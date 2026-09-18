"""SoulX-FlashHead on Modal, as a resident renderer.

FlashHead is CUDA-native — flash_attn, sageattention and xfuser are all built
for it, and upstream's published numbers are on an RTX 4090 — so it runs on a
rented GPU. This module deploys it as a class whose container loads the
pipeline once (`@modal.enter`) and then serves render calls in-process, so a
warm container answers in seconds instead of re-loading 8 GB of weights and
re-running data prep for every clip.

    modal deploy app.py                       # persistent app "flashhead"
    modal run app.py --audio in.wav           # CLI probe against the deployed class

Keep-warm is a runtime toggle, not a deploy-time constant — see
`Renderer().update_autoscaler(min_containers=1|0)` from the studio. An idle
L4 bills ~$0.80/hr, so the default is scale-to-zero after `SCALEDOWN_S`.
"""

import json
import os
import pathlib
import threading
import time
from collections import deque

import modal

from renderer import (CACHE, CKPT, COMPILE, DONE_MARK, GPU, IDLE_CHUNKS, IDLE_DIR,
                      LIVE_MAX_SESSIONS, SRC, WARMUP_IMAGE, WAV2VEC, WEIGHTS, Host,
                      RendererCore, _HlsShipper, _idle_loop_path, _live_ffmpeg,
                      _write_mp4, print_summary)

# Both are overridable so a second, differently-sized deployment can be stood up
# beside the working one instead of replacing it:
#
#   FLASHHEAD_APP=flashhead-h100 FLASHHEAD_GPU=H100 modal deploy app.py
#   FLASHHEAD_APP=flashhead-h100 FLASHHEAD_GPU=H100 modal run app.py::mux --sessions 6
#
# The studio keeps pointing at the default app, so an experiment on a faster card
# never disturbs a demo that is running.
APP_NAME = os.environ.get("FLASHHEAD_APP", "flashhead")
REPO = "https://github.com/Soul-AILab/SoulX-FlashHead.git"

# Modal has no RTX 4090 (upstream's benchmark card). L4 24 GB is the cheapest
# 24 GB option; A10 is faster but pricier. L4's 300 GB/s bandwidth is well under
# the 4090's ~1 TB/s, so expect the 96 FPS Lite figure not to carry over.
#
# The scheduler's per-slot budget is floor(0.96 s / chunk time), so the card
# decides how many sessions may speak at the same moment: L4 measures 0.746 s
# → 1. Two concurrent speakers need a chunk at or under 0.48 s.

# torch.compile. Upstream hardcodes it on for its realtime-streaming case —
# one resident process, fixed shapes, thousands of chunks — which is exactly
# what the live path became. It was turned off for the original use (offline
# short clips, where a 7.8 s job spent 430 s compiling), and that reason no
# longer covers live. The inductor and triton caches sit on a Volume, so the
# cost should be paid once per image rather than once per container.

# Idle seconds before a warm container is torn down. Long enough to cover a demo
# session's gaps, short enough that a forgotten toggle does not burn the budget.
SCALEDOWN_S = 600

# requirements.txt lines that are either unused by the inference path or that
# break resolution. Verified by grepping the imports under flash_head/:
#   xformers        -> never imported (only in requirements)
#   nvidia-nccl-cu12 -> pinned at 2.27.3, which conflicts with xformers and is
#                       only needed for multi-GPU; torch brings its own
#   decord, flask, gradio -> data prep and the web demo, not inference
#   nvidia-nccl pin removal also lets torch keep a consistent CUDA stack
_DROP = ("xformers", "nvidia-nccl-cu12", "decord", "flask", "gradio")

# NOTE: `Dockerfile` in this directory builds the same environment for hosts
# that are not Modal, and `setup.sh` does it against a box's own Python. The
# three are deliberately parallel — change one, change the others. They are not
# unified through `modal.Image.from_dockerfile` yet because that changes how the
# working deployment builds, and there is no Modal credit to verify it with.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6")
    # index_url, not extra_index_url: on PyPI today plain `torch` resolves to a
    # CUDA 13 build, whose nvidia-* stack then fights requirements.txt.
    .pip_install(
        "torch", "torchvision", "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(f"git clone --depth 1 {REPO} {SRC}")
    .run_commands(
        "python - <<'EOF'\n"
        "drop = " + repr(_DROP) + "\n"
        f"src = '{SRC}/requirements.txt'\n"
        "keep = [l for l in open(src) if l.strip() and not any(l.lower().startswith(d) for d in drop)]\n"
        f"open('{SRC}/requirements.modal.txt','w').writelines(keep)\n"
        "print('kept', len(keep), 'requirement lines')\n"
        "EOF"
    )
    .run_commands(f"pip install -r {SRC}/requirements.modal.txt")
    # Optional CUDA attention kernels: faster when present, and the model falls
    # back to F.scaled_dot_product_attention when not, so never fail the build.
    .run_commands(
        "pip install flash_attn==2.8.0.post2 --no-build-isolation || "
        "echo 'flash_attn unavailable, using SDPA fallback'"
    )
    .pip_install("huggingface_hub[cli]")
    # T1: turn off torch.compile. Upstream hardcodes COMPILE_MODEL/COMPILE_VAE = True
    # (flash_head_pipeline.py:19-20) for its realtime-streaming use case, where one
    # process serves hundreds of chunks and compile pays back. Ours is offline
    # short clips: a 7.8 s job spent 430 s in chunk-0 compiling (denoise 174 s +
    # 161 s, VAE encode 78 s) and 4 s actually generating. The persisted inductor
    # cache did not help either -- `once` mode feeds different tensor shapes than
    # the `stream` runs that populated it. Placed last so cached layers survive.
    .run_commands(
        f"sed -i 's/^COMPILE_MODEL = {str(not COMPILE)}/COMPILE_MODEL = {str(COMPILE)}/; "
        f"s/^COMPILE_VAE = {str(not COMPILE)}/COMPILE_VAE = {str(COMPILE)}/' "
        f"{SRC}/flash_head/src/pipeline/flash_head_pipeline.py",
        f"grep -q '^COMPILE_MODEL = {str(COMPILE)}' {SRC}/flash_head/src/pipeline/flash_head_pipeline.py "
        f"&& grep -q '^COMPILE_VAE = {str(COMPILE)}' {SRC}/flash_head/src/pipeline/flash_head_pipeline.py "
        f"|| (echo 'compile flags NOT patched' && exit 1)",
    )
    # T2: drop the per-step timing instrumentation from generate(). It forces
    # torch.cuda.synchronize() eight times per chunk — twice around every denoise
    # step, then around decode, colour correction and the motion encode — purely
    # so it can print each stage. On L4 a denoise step is ~105 ms and the stalls
    # hide in it; on H100 a step is ~29 ms and they do not. _run_chunk's own
    # swap/model/transfer split replaces what these prints told us.
    .run_commands(
        f"sed -i '/torch\\.cuda\\.synchronize()/d; "
        f"s/^\\( *\\)print(f.\\[generate\\].*/\\1pass/' "
        f"{SRC}/flash_head/src/pipeline/flash_head_pipeline.py",
        f"! grep -q 'torch\\.cuda\\.synchronize()' {SRC}/flash_head/src/pipeline/flash_head_pipeline.py "
        f"|| (echo 'sync instrumentation NOT stripped' && exit 1)",
        f"python -c \"import ast,sys; ast.parse(open('{SRC}/flash_head/src/pipeline/flash_head_pipeline.py').read())\" "
        f"|| (echo 'pipeline no longer parses after strip' && exit 1)",
    )
    .env(
        {
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE}/inductor",
            "TRITON_CACHE_DIR": f"{CACHE}/triton",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "HF_HOME": f"{CACHE}/hf",
        }
    )
)
# `base_image` has no local files, so other apps (bench_multi.py, probe_gpu.py)
# can add build steps on top of it; add_local_* must come last on an Image.
base_image = image
# Default avatar, used only to warm the container in `Renderer.load`, plus the
# host-free renderer itself — Modal does not mount imported local modules on its
# own, so without this the container would import app.py and fail on `renderer`.
image = (base_image
         .add_local_python_source("renderer")
         .add_local_file(pathlib.Path(__file__).parent / "inputs" / "newscaster.png", "/root/warmup.png"))

weights = modal.Volume.from_name("flashhead-weights", create_if_missing=True)
cache = modal.Volume.from_name("flashhead-compile-cache", create_if_missing=True)
# Per-job progress lines, one partition per job id. A deployed method's stdout is
# not returned to the caller, and a long render with no visible progress is
# indistinguishable from a hang — this is how the studio tails it instead.
progress = modal.Queue.from_name("flashhead-progress", create_if_missing=True)
# Resident-container readiness. `modal container list` shows a container the
# moment it is scheduled, a minute or two before `enter()` finishes loading;
# this flag flips only when the pipeline can actually take a call.
state = modal.Dict.from_name("flashhead-state", create_if_missing=True)

app = modal.App(APP_NAME)



@app.function(image=image, volumes={WEIGHTS: weights}, timeout=3600)
def fetch_weights(include_pro: bool = False) -> str:
    """Pull the checkpoints into the Volume once. Idempotent."""
    import subprocess

    from huggingface_hub import snapshot_download

    ignore = None if include_pro else ["Model_Pro/*"]
    snapshot_download("Soul-AILab/SoulX-FlashHead-1_3B", local_dir=CKPT, ignore_patterns=ignore)
    snapshot_download(
        "facebook/wav2vec2-base-960h",
        local_dir=WAV2VEC,
        ignore_patterns=["*.h5", "*.msgpack", "*tf_model*"],
    )
    weights.commit()
    return subprocess.run(["du", "-sh", WEIGHTS], capture_output=True, text=True).stdout.strip()





class ModalHost(Host):
    """The renderer's four calls, on Modal's primitives."""

    def put(self, item, partition: str) -> None:
        progress.put(item, block=False, partition=partition)

    def get_many(self, n: int, partition: str, block: bool = False,
                 timeout: float | None = None) -> list:
        return progress.get_many(n, block=block, timeout=timeout, partition=partition)

    def publish_ready(self, key: str, payload: dict | None) -> None:
        state.pop(key, None) if payload is None else state.__setitem__(key, payload)

    def commit(self) -> None:
        weights.commit()


MODAL_HOST = ModalHost()


@app.cls(
    image=image,
    gpu=GPU,
    volumes={WEIGHTS: weights, CACHE: cache},
    scaledown_window=SCALEDOWN_S,
    min_containers=0,
    max_containers=1,   # one GPU; sessions share it rather than fanning out
    timeout=3600,
)
@modal.concurrent(max_inputs=LIVE_MAX_SESSIONS)
class Renderer:
    """Modal adapter around `renderer.RendererCore`.

    Everything here is Modal: the lifecycle hooks, the exposed methods, and the
    Host that maps the renderer's four calls onto Queue/Dict/Volume. The renderer
    itself is held by composition rather than inheritance — Modal rewrites the
    classes it decorates, and a 900-line base class is not something to hand it
    when the same delegation costs six lines.
    """

    # "lite" (default) or "pro". Each value gets its own container pool and its
    # own autoscaler settings, so Renderer(model_type="pro") never competes with
    # the resident Lite streamer for the GPU, and scales to zero on its own.
    model_type: str = modal.parameter(default="lite")

    @modal.enter()
    def _enter(self) -> None:
        self.core = RendererCore(host=MODAL_HOST, model_type=self.model_type)
        self.core.load()

    @modal.exit()
    def _exit(self) -> None:
        self.core.unload()

    @modal.method()
    def ping(self) -> dict:
        return self.core.ping()

    @modal.method()
    def render(self, *args, **kwargs):
        return self.core.render(*args, **kwargs)

    @modal.method()
    def render_stream(self, *args, **kwargs):
        return self.core.render_stream(*args, **kwargs)

    @modal.method()
    def live(self, *args, **kwargs):
        return self.core.live(*args, **kwargs)


@app.function(image=image, gpu=GPU, timeout=900)
def smoke() -> dict:
    """Cheap environment check. Surfaces image build and import problems in a
    minute instead of after a long GPU job."""
    import subprocess
    import sys

    out: dict[str, str] = {"python": sys.version.split()[0]}
    try:
        out["nvidia-smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        out["nvidia-smi"] = f"unavailable: {exc}"
    import torch

    out["torch"] = f"{torch.__version__} cuda={torch.cuda.is_available()}"
    for mod in ("xfuser", "diffusers", "transformers", "flash_attn", "sageattention", "mediapipe"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"MISSING ({type(exc).__name__})"
    import importlib
    import os

    os.chdir(SRC)
    sys.path.insert(0, SRC)
    for target in ("flash_head.src.modules.flash_head_model", "flash_head.src.pipeline.flash_head_pipeline", "flash_head.inference"):
        try:
            importlib.import_module(target)
            out[target.rsplit(".", 1)[-1]] = "imports"
        except Exception as exc:  # noqa: BLE001
            out[target.rsplit(".", 1)[-1]] = f"FAILED {type(exc).__name__}: {str(exc)[:200]}"
    print("=== FlashHead environment ===", flush=True)
    for k, v in out.items():
        print(f"  {k:24s} {v}", flush=True)
    return out


def tail_progress(job_id: str, call: modal.FunctionCall, poll_s: float = 1.0):
    """Yield progress lines for `job_id` until the call returns; final item is
    ("result", bytes). Shared by the CLI entrypoint and the studio server."""
    while True:
        while (line := progress.get(block=False, partition=job_id)) is not None:
            yield ("line", line)
        try:
            yield ("result", call.get(timeout=0))
            return
        except TimeoutError:
            time.sleep(poll_s)


@app.local_entrypoint()
def main(
    image: str = "inputs/newscaster.png",
    audio: str = "inputs/korean_short.wav",
    out: str = "",
    seed: int = 42,
    fetch: bool = False,
):
    """CLI probe against the *deployed* Renderer (run `modal deploy app.py` first)."""
    import uuid

    if fetch:
        print("weights:", fetch_weights.remote())
    img_p, wav_p = pathlib.Path(image), pathlib.Path(audio)
    job_id = uuid.uuid4().hex[:10]
    renderer = modal.Cls.from_name(APP_NAME, "Renderer")()
    t0 = time.time()
    call = renderer.render.spawn(img_p.read_bytes(), wav_p.read_bytes(), job_id, seed=seed)
    mp4 = b""
    for kind, item in tail_progress(job_id, call):
        if kind == "line":
            print(item, flush=True)
        else:
            mp4 = item
    dest = pathlib.Path(out or f"out_{img_p.stem}_{wav_p.stem}_resident.mp4")
    dest.write_bytes(mp4)
    print(f"wrote {dest} ({dest.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s wall")


# ---------------------------------------------------------------------------
# Pro live sessions on 2×H100 (jp): the measured real-time configuration
# (0.66 s per 1.12 s chunk, FA2, compile off, xfuser Ulysses degree 2). The
# container keeps a torchrun worker pair (live_worker.py) resident; this class
# only supervises: it relays studio audio into the workers' control FIFO,
# tails their status file, and ships the HLS files they write.
FA_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
# xfuser 0.6.0 (what the base image resolved) needs an unreleased yunchang; 0.4.5 is FlashHead's era.
image_live_pro = (
    base_image.pip_install(FA_WHEEL, "xfuser==0.4.5")
    .add_local_python_source("live_worker", "renderer")
    .add_local_file(pathlib.Path(__file__).parent / "inputs" / "newscaster.png", "/root/warmup.png")
)
LIVE_PRO_GPU = "H100:2"
LIVE_PRO_REGION = ["jp"]


@app.cls(
    image=image_live_pro,
    gpu=LIVE_PRO_GPU,
    region=LIVE_PRO_REGION,
    volumes={WEIGHTS: weights, CACHE: cache},
    scaledown_window=300,
    min_containers=0,
    max_containers=1,
    timeout=3600,
)
class LivePro:
    @modal.enter()
    def start_workers(self) -> None:
        import os
        import subprocess
        import threading

        self.root = pathlib.Path("/tmp/livepro")
        self.root.mkdir(exist_ok=True)
        self.ctl_path = self.root / "control.fifo"
        os.mkfifo(self.ctl_path)
        self.status_path = self.root / "status.jsonl"
        self.status_path.touch()
        self.status_pos = 0
        env = {**os.environ, "NCCL_MIN_NCHANNELS": "4", "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": "/root:" + os.environ.get("PYTHONPATH", "")}
        cmd = ["torchrun", "--nproc_per_node", "2", "--master_port", "29513", "-m", "live_worker",
               "--ckpt_dir", CKPT, "--wav2vec_dir", WAV2VEC, "--model_type", "pro",
               "--control_fifo", str(self.ctl_path), "--status_file", str(self.status_path), "--warmup_image", WARMUP_IMAGE]
        self.worker_log = open(self.root / "worker.log", "wb")
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=self.worker_log, stderr=subprocess.STDOUT)
        ready = None
        while time.time() - t0 < 900 and ready is None:
            for ev in self._status_events():
                if ev.get("event") == "worker_ready":
                    ready = ev
            if self.proc.poll() is not None:
                raise RuntimeError("live worker exited during startup:\n" + self._worker_tail())
            time.sleep(1)
        if ready is None:
            raise TimeoutError("live worker not ready within 900 s")
        # the worker opened the read end before reporting ready, so this does not block
        self.ctl = open(self.ctl_path, "w")
        self._ready = {"since": time.time(), "startup_s": round(time.time() - t0), "worker": ready}

        def beat() -> None:
            while True:
                try:
                    state["ready:pro-live"] = {**self._ready, "beat": time.time()}
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(20)

        threading.Thread(target=beat, daemon=True).start()
        print(f"live workers resident (pro, 2 GPUs): {time.time()-t0:.0f}s, load {ready.get('load_s')}s", flush=True)

    @modal.exit()
    def stop_workers(self) -> None:
        state.pop("ready:pro-live", None)
        try:
            self._ctl_send({"type": "quit"})
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            pass
        if self.proc.poll() is None:
            self.proc.kill()

    def _ctl_send(self, msg: dict) -> None:
        import json

        self.ctl.write(json.dumps(msg) + "\n")
        self.ctl.flush()

    def _status_events(self) -> list[dict]:
        import json

        out: list[dict] = []
        with open(self.status_path, "rb") as f:
            f.seek(self.status_pos)
            data = f.read()
        if not data:
            return out
        lines = data.split(b"\n")
        complete = lines[:-1]
        self.status_pos += sum(len(l) + 1 for l in complete)
        for l in complete:
            if l.strip():
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
        return out

    def _worker_tail(self) -> str:
        try:
            return (self.root / "worker.log").read_text(errors="replace")[-3000:]
        except OSError:
            return ""

    @modal.method()
    def ping(self) -> dict:
        return {"warm": True, "model_type": "pro-live", **self._ready}

    @modal.method()
    def live(
        self,
        session_id: str,
        image_bytes: bytes,
        seed: int = 42,
        idle_timeout_s: float = 180.0,
        max_seconds: float = 1800.0,
        lead_chunks: float = 1.0,
    ) -> dict:
        """Same contract as Renderer.live (audio in on `<sid>:audio`, HLS out on
        `<sid>`), executed by the resident 2-rank Pro worker."""
        import base64
        import shutil

        t0 = time.time()
        sess = self.root / session_id
        hls = sess / "hls"
        hls.mkdir(parents=True, exist_ok=True)
        ref = sess / "ref.png"
        ref.write_bytes(image_bytes)
        ship = _HlsShipper(MODAL_HOST, session_id, hls, t0)
        audio_part = f"{session_id}:audio"
        stats: dict | None = None
        n_audio = 0
        try:
            ship.log("live(pro): starting session on the resident 2×H100 worker")
            self._ctl_send({"type": "start", "sid": session_id, "ref": str(ref), "hls_dir": str(hls),
                            "audio_fifo": str(sess / "audio.fifo"), "seed": seed, "idle_timeout": idle_timeout_s,
                            "max_seconds": max_seconds, "lead_chunks": lead_chunks})
            while True:
                for item in progress.get_many(50, block=False, partition=audio_part):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "end":
                        self._ctl_send({"type": "end"})
                    elif item.get("type") == "audio":
                        # bulk PCM goes through a file, not the FIFO: a 20 s sentence
                        # as one base64 line took ~7 s to cross the pipe (64 KB reads)
                        pcm = item.get("pcm") or b"".join(item.get("chunks", []))
                        n_audio += 1
                        pcm_path = sess / f"audio_{n_audio:04d}.pcm"
                        pcm_path.write_bytes(pcm)
                        self._ctl_send({"type": "audio_file", "path": str(pcm_path), "final": bool(item.get("final", True))})
                for ev in self._status_events():
                    e = ev.get("event")
                    if e == "session_ready":
                        ship.log(f"live: ready — 1.12s chunks (pro, 2 GPUs), prep {ev.get('prep_s')}s, idle timeout {idle_timeout_s:.0f}s")
                    elif e == "chunk":
                        if ev["speech"] or ev["k"] % 10 == 0 or ev["k"] <= 3:
                            ship.log(f"live: chunk {ev['k']} {'speech' if ev['speech'] else 'silence'} gen {ev['gen']:.2f}s behind {ev['behind']:.2f}s queued {ev['queued']:.1f}s")
                    elif e == "session_end":
                        stats = ev
                    elif e == "error":
                        raise RuntimeError(f"worker error: {ev.get('error')}")
                if stats is not None:
                    break
                if self.proc.poll() is not None:
                    raise RuntimeError("live worker died:\n" + self._worker_tail())
                if time.time() - t0 > max_seconds + 300:
                    self._ctl_send({"type": "end"})
                    raise TimeoutError("live session exceeded max_seconds")
                ship.ship_new_files()
                time.sleep(0.2)
            ship.ship_new_files()
            ship.log(f"{DONE_MARK} live ended ({stats.get('reason')}): {stats.get('chunks')} chunks, {stats.get('speech_chunks')} speech, {ship.streamed/1e6:.1f} MB")
            ship.close()
            return {"chunks": stats.get("chunks"), "speech_chunks": stats.get("speech_chunks"), "ended_by": stats.get("reason"),
                    "gen_median_s": stats.get("gen_median_s"), "ffmpeg_rc": stats.get("ffmpeg_rc"), "seconds": stats.get("seconds")}
        finally:
            if ship.thread.is_alive():
                ship.q.put(None)
            shutil.rmtree(sess, ignore_errors=True)





# ---------------------------------------------------------------------------
# Spike: can a pre-rendered idle loop hand off to live generation seamlessly?
#
# The live loop currently generates silent chunks while the candidate speaks —
# measured 44-84% of a session's GPU work. If those frames can come from a
# pre-rendered loop instead, GPUs are only needed while the avatar speaks.
#
# The model conditions each chunk on `latent_motion_frames` = VAE-encoded last 9
# frames of the previous chunk, so injecting the loop's last frames there should
# continue the video. The risk is the round trip: in production the loop lives as
# an h264 file, so the frames come back quantised to uint8 and lossily coded.
# Three conditions isolate each loss stage, same seed and same audio throughout:
#   C (control) — the pipeline's own float latents, never left the GPU
#   A           — re-encoded from uint8 frames held in memory (quantisation only)
#   B           — re-encoded from frames decoded out of the saved mp4 (+ h264)
@app.function(image=image, gpu=GPU, volumes={WEIGHTS: weights, CACHE: cache}, timeout=1800)
def spike_seam(
    image_bytes: bytes,
    audio_bytes: bytes,
    seed: int = 42,
    idle_chunks: int = 10,
    speech_chunks: int = 6,
) -> dict:
    import os
    import sys
    import tempfile
    from collections import deque

    os.chdir(SRC)
    sys.path.insert(0, SRC)

    import imageio
    import librosa
    import numpy as np
    import torch

    from flash_head.inference import (
        get_audio_embedding,
        get_base_data,
        get_infer_params,
        get_pipeline,
        run_pipeline,
    )

    work = pathlib.Path(tempfile.mkdtemp(prefix="spike-"))
    (work / "ref.png").write_bytes(image_bytes)
    (work / "speech.wav").write_bytes(audio_bytes)

    t0 = time.time()
    pipeline = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type="lite", wav2vec_dir=WAV2VEC)
    get_base_data(pipeline, cond_image_path_or_dir=str(work / "ref.png"), base_seed=seed, use_face_crop=True)
    p = get_infer_params()
    sr, fps = p["sample_rate"], p["tgt_fps"]
    frame_num, motion = p["frame_num"], p["motion_frames_num"]
    slice_len = frame_num - motion
    chunk_samples = slice_len * sr // fps
    cached = p["cached_audio_duration"] * sr
    end_idx = p["cached_audio_duration"] * fps
    start_idx = end_idx - frame_num
    print(f"pipeline ready in {time.time()-t0:.0f}s", flush=True)

    def gen(audio_dq: deque, pcm: np.ndarray) -> np.ndarray:
        """One chunk: push 0.96 s of audio, generate, return new frames (T,H,W,C) uint8."""
        audio_dq.extend(pcm.tolist())
        emb = get_audio_embedding(pipeline, np.array(audio_dq, dtype=np.float32), start_idx, end_idx)
        return run_pipeline(pipeline, emb)[motion:].to(torch.uint8).cpu().numpy()

    def encode_motion(frames_uint8: np.ndarray) -> torch.Tensor:
        """uint8 (T,H,W,C) -> the latent the pipeline expects in latent_motion_frames."""
        t = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(pipeline.device, dtype=pipeline.param_dtype)
        t = (t / 255 - 0.5) * 2
        return pipeline.vae.encode(t.permute(3, 0, 1, 2).unsqueeze(0))   # 1 C T H W

    silence = np.zeros(chunk_samples, dtype=np.float32)

    # ---- phase 1: the idle loop (pure silence), exactly as a live session idles
    idle_dq: deque = deque([0.0] * cached, maxlen=cached)
    idle_frames = [gen(idle_dq, silence) for _ in range(idle_chunks)]
    idle = np.concatenate(idle_frames)
    control_latent = pipeline.latent_motion_frames.clone()          # condition C
    audio_state = list(idle_dq)                                     # the 8 s window at hand-off
    loop_mp4 = work / "idle_loop.mp4"
    with imageio.get_writer(str(loop_mp4), format="mp4", mode="I", fps=fps, codec="h264",
                            ffmpeg_params=["-bf", "0", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"]) as w:
        for f in idle:
            w.append_data(f)
    reader = imageio.get_reader(str(loop_mp4))
    decoded = np.stack([np.asarray(f) for f in reader])
    reader.close()
    print(f"idle loop: {len(idle)} frames, mp4 {loop_mp4.stat().st_size/1e6:.1f} MB, decoded {decoded.shape}", flush=True)

    # ---- phase 2: the same speech from each hand-off condition
    speech, _ = librosa.load(str(work / "speech.wav"), sr=sr, mono=True)
    need = speech_chunks * chunk_samples
    speech = np.concatenate([speech, np.zeros(max(0, need - len(speech)), dtype=np.float32)])[:need]

    conditions = {
        "C_control_float": None,                       # keep the pipeline's own latents
        "A_uint8_memory": idle[-motion:],              # quantisation only
        "B_h264_file": decoded[-motion:],              # quantisation + h264
    }
    out: dict = {}
    per_chunk: dict = {}
    for name, frames in conditions.items():
        pipeline.latent_motion_frames = control_latent.clone() if frames is None else encode_motion(frames)
        pipeline.generator.manual_seed(seed)           # identical noise in every condition
        dq: deque = deque(audio_state, maxlen=cached)
        got = [gen(dq, speech[i * chunk_samples : (i + 1) * chunk_samples]) for i in range(speech_chunks)]
        per_chunk[name] = got
        out[name] = np.concatenate(got)
        print(f"{name}: generated {len(out[name])} frames", flush=True)

    # ---- phase 3: measure
    def mad(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    last_idle = idle[-1]
    intra_idle = float(np.mean([mad(idle[i], idle[i + 1]) for i in range(len(idle) - 1)]))
    res: dict = {
        "idle_frames": int(len(idle)),
        "idle_mp4_mb": round(loop_mp4.stat().st_size / 1e6, 2),
        "h264_roundtrip_mad": round(mad(idle[-motion:], decoded[-motion:]), 3),
        "intra_frame_mad_idle": round(intra_idle, 2),
        "conditions": {},
    }
    ctrl = out["C_control_float"]
    for name, frames in out.items():
        intra = float(np.mean([mad(frames[i], frames[i + 1]) for i in range(min(len(frames) - 1, 24))]))
        res["conditions"][name] = {
            # the jump across the hand-off, next to what an ordinary frame step looks like
            "boundary_mad": round(mad(last_idle, frames[0]), 2),
            "intra_frame_mad_speech": round(intra, 2),
            "divergence_from_control": round(mad(frames, ctrl), 3),
            "mean_rgb_first3": [round(float(frames[:3, :, :, c].mean()), 1) for c in range(3)],
            "mean_rgb_last3": [round(float(frames[-3:, :, :, c].mean()), 1) for c in range(3)],
            # does the injected error grow over a long speaking turn, or stay bounded?
            "divergence_per_chunk": [round(mad(per_chunk[name][i], per_chunk["C_control_float"][i]), 2)
                                     for i in range(len(per_chunk[name]))],
        }
    res["mean_rgb_last_idle"] = [round(float(last_idle[:, :, c].mean()), 1) for c in range(3)]

    # ---- strips across the hand-off for the eye: 3 idle frames then 5 generated
    strips = {}
    for name, frames in out.items():
        strip = np.concatenate(list(idle[-3:]) + list(frames[:5]), axis=1)
        path = work / f"strip_{name}.png"
        imageio.imwrite(str(path), strip)
        strips[name] = path.read_bytes()
    # ---- hand-off clips for the eye: 1 s of idle loop, then the generated speech,
    # with matching audio (silence over the idle tail) so the sync is visible too
    import subprocess
    import wave as wavmod

    tail = 25
    mux_wav = work / "mux.wav"
    with wavmod.open(str(mux_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        pad = np.zeros(tail * sr // fps, dtype=np.float32)
        w.writeframes((np.clip(np.concatenate([pad, speech]), -1, 1) * 32767).astype(np.int16).tobytes())
    clips = {}
    for name, frames in out.items():
        silent = work / f"{name}_silent.mp4"
        _write_mp4(silent, np.concatenate([idle[-tail:], frames]), fps)
        final = work / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(mux_wav),
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)], check=True)
        clips[name] = final.read_bytes()

    res["seconds"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(res), flush=True)
    return {"metrics": res, "strips": strips, "idle_loop_mp4": loop_mp4.read_bytes(),
            "clips": clips, "handoff_frame": tail}



@app.local_entrypoint()
def mux(image: str = "inputs/newscaster.png", audio: str = "inputs/korean_short.wav",
        sessions: int = 3, minutes: float = 3.0, gap_s: float = 18.0, stagger_s: float = 4.0):
    """한 컨테이너에 여러 라이브 세션을 붙여 용량을 잰다.

    각 세션은 `gap_s`마다 한 문장씩 말하고 나머지 시간은 대기 루프를 재생한다.
    세션들은 `stagger_s`만큼 어긋나게 시작해 발화가 겹치는 순간이 생기게 한다.
    """
    import threading
    import uuid
    import wave

    with wave.open(audio, "rb") as w:
        pcm = w.readframes(w.getnframes())
        speech_s = w.getnframes() / w.getframerate()
    img = pathlib.Path(image).read_bytes()
    ids = [f"mux-{uuid.uuid4().hex[:8]}" for _ in range(sessions)]
    print(f"{sessions} sessions · 문장 {speech_s:.1f}s 마다 {gap_s:.0f}s · {minutes:.0f}분 "
          f"(예상 발화 비중 {speech_s/gap_s*100:.0f}%)", flush=True)

    renderer = modal.Cls.from_name(APP_NAME, "Renderer")()
    calls = {sid: renderer.live.spawn(sid, img, seed=42, idle_timeout_s=minutes * 60 + 120,
                                      max_seconds=minutes * 60 + 120) for sid in ids}
    deadline = time.time() + minutes * 60
    said = {sid: 0 for sid in ids}
    ready = {sid: None for sid in ids}
    finished = threading.Event()        # daemon threads must stop before the client closes

    def drive(sid: str, delay: float) -> None:
        if finished.wait(delay):
            return
        while time.time() < deadline and not finished.is_set():
            progress.put({"type": "audio", "pcm": pcm, "final": True}, block=False, partition=f"{sid}:audio")
            said[sid] += 1
            finished.wait(gap_s)
        if not finished.is_set():
            progress.put({"type": "end"}, block=False, partition=f"{sid}:audio")

    def drain(sid: str) -> None:
        # the studio would be consuming the HLS; here we just keep the queue empty
        while not finished.is_set():
            try:
                for item in progress.get_many(50, block=False, partition=sid):
                    if isinstance(item, dict) and item.get("type") == "batch":
                        for it in item["items"]:
                            if it.get("type") == "log" and "live: ready" in it.get("line", "") and ready[sid] is None:
                                ready[sid] = time.time()
            except Exception:  # noqa: BLE001 — the client closes as the run ends
                return
            finished.wait(0.5)

    threads = [threading.Thread(target=drive, args=(sid, i * stagger_s), daemon=True) for i, sid in enumerate(ids)]
    threads += [threading.Thread(target=drain, args=(sid,), daemon=True) for sid in ids]
    for t in threads:
        t.start()

    results = {}
    for sid, call in calls.items():
        try:
            results[sid] = call.get(timeout=minutes * 60 + 240)
        except Exception as exc:  # noqa: BLE001
            results[sid] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    finished.set()
    for t in threads:
        t.join(timeout=3)

    print_summary(results)


@app.local_entrypoint()
def seam(image: str = "inputs/newscaster.png", audio: str = "inputs/korean_flashhead.wav",
         out_dir: str = "spike-seam", speech_chunks: int = 25):
    """대기 루프 → 생성 전환 이음새 스파이크. 결과는 <out_dir>/ 에 저장."""
    import json as _json

    d = pathlib.Path(out_dir)
    d.mkdir(exist_ok=True)
    r = spike_seam.remote(pathlib.Path(image).read_bytes(), pathlib.Path(audio).read_bytes(), speech_chunks=speech_chunks)
    (d / "metrics.json").write_text(_json.dumps(r["metrics"], indent=1, ensure_ascii=False))
    for name, data in r["strips"].items():
        (d / f"strip_{name}.png").write_bytes(data)
    (d / "idle_loop.mp4").write_bytes(r["idle_loop_mp4"])
    for name, data in r["clips"].items():
        (d / f"handoff_{name}.mp4").write_bytes(data)
    # 나란히 비교용 (왼쪽부터 C 대조군 · A uint8 · B h264)
    import subprocess as _sp

    order = ["C_control_float", "A_uint8_memory", "B_h264_file"]
    _sp.run(["ffmpeg", "-y", "-loglevel", "error",
             *sum(([" -i".strip(), str(d / f"handoff_{n}.mp4")] for n in order), []),
             "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
             "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(d / "handoff_CAB_side_by_side.mp4")], check=True)
    print(f"전환 시점: {r['handoff_frame']}번째 프레임 ({r['handoff_frame']/25:.1f}초)")
    m = r["metrics"]
    print(f"\nh264 왕복 손실(MAD): {m['h264_roundtrip_mad']}   대기 루프 프레임 간 변화: {m['intra_frame_mad_idle']}")
    for name, c in m["conditions"].items():
        pc = c["divergence_per_chunk"]
        print(f"  {name:18s} 경계 점프 {c['boundary_mad']:6.2f} | 발화 중 프레임 간 {c['intra_frame_mad_speech']:6.2f} | 대조군 대비 {c['divergence_from_control']:7.3f}")
        print(f"                     청크별 발산: {' '.join(f'{v:.1f}' for v in pc[:12])}{' …' if len(pc) > 12 else ''}  (마지막 {pc[-1]:.1f})")
    print(f"\n저장: {d}/")


# ---------------------------------------------------------------------------
# Spike 2: the reverse hand-off — generation back to the idle loop.
#
# `spike_seam` proved loop -> generation is invisible. A turn also has to END:
# when the avatar stops talking the stream must return to the loop, and the
# loop's frames are fixed while the last generated frame is whatever the model
# produced. Four strategies, measured against the natural frame-to-frame step:
#   R1 cut to the loop's first frame            (naive)
#   R2 cut to the loop frame nearest that pose  (the loop is ~240 frames, so search it)
#   R3 R2 plus a short cross-fade
#   R4 generate a couple of silent chunks first to let the face settle, then R3
@app.function(image=image, gpu=GPU, volumes={WEIGHTS: weights, CACHE: cache}, timeout=1800)
def spike_return(
    image_bytes: bytes,
    audio_bytes: bytes,
    seed: int = 42,
    idle_chunks: int = 10,
    speech_chunks: int = 8,
    settle_chunks: int = 2,
    fade_frames: int = 4,
) -> dict:
    import os
    import sys
    import tempfile
    from collections import deque

    os.chdir(SRC)
    sys.path.insert(0, SRC)

    import imageio
    import librosa
    import numpy as np
    import torch

    from flash_head.inference import (
        get_audio_embedding,
        get_base_data,
        get_infer_params,
        get_pipeline,
        run_pipeline,
    )

    work = pathlib.Path(tempfile.mkdtemp(prefix="spike-ret-"))
    (work / "ref.png").write_bytes(image_bytes)
    (work / "speech.wav").write_bytes(audio_bytes)

    t0 = time.time()
    pipeline = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type="lite", wav2vec_dir=WAV2VEC)
    get_base_data(pipeline, cond_image_path_or_dir=str(work / "ref.png"), base_seed=seed, use_face_crop=True)
    p = get_infer_params()
    sr, fps = p["sample_rate"], p["tgt_fps"]
    frame_num, motion = p["frame_num"], p["motion_frames_num"]
    slice_len = frame_num - motion
    chunk_samples = slice_len * sr // fps
    cached = p["cached_audio_duration"] * sr
    end_idx = p["cached_audio_duration"] * fps
    start_idx = end_idx - frame_num

    def gen(dq: deque, pcm: np.ndarray) -> np.ndarray:
        dq.extend(pcm.tolist())
        emb = get_audio_embedding(pipeline, np.array(dq, dtype=np.float32), start_idx, end_idx)
        return run_pipeline(pipeline, emb)[motion:].to(torch.uint8).cpu().numpy()

    def encode_motion(frames_uint8: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(pipeline.device, dtype=pipeline.param_dtype)
        t = (t / 255 - 0.5) * 2
        return pipeline.vae.encode(t.permute(3, 0, 1, 2).unsqueeze(0))

    def mad(a, b) -> float:
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    silence = np.zeros(chunk_samples, dtype=np.float32)

    # idle loop, stored and read back as mp4 like production would
    dq: deque = deque([0.0] * cached, maxlen=cached)
    idle = np.concatenate([gen(dq, silence) for _ in range(idle_chunks)])
    loop_mp4 = work / "idle_loop.mp4"
    with imageio.get_writer(str(loop_mp4), format="mp4", mode="I", fps=fps, codec="h264",
                            ffmpeg_params=["-bf", "0", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"]) as w:
        for f in idle:
            w.append_data(f)
    reader = imageio.get_reader(str(loop_mp4))
    loop = np.stack([np.asarray(f) for f in reader])
    reader.close()
    audio_state = list(dq)

    # a speaking turn, entered from the loop (the proven direction)
    speech, _ = librosa.load(str(work / "speech.wav"), sr=sr, mono=True)
    need = (speech_chunks + settle_chunks) * chunk_samples
    speech = np.concatenate([speech, np.zeros(max(0, need - len(speech)), dtype=np.float32)])
    pipeline.latent_motion_frames = encode_motion(loop[-motion:])
    pipeline.generator.manual_seed(seed)
    dq = deque(audio_state, maxlen=cached)
    spoken = np.concatenate([gen(dq, speech[i * chunk_samples : (i + 1) * chunk_samples]) for i in range(speech_chunks)])
    # optional settle: keep generating, now on silence, so the face relaxes
    settled = np.concatenate([gen(dq, silence) for _ in range(settle_chunks)]) if settle_chunks else spoken[:0]

    def nearest(frame: np.ndarray) -> tuple:
        d = [mad(frame, loop[i]) for i in range(len(loop))]
        i = int(np.argmin(d))
        return i, float(d[i])

    def fade(a_tail: np.ndarray, b_head: np.ndarray, n: int) -> np.ndarray:
        w = np.linspace(0, 1, n + 2)[1:-1].reshape(-1, 1, 1, 1)
        return (a_tail.astype(np.float32) * (1 - w) + b_head.astype(np.float32) * w).astype(np.uint8)

    natural = float(np.mean([mad(loop[i], loop[i + 1]) for i in range(len(loop) - 1)]))
    res = {"natural_frame_step_idle": round(natural, 2),
           "intra_frame_step_speech": round(float(np.mean([mad(spoken[i], spoken[i + 1]) for i in range(len(spoken) - 1)])), 2),
           "strategies": {}}
    clips = {}

    def evaluate(name: str, last: np.ndarray, resume_at: int, fade_n: int, pre: np.ndarray) -> None:
        head = loop[resume_at : resume_at + 50]
        if len(head) < 50:                                   # wrap the loop
            head = np.concatenate([head, loop[: 50 - len(head)]])
        bridge = (fade(np.repeat(last[None], fade_n, axis=0), head[:fade_n], fade_n)
                  if fade_n else np.empty((0,) + last.shape, dtype=np.uint8))
        joined = np.concatenate([pre[-50:], bridge, head])
        # the jump the eye would see: last frame before the loop vs first loop frame shown
        before = bridge[-1] if fade_n else last
        res["strategies"][name] = {"resume_frame": resume_at, "fade_frames": fade_n,
                                   "jump_mad": round(mad(before, head[0] if not fade_n else head[fade_n]), 2),
                                   "vs_natural": round((mad(before, head[0] if not fade_n else head[fade_n])) / natural, 2)}
        clips[name] = joined

    last_spoken = spoken[-1]
    evaluate("R1_cut_to_loop_start", last_spoken, 0, 0, spoken)
    i2, d2 = nearest(last_spoken)
    evaluate("R2_cut_to_nearest", last_spoken, i2, 0, spoken)
    res["strategies"]["R2_cut_to_nearest"]["nearest_distance"] = round(d2, 2)
    evaluate("R3_nearest_plus_fade", last_spoken, i2, fade_frames, spoken)
    if settle_chunks:
        last_settled = settled[-1]
        i4, d4 = nearest(last_settled)
        evaluate("R4_settle_then_fade", last_settled, i4, fade_frames, np.concatenate([spoken, settled]))
        res["strategies"]["R4_settle_then_fade"]["nearest_distance"] = round(d4, 2)
        res["strategies"]["R4_settle_then_fade"]["settle_chunks"] = settle_chunks

    out_clips = {}
    for name, frames in clips.items():
        out_clips[name] = _write_mp4(work / f"{name}.mp4", frames, fps)
    res["seconds"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(res), flush=True)
    return {"metrics": res, "clips": out_clips}


@app.local_entrypoint()
def ret(image: str = "inputs/newscaster.png", audio: str = "inputs/korean_flashhead.wav", out_dir: str = "spike-return"):
    """생성 → 대기 루프 복귀 전환 스파이크."""
    import json as _json

    d = pathlib.Path(out_dir)
    d.mkdir(exist_ok=True)
    r = spike_return.remote(pathlib.Path(image).read_bytes(), pathlib.Path(audio).read_bytes())
    (d / "metrics.json").write_text(_json.dumps(r["metrics"], indent=1, ensure_ascii=False))
    for name, data in r["clips"].items():
        (d / f"{name}.mp4").write_bytes(data)
    m = r["metrics"]
    print(f"\n자연스러운 프레임 간 변화: 무음 {m['natural_frame_step_idle']} / 발화 {m['intra_frame_step_speech']}")
    for name, v in m["strategies"].items():
        extra = f" · 최근접 거리 {v['nearest_distance']}" if "nearest_distance" in v else ""
        print(f"  {name:24s} 복귀 점프 {v['jump_mad']:6.2f}  (자연 변화의 {v['vs_natural']:.1f}배){extra}")
    print(f"\n저장: {d}/")


# ---------------------------------------------------------------------------
# One-off A/B helper (not used by the studio): load either model, render the
# same clip in `once` mode, return the mp4 plus timings so Lite and Pro can be
# compared on identical inputs. Runs as an ephemeral `modal run`, so the
# deployed Lite Renderer is untouched.
@app.function(image=image, gpu=GPU, volumes={WEIGHTS: weights, CACHE: cache}, timeout=3600)
def render_once(
    model_type: str,
    image_bytes: bytes,
    audio_bytes: bytes,
    seed: int = 42,
    use_face_crop: bool = True,
    tail_seconds: float = 0.5,
) -> dict:
    import math
    import os
    import subprocess
    import sys
    import tempfile
    import wave

    os.chdir(SRC)
    sys.path.insert(0, SRC)
    import imageio
    import librosa
    import numpy as np
    import torch

    from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, get_pipeline, run_pipeline

    def say(msg: str) -> None:
        print(f"[{model_type}] {msg}", flush=True)

    t0 = time.time()
    pipeline = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type=model_type, wav2vec_dir=WAV2VEC)
    load_s = time.time() - t0
    say(f"pipeline loaded in {load_s:.0f}s")

    work = pathlib.Path(tempfile.mkdtemp(prefix=f"ab-{model_type}-"))
    img, wav, padded_wav, tmp, out = (work / n for n in ("ref.png", "speech.wav", "speech_padded.wav", "noaudio.mp4", "video.mp4"))
    img.write_bytes(image_bytes)
    wav.write_bytes(audio_bytes)

    t1 = time.time()
    get_base_data(pipeline, cond_image_path_or_dir=str(img), base_seed=seed, use_face_crop=use_face_crop)
    p = get_infer_params()
    sr, fps = p["sample_rate"], p["tgt_fps"]
    frame_num, motion = p["frame_num"], p["motion_frames_num"]
    slice_len = frame_num - motion
    speech, _ = librosa.load(str(wav), sr=sr, mono=True)
    speech_s = len(speech) / sr
    need = math.ceil(speech_s * fps) + math.ceil(tail_seconds * fps)
    k = max(1, math.ceil((need - motion) / slice_len))
    target = (frame_num + k * slice_len) * sr // fps
    if len(speech) < target:
        speech = np.concatenate([speech, np.zeros(target - len(speech), dtype=speech.dtype)])
    with wave.open(str(padded_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    emb = get_audio_embedding(pipeline, speech)
    n_chunks = (emb.shape[1] - frame_num) // slice_len
    prep_s = time.time() - t1
    say(f"prep {prep_s:.1f}s; {n_chunks} chunks of {slice_len} new frames (motion overlap {motion})")

    chunk_s: list[float] = []
    frames = []
    for i in range(n_chunks):
        torch.cuda.synchronize(); c0 = time.time()
        video = run_pipeline(pipeline, emb[:, i * slice_len : i * slice_len + frame_num].contiguous())
        if i:
            video = video[motion:]
        frames.append(video.cpu())
        torch.cuda.synchronize(); chunk_s.append(time.time() - c0)
        if i < 3 or i == n_chunks - 1:
            say(f"chunk {i+1}/{n_chunks} {chunk_s[-1]:.2f}s")
    t2 = time.time()
    with imageio.get_writer(str(tmp), format="mp4", mode="I", fps=fps, codec="h264", ffmpeg_params=["-bf", "0", "-crf", "17"]) as w:
        for block in frames:
            arr = block.numpy().astype(np.uint8)
            for j in range(arr.shape[0]):
                w.append_data(arr[j])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(padded_wav),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    encode_s = time.time() - t2
    total_frames = sum(int(b.shape[0]) for b in frames)
    say(f"done: {total_frames} frames, encode {encode_s:.1f}s, total {time.time()-t0:.0f}s, peak vram {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    return {
        "model_type": model_type, "mp4": out.read_bytes(), "load_s": round(load_s, 1), "prep_s": round(prep_s, 1),
        "chunk_s": [round(c, 3) for c in chunk_s], "encode_s": round(encode_s, 1), "frames": total_frames,
        "chunks": n_chunks, "slice_len": slice_len, "motion_frames": motion, "speech_s": round(speech_s, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "total_s": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def ab(image: str = "inputs/newscaster.png", audio: str = "ab/ab.wav", out_dir: str = "ab", seed: int = 42):
    """Lite vs Pro on identical inputs: `modal run app.py::ab`. Downloads Model_Pro
    into the Volume on first use, renders both in parallel containers, writes
    <out_dir>/{lite,pro}.mp4 and <out_dir>/timings.json."""
    import json

    print("weights:", fetch_weights.remote(include_pro=True), flush=True)
    img_b, wav_b = pathlib.Path(image).read_bytes(), pathlib.Path(audio).read_bytes()
    calls = {m: render_once.spawn(m, img_b, wav_b, seed=seed) for m in ("lite", "pro")}
    results = {}
    for m, call in calls.items():
        r = call.get()
        dest = pathlib.Path(out_dir) / f"{m}.mp4"
        dest.write_bytes(r.pop("mp4"))
        results[m] = r
        cs = r["chunk_s"]
        steady = sorted(cs[1:])[len(cs[1:]) // 2] if len(cs) > 1 else cs[0]
        print(f"{m:5s} load {r['load_s']}s  prep {r['prep_s']}s  chunks {r['chunks']}×{r['slice_len']}f  "
              f"chunk median {steady:.2f}s (first {cs[0]:.2f}s)  → {r['slice_len']/25/steady:.2f}× realtime  "
              f"encode {r['encode_s']}s  total {r['total_s']}s  vram {r['peak_vram_gb']} GB  → {dest}", flush=True)
    (pathlib.Path(out_dir) / "timings.json").write_text(json.dumps(results, indent=1))
