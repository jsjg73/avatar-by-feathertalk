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
import pathlib
import time

import modal

APP_NAME = "flashhead"
REPO = "https://github.com/Soul-AILab/SoulX-FlashHead.git"
SRC = "/root/SoulX-FlashHead"

# Modal has no RTX 4090 (upstream's benchmark card). L4 24 GB is the cheapest
# 24 GB option; A10 is faster but pricier. L4's 300 GB/s bandwidth is well under
# the 4090's ~1 TB/s, so expect the 96 FPS Lite figure not to carry over.
GPU = "L4"

CACHE = "/cache"
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
        f"sed -i 's/^COMPILE_MODEL = True/COMPILE_MODEL = False/; s/^COMPILE_VAE = True/COMPILE_VAE = False/' "
        f"{SRC}/flash_head/src/pipeline/flash_head_pipeline.py",
        f"grep -q '^COMPILE_MODEL = False' {SRC}/flash_head/src/pipeline/flash_head_pipeline.py "
        f"&& grep -q '^COMPILE_VAE = False' {SRC}/flash_head/src/pipeline/flash_head_pipeline.py "
        f"|| (echo 'compile flags NOT patched' && exit 1)",
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
# Default avatar, used only to warm the container in `Renderer.load`.
image = base_image.add_local_file(pathlib.Path(__file__).parent / "inputs" / "newscaster.png", "/root/warmup.png")
WARMUP_IMAGE = "/root/warmup.png"

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

WEIGHTS = "/weights"
CKPT = f"{WEIGHTS}/SoulX-FlashHead-1_3B"
WAV2VEC = f"{WEIGHTS}/wav2vec2-base-960h"
DONE_MARK = "__done__"


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


IDLE_DIR = f"{WEIGHTS}/idle_loops"
IDLE_CHUNKS = 10                      # ~9.6 s of loop; long enough that a repeat is not obvious


def _idle_loop_path(image_bytes: bytes, seed: int, model_type: str) -> pathlib.Path:
    import hashlib

    key = hashlib.sha256(image_bytes + f"|{seed}|{model_type}|{IDLE_CHUNKS}".encode()).hexdigest()[:16]
    return pathlib.Path(IDLE_DIR) / f"{key}.mp4"


def _emit(job_id: str, line: str) -> None:
    print(line, flush=True)
    try:
        progress.put(line, block=False, partition=job_id)
    except Exception:  # noqa: BLE001 - progress is best-effort, never fail a render over it
        pass


@app.cls(
    image=image,
    gpu=GPU,
    volumes={WEIGHTS: weights, CACHE: cache},
    scaledown_window=SCALEDOWN_S,
    min_containers=0,
    max_containers=1,   # the studio serialises jobs; never let a queue fan out to N GPUs
    timeout=3600,
)
class Renderer:
    # "lite" (default) or "pro". Each value gets its own container pool and its
    # own autoscaler settings, so Renderer(model_type="pro") never competes with
    # the resident Lite streamer for the GPU, and scales to zero on its own.
    model_type: str = modal.parameter(default="lite")

    @modal.enter()
    def load(self) -> None:
        import os
        import sys
        import threading

        # flash_head/inference.py opens flash_head/configs/infer_params.yaml by a
        # path relative to the CWD at import time, so chdir first.
        os.chdir(SRC)
        sys.path.insert(0, SRC)
        t0 = time.time()
        from flash_head.inference import get_pipeline

        self.pipeline = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type=self.model_type, wav2vec_dir=WAV2VEC)
        load_s = time.time() - t0
        # First-call costs (mediapipe init, cuDNN autotune, first VAE/denoise
        # pass) were 24 s on the first render of a fresh container and 0.3 s
        # after. Pay them here so a pre-warmed container answers in seconds.
        t1 = time.time()
        import numpy as np
        from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline

        get_base_data(self.pipeline, cond_image_path_or_dir=WARMUP_IMAGE, base_seed=0, use_face_crop=True)
        p = get_infer_params()
        n = p["frame_num"] * p["sample_rate"] // p["tgt_fps"]
        emb = get_audio_embedding(self.pipeline, np.zeros(n, dtype=np.float32))
        video = run_pipeline(self.pipeline, emb[:, : p["frame_num"]].contiguous())
        gpu_s = time.time() - t1
        # librosa JIT-compiles (numba) on first use and imageio spins up its
        # ffmpeg plugin on first write — together ~25 s on a fresh container.
        t2 = time.time()
        import tempfile

        import wave

        import imageio
        import librosa

        with tempfile.TemporaryDirectory() as d:
            with wave.open(f"{d}/w.wav", "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(p["sample_rate"]); w.writeframes(b"\x00\x00" * n)
            librosa.load(f"{d}/w.wav", sr=p["sample_rate"], mono=True)
            with imageio.get_writer(f"{d}/w.mp4", format="mp4", mode="I", fps=p["tgt_fps"], codec="h264", ffmpeg_params=["-bf", "0"]) as w:
                w.append_data(video[0].cpu().numpy().astype(np.uint8))
        io_s = time.time() - t2
        self.load_seconds = time.time() - t0
        self.loaded_at = time.time()
        self._ready = {"since": self.loaded_at, "weights_s": round(load_s), "warmup_s": round(gpu_s + io_s)}

        # The flag carries a heartbeat: a deploy replaces this container and the
        # exit hook is not guaranteed to run, so readers treat a flag whose beat
        # is older than ~60 s as stale instead of trusting its mere presence.
        ready_key = f"ready:{self.model_type}"

        def beat() -> None:
            while True:
                try:
                    state[ready_key] = {**self._ready, "beat": time.time()}
                except Exception:  # noqa: BLE001 - best effort
                    pass
                time.sleep(20)

        threading.Thread(target=beat, daemon=True, name="ready-beat").start()
        print(f"pipeline resident ({self.model_type}): weights {load_s:.0f}s + gpu warmup {gpu_s:.0f}s + io warmup {io_s:.0f}s", flush=True)

    @modal.exit()
    def unload(self) -> None:
        state.pop(f"ready:{self.model_type}", None)
        print("container exiting; ready flag cleared", flush=True)

    @modal.method()
    def ping(self) -> dict:
        """Cheap liveness check for a *warm* container. Calling this on a cold
        class spins one up, so the studio never uses it for status polling."""
        return {"warm": True, "model_type": self.model_type, "load_seconds": round(self.load_seconds, 1), "resident_for": round(time.time() - self.loaded_at)}

    @modal.method()
    def render(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        job_id: str,
        seed: int = 42,
        use_face_crop: bool = True,
        tail_seconds: float = 0.5,
    ) -> bytes:
        """One clip: reference image + 16 kHz mono WAV -> mp4 bytes.

        Mirrors the `once` branch of upstream generate_video.py::generate —
        encode the whole audio in one wav2vec2 pass, slice per chunk — against
        the already-loaded pipeline. Only the reference image and seed are
        re-prepared per call.

        `tail_seconds` is the minimum silence the video keeps after the speech
        ends. Upstream's `once` padding only completes the last chunk, and its
        chunk count is one short of what the audio allows, so the video always
        came out 24 frames (0.96 s) shorter than the padded audio — depending
        on where the speech fell on the 0.96 s chunk grid, up to that much of
        the last word was cut at the mux. Here the audio is padded with silence
        until the rendered frames cover speech + tail (the grid makes the
        actual tail land in [tail, tail + 0.96 s)).
        """
        import math
        import os
        import subprocess
        import tempfile
        import threading
        import wave

        import imageio
        import librosa
        import numpy as np
        import torch

        from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline

        t0 = time.time()
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
        img = work / "ref.png"
        wav = work / "speech.wav"
        out = work / "video.mp4"
        img.write_bytes(image_bytes)
        wav.write_bytes(audio_bytes)

        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(10):
                _emit(job_id, f"[{time.time()-t0:5.1f}s] .. running, vram {torch.cuda.memory_allocated()/1e9:.1f} GB")

        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            _emit(job_id, f"[{time.time()-t0:5.1f}s] preparing reference (face crop, seed {seed})")
            get_base_data(self.pipeline, cond_image_path_or_dir=str(img), base_seed=seed, use_face_crop=use_face_crop)
            _emit(job_id, f"[{time.time()-t0:5.1f}s] reference ready")
            p = get_infer_params()
            sr, fps = p["sample_rate"], p["tgt_fps"]
            frame_num, motion_frames_num = p["frame_num"], p["motion_frames_num"]
            slice_len = frame_num - motion_frames_num

            speech, _ = librosa.load(str(wav), sr=sr, mono=True)
            speech_s = len(speech) / sr
            _emit(job_id, f"[{time.time()-t0:5.1f}s] audio loaded ({speech_s:.2f}s)")
            # k chunks render frame_num + (k-1)*slice_len frames and, with the
            # upstream chunk-count rule kept below, need frame_num + k*slice_len
            # frames of audio (each rendered frame sees >= one slice of
            # look-ahead, as upstream). Pick k so the video covers speech + tail.
            need_frames = math.ceil(speech_s * fps) + math.ceil(tail_seconds * fps)
            k = max(1, math.ceil((need_frames - motion_frames_num) / slice_len))
            audio_frames = frame_num + k * slice_len
            target = audio_frames * sr // fps
            if len(speech) < target:
                speech = np.concatenate([speech, np.zeros(target - len(speech), dtype=speech.dtype)])
            video_s = (motion_frames_num + k * slice_len) / fps
            _emit(job_id, f"[{time.time()-t0:5.1f}s] speech {speech_s:.2f}s → video {video_s:.2f}s ({k} chunks, tail ≥{tail_seconds:.1f}s)")
            # Mux the *padded* audio: it is longer than the video, so `-shortest`
            # trims silence off the audio instead of trimming frames off the video.
            padded_wav = work / "speech_padded.wav"
            with wave.open(str(padded_wav), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

            _emit(job_id, f"[{time.time()-t0:5.1f}s] encoding audio ({len(speech)/sr:.1f}s) in one pass")
            emb = get_audio_embedding(self.pipeline, speech)
            n_chunks = (emb.shape[1] - frame_num) // slice_len
            chunks = [emb[:, i * slice_len : i * slice_len + frame_num].contiguous() for i in range(n_chunks)]
            _emit(job_id, f"[{time.time()-t0:5.1f}s] {n_chunks} chunks to generate")

            frames = []
            for i, chunk in enumerate(chunks):
                torch.cuda.synchronize()
                c0 = time.time()
                video = run_pipeline(self.pipeline, chunk)
                if i != 0:
                    video = video[motion_frames_num:]
                torch.cuda.synchronize()
                frames.append(video.cpu())
                _emit(job_id, f"[{time.time()-t0:5.1f}s] chunk {i+1}/{n_chunks} done ({time.time()-c0:.2f}s)")

            _emit(job_id, f"[{time.time()-t0:5.1f}s] encoding mp4")
            tmp = work / "video_noaudio.mp4"
            with imageio.get_writer(str(tmp), format="mp4", mode="I", fps=fps, codec="h264", ffmpeg_params=["-bf", "0"]) as w:
                for block in frames:
                    arr = block.numpy().astype(np.uint8)
                    for k in range(arr.shape[0]):
                        w.append_data(arr[k])
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(padded_wav),
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
                check=True,
            )
            total = time.time() - t0
            _emit(job_id, f"[{total:5.1f}s] {DONE_MARK} {n_chunks} chunks, {out.stat().st_size/1e6:.1f} MB")
            return out.read_bytes()
        finally:
            stop.set()
            try:
                for f in work.iterdir():
                    f.unlink()
                work.rmdir()
            except OSError:
                pass


    @modal.method()
    def live(
        self,
        session_id: str,
        image_bytes: bytes,
        seed: int = 42,
        use_face_crop: bool = True,
        idle_timeout_s: float = 180.0,
        max_seconds: float = 1800.0,
        lead_chunks: float = 1.0,
        use_idle_loop: bool = True,
        fade_frames: int = 4,
    ) -> dict:
        """A live session: upstream *stream* mode paced to the wall clock.

        Every 0.96 s (one chunk) the loop takes the next 0.96 s of speech from
        the Queue partition `<session_id>:audio` — or a silent chunk if nothing
        is queued, which keeps the avatar idling — runs wav2vec2 over the 8 s
        rolling window and generates 24 new frames, then ships HLS segments to
        partition `<session_id>` like render_stream. Ends on an {"type": "end"}
        item, after `idle_timeout_s` without speech, or at `max_seconds`.

        With `use_idle_loop` the silent chunks are not generated at all: a loop
        rendered once per avatar (cached in the weights Volume) plays instead, so
        the GPU only works while the avatar speaks. Both hand-offs were measured
        (`spike_seam`, `spike_return`): entering generation from the loop is free
        — the loop's last 9 frames go into `latent_motion_frames`, which is what
        the model conditions on anyway — while returning needs a short cross-fade,
        since a hard cut jumps 6.25 MAD against a natural frame step of 2.7 and a
        4-frame fade brings it to 1.66.

        Latency budget on a warm L4 (Lite): audio put→get ≈0.7 s, wait for the
        next chunk slot ≤0.96 s, generate 0.8 s, segment closes with the next
        chunk ≈0.96 s, ship ≈0.7 s, player 1–2 s behind the live edge.
        """
        import os
        import tempfile
        import threading
        from collections import deque

        import numpy as np
        import torch

        from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline

        t0 = time.time()
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"live-{session_id}-"))
        hls = work / "hls"
        hls.mkdir()
        (work / "ref.png").write_bytes(image_bytes)
        ship = _HlsShipper(session_id, hls, t0)
        audio_part = f"{session_id}:audio"
        proc = None
        stats = {"chunks": 0, "speech_chunks": 0, "gpu_chunks": 0, "gen_s": [], "behind_s": [], "ended_by": ""}
        try:
            ship.log(f"live: preparing reference (face crop, seed {seed})")
            get_base_data(self.pipeline, cond_image_path_or_dir=str(work / "ref.png"), base_seed=seed, use_face_crop=use_face_crop)
            p = get_infer_params()
            sr, fps = p["sample_rate"], p["tgt_fps"]
            frame_num, motion = p["frame_num"], p["motion_frames_num"]
            slice_len = frame_num - motion
            chunk_samples = slice_len * sr // fps                 # 15360 = 0.96 s
            chunk_s = slice_len / fps
            cached = p["cached_audio_duration"] * sr              # 8 s rolling window
            audio_end_idx = p["cached_audio_duration"] * fps
            audio_start_idx = audio_end_idx - frame_num
            audio_dq: deque = deque([0.0] * cached, maxlen=cached)

            fifo = str(work / "audio.pipe")
            os.mkfifo(fifo)
            proc = _live_ffmpeg(hls, p["width"], p["height"], fps, sr, fifo, slice_len)
            # two writers: ffmpeg interleaves by timestamp, so each stream needs
            # its own feeder or the muxer deadlocks waiting on the other pipe
            import queue as _q

            vq: _q.Queue = _q.Queue()
            aq: _q.Queue = _q.Queue()

            def vwriter() -> None:
                assert proc is not None and proc.stdin is not None
                try:
                    while (arr := vq.get()) is not None:
                        proc.stdin.write(arr.tobytes())
                finally:
                    proc.stdin.close()

            def awriter() -> None:
                with open(fifo, "wb") as f:              # blocks until ffmpeg opens the reader
                    while (pcm := aq.get()) is not None:
                        f.write(pcm)

            threading.Thread(target=vwriter, daemon=True).start()
            threading.Thread(target=awriter, daemon=True).start()

            pending: deque = deque()                      # float32 arrays of chunk_samples
            buf = np.zeros(0, dtype=np.float32)            # sentence PCM not yet cut into chunks
            ended = False
            last_speech = time.time()
            stream_t0 = None
            k = 0
            # ---- idle loop: rendered once per avatar, then reused by every session
            idle_loop = None
            if use_idle_loop:
                import imageio

                loop_path = _idle_loop_path(image_bytes, seed, self.model_type)
                loop_path.parent.mkdir(parents=True, exist_ok=True)
                if not loop_path.exists():
                    t_loop = time.time()
                    dq: deque = deque([0.0] * cached, maxlen=cached)
                    made = []
                    for _ in range(IDLE_CHUNKS):
                        dq.extend(np.zeros(chunk_samples, dtype=np.float32).tolist())
                        e = get_audio_embedding(self.pipeline, np.array(dq, dtype=np.float32), audio_start_idx, audio_end_idx)
                        made.append(run_pipeline(self.pipeline, e)[motion:].cpu().numpy().astype(np.uint8))
                    tmp_loop = loop_path.with_name(f".{loop_path.stem}.{os.getpid()}.tmp.mp4")
                    with imageio.get_writer(str(tmp_loop), format="mp4", mode="I", fps=fps, codec="h264",
                                            ffmpeg_params=["-bf", "0", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"]) as w:
                        for block in made:
                            for f in block:
                                w.append_data(f)
                    tmp_loop.replace(loop_path)
                    weights.commit()
                    ship.log(f"live: idle loop rendered in {time.time()-t_loop:.0f}s → {loop_path.name}")
                reader = imageio.get_reader(str(loop_path))
                idle_loop = np.stack([np.asarray(f) for f in reader])
                reader.close()
                ship.log(f"live: idle loop ready ({len(idle_loop)} frames) — GPU rests while the avatar listens")
            loop_pos = 0
            speaking = False
            last_frame = None

            ship.log(f"live: ready — {chunk_s:.2f}s chunks, idle timeout {idle_timeout_s:.0f}s")
            while True:
                # ingest speech / control items
                for item in progress.get_many(50, block=False, partition=audio_part):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "end":
                        ended = True
                    elif item.get("type") == "audio":
                        raws = item.get("chunks") or [item.get("pcm", b"")]
                        arr = np.concatenate([np.frombuffer(r, dtype=np.int16).astype(np.float32) / 32768.0 for r in raws]) if raws else np.zeros(0, np.float32)
                        buf = np.concatenate([buf, arr])
                        while len(buf) >= chunk_samples:
                            pending.append(buf[:chunk_samples]); buf = buf[chunk_samples:]
                        if item.get("final", True) and len(buf):   # pad the sentence tail instead of waiting for more speech
                            pending.append(np.concatenate([buf, np.zeros(chunk_samples - len(buf), dtype=np.float32)])); buf = buf[:0]
                now = time.time()
                if ended and not pending:
                    stats["ended_by"] = "stop"; break
                if not pending and now - last_speech > idle_timeout_s:
                    stats["ended_by"] = "idle"; break
                if now - t0 > max_seconds:
                    stats["ended_by"] = "max_seconds"; break
                # pace to the wall clock, staying `lead_chunks` ahead of playback time
                if stream_t0 is None:
                    stream_t0 = now
                sched = stream_t0 + (k - lead_chunks) * chunk_s
                if now < sched:
                    time.sleep(min(sched - now, 0.25))
                    continue                              # re-check the queue while waiting
                behind = now - sched
                if pending:
                    pcm = pending.popleft(); is_speech = True; last_speech = now
                else:
                    pcm = np.zeros(chunk_samples, dtype=np.float32); is_speech = False
                g0 = time.time()
                audio_dq.extend(pcm.tolist())          # the 8 s window advances either way
                if idle_loop is not None and not is_speech:
                    frames = np.take(idle_loop, range(loop_pos, loop_pos + slice_len), axis=0, mode="wrap")
                    loop_pos = (loop_pos + slice_len) % len(idle_loop)
                    if speaking:                       # first silent chunk after a turn: fade back into the loop
                        frames = frames.copy()
                        wts = np.linspace(0, 1, fade_frames + 2)[1:-1].reshape(-1, 1, 1, 1)
                        frames[:fade_frames] = (last_frame.astype(np.float32) * (1 - wts)
                                                + frames[:fade_frames].astype(np.float32) * wts).astype(np.uint8)
                        speaking = False
                    gen = 0.0
                else:
                    if idle_loop is not None and not speaking:
                        # entering generation: continue from the frames the viewer is seeing
                        seen = np.take(idle_loop, range(loop_pos - motion, loop_pos), axis=0, mode="wrap")
                        t = torch.from_numpy(np.ascontiguousarray(seen)).to(self.pipeline.device, dtype=self.pipeline.param_dtype)
                        self.pipeline.latent_motion_frames = self.pipeline.vae.encode(((t / 255 - 0.5) * 2).permute(3, 0, 1, 2).unsqueeze(0))
                        speaking = True
                    emb = get_audio_embedding(self.pipeline, np.array(audio_dq, dtype=np.float32), audio_start_idx, audio_end_idx)
                    frames = run_pipeline(self.pipeline, emb)[motion:].cpu().numpy().astype(np.uint8)
                    gen = time.time() - g0
                vq.put(frames)
                last_frame = frames[-1]
                aq.put((np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
                k += 1
                stats["chunks"] = k; stats["speech_chunks"] += int(is_speech)
                stats["gpu_chunks"] += int(gen > 0)
                if gen > 0:
                    stats["gen_s"].append(round(gen, 3))
                stats["behind_s"].append(round(behind, 2))
                ship.ship_new_files()
                if is_speech or k % 10 == 0 or k <= 3:
                    ship.log(f"live: chunk {k} {'speech' if is_speech else 'silence'} gen {gen:.2f}s behind {behind:.2f}s queued {len(pending)*chunk_s:.1f}s")
            # finalize
            vq.put(None); aq.put(None)
            rc = proc.wait(timeout=120)
            ship.ship_new_files()
            saved = 100 * (1 - stats["gpu_chunks"] / k) if k else 0
            ship.log(f"{DONE_MARK} live ended ({stats['ended_by']}): {k} chunks, {stats['speech_chunks']} speech, "
                     f"GPU {stats['gpu_chunks']} chunks ({saved:.0f}% saved), {ship.streamed/1e6:.1f} MB")
            ship.close()
            stats["gen_median_s"] = sorted(stats["gen_s"])[len(stats["gen_s"]) // 2] if stats["gen_s"] else None
            stats["gpu_chunk_ratio"] = round(stats["gpu_chunks"] / k, 3) if k else None
            stats["behind_max_s"] = max(stats["behind_s"]) if stats["behind_s"] else None
            stats["ffmpeg_rc"] = rc
            stats.pop("gen_s"); stats.pop("behind_s")
            return stats
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
            if ship.thread.is_alive():
                ship.q.put(None)
            try:
                for f in sorted(work.rglob("*"), reverse=True):
                    f.unlink() if f.is_file() else f.rmdir()
                work.rmdir()
            except OSError:
                pass

    @modal.method()
    def render_stream(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        job_id: str,
        seed: int = 42,
        use_face_crop: bool = True,
        tail_seconds: float = 0.5,
    ) -> dict:
        """`render()` for live playback: the same `once`-mode generation, but frames
        go into one long-lived ffmpeg that emits HLS (fMP4) segments of one chunk
        (0.96 s) each, and every finished segment is pushed to the progress Queue
        (partition = job_id) the moment ffmpeg closes it. A player can start ~2
        chunks in and never catch up, because generation runs at 1.17x realtime
        (eager) on the L4.

        Queue items: {"type": "log", "line": str} and
        {"type": "file", "name": "init.mp4" | "seg_000.m4s" | "index.m3u8",
         "part": i, "parts": n, "data": bytes}  (files > FILE_PART bytes are split;
        Queue items are capped at 1 MiB). Files precede the playlist that lists
        them; the final playlist carries EXT-X-ENDLIST. The caller remuxes to mp4.

        Delivery goes through the Queue rather than a Modal generator: generator
        outputs were measured arriving 5-20 s late and in bursts, which starves
        the player; Queue put->get is 0.6-0.8 s.
        """
        import math
        import queue
        import re
        import subprocess
        import tempfile
        import threading
        import wave

        import librosa
        import numpy as np
        import torch

        from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline

        FILE_PART = 900_000
        t0 = time.time()
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
        hls = work / "hls"
        hls.mkdir()
        img, wav, padded_wav = work / "ref.png", work / "speech.wav", work / "speech_padded.wav"
        img.write_bytes(image_bytes)
        wav.write_bytes(audio_bytes)

        # Everything a chunk produces (log lines, segment, playlist) travels as ONE
        # Queue item: the consumer pays a ~0.5 s round trip per item, and 3 items
        # per 0.8 s chunk fell ~10 s behind by chunk 20. Batches stay < 1 MiB.
        batch: list[dict] = []
        batch_bytes = [0]
        # Queue.put is a network round trip (~0.1 s); done inline it cost the GPU
        # loop ~0.1 s per chunk (0.82 -> 0.91 s). A single sender thread keeps
        # FIFO order and takes it off the critical path.
        send_q: queue.Queue = queue.Queue()
        send_errors: list[str] = []

        def sender() -> None:
            while (item := send_q.get()) is not None:
                try:
                    progress.put(item, block=False, partition=job_id)
                except Exception as exc:  # noqa: BLE001 - keep streaming; report at the end
                    send_errors.append(f"{type(exc).__name__}: {exc}"[:200])

        sender_thread = threading.Thread(target=sender, daemon=True)
        sender_thread.start()

        def flush() -> None:
            if batch:
                send_q.put({"type": "batch", "items": list(batch)})
                batch.clear()
                batch_bytes[0] = 0

        def put(item: dict) -> None:
            size = len(item.get("data", b"")) + 200
            if batch and batch_bytes[0] + size > FILE_PART:
                flush()
            batch.append(item)
            batch_bytes[0] += size

        def log(line: str) -> None:
            line = f"[{time.time()-t0:5.1f}s] {line}"
            print(line, flush=True)
            put({"type": "log", "line": line})
            flush()                              # log lines go out immediately, not with the next segment

        def send_file(name: str, data: bytes) -> None:
            parts = max(1, math.ceil(len(data) / FILE_PART))
            for i in range(parts):
                put({"type": "file", "name": name, "part": i, "parts": parts, "data": data[i * FILE_PART : (i + 1) * FILE_PART]})

        proc = None
        try:
            log(f"preparing reference (face crop, seed {seed})")
            get_base_data(self.pipeline, cond_image_path_or_dir=str(img), base_seed=seed, use_face_crop=use_face_crop)
            p = get_infer_params()
            sr, fps = p["sample_rate"], p["tgt_fps"]
            frame_num, motion_frames_num = p["frame_num"], p["motion_frames_num"]
            slice_len = frame_num - motion_frames_num
            height, width = p["height"], p["width"]

            speech, _ = librosa.load(str(wav), sr=sr, mono=True)
            speech_s = len(speech) / sr
            need_frames = math.ceil(speech_s * fps) + math.ceil(tail_seconds * fps)
            k = max(1, math.ceil((need_frames - motion_frames_num) / slice_len))
            target = (frame_num + k * slice_len) * sr // fps
            if len(speech) < target:
                speech = np.concatenate([speech, np.zeros(target - len(speech), dtype=speech.dtype)])
            video_s = (motion_frames_num + k * slice_len) / fps
            with wave.open(str(padded_wav), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            log(f"speech {speech_s:.2f}s → video {video_s:.2f}s ({k} chunks, tail ≥{tail_seconds:.1f}s)")
            flush()

            emb = get_audio_embedding(self.pipeline, speech)
            n_chunks = (emb.shape[1] - frame_num) // slice_len
            chunks = [emb[:, i * slice_len : i * slice_len + frame_num].contiguous() for i in range(n_chunks)]

            # One keyframe per chunk (GOP = slice_len) so each HLS segment is exactly
            # one chunk and closes as soon as the next chunk's first frame arrives.
            # temp_file: segments/playlist appear atomically, so anything listed in
            # index.m3u8 is complete. Audio is cut to the video length up front
            # instead of relying on -shortest against a slow pipe.
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
                "-t", f"{video_s:.3f}", "-i", str(padded_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-crf", "20",
                "-g", str(slice_len), "-keyint_min", str(slice_len), "-sc_threshold", "0", "-bf", "0",
                "-c:a", "aac", "-b:a", "128k",
                "-f", "hls", "-hls_time", f"{slice_len / fps:.2f}", "-hls_segment_type", "fmp4",
                "-hls_playlist_type", "event", "-hls_list_size", "0",
                "-hls_flags", "independent_segments+temp_file",
                "-hls_fmp4_init_filename", "init.mp4",
                "-hls_segment_filename", str(hls / "seg_%03d.m4s"),
                str(hls / "index.m3u8"),
            ]
            ffmpeg_log = open(work / "ffmpeg.log", "wb")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffmpeg_log)
            frames_q: queue.Queue = queue.Queue()

            def writer() -> None:
                # decoupled from the GPU loop so a slow encode never stalls generation
                assert proc is not None and proc.stdin is not None
                try:
                    while (arr := frames_q.get()) is not None:
                        proc.stdin.write(arr.tobytes())
                finally:
                    proc.stdin.close()

            wt = threading.Thread(target=writer, daemon=True)
            wt.start()

            sent: set[str] = set()
            last_playlist = [""]
            seg_re = re.compile(r"^(seg_\d+\.m4s)$", re.M)
            streamed = [0]

            def ship_new_files() -> None:
                pl = hls / "index.m3u8"
                if pl.exists():
                    text = pl.read_text()
                    # The fMP4 init file is opened at start but only gets its moov
                    # (delay_moov) when the first segment is flushed — i.e. right
                    # before the playlist first lists a segment. Reading it earlier
                    # shipped 0 bytes once; so ship it only then, and only if non-empty.
                    init = hls / "init.mp4"
                    if "init.mp4" not in sent and seg_re.search(text) and init.exists() and init.stat().st_size > 0:
                        sent.add("init.mp4")
                        data = init.read_bytes(); streamed[0] += len(data); send_file("init.mp4", data)
                    if "init.mp4" not in sent:
                        return                      # never advertise segments before their init
                    for name in seg_re.findall(text):
                        if name not in sent and (hls / name).exists():
                            sent.add(name)
                            data = (hls / name).read_bytes(); streamed[0] += len(data); send_file(name, data)
                    if text != last_playlist[0] and text.endswith("\n"):
                        last_playlist[0] = text
                        send_file("index.m3u8", text.encode())

            log(f"{n_chunks} chunks to generate, streaming {slice_len / fps:.2f}s segments")
            for i, chunk in enumerate(chunks):
                c0 = time.time()
                video = run_pipeline(self.pipeline, chunk)
                if i != 0:
                    video = video[motion_frames_num:]
                frames_q.put(video.cpu().numpy().astype(np.uint8))
                log(f"chunk {i+1}/{n_chunks} done ({time.time()-c0:.2f}s)")
                ship_new_files()
                flush()
            frames_q.put(None)
            wt.join(timeout=120)
            rc = proc.wait(timeout=120)
            ffmpeg_log.close()
            if rc != 0:
                raise RuntimeError("ffmpeg failed: " + (work / "ffmpeg.log").read_text()[-2000:])
            ship_new_files()
            if "#EXT-X-ENDLIST" not in last_playlist[0]:
                raise RuntimeError("ffmpeg exited without finalizing the playlist")
            log(f"{DONE_MARK} {n_chunks} chunks, {len(sent)-1} segments, {streamed[0]/1e6:.1f} MB streamed")
            flush()
            # everything must be in the Queue before the call returns: the studio
            # does its final drain right after FunctionCall.get() succeeds
            send_q.put(None)
            sender_thread.join(timeout=120)
            if send_errors:
                raise RuntimeError(f"{len(send_errors)} Queue puts failed, e.g. {send_errors[0]}")
            return {"chunks": n_chunks, "segments": len(sent) - 1, "bytes": streamed[0], "seconds": round(time.time() - t0, 1)}
        finally:
            if sender_thread.is_alive():
                send_q.put(None)
            if proc is not None and proc.poll() is None:
                proc.kill()
            try:
                for f in sorted(work.rglob("*"), reverse=True):
                    f.unlink() if f.is_file() else f.rmdir()
                work.rmdir()
            except OSError:
                pass


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
    .add_local_python_source("live_worker")
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
        ship = _HlsShipper(session_id, hls, t0)
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
# Live session support: HLS shipping shared by the live loop (segments and
# log lines batched into Queue items, sent from a thread, see render_stream).
class _HlsShipper:
    FILE_PART = 900_000

    def __init__(self, partition: str, hls_dir: pathlib.Path, t0: float):
        import queue
        import re
        import threading

        self.partition, self.hls, self.t0 = partition, hls_dir, t0
        self.batch: list[dict] = []
        self.batch_bytes = 0
        self.sent: set[str] = set()
        self.last_playlist = ""
        self.streamed = 0
        self.errors: list[str] = []
        self.seg_re = re.compile(r"^(seg_\d+\.m4s)$", re.M)
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._sender, daemon=True)
        self.thread.start()

    def _sender(self) -> None:
        while (item := self.q.get()) is not None:
            try:
                progress.put(item, block=False, partition=self.partition)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}"[:200])

    def put(self, item: dict) -> None:
        size = len(item.get("data", b"")) + 200
        if self.batch and self.batch_bytes + size > self.FILE_PART:
            self.flush()
        self.batch.append(item)
        self.batch_bytes += size

    def flush(self) -> None:
        if self.batch:
            self.q.put({"type": "batch", "items": list(self.batch)})
            self.batch.clear()
            self.batch_bytes = 0

    def log(self, line: str) -> None:
        line = f"[{time.time()-self.t0:6.1f}s] {line}"
        print(line, flush=True)
        self.put({"type": "log", "line": line})
        self.flush()

    def send_file(self, name: str, data: bytes) -> None:
        import math

        parts = max(1, math.ceil(len(data) / self.FILE_PART))
        for i in range(parts):
            self.put({"type": "file", "name": name, "part": i, "parts": parts,
                      "data": data[i * self.FILE_PART : (i + 1) * self.FILE_PART]})
        self.streamed += len(data)

    def ship_new_files(self) -> int:
        """Ship init (once it is complete), newly listed segments, then the playlist."""
        pl = self.hls / "index.m3u8"
        if not pl.exists():
            return 0
        text = pl.read_text()
        n = 0
        init = self.hls / "init.mp4"
        if "init.mp4" not in self.sent and self.seg_re.search(text) and init.exists() and init.stat().st_size > 0:
            self.sent.add("init.mp4")
            self.send_file("init.mp4", init.read_bytes())
        if "init.mp4" not in self.sent:
            return 0
        for name in self.seg_re.findall(text):
            if name not in self.sent and (self.hls / name).exists():
                self.sent.add(name)
                self.send_file(name, (self.hls / name).read_bytes())
                n += 1
        if text != self.last_playlist and text.endswith("\n"):
            self.last_playlist = text
            self.send_file("index.m3u8", text.encode())
        self.flush()
        return n

    def close(self) -> None:
        self.flush()
        self.q.put(None)
        self.thread.join(timeout=120)


def _live_ffmpeg(hls: pathlib.Path, width: int, height: int, fps: int, sr: int, audio_fifo: str, slice_len: int):
    """Long-lived HLS encoder for a live session: raw frames on stdin, raw PCM on a FIFO."""
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-thread_queue_size", "1024", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-thread_queue_size", "1024", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", audio_fifo,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-crf", "20",
        "-g", str(slice_len), "-keyint_min", str(slice_len), "-sc_threshold", "0", "-bf", "0",
        "-c:a", "aac", "-b:a", "128k",
        "-f", "hls", "-hls_time", f"{slice_len / fps:.2f}", "-hls_segment_type", "fmp4",
        "-hls_playlist_type", "event", "-hls_list_size", "0",
        "-hls_flags", "independent_segments+temp_file",
        "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", str(hls / "seg_%04d.m4s"),
        str(hls / "index.m3u8"),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=open(hls.parent / "ffmpeg.log", "wb"))


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
        return run_pipeline(pipeline, emb)[motion:].cpu().numpy().astype(np.uint8)

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


def _write_mp4(path: pathlib.Path, frames, fps: int) -> bytes:
    import imageio

    with imageio.get_writer(str(path), format="mp4", mode="I", fps=fps, codec="h264",
                            ffmpeg_params=["-bf", "0", "-crf", "18", "-pix_fmt", "yuv420p"]) as w:
        for f in frames:
            w.append_data(f)
    return path.read_bytes()


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
        return run_pipeline(pipeline, emb)[motion:].cpu().numpy().astype(np.uint8)

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
