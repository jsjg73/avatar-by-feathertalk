"""Pro real-time benchmark on big GPUs: upstream's own multi-GPU path
(`torchrun` + xfuser Ulysses sequence parallel, stream mode) with flash-attn 2
or SageAttention, compile on/off, pinned to a region. Reports per-chunk times.

    modal run bench_multi.py                 # all runs in parallel, JSON to bench/

Ephemeral: nothing here touches the deployed studio app.
"""
import json
import os
import pathlib
import re
import subprocess
import threading
import time

import modal

from app import CACHE, CKPT, SRC, WAV2VEC, WEIGHTS, base_image as _base, cache, weights  # noqa: E402

# Prebuilt FA2 wheel for torch 2.6 / cu12 / cp311 (cxx11abi FALSE matches the PyPI torch 2.6 wheels).
FA_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
# USP attention (xfuser/yunchang) needs FA2 or SageAttention; SDPA is not an option there.
# xfuser 0.6.0 (released 2026-09-01, what the base image resolved) imports
# `update_npu_out` from a yunchang that has not been released -> USP forward
# dies with ImportError on every rank. FlashHead was built against 0.4.x.
PINS = ("xfuser==0.4.5",)
image_fa = _base.pip_install(FA_WHEEL, *PINS).add_local_python_source("app")
image_sage = _base.pip_install(FA_WHEEL, "sageattention==1.0.6", *PINS).add_local_python_source("app")

app = modal.App("flashhead-bench")
REGION = ["jp"]
CHUNK_RE = re.compile(r"chunk-(\d+) done, cost time: ([\d.]+)s")
STEP_RE = re.compile(r"denoise per step: ([\d.]+)s")


def _bench(label: str, nproc: int, compile_on: bool, model_type: str, image_bytes: bytes, audio_bytes: bytes,
           mode: str = "stream", max_seconds: int = 1500) -> dict:
    import shutil

    t0 = time.time()
    env_info: dict = {}
    for mod in ("flash_attn", "sageattention", "xfuser", "yunchang"):
        try:
            m = __import__(mod)
            env_info[mod] = getattr(m, "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            env_info[mod] = f"missing ({type(exc).__name__})"
    src = SRC
    if compile_on:
        # the image ships with compile patched off; flip it on a private copy
        src = "/tmp/src"
        shutil.copytree(SRC, src, dirs_exist_ok=True)
        pipe = f"{src}/flash_head/src/pipeline/flash_head_pipeline.py"
        subprocess.run(["sed", "-i", "s/^COMPILE_MODEL = False/COMPILE_MODEL = True/; s/^COMPILE_VAE = False/COMPILE_VAE = True/", pipe], check=True)
    work = pathlib.Path("/tmp/in"); work.mkdir(exist_ok=True)
    (work / "ref.png").write_bytes(image_bytes)
    (work / "speech.wav").write_bytes(audio_bytes)
    out = "/tmp/out.mp4"
    cmd = ["torchrun", "--nproc_per_node", str(nproc), "--master_port", "29512", "generate_video.py",
           "--ckpt_dir", CKPT, "--wav2vec_dir", WAV2VEC, "--model_type", model_type,
           "--cond_image", str(work / "ref.png"), "--audio_path", str(work / "speech.wav"),
           "--audio_encode_mode", mode, "--save_file", out]
    env = {**os.environ, "NCCL_MIN_NCHANNELS": "4", "PYTHONUNBUFFERED": "1", "NCCL_DEBUG": "WARN", "TORCHELASTIC_ERROR_FILE": "/tmp/torchelastic_error.json"}
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=src, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    chunks: dict[int, float] = {}
    steps: list[float] = []
    tail: list[str] = []
    full: list[str] = []
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(30):
            smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip().replace("\n", " | ")
            print(f"[{label}] .. {time.time()-t0:5.0f}s, chunks {len(chunks)}, gpu {smi}", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        tail.append(line[:300]); tail[:] = tail[-60:]
        if len(full) < 4000:
            full.append(line[:400])
        m = CHUNK_RE.search(line)
        if m:
            chunks[int(m.group(1))] = float(m.group(2))
            if len(chunks) <= 3 or len(chunks) % 10 == 0:
                print(f"[{label}] {time.time()-t0:5.0f}s chunk-{m.group(1)} {m.group(2)}s", flush=True)
        m = STEP_RE.search(line)
        if m:
            steps.append(float(m.group(1)))
        if time.time() - t0 > max_seconds:
            proc.kill()
            break
    rc = proc.wait()
    stop.set()
    order = [chunks[i] for i in sorted(chunks)]
    steady = sorted(order[1:])[len(order[1:]) // 2] if len(order) > 1 else (order[0] if order else None)
    res = {"label": label, "nproc": nproc, "compile": compile_on, "model": model_type, "mode": mode, "rc": rc,
           "env": env_info, "wall_s": round(time.time() - t0, 1), "chunks": len(order),
           "chunk_first_s": order[0] if order else None, "chunk_median_s": steady,
           "chunk_min_s": min(order[1:]) if len(order) > 1 else None, "chunk_max_s": max(order[1:]) if len(order) > 1 else None,
           "denoise_step_last5": steps[-5:], "chunk_times": order,
           "out_mb": round(os.path.getsize(out) / 1e6, 2) if os.path.exists(out) else None}
    if rc != 0 or not order:
        res["tail"] = tail[-25:]
        # the child traceback sits well above torchrun's own wrapper: keep it all
        res["full_output"] = [l for l in full if not any(x in l for x in ("denoise per step", "Copyright", "License"))]
    print(f"[{label}] done rc={rc} chunks={len(order)} median={steady}", flush=True)
    return res


VOL = {WEIGHTS: weights, CACHE: cache}


@app.function(image=image_fa, gpu="H100:2", region=REGION, volumes=VOL, timeout=1800)
def fa_x2(compile_on: bool, image_bytes: bytes, audio_bytes: bytes) -> dict:
    return _bench(f"H100x2 fa2 compile={'on' if compile_on else 'off'}", 2, compile_on, "pro", image_bytes, audio_bytes)


@app.function(image=image_fa, gpu="H100", region=REGION, volumes=VOL, timeout=1800)
def fa_x1(compile_on: bool, image_bytes: bytes, audio_bytes: bytes) -> dict:
    return _bench(f"H100x1 fa2 compile={'on' if compile_on else 'off'}", 1, compile_on, "pro", image_bytes, audio_bytes)


@app.function(image=image_sage, gpu="H100:2", region=REGION, volumes=VOL, timeout=1800)
def sage_x2(compile_on: bool, image_bytes: bytes, audio_bytes: bytes) -> dict:
    return _bench(f"H100x2 sage compile={'on' if compile_on else 'off'}", 2, compile_on, "pro", image_bytes, audio_bytes)


@app.local_entrypoint()
def debug(image: str = "inputs/newscaster.png", audio: str = "inputs/korean_short.wav"):
    """One fast 2-GPU run (compile off) returning the full torchrun output."""
    img, wav = pathlib.Path(image).read_bytes(), pathlib.Path(audio).read_bytes()
    r = fa_x2.remote(False, img, wav)
    pathlib.Path("bench").mkdir(exist_ok=True)
    pathlib.Path("bench/debug_x2.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print("RESULT rc", r["rc"], "chunks", r["chunks"], "median", r["chunk_median_s"])
    for l in r.get("full_output", []):
        if any(k in l for k in ("Traceback", "Error", "error", "File \"/root", "File \"/tmp", "NCCL", "raise", "assert", "chunk-")):
            print("  ", l[:300])


@app.local_entrypoint()
def main(image: str = "inputs/newscaster.png", audio: str = "inputs/korean_flashhead.wav", out_dir: str = "bench"):
    img, wav = pathlib.Path(image).read_bytes(), pathlib.Path(audio).read_bytes()
    calls = {
        "A_x2_fa_off": fa_x2.spawn(False, img, wav),
        "B_x2_fa_on": fa_x2.spawn(True, img, wav),
        "D_x2_sage_on": sage_x2.spawn(True, img, wav),
    }
    results = {}
    deadline = time.time() + 1700
    while calls and time.time() < deadline:
        for name, c in list(calls.items()):
            try:
                r = c.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                r = {"label": name, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            results[name] = r
            calls.pop(name)
            summary = {k: r.get(k) for k in ("label", "rc", "chunks", "chunk_first_s", "chunk_median_s", "chunk_min_s", "chunk_max_s", "wall_s", "env", "error")}
            print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
        time.sleep(5)
    for name, c in calls.items():
        results[name] = {"label": name, "error": "not finished within deadline"}
        c.cancel()
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    (pathlib.Path(out_dir) / "h100_jp_pro.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("saved", out_dir + "/h100_jp_pro.json", flush=True)
