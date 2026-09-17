"""Is there a cheaper GPU than L4 for FlashHead Lite?

Measures raw bf16/fp16 throughput at the shapes the Lite DiT actually uses,
without downloading weights, so each probe is a ~90 s container.
"""
import json
import modal

app = modal.App("flashhead-probe-cheap")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
)


def bench():
    import time
    import subprocess
    import torch
    import torch.nn.functional as F

    out = {}
    out["smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    out["capability"] = list(torch.cuda.get_device_capability())
    try:
        out["is_bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as e:                      # noqa: BLE001
        out["is_bf16_supported"] = f"error: {e}"

    def timed(fn, warmup=5, iters=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    # 1.3B DiT: hidden 1536, 12 heads x 128. Chunk of 33 frames at 16x16 latent
    # patchifies to roughly 1.3k tokens; 4096 covers the wider attention too.
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        d = {}
        try:
            a = torch.randn(4096, 1536, device="cuda", dtype=dtype)
            w = torch.randn(1536, 1536, device="cuda", dtype=dtype)
            d["gemm_4096x1536x1536_ms"] = round(timed(lambda: a @ w) * 1e3, 3)

            q = torch.randn(1, 12, 1280, 128, device="cuda", dtype=dtype)
            k = torch.randn(1, 12, 1280, 128, device="cuda", dtype=dtype)
            v = torch.randn(1, 12, 1280, 128, device="cuda", dtype=dtype)
            d["sdpa_1280tok_ms"] = round(timed(lambda: F.scaled_dot_product_attention(q, k, v)) * 1e3, 3)

            big = torch.randn(8192, 8192, device="cuda", dtype=dtype)
            d["gemm_8192_tflops"] = round(
                2 * 8192 ** 3 / timed(lambda: big @ big) / 1e12, 1)
        except Exception as e:                  # noqa: BLE001
            d["error"] = f"{type(e).__name__}: {e}"
        out[dtype_name] = d

    # conv3d shows up in the VAE decode, which is the bandwidth-heavy half
    try:
        import torch.nn as nn
        conv = nn.Conv3d(128, 128, 3, padding=1).cuda().to(torch.bfloat16)
        x = torch.randn(1, 128, 5, 64, 64, device="cuda", dtype=torch.bfloat16)
        out["vae_conv3d_bf16_ms"] = round(timed(lambda: conv(x)) * 1e3, 3)
    except Exception as e:                      # noqa: BLE001
        out["vae_conv3d_bf16_ms"] = f"error: {type(e).__name__}: {e}"

    print(json.dumps(out, indent=2))
    return out


@app.function(image=image, gpu="T4", timeout=600)
def t4():
    return bench()


@app.function(image=image, gpu="L4", timeout=600)
def l4():
    return bench()


@app.function(image=image, gpu="A10", timeout=600)
def a10():
    return bench()


@app.function(image=image, gpu="L40S", timeout=600)
def l40s():
    return bench()


@app.function(image=image, gpu="H100", timeout=600)
def h100():
    return bench()


@app.local_entrypoint()
def big():
    for name, fn in (("L40S", l40s), ("H100", h100)):
        try:
            fn.remote()
        except Exception as e:                  # noqa: BLE001
            print(f"{name}: {type(e).__name__}: {e}")
        print(f"--- {name} done ---", flush=True)


@app.local_entrypoint()
def main():
    import json as _j
    res = {}
    for name, fn in (("T4", t4), ("L4", l4), ("A10", a10)):
        try:
            res[name] = fn.remote()
        except Exception as e:                  # noqa: BLE001
            res[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"--- {name} done ---", flush=True)
    print(_j.dumps(res, indent=2))
