"""Scheduling / topology probe for multi-GPU candidates. Each probe runs a few
seconds: nvidia-smi, NVLink topology, torch device count, capability, a matmul
on every GPU, and peer access between GPU 0 and 1. Uses the FlashHead image so
the CUDA 12.4 / torch cu124 stack is what gets tested (Blackwell needs cu128)."""
import subprocess
import time

import modal

from app import image as _base_image  # (add_local_python_source only, so the local-file layer is fine)  # noqa: E402  (same image as the renderer; import does not deploy)

# The container imports this module too, and it imports `app` — so app.py must
# be shipped with the image (Modal 1.x does not auto-mount local modules).
image = _base_image.add_local_python_source("app")

app = modal.App("flashhead-gpu-probe")


def _probe(label: str) -> dict:
    import torch

    import os

    out: dict = {"label": label, "region": os.environ.get("MODAL_REGION"), "cloud": os.environ.get("MODAL_CLOUD_PROVIDER")}
    import socket

    for host in ("www.naver.com", "www.google.com"):
        best = None
        for _ in range(3):
            try:
                t = time.time(); socket.create_connection((host, 443), timeout=5).close()
                ms = (time.time() - t) * 1000
                best = ms if best is None else min(best, ms)
            except Exception:  # noqa: BLE001
                pass
        out[f"tcp_connect_ms_{host.split('.')[1]}"] = round(best) if best is not None else None
    t0 = time.time()
    out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60).stdout
    out["topo_first_rows"] = [l for l in topo.splitlines()[:4]]
    out["nvlink"] = "NV" in topo
    out["torch"] = torch.__version__
    out["device_count"] = torch.cuda.device_count()
    out["capability"] = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    try:
        for i in range(torch.cuda.device_count()):
            a = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
            torch.cuda.synchronize(i); t = time.time()
            for _ in range(20):
                a = a @ a * 1e-3
            torch.cuda.synchronize(i)
            out.setdefault("bf16_tflops", []).append(round(20 * 2 * 4096**3 / (time.time() - t) / 1e12, 1))
        if torch.cuda.device_count() > 1:
            out["p2p_0_1"] = torch.cuda.can_device_access_peer(0, 1)
            x = torch.randn(256, 1024, 1024, device="cuda:0"); torch.cuda.synchronize()
            t = time.time(); y = x.to("cuda:1"); torch.cuda.synchronize()
            out["copy_0_to_1_GBps"] = round(x.numel() * 4 / (time.time() - t) / 1e9, 1)
    except Exception as exc:  # noqa: BLE001
        out["kernel_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


@app.function(image=image, gpu="H100:2", timeout=900)
def h100x2() -> dict: return _probe("H100:2")

@app.function(image=image, gpu="H200:2", timeout=900)
def h200x2() -> dict: return _probe("H200:2")

@app.function(image=image, gpu="A100-80GB:2", timeout=900)
def a100x2() -> dict: return _probe("A100-80GB:2")

@app.function(image=image, gpu="B200", timeout=900)
def b200() -> dict: return _probe("B200")

@app.function(image=image, gpu="H100:4", timeout=900)
def h100x4() -> dict: return _probe("H100:4")


# Region-pinned variants: Korea-facing latency needs containers in Japan /
# ap-northeast, so check that big-GPU pairs are actually schedulable there.
@app.function(image=image, gpu="H200:2", region=["jp"], timeout=900)
def h200x2_jp() -> dict: return _probe("H200:2 @jp")

@app.function(image=image, gpu="H200:2", region=["ap-northeast"], timeout=900)
def h200x2_apne() -> dict: return _probe("H200:2 @ap-northeast")

@app.function(image=image, gpu="H100:2", region=["jp", "ap-northeast"], timeout=900)
def h100x2_jp() -> dict: return _probe("H100:2 @jp|ap-northeast")

@app.function(image=image, gpu="H200:4", timeout=900)
def h200x4() -> dict: return _probe("H200:4")


@app.function(image=image, gpu="L4", region=["jp"], timeout=600)
def l4_jp() -> dict: return _probe("L4 @jp")

@app.function(image=image, gpu="L4", region=["ap-northeast"], timeout=600)
def l4_apne() -> dict: return _probe("L4 @ap-northeast")

@app.function(image=image, gpu="L4", region=["ap"], timeout=600)
def l4_ap() -> dict: return _probe("L4 @ap (any Asia-Pacific)")


@app.function(image=image, gpu="L4", region=["ap-southeast"], timeout=600)
def l4_apse() -> dict: return _probe("L4 @ap-southeast")

@app.function(image=image, gpu="L4", region=["ap-south"], timeout=600)
def l4_aps() -> dict: return _probe("L4 @ap-south")

# note: gpu='L4' + region=['ap-melbourne'] is rejected at app creation (no L4 worker type there)

@app.function(image=image, gpu="L4", timeout=600)
def l4_any() -> dict: return _probe("L4 @default")

@app.function(image=image, gpu="H100", region=["jp"], timeout=600)
def h100_jp() -> dict: return _probe("H100 @jp")


@app.local_entrypoint()
def l4kr():
    """L4 placement for Korea: how long do ap-northeast / jp / ap pins take, where do they land, RTT to Korea?"""
    import json
    t0 = time.time()
    calls = {"ap-northeast#0": l4_apne.spawn(), "ap-northeast#1": l4_apne.spawn(), "ap-northeast#2": l4_apne.spawn(),
             "jp": l4_jp.spawn(), "ap#0": l4_ap.spawn(), "ap#1": l4_ap.spawn(), "H100 jp": h100_jp.spawn()}
    deadline = time.time() + 300
    while calls and time.time() < deadline:
        for name, c in list(calls.items()):
            try:
                r = c.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(json.dumps({"probe": name, "scheduled_after_s": round(time.time() - t0 - r.get("seconds", 0), 1), "region": r.get("region"),
                              "naver_ms": r.get("tcp_connect_ms_naver"), "google_ms": r.get("tcp_connect_ms_google"), "error": r.get("error")}), flush=True)
            calls.pop(name)
        time.sleep(2)
    for name, c in calls.items():
        print(json.dumps({"probe": name, "error": "not scheduled within 300s"}), flush=True)
        c.cancel()


@app.local_entrypoint()
def l4ap():
    """Where in Asia-Pacific do L4s actually land, and how far are they from Korea?"""
    import json
    t0 = time.time()
    calls = {"ap#0": l4_ap.spawn(), "ap#1": l4_ap.spawn(), "ap#2": l4_ap.spawn(), "ap-southeast": l4_apse.spawn(),
             "ap-south": l4_aps.spawn(), "default": l4_any.spawn(), "H100 jp": h100_jp.spawn()}
    deadline = time.time() + 180
    while calls and time.time() < deadline:
        for name, c in list(calls.items()):
            try:
                r = c.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(json.dumps({"probe": name, "t_s": round(time.time() - t0, 1), "region": r.get("region"), "cloud": r.get("cloud"),
                              "naver_ms": r.get("tcp_connect_ms_naver"), "google_ms": r.get("tcp_connect_ms_google"), "error": r.get("error")}), flush=True)
            calls.pop(name)
        time.sleep(2)
    for name, c in calls.items():
        print(json.dumps({"probe": name, "error": "not scheduled within 180s"}), flush=True)
        c.cancel()


@app.local_entrypoint()
def l4jp(n: int = 6):
    """Can L4 containers be scheduled in Japan, and several at once? Spawns `n`
    jp probes in parallel plus one ap-northeast and one ap-wide probe."""
    import json
    t0 = time.time()
    calls = {f"jp#{i}": l4_jp.spawn() for i in range(n)}
    calls["ap-northeast"] = l4_apne.spawn()
    calls["ap"] = l4_ap.spawn()
    deadline = time.time() + 240
    while calls and time.time() < deadline:
        for name, c in list(calls.items()):
            try:
                r = c.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(json.dumps({"probe": name, "t_done_s": round(time.time() - t0, 1), "gpu": (r.get("nvidia_smi") or ["?"])[0][:40], "error": r.get("error")}), flush=True)
            calls.pop(name)
        time.sleep(2)
    for name, c in calls.items():
        print(json.dumps({"probe": name, "error": "not scheduled within 240s"}), flush=True)
        c.cancel()


@app.local_entrypoint()
def regions():
    import json
    calls = {"h200x2_jp": h200x2_jp.spawn(), "h200x2_apne": h200x2_apne.spawn(), "h100x2_jp": h100x2_jp.spawn(), "h200x4": h200x4.spawn()}
    deadline = time.time() + 240
    while calls and time.time() < deadline:
        for name, c in list(calls.items()):
            try:
                r = c.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                r = {"label": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(json.dumps({k: v for k, v in r.items() if k in ("label", "nvidia_smi", "device_count", "p2p_0_1", "copy_0_to_1_GBps", "error", "kernel_error")}, ensure_ascii=False), flush=True)
            calls.pop(name)
        time.sleep(3)
    for name, c in calls.items():
        print(json.dumps({"label": name, "error": "not scheduled within 240s"}), flush=True)
        c.cancel()


@app.local_entrypoint()
def main():
    import json
    calls = {f.__name__ if hasattr(f, "__name__") else n: f.spawn() for n, f in
             {"h100x2": h100x2, "h200x2": h200x2, "a100x2": a100x2, "b200": b200, "h100x4": h100x4}.items()}
    deadline = time.time() + 300
    pending = dict(calls)
    while pending and time.time() < deadline:
        for name, c in list(pending.items()):
            try:
                r = c.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                r = {"label": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            pending.pop(name)
        time.sleep(3)
    for name in pending:
        print(json.dumps({"label": name, "error": f"not scheduled within 300s"}), flush=True)
        pending[name].cancel()
