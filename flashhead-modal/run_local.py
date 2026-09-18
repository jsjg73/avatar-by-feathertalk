"""Run the renderer as an ordinary process, on any box with an NVIDIA GPU.

This is the same experiment `app.py::mux` runs on Modal, with the cloud taken
out: one `RendererCore`, `LocalHost` for transport, and threads instead of
concurrent Modal inputs. Nothing here is specific to Lightning, RunPod, vast.ai
or Lambda — they are all "a Linux box with a GPU", which is what this assumes.

    python run_local.py --sessions 6 --minutes 3 --gap-s 18

Before the first run, fetch the weights (once per machine; ~8 GB):

    python run_local.py --fetch-weights

Paths follow the environment, so nothing has to live where Modal put it:

    FLASHHEAD_SRC=~/SoulX-FlashHead FLASHHEAD_WEIGHTS=~/weights \\
    FLASHHEAD_CACHE=~/cache FLASHHEAD_WARMUP=~/newscaster.png python run_local.py
"""

import argparse
import pathlib
import threading
import time
import uuid
import wave

from renderer import CKPT, WAV2VEC, LocalHost, RendererCore, print_summary


def fetch_weights(include_pro: bool = False) -> None:
    """Download the checkpoints this machine needs. Same source as on Modal."""
    from huggingface_hub import snapshot_download

    ignore = None if include_pro else ["*14B*", "*pro*"]
    print(f"→ {CKPT}", flush=True)
    snapshot_download("Soul-AILab/SoulX-FlashHead-1_3B", local_dir=CKPT, ignore_patterns=ignore)
    print(f"→ {WAV2VEC}", flush=True)
    snapshot_download("facebook/wav2vec2-base-960h", local_dir=WAV2VEC)
    print("weights ready", flush=True)


def bench(image: str, audio: str, sessions: int, minutes: float, gap_s: float,
          stagger_s: float, model_type: str, out_dir: str | None) -> dict:
    """Drive `sessions` live sessions and report the same numbers `mux` does."""
    with wave.open(audio, "rb") as w:
        pcm = w.readframes(w.getnframes())
        speech_s = w.getnframes() / w.getframerate()
    img = pathlib.Path(image).read_bytes()
    ids = [f"local-{uuid.uuid4().hex[:8]}" for _ in range(sessions)]
    print(f"{sessions} sessions · 문장 {speech_s:.1f}s 마다 {gap_s:.0f}s · {minutes:.0f}분 "
          f"(예상 발화 비중 {speech_s / gap_s * 100:.0f}%)", flush=True)

    host = LocalHost()
    core = RendererCore(host=host, model_type=model_type)
    core.load()

    budget_s = minutes * 60 + 120
    deadline = time.time() + minutes * 60
    finished = threading.Event()
    results: dict = {}
    out = pathlib.Path(out_dir) if out_dir else None

    def session(sid: str) -> None:
        try:
            results[sid] = core.live(sid, img, seed=42, idle_timeout_s=budget_s, max_seconds=budget_s)
        except Exception as exc:  # noqa: BLE001 — one session must not sink the run
            results[sid] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    def drive(sid: str, delay: float) -> None:
        if finished.wait(delay):
            return
        while time.time() < deadline and not finished.is_set():
            host.put({"type": "audio", "pcm": pcm, "final": True}, f"{sid}:audio")
            finished.wait(gap_s)
        host.put({"type": "end"}, f"{sid}:audio")

    def drain(sid: str) -> None:
        """Keep the session's outbound queue empty, and optionally keep the video.

        Whatever drives the renderer has to consume what it produces — the studio
        does this to serve HLS. Here it is either written to `out_dir` or dropped,
        but it can never just be left: HLS segments arrive every 0.96 s per
        session and an unread queue is a memory leak with a slow fuse.
        """
        d = out / sid if out else None
        if d:
            d.mkdir(parents=True, exist_ok=True)
        parts: dict = {}
        while not finished.is_set():
            for item in host.get_many(50, sid, block=True, timeout=0.5):
                for it in (item["items"] if item.get("type") == "batch" else [item]):
                    if it.get("type") == "log":
                        print(f"  [{sid[-4:]}] {it['line']}", flush=True)
                    elif it.get("type") == "file" and d:
                        buf = parts.setdefault(it["name"], {})
                        buf[it["part"]] = it["data"]
                        if len(buf) == it["parts"]:
                            (d / it["name"]).write_bytes(b"".join(buf[i] for i in sorted(buf)))
                            parts.pop(it["name"])

    threads = [threading.Thread(target=session, args=(sid,), name=f"live-{sid}") for sid in ids]
    threads += [threading.Thread(target=drive, args=(sid, i * stagger_s), daemon=True)
                for i, sid in enumerate(ids)]
    threads += [threading.Thread(target=drain, args=(sid,), daemon=True) for sid in ids]
    for t in threads:
        t.start()
    for t in threads[:sessions]:                 # the session threads own the run's length
        t.join(timeout=budget_s + 120)
    finished.set()
    for t in threads[sessions:]:
        t.join(timeout=3)

    print_summary(results)
    if out:
        print(f"\nHLS: {out}/<session>/index.m3u8", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fetch-weights", action="store_true", help="download checkpoints and exit")
    p.add_argument("--include-pro", action="store_true", help="with --fetch-weights, also get 14B")
    p.add_argument("--image", default="inputs/newscaster.png")
    p.add_argument("--audio", default="inputs/korean_short.wav")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--minutes", type=float, default=3.0)
    p.add_argument("--gap-s", type=float, default=18.0)
    p.add_argument("--stagger-s", type=float, default=4.0)
    p.add_argument("--model", default="lite", choices=("lite", "pro"))
    p.add_argument("--out", default=None, help="write HLS here (default: drop it)")
    a = p.parse_args()
    if a.fetch_weights:
        fetch_weights(a.include_pro)
        return
    bench(a.image, a.audio, a.sessions, a.minutes, a.gap_s, a.stagger_s, a.model, a.out)


if __name__ == "__main__":
    main()
