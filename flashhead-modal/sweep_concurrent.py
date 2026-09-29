"""Repeatable concurrency-limit finder for FeatherTalk real-time generation.

Launches bench_feathertalk_concurrent.py N times as SEPARATE OS processes
(matching how N independently-driven interview sessions would actually
compete for this box's GPU+CPU -- not N threads in one process, which would
understate real contention from N separate CUDA contexts and N separate
Python interpreters). Sweeps a list of N values in one run and reports, for
each N, the min/avg/max fps across sessions and whether the worst session
stayed at-or-above the real-time target (default 25fps, minus a small
tolerance for measurement jitter).

Usage (run ON the GPU box, from the FeatherTalk directory):
    python sweep_concurrent.py --dataset data/kjs --checkpoint ckpt_full/last.pth \
        --audio_feat data/kjs/aud_hu.npy --ns 1,4,8,10,12,16,20,24 --seconds 8

Re-run any time (e.g. after a code change meant to raise capacity) to get a
fresh, comparable number -- that repeatability, not any single run's output,
is the point of this script.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_FPS = 25.0
TOLERANCE = 0.5  # a session at 24.6fps isn't meaningfully "behind" 25fps


def run_one_n(n: int, dataset: str, checkpoint: str, audio_feat: str, seconds: float, tmp_dir: str, worker: str) -> dict:
    out_paths = [os.path.join(tmp_dir, f"sess_{n}_{i}.json") for i in range(n)]
    for p in out_paths:
        if os.path.exists(p):
            os.remove(p)

    procs = []
    t0 = time.time()
    for i, out_path in enumerate(out_paths):
        cmd = [
            sys.executable, os.path.join(HERE, worker),
            "--dataset", dataset, "--checkpoint", checkpoint, "--audio_feat", audio_feat,
            "--seconds", str(seconds), "--out", out_path,
        ]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))

    results = []
    for p, out_path in zip(procs, out_paths):
        _, stderr = p.communicate()
        if p.returncode != 0:
            print(f"  !! a session process failed (exit {p.returncode}): {stderr.decode()[-500:]}", flush=True)
            continue
        with open(out_path) as f:
            results.append(json.load(f))
    wall = time.time() - t0

    if not results:
        return {"n": n, "wall_s": wall, "error": "all sessions failed"}

    fpss = [r["fps"] for r in results]
    ok = min(fpss) >= TARGET_FPS - TOLERANCE
    return {
        "n": n, "wall_s": round(wall, 1), "completed": len(results),
        "min_fps": round(min(fpss), 1), "avg_fps": round(sum(fpss) / len(fpss), 1),
        "max_fps": round(max(fpss), 1), "real_time_ok": ok,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio_feat", required=True)
    p.add_argument("--ns", default="1,4,8,10,12,16,20,24", help="comma-separated session counts to test")
    p.add_argument("--seconds", type=float, default=8.0, help="measurement duration per N")
    p.add_argument("--tmp_dir", default="/tmp/sweep_concurrent")
    p.add_argument("--out", default=None, help="write the full JSON report here")
    p.add_argument("--worker", default="bench_feathertalk_concurrent.py",
                   help="which per-session worker script to launch (e.g. the in-memory-cached variant)")
    args = p.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    ns = [int(x) for x in args.ns.split(",")]

    report = []
    print(f"{'N':>4} {'min fps':>8} {'avg fps':>8} {'max fps':>8}  status", flush=True)
    knee_found = False
    for n in ns:
        r = run_one_n(n, args.dataset, args.checkpoint, args.audio_feat, args.seconds, args.tmp_dir, args.worker)
        report.append(r)
        if "error" in r:
            print(f"{n:>4}  ERROR: {r['error']}", flush=True)
            continue
        status = "OK" if r["real_time_ok"] else "BEHIND <-- knee"
        if not r["real_time_ok"] and not knee_found:
            knee_found = True
        print(f"{n:>4} {r['min_fps']:>8} {r['avg_fps']:>8} {r['max_fps']:>8}  {status}", flush=True)

    safe = [r["n"] for r in report if r.get("real_time_ok")]
    print(f"\nSafe (all sessions >= {TARGET_FPS}fps): N <= {max(safe) if safe else 0}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"target_fps": TARGET_FPS, "seconds_per_n": args.seconds, "results": report}, f, indent=2)
        print(f"Full report written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
