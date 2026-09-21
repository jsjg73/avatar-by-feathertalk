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
import json
import pathlib
import random
import sys
import threading
import time
import uuid
import wave

from renderer import (CKPT, GATES, LIVE_MAX_SESSIONS, WAV2VEC, LocalHost, RendererCore,
                      print_summary, verdict, verdict_line)


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
          stagger_s: float, model_type: str, out_dir: str | None,
          core: RendererCore | None = None, host: LocalHost | None = None,
          jitter: float = 0.5, seed: int = 0, lanes: int = 0) -> dict:
    """Drive `sessions` live sessions and report the same numbers `mux` does.

    `core`/`host` let a sweep load the pipeline once instead of paying the 85-125 s
    warm-up at every point. `jitter` is the fraction of `gap_s` by which each
    session's turn is randomly displaced: without it every session speaks on the
    same period with a constant offset, so the run measures one arbitrary phase
    alignment rather than how often turns actually collide.

    `lanes` (0 = off) instead assigns turns to a deterministic schedule: sessions
    are split into ceil(sessions/lanes) waves of `lanes` sessions each, and every
    session in a wave starts speaking at the same instant, `gap_s/waves` apart
    from the next wave — the L4 6-session x 7 s trick generalised to `lanes`
    concurrent speakers. It exists to answer one question empirically: can a
    server-controlled schedule, not random arrival, pack `lanes` x (gap_s/utterance)
    sessions onto one budget with near-zero wait? Pass jitter=0 alongside it —
    any per-turn jitter re-randomises the very thing being held fixed.
    """
    with wave.open(audio, "rb") as w:
        pcm = w.readframes(w.getnframes())
        speech_s = w.getnframes() / w.getframerate()
    img = pathlib.Path(image).read_bytes()
    ids = [f"local-{uuid.uuid4().hex[:8]}" for _ in range(sessions)]
    print(f"{sessions} sessions · 문장 {speech_s:.1f}s 마다 {gap_s:.0f}s · {minutes:.0f}분 "
          f"(예상 발화 비중 {speech_s / gap_s * 100:.0f}%)", flush=True)

    own = core is None
    if own:
        host = LocalHost()
        core = RendererCore(host=host, model_type=model_type)
        core.load()
    else:
        # container-wide counters; a sweep point must not inherit the previous one's
        core.slot_hist = [0] * len(core.slot_hist)
        core.slot_overruns = 0
    rng = random.Random(seed)

    budget_s = minutes * 60 + 120
    deadline = time.time() + minutes * 60
    finished = threading.Event()
    results: dict = {}
    out = pathlib.Path(out_dir) if out_dir else None

    def session(sid: str, delay: float = 0.0) -> None:
        # `stagger_s` now spaces the registrations, not the first utterances.
        # Registering a session crops the face and builds its reference latent, so
        # starting twelve at the same instant is a burst the production path never
        # sees — interviews are opened by people, minutes apart.
        if delay and finished.wait(delay):
            return
        try:
            results[sid] = core.live(sid, img, seed=42, idle_timeout_s=budget_s, max_seconds=budget_s)
        except Exception as exc:  # noqa: BLE001 — one session must not sink the run
            results[sid] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    def drive(sid: str, delay: float) -> None:
        if finished.wait(delay):
            return
        while time.time() < deadline and not finished.is_set():
            host.put({"type": "audio", "pcm": pcm, "final": True}, f"{sid}:audio")
            # uniform jitter around gap_s, mean preserved, so the duty cycle is the
            # same but the phase between sessions keeps drifting
            finished.wait(max(speech_s, gap_s + rng.uniform(-jitter, jitter) * gap_s))
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

    threads = [threading.Thread(target=session, args=(sid, i * stagger_s), name=f"live-{sid}")
               for i, sid in enumerate(ids)]
    # Random phase, not a ladder. Real interviews begin at random times of day, so
    # in steady state a session's turn sits anywhere in the interval — whereas
    # `i * stagger_s` starts every session's first utterance 4 s apart, which with
    # a 7.07 s utterance means the opening turns collide by construction and then
    # take several cycles of jitter to spread out. That transient was being
    # measured as if it were load.
    if lanes:
        waves = -(-sessions // lanes)                       # ceil
        speak_delay = [(i // lanes) * (gap_s / waves) for i in range(sessions)]
    else:
        speak_delay = [i * stagger_s + rng.uniform(0, gap_s) for i in range(sessions)]
    threads += [threading.Thread(target=drive, args=(sid, speak_delay[i]),
                                 daemon=True) for i, sid in enumerate(ids)]
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


def sweep(image: str, audio: str, counts: list[int], minutes: float, gap_s: float,
          stagger_s: float, model_type: str, jitter: float, out_json: str) -> dict:
    """Run the same experiment at several session counts and write one JSON.

    The pipeline is loaded once. Point 1 is the baseline: a single session has
    nobody to collide with, so its freeze-out is 0 by construction and its
    generation time is what this particular box can do. Every later point is read
    against it — without it a bad number cannot be told apart from a slow box.
    """
    import torch

    with wave.open(audio, "rb") as w:
        speech_s = w.getnframes() / w.getframerate()
    # Precision check before anything is spent. freeze-out is a proportion measured
    # over turns, so its standard error is sqrt(p(1-p)/T). The gate sits at 1%, and
    # a point with 32 turns has SE 1.8% — it cannot tell 0% from 3%, let alone
    # resolve the gate. Better to see that now than after paying for the GPU hour.
    print("\n실행 전 점검 — freeze-out 1% 게이트를 분해할 만큼 턴이 나오는가")
    print(f"  {'세션':>5}{'예상 턴':>9}{'1%에서의 표준오차':>18}  판정")
    coarse = []
    for n in counts:
        turns = int(n * minutes * 60 / gap_s)
        se = (0.01 * 0.99 / turns) ** 0.5 if turns else 1.0
        fine = se <= 0.005
        coarse.append(not fine)
        print(f"  {n:>5}{turns:>9}{se*100:>17.2f}%  {'분해 가능' if fine else '거칠다 — 무릎 찾기용'}")
    if all(coarse):
        need = 0.01 * 0.99 / (0.005 ** 2)
        print(f"  ※ 어느 지점도 1% 를 분해하지 못한다. 지점당 턴 {need:.0f}개가 필요하고,")
        print(f"     세션 {counts[-1]}·간격 {gap_s:.0f}s 면 {need*gap_s/counts[-1]/60:.0f}분이 든다.")
        print("     이 스윕은 '무릎이 어디인가'를 찾는 용도로 쓰고, 그 근처 두세 점만")
        print("     --minutes 를 늘려 다시 재는 2단계 설계를 권한다.\n", flush=True)
    else:
        print(flush=True)

    host = LocalHost()
    core = RendererCore(host=host, model_type=model_type)
    t_load = time.time()
    core.load()
    load_s = round(time.time() - t_load, 1)

    run = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpus": torch.cuda.device_count(),
        "model": model_type,
        "load_s": load_s,
        "utterance_s": round(speech_s, 2),
        "gap_s": gap_s,
        "jitter": jitter,
        "minutes": minutes,
        "duty": round(speech_s / gap_s, 4),
        "slot_s": None,
        "gates": GATES,
        "points": [],
    }
    for n in counts:
        if n > LIVE_MAX_SESSIONS:
            print(f"\n!! {n} 세션은 LIVE_MAX_SESSIONS={LIVE_MAX_SESSIONS} 를 넘는다 — "
                  f"FLASHHEAD_MAX_SESSIONS 를 올려서 다시 실행할 것", flush=True)
            break
        print(f"\n{'='*70}\n  {n} 세션\n{'='*70}", flush=True)
        res = bench(image, audio, n, minutes, gap_s, stagger_s, model_type, None,
                    core=core, host=host, jitter=jitter, seed=n)
        v = verdict(res)
        v["slot_hist"] = list(core.slot_hist)
        # the measured freeze-out's own uncertainty, so the report can say whether a
        # point actually cleared the gate or merely failed to see a violation
        v["freeze_out_se"] = round((max(v["freeze_out"], 1e-9) * (1 - v["freeze_out"])
                                    / v["turns"]) ** 0.5, 4) if v["turns"] else None
        run["points"].append(v)
        run["slot_s"] = run["slot_s"] or next(
            (r["seconds"] / r["chunks"] for r in res.values()
             if "error" not in r and r.get("chunks")), None)
        pathlib.Path(out_json).write_text(json.dumps(run, indent=2, ensure_ascii=False))
        print(f"  → {out_json} ({len(run['points'])}개 지점 기록)", flush=True)

    print(f"\n{'='*70}\n  스윕 요약\n{'='*70}")
    print(f"  {'세션':>5}{'예산':>5}{'턴':>6}{'freeze-out':>16}{'대기 p95':>10}{'최대':>8}{'영상지연':>10}  판정")
    for v in run["points"]:
        se = v.get("freeze_out_se")
        fo = f"{v['freeze_out']*100:.1f}±{se*100:.1f}%" if se else f"{v['freeze_out']*100:.1f}%"
        print(f"  {v['sessions']:>5}{v['slot_budget']:>5}{v['turns']:>6}{fo:>16}"
              f"{v['wait_p95_s']:>9.2f}s{v['wait_max_s']:>7.1f}s"
              f"{v['behind_tail_s']:>9.2f}s  {'PASS' if v['pass'] else 'FAIL ' + ','.join(v['failed_gates'])}")
    good = [v["sessions"] for v in run["points"] if v["pass"]]
    print(f"\n  지연 없이 가능한 최대 세션: {max(good) if good else 0}"
          f"  (게이트: freeze-out ≤ 1%, 대기 p95 ≤ 0.5s, 영상 지연 ≤ 0.5s)")
    return run


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
    p.add_argument("--sweep", default=None,
                   help="session counts to run in one go, e.g. 1,2,4,8,12,16 (1 = baseline)")
    p.add_argument("--jitter", type=float, default=0.5,
                   help="random displacement of each turn as a fraction of --gap-s (0 = fixed phase)")
    p.add_argument("--out-json", default="bench/sweep.json", help="where --sweep writes its result")
    p.add_argument("--lanes", type=int, default=0,
                   help="deterministic schedule instead of random arrival: N concurrent "
                        "speaker slots, filled in waves gap_s/waves apart. Pass --jitter 0 with it.")
    p.add_argument("--verdict", action="store_true", help="print pass/fail against GATES and exit non-zero on fail")
    a = p.parse_args()
    if a.fetch_weights:
        fetch_weights(a.include_pro)
        return
    if a.sweep:
        pathlib.Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        sweep(a.image, a.audio, [int(x) for x in a.sweep.split(",")], a.minutes, a.gap_s,
              a.stagger_s, a.model, a.jitter, a.out_json)
        return
    res = bench(a.image, a.audio, a.sessions, a.minutes, a.gap_s, a.stagger_s, a.model, a.out,
               jitter=a.jitter, lanes=a.lanes)
    if a.verdict:
        v = verdict(res)
        print(verdict_line(res))
        if not v.get("pass"):
            sys.exit(1)


if __name__ == "__main__":
    main()
