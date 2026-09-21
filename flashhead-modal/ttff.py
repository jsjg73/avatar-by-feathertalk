#!/usr/bin/env python3
"""무경합 상황의 "오디오 도착 → 첫 프레임" 지연을 정밀하게 잰다.

`wait_to_speak_s` 는 이걸 못 잰다 — 슬롯 시작에서 한 번 찍은 `now` 를 admission
과 emit 양쪽에서 그대로 쓰기 때문에(renderer.py:1137,1249,1306), 한 슬롯 안에서
바로 생성·송출된 세션은 실제 GPU 시간과 무관하게 대기가 정확히 0.00 으로
찍힌다. "대기 없음"과 "즉시 도착"은 다른 말이다.

여기서는 renderer 내부를 우회해 바깥에서 직접 시계를 잰다:

    t_send   = host.put() 를 호출하기 직전
    t_frame  = LocalHost 의 영상 큐에 그 세션의 첫 비-유휴 프레임이 들어온 시각

세션을 하나만 띄우므로(경합 없음) 이 값이 곧 바닥 지연이다. 예산이 남아도는
한 세션 초과분(2번째, 3번째...)까지 반복해 표본을 늘린다 — 매번 유일한
발화자이므로 여전히 무경합이다.

    python ttff.py --reps 10
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from renderer import RendererCore, LocalHost  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--audio", default="inputs/q_avatar_7s.wav")
    ap.add_argument("--image", default="inputs/newscaster.png")
    a = ap.parse_args()

    import wave

    with wave.open(a.audio, "rb") as w:
        pcm = w.readframes(w.getnframes())
    img = open(a.image, "rb").read()

    host = LocalHost()
    core = RendererCore(host=host, model_type="lite")
    t0 = time.time()
    core.load()
    print(f"로드 {time.time()-t0:.1f}s · gen_ema {core.gen_ema:.3f}s", flush=True)

    import threading

    sid = "ttff-probe"
    results: dict = {}

    def session() -> None:
        results["r"] = core.live(sid, img, seed=42, idle_timeout_s=90, max_seconds=90)

    th = threading.Thread(target=session, daemon=True)
    th.start()

    # registration (face crop, idle-loop build) takes well under a second on
    # this card (measured ~0.3 s); give it comfortable margin before sending audio
    deadline = time.time() + 30
    while sid not in core.live_sessions and time.time() < deadline:
        time.sleep(0.02)
    if sid not in core.live_sessions:
        sys.exit("세션이 30초 안에 등록되지 않았습니다")
    time.sleep(1.0)

    gaps = []
    for i in range(a.reps):
        prev = core.live_sessions[sid]["stats"].get("speech_chunks", 0)
        t_send = time.time()
        host.put({"type": "audio", "pcm": pcm, "final": True}, f"{sid}:audio")
        # Poll the session's own stats for the first NEW speech chunk since this
        # rep's send — `_emit`'s `st["speech_chunks"] += 1` happens the instant a
        # frame is queued for the viewer (renderer.py:_emit), which is the
        # earliest externally observable "a frame now exists" without hooking
        # the scheduler's internal (and, per wait_to_speak_s, stale-timestamped)
        # clock at all.
        t_frame = None
        deadline = time.time() + 15
        while time.time() < deadline:
            if core.live_sessions[sid]["stats"].get("speech_chunks", 0) > prev:
                t_frame = time.time()
                break
            time.sleep(0.005)
        if t_frame is None:
            print(f"  rep {i}: 15초 안에 발화 청크가 안 잡힘 (건너뜀)")
            continue
        gap = t_frame - t_send
        gaps.append(gap)
        print(f"  rep {i}: {gap*1000:.0f} ms", flush=True)
        time.sleep(8.0)  # let this turn's speech drain before sending the next

    if gaps:
        gaps.sort()
        n = len(gaps)
        print(f"\n{n}회 · 중앙값 {gaps[n//2]*1000:.0f} ms · "
              f"최소 {gaps[0]*1000:.0f} ms · 최대 {gaps[-1]*1000:.0f} ms")
    core.live_sessions[sid]["ended"] = True
    th.join(timeout=15)


if __name__ == "__main__":
    main()
