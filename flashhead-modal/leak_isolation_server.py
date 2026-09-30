"""Isolates the /speak HTTP + queue + consumer-thread layer from live_server.py,
with the two compute paths already cleared of suspicion (audio feature
extraction, video frame generation -- both tested flat in isolation on
2026-09-29) replaced by cheap fakes of the same shape/size. If a leak still
reproduces here, it's in this glue layer -- FastAPI/UploadFile handling, or
the queue<->consumer-thread handoff itself -- not in the model code.

Not a real live_server.py alternative -- diagnostic only, throwaway.

Usage:
    python leak_isolation_server.py            # starts the server on :8999
    python leak_isolation_pump.py               # hits it repeatedly, reports RSS
"""
import threading
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, UploadFile
import uvicorn

FPS = 25
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FAKE_N_FRAMES = 187  # matches test_line.wav's real queued_frames from earlier sessions

app = FastAPI()
STATE = {"queue": deque(), "lock": threading.Lock(), "interrupt": False, "speaking": False}


def _fake_extract_features() -> np.ndarray:
    # Same shape class as the real _extract_features return (~374 x 2 x 1024
    # float32 for a ~7.5s clip) -- a fresh allocation every call, like the real one.
    return np.zeros((374, 2, 1024), dtype=np.float32)


@app.post("/speak")
async def speak(file: UploadFile):
    raw = await file.read()  # real UploadFile read, same as live_server.py
    features = _fake_extract_features()
    pcm = bytes(SAMPLES_PER_FRAME * 2 * FAKE_N_FRAMES)  # same size class as real pcm
    with STATE["lock"]:
        STATE["queue"].append({"features": features, "pcm": pcm, "idx": 0, "n_frames": FAKE_N_FRAMES})
    return {"queued_frames": FAKE_N_FRAMES, "received_bytes": len(raw)}


@app.get("/status")
def status():
    return {"pending": len(STATE["queue"])}


def _consumer_loop() -> None:
    """Mimics _gen_loop's dequeue/advance/discard cadence, at 25fps."""
    current = None
    while True:
        t0 = time.time()
        with STATE["lock"]:
            if current is None and STATE["queue"]:
                current = STATE["queue"].popleft()
        if current is not None:
            current["idx"] += 1
            if current["idx"] >= current["n_frames"]:
                current = None
        elapsed = time.time() - t0
        sleep_s = (1.0 / FPS) - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    threading.Thread(target=_consumer_loop, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8999)
