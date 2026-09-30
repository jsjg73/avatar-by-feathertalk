"""One-shot smoke test for the gateway + idle-cache feature: creates a
session, captures the /ws fMP4 stream during an idle period (no /speak
call yet) and again during/after a /speak call, writing each capture to
its own file for later ffmpeg frame-extraction and visual inspection.

Throwaway diagnostic -- not part of the production client.

Usage: python smoke_test_idle.py
"""
import asyncio
import json
import time

import httpx
import websockets

BASE = "http://127.0.0.1:8000"
WS_BASE = "ws://127.0.0.1:8000"


async def capture_ws(session_id: str, out_path: str, duration_s: float) -> int:
    url = f"{WS_BASE}/sessions/{session_id}/ws"
    total = 0
    async with websockets.connect(url, max_size=None) as ws:
        with open(out_path, "wb") as f:
            deadline = time.time() + duration_s
            while time.time() < deadline:
                try:
                    frag = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
                except asyncio.TimeoutError:
                    break
                f.write(frag)
                total += len(frag)
    return total


async def main() -> None:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/sessions")
        r.raise_for_status()
        session_id = r.json()["session_id"]
        print("session:", session_id)

    print("capturing idle period (6s)...")
    n = await capture_ws(session_id, "idle_capture.mp4", 6.0)
    print(f"  idle_capture.mp4: {n} bytes")

    async with httpx.AsyncClient() as client:
        with open("test_line.wav", "rb") as f:
            r = await client.post(
                f"{BASE}/sessions/{session_id}/speak",
                files={"file": ("test_line.wav", f.read(), "audio/wav")},
                timeout=30.0,
            )
        r.raise_for_status()
        print("speak response:", r.json())

    print("capturing speak period (10s)...")
    n = await capture_ws(session_id, "speak_capture.mp4", 10.0)
    print(f"  speak_capture.mp4: {n} bytes")

    async with httpx.AsyncClient() as client:
        await client.post(f"{BASE}/sessions/{session_id}/end")
    print("session ended")


if __name__ == "__main__":
    asyncio.run(main())
