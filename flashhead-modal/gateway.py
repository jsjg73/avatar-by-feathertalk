"""Multi-session gateway in front of N live_server.py worker processes.

Each session gets exclusive use of one live_server.py process (its own CUDA
context, resident model, and read-only view of the shared memmap image
cache) for its lifetime -- this is the N-separate-process concurrency
architecture verified up to N=24 (sweep_concurrent.py), not a batched
producer-consumer server. See docs/live-serving/DESIGN.md, section 0, for
why that's a deliberate choice: batching was tried once in this project and
regressed performance (bench_feathertalk_batched.py).

The gateway proxies both the HTTP control calls (/speak, /interrupt) and the
media WebSocket (/ws) so that only ONE public port needs to be reachable --
a rented GPU box typically has one (or a few) port mapped, not a whole range
per worker.

Usage:
    python gateway.py --dataset data/kjs --checkpoint ckpt_full/last.pth \
        --workers 24 --base_port 9000 --port 8080

API:
    POST /sessions                    -> {"session_id": ...} | 503 no_capacity
    POST /sessions/{id}/speak         multipart file, proxied to the worker's /speak
    POST /sessions/{id}/interrupt     barge-in: clears the worker's queue + current clip
    GET  /sessions/{id}               proxied worker /status, plus session "status"
    GET  /sessions/{id}/ws            WebSocket, relays the worker's fMP4 fragment stream
    POST /sessions/{id}/end           frees the worker back to the pool
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
import uuid

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

POOL: list[dict] = []          # [{"port": int, "process": Popen, "session_id": str | None}]
SESSIONS: dict[str, dict] = {}  # session_id -> the POOL entry it owns
READY_TIMEOUT_S = 120


async def _wait_ready(port: int) -> None:
    """Blocks until a freshly-spawned worker answers /status (model loaded,
    ffmpeg started) or raises after READY_TIMEOUT_S -- a worker that never
    comes up (bad checkpoint path, OOM, ...) should fail loudly at startup,
    not silently sit in the pool as a dead slot that 503s every session."""
    deadline = time.time() + READY_TIMEOUT_S
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"http://127.0.0.1:{port}/status", timeout=2.0)
                if r.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"worker on port {port} did not become ready within {READY_TIMEOUT_S}s")


async def _spawn_pool(n: int, base_port: int, dataset: str, checkpoint: str, fh_checkpoint: str) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    procs = []
    for i in range(n):
        port = base_port + i
        cmd = [sys.executable, f"{here}/live_server.py",
               "--dataset", dataset, "--checkpoint", checkpoint,
               "--fh_checkpoint", fh_checkpoint, "--port", str(port)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((port, proc))
        print(f"[gateway] worker {i+1}/{n} launched on port {port} (pid {proc.pid})", flush=True)

    # Wait for all workers concurrently -- serially would be N * (model load
    # time), which at ~24 workers turns a ~10s load into a 4-minute startup.
    await asyncio.gather(*(_wait_ready(port) for port, _ in procs))
    for port, proc in procs:
        POOL.append({"port": port, "process": proc, "session_id": None})
    print(f"[gateway] all {n} workers ready", flush=True)


def _find_free() -> dict | None:
    for slot in POOL:
        if slot["session_id"] is None:
            return slot
    return None


def _get_session(session_id: str) -> dict:
    slot = SESSIONS.get(session_id)
    if slot is None:
        raise HTTPException(404, "unknown or ended session")
    return slot


@app.post("/sessions")
async def create_session(avatar: str = "kjs"):
    slot = _find_free()
    if slot is None:
        return JSONResponse({"error": "no_capacity"}, status_code=503)
    session_id = uuid.uuid4().hex
    slot["session_id"] = session_id
    SESSIONS[session_id] = slot
    print(f"[gateway] session {session_id} -> port {slot['port']} ({len(SESSIONS)}/{len(POOL)} busy)", flush=True)
    return {"session_id": session_id}


@app.post("/sessions/{session_id}/speak")
async def speak(session_id: str, file: UploadFile):
    slot = _get_session(session_id)
    data = await file.read()
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"http://127.0.0.1:{slot['port']}/speak",
            files={"file": (file.filename or "speak.wav", data, "audio/wav")},
            timeout=30.0,
        )
    r.raise_for_status()
    return r.json()


@app.post("/sessions/{session_id}/interrupt")
async def interrupt(session_id: str):
    slot = _get_session(session_id)
    async with httpx.AsyncClient() as client:
        r = await client.post(f"http://127.0.0.1:{slot['port']}/interrupt", timeout=5.0)
    r.raise_for_status()
    return r.json()


@app.get("/sessions/{session_id}")
async def session_status(session_id: str):
    slot = SESSIONS.get(session_id)
    if slot is None:
        return {"status": "ended"}
    async with httpx.AsyncClient() as client:
        r = await client.get(f"http://127.0.0.1:{slot['port']}/status", timeout=5.0)
    r.raise_for_status()
    return {"status": "running", **r.json()}


@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str):
    slot = _get_session(session_id)
    # Clear whatever's queued/playing before returning the worker to the pool
    # -- otherwise the next session assigned to this same process would
    # start by finishing the previous candidate's leftover speech.
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"http://127.0.0.1:{slot['port']}/interrupt", timeout=5.0)
        except httpx.RequestError:
            pass
    del SESSIONS[session_id]
    slot["session_id"] = None
    print(f"[gateway] session {session_id} ended, port {slot['port']} freed", flush=True)
    return {"ok": True}


@app.websocket("/sessions/{session_id}/ws")
async def media_ws(websocket: WebSocket, session_id: str):
    """One-directional relay: worker's /ws (fMP4 fragments) -> this client.
    The client never sends anything here (audio goes through POST /speak
    instead), so the only thing to watch for is either side hanging up --
    exact exception types vary by websockets/Starlette version, so this
    treats any break in the pipe as a normal end-of-session, not an error
    worth surfacing."""
    slot = SESSIONS.get(session_id)
    if slot is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    upstream_url = f"ws://127.0.0.1:{slot['port']}/ws"
    try:
        async with websockets.connect(upstream_url, max_size=None) as upstream:
            while True:
                frag = await upstream.recv()
                await websocket.send_bytes(frag)
    except Exception as exc:
        print(f"[gateway] media relay for {session_id} ended: {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # already closed by the client or the upstream disconnecting


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fh_checkpoint", default="./feather_hubert.pth")
    p.add_argument("--workers", type=int, default=24, help="pool size -- verified safe up to 24 (sweep_concurrent.py)")
    p.add_argument("--base_port", type=int, default=9000, help="workers occupy [base_port, base_port + workers)")
    p.add_argument("--port", type=int, default=8080, help="gateway's own public port")
    args = p.parse_args()

    @app.on_event("startup")
    async def _startup() -> None:
        await _spawn_pool(args.workers, args.base_port, args.dataset, args.checkpoint, args.fh_checkpoint)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
