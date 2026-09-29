"""FeatherTalk live HLS streaming server -- persistent-ffmpeg version.

One long-lived ffmpeg process (matching FlashHead's studio/server.py design)
receives raw video over one named pipe and raw PCM audio over another, and
muxes+segments them into a live HLS playlist continuously. This replaces the
earlier one-shot-per-segment approach, which had to fake timestamp continuity
across independently-encoded segments; here ffmpeg keeps one running PTS
clock, so that whole class of bug doesn't exist.

The earlier attempt at this deadlocked: opening a FIFO for write blocks until
a reader connects, and if both fifos are opened from the *same* thread in
sequence, ffmpeg's own need to read a little from the first input before it
opens the second can leave both sides waiting on each other. Fix: two
independent writer threads, one per fifo, each blocking only on its own
open() -- neither can block the other's handshake. A small model-inference
loop feeds both writers through queues and is decoupled from both.

Usage:
    ./.venv/bin/python live_server.py --dataset data/kjs --checkpoint ckpt_full/last.pth
"""

import argparse
import asyncio
import os
import queue
import subprocess
import sys
import threading
import time
import wave

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from face_utils import gather_audio_window, reshape_audio_feat
from inference import FramePicker, prepare_model_input, paste_prediction, load_model

# feather_hubert.py lives in a subfolder with no __init__.py; import it directly
# rather than shelling out to it per-request (a fresh `python feather_hubert.py`
# process pays torch-import + checkpoint-load cost -- ~1.6s measured -- on
# *every single* /speak call, which was most of the observed 4s+ latency).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_utils", "feather_hubert"))
import feather_hubert as _fh  # noqa: E402

FPS = 25
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS  # 640
MAX_CLIENT_BACKLOG = 15  # fragments a slow client can queue before we drop old ones

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE: dict = {}


def _make_fifos(hls_dir: str) -> tuple[str, str]:
    os.makedirs(hls_dir, exist_ok=True)
    for f in os.listdir(hls_dir):
        os.remove(os.path.join(hls_dir, f))
    vfifo = os.path.join(hls_dir, "video.fifo")
    afifo = os.path.join(hls_dir, "audio.fifo")
    os.mkfifo(vfifo)
    os.mkfifo(afifo)
    return vfifo, afifo


def _start_ffmpeg(hls_dir: str, vfifo: str, afifo: str, width: int, height: int) -> subprocess.Popen:
    """Fragmented MP4 to stdout, not HLS files -- a WebSocket relay (see
    _stdout_reader/_ws_broadcast below) pushes each chunk to browsers the
    moment it's produced, instead of them polling a playlist for whole
    segment files. `empty_moov` makes ffmpeg emit the init segment (ftyp+moov,
    no samples) as the very first thing it writes, before any moof/mdat
    fragment -- _extract_init_segment relies on exactly that ordering.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", vfifo,
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", afifo,
        "-vf", "scale=960:-2",
        # The <video> element is CSS-capped at 480px wide, so encoding the
        # model's native 1620x1080 output was ~2.85x more pixels than the
        # viewer ever sees -- pure waste that starved every pixel of bits at
        # any reasonable bitrate (that's what was actually blocky, not the
        # bitrate cap itself). Scaling down first lets the same bitrate buy
        # far more quality per visible pixel, and lets "veryfast" run
        # comfortably in real time (measured >1x at full res with this
        # preset+tune -- too close to the edge to risk under load).
        # x264 preset speed order is ultrafast < superfast < veryfast < faster
        # < fast < medium -- "faster" and "veryfast" both landed under 1x
        # real-time even at 960px (got the ordering backwards the first try).
        "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency",
        "-g", "3", "-maxrate", "3M", "-bufsize", "3M",
        "-c:a", "aac", "-b:a", "96k",
        "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "pipe:1",
    ]
    errlog = open(os.path.join(hls_dir, "ffmpeg.log"), "wb")
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errlog, bufsize=0)


def _read_exact(read_fn, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = read_fn(n - len(buf))
        if not chunk:
            raise EOFError("ffmpeg stdout closed while reading a box")
        buf += chunk
    return buf


def _read_one_box(read_fn) -> tuple[bytes, bytes]:
    """Read exactly one top-level MP4 box (4-byte size + 4-byte type + body,
    with the 64-bit extended-size case handled). Returns (raw_bytes, boxtype)."""
    header = _read_exact(read_fn, 8)
    size = int.from_bytes(header[:4], "big")
    boxtype = header[4:8]
    if size == 1:
        ext = _read_exact(read_fn, 8)
        size = int.from_bytes(ext, "big")
        body = _read_exact(read_fn, size - 16)
        return header + ext + body, boxtype
    body = _read_exact(read_fn, size - 8)
    return header + body, boxtype


def _extract_init_segment(read_fn) -> bytes:
    """Read top-level MP4 boxes until a `moov` box has been fully consumed.
    Returns exactly those bytes (init segment); the stream is left positioned
    at the first fragment (`styp`/`moof`/...)."""
    buf = b""
    while True:
        box, boxtype = _read_one_box(read_fn)
        buf += box
        if boxtype == b"moov":
            return buf


def _stdout_reader(proc: subprocess.Popen) -> None:
    """Broadcast one complete fragment (everything up to and including its
    `mdat` box) at a time -- never a raw byte chunk. A client that joins
    ws_clients between two fragments always starts on a fragment boundary;
    broadcasting arbitrary 64KB read()-sized chunks instead let a late-joining
    client start mid-fragment, which MSE can't decode and silently stalls on."""
    read_fn = proc.stdout.read
    STATE["init_segment"] = _extract_init_segment(read_fn)
    print(f"[live] init segment captured ({len(STATE['init_segment'])} bytes)", flush=True)
    frag = b""
    while True:
        try:
            box, boxtype = _read_one_box(read_fn)
        except EOFError:
            print("[live] ffmpeg stdout closed", flush=True)
            return
        frag += box
        if boxtype == b"mdat":
            # Each fragment starts with its own keyframe (frag_keyframe), so
            # dropping whole stale fragments is always safe to decode -- unlike
            # an unbounded queue, which just lets a client that can't drain fast
            # enough (a network slower than the encode bitrate) pile up an
            # ever-growing backlog it can never catch up from.
            for q in list(STATE["ws_clients"]):
                while q.qsize() > MAX_CLIENT_BACKLOG:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
                q.put(frag)
            frag = b""


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    while STATE.get("init_segment") is None:
        await asyncio.sleep(0.05)
    await websocket.send_bytes(STATE["init_segment"])

    q: "queue.Queue[bytes]" = queue.Queue()
    STATE["ws_clients"].append(q)
    loop = asyncio.get_event_loop()
    try:
        while True:
            chunk = await loop.run_in_executor(None, q.get)
            await websocket.send_bytes(chunk)
    except WebSocketDisconnect:
        pass
    finally:
        STATE["ws_clients"].remove(q)


def _video_writer(vfifo: str, q: "queue.Queue[bytes]") -> None:
    f = open(vfifo, "wb")  # blocks until ffmpeg opens its side for read
    print("[live] video fifo connected", flush=True)
    while True:
        f.write(q.get())


def _audio_writer(afifo: str, q: "queue.Queue[bytes]") -> None:
    f = open(afifo, "wb")  # independent of the video handshake -- separate thread
    print("[live] audio fifo connected", flush=True)
    while True:
        f.write(q.get())


def _gen_loop() -> None:
    """Runs at 25 fps. Checks STATE["next"] every frame (not just when idle)
    so a fresh /speak call always barge-in-interrupts whatever is currently
    playing -- a real-time avatar shouldn't finish a queued backlog before
    reacting to the newest thing it was told to say."""
    s = STATE
    silence = b"\x00\x00" * SAMPLES_PER_FRAME
    idle_frame = cv2.imread(os.path.join(s["image_dir"], f"{s['idle_frame_idx']}.jpg"))
    current = None
    last_idle_heartbeat = 0.0

    while True:
        t0 = time.time()

        with s["lock"]:
            nxt = s["next"]
            s["next"] = None
        if nxt is not None:
            current = nxt
            s["speaking"] = True
            print(f"[live] speaking: {current['n_frames']} frames", flush=True)

        if current is None and (t0 - last_idle_heartbeat) < 1.0:
            # A full stop breaks the muxer -- frag_keyframe closes a fragment
            # only when the NEXT keyframe arrives, so one idle frame then
            # silence forever leaves ffmpeg stuck without ever emitting even
            # the init segment. A once-a-second heartbeat keeps fragments
            # closing while still cutting idle CPU/bitrate by ~25x vs 25fps.
            time.sleep(1.0 / FPS)
            continue
        if current is None:
            last_idle_heartbeat = t0

        if current is not None:
            i = current["idx"]
            audio_feat = reshape_audio_feat(gather_audio_window(current["features"], i)).unsqueeze(0).to(s["device"])
            resource_index = s["picker"].next()
            image = cv2.imread(os.path.join(s["image_dir"], f"{resource_index}.jpg"))
            landmark_path = os.path.join(s["landmark_dir"], f"{resource_index}.lms")
            model_input, face_crop, bbox, original_size = prepare_model_input(image, landmark_path, s["device"])
            with torch.no_grad():
                prediction = s["model"](model_input, audio_feat)[0]
            prediction = (prediction.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            paste_prediction(image, prediction, face_crop, bbox, original_size)
            frame = image

            a0 = i * SAMPLES_PER_FRAME
            a1 = a0 + SAMPLES_PER_FRAME
            pcm_chunk = current["pcm"][a0 * 2: a1 * 2]
            if len(pcm_chunk) < SAMPLES_PER_FRAME * 2:
                pcm_chunk = pcm_chunk + b"\x00" * (SAMPLES_PER_FRAME * 2 - len(pcm_chunk))

            current["idx"] += 1
            if current["idx"] >= current["n_frames"]:
                current = None
                s["speaking"] = False
        else:
            frame = idle_frame
            pcm_chunk = silence

        s["video_q"].put(frame.tobytes())
        s["audio_q"].put(pcm_chunk)

        elapsed = time.time() - t0
        sleep_s = (1.0 / FPS) - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)


PAD_MS = 200  # feather_hubert's CNN feature extractor has no frames near a
              # clip's raw edges (no context to convolve over), so without
              # padding the first/last ~30ms of REAL speech silently has no
              # feature vector and never gets played -- audible clipping at
              # both ends. Padding with silence moves the edges into pure
              # silence instead, at the cost of ~2*PAD_MS of harmless quiet.
PAD_SAMPLES = int(SAMPLE_RATE * PAD_MS / 1000)


def _pad_wav_with_silence(src_path: str, dst_path: str) -> None:
    with wave.open(src_path, "rb") as r:
        assert r.getframerate() == SAMPLE_RATE and r.getnchannels() == 1
        sampwidth = r.getsampwidth()
        pcm = r.readframes(r.getnframes())
    silence = b"\x00" * (PAD_SAMPLES * sampwidth)
    with wave.open(dst_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(silence + pcm + silence)


@app.post("/speak")
async def speak(file: UploadFile):
    raw = await file.read()
    tmp_wav = f"/tmp/speak_{time.time_ns()}.wav"
    padded_wav = f"/tmp/speak_{time.time_ns()}_padded.wav"
    with open(tmp_wav, "wb") as f:
        f.write(raw)
    _pad_wav_with_silence(tmp_wav, padded_wav)
    os.remove(tmp_wav)

    with wave.open(padded_wav, "rb") as r:
        pcm = r.readframes(r.getnframes())

    features = _extract_features(padded_wav)
    os.remove(padded_wav)

    n_frames = features.shape[0]
    with STATE["lock"]:
        STATE["next"] = {"features": features, "pcm": pcm, "idx": 0, "n_frames": n_frames}
    return {"queued_frames": n_frames, "seconds": n_frames / FPS}


def _extract_features(wav_path: str) -> np.ndarray:
    """In-process now -- the resident model loaded once at startup (STATE['fh_model']),
    not a fresh `python feather_hubert.py` subprocess per call."""
    speech = _fh.read_wav_16k(wav_path)
    hidden = _fh.get_feather_hubert_from_16k_speech(speech, STATE["fh_model"], device=STATE["device"])
    hidden = _fh.make_even_first_dim(hidden).reshape(-1, 2, STATE["fh_model"].config.output_dim)
    return hidden.numpy().astype(np.float32)


@app.get("/status")
def status():
    return {"speaking": STATE.get("speaking", False), "pending": STATE.get("next") is not None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--idle_frame", type=int, default=0, help="resource frame index to hold on while idle")
    p.add_argument("--fh_checkpoint", type=str, default="./feather_hubert.pth")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = os.path.join(args.dataset, "full_body_img")
    landmark_dir = os.path.join(args.dataset, "landmarks")
    frame_count = sum(f.endswith(".jpg") for f in os.listdir(image_dir))
    first_image = cv2.imread(os.path.join(image_dir, "0.jpg"))
    height, width = first_image.shape[:2]

    hls_dir = os.path.join(os.getcwd(), "hls_live")
    vfifo, afifo = _make_fifos(hls_dir)

    STATE.update({
        "device": device,
        "image_dir": image_dir,
        "landmark_dir": landmark_dir,
        "model": load_model(args.checkpoint, device),
        "fh_model": _fh.load_feather_hubert(args.fh_checkpoint, device=device),
        "picker": FramePicker(frame_count),
        "idle_frame_idx": args.idle_frame,
        "next": None,
        "speaking": False,
        "lock": threading.Lock(),
        "video_q": queue.Queue(),
        "audio_q": queue.Queue(),
        "ws_clients": [],
        "init_segment": None,
    })

    print(f"[live] {frame_count} resource frames, {width}x{height}, idle frame #{args.idle_frame}", flush=True)

    ffmpeg_proc = _start_ffmpeg(hls_dir, vfifo, afifo, width, height)
    STATE["ffmpeg"] = ffmpeg_proc

    threading.Thread(target=_video_writer, args=(vfifo, STATE["video_q"]), daemon=True).start()
    threading.Thread(target=_audio_writer, args=(afifo, STATE["audio_q"]), daemon=True).start()
    threading.Thread(target=_gen_loop, daemon=True).start()
    threading.Thread(target=_stdout_reader, args=(ffmpeg_proc,), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
