"""Resident multi-GPU live worker for SoulX-FlashHead (run under torchrun).

Both ranks load the pipeline once and then serve live sessions forever. Rank 0
owns the control FIFO (JSON lines from the supervisor), the wall-clock pacing,
the ffmpeg HLS encoder and the status file; every chunk it broadcasts a control
flag and the 1.12 s of PCM to the other ranks, and all ranks run the sequence-
parallel generation together (xfuser USP, same seed -> same noise on every rank).

Control lines (supervisor -> rank 0):
  {"type": "start", "sid", "ref", "hls_dir", "audio_fifo", "seed", "idle_timeout", "max_seconds", "lead_chunks"}
  {"type": "audio", "pcm_b64": <int16 mono 16 kHz>, "final": bool}   # small payloads; cut into chunks here
  {"type": "audio_file", "path": <raw int16 mono 16 kHz file>, "final": bool}  # bulk PCM: read from disk, file removed
  {"type": "end"}                                                     # finish the current session
  {"type": "quit"}                                                    # exit the worker
Status lines (rank 0 -> supervisor), JSON per line:
  {"event": "worker_ready" | "session_ready" | "chunk" | "session_end" | "error", ...}
"""
import argparse
import base64
import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from collections import deque

SRC = "/root/SoulX-FlashHead"
os.chdir(SRC)
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, get_pipeline, run_pipeline  # noqa: E402


class Control:
    """Non-blocking JSON-lines reader on a FIFO (rank 0 only)."""

    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.buf = b""

    def poll(self, timeout: float = 0.0) -> list[dict]:
        out: list[dict] = []
        r, _, _ = select.select([self.fd], [], [], timeout)
        if r:
            # drain everything available: a pipe holds 64 KB, and reading once per
            # loop iteration made a long line crawl through at 64 KB per chunk
            while True:
                try:
                    data = os.read(self.fd, 1 << 20)
                except BlockingIOError:
                    break
                if not data:
                    break
                self.buf += data
                if len(data) < 65536:
                    break
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    def wait_for(self, kinds: tuple[str, ...]) -> dict:
        while True:
            for msg in self.poll(timeout=0.5):
                if msg.get("type") in kinds:
                    return msg


def start_ffmpeg(hls_dir: str, audio_fifo: str, width: int, height: int, fps: int, sr: int, slice_len: int, log_path: str):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-thread_queue_size", "1024", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-thread_queue_size", "1024", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", audio_fifo,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-crf", "20",
        "-g", str(slice_len), "-keyint_min", str(slice_len), "-sc_threshold", "0", "-bf", "0",
        "-c:a", "aac", "-b:a", "128k",
        "-f", "hls", "-hls_time", f"{slice_len / fps:.2f}", "-hls_segment_type", "fmp4",
        "-hls_playlist_type", "event", "-hls_list_size", "0",
        "-hls_flags", "independent_segments+temp_file",
        "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", os.path.join(hls_dir, "seg_%04d.m4s"),
        os.path.join(hls_dir, "index.m3u8"),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=open(log_path, "wb"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--wav2vec_dir", required=True)
    ap.add_argument("--model_type", default="pro")
    ap.add_argument("--control_fifo", required=True)
    ap.add_argument("--status_file", required=True)
    ap.add_argument("--warmup_image", default="")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    status = open(args.status_file, "a") if rank == 0 else None

    def emit(obj: dict) -> None:
        if status is not None:
            status.write(json.dumps(obj) + "\n")
            status.flush()

    t_load = time.time()
    pipeline = get_pipeline(world, args.ckpt_dir, args.model_type, args.wav2vec_dir)
    device = pipeline.device
    p = get_infer_params()
    sr, fps = p["sample_rate"], p["tgt_fps"]
    frame_num, motion = p["frame_num"], p["motion_frames_num"]
    slice_len = frame_num - motion
    chunk_samples = slice_len * sr // fps
    chunk_s = slice_len / fps
    cached = p["cached_audio_duration"] * sr
    end_idx = p["cached_audio_duration"] * fps
    start_idx = end_idx - frame_num
    warm_s = 0.0
    if args.warmup_image and os.path.exists(args.warmup_image):
        # first generation in a fresh process costs ~1.9 s instead of 0.7 s
        # (allocator, cuDNN, NCCL warm-up); take it here, not on the first live chunk
        t_w = time.time()
        get_base_data(pipeline, cond_image_path_or_dir=args.warmup_image, base_seed=0, use_face_crop=True)
        dq = deque([0.0] * cached, maxlen=cached)
        emb = get_audio_embedding(pipeline, np.array(dq, dtype=np.float32), start_idx, end_idx)
        for _ in range(2):
            run_pipeline(pipeline, emb)
        torch.cuda.synchronize()
        warm_s = time.time() - t_w
    ctl = Control(args.control_fifo) if rank == 0 else None
    emit({"event": "worker_ready", "model": args.model_type, "world": world, "chunk_s": chunk_s, "load_s": round(time.time() - t_load, 1), "warmup_s": round(warm_s, 1)})

    while True:
        info = [None]
        if rank == 0:
            info[0] = ctl.wait_for(("start", "quit"))
        dist.broadcast_object_list(info, src=0)
        info = info[0]
        if info["type"] == "quit":
            break
        try:
            run_session(info, pipeline, device, rank, ctl, emit, p, sr, fps, motion, slice_len, chunk_samples, chunk_s, cached, start_idx, end_idx)
        except Exception as exc:  # noqa: BLE001
            emit({"event": "error", "sid": info.get("sid"), "error": f"{type(exc).__name__}: {exc}"[:400]})
            raise
    dist.barrier()
    dist.destroy_process_group()


def run_session(info, pipeline, device, rank, ctl, emit, p, sr, fps, motion, slice_len, chunk_samples, chunk_s, cached, start_idx, end_idx) -> None:
    sid = info["sid"]
    t0 = time.time()
    get_base_data(pipeline, cond_image_path_or_dir=info["ref"], base_seed=int(info.get("seed", 42)), use_face_crop=True)
    audio_dq: deque = deque([0.0] * cached, maxlen=cached)
    lead = float(info.get("lead_chunks", 1.0))
    idle_timeout = float(info.get("idle_timeout", 180.0))
    max_seconds = float(info.get("max_seconds", 1800.0))

    proc = None
    vq: queue.Queue = queue.Queue()
    aq: queue.Queue = queue.Queue()
    if rank == 0:
        os.makedirs(info["hls_dir"], exist_ok=True)
        if not os.path.exists(info["audio_fifo"]):
            os.mkfifo(info["audio_fifo"])
        proc = start_ffmpeg(info["hls_dir"], info["audio_fifo"], p["width"], p["height"], fps, sr, slice_len, os.path.join(os.path.dirname(info["hls_dir"]), "ffmpeg.log"))

        def vwriter() -> None:
            try:
                while (arr := vq.get()) is not None:
                    proc.stdin.write(arr.tobytes())
            finally:
                proc.stdin.close()

        def awriter() -> None:
            with open(info["audio_fifo"], "wb") as f:
                while (pcm := aq.get()) is not None:
                    f.write(pcm)

        threading.Thread(target=vwriter, daemon=True).start()
        threading.Thread(target=awriter, daemon=True).start()
        emit({"event": "session_ready", "sid": sid, "prep_s": round(time.time() - t0, 1)})

    buf = np.zeros(0, dtype=np.float32)
    pending: deque = deque()
    ended = False
    last_speech = time.time()
    stream_t0 = None
    k = 0
    speech_chunks = 0
    gens: list[float] = []
    reason = ""
    flag_t = torch.zeros(1, dtype=torch.int64, device=device)
    pcm_t = torch.zeros(chunk_samples, dtype=torch.float32, device=device)
    while True:
        behind = 0.0
        if rank == 0:
            for msg in ctl.poll():
                t = msg.get("type")
                if t == "end":
                    ended = True
                elif t in ("audio", "audio_file"):
                    if t == "audio_file":
                        arr = np.fromfile(msg["path"], dtype=np.int16).astype(np.float32) / 32768.0
                        try:
                            os.remove(msg["path"])
                        except OSError:
                            pass
                    else:
                        arr = np.frombuffer(base64.b64decode(msg["pcm_b64"]), dtype=np.int16).astype(np.float32) / 32768.0
                    buf = np.concatenate([buf, arr])
                    while len(buf) >= chunk_samples:
                        pending.append(buf[:chunk_samples]); buf = buf[chunk_samples:]
                    if msg.get("final") and len(buf):
                        pending.append(np.concatenate([buf, np.zeros(chunk_samples - len(buf), dtype=np.float32)])); buf = buf[:0]
                elif t == "quit":
                    ended = True
            now = time.time()
            if ended and not pending:
                flag = -1; reason = "stop"
            elif not pending and now - last_speech > idle_timeout:
                flag = -1; reason = "idle"
            elif now - t0 > max_seconds:
                flag = -1; reason = "max_seconds"
            else:
                if stream_t0 is None:
                    stream_t0 = now
                sched = stream_t0 + (k - lead) * chunk_s
                if now < sched:
                    time.sleep(min(sched - now, 0.2))
                    continue
                behind = now - sched
                if pending:
                    pcm = pending.popleft(); flag = 1; last_speech = now
                else:
                    pcm = np.zeros(chunk_samples, dtype=np.float32); flag = 0
                pcm_t.copy_(torch.from_numpy(np.ascontiguousarray(pcm)))
            flag_t.fill_(flag)
        dist.broadcast(flag_t, src=0)
        flag = int(flag_t.item())
        if flag < 0:
            break
        dist.broadcast(pcm_t, src=0)
        pcm = pcm_t.cpu().numpy()
        g0 = time.time()
        audio_dq.extend(pcm.tolist())
        emb = get_audio_embedding(pipeline, np.array(audio_dq, dtype=np.float32), start_idx, end_idx)
        video = run_pipeline(pipeline, emb)[motion:]
        k += 1
        if rank == 0:
            vq.put(video.cpu().numpy().astype(np.uint8))
            aq.put((np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            gen = time.time() - g0
            gens.append(gen)
            speech_chunks += int(flag == 1)
            emit({"event": "chunk", "sid": sid, "k": k, "speech": flag == 1, "gen": round(gen, 3), "behind": round(behind, 2),
                  "queued": round(len(pending) * chunk_s, 1)})
    if rank == 0:
        vq.put(None); aq.put(None)
        rc = proc.wait(timeout=120) if proc else None
        gens_sorted = sorted(gens)
        emit({"event": "session_end", "sid": sid, "reason": reason, "chunks": k, "speech_chunks": speech_chunks,
              "gen_median_s": round(gens_sorted[len(gens_sorted) // 2], 3) if gens else None, "ffmpeg_rc": rc,
              "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
