"""SoulX-FlashHead as a resident renderer — the part that has no host in it.

This module is plain Python and PyTorch: the pipeline, the per-slot scheduler,
idle-loop playback and HLS shipping. It imports no cloud SDK, so the same code
runs as a Modal class, as a process on a rented GPU box (Lightning, RunPod,
vast.ai, Lambda, a machine under a desk), or locally.

Everything a host provides arrives through one small seam, `Host`:

    put / get_many     move items between the renderer and whatever drives it
    publish_ready      announce that the pipeline can take calls (optional)
    commit             persist shared storage after a write (optional)

`LocalHost` below is the whole implementation for a single box — the renderer
and its driver share a process, so "transport" is a `queue.Queue`. `app.py`
holds the Modal adapter, which maps the same four methods onto `modal.Queue`,
`modal.Dict` and `modal.Volume`.

Paths and sizes come from the environment so a host can place them anywhere:

    FLASHHEAD_SRC       upstream checkout            (default /root/SoulX-FlashHead)
    FLASHHEAD_WEIGHTS   model weights                (default /weights)
    FLASHHEAD_CACHE     inductor/triton/HF caches    (default /cache)
    FLASHHEAD_WARMUP    warm-up reference image      (default /root/warmup.png)
    FLASHHEAD_MAX_SESSIONS  live sessions per container (default 8)
    FLASHHEAD_COMPILE   1 to leave upstream's torch.compile on
"""

import json
import os
import pathlib
import queue as _queue
import threading
import time
from collections import deque

SRC = os.environ.get("FLASHHEAD_SRC", "/root/SoulX-FlashHead")
WEIGHTS = os.environ.get("FLASHHEAD_WEIGHTS", "/weights")
CACHE = os.environ.get("FLASHHEAD_CACHE", "/cache")
WARMUP_IMAGE = os.environ.get("FLASHHEAD_WARMUP", "/root/warmup.png")
CKPT = f"{WEIGHTS}/SoulX-FlashHead-1_3B"
WAV2VEC = f"{WEIGHTS}/wav2vec2-base-960h"
GPU = os.environ.get("FLASHHEAD_GPU", "L4")
COMPILE = os.environ.get("FLASHHEAD_COMPILE", "0") == "1"
# Per-stage timing inside a batched generate. Off by default: it needs a
# cuda.synchronize between stages, which is the very thing stripped from
# upstream for costing a slot 8 stalls per chunk.
STAGE_TIMING = os.environ.get("FLASHHEAD_STAGE_TIMING", "0") == "1"
# Decompose the residency cost — what one live session adds to a slot whether
# or not it speaks. Capacity is (slot - fixed) / (residency + speech x duty),
# and at a 17% duty the residency term is weighted six times heavier, so this
# is the number to know. Off by default; it adds a perf_counter per section.
RESIDENCY_TIMING = os.environ.get("FLASHHEAD_RESIDENCY_TIMING", "0") == "1"
# Turns that began within this many seconds of a session's start are left out of
# the wait statistics. See _stats: a benchmark is observed from an empty system
# and its first turns never queue, which biases every point optimistic.
WARMUP_S = float(os.environ.get("FLASHHEAD_WARMUP_S", "90"))
# How many chunks a session may generate ahead of what it is playing out.
#
# The audio for a turn arrives as a whole TTS file, so there is no reason to
# generate it in lockstep with playback. Generating 0.96 s of video costs about
# a quarter of a second of GPU, so an utterance needs ~1.8 s of GPU spread over
# the 7 s it takes to say — but the lockstep scheduler demanded a budget slot in
# *every one* of those slots, and a speaker held one for the whole utterance.
# With a lead, a session that is already a few chunks ahead asks for nothing,
# and the budget it is not using goes to whoever wants to start speaking.
#
# Generation runs ahead; emission does not. Exactly one chunk leaves per slot,
# so the viewer's buffer depth — and with it the latency from "decided to speak"
# to "the candidate sees it" — is unchanged. Shipping early would trade the wait
# we are trying to remove for an equal delay in every frame.
#
# 0 restores the old lockstep behaviour.
LOOKAHEAD = int(os.environ.get("FLASHHEAD_LOOKAHEAD", "3"))
DONE_MARK = "__done__"


class Host:
    """What the renderer needs from wherever it is running.

    The renderer never asks for more than this, which is why moving hosts is a
    new adapter rather than a new renderer. Only `put`/`get_many` carry work;
    the other two are announcements a single-box host can ignore.
    """

    def put(self, item, partition: str) -> None:
        raise NotImplementedError

    def get_many(self, n: int, partition: str, block: bool = False,
                 timeout: float | None = None) -> list:
        raise NotImplementedError

    def publish_ready(self, key: str, payload: dict | None) -> None:
        """Announce readiness. `None` clears it. No-op where nothing watches."""

    def commit(self) -> None:
        """Flush shared storage after writing to it. No-op on a real filesystem."""


class LocalHost(Host):
    """Everything a single box needs: the driver is in this process, so items
    go through an in-memory queue per partition and storage is just the disk."""

    def __init__(self) -> None:
        self._qs: dict[str, _queue.Queue] = {}
        self._lock = threading.Lock()
        self.ready: dict[str, dict] = {}

    def _q(self, partition: str) -> _queue.Queue:
        with self._lock:
            return self._qs.setdefault(partition, _queue.Queue())

    def put(self, item, partition: str) -> None:
        self._q(partition).put(item)

    def get_many(self, n: int, partition: str, block: bool = False,
                 timeout: float | None = None) -> list:
        q, out = self._q(partition), []
        try:
            out.append(q.get(block=block, timeout=timeout))
        except _queue.Empty:
            return out
        while len(out) < n:
            try:
                out.append(q.get_nowait())
            except _queue.Empty:
                break
        return out

    def publish_ready(self, key: str, payload: dict | None) -> None:
        self.ready.pop(key, None) if payload is None else self.ready.update({key: payload})


IDLE_DIR = f"{WEIGHTS}/idle_loops"
IDLE_CHUNKS = 10                      # ~9.6 s of loop; long enough that a repeat is not obvious


def _idle_loop_path(image_bytes: bytes, seed: int, model_type: str) -> pathlib.Path:
    import hashlib

    key = hashlib.sha256(image_bytes + f"|{seed}|{model_type}|{IDLE_CHUNKS}".encode()).hexdigest()[:16]
    return pathlib.Path(IDLE_DIR) / f"{key}.mp4"


def _emit(host, job_id: str, line: str) -> None:
    print(line, flush=True)
    try:
        host.put(line, partition=job_id)
    except Exception:  # noqa: BLE001 - progress is best-effort, never fail a render over it
        pass


# One container, many sessions. The GPU can only draw for one session at a time
# (a chunk takes 0.82 s of its 0.96 s slot on an L4), but sessions mostly *listen*
# — and a listening avatar plays a pre-rendered loop, costing nothing. So N
# sessions share one GPU as long as they rarely speak at the same moment; when
# they do, the loser keeps playing its loop and its speech starts a beat later.
LIVE_MAX_SESSIONS = int(os.environ.get("FLASHHEAD_MAX_SESSIONS", "8"))   # per container; a faster card holds more

# ---------------------------------------------------------------------------
# Live session support: HLS shipping shared by the live loop (segments and
# log lines batched into Queue items, sent from a thread, see render_stream).
class _HlsShipper:
    FILE_PART = 900_000

    def __init__(self, host: Host, partition: str, hls_dir: pathlib.Path, t0: float):
        import queue
        import re
        import threading

        self.host = host
        self.partition, self.hls, self.t0 = partition, hls_dir, t0
        self.batch: list[dict] = []
        self.batch_bytes = 0
        self.sent: set[str] = set()
        self.last_playlist = ""
        self.streamed = 0
        self.errors: list[str] = []
        self.seg_re = re.compile(r"^(seg_\d+\.m4s)$", re.M)
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._sender, daemon=True)
        self.thread.start()

    def _sender(self) -> None:
        while (item := self.q.get()) is not None:
            try:
                self.host.put(item, partition=self.partition)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}"[:200])

    def put(self, item: dict) -> None:
        size = len(item.get("data", b"")) + 200
        if self.batch and self.batch_bytes + size > self.FILE_PART:
            self.flush()
        self.batch.append(item)
        self.batch_bytes += size

    def flush(self) -> None:
        if self.batch:
            self.q.put({"type": "batch", "items": list(self.batch)})
            self.batch.clear()
            self.batch_bytes = 0

    def log(self, line: str) -> None:
        line = f"[{time.time()-self.t0:6.1f}s] {line}"
        print(line, flush=True)
        self.put({"type": "log", "line": line})
        self.flush()

    def send_file(self, name: str, data: bytes) -> None:
        import math

        parts = max(1, math.ceil(len(data) / self.FILE_PART))
        for i in range(parts):
            self.put({"type": "file", "name": name, "part": i, "parts": parts,
                      "data": data[i * self.FILE_PART : (i + 1) * self.FILE_PART]})
        self.streamed += len(data)

    def ship_new_files(self) -> int:
        """Ship init (once it is complete), newly listed segments, then the playlist."""
        pl = self.hls / "index.m3u8"
        if not pl.exists():
            return 0
        text = pl.read_text()
        n = 0
        init = self.hls / "init.mp4"
        if "init.mp4" not in self.sent and self.seg_re.search(text) and init.exists() and init.stat().st_size > 0:
            self.sent.add("init.mp4")
            self.send_file("init.mp4", init.read_bytes())
        if "init.mp4" not in self.sent:
            return 0
        for name in self.seg_re.findall(text):
            if name not in self.sent and (self.hls / name).exists():
                self.sent.add(name)
                self.send_file(name, (self.hls / name).read_bytes())
                n += 1
        if text != self.last_playlist and text.endswith("\n"):
            self.last_playlist = text
            self.send_file("index.m3u8", text.encode())
        self.flush()
        return n

    def close(self) -> None:
        self.flush()
        self.q.put(None)
        self.thread.join(timeout=120)


class _AudioWindow:
    """The rolling audio window a chunk is conditioned on, as one float32 array.

    It used to be a `deque` of 128,000 Python floats: every slot turned a numpy
    chunk into 15,360 float objects to append, and every speaking slot turned the
    whole deque back into an array. Both directions were pure Python and held the
    GIL, on the one thread that drives every session.

    Writes advance a cursor in a doubled buffer so the live window is always one
    contiguous slice — no copy to read, no allocation to write.
    """

    def __init__(self, size: int):
        import numpy as np

        self.n = size
        self._buf = np.zeros(size * 2, dtype=np.float32)
        self._pos = size                      # window is _buf[_pos - n : _pos]

    def push(self, pcm) -> None:
        k = len(pcm)
        if self._pos + k > len(self._buf):    # wrap: carry the live window to the front
            self._buf[: self.n] = self._buf[self._pos - self.n : self._pos]
            self._pos = self.n
        self._buf[self._pos : self._pos + k] = pcm
        self._pos += k

    def view(self):
        """The current window. A view — the caller must not write into it."""
        return self._buf[self._pos - self.n : self._pos]


def _loop_window(loop, pos: int, n: int):
    """`n` frames of `loop` starting at `pos` — a view unless the window wraps.

    This runs on the scheduler thread for every listening session in every 0.96 s
    slot, and one chunk is 24 × 512 × 512 × 3 = 18.9 MB. `np.take` always copies:
    at eight sessions that was 158 MB/s of pure memcpy on a single thread, which
    capped the container at ~8 sessions while the GPU sat idle. A slice of a
    C-contiguous array costs nothing, and only the wrap — once per loop cycle,
    about one slot in ten — has to concatenate.

    The result may alias `loop`, so callers must not write into it. The one that
    does (the fade back into the loop) copies first.
    """
    import numpy as np

    end = pos + n
    if end <= len(loop):
        return loop[pos:end]
    return np.concatenate((loop[pos:], loop[:end - len(loop)]))


# ---------------------------------------------------------------------------
# Batching. Upstream generates one chunk per call, which leaves a big card idle:
# the weights have to be read from HBM whatever the batch size, so on a card with
# compute to spare several sessions cost barely more than one. Measured on an
# H200: 8 chunks together took 0.597 s against 8 × 0.173 s apart — 2.3× more work
# per second. On an L4 it is exactly linear (already compute-bound at batch 1),
# so the win is real but card-dependent.
#
# Three places upstream assume a batch of one. Two of them fail silently.
def _rope_apply_batched(x, freqs, grid_sizes, use_usp=False, sp_size=1, sp_rank=0):
    """Rotary embedding over the batch. Upstream reads `x[0]` and re-wraps it, so
    for B>1 it hands item 0's result to every session — no error, just four
    identical faces. Every session in a slot shares the resolution and frame
    count, so the rotary table is shared and only `x` carries a batch."""
    import torch

    b, s_len, n, c2 = x.shape
    c = c2 // 2
    fq = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes
    seq_len = f * h * w
    freqs_i = torch.cat([
        fq[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fq[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fq[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(1, seq_len, 1, -1)
    xc = torch.view_as_complex(x[:, :seq_len].to(torch.float64).reshape(b, seq_len, n, -1, 2))
    out = torch.view_as_real(xc * freqs_i).flatten(3)
    if s_len > seq_len:
        out = torch.cat([out, x[:, seq_len:]], dim=1)
    return out.to(x.dtype)


def _block_forward_batched(self, x, context, t_mod, freqs, grid_sizes):
    """The DiT block's cross-attention. The `(b f)` folding was already there;
    only `context.squeeze(0)` and the `.unsqueeze(0)` after it assumed one item."""
    from einops import rearrange

    b = x.shape[0]
    e = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], freqs, grid_sizes)
    x = x + y * e[2]
    f = context.shape[1]
    x_1 = rearrange(self.norm3(x), "b (f l) c -> (b f) l c", f=f)
    context_1 = rearrange(context, "b f m c -> (b f) m c")
    x = x + rearrange(self.cross_attn(x_1, context_1), "(b f) l c -> b (f l) c", b=b)
    y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
    return x + y * e[5]


def _enable_batching() -> str:
    """Patch upstream in place so a batch above one is computed correctly.

    The third assumption — `Head.forward` reading its batch from `t_mod`'s rows
    (its docstring says `[B*21, C]`) — needs no patch: widening `timestep` to one
    row per item is enough, which `_generate_batch` does.
    """
    from flash_head.src.modules import flash_head_model as M

    if getattr(M, "_flashhead_batched", False):
        return "already"
    # `cross_attn` / `self_attn` are set in __init__, so they are instance
    # attributes and cannot be used to find the class. Match on the name and on
    # the source instead, and refuse to guess if that is not exactly one class —
    # patching the wrong forward would corrupt output without raising.
    import inspect

    cands = [c for n in dir(M) if isinstance(c := getattr(M, n), type)
             and n.endswith("Block") and "cross_attn" in (inspect.getsource(c.forward)
                                                          if hasattr(c, "forward") else "")]
    if len(cands) != 1:
        raise RuntimeError(f"배치 패치 대상 블록을 특정하지 못했다: {[c.__name__ for c in cands]}")
    M.rope_apply = _rope_apply_batched
    cands[0].forward = _block_forward_batched
    M._flashhead_batched = True
    return cands[0].__name__


def _live_ffmpeg(hls: pathlib.Path, width: int, height: int, fps: int, sr: int, audio_fifo: str, slice_len: int):
    """Long-lived HLS encoder for a live session: raw frames on stdin, raw PCM on a FIFO."""
    import subprocess

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
        "-hls_segment_filename", str(hls / "seg_%04d.m4s"),
        str(hls / "index.m3u8"),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=open(hls.parent / "ffmpeg.log", "wb"))

def _write_mp4(path: pathlib.Path, frames, fps: int) -> bytes:
    import imageio

    with imageio.get_writer(str(path), format="mp4", mode="I", fps=fps, codec="h264",
                            ffmpeg_params=["-bf", "0", "-crf", "18", "-pix_fmt", "yuv420p"]) as w:
        for f in frames:
            w.append_data(f)
    return path.read_bytes()


class RendererCore:
    """The renderer itself. `host` is the only thing it knows about the
    world outside; `model_type` is "lite" or "pro"."""

    def __init__(self, host: Host | None = None, model_type: str = "lite") -> None:
        self.host = host or LocalHost()
        self.model_type = model_type

    def load(self) -> None:
        import os
        import sys
        import threading

        # flash_head/inference.py opens flash_head/configs/infer_params.yaml by a
        # path relative to the CWD at import time, so chdir first.
        os.chdir(SRC)
        sys.path.insert(0, SRC)
        t0 = time.time()
        from flash_head.inference import get_pipeline

        self.pipeline = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type=self.model_type, wav2vec_dir=WAV2VEC)
        load_s = time.time() - t0
        # First-call costs (mediapipe init, cuDNN autotune, first VAE/denoise
        # pass) were 24 s on the first render of a fresh container and 0.3 s
        # after. Pay them here so a pre-warmed container answers in seconds.
        t1 = time.time()
        import numpy as np
        import torch
        from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline

        get_base_data(self.pipeline, cond_image_path_or_dir=WARMUP_IMAGE, base_seed=0, use_face_crop=True)
        p = get_infer_params()
        n = p["frame_num"] * p["sample_rate"] // p["tgt_fps"]
        emb = get_audio_embedding(self.pipeline, np.zeros(n, dtype=np.float32))
        t_gen = time.time()
        video = run_pipeline(self.pipeline, emb[:, : p["frame_num"]].contiguous())
        # Seed the per-slot budget's estimate. The pull to host memory belongs in
        # the measurement: the scheduler holds the GPU lock across it, so it is
        # part of what one slot costs. Timing run_pipeline alone read ~0.12 s
        # optimistic against the in-session median on the same container, which
        # on a faster GPU is enough to hand the first slots a budget too large.
        warm = video[self.pipeline.motion_frames_num:].to(torch.uint8).cpu().numpy()
        self.gen_ema = time.time() - t_gen
        del warm
        gpu_s = time.time() - t1
        # librosa JIT-compiles (numba) on first use and imageio spins up its
        # ffmpeg plugin on first write — together ~25 s on a fresh container.
        t2 = time.time()
        import tempfile

        import wave

        import imageio
        import librosa

        with tempfile.TemporaryDirectory() as d:
            with wave.open(f"{d}/w.wav", "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(p["sample_rate"]); w.writeframes(b"\x00\x00" * n)
            librosa.load(f"{d}/w.wav", sr=p["sample_rate"], mono=True)
            with imageio.get_writer(f"{d}/w.mp4", format="mp4", mode="I", fps=p["tgt_fps"], codec="h264", ffmpeg_params=["-bf", "0"]) as w:
                w.append_data(video[0].to(torch.uint8).cpu().numpy())
        io_s = time.time() - t2
        self.load_seconds = time.time() - t0
        self.loaded_at = time.time()
        self._ready = {"since": self.loaded_at, "weights_s": round(load_s), "warmup_s": round(gpu_s + io_s)}

        # The flag carries a heartbeat: a deploy replaces this container and the
        # exit hook is not guaranteed to run, so readers treat a flag whose beat
        # is older than ~60 s as stale instead of trusting its mere presence.
        # every GPU op goes through this: `generate()` and `reset_person_name()`
        # write pipeline instance state, so concurrent sessions must take turns
        self.gpu_lock = threading.Lock()
        self.avatars: dict = {}            # sha -> registered in the pipeline
        self.sessions: dict = {}           # session_id -> per-session pipeline state
        self.loop_locks: dict = {}         # one idle loop per avatar, rendered once
        self.loop_locks_guard = threading.Lock()
        self.idle_loops: dict = {}         # avatar -> decoded loop frames, shared by sessions
        self.live_sessions: dict = {}      # session_id -> everything the scheduler needs
        self.sched_thread = None           # the one thread that produces live video
        self.sched_guard = threading.Lock()
        self.slot_t0 = None
        self.slot = 0
        self.slot_overruns = 0             # slots where ongoing utterances exceeded the budget
        # how many sessions actually generated speech in the same slot, counted per
        # slot. This is the whole point of a faster card: on L4 the budget is 1 and
        # this histogram can only ever fill index 1.
        self.slot_hist = [0] * (LIVE_MAX_SESSIONS + 1)
        self.slot_budget = 0
        # NOTE: gen_ema is seeded by the warm-up above; do not reset it here
        self.t_swap = self.t_pipe = self.t_xfer = 0.0   # _run_chunk's three parts
        # (batch size, seconds) for the slots that generated; the per-slot budget
        # is fitted from these because with batching the cost is no longer
        # proportional to the number of speakers
        self.batch_samples = deque(maxlen=200)
        self.res_t: dict = {}      # residency seconds by section
        self.res_n = 0             # session-slots those seconds cover
        self.res_slots = 0
        batched = _enable_batching()
        ready_key = f"ready:{self.model_type}"

        def beat() -> None:
            while True:
                try:
                    self.host.publish_ready(ready_key, {**self._ready, "beat": time.time()})
                except Exception:  # noqa: BLE001 - best effort
                    pass
                time.sleep(20)

        threading.Thread(target=beat, daemon=True, name="ready-beat").start()
        # Which attention kernel actually got picked, and at what precision: the
        # image installs flash_attn but swallows a build failure, so without this
        # a silent fall back to SDPA looks identical to success.
        from flash_head.src.modules import flash_head_model as fhm

        attn = ("SageAttention" if fhm.SAGE_ATTN_AVAILABLE else
                "FlashAttention-3" if fhm.FLASH_ATTN_3_AVAILABLE else
                "FlashAttention-2" if fhm.FLASH_ATTN_2_AVAILABLE else "SDPA")
        print(f"pipeline resident ({self.model_type}): weights {load_s:.0f}s + gpu warmup {gpu_s:.0f}s "
              f"+ io warmup {io_s:.0f}s · chunk {self.gen_ema:.2f}s → slot budget "
              f"{self._batch_budget((p['frame_num'] - p['motion_frames_num']) / p['tgt_fps'])}\n"
              f"  {GPU} · attn {attn} · dtype {self.pipeline.param_dtype} · compile {COMPILE}"
              f" · batching {batched}", flush=True)

    def unload(self) -> None:
        self.host.publish_ready(f"ready:{self.model_type}", None)
        print("container exiting; ready flag cleared", flush=True)

    def _ensure_avatar(self, image_bytes: bytes, use_face_crop: bool = True) -> str:
        """Register one avatar in the pipeline without disturbing the others.

        `prepare_params` REPLACES cond_image_dict, so calling it per session would
        wipe every other session's reference. This repeats just its per-avatar body
        — the pipeline keeps a dict of them precisely so `reset_person_name` can
        switch between avatars at zero cost."""
        import hashlib
        import tempfile

        from flash_head.src.pipeline.flash_head_pipeline import get_cond_image_dict
        from flash_head.utils.utils import resize_and_centercrop

        name = "av" + hashlib.sha256(image_bytes).hexdigest()[:12]
        with self.gpu_lock:
            if name in self.avatars:
                return name
            d = pathlib.Path(tempfile.mkdtemp(prefix="av-"))
            path = d / f"{name}.png"
            path.write_bytes(image_bytes)
            pil = get_cond_image_dict(str(path), use_face_crop)[name]
            pl = self.pipeline
            t = resize_and_centercrop(pil, (pl.target_h, pl.target_w)).to(pl.device, dtype=pl.param_dtype)
            t = (t / 255 - 0.5) * 2
            pl.cond_image_dict[name] = pil
            pl.cond_image_tensor_dict[name] = t
            pl.ref_img_latent_dict[name] = pl.vae.encode(t.repeat(1, 1, pl.frame_num, 1, 1))
            self.avatars[name] = True
            path.unlink()
            d.rmdir()
            return name

    def _gpu_free(self) -> bool:
        """Is the GPU idle right now? Used to decide whether an utterance may start;
        best-effort, since another session can take it a moment later."""
        if self.gpu_lock.acquire(blocking=False):
            self.gpu_lock.release()
            return True
        return False

    def _run_chunk(self, sess: dict, emb, drop_motion: bool = True):
        """Generate one chunk for `sess`, swapping its pipeline state in and out
        under the lock. The audio encoder is read-only, so its work stays outside.
        Returns (frames uint8, generate seconds, seconds spent waiting for the GPU)."""
        import torch

        from flash_head.inference import run_pipeline

        w0 = time.time()
        with self.gpu_lock:
            t0 = time.time()
            pl = self.pipeline
            pl.reset_person_name(sess["avatar"])          # also resets latent_motion_frames
            if sess.get("motion") is not None:
                pl.latent_motion_frames = sess["motion"]
            pl.generator = sess["generator"]
            t_swap = time.time()
            video = run_pipeline(pl, emb)
            t_pipe = time.time()
            sess["motion"] = pl.latent_motion_frames
            if drop_motion:
                video = video[pl.motion_frames_num:]
            frames = video.to(torch.uint8).cpu().numpy()
        gen = time.time() - t0
        # Split the slot cost three ways. The model's own stages account for
        # ~0.75 s; the rest is state swap and the pull to host memory, and only
        # a split measured here says which of the two moved when the total does.
        self.t_swap = 0.9 * self.t_swap + 0.1 * (t_swap - t0) if self.t_swap else t_swap - t0
        self.t_pipe = 0.9 * self.t_pipe + 0.1 * (t_pipe - t_swap) if self.t_pipe else t_pipe - t_swap
        self.t_xfer = 0.9 * self.t_xfer + 0.1 * (gen + t0 - t_pipe) if self.t_xfer else gen + t0 - t_pipe
        # Record the measurement here, where it is taken: every live generation
        # path goes through this method (idle-loop render, clip render, the
        # scheduler); the warm-up in load() seeds it separately. Keeping it in
        # the scheduler alone threw away the measurements a container makes
        # before its first live chunk, leaving the per-slot budget to be
        # computed from an empty estimate.
        self.gen_ema = 0.9 * self.gen_ema + 0.1 * gen if self.gen_ema else gen
        return frames, gen, t0 - w0

    def _audio_embed_batch(self, windows: list, start_idx: int, end_idx: int):
        """`get_audio_embedding` for several sessions at once.

        Upstream runs wav2vec one session at a time — `preprocess_audio` does
        `unsqueeze(0)` on the way in and `squeeze(0)` on the way out — so a slot
        with eight speakers paid eight forwards. The encoder itself is batch
        native; only the wrapper was not. This was 58% of what a session costs the
        scheduler in a slot, the single largest item after generation itself.

        Every window is the same length, so nothing has to be padded.
        """
        import numpy as np
        import torch
        from einops import rearrange

        from flash_head.inference import infer_params

        pl = self.pipeline
        sr, fps = infer_params["sample_rate"], infer_params["tgt_fps"]
        feats = pl.wav2vec_feature_extractor(list(windows), sampling_rate=sr).input_values
        feat = torch.as_tensor(np.asarray(feats, dtype=np.float32), device=pl.device)
        with torch.no_grad():
            out = pl.audio_encoder(feat, seq_len=int(len(windows[0]) * fps / sr),
                                   output_hidden_states=True)
        emb = rearrange(torch.stack(out.hidden_states[1:], dim=1), "b l s d -> b s l d")

        # the same five-frame window upstream takes around each frame
        idx = (torch.arange(2 * 2 + 1) - 2).unsqueeze(0) + torch.arange(start_idx, end_idx).unsqueeze(1)
        return emb[:, torch.clamp(idx, min=0, max=end_idx - 1)].contiguous()

    def _generate_batch(self, sessions: list, embs: list):
        """`FlashHeadPipeline.generate` for several sessions at once.

        Mirrors upstream step for step, with a batch dimension kept throughout and
        the per-session state stacked instead of swapped in and out. Returns the
        frames for each session and its new motion latents.

        Noise is drawn per session from that session's own generator, so a session
        renders identically whether it ran alone or in a batch.
        """
        import torch
        from flash_head.utils.utils import match_and_blend_colors_torch

        pl = self.pipeline
        b = len(sessions)
        lat = ((pl.frame_num - 1) // pl.config.vae_stride[0] + 1, pl.lat_h, pl.lat_w)
        shape = (pl.config.out_dim, *lat)

        x = torch.stack([torch.randn(shape, dtype=pl.param_dtype, device=pl.device,
                                     generator=s["generator"]) for s in sessions])
        motion = torch.stack([s["motion"] if s.get("motion") is not None
                              else pl.ref_img_latent_dict[s["avatar"]][:, :1] for s in sessions])
        ref = torch.stack([pl.ref_img_latent_dict[s["avatar"]] for s in sessions])
        # embs is either the slot's batched context (B, ...) or one tensor per
        # session, which is what the single-session callers still hand over
        ctx = (embs.to(pl.device) if torch.is_tensor(embs)
               else torch.cat([e.to(pl.device) for e in embs], dim=0))
        ts, n_ts = pl.timesteps, pl.num_timesteps

        stage: dict = {}

        def mark(name: str, t0: float) -> float:
            if not STAGE_TIMING:
                return 0.0
            torch.cuda.synchronize()
            now = time.time()
            stage[name] = stage.get(name, 0.0) + now - t0
            return now

        t = time.time() if STAGE_TIMING else 0.0
        with torch.no_grad():
            for i in range(len(ts) - 1):
                x[:, :, :motion.shape[2]] = motion
                # the head reads its batch from t_mod's rows, so the shared
                # timestep is widened to one row per session
                flow = pl.model(x=x, timestep=ts[i].expand(b), context=ctx, y=ref)
                flow = flow[0] if isinstance(flow, (tuple, list)) else flow
                t_i = (ts[i][:, None, None, None] / n_ts).to(x.dtype)
                t_i1 = (ts[i + 1][:, None, None, None] / n_ts).to(x.dtype)
                x0 = x - flow * t_i
                x = (1 - t_i1) * x0 + t_i1 * torch.stack(
                    [torch.randn(shape, dtype=pl.param_dtype, device=pl.device,
                                 generator=s["generator"]) for s in sessions])
            x[:, :, :motion.shape[2]] = motion
            t = mark("denoise", t)

            # LtxVAE.decode adds a batch dim of its own; feed it one instead
            videos = pl.vae.model.decode(pl.vae.un_normalize_latents(x),
                                         return_dict=False, target_shape=x.shape)[0]
            t = mark("decode", t)
            if pl.color_correction_strength > 0.0:
                # already batched upstream (B, C, T, H, W) against (B, C, 1, H, W)
                refimg = torch.stack([pl.cond_image_tensor_dict[s["avatar"]][0] for s in sessions])
                videos = match_and_blend_colors_torch(videos, refimg, pl.color_correction_strength)
            t = mark("color", t)
            cond = videos[:, :, -pl.motion_frames_num:].to(pl.device)
            new_motion = pl.vae.normalize_latents(
                pl.vae.model.encode(cond, return_dict=False)[0].sample())
            t = mark("motion", t)

        # run_pipeline's tail, with the batch kept: (B,C,F,H,W) -> (B,F,H,W,C)
        frames = (((videos.to(torch.float32) + 1) / 2).permute(0, 2, 3, 4, 1).clip(0, 1) * 255)
        frames = frames[:, pl.motion_frames_num:].contiguous().to(torch.uint8).cpu().numpy()
        mark("transfer", t)
        if STAGE_TIMING:
            self.stage_last = stage
        return frames, new_motion

    def _run_batch(self, sessions: list, embs: list) -> dict:
        """One guarded GPU call for every session speaking in this slot."""
        t0 = time.time()
        with self.gpu_lock:
            frames, motion = self._generate_batch(sessions, embs)
        took = time.time() - t0
        for i, sess in enumerate(sessions):
            sess["motion"] = motion[i]
        # the slot's cost as a function of how many spoke; the budget is fitted
        # from these, because with batching it is no longer B x (one chunk)
        self.batch_samples.append((len(sessions), took))
        per = took / len(sessions)
        self.gen_ema = 0.9 * self.gen_ema + 0.1 * per if self.gen_ema else per
        return {s["id"]: {"frames": frames[i], "gen": took if i == 0 else 0.0}
                for i, s in enumerate(sessions)}

    def _slot_capacity(self, slot_s: float) -> float:
        """Chunks that fit in a slot, unrounded.

        `_batch_budget` floors this, and on the 4090 the floor throws away 0.89
        of the 3.89 chunks a slot can hold — 23% of the card. Lockstep could not
        have used the remainder anyway (a speaker needs a whole slot every slot),
        but with a lookahead the fraction can be carried and banked, which is why
        the two changes are worth far more together than apart.
        """
        return max(1.0, slot_s / self.gen_ema) if self.gen_ema else 1.0

    def _batch_budget(self, slot_s: float) -> int:
        """How many sessions may speak in one slot, fitted from what batches cost.

        Without batching this was floor(slot / chunk). With it the slot cost is
        roughly fixed + marginal x B — on an H200 about 0.11 s + 0.06 s per
        session — so the budget is the largest B that still fits. Until there are
        samples at two different sizes there is nothing to fit, and the old
        per-chunk rule stands.
        """
        pts = list(self.batch_samples)
        sizes = {b for b, _ in pts}
        if len(pts) >= 8 and len(sizes) >= 2:
            n = len(pts)
            sx = sum(b for b, _ in pts); sy = sum(t for _, t in pts)
            sxx = sum(b * b for b, _ in pts); sxy = sum(b * t for b, t in pts)
            den = n * sxx - sx * sx
            if den > 0:
                slope = (n * sxy - sx * sy) / den
                fixed = (sy - slope * sx) / n
                if slope > 1e-4:
                    return max(1, min(LIVE_MAX_SESSIONS, int((slot_s - max(0.0, fixed)) / slope)))
        return max(1, int(slot_s / self.gen_ema)) if self.gen_ema else 1

    def _chunk(self, sess: dict, audio_window, start_idx: int, end_idx: int):
        """`_run_chunk` for the streaming path: build the rolling-window embedding first."""
        import numpy as np

        from flash_head.inference import get_audio_embedding

        emb = get_audio_embedding(self.pipeline, np.array(audio_window, dtype=np.float32), start_idx, end_idx)
        return self._run_chunk(sess, emb)

    def _session(self, image_bytes: bytes, seed: int, use_face_crop: bool, session_id: str) -> dict:
        """Register the avatar and open a per-call pipeline-state slot."""
        import torch

        sess = {"avatar": self._ensure_avatar(image_bytes, use_face_crop), "motion": None,
                "generator": torch.Generator(device=self.pipeline.device).manual_seed(seed)}
        self.sessions[session_id] = sess
        return sess

    def ping(self) -> dict:
        """Cheap liveness check for a *warm* container. Calling this on a cold
        class spins one up, so the studio never uses it for status polling."""
        return {"warm": True, "model_type": self.model_type, "load_seconds": round(self.load_seconds, 1), "resident_for": round(time.time() - self.loaded_at)}

    def render(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        job_id: str,
        seed: int = 42,
        use_face_crop: bool = True,
        tail_seconds: float = 0.5,
    ) -> bytes:
        """One clip: reference image + 16 kHz mono WAV -> mp4 bytes.

        Mirrors the `once` branch of upstream generate_video.py::generate —
        encode the whole audio in one wav2vec2 pass, slice per chunk — against
        the already-loaded pipeline. Only the reference image and seed are
        re-prepared per call.

        `tail_seconds` is the minimum silence the video keeps after the speech
        ends. Upstream's `once` padding only completes the last chunk, and its
        chunk count is one short of what the audio allows, so the video always
        came out 24 frames (0.96 s) shorter than the padded audio — depending
        on where the speech fell on the 0.96 s chunk grid, up to that much of
        the last word was cut at the mux. Here the audio is padded with silence
        until the rendered frames cover speech + tail (the grid makes the
        actual tail land in [tail, tail + 0.96 s)).
        """
        import math
        import os
        import subprocess
        import tempfile
        import threading
        import wave

        import imageio
        import librosa
        import numpy as np
        import torch

        from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline

        t0 = time.time()
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
        img = work / "ref.png"
        wav = work / "speech.wav"
        out = work / "video.mp4"
        img.write_bytes(image_bytes)
        wav.write_bytes(audio_bytes)

        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(10):
                _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] .. running, vram {torch.cuda.memory_allocated()/1e9:.1f} GB")

        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] preparing reference (face crop, seed {seed})")
            sess = self._session(image_bytes, seed, use_face_crop, job_id)
            _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] reference ready")
            p = get_infer_params()
            sr, fps = p["sample_rate"], p["tgt_fps"]
            frame_num, motion_frames_num = p["frame_num"], p["motion_frames_num"]
            slice_len = frame_num - motion_frames_num

            speech, _ = librosa.load(str(wav), sr=sr, mono=True)
            speech_s = len(speech) / sr
            _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] audio loaded ({speech_s:.2f}s)")
            # k chunks render frame_num + (k-1)*slice_len frames and, with the
            # upstream chunk-count rule kept below, need frame_num + k*slice_len
            # frames of audio (each rendered frame sees >= one slice of
            # look-ahead, as upstream). Pick k so the video covers speech + tail.
            need_frames = math.ceil(speech_s * fps) + math.ceil(tail_seconds * fps)
            k = max(1, math.ceil((need_frames - motion_frames_num) / slice_len))
            audio_frames = frame_num + k * slice_len
            target = audio_frames * sr // fps
            if len(speech) < target:
                speech = np.concatenate([speech, np.zeros(target - len(speech), dtype=speech.dtype)])
            video_s = (motion_frames_num + k * slice_len) / fps
            _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] speech {speech_s:.2f}s → video {video_s:.2f}s ({k} chunks, tail ≥{tail_seconds:.1f}s)")
            # Mux the *padded* audio: it is longer than the video, so `-shortest`
            # trims silence off the audio instead of trimming frames off the video.
            padded_wav = work / "speech_padded.wav"
            with wave.open(str(padded_wav), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

            _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] encoding audio ({len(speech)/sr:.1f}s) in one pass")
            emb = get_audio_embedding(self.pipeline, speech)
            n_chunks = (emb.shape[1] - frame_num) // slice_len
            chunks = [emb[:, i * slice_len : i * slice_len + frame_num].contiguous() for i in range(n_chunks)]
            _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] {n_chunks} chunks to generate")

            frames = []
            for i, chunk in enumerate(chunks):
                block, gen_s, _ = self._run_chunk(sess, chunk, drop_motion=i != 0)
                frames.append(torch.from_numpy(block))
                _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] chunk {i+1}/{n_chunks} done ({gen_s:.2f}s)")

            _emit(self.host, job_id, f"[{time.time()-t0:5.1f}s] encoding mp4")
            tmp = work / "video_noaudio.mp4"
            with imageio.get_writer(str(tmp), format="mp4", mode="I", fps=fps, codec="h264", ffmpeg_params=["-bf", "0"]) as w:
                for block in frames:
                    arr = block.numpy().astype(np.uint8)
                    for k in range(arr.shape[0]):
                        w.append_data(arr[k])
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(padded_wav),
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
                check=True,
            )
            total = time.time() - t0
            _emit(self.host, job_id, f"[{total:5.1f}s] {DONE_MARK} {n_chunks} chunks, {out.stat().st_size/1e6:.1f} MB")
            return out.read_bytes()
        finally:
            stop.set()
            self.sessions.pop(job_id, None)
            try:
                for f in work.iterdir():
                    f.unlink()
                work.rmdir()
            except OSError:
                pass


    def live(
        self,
        session_id: str,
        image_bytes: bytes,
        seed: int = 42,
        use_face_crop: bool = True,
        idle_timeout_s: float = 180.0,
        max_seconds: float = 1800.0,
        lead_chunks: float = 1.0,
        use_idle_loop: bool = True,
        fade_frames: int = 4,
    ) -> dict:
        """Join a live session. The container's single scheduler drives it.

        Sessions do not pace themselves. An earlier version gave each session its
        own clock and had them compete for the GPU lock, which behaves exactly as
        uncoordinated greedy actors do: a session that fell behind generated flat
        out, starved its peers, and pushed them behind too. Here one scheduler
        thread produces exactly one chunk per session per 0.96 s slot and decides
        who gets the GPU, so no session can be delayed by another's backlog.

        While an avatar listens it plays a pre-rendered idle loop, which costs
        nothing; only speech needs the GPU. The GPU fits one generation per slot
        on an L4 (0.82 s of 0.96 s), so when several avatars want to talk at once
        the others' *speech* waits a slot or two while their *video* keeps running
        at realtime. That is the capacity limit made explicit rather than emergent.
        """
        sess = self._open_live(session_id, image_bytes, seed, use_face_crop, use_idle_loop,
                               fade_frames, idle_timeout_s, max_seconds)
        try:
            threading.Thread(target=self._ingest, args=(sess,), daemon=True,
                             name=f"ingest-{session_id}").start()
            self._ensure_scheduler()
            sess["done"].wait(timeout=max_seconds + 300)
            return self._close_live(sess)
        finally:
            self.live_sessions.pop(session_id, None)
            self.sessions.pop(session_id, None)

    # ---- live session plumbing (all driven by _scheduler_loop) ----------------
    def _idle_loop_for(self, image_bytes: bytes, seed: int, sess: dict, p: dict) -> "object":
        """The idle loop for this avatar: rendered once, then shared by every
        session using it. Concurrent starts would otherwise each render the same
        9 s of video — three at once cost 24 s of GPU and put all three behind."""
        import imageio
        import numpy as np

        name = sess["avatar"]
        cached_loop = self.idle_loops.get(name)
        if cached_loop is not None:
            return cached_loop
        with self.loop_locks_guard:
            lock = self.loop_locks.setdefault(name, threading.Lock())
        with lock:
            if name in self.idle_loops:
                return self.idle_loops[name]
            path = _idle_loop_path(image_bytes, seed, self.model_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                t0 = time.time()
                sr, fps = p["sample_rate"], p["tgt_fps"]
                slice_len = p["frame_num"] - p["motion_frames_num"]
                dq = deque([0.0] * (p["cached_audio_duration"] * sr), maxlen=p["cached_audio_duration"] * sr)
                end_idx = p["cached_audio_duration"] * fps
                made = []
                for _ in range(IDLE_CHUNKS):
                    dq.extend(np.zeros(slice_len * sr // fps, dtype=np.float32).tolist())
                    made.append(self._chunk(sess, dq, end_idx - p["frame_num"], end_idx)[0])
                sess["motion"] = None          # the loop is a shared asset, not this session's history
                tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mp4")
                with imageio.get_writer(str(tmp), format="mp4", mode="I", fps=fps, codec="h264",
                                        ffmpeg_params=["-bf", "0", "-crf", "20", "-preset", "veryfast",
                                                       "-pix_fmt", "yuv420p"]) as w:
                    for block in made:
                        for f in block:
                            w.append_data(f)
                tmp.replace(path)
                self.host.commit()
                print(f"idle loop rendered in {time.time()-t0:.0f}s -> {path.name}", flush=True)
            reader = imageio.get_reader(str(path))
            loop = np.stack([np.asarray(f) for f in reader])
            reader.close()
            self.idle_loops[name] = loop
            return loop

    def _open_live(self, session_id: str, image_bytes: bytes, seed: int, use_face_crop: bool,
                   use_idle_loop: bool, fade_frames: int, idle_timeout_s: float, max_seconds: float) -> dict:
        import queue as _q
        import tempfile

        import numpy as np

        from flash_head.inference import get_infer_params

        t0 = time.time()
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"live-{session_id}-"))
        hls = work / "hls"
        hls.mkdir()
        (work / "ref.png").write_bytes(image_bytes)
        ship = _HlsShipper(self.host, session_id, hls, t0)
        ship.log(f"live: preparing reference (face crop, seed {seed})")

        sess = self._session(image_bytes, seed, use_face_crop, session_id)
        p = get_infer_params()
        sr, fps = p["sample_rate"], p["tgt_fps"]
        frame_num, motion = p["frame_num"], p["motion_frames_num"]
        slice_len = frame_num - motion
        cached = p["cached_audio_duration"] * sr
        sess.update({
            "id": session_id, "ship": ship, "work": work, "hls": hls, "t0": t0,
            "sr": sr, "fps": fps, "motion": sess.get("motion"), "motion_n": motion,
            "slice_len": slice_len, "chunk_samples": slice_len * sr // fps, "chunk_s": slice_len / fps,
            "audio_end_idx": p["cached_audio_duration"] * fps,
            "audio_start_idx": p["cached_audio_duration"] * fps - frame_num,
            "audio_dq": _AudioWindow(cached),
            # generated but not yet shown: (frames, pcm) in playback order
            "ready": deque(),
            "playing": False,        # emission-side counterpart of "speaking"
            "pending": deque(), "buf": np.zeros(0, dtype=np.float32),
            "fade_frames": fade_frames, "idle_timeout_s": idle_timeout_s, "max_seconds": max_seconds,
            "speaking": False, "loop_pos": 0, "last_frame": None, "want_since": None,
            "ended": False, "open": True, "done": threading.Event(), "reason": "",
            "last_speech": time.time(), "k": 0,
            # the overrun counter is container-wide, so a session reports the
            # slots that overran while it was open, not the container's lifetime total
            "overruns_at_open": self.slot_overruns,
            "stats": {"chunks": 0, "speech_chunks": 0, "gpu_chunks": 0, "deferred_chunks": 0,
                      "gen_s": [], "behind_s": [], "wait_to_speak_s": [],
                      "wait_at_s": []},
        })
        sess["idle_loop"] = self._idle_loop_for(image_bytes, seed, sess, p) if use_idle_loop else None

        fifo = str(work / "audio.pipe")
        os.mkfifo(fifo)
        proc = _live_ffmpeg(hls, p["width"], p["height"], fps, sr, fifo, slice_len)
        vq: _q.Queue = _q.Queue()
        aq: _q.Queue = _q.Queue()

        def vwriter() -> None:
            # `tobytes()` copied the whole 18.9 MB chunk again just to hand it to
            # a pipe. Writing the array's own buffer skips that; the copy held the
            # GIL, so it was stealing from the scheduler thread as well as costing
            # bandwidth. Idle frames arrive as views of the shared loop, which are
            # contiguous, and the few that are not get made so here.
            try:
                while (arr := vq.get()) is not None:
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    proc.stdin.write(memoryview(arr).cast("B"))
            finally:
                proc.stdin.close()

        def awriter() -> None:
            with open(fifo, "wb") as f:               # blocks until ffmpeg opens the reader
                while (pcm := aq.get()) is not None:
                    f.write(pcm)

        threading.Thread(target=vwriter, daemon=True).start()
        threading.Thread(target=awriter, daemon=True).start()
        sess.update({"proc": proc, "vq": vq, "aq": aq})
        if sess["idle_loop"] is not None:
            ship.log(f"live: idle loop ready ({len(sess['idle_loop'])} frames) — GPU rests while the avatar listens")
        ship.log(f"live: ready — {sess['chunk_s']:.2f}s chunks, idle timeout {idle_timeout_s:.0f}s")
        self.live_sessions[session_id] = sess
        return sess

    def _ingest(self, sess: dict) -> None:
        """Audio arrives on the session's Queue partition. Network I/O, so it runs
        off the scheduler: the scheduler must never block on a remote read."""
        import numpy as np

        n = sess["chunk_samples"]
        while sess["open"]:
            try:
                items = self.host.get_many(50, partition=f"{sess['id']}:audio", block=True, timeout=1)
            except Exception:  # noqa: BLE001  (queue.Empty and transient errors alike)
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "end":
                    sess["ended"] = True
                elif item.get("type") == "audio":
                    raws = item.get("chunks") or [item.get("pcm", b"")]
                    arr = np.concatenate([np.frombuffer(r, dtype=np.int16).astype(np.float32) / 32768.0
                                          for r in raws]) if raws else np.zeros(0, np.float32)
                    buf = np.concatenate([sess["buf"], arr])
                    while len(buf) >= n:
                        sess["pending"].append(buf[:n]); buf = buf[n:]
                    if item.get("final", True) and len(buf):
                        sess["pending"].append(np.concatenate([buf, np.zeros(n - len(buf), dtype=np.float32)]))
                        buf = buf[:0]
                    sess["buf"] = buf

    def _ensure_scheduler(self) -> None:
        with self.sched_guard:
            if self.sched_thread is None or not self.sched_thread.is_alive():
                self.sched_thread = threading.Thread(target=self._scheduler_loop, daemon=True,
                                                     name="live-scheduler")
                self.sched_thread.start()

    def _scheduler_loop(self) -> None:
        """One chunk per session per slot, forever. The only place GPU work is
        scheduled, so the budget below is the whole capacity story."""
        import numpy as np

        while True:
            live = [s for s in list(self.live_sessions.values()) if s["open"]]
            if not live:
                self.slot_t0 = None
                time.sleep(0.05)
                continue
            chunk_s = live[0]["chunk_s"]
            if self.slot_t0 is None:
                self.slot_t0, self.slot = time.time(), 0
            target = self.slot_t0 + self.slot * chunk_s
            now = time.time()
            if now < target:
                time.sleep(min(target - now, chunk_s))
                continue
            behind = now - target
            self.slot += 1

            # How many generations fit in a slot, from the generation time we
            # measure. Before any measurement exists, admit one: the old fallback
            # divided by 0.05 s and so advertised a budget of 19, which admitted
            # every waiting session into a single slot and then preempted all but
            # the first once the real 0.8 s landed a slot later.
            budget = self._batch_budget(chunk_s)
            if LOOKAHEAD:
                # Spend chunk generations, not sessions, and carry what a slot
                # could not use into the next one. Alone this buys nothing — a
                # lockstep speaker needs a whole slot every slot and has no use
                # for a spare fifth of one — but with a lead it can be banked.
                # The credit is deducted whether or not the quota gets spent,
                # so what survives a slot is only the sub-1 fraction the integer
                # floor would have discarded — never idle time. That is the right
                # bound and it needs no cap: silicon-seconds nobody used are
                # gone, and a credit that grew while the room was quiet would be
                # promising work the next slot cannot physically do.
                self.gen_credit = getattr(self, "gen_credit", 0.0) + self._slot_capacity(chunk_s)
                quota = int(self.gen_credit)
                self.gen_credit -= quota
            else:
                quota = budget
            speakers = [s for s in live if s["pending"]]
            # want_since marks the start of a TURN, not of a chunk. A session that is
            # already speaking has had its want_since cleared, and must not set it
            # again mid-utterance: every later chunk of that utterance would then
            # record a zero wait and dilute freeze-out — the fraction of turns that
            # could not start at once — until it means nothing. The transition being
            # timed is silence -> speech, which happens once per turn.
            for s in speakers:
                if s["want_since"] is None and not s["speaking"]:
                    s["want_since"] = now
            # A session mid-utterance is never cut: once speech is being played
            # out there is no slack in its audio timeline, so a missed slot shows
            # up as a second of silence in the middle of a sentence. Slicing the
            # combined list would have done exactly that whenever the budget
            # shrank below the number of speakers already talking (possible on any
            # GPU fast enough for a budget above 1). Overrunning the slot instead
            # is recoverable — idle chunks are free, so the lag is worked off.
            if LOOKAHEAD:
                # Who needs the GPU *this slot* — not who happens to be speaking.
                # A session that is already `LOOKAHEAD` chunks ahead asks for
                # nothing and its budget slot goes to someone who wants to start.
                # That is the whole change: an utterance no longer reserves a
                # slot for its full seven seconds, it just has to be kept fed.
                #
                # But a sort is only a preference, not a guarantee — and the
                # comment above this block promises a guarantee. If more sessions
                # are simultaneously `speaking` with `ready == 0` than the quota
                # covers, sorting them to the front still truncates the excess at
                # `[:quota]`. Those tied-for-first sessions ARE mid-utterance and
                # about to go silent, which is exactly the failure this design is
                # supposed to rule out. So they are force-admitted, the same way
                # lockstep force-admitted every `ongoing` session — the slot may
                # run over its nominal quota, which is recoverable; a gap is not.
                must = {s["id"] for s in speakers if s["speaking"] and not s["ready"]}
                rest_quota = max(0, quota - len(must))
                cands = [s for s in speakers
                         if s["id"] not in must and len(s["ready"]) < LOOKAHEAD]
                # Order by how close each one is to running dry, among the
                # discretionary rest: a session with some lead left but low can
                # go next slot without a gap; a fresh start can wait a slot longer.
                cands.sort(key=lambda s: (len(s["ready"]),
                                          0 if s["speaking"] else 1,
                                          s["want_since"] or 0.0))
                chosen = must | {s["id"] for s in cands[:rest_quota]}
                ongoing = [s for s in speakers if s["speaking"]]
                if len(must) > quota:
                    self.slot_overruns += 1
            else:
                ongoing = [s for s in speakers if s["speaking"]]
                waiting = sorted((s for s in speakers if not s["speaking"]),
                                 key=lambda s: s["want_since"])
                # compare by id: a session dict holds tensors, so `in` would try to
                # compare those element-wise and raise
                chosen = {s["id"] for s in ongoing}
                chosen |= {s["id"] for s in waiting[:max(0, budget - len(ongoing))]}
            chosen |= {s["id"] for s in live if s["idle_loop"] is None}   # no loop to fall back on
            if not LOOKAHEAD and len(ongoing) > budget:
                self.slot_overruns += 1
            self.slot_budget = budget
            # A session once sat deferred for 77 s while only one other spoke and
            # two budget slots looked free. Nothing in the log could say what the
            # budget actually was at that moment, so it gets written down whenever
            # it moves — the number that decides who speaks must be observable.
            if budget != getattr(self, "_budget_last", None):
                print(f"budget {getattr(self, '_budget_last', '-')} -> {budget}  "
                      f"(gen_ema {self.gen_ema:.3f}s, samples {len(self.batch_samples)}, "
                      f"sizes {sorted({b for b, _ in self.batch_samples})}, "
                      f"ongoing {len(ongoing)}, chosen {len(chosen)})", flush=True)
                self._budget_last = budget
            n_spk = sum(1 for s in speakers if s["id"] in chosen)   # generating, not playing
            self.slot_hist[min(n_spk, LIVE_MAX_SESSIONS)] += 1

            # Three phases, so the GPU sees one call per slot instead of one per
            # speaker. Phase 1 takes every session to the point where only the
            # generate is left, phase 2 runs the speakers together, phase 3 emits.
            def failed(s, exc) -> None:
                s["reason"] = f"error: {type(exc).__name__}: {exc}"[:200]
                s["open"] = False
                s["done"].set()

            prep: dict = {}
            for s in live:
                try:
                    p = self._slot_prep(s, s["id"] in chosen, now)
                    if p is not None:
                        prep[s["id"]] = p
                except Exception as exc:  # noqa: BLE001 — one bad session must not stop the rest
                    failed(s, exc)

            talking = [s for s in live if prep.get(s["id"], {}).get("emb") is not None]
            if talking:
                try:
                    # one wav2vec forward for the whole slot, then one generate
                    embs = self._audio_embed_batch([prep[s["id"]]["emb"] for s in talking],
                                                   talking[0]["audio_start_idx"],
                                                   talking[0]["audio_end_idx"])
                    out = self._run_batch(talking, embs)
                    for s in talking:
                        r = out[s["id"]]
                        prep[s["id"]].update(r)
                        self._bank(s, r, prep[s["id"]]["pcm"])
                except Exception as exc:  # noqa: BLE001
                    for s in talking:
                        prep.pop(s["id"], None)
                        failed(s, exc)
                quota -= len(talking)

            # Extra rounds: spend what is left of the slot's quota building a
            # lead for whoever is closest to running dry. This is what makes the
            # lookahead real — without it a session can only ever keep pace, and
            # `ready` never rises above the one chunk it just made.
            t_slot = time.time()
            while LOOKAHEAD and quota > 0 and time.time() - t_slot < chunk_s * 0.7:
                c = [s for s in live
                     if s["open"] and s["pending"] and len(s["ready"]) < LOOKAHEAD]
                if not c:
                    break
                c.sort(key=lambda s: (len(s["ready"]),
                                      0 if s["speaking"] else 1,
                                      s["want_since"] or 0.0))
                take = c[:quota]
                try:
                    gs = [self._gen_one(s, now) for s in take]
                    embs = self._audio_embed_batch([g["emb"] for g in gs],
                                                   take[0]["audio_start_idx"],
                                                   take[0]["audio_end_idx"])
                    out = self._run_batch(take, embs)
                    for s, g in zip(take, gs):
                        self._bank(s, out[s["id"]], g["pcm"])
                except Exception as exc:  # noqa: BLE001
                    for s in take:
                        failed(s, exc)
                    break
                quota -= len(take)

            for s in live:
                p = prep.get(s["id"])
                if p is None:
                    continue
                try:
                    self._emit(s, p, behind, now)
                except Exception as exc:  # noqa: BLE001
                    failed(s, exc)

            if RESIDENCY_TIMING:
                self.res_slots += 1
                if self.res_slots % 100 == 0 and self.res_n:
                    # Diagnostics must never take down what they measure. A timer
                    # variable once collided with a tensor named `t` and the
                    # formatting raised, which killed this thread and with it every
                    # session — the run looked like a hang, not a bug in a print.
                    try:
                        per = {k: float(v) / self.res_n for k, v in sorted(self.res_t.items())}
                        tot = sum(per.values())
                        print(f"residency over {self.res_slots} slots, {len(live)} live: "
                              f"{tot*1000:.2f} ms per session-slot  "
                              + "  ".join(f"{k.split(':')[-1]} {v*1000:.2f}" for k, v in per.items()),
                              flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"residency report failed: {type(exc).__name__}: {exc}", flush=True)

    def _res(self, name: str, t0: float) -> float:
        """Charge the time since `t0` to a residency section. Returns the new mark."""
        if not RESIDENCY_TIMING:
            return 0.0
        now = time.perf_counter()
        self.res_t[name] = self.res_t.get(name, 0.0) + now - t0
        return now

    def _bank(self, sess: dict, r: dict, pcm) -> None:
        """Queue a generated chunk behind whatever lead the session already has.

        One place, because round one and the extra rounds must append in the same
        order they were generated — the motion latents chain, so a chunk shown out
        of order would show a face continuing from a frame it never saw.
        """
        if r.get("frames") is None:
            return
        sess["ready"].append((r["frames"], pcm))
        st = sess["stats"]
        st["gen_s"].append(round(r.get("gen", 0.0), 3))
        st["gpu_chunks"] += 1

    def _gen_one(self, sess: dict, now: float) -> dict:
        """Take one pending chunk and build what `generate` needs for it.

        Generation only — no emission, no bookkeeping that belongs once per slot.
        A slot may call this several times for the same session to bank a lead;
        consecutive chunks are sequentially dependent through the motion latents,
        so they are separate calls, but they cost the same as anyone else's.
        """
        import numpy as np  # noqa: F401  (kept for symmetry with _slot_prep)
        import torch

        pcm = sess["pending"].popleft()
        sess["last_speech"] = now
        sess["audio_dq"].push(pcm)
        if sess["idle_loop"] is not None and not sess["speaking"]:
            # entering generation: continue from the frames the viewer is seeing
            seen = _loop_window(sess["idle_loop"],
                                (sess["loop_pos"] - sess["motion_n"]) % len(sess["idle_loop"]),
                                sess["motion_n"])
            with self.gpu_lock:
                t = torch.from_numpy(np.ascontiguousarray(seen)).to(self.pipeline.device,
                                                                    dtype=self.pipeline.param_dtype)
                sess["motion"] = self.pipeline.vae.encode(((t / 255 - 0.5) * 2).permute(3, 0, 1, 2).unsqueeze(0))
            sess["speaking"] = True
        return {"pcm": pcm, "emb": sess["audio_dq"].view()}

    def _slot_prep(self, sess: dict, may_speak: bool, now: float):
        """Take one session as far as it can go without the GPU.

        Ends the session if its time is up, decides whether it speaks this slot,
        advances its audio window, and — for a session that is speaking — builds
        the embedding the batched generate will consume. Returns None once the
        session is closed, otherwise what `_emit` needs to finish the slot.

        This used to be the front half of `_produce`. It is separate so that
        every speaker in a slot is ready before any of them touches the GPU,
        which is what lets them share one call.
        """
        import numpy as np

        from flash_head.inference import get_audio_embedding

        st = sess["stats"]
        n = sess["chunk_samples"]
        if sess["ended"] and not sess["pending"] and not sess["ready"]:
            sess["reason"] = "stop"; sess["open"] = False; sess["done"].set(); return None
        if not sess["pending"] and not sess["ready"] and now - sess["last_speech"] > sess["idle_timeout_s"]:
            sess["reason"] = "idle"; sess["open"] = False; sess["done"].set(); return None
        if now - sess["t0"] > sess["max_seconds"]:
            sess["reason"] = "max_seconds"; sess["open"] = False; sess["done"].set(); return None

        _rt = time.perf_counter() if RESIDENCY_TIMING else 0.0
        gen_speech = bool(sess["pending"]) and may_speak
        if gen_speech:
            g = self._gen_one(sess, now)
            pcm, emb = g["pcm"], g["emb"]
        else:
            pcm, emb = np.zeros(n, dtype=np.float32), None
            if sess["pending"] and not sess["ready"]:
                # wanted to speak, has nothing made, did not get the GPU
                st["deferred_chunks"] += 1
            # The window follows the GENERATION head, not the playhead: it holds
            # the eight seconds around the chunk being made, so it advances once
            # per generated chunk, in order. A session running ahead — frames
            # ready, not generating this slot — has not moved its head, and
            # pushing silence here would punch a hole into the middle of the
            # utterance it is in the middle of making.
            if not sess["ready"]:
                sess["audio_dq"].push(pcm)
            if sess["idle_loop"] is None:          # nothing to fall back on yet
                emb = sess["audio_dq"].view()
        _rt = self._res("prep:decide", _rt)
        self.res_n += 1 if RESIDENCY_TIMING else 0
        return {"pcm": pcm, "gen_speech": gen_speech, "emb": emb, "frames": None, "gen": 0.0}

    def _emit(self, sess: dict, p: dict, behind: float, now: float) -> None:
        """Finish one session's slot: pick the frames, queue them, record the chunk."""
        import numpy as np

        st = sess["stats"]
        _rt = time.perf_counter() if RESIDENCY_TIMING else 0.0
        gen = p["gen"]
        # Exactly one chunk leaves per slot whatever generation did. That is what
        # keeps the viewer's buffer — and so the latency from "decided to speak"
        # to "the candidate sees it" — the same as it was before the lookahead.
        is_speech = bool(sess["ready"])
        if is_speech:
            frames, pcm = sess["ready"].popleft()
            if sess["want_since"] is not None:
                # the wait ends when the mouth moves, not when the GPU ran
                st["wait_to_speak_s"].append(round(now - sess["want_since"], 2))
                # when this turn began, relative to the session's own start. A run
                # starts with an empty GPU, so its first turns wait for nobody and
                # report a wait no steady state would give — the transient flatters
                # the numbers. Keeping the time lets the statistics drop it.
                st["wait_at_s"].append(round(sess["want_since"] - sess["t0"], 1))
                sess["want_since"] = None
            sess["playing"] = True
        else:                                    # nothing ready: play the idle loop
            pcm = p["pcm"]
            loop = sess["idle_loop"]
            if loop is None:
                # Before the first idle loop exists a session has nothing to fall
                # back on. It is force-admitted for exactly that reason, so this
                # only happens if its generation failed; emitting nothing is the
                # honest response, and the next slot tries again.
                return
            frames = _loop_window(loop, sess["loop_pos"], sess["slice_len"])
            sess["loop_pos"] = (sess["loop_pos"] + sess["slice_len"]) % len(loop)
            if sess["playing"]:                  # returning from a turn: fade into the loop
                f = sess["fade_frames"]
                frames = frames.copy()
                w = np.linspace(0, 1, f + 2)[1:-1].reshape(-1, 1, 1, 1)
                frames[:f] = (sess["last_frame"].astype(np.float32) * (1 - w)
                              + frames[:f].astype(np.float32) * w).astype(np.uint8)
                sess["playing"] = False
            if not sess["pending"]:
                sess["speaking"] = False         # generation head has left the utterance

        _rt = self._res("emit:frames", _rt)
        sess["vq"].put(frames)
        sess["last_frame"] = frames[-1]
        sess["aq"].put((np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        _rt = self._res("emit:queues", _rt)
        sess["k"] += 1
        st["chunks"] = sess["k"]
        st["speech_chunks"] += int(is_speech)
        st["lead"] = max(st.get("lead", 0), len(sess["ready"]))
        st["behind_s"].append(round(behind, 2))
        _rt = self._res("emit:stats", _rt)
        sess["ship"].ship_new_files()
        _rt = self._res("emit:ship_files", _rt)
        kind = "speech" if is_speech else ("deferred" if sess["pending"] else "silence")
        # every speech and every deferred chunk is logged: the studio reconstructs
        # the Gantt's spans from these lines, so a skipped one is a hole in the chart
        if kind != "silence" or sess["k"] % 20 == 0:
            sess["ship"].log(f"live: chunk {sess['k']} {kind} "
                             f"gen {gen:.2f}s behind {behind:.2f}s "
                             f"lead {len(sess['ready'])} "
                             f"queued {len(sess['pending'])*sess['chunk_s']:.1f}s")
        self._res("emit:log", _rt)

    def _close_live(self, sess: dict) -> dict:
        sess["open"] = False
        sess["vq"].put(None)
        sess["aq"].put(None)
        rc = sess["proc"].wait(timeout=120)
        ship = sess["ship"]
        ship.ship_new_files()
        st = sess["stats"]
        k = max(1, st["chunks"])
        saved = 100 * (1 - st["gpu_chunks"] / k)
        ship.log(f"{DONE_MARK} live ended ({sess['reason']}): {st['chunks']} chunks, {st['speech_chunks']} speech, "
                 f"GPU {st['gpu_chunks']} chunks ({saved:.0f}% saved), {ship.streamed/1e6:.1f} MB")
        ship.close()
        gens = sorted(st["gen_s"])
        # Steady state only. WARMUP_S of turns are dropped because at t=0 nothing is
        # speaking: the first turns of a run find every budget slot free whatever
        # the session count is, which is an artefact of starting a benchmark, not
        # a property of the load. In production interviews begin at random times
        # and the system is never observed from empty.
        at = st.get("wait_at_s") or []
        waits = [w for w, t in zip(st["wait_to_speak_s"], at) if t >= WARMUP_S] \
            if len(at) == len(st["wait_to_speak_s"]) else st["wait_to_speak_s"]
        if not waits:                              # a run shorter than the warm-up
            waits = st["wait_to_speak_s"]
        tail = st["behind_s"][-20:]
        # freeze-out: the share of turns the avatar could not begin at once because
        # no budget was free. TASI's name and TASI's design target (0.5%). A turn is
        # counted as frozen once it waits more than a hundredth of a slot, which is
        # below anything a viewer could see and above scheduler jitter.
        turns = len(waits)
        frozen = [w for w in waits if w > 0.01]
        out = {"chunks": st["chunks"], "speech_chunks": st["speech_chunks"], "gpu_chunks": st["gpu_chunks"],
               "gpu_chunk_ratio": round(st["gpu_chunks"] / k, 3), "deferred_chunks": st["deferred_chunks"],
               "gen_median_s": gens[len(gens) // 2] if gens else None,
               "gen_swap_s": round(self.t_swap, 3), "gen_pipe_s": round(self.t_pipe, 3),
               "gen_xfer_s": round(self.t_xfer, 3),
               "behind_max_s": max(st["behind_s"]) if st["behind_s"] else None,
               "behind_tail_s": max(tail) if tail else None,
               "wait_to_speak_max_s": max(waits) if waits else 0.0,
               "wait_to_speak_median_s": _pct(waits, 50),
               # Exact GPU seconds, not a count times an average. `_run_batch`
               # gives the whole slot's time to the first session of the batch
               # and 0.0 to the rest, so summing these over every session counts
               # each slot once — which is what occupancy needs.
               "gpu_seconds": round(sum(st["gen_s"]), 2),
               "lead_max": st.get("lead", 0),
               "turns": turns, "turns_frozen": len(frozen),
               "freeze_out": round(len(frozen) / turns, 4) if turns else 0.0,
               "wait_p50_s": _pct(waits, 50), "wait_p95_s": _pct(waits, 95),
               # the raw waits, so a run's percentile can be taken over every
               # turn it produced rather than over one session's dozen
               "waits_s": [round(w, 2) for w in waits],
               "wait_frozen_mean_s": round(sum(frozen) / len(frozen), 2) if frozen else 0.0,
               "ended_by": sess["reason"], "ffmpeg_rc": rc, "slot_overruns": self.slot_overruns - sess["overruns_at_open"],
               "slot_budget": self.slot_budget, "slot_speakers": list(self.slot_hist),
               "peer_sessions": len(self.live_sessions) - 1,
               "seconds": round(time.time() - sess["t0"], 1)}
        try:
            for f in sorted(sess["work"].rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            sess["work"].rmdir()
        except OSError:
            pass
        return out
    def render_stream(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        job_id: str,
        seed: int = 42,
        use_face_crop: bool = True,
        tail_seconds: float = 0.5,
    ) -> dict:
        """`render()` for live playback: the same `once`-mode generation, but frames
        go into one long-lived ffmpeg that emits HLS (fMP4) segments of one chunk
        (0.96 s) each, and every finished segment is pushed to the progress Queue
        (partition = job_id) the moment ffmpeg closes it. A player can start ~2
        chunks in and never catch up, because generation runs at 1.17x realtime
        (eager) on the L4.

        Queue items: {"type": "log", "line": str} and
        {"type": "file", "name": "init.mp4" | "seg_000.m4s" | "index.m3u8",
         "part": i, "parts": n, "data": bytes}  (files > FILE_PART bytes are split;
        Queue items are capped at 1 MiB). Files precede the playlist that lists
        them; the final playlist carries EXT-X-ENDLIST. The caller remuxes to mp4.

        Delivery goes through the Queue rather than a Modal generator: generator
        outputs were measured arriving 5-20 s late and in bursts, which starves
        the player; Queue put->get is 0.6-0.8 s.
        """
        import math
        import queue
        import re
        import subprocess
        import tempfile
        import threading
        import wave

        import librosa
        import numpy as np
        import torch

        from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline

        FILE_PART = 900_000
        t0 = time.time()
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
        hls = work / "hls"
        hls.mkdir()
        img, wav, padded_wav = work / "ref.png", work / "speech.wav", work / "speech_padded.wav"
        img.write_bytes(image_bytes)
        wav.write_bytes(audio_bytes)

        # Everything a chunk produces (log lines, segment, playlist) travels as ONE
        # Queue item: the consumer pays a ~0.5 s round trip per item, and 3 items
        # per 0.8 s chunk fell ~10 s behind by chunk 20. Batches stay < 1 MiB.
        batch: list[dict] = []
        batch_bytes = [0]
        # Queue.put is a network round trip (~0.1 s); done inline it cost the GPU
        # loop ~0.1 s per chunk (0.82 -> 0.91 s). A single sender thread keeps
        # FIFO order and takes it off the critical path.
        send_q: queue.Queue = queue.Queue()
        send_errors: list[str] = []

        def sender() -> None:
            while (item := send_q.get()) is not None:
                try:
                    self.host.put(item, partition=job_id)
                except Exception as exc:  # noqa: BLE001 - keep streaming; report at the end
                    send_errors.append(f"{type(exc).__name__}: {exc}"[:200])

        sender_thread = threading.Thread(target=sender, daemon=True)
        sender_thread.start()

        def flush() -> None:
            if batch:
                send_q.put({"type": "batch", "items": list(batch)})
                batch.clear()
                batch_bytes[0] = 0

        def put(item: dict) -> None:
            size = len(item.get("data", b"")) + 200
            if batch and batch_bytes[0] + size > FILE_PART:
                flush()
            batch.append(item)
            batch_bytes[0] += size

        def log(line: str) -> None:
            line = f"[{time.time()-t0:5.1f}s] {line}"
            print(line, flush=True)
            put({"type": "log", "line": line})
            flush()                              # log lines go out immediately, not with the next segment

        def send_file(name: str, data: bytes) -> None:
            parts = max(1, math.ceil(len(data) / FILE_PART))
            for i in range(parts):
                put({"type": "file", "name": name, "part": i, "parts": parts, "data": data[i * FILE_PART : (i + 1) * FILE_PART]})

        proc = None
        try:
            log(f"preparing reference (face crop, seed {seed})")
            sess = self._session(image_bytes, seed, use_face_crop, job_id)
            p = get_infer_params()
            sr, fps = p["sample_rate"], p["tgt_fps"]
            frame_num, motion_frames_num = p["frame_num"], p["motion_frames_num"]
            slice_len = frame_num - motion_frames_num
            height, width = p["height"], p["width"]

            speech, _ = librosa.load(str(wav), sr=sr, mono=True)
            speech_s = len(speech) / sr
            need_frames = math.ceil(speech_s * fps) + math.ceil(tail_seconds * fps)
            k = max(1, math.ceil((need_frames - motion_frames_num) / slice_len))
            target = (frame_num + k * slice_len) * sr // fps
            if len(speech) < target:
                speech = np.concatenate([speech, np.zeros(target - len(speech), dtype=speech.dtype)])
            video_s = (motion_frames_num + k * slice_len) / fps
            with wave.open(str(padded_wav), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            log(f"speech {speech_s:.2f}s → video {video_s:.2f}s ({k} chunks, tail ≥{tail_seconds:.1f}s)")
            flush()

            emb = get_audio_embedding(self.pipeline, speech)
            n_chunks = (emb.shape[1] - frame_num) // slice_len
            chunks = [emb[:, i * slice_len : i * slice_len + frame_num].contiguous() for i in range(n_chunks)]

            # One keyframe per chunk (GOP = slice_len) so each HLS segment is exactly
            # one chunk and closes as soon as the next chunk's first frame arrives.
            # temp_file: segments/playlist appear atomically, so anything listed in
            # index.m3u8 is complete. Audio is cut to the video length up front
            # instead of relying on -shortest against a slow pipe.
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
                "-t", f"{video_s:.3f}", "-i", str(padded_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-crf", "20",
                "-g", str(slice_len), "-keyint_min", str(slice_len), "-sc_threshold", "0", "-bf", "0",
                "-c:a", "aac", "-b:a", "128k",
                "-f", "hls", "-hls_time", f"{slice_len / fps:.2f}", "-hls_segment_type", "fmp4",
                "-hls_playlist_type", "event", "-hls_list_size", "0",
                "-hls_flags", "independent_segments+temp_file",
                "-hls_fmp4_init_filename", "init.mp4",
                "-hls_segment_filename", str(hls / "seg_%03d.m4s"),
                str(hls / "index.m3u8"),
            ]
            ffmpeg_log = open(work / "ffmpeg.log", "wb")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffmpeg_log)
            frames_q: queue.Queue = queue.Queue()

            def writer() -> None:
                # decoupled from the GPU loop so a slow encode never stalls generation
                assert proc is not None and proc.stdin is not None
                try:
                    while (arr := frames_q.get()) is not None:
                        proc.stdin.write(arr.tobytes())
                finally:
                    proc.stdin.close()

            wt = threading.Thread(target=writer, daemon=True)
            wt.start()

            sent: set[str] = set()
            last_playlist = [""]
            seg_re = re.compile(r"^(seg_\d+\.m4s)$", re.M)
            streamed = [0]

            def ship_new_files() -> None:
                pl = hls / "index.m3u8"
                if pl.exists():
                    text = pl.read_text()
                    # The fMP4 init file is opened at start but only gets its moov
                    # (delay_moov) when the first segment is flushed — i.e. right
                    # before the playlist first lists a segment. Reading it earlier
                    # shipped 0 bytes once; so ship it only then, and only if non-empty.
                    init = hls / "init.mp4"
                    if "init.mp4" not in sent and seg_re.search(text) and init.exists() and init.stat().st_size > 0:
                        sent.add("init.mp4")
                        data = init.read_bytes(); streamed[0] += len(data); send_file("init.mp4", data)
                    if "init.mp4" not in sent:
                        return                      # never advertise segments before their init
                    for name in seg_re.findall(text):
                        if name not in sent and (hls / name).exists():
                            sent.add(name)
                            data = (hls / name).read_bytes(); streamed[0] += len(data); send_file(name, data)
                    if text != last_playlist[0] and text.endswith("\n"):
                        last_playlist[0] = text
                        send_file("index.m3u8", text.encode())

            log(f"{n_chunks} chunks to generate, streaming {slice_len / fps:.2f}s segments")
            for i, chunk in enumerate(chunks):
                block, gen_s, _ = self._run_chunk(sess, chunk, drop_motion=i != 0)
                frames_q.put(block)
                log(f"chunk {i+1}/{n_chunks} done ({gen_s:.2f}s)")
                ship_new_files()
                flush()
            frames_q.put(None)
            wt.join(timeout=120)
            rc = proc.wait(timeout=120)
            ffmpeg_log.close()
            if rc != 0:
                raise RuntimeError("ffmpeg failed: " + (work / "ffmpeg.log").read_text()[-2000:])
            ship_new_files()
            if "#EXT-X-ENDLIST" not in last_playlist[0]:
                raise RuntimeError("ffmpeg exited without finalizing the playlist")
            log(f"{DONE_MARK} {n_chunks} chunks, {len(sent)-1} segments, {streamed[0]/1e6:.1f} MB streamed")
            flush()
            # everything must be in the Queue before the call returns: the studio
            # does its final drain right after FunctionCall.get() succeeds
            send_q.put(None)
            sender_thread.join(timeout=120)
            if send_errors:
                raise RuntimeError(f"{len(send_errors)} Queue puts failed, e.g. {send_errors[0]}")
            return {"chunks": n_chunks, "segments": len(sent) - 1, "bytes": streamed[0], "seconds": round(time.time() - t0, 1)}
        finally:
            self.sessions.pop(job_id, None)
            if sender_thread.is_alive():
                send_q.put(None)
            if proc is not None and proc.poll() is None:
                proc.kill()
            try:
                for f in sorted(work.rglob("*"), reverse=True):
                    f.unlink() if f.is_file() else f.rmdir()
                work.rmdir()
            except OSError:
                pass


def _pct(xs: list, q: int) -> float:
    """q-th percentile, nearest-rank. Small samples are the norm here (a session
    holds a few dozen turns), so an interpolating definition would invent values
    between measurements that never happened."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, (q * len(s) + 99) // 100 - 1))]


# The three gates that decide whether a session count "works". All three must
# hold: a run can keep video realtime (gate 3) while freezing a third of the
# avatar's turns (gate 1), which is the failure this whole experiment is for.
# What counts as passing. These are product decisions, not measurements, so they
# come from the environment: the wait is the silence a candidate hears before the
# avatar answers, and how much of it is tolerable is not something the scheduler
# can know. freeze_out — the share of turns delayed at all, even by 10 ms — is
# kept as a diagnostic rather than a gate, because a wait too short to hear is
# not a failure; set FLASHHEAD_GATE_FREEZEOUT to put teeth back in it.
GATES = {"freeze_out": float(os.environ.get("FLASHHEAD_GATE_FREEZEOUT", "0.01")),
         "wait_p95_s": float(os.environ.get("FLASHHEAD_GATE_WAIT", "0.5")),
         "behind_tail_s": float(os.environ.get("FLASHHEAD_GATE_BEHIND", "0.5"))}


def verdict(results: dict) -> dict:
    """Aggregate a run to the numbers the gates are read from, worst session wins."""
    ok = [r for r in results.values() if "error" not in r]
    if not ok:
        return {"pass": False, "reason": "every session errored"}
    turns = sum(r.get("turns", 0) for r in ok)
    frozen = sum(r.get("turns_frozen", 0) for r in ok)
    pooled = sorted(w for r in ok for w in (r.get("waits_s") or []))
    agg = {
        "sessions": len(results),
        "errors": len(results) - len(ok),
        "turns": turns,
        "turns_frozen": frozen,
        # run-wide, not the mean of per-session rates: a session with three turns
        # must not weigh the same as one with forty
        "freeze_out": round(frozen / turns, 4) if turns else 0.0,
        # A true percentile over every turn in the run, not the worst of the
        # per-session percentiles. Each session contributes about fifteen turns,
        # so its own p95 is its worst turn, and taking the max across sessions
        # made the gate read "no single turn anywhere waited more than the limit".
        # That is a maximum wearing a percentile's name, and it gets *stricter*
        # the longer the run — more draws, more chances at one bad one — which
        # turns a better-sampled measurement into a harsher one.
        "wait_p50_s": _pct(pooled, 50),
        "wait_p95_s": _pct(pooled, 95),
        "wait_p99_s": _pct(pooled, 99),
        "wait_max_s": max(pooled, default=0.0),
        "behind_tail_s": max((r.get("behind_tail_s") or 0.0 for r in ok), default=0.0),
        "gen_median_s": _pct([r["gen_median_s"] for r in ok if r.get("gen_median_s")], 50),
        "slot_budget": max((r.get("slot_budget", 0) for r in ok), default=0),
        "slot_overruns": max((r.get("slot_overruns", 0) for r in ok), default=0),
    }
    failed = [k for k, lim in GATES.items() if agg.get(k, 0) > lim]
    agg["pass"] = not failed and not agg["errors"]
    agg["failed_gates"] = failed
    return agg


def verdict_line(results: dict) -> str:
    v = verdict(results)
    mark = "PASS" if v.get("pass") else "FAIL"
    if "reason" in v:
        return f"  판정: FAIL — {v['reason']}"
    return (f"  판정: {mark} — freeze-out {v['freeze_out']*100:.1f}% (한계 {GATES['freeze_out']*100:.0f}%) · "
            f"대기 p95 {v['wait_p95_s']}s (한계 {GATES['wait_p95_s']:g}s) · "
            f"영상 지연 끝 {v['behind_tail_s']}s (한계 {GATES['behind_tail_s']:g}s)"
            + (f" · 불합격 게이트: {', '.join(v['failed_gates'])}" if v["failed_gates"] else ""))


def print_summary(results: dict) -> None:
    """Report a multi-session run. Shared by the Modal entrypoint and the
    plain-process harness so both read the same way."""
    print("\n=== 세션별")
    tot_chunks = tot_gpu = 0
    for i, (sid, r) in enumerate(results.items()):
        if "error" in r:
            print(f"  {i+1}: {r['error']}")
            continue
        tot_chunks += r["chunks"]; tot_gpu += r["gpu_chunks"]
        print(f"  {i+1}: 청크 {r['chunks']:4d} · GPU {r['gpu_chunks']:3d} ({r['gpu_chunk_ratio']*100:.0f}%) · "
              f"생성 {r['gen_median_s']}s · 영상 지연 최대 {r['behind_max_s']}s / 끝 {r['behind_tail_s']}s · "
              f"턴 {r.get('turns', 0)} 중 {r.get('turns_frozen', 0)} 밀림 "
              f"(freeze-out {r.get('freeze_out', 0)*100:.1f}%) · "
              f"대기 p50 {r.get('wait_p50_s', 0)}s / p95 {r.get('wait_p95_s', 0)}s / 최대 {r['wait_to_speak_max_s']}s "
              f"· 동거 {r['peer_sessions']}")
        print(f"     생성 분해: 상태교체 {r.get('gen_swap_s')}s + 모델 {r.get('gen_pipe_s')}s "
              f"+ 호스트전송 {r.get('gen_xfer_s')}s")
    if tot_chunks:
        ok = [r for r in results.values() if "error" not in r]
        span = max(r["seconds"] for r in ok)
        print(f"\n=== 합계: {len(results)} 세션이 컨테이너 1대를 공유 (세션 최장 {span:.0f}s)")
        print(f"  영상 청크 {tot_chunks} (= {tot_chunks*0.96/60:.1f}분 분량) · GPU 청크 {tot_gpu} "
              f"({tot_gpu/tot_chunks*100:.0f}%) · GPU가 만든 시간 {tot_gpu*0.96/60:.1f}분")
        print(f"  실시간 유지: 세션 길이가 요구하는 청크 {span/0.96:.0f} vs 실제 {[r['chunks'] for r in ok]}")
        # occupancy is GPU seconds over wall seconds. The previous form multiplied
        # a per-SESSION chunk count by a per-SLOT batch time, so every slot with
        # more than one speaker was counted as many times as it had speakers: at
        # 16 sessions it printed 210%, which is not a thing GPU time can do. An
        # earlier pass fixed the constant it used and left the unit mismatch.
        gens = sorted(r["gen_median_s"] for r in ok if r.get("gen_median_s"))
        gen = gens[len(gens) // 2] if gens else 0.0
        busy = sum(r.get("gpu_seconds") or 0.0 for r in ok)
        print(f"  GPU 점유율 {busy/span*100:.0f}% "
              f"(청크 {gen:.3f}s 기준) · 세션당 GPU 비용은 {len(results)}분의 1")
        print(verdict_line(results))
        # the histogram is container-wide, so any session's copy is the whole run's
        hist = next((r.get("slot_speakers") for r in ok if r.get("slot_speakers")), None)
        if hist:
            budget = next(r.get("slot_budget") for r in ok if r.get("slot_speakers"))
            slots = sum(hist) or 1
            spoken = sum(hist[1:])
            print(f"\n=== 동시 발화 (슬롯당 예산 {budget})")
            print(f"  발화가 있었던 슬롯 {spoken} / 전체 {slots}")
            for n, c in enumerate(hist):
                if c and n:
                    print(f"    {n}개 세션이 동시에 발화: {c:5d} 슬롯 ({c/spoken*100:4.1f}% of 발화 슬롯)")
            top = max((n for n, c in enumerate(hist) if c), default=0)
            print(f"  최대 동시 발화 {top}개 · 평균 {sum(n*c for n, c in enumerate(hist))/spoken:.2f}개"
                  if spoken else "  발화 없음")
