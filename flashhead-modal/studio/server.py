"""FlashHead Studio — type a line, get a talking-head clip.

Runs on the Mac because two of the three stages have to:
  1. text  -> WAV      macOS `say -v "Yuna (Premium)"`, 16 kHz mono (Mac only)
  2. WAV   -> MP4      the deployed Modal app "flashhead" (`modal deploy ../app.py`),
                       a resident `Renderer` class on an L4 called via the SDK
  3. serve the MP4     plain static file

Stage 2 keeps the pipeline loaded between calls, so a warm container renders a
30 s clip in ~30 s while a cold one first pays ~70 s of container start + model
load. The "예열" toggle pins one container (`update_autoscaler(min_containers=1)`)
at ~$0.80/hr; off, it scales to zero `SCALEDOWN_S` after the last job.

Jobs run one at a time: parallel containers would multiply GPU cost, and this
is a demo, not a service.

    ./studio.sh            # http://127.0.0.1:8300
"""

from __future__ import annotations

import faulthandler
import hashlib
import importlib.util
import json
import math
import os
import queue as stdqueue
import shutil
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
APP_PY = HERE.parent / "app.py"
MODAL = Path("/Users/kjs0703/playground/opentalk/opentalking/.venv/bin/modal")
JOBS_DIR = HERE / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# Import app.py for its constants and `tail_progress` without triggering a
# deploy — module import only builds the Image *definition*. app.py imports the
# host-free core beside it, so that directory has to be importable from here.
if str(APP_PY.parent) not in sys.path:
    sys.path.insert(0, str(APP_PY.parent))
_spec = importlib.util.spec_from_file_location("flashhead_app", APP_PY)
fh = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(fh)

VOICE = "Yuna (Premium)"
MAX_CHARS = 1000
# Avatar library: every image in this folder is selectable; the file stem is the
# id shown in the UI. Drop files in by hand or upload from the page.
AVATARS_DIR = HERE / "avatars"
AVATARS_DIR.mkdir(exist_ok=True)
AVATAR_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_AVATAR = "newscaster"
AVATAR_MAX_SIDE = 2048     # FlashHead works at 512², anything bigger is just upload weight
AVATAR_MIN_SIDE = 256      # below this the face crop has nothing to work with
# Minimum silence after the last word. The renderer pads the audio so the video
# covers speech + this tail (the 0.96 s chunk grid can add up to one more slice).
VIDEO_TAIL_S = 0.5
# Lite generates 33-frame chunks that advance 24 new frames each, at 25 fps.
FRAME_NUM, SLICE_LEN, FPS = 33, 24, 25
# Pro (Wan VAE) advances 28 frames per chunk with a 5-frame overlap and, on the
# L4, takes ~8.6 s per chunk (A/B 2026-09-14) — a final-render option, not live.
PRO_SLICE_LEN, PRO_MOTION, PRO_CHUNK_S = 28, 5, 8.6
# A cold render is bounded by container start + load; anything beyond this is a hang.
RENDER_TIMEOUT_S = 1800


@dataclass
class Job:
    id: str
    text: str                       # the line that was spoken, or the uploaded file's name
    created: float
    source: str = "tts"             # tts (macOS say) | upload (user-provided audio)
    status: str = "queued"          # queued | tts | render | done | error
    avatar: str = DEFAULT_AVATAR
    hls_url: str = ""               # set once init + first segment + playlist are on disk
    segments: int = 0
    error: str = ""
    # optional Pro final render of the same speech/avatar/seed
    pro_status: str = ""            # "" | queued | render | done | error
    pro_error: str = ""
    pro_chunks_done: int = 0
    pro_expected_chunks: int = 0
    pro_started: float = 0.0
    pro_finished: float = 0.0
    pro_log: deque = field(default_factory=lambda: deque(maxlen=100))
    audio_seconds: float = 0.0
    expected_chunks: int = 0
    chunks_done: int = 0
    started: float = 0.0
    finished: float = 0.0
    log: deque = field(default_factory=lambda: deque(maxlen=200))

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def public(self) -> dict:
        d = asdict(self)
        d["log"] = list(self.log)
        d["pro_log"] = list(self.pro_log)
        d["elapsed"] = round((self.finished or time.time()) - self.started, 1) if self.started else 0.0
        d["pro_elapsed"] = round((self.pro_finished or time.time()) - self.pro_started, 1) if self.pro_started else 0.0
        d["pro_video_url"] = f"/videos/pro/{self.id}.mp4" if self.pro_status == "done" else None
        d["pro_video_path"] = str(self.dir / "video_pro.mp4") if self.pro_status == "done" else None
        d["pro_eta_s"] = int(self.audio_seconds * PRO_CHUNK_S / (PRO_SLICE_LEN / FPS) + 90)   # + cold start
        d["video_url"] = f"/videos/{self.id}.mp4" if self.status == "done" else None
        d["video_path"] = str(self.dir / "video.mp4") if self.status == "done" else None
        av = _avatars().get(self.avatar)
        d["avatar_url"] = f"/avatars/{av.name}" if av else None
        return d

    def persist(self) -> None:
        (self.dir / "job.json").write_text(json.dumps(self.public(), ensure_ascii=False, indent=1))


jobs: dict[str, Job] = {}
queue: deque[str] = deque()
lock = threading.Lock()
wake = threading.Event()


def _log(job: Job, line: str) -> None:
    job.log.append(line)


def _avatars() -> dict[str, Path]:
    """id (file stem) -> image path, default first, then by name."""
    files = [p for p in AVATARS_DIR.iterdir() if p.suffix.lower() in AVATAR_EXTS and not p.name.startswith(".")]
    files.sort(key=lambda p: (p.stem != DEFAULT_AVATAR, p.stem.lower()))
    return {p.stem: p for p in files}


AUDIO_MAX_BYTES = 200_000_000
AUDIO_MAX_SECONDS = 900.0


def _probe_seconds(src: Path) -> float:
    """Duration of an audio/video file, or raise — cheap enough (~50 ms) to run at
    upload time so a bad file is rejected in the request instead of as a dead card."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(src)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise ValueError("오디오를 읽을 수 없습니다 (지원하지 않는 형식이거나 손상된 파일)") from None


def _to_speech_wav(src: Path, dest: Path) -> float:
    """Any audio (or a video's audio track) -> 16 kHz mono PCM WAV, the only format
    FlashHead's wav2vec2 front end takes. Returns the duration in seconds."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0 or not dest.exists():
        raise ValueError(f"오디오를 읽을 수 없습니다: {r.stderr.strip()[:200]}")
    with wave.open(str(dest), "rb") as w:
        seconds = w.getnframes() / w.getframerate()
    if seconds < 0.3:
        raise ValueError(f"오디오가 너무 짧습니다 ({seconds:.1f}s)")
    if seconds > AUDIO_MAX_SECONDS:
        raise ValueError(f"오디오가 너무 깁니다 ({seconds:.0f}s); {AUDIO_MAX_SECONDS:.0f}s 이하만 받습니다")
    return seconds


def _set_audio_plan(job: Job, seconds: float) -> None:
    job.audio_seconds = seconds
    # same rule as Renderer.render: k chunks render 9 + 24k frames >= speech + tail
    need = math.ceil(seconds * FPS) + math.ceil(VIDEO_TAIL_S * FPS)
    job.expected_chunks = max(1, math.ceil((need - (FRAME_NUM - SLICE_LEN)) / SLICE_LEN))


def _prepare_audio(job: Job) -> Path:
    """Produce jobs/<id>/speech.wav, either by synthesizing the text or by
    converting the audio the user uploaded (already saved as `upload.*`)."""
    job.status = "tts"
    job.dir.mkdir(parents=True, exist_ok=True)
    wav = job.dir / "speech.wav"
    if job.source == "upload":
        src = next((p for p in job.dir.glob("upload.*")), None)
        if src is None:
            raise FileNotFoundError("업로드된 오디오 파일을 찾을 수 없습니다")
        _log(job, f"audio: {src.name} → 16 kHz mono")
        seconds = _to_speech_wav(src, wav)
        _set_audio_plan(job, seconds)
        _log(job, f"audio ready: {seconds:.1f}s → {job.expected_chunks} chunks")
        return wav
    txt = job.dir / "text.txt"
    txt.write_text(job.text, encoding="utf-8")
    _log(job, f"TTS: say -v \"{VOICE}\"")
    subprocess.run(
        ["say", "-v", VOICE, "-o", str(wav), "--data-format", "LEI16@16000", "-f", str(txt)],
        check=True, capture_output=True, timeout=120,
    )
    with wave.open(str(wav), "rb") as r:
        _set_audio_plan(job, r.getnframes() / r.getframerate())
    _log(job, f"TTS done: {job.audio_seconds:.1f}s → {job.expected_chunks} chunks")
    return wav


# ----------------------------------------------------------------------------
# Modal side. `Renderer` is looked up lazily so the studio starts even if the
# app is not deployed yet; the first job then reports the lookup error.
_renderer = None


def renderer(fresh: bool = False):
    """Handle to the deployed class. `fresh=True` drops the cached handle — after
    the app is stopped in the Modal console and redeployed, the old handle keeps
    failing with ConflictError even though the new deployment is fine."""
    global _renderer
    if _renderer is None or fresh:
        _renderer = modal.Cls.from_name(fh.APP_NAME, "Renderer")()
    return _renderer


def _spawn_render(*args, **kwargs):
    try:
        return renderer().render_stream.spawn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if "stopped" in msg or "disabled" in msg or "NotFound" in msg:
            # app was stopped/redeployed under us: re-resolve once
            return renderer(fresh=True).render_stream.spawn(*args, **kwargs)
        raise


_CHUNK_RE = re.compile(r"chunk (\d+)/(\d+) done")
_HLS_NAME = re.compile(r"^(init\.mp4|seg_\d+\.m4s|index\.m3u8)$")
_HLS_MEDIA = {".m3u8": "application/vnd.apple.mpegurl", ".mp4": "video/mp4", ".m4s": "video/iso.segment"}
_LIVE_RE = re.compile(r"live: chunk (\d+) (speech|deferred|silence) gen ([\d.]+)s behind ([\d.]+)s queued ([\d.]+)s")


def _hls_sink(hls_dir: Path, item: dict, parts: dict) -> str | None:
    """Write one streamed file item (possibly split into parts) atomically. Returns the file name when complete."""
    name = item["name"]
    if not _HLS_NAME.match(name):
        return None
    if item.get("parts", 1) > 1:
        bucket = parts.setdefault(name, {})
        bucket[item["part"]] = item["data"]
        if len(bucket) < item["parts"]:
            return None
        data = b"".join(bucket[i] for i in range(item["parts"]))
        parts.pop(name, None)
    else:
        data = item["data"]
    tmp = hls_dir / (name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, hls_dir / name)
    return name
_TS_RE = re.compile(r"^\[\s*([\d.]+)s\]")


def _render(job: Job, wav: Path) -> Path:
    """Stream the render: HLS files arrive per chunk and are served live from
    jobs/<id>/hls/; when the generator finishes, the segments are remuxed
    (no re-encode) into jobs/<id>/video.mp4 for the gallery."""
    job.status = "render"
    out = job.dir / "video.mp4"
    hls_dir = job.dir / "hls"
    hls_dir.mkdir(exist_ok=True)
    image = _avatars().get(job.avatar)
    if image is None:
        raise FileNotFoundError(f"avatar '{job.avatar}' is no longer in {AVATARS_DIR.name}/")
    if _fresh(warm_state["ready"]):
        warm_hint = "warm"
    elif warm_state["containers"]:
        warm_hint = "container still loading — the call waits for it"
    else:
        warm_hint = "cold — container start + model load first (~1.5–2.5 min)"
    _log(job, f"Modal: Renderer.render_stream → {fh.APP_NAME}, avatar {image.name} ({warm_hint})")
    t0 = time.time()
    have: set[str] = set()
    parts: dict[str, dict] = {}          # name -> {part index: bytes} for split files
    call_started = [0.0]                 # set when the container's first line arrives (t=0 there)

    def handle(item: dict) -> None:
        if item["type"] == "batch":                 # one item per chunk: logs + segment + playlist
            for sub_item in item["items"]:
                handle(sub_item)
            return
        if not call_started[0] and item["type"] == "log":
            ts = _TS_RE.match(item["line"])
            if ts:
                call_started[0] = time.time() - float(ts.group(1))
        if item["type"] == "log":
            line = item["line"]
            m = _CHUNK_RE.search(line)
            if m:
                job.chunks_done = int(m.group(1))
            # container-relative stamp is in the line; append how late it reached us
            ts = _TS_RE.match(line)
            if ts and call_started[0]:
                lag = (time.time() - call_started[0]) - float(ts.group(1))
                line = f"{line[:150]}  (+{lag:.1f}s)"
            _log(job, line[:170])
            return
        if item["type"] != "file" or not _HLS_NAME.match(item["name"]):
            return
        name = item["name"]
        if item.get("parts", 1) > 1:
            bucket = parts.setdefault(name, {})
            bucket[item["part"]] = item["data"]
            if len(bucket) < item["parts"]:
                return
            data = b"".join(bucket[i] for i in range(item["parts"]))
            parts.pop(name, None)
        else:
            data = item["data"]
        tmp = hls_dir / (name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, hls_dir / name)          # never let the player read a partial file
        have.add(name)
        if name.endswith(".m4s"):
            job.segments += 1
        # start playback once three segments are on disk: ~3 s of buffer against
        # the ~1 s Queue delivery jitter; generation then stays ahead by itself
        init_ok = (hls_dir / "init.mp4").exists() and (hls_dir / "init.mp4").stat().st_size > 0
        if not job.hls_url and init_ok and "index.m3u8" in have and job.segments >= 3:
            job.hls_url = f"/hls/{job.id}/index.m3u8"
            _log(job, f"streaming: {job.segments} segments ready after {time.time()-t0:.1f}s → playback starts")

    call = _spawn_render(image.read_bytes(), wav.read_bytes(), job.id, seed=42, tail_seconds=VIDEO_TAIL_S)
    while True:
        if time.time() - t0 > RENDER_TIMEOUT_S:
            call.cancel()
            raise TimeoutError(f"render exceeded {RENDER_TIMEOUT_S}s")
        try:
            items = fh.progress.get_many(20, block=True, timeout=1, partition=job.id)
        except stdqueue.Empty:
            items = []
        if items:
            for item in items:             # one round trip drains the whole backlog
                handle(item)
            continue
        try:
            call.get(timeout=0)
            break
        except TimeoutError:
            pass
    for item in fh.progress.get_many(100, block=False, partition=job.id):
        handle(item)
    if not job.segments:
        raise RuntimeError("render finished but no segments were streamed")
    # remux the finished stream into a plain mp4 (copy, no re-encode)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(hls_dir / "index.m3u8"),
         "-c", "copy", "-movflags", "+faststart", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out.exists():
        # fallback: fMP4 init + segments concatenate into a valid fragmented mp4
        blob = job.dir / "concat.mp4"
        with blob.open("wb") as f:
            f.write((hls_dir / "init.mp4").read_bytes())
            for seg in sorted(hls_dir.glob("seg_*.m4s")):
                f.write(seg.read_bytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(blob), "-c", "copy",
                        "-movflags", "+faststart", str(out)], check=True)
        blob.unlink()
        _log(job, "remux via playlist failed, used concat fallback: " + r.stderr.strip()[:120])
    return out


def _worker() -> None:
    while True:
        wake.wait()
        with lock:
            if not queue:
                wake.clear()
                continue
            job = jobs[queue.popleft()]
        job.started = time.time()
        try:
            wav = _prepare_audio(job)
            _render(job, wav)
            job.status = "done"
            _log(job, "done")
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            _log(job, "ERROR " + job.error)
        finally:
            job.finished = time.time()
            job.persist()
            _poll_containers()


threading.Thread(target=_worker, daemon=True, name="studio-worker").start()


# ----------------------------------------------------------------------------
# Live sessions: Renderer.live() generates continuously (speech or silence) on the
# Lite container; the studio feeds sentences as PCM chunks and mirrors the HLS.
LIVE_DIR = HERE / "live"
LIVE_DIR.mkdir(exist_ok=True)
LIVE_CHUNK_SAMPLES = SLICE_LEN * 16000 // FPS          # 15360 samples = 0.96 s
LIVE_IDLE_TIMEOUT_S = 600.0
# One container serves several sessions (app.py: @modal.concurrent(max_inputs=8)).
# follow the container's own cap: app.py sets @modal.concurrent(max_inputs=…)
# from it, and a studio that let more sessions through than the container
# accepts would simply queue them outside the scheduler
LIVE_MAX_SESSIONS = fh.LIVE_MAX_SESSIONS
# The interview rhythm we are modelling: the avatar asks for ~7 s, the candidate
# answers for ~35 s. That 1-in-6 speaking ratio is what sets how many sessions
# fit on one GPU, so the demo drives it rather than leaving it to hand-typing.
LIVE_SPEAK_S = 7.0
LIVE_CYCLE_S = 42.0
# One scripted session is (아바타 발화 → 응시자 발화) × 3, then a closing avatar
# turn: four avatar utterances, three candidate gaps, then the session ends.
# Anything longer only repeats what the first cycles already showed.
LIVE_TURNS = 4
# Sessions start one at a time. Each start uploads the avatar PNG (~0.8 MB) in the
# spawn payload, so six "+" clicks in two seconds push ~5 MB through one Modal
# client at once — enough for a 1-second QueueGet to miss its deadline.
LIVE_START_SPACING_S = 1.5
# A failed poll says nothing about the session: the container keeps generating and
# the queue keeps the items. Only give up after this many consecutive failures.
LIVE_CLIENT_RETRIES = 30
# ~40-45 Korean characters is ~7 s in Yuna Premium (measured).
INTERVIEW_QUESTIONS = [
    "자기소개를 부탁드립니다. 어떤 일을 해오셨는지 편하게 말씀해 주시면 좋겠습니다.",
    "최근에 맡으신 프로젝트 중에서 가장 어려웠던 문제는 무엇이었고, 어떻게 해결하셨나요?",
    "팀에서 의견이 갈렸던 경험이 있다면, 그때 어떤 방식으로 합의를 이끌어내셨는지 궁금합니다.",
    "지금까지의 경력에서 가장 크게 성장했다고 느낀 순간은 언제였나요? 구체적으로 말씀해 주세요.",
    "새로운 기술을 익혀야 했던 상황에서 어떤 순서로 접근하시는지, 최근 사례를 들어 설명해 주세요.",
    "일정이 촉박한 상황에서 품질과 속도 사이를 어떻게 조율하시는지 경험을 들려주시겠어요?",
    "동료에게 피드백을 줄 때 특별히 신경 쓰시는 점이 있다면 무엇인지 말씀해 주시면 좋겠습니다.",
    "마지막으로, 저희 팀에 궁금하신 점이나 더 하고 싶으신 말씀이 있으면 편하게 이야기해 주세요.",
]


# A playground is one demo run: the sessions you opened together, their
# recordings and their Gantt. Starting a new one clears the screen without
# throwing anything away — old runs stay loadable.
PLAYGROUNDS_FILE = LIVE_DIR / "playgrounds.json"


@dataclass
class Playground:
    id: str
    name: str
    created: float

    def public(self) -> dict:
        mine = [s for s in live_sessions.values() if s.playground == self.id]
        chunks = sum(s.chunks for s in mine)
        speech = sum(s.speech_chunks for s in mine)
        spans = [sp for s in mine for sp in s.spans]
        return {**asdict(self), "sessions": len(mine),
                "active": sum(1 for s in mine if s.state in ("starting", "live")),
                "video_s": round(chunks * SLICE_LEN / FPS, 1),
                "gpu_s": round(speech * SLICE_LEN / FPS, 1),
                "span_s": round(max((b for _, b in spans), default=0) - min((a for a, _ in spans), default=0), 1)}


playgrounds: dict[str, Playground] = {}
current_playground = ""


def _save_playgrounds() -> None:
    PLAYGROUNDS_FILE.write_text(json.dumps(
        {"current": current_playground,
         "items": [{"id": p.id, "name": p.name, "created": p.created} for p in playgrounds.values()]},
        ensure_ascii=False, indent=1))


def _new_playground(name: str = "") -> Playground:
    global current_playground
    p = Playground(id=uuid.uuid4().hex[:8],
                   name=name or time.strftime("%m/%d %H:%M 시연"), created=time.time())
    playgrounds[p.id] = p
    current_playground = p.id
    _save_playgrounds()
    return p


def _subtract(spans: list, holes: list) -> list:
    """`spans` minus `holes`, both as [start, end] pairs on one clock."""
    out = []
    for a, b in spans:
        cur = [[a, b]]
        for ha, hb in holes:
            nxt = []
            for ca, cb in cur:
                if hb <= ca or ha >= cb:
                    nxt.append([ca, cb])
                    continue
                if ha > ca:
                    nxt.append([ca, ha])
                if hb < cb:
                    nxt.append([hb, cb])
            cur = nxt
        out.extend(c for c in cur if c[1] - c[0] > 0.3)
    return out


@dataclass
class LiveSession:
    id: str
    avatar: str
    created: float
    model: str = "lite"            # lite (L4, resident Renderer) | pro (2×H100 jp, resident LivePro workers)
    state: str = "starting"        # starting | live | ending | ended | error
    error: str = ""
    hls_url: str = ""
    segments: int = 0
    chunks: int = 0
    speech_chunks: int = 0
    queued_s: float = 0.0          # speech seconds queued on the container (from its log lines)
    behind_s: float = 0.0
    gen_s: float = 0.0
    said: list = field(default_factory=list)
    auto: bool = False              # drive the interview rhythm from the server
    turns: int = 0
    # when this session held the GPU, as [start, end] epoch pairs — the Gantt below
    # the grid is drawn straight from these
    spans: list = field(default_factory=list)
    # slots where this session had speech ready but another session held the GPU
    wait_spans: list = field(default_factory=list)
    container_t0: float = 0.0       # epoch of the container's own t=0 for this session
    playground: str = ""            # which demo run this session belongs to
    started_live: float = 0.0
    finished: float = 0.0
    recording_url: str = ""
    log: deque = field(default_factory=lambda: deque(maxlen=200))
    # the container's end-of-session stats, kept whole: this is the run's
    # record (gen_median_s, behind_*, slot_overruns, …) and the only place
    # the numbers survive once the container is gone
    stats: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return LIVE_DIR / self.id

    def listen_spans(self) -> list:
        """When the candidate is assumed to be answering.

        The demo asks one question per `LIVE_CYCLE_S` and the avatar's own speech
        is measured, so the candidate's turn is everything between the end of one
        question and the start of the next. The part where the avatar had its next
        question ready but was queued behind another session is cut out — that is
        the system stalling, not the candidate talking, and drawing it as the
        candidate's turn would hide exactly the contention this chart is for."""
        if not self.spans:
            return []
        edge = time.time() if self.state in ("starting", "live") else (self.finished or self.spans[-1][1])
        gaps = [[b, (self.spans[i + 1][0] if i + 1 < len(self.spans) else edge)]
                for i, (_, b) in enumerate(self.spans)]
        return _subtract([g for g in gaps if g[1] - g[0] > 0.5], self.wait_spans)

    def public(self) -> dict:
        d = asdict(self)
        d["log"] = list(self.log)
        d["elapsed"] = round((self.finished or time.time()) - self.created, 1)
        d["live_for"] = round((self.finished or time.time()) - self.started_live, 1) if self.started_live else 0.0
        d["gpu_ratio"] = round(self.speech_chunks / self.chunks, 3) if self.chunks else 0.0
        d["spans"] = [[round(a, 2), round(b, 2)] for a, b in self.spans]
        d["wait_spans"] = [[round(a, 2), round(b, 2)] for a, b in self.wait_spans]
        d["listen_spans"] = [[round(a, 2), round(b, 2)] for a, b in self.listen_spans()]
        return d


live_sessions: dict[str, LiveSession] = {}
_live_pro = None


def live_pro(fresh: bool = False):
    global _live_pro
    if _live_pro is None or fresh:
        _live_pro = modal.Cls.from_name(fh.APP_NAME, "LivePro")()
    return _live_pro


_live_start_gate = threading.Lock()
_live_last_start = 0.0


def _await_start_slot(sess: LiveSession) -> None:
    """Space session starts out so their avatar uploads do not collide."""
    global _live_last_start
    with _live_start_gate:
        wait = LIVE_START_SPACING_S - (time.time() - _live_last_start)
        if wait > 0:
            sess.log.append(f"앞 세션과 {wait:.1f}s 간격을 두고 시작")
            time.sleep(wait)
        _live_last_start = time.time()


def _live_end(sess: LiveSession) -> None:
    """Tell the container this session is done. Safe to call twice."""
    if sess.state in ("starting", "live"):
        _put_async({"type": "end"}, f"{sess.id}:audio", f"end {sess.id[:6]}")
        sess.state = "ending"
        sess.log.append("stop requested")


def _await_speech_drained(sess: LiveSession, timeout_s: float = 90.0) -> None:
    """Wait until the container has played out everything queued for this session.

    Ending a session discards whatever audio it still holds, so closing right
    after pushing the last question would cut the avatar off mid-sentence. The
    container reports its queue in every chunk line; speech is done when that
    reaches zero and no further speech chunk arrives."""
    t0, last, quiet = time.time(), sess.speech_chunks, 0.0
    while time.time() - t0 < timeout_s:
        time.sleep(0.5)
        if sess.state in ("ended", "error"):
            return
        if sess.speech_chunks != last:
            last, quiet = sess.speech_chunks, 0.0
            continue
        quiet += 0.5
        if quiet >= 2.5 and sess.speech_chunks and sess.queued_s <= 0.05:
            return
    sess.log.append(f"auto: 마지막 발화가 {timeout_s:.0f}s 안에 끝나지 않아 그대로 종료한다")


def _auto_driver(sess: LiveSession) -> None:
    """Ask a question every `LIVE_CYCLE_S`, which is the cadence the capacity
    numbers assume.

    The first question waits for the container to say `live: ready`, not for the
    studio to hold two HLS segments. Waiting for segments added ~4 s to the first
    response for nothing: the playlist is an EVENT playlist with every segment
    retained, so a viewer who attaches later still starts from the first frame,
    and the container's ingest thread buffers audio that arrives before then."""
    while not sess.started_live and sess.state not in ("ended", "error"):
        time.sleep(0.2)
    while sess.auto and sess.state in ("starting", "live"):
        q = INTERVIEW_QUESTIONS[sess.turns % len(INTERVIEW_QUESTIONS)]
        sess.turns += 1
        try:
            _live_say(sess, q)
        except Exception as exc:  # noqa: BLE001
            sess.log.append(f"auto say failed: {type(exc).__name__}: {exc}")
        if sess.turns >= LIVE_TURNS:
            _await_speech_drained(sess)
            sess.log.append(f"auto: 아바타 발화 {LIVE_TURNS}회를 마쳐 세션을 종료한다")
            _live_end(sess)
            return
        # the candidate's turn: the rest of the cycle after the avatar's question
        for _ in range(int(LIVE_CYCLE_S * 2)):
            if not (sess.auto and sess.state in ("starting", "live")):
                return
            time.sleep(0.5)


def _live_consumer(sess: LiveSession) -> None:
    hls_dir = sess.dir / "hls"
    hls_dir.mkdir(parents=True, exist_ok=True)
    parts: dict = {}
    image = _avatars().get(sess.avatar)
    try:
        if image is None:
            raise FileNotFoundError(f"avatar '{sess.avatar}' missing")
        handle = live_pro if sess.model == "pro" else renderer
        where = "LivePro.live (2×H100 jp)" if sess.model == "pro" else "Renderer.live (L4)"
        sess.log.append(f"Modal: {where} → {fh.APP_NAME}, avatar {image.name}")
        _await_start_slot(sess)
        try:
            call = handle().live.spawn(sess.id, image.read_bytes(), seed=42, idle_timeout_s=LIVE_IDLE_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            if not any(k in str(exc) for k in ("stopped", "disabled", "NotFound")):
                raise
            call = handle(fresh=True).live.spawn(sess.id, image.read_bytes(), seed=42, idle_timeout_s=LIVE_IDLE_TIMEOUT_S)

        def handle(item) -> None:
            if not isinstance(item, dict):
                return
            if item["type"] == "batch":
                for it in item["items"]:
                    handle(it)
                return
            if item["type"] == "log":
                line = item["line"]
                ts = _TS_RE.match(line)
                if ts and not sess.container_t0:
                    # the container stamps every line with its own elapsed time;
                    # anchoring it once puts every session on one wall clock
                    sess.container_t0 = time.time() - float(ts.group(1))
                m = _LIVE_RE.search(line)
                if m:
                    sess.chunks = int(m.group(1)); sess.gen_s = float(m.group(3)); sess.behind_s = float(m.group(4)); sess.queued_s = float(m.group(5))
                    kind = m.group(2)
                    if kind == "speech":
                        sess.speech_chunks += 1
                    if kind in ("speech", "deferred") and ts and sess.container_t0:
                        at = sess.container_t0 + float(ts.group(1))
                        step = SLICE_LEN / FPS
                        into = sess.spans if kind == "speech" else sess.wait_spans
                        if into and at - into[-1][1] <= step * 1.6:
                            into[-1][1] = at + step                # same run, extend it
                        else:
                            into.append([at, at + step])
                if "live: ready" in line and not sess.started_live:
                    sess.started_live = time.time()
                sess.log.append(line[:160])
                return
            if item["type"] == "file":
                name = _hls_sink(hls_dir, item, parts)
                if name and name.endswith(".m4s"):
                    sess.segments += 1
                init_ok = (hls_dir / "init.mp4").exists() and (hls_dir / "init.mp4").stat().st_size > 0
                if not sess.hls_url and init_ok and (hls_dir / "index.m3u8").exists() and sess.segments >= 2:
                    sess.hls_url = f"/hls/live/{sess.id}/index.m3u8"
                    sess.state = "live"
                    sess.log.append("streaming: live playlist ready")

        t0 = time.time()
        transient = 0          # consecutive client-side failures
        while True:
            if time.time() - t0 > RENDER_TIMEOUT_S + 600:
                call.cancel()
                raise TimeoutError("live session exceeded the maximum duration")
            if transient > LIVE_CLIENT_RETRIES:
                raise ConnectionError(f"컨테이너와 {transient}회 연속 통신 실패")
            try:
                items = fh.progress.get_many(20, block=True, timeout=1, partition=sess.id)
                transient = 0
            except stdqueue.Empty:
                items, transient = [], 0
            except Exception as exc:  # noqa: BLE001 — the session outlives a failed poll
                transient += 1
                if transient == 1 or transient % 10 == 0:
                    sess.log.append(f"client: {type(exc).__name__}: {str(exc)[:70]} — 재시도 {transient}")
                time.sleep(min(0.3 * transient, 3.0))
                continue
            if items:
                for it in items:
                    handle(it)
                continue
            try:
                stats = call.get(timeout=0)
                break
            except TimeoutError:
                pass               # still running
            except Exception as exc:  # noqa: BLE001
                transient += 1
                if transient == 1 or transient % 10 == 0:
                    sess.log.append(f"client: {type(exc).__name__}: {str(exc)[:70]} — 재시도 {transient}")
                time.sleep(min(0.3 * transient, 3.0))
        for it in fh.progress.get_many(100, block=False, partition=sess.id):
            handle(it)
        sess.stats = stats
        sess.log.append("container stats: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
        # keep a recording of the whole session
        if sess.segments:
            out = sess.dir / "video.mp4"
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(hls_dir / "index.m3u8"),
                                "-c", "copy", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
            if r.returncode == 0 and out.exists():
                sess.recording_url = f"/videos/live/{sess.id}.mp4"
        sess.state = "ended"
    except Exception as exc:  # noqa: BLE001
        sess.state = "error"
        sess.error = f"{type(exc).__name__}: {exc}"
        sess.log.append("ERROR " + sess.error)
    finally:
        sess.finished = time.time()
        (sess.dir / "session.json").write_text(json.dumps(sess.public(), ensure_ascii=False, indent=1))


_outbox: stdqueue.Queue = stdqueue.Queue()


def _put_async(item: dict, partition: str, label: str) -> None:
    """Hand a queue put to the sender thread instead of waiting on Modal here.

    A `progress.put` is a gRPC call, and a congested Modal client makes it slow
    or hang. Called straight from a route it holds an anyio worker thread for
    that whole time, and enough of those stall the whole studio. Order within a
    session is preserved because one thread drains this queue in FIFO order."""
    _outbox.put((item, partition, label))


def _outbox_worker() -> None:
    while True:
        item, partition, label = _outbox.get()
        try:
            fh.progress.put(item, block=False, partition=partition)
        except Exception as exc:  # noqa: BLE001
            print(f"outbox: {label} 실패: {type(exc).__name__}: {exc}", flush=True)


threading.Thread(target=_outbox_worker, daemon=True, name="modal-outbox").start()


def _live_push_pcm(sess: LiveSession, pcm: bytes, label: str, tts_s: float = 0.0) -> float:
    """Send raw 16 kHz mono PCM to the container in ≤800 KB parts (Queue items are
    capped at 1 MiB); the last part carries `final` so the tail gets padded."""
    seconds = len(pcm) / 2 / 16000
    part = 800_000
    parts = [pcm[j:j + part] for j in range(0, len(pcm), part)] or [b""]
    for j, chunk in enumerate(parts):
        _put_async({"type": "audio", "pcm": chunk, "final": j == len(parts) - 1},
                   f"{sess.id}:audio", f"audio {sess.id[:6]} {j + 1}/{len(parts)}")
    extra = f" (tts {tts_s:.1f}s)" if tts_s else " (tts 캐시)"
    sess.log.append(f"say: {seconds:.1f}s of speech pushed{extra} → \"{label[:40]}\"")
    return seconds


def _live_say_audio(sess: LiveSession, src: Path, name: str) -> None:
    """Speak an uploaded audio file in a live session."""
    sess.dir.mkdir(parents=True, exist_ok=True)
    n = len(sess.said)
    entry = {"text": name, "seconds": None, "at": time.time(), "source": "upload"}
    sess.said.append(entry)
    wav = sess.dir / f"say_{n:03d}.wav"
    seconds = _to_speech_wav(src, wav)
    with wave.open(str(wav), "rb") as r:
        pcm = r.readframes(r.getnframes())
    entry["seconds"] = round(_live_push_pcm(sess, pcm, name), 2)
    src.unlink(missing_ok=True)


TTS_CACHE = HERE / "tts-cache"
TTS_CACHE.mkdir(exist_ok=True)
_tts_locks: dict[str, threading.Lock] = {}
_tts_locks_guard = threading.Lock()


def _tts_pcm(text: str) -> tuple[bytes, float]:
    """16 kHz mono PCM for `text`, synthesized once and kept on disk.

    The interview questions repeat for every session and every cycle, and each
    `say -v "Yuna (Premium)"` costs ~2-3 s of process and voice start-up. Caching
    by (voice, text) turns that into one run per distinct sentence, ever, and
    removes the delay before the avatar starts a question.
    Returns (pcm, seconds spent synthesizing — 0 on a cache hit).
    """
    key = hashlib.sha256(f"{VOICE}|{text}".encode()).hexdigest()[:20]
    wav = TTS_CACHE / f"{key}.wav"
    with _tts_locks_guard:                 # two sessions asking the same question
        lock = _tts_locks.setdefault(key, threading.Lock())
    took = 0.0
    with lock:
        if not wav.exists():
            t0 = time.time()
            txt = TTS_CACHE / f"{key}.txt"
            txt.write_text(text, encoding="utf-8")
            tmp = wav.with_suffix(".tmp.wav")
            subprocess.run(["say", "-v", VOICE, "-o", str(tmp), "--data-format", "LEI16@16000",
                            "-f", str(txt)], check=True, capture_output=True, timeout=120)
            tmp.replace(wav)
            took = time.time() - t0
    with wave.open(str(wav), "rb") as r:
        return r.readframes(r.getnframes()), took


def _prewarm_tts() -> None:
    """Synthesize the question bank up front so the first question of a session is
    not waiting on `say`."""
    for q in INTERVIEW_QUESTIONS:
        try:
            _tts_pcm(q)
        except Exception:  # noqa: BLE001 — a missing voice must not stop the server
            return


def _live_say(sess: LiveSession, text: str) -> None:
    """Push a sentence to the container as speech. The audio comes from the TTS
    cache, so a repeated question costs nothing but the queue put."""
    entry = {"text": text, "seconds": None, "at": time.time()}
    sess.said.append(entry)
    pcm, tts_s = _tts_pcm(text)
    entry["seconds"] = round(_live_push_pcm(sess, pcm, text, tts_s=tts_s), 2)


# ----------------------------------------------------------------------------
# Pro final render: same speech.wav, avatar and seed through Renderer(model_type="pro"),
# which has its own container pool, so it runs alongside Lite streaming jobs.
pro_queue: deque[str] = deque()
pro_wake = threading.Event()
_renderer_pro = None


def renderer_pro(fresh: bool = False):
    global _renderer_pro
    if _renderer_pro is None or fresh:
        _renderer_pro = modal.Cls.from_name(fh.APP_NAME, "Renderer")(model_type="pro")
    return _renderer_pro


def _render_pro(job: Job) -> Path:
    job.pro_status = "render"
    job.pro_started = time.time()
    job.pro_chunks_done = 0
    out = job.dir / "video_pro.mp4"
    wav = job.dir / "speech.wav"
    if not wav.exists():
        raise FileNotFoundError("speech.wav is gone; re-create the job")
    image = _avatars().get(job.avatar)
    if image is None:
        raise FileNotFoundError(f"avatar '{job.avatar}' is no longer in {AVATARS_DIR.name}/")
    need = math.ceil(job.audio_seconds * FPS) + math.ceil(VIDEO_TAIL_S * FPS)
    job.pro_expected_chunks = max(1, math.ceil((need - PRO_MOTION) / PRO_SLICE_LEN))
    part = f"{job.id}:pro"
    job.pro_log.append(f"Modal: Renderer(pro).render, avatar {image.name}, {job.pro_expected_chunks} chunks × {PRO_SLICE_LEN}f (~{PRO_CHUNK_S}s each on L4)")
    try:
        call = renderer_pro().render.spawn(image.read_bytes(), wav.read_bytes(), part, seed=42, tail_seconds=VIDEO_TAIL_S)
    except Exception as exc:  # noqa: BLE001
        if not any(k in str(exc) for k in ("stopped", "disabled", "NotFound")):
            raise
        call = renderer_pro(fresh=True).render.spawn(image.read_bytes(), wav.read_bytes(), part, seed=42, tail_seconds=VIDEO_TAIL_S)
    t0 = time.time()

    def handle(line: object) -> None:
        if not isinstance(line, str):
            return
        m = _CHUNK_RE.search(line)
        if m:
            job.pro_chunks_done = int(m.group(1))
        if ".. running" in line and job.pro_chunks_done:
            return                                   # heartbeat noise once chunks are flowing
        job.pro_log.append(line[:160])

    while True:
        if time.time() - t0 > RENDER_TIMEOUT_S:
            call.cancel()
            raise TimeoutError(f"pro render exceeded {RENDER_TIMEOUT_S}s")
        try:
            items = fh.progress.get_many(20, block=True, timeout=1, partition=part)
        except stdqueue.Empty:
            items = []
        if items:
            for it in items:
                handle(it)
            continue
        try:
            mp4 = call.get(timeout=0)
            break
        except TimeoutError:
            pass
    for it in fh.progress.get_many(100, block=False, partition=part):
        handle(it)
    if not mp4:
        raise RuntimeError("pro render returned no video")
    out.write_bytes(mp4)
    return out


def _pro_worker() -> None:
    while True:
        pro_wake.wait()
        with lock:
            if not pro_queue:
                pro_wake.clear()
                continue
            job = jobs[pro_queue.popleft()]
        try:
            _render_pro(job)
            job.pro_status = "done"
            job.pro_log.append("done")
        except Exception as exc:  # noqa: BLE001
            job.pro_status = "error"
            job.pro_error = f"{type(exc).__name__}: {exc}"
            job.pro_log.append("ERROR " + job.pro_error)
        finally:
            job.pro_finished = time.time()
            job.persist()


threading.Thread(target=_pro_worker, daemon=True, name="studio-pro-worker").start()


# ----------------------------------------------------------------------------
# Warm state. `keep_warm` is what we asked for; `containers` is what Modal
# actually has running for the app right now (polled via the CLI, which is the
# only public way to list containers). The two disagree during spin-up and for
# `SCALEDOWN_S` after the last job — the UI shows both.
warm_state = {"keep_warm": False, "containers": 0, "ready": None, "checked": 0.0, "error": ""}
STATE_FILE = HERE / "warm.json"


READY_STALE_S = 60


def _fresh(flag: dict | None) -> dict | None:
    if not flag:
        return None
    return flag if time.time() - flag.get("beat", flag.get("since", 0)) < READY_STALE_S else None


def _poll_containers() -> None:
    try:
        r = subprocess.run([str(MODAL), "container", "list", "--json"], capture_output=True, text=True, timeout=30)
        rows = json.loads(r.stdout or "[]")
        warm_state["containers"] = sum(1 for c in rows if c.get("app_name") == fh.APP_NAME)
        # set by Renderer.load once the pipeline can take a call; it heartbeats
        # every 20 s, so a flag left behind by a replaced container goes stale
        warm_state["ready"] = _fresh(fh.state.get("ready:lite")) if warm_state["containers"] else None
        warm_state["error"] = ""
    except Exception as exc:  # noqa: BLE001
        warm_state["error"] = f"{type(exc).__name__}: {exc}"
    warm_state["checked"] = time.time()


def _container_poller() -> None:
    while True:
        _poll_containers()
        # `modal deploy` resets autoscaler overrides; if the switch says on but
        # nothing is running, re-apply (idempotent) instead of waiting for a restart
        if warm_state["keep_warm"] and not warm_state["containers"]:
            try:
                renderer().update_autoscaler(min_containers=1)
            except Exception as exc:  # noqa: BLE001
                warm_state["error"] = f"re-assert keep_warm failed: {type(exc).__name__}: {exc}"
        # cheap CLI call; faster while a container is expected to be changing state
        # poll fast while the container is between states (starting, loading, draining)
        settled = warm_state["keep_warm"] == bool(warm_state["ready"]) == bool(warm_state["containers"])
        time.sleep(60 if settled else 10)


def _set_keep_warm(on: bool) -> None:
    # min_containers=1 keeps one container resident indefinitely; 0 restores
    # scale-to-zero after SCALEDOWN_S. max stays 1 (see app.cls) either way.
    try:
        renderer().update_autoscaler(min_containers=1 if on else 0)
    except Exception as exc:  # noqa: BLE001
        if not any(k in str(exc) for k in ("stopped", "disabled", "NotFound")):
            raise
        renderer(fresh=True).update_autoscaler(min_containers=1 if on else 0)
    warm_state["keep_warm"] = on
    STATE_FILE.write_text(json.dumps({"keep_warm": on}))


# Restore the toggle across restarts and re-assert it on Modal: `update_autoscaler`
# overrides are reset by every `modal deploy`, so the studio (not Modal) is the
# source of truth for "keep warm" and re-applies it at boot.
try:
    warm_state["keep_warm"] = bool(json.loads(STATE_FILE.read_text()).get("keep_warm"))
except Exception:  # noqa: BLE001
    pass


def _reassert_keep_warm() -> None:
    if warm_state["keep_warm"]:
        try:
            _set_keep_warm(True)
        except Exception as exc:  # noqa: BLE001
            warm_state["error"] = f"re-assert keep_warm failed: {type(exc).__name__}: {exc}"


threading.Thread(target=_reassert_keep_warm, daemon=True, name="keep-warm-reassert").start()
threading.Thread(target=_container_poller, daemon=True, name="container-poller").start()


# ----------------------------------------------------------------------------
app = FastAPI(title="FlashHead Studio")

# Every route here is a sync `def`, so each in-flight request holds one of
# anyio's worker threads — 40 by default. Six live sessions pull HLS segments
# continuously while the page polls, and any handler that blocks holds its
# thread for as long as it blocks. When all 40 are held the studio stops
# answering *everything*, segments included, while looking idle. 200 gives the
# demo room; `_put_async` below keeps the blocking calls off these threads.
@app.on_event("startup")
async def _widen_threadpool() -> None:
    import anyio.to_thread

    anyio.to_thread.current_default_thread_limiter().total_tokens = 200


# `kill -USR1 <pid>` dumps every thread's stack into studio.log. Without this a
# wedged studio can only be guessed at: py-spy needs root on macOS.
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


class CreateJob(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    avatar: str = DEFAULT_AVATAR


class WarmBody(BaseModel):
    on: bool


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/live")
def live_page() -> FileResponse:
    """Dedicated screen for the concurrency demo: a grid of sessions plus a Gantt
    of who held the GPU when."""
    return FileResponse(HERE / "live.html")


@app.get("/api/config")
def config() -> dict:
    return {
        "voice": VOICE, "default_avatar": DEFAULT_AVATAR, "max_chars": MAX_CHARS, "mode": "once", "model": "lite",
        "gpu": fh.GPU, "scaledown_s": fh.SCALEDOWN_S, "app": fh.APP_NAME, "tail_s": VIDEO_TAIL_S,
        "audio_max_mb": AUDIO_MAX_BYTES // 1_000_000, "audio_max_seconds": AUDIO_MAX_SECONDS,
        "live_max_sessions": LIVE_MAX_SESSIONS, "live_speak_s": LIVE_SPEAK_S, "live_cycle_s": LIVE_CYCLE_S, "live_turns": LIVE_TURNS,
    }


@app.get("/api/avatars")
def list_avatars() -> list[dict]:
    return [{"id": stem, "name": stem, "url": f"/avatars/{p.name}"} for stem, p in _avatars().items()]


@app.post("/api/avatars")
async def add_avatar(file: UploadFile = File(...)) -> dict:
    """Add a portrait to the library. Re-encoded to PNG (EXIF orientation applied,
    longest side capped) so whatever the browser sends becomes a clean reference."""
    import io

    from PIL import Image, ImageOps, UnidentifiedImageError

    data = await file.read()
    if len(data) > 30_000_000:
        raise HTTPException(413, "30 MB 이하 이미지만 받습니다")
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(415, "이미지 파일로 읽을 수 없습니다") from exc
    if min(im.size) < AVATAR_MIN_SIDE:
        raise HTTPException(422, f"너무 작습니다 ({im.size[0]}×{im.size[1]}); 짧은 변 {AVATAR_MIN_SIDE}px 이상이어야 합니다")
    if max(im.size) > AVATAR_MAX_SIDE:
        im.thumbnail((AVATAR_MAX_SIDE, AVATAR_MAX_SIDE))
    stem = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", Path(file.filename or "avatar").stem).strip("-") or "avatar"
    existing = _avatars()
    candidate, n = stem, 2
    while candidate in existing:
        candidate, n = f"{stem}-{n}", n + 1
    dest = AVATARS_DIR / f"{candidate}.png"
    im.save(dest, "PNG")
    return {"id": candidate, "name": candidate, "url": f"/avatars/{dest.name}", "size": list(im.size)}


@app.get("/api/warm")
def warm_status() -> dict:
    # re-check staleness at request time so a cached flag expires between polls
    warm_state["ready"] = _fresh(warm_state["ready"])
    return warm_state


@app.post("/api/warm")
def warm_set(body: WarmBody) -> dict:
    try:
        _set_keep_warm(body.on)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    threading.Thread(target=_poll_containers, daemon=True).start()
    return warm_state


@app.post("/api/jobs")
def create(body: CreateJob) -> JSONResponse:
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "멘트가 비어 있습니다")
    if body.avatar not in _avatars():
        raise HTTPException(422, f"없는 아바타입니다: {body.avatar}")
    job = Job(id=uuid.uuid4().hex[:10], text=text, created=time.time(), avatar=body.avatar)
    with lock:
        jobs[job.id] = job
        queue.append(job.id)
        position = len(queue)
    wake.set()
    _log(job, f"queued (position {position})")
    return JSONResponse(job.public(), status_code=201)


@app.post("/api/jobs/audio")
async def create_from_audio(file: UploadFile = File(...), avatar: str = Form(DEFAULT_AVATAR)) -> JSONResponse:
    """Create a job from an uploaded audio file instead of typed text."""
    if avatar not in _avatars():
        raise HTTPException(422, f"없는 아바타입니다: {avatar}")
    data = await file.read()
    if not data:
        raise HTTPException(422, "빈 파일입니다")
    if len(data) > AUDIO_MAX_BYTES:
        raise HTTPException(413, f"{AUDIO_MAX_BYTES // 1_000_000} MB 이하 파일만 받습니다")
    name = Path(file.filename or "audio").name
    job = Job(id=uuid.uuid4().hex[:10], text=name, created=time.time(), avatar=avatar, source="upload")
    job.dir.mkdir(parents=True, exist_ok=True)
    src = job.dir / f"upload{Path(name).suffix.lower() or '.bin'}"
    src.write_bytes(data)
    try:
        seconds = _probe_seconds(src)
        if seconds > AUDIO_MAX_SECONDS:
            raise ValueError(f"오디오가 너무 깁니다 ({seconds:.0f}s); {AUDIO_MAX_SECONDS:.0f}s 이하만 받습니다")
    except ValueError as exc:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc
    with lock:
        jobs[job.id] = job
        queue.append(job.id)
        position = len(queue)
    wake.set()
    _log(job, f"queued (position {position}) — 업로드 오디오 {name} ({len(data)/1e6:.1f} MB)")
    return JSONResponse(job.public(), status_code=201)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.public() for j in sorted(jobs.values(), key=lambda j: j.created, reverse=True)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.public()


class LiveCreate(BaseModel):
    avatar: str = DEFAULT_AVATAR
    model: str = "lite"
    auto: bool = True


class LiveSay(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class LiveAuto(BaseModel):
    on: bool


class PlaygroundNew(BaseModel):
    name: str = ""


@app.post("/api/live")
def live_start(body: LiveCreate) -> JSONResponse:
    if body.avatar not in _avatars():
        raise HTTPException(422, f"없는 아바타입니다: {body.avatar}")
    if body.model not in ("lite", "pro"):
        raise HTTPException(422, "model은 lite 또는 pro")
    active = [s for s in live_sessions.values() if s.state in ("starting", "live", "ending")]
    if len(active) >= LIVE_MAX_SESSIONS:
        raise HTTPException(409, f"동시 세션은 {LIVE_MAX_SESSIONS}개까지입니다 (컨테이너 1대 기준)")
    sess = LiveSession(id=uuid.uuid4().hex[:10], avatar=body.avatar, created=time.time(),
                       model=body.model, auto=body.auto, playground=current_playground)
    live_sessions[sess.id] = sess
    sess.log.append(f"session created (auto={body.auto})")
    threading.Thread(target=_live_consumer, args=(sess,), daemon=True, name=f"live-{sess.id}").start()
    if sess.auto:
        threading.Thread(target=_auto_driver, args=(sess,), daemon=True, name=f"auto-{sess.id}").start()
    return JSONResponse(sess.public(), status_code=201)


@app.get("/api/live")
def live_list(playground: str = "") -> list[dict]:
    """Sessions of one demo run — the current one unless another is asked for."""
    pg = playground or current_playground
    return [s.public() for s in sorted(live_sessions.values(), key=lambda s: s.created, reverse=True)
            if s.playground == pg]


@app.get("/api/playgrounds")
def playground_list() -> dict:
    items = sorted((p.public() for p in playgrounds.values()), key=lambda p: p["created"], reverse=True)
    return {"current": current_playground, "items": items}


@app.post("/api/playgrounds")
def playground_new(body: PlaygroundNew) -> dict:
    """Start a fresh run: the screen empties, nothing is deleted."""
    mine = [s for s in live_sessions.values() if s.playground == current_playground]
    if any(s.state in ("starting", "live") for s in mine):
        raise HTTPException(409, "진행 중인 세션이 있습니다. 먼저 종료해 주세요")
    if not mine and current_playground in playgrounds and not body.name.strip():
        return playgrounds[current_playground].public()      # already empty: no need for another
    return _new_playground(body.name.strip()).public()


@app.post("/api/playgrounds/{pid}/select")
def playground_select(pid: str) -> dict:
    """Load an earlier run. New sessions then join it too."""
    global current_playground
    if pid not in playgrounds:
        raise HTTPException(404, "no such playground")
    if any(s.state in ("starting", "live") for s in live_sessions.values() if s.playground == current_playground):
        raise HTTPException(409, "진행 중인 세션이 있습니다. 먼저 종료해 주세요")
    current_playground = pid
    _save_playgrounds()
    return playgrounds[pid].public()


@app.get("/api/live/{sid}")
def live_get(sid: str) -> dict:
    sess = live_sessions.get(sid)
    if not sess:
        raise HTTPException(404, "no such session")
    return sess.public()


@app.post("/api/live/{sid}/say")
def live_say(sid: str, body: LiveSay) -> dict:
    sess = live_sessions.get(sid)
    if not sess:
        raise HTTPException(404, "no such session")
    if sess.state not in ("starting", "live"):
        raise HTTPException(409, "세션이 끝났습니다")
    text = body.text.strip()

    def run() -> None:
        try:
            _live_say(sess, text)
        except Exception as exc:  # noqa: BLE001
            sess.log.append(f"say failed: {type(exc).__name__}: {exc}")

    threading.Thread(target=run, daemon=True).start()
    sess.log.append(f"say queued: \"{text[:40]}\"")
    return sess.public()


@app.post("/api/live/{sid}/say_audio")
async def live_say_audio(sid: str, file: UploadFile = File(...)) -> dict:
    sess = live_sessions.get(sid)
    if not sess:
        raise HTTPException(404, "no such session")
    if sess.state not in ("starting", "live"):
        raise HTTPException(409, "세션이 끝났습니다")
    data = await file.read()
    if not data:
        raise HTTPException(422, "빈 파일입니다")
    if len(data) > AUDIO_MAX_BYTES:
        raise HTTPException(413, f"{AUDIO_MAX_BYTES // 1_000_000} MB 이하 파일만 받습니다")
    name = Path(file.filename or "audio").name
    sess.dir.mkdir(parents=True, exist_ok=True)
    tmp = sess.dir / f"upload_{len(sess.said):03d}{Path(name).suffix.lower() or '.bin'}"
    tmp.write_bytes(data)
    try:
        _probe_seconds(tmp)
    except ValueError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc

    def run() -> None:
        try:
            _live_say_audio(sess, tmp, name)
        except Exception as exc:  # noqa: BLE001
            sess.log.append(f"say(audio) failed: {type(exc).__name__}: {exc}")

    threading.Thread(target=run, daemon=True).start()
    sess.log.append(f"say queued (audio): {name} ({len(data)/1e6:.1f} MB)")
    return sess.public()


@app.post("/api/live/{sid}/auto")
def live_auto(sid: str, body: LiveAuto) -> dict:
    sess = live_sessions.get(sid)
    if not sess:
        raise HTTPException(404, "no such session")
    was, sess.auto = sess.auto, body.on
    if body.on and not was and sess.state in ("starting", "live"):
        threading.Thread(target=_auto_driver, args=(sess,), daemon=True, name=f"auto-{sid}").start()
    return sess.public()


@app.post("/api/live/{sid}/stop")
def live_stop(sid: str) -> dict:
    sess = live_sessions.get(sid)
    if not sess:
        raise HTTPException(404, "no such session")
    _live_end(sess)
    return sess.public()


@app.delete("/api/live/{sid}")
def live_forget(sid: str) -> dict:
    """Drop one ended session from the demo screen. Files on disk are kept."""
    sess = live_sessions.get(sid)
    if not sess:
        raise HTTPException(404, "no such session")
    if sess.state in ("starting", "live", "ending"):
        raise HTTPException(409, "진행 중인 세션입니다")
    live_sessions.pop(sid, None)
    # also stop it coming back on the next restart; the recording stays on disk
    meta = sess.dir / "session.json"
    if meta.exists():
        meta.rename(sess.dir / "session.hidden.json")
    return {"removed": sid}


@app.get("/hls/live/{sid}/{name}")
def hls_live_file(sid: str, name: str) -> FileResponse:
    if not _HLS_NAME.match(name):
        raise HTTPException(404, "no such file")
    p = LIVE_DIR / sid / "hls" / name
    if not p.exists():
        raise HTTPException(404, "not streamed yet")
    return FileResponse(p, media_type=_HLS_MEDIA[p.suffix], headers={"Cache-Control": "no-store"})


@app.get("/videos/live/{sid}.mp4")
def video_live(sid: str) -> FileResponse:
    p = LIVE_DIR / sid / "video.mp4"
    if not p.exists():
        raise HTTPException(404, "no recording")
    return FileResponse(p, media_type="video/mp4")


@app.post("/api/jobs/{job_id}/pro")
def start_pro(job_id: str) -> dict:
    """Queue a Pro final render of a finished job (same speech, avatar, seed)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job.status != "done":
        raise HTTPException(409, "Lite 렌더가 끝난 뒤에 요청할 수 있습니다")
    if job.pro_status in ("queued", "render"):
        raise HTTPException(409, "이미 Pro 렌더가 진행 중입니다")
    job.pro_status, job.pro_error, job.pro_chunks_done = "queued", "", 0
    job.pro_started = job.pro_finished = 0.0
    job.pro_log.clear()
    with lock:
        pro_queue.append(job.id)
        position = len(pro_queue)
    pro_wake.set()
    job.pro_log.append(f"queued (position {position})")
    return job.public()


@app.post("/api/jobs/{job_id}/reveal")
def reveal(job_id: str, which: str = "lite") -> dict:
    """Select the rendered file in Finder. Local demo tool, so shelling out to
    `open -R` is the whole feature."""
    p = JOBS_DIR / job_id / ("video_pro.mp4" if which == "pro" else "video.mp4")
    if not p.exists():
        raise HTTPException(404, "not rendered")
    subprocess.Popen(["open", "-R", str(p)])
    return {"revealed": str(p)}


@app.get("/hls/{job_id}/{name}")
def hls_file(job_id: str, name: str) -> FileResponse:
    if not _HLS_NAME.match(name):
        raise HTTPException(404, "no such file")
    p = JOBS_DIR / job_id / "hls" / name
    if not p.exists():
        raise HTTPException(404, "not streamed yet")
    return FileResponse(p, media_type=_HLS_MEDIA[p.suffix], headers={"Cache-Control": "no-store"})


@app.get("/videos/pro/{job_id}.mp4")
def video_pro(job_id: str) -> FileResponse:
    p = JOBS_DIR / job_id / "video_pro.mp4"
    if not p.exists():
        raise HTTPException(404, "not rendered")
    return FileResponse(p, media_type="video/mp4")


@app.get("/videos/{job_id}.mp4")
def video(job_id: str) -> FileResponse:
    p = JOBS_DIR / job_id / "video.mp4"
    if not p.exists():
        raise HTTPException(404, "not rendered")
    return FileResponse(p, media_type="video/mp4")


# playgrounds first: sessions restored below are filed under theirs
try:
    _saved = json.loads(PLAYGROUNDS_FILE.read_text())
    for it in _saved.get("items", []):
        playgrounds[it["id"]] = Playground(id=it["id"], name=it["name"], created=it["created"])
    current_playground = _saved.get("current", "")
except Exception:  # noqa: BLE001
    pass

# reload finished live sessions too: the demo screen keeps their recording and
# their Gantt row, so a server restart must not erase the history
for meta in LIVE_DIR.glob("*/session.json"):
    try:
        d = json.loads(meta.read_text())
        if d["id"] in live_sessions:
            continue
        s = LiveSession(id=d["id"], avatar=d.get("avatar", DEFAULT_AVATAR), created=d.get("created", 0),
                        model=d.get("model", "lite"))
        s.state = "ended" if d.get("state") in ("ended", "ending", "live", "starting") else d.get("state", "ended")
        s.error = d.get("error", "")
        s.chunks = d.get("chunks", 0); s.speech_chunks = d.get("speech_chunks", 0)
        s.segments = d.get("segments", 0); s.turns = d.get("turns", 0)
        s.started_live = d.get("started_live", 0); s.finished = d.get("finished", 0)
        s.spans = [list(x) for x in d.get("spans", [])]
        s.wait_spans = [list(x) for x in d.get("wait_spans", [])]
        s.said = d.get("said", [])
        s.playground = d.get("playground", "")
        s.recording_url = d.get("recording_url", "") if (s.dir / "video.mp4").exists() else ""
        s.log.extend(d.get("log", [])[-30:])
        live_sessions[s.id] = s
    except Exception:  # noqa: BLE001
        continue

# sessions from before playgrounds existed keep their history under one heading
_orphans = [s for s in live_sessions.values() if not s.playground]
if _orphans:
    _legacy = Playground(id="legacy00", name="이전 기록", created=min(s.created for s in _orphans))
    playgrounds.setdefault(_legacy.id, _legacy)
    for s in _orphans:
        s.playground = _legacy.id
if current_playground not in playgrounds:
    _new_playground()
else:
    _save_playgrounds()


# reload finished jobs from disk so a server restart keeps the gallery
for meta in JOBS_DIR.glob("*/job.json"):
    try:
        d = json.loads(meta.read_text())
        if d.get("status") == "done":
            j = Job(id=d["id"], text=d["text"], created=d["created"], status="done", avatar=d.get("avatar", DEFAULT_AVATAR),
                    source=d.get("source", "tts"),
                    audio_seconds=d.get("audio_seconds", 0), expected_chunks=d.get("expected_chunks", 0),
                    chunks_done=d.get("chunks_done", 0), started=d.get("started", 0), finished=d.get("finished", 0))
            j.log.extend(d.get("log", [])[-50:])
            if (j.dir / "video_pro.mp4").exists():
                j.pro_status = "done"
                j.pro_started, j.pro_finished = d.get("pro_started", 0), d.get("pro_finished", 0)
                j.pro_chunks_done = j.pro_expected_chunks = d.get("pro_expected_chunks", 0)
                j.pro_log.extend(d.get("pro_log", [])[-30:])
            jobs[j.id] = j
    except Exception:  # noqa: BLE001
        continue

# after the definitions above, so the bank is ready before the first session
threading.Thread(target=_prewarm_tts, daemon=True, name="tts-prewarm").start()

app.mount("/avatars", StaticFiles(directory=AVATARS_DIR), name="avatars")
app.mount("/static", StaticFiles(directory=HERE), name="static")
