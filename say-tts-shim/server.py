"""OpenAI-compatible TTS shim backed by the macOS `say` command.

OpenTalking's `openai_compatible` TTS provider posts to `{base_url}/audio/speech`
and expects 16-bit PCM WAV back. `say --data-format=LEI16@16000` produces exactly
that, so no transcoding is needed anywhere in the chain.

    ./.venv/bin/python server.py            # listens on 127.0.0.1:9999
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

SAMPLE_RATE = 16_000
DATA_FORMAT = f"LEI16@{SAMPLE_RATE}"
BASE_WPM = 180  # `say -r` baseline that maps OpenAI's speed=1.0
SAY_TIMEOUT_S = 120.0

DEFAULT_VOICE = os.environ.get("SAY_SHIM_DEFAULT_VOICE", "Yuna")
# OpenTalking normalizes voice ids before sending them and drops the
# parenthesized variant, so "Yuna (Premium)" arrives as "Yuna". When this is
# on, an exact match is upgraded to its installed Premium/Enhanced sibling.
PREFER_HQ_VARIANT = os.environ.get("SAY_SHIM_PREFER_HQ_VARIANT", "1").strip().lower() not in {"0", "false", "no", "off"}
HQ_SUFFIXES = ("(Premium)", "(Enhanced)")

logger = logging.getLogger("say-shim")

app = FastAPI(title="macOS say TTS shim")


class SpeechRequest(BaseModel):
    model: str = ""
    input: str = ""
    voice: str = ""
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


def _list_voices() -> dict[str, str]:
    """Map lowercased macOS voice name -> BCP-47-ish locale (e.g. "ko_KR")."""
    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=15, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not enumerate `say` voices: %s", exc)
        return {}

    voices: dict[str, str] = {}
    for line in out.splitlines():
        head = line.split("#", 1)[0].strip()
        if not head:
            continue
        # "Yuna (Premium)      ko_KR" -> name is everything before the locale token
        match = re.match(r"^(.*?)\s+([a-z]{2}(?:[-_][A-Za-z0-9]+)*)$", head)
        if not match:
            continue
        name, locale = match.group(1).strip(), match.group(2).replace("-", "_")
        if name:
            voices[name.lower()] = locale
    return voices


VOICES = _list_voices()


def _voice_for_locale(locale: str) -> str | None:
    """First installed voice whose locale matches, preferring non-Premium names."""
    target = locale.replace("-", "_").lower()
    matches = [name for name, loc in VOICES.items() if loc.lower() == target]
    if not matches:
        # fall back to the language subtag, so "ko-KR" still finds a "ko_KR" voice
        lang = target.split("_", 1)[0]
        matches = [name for name, loc in VOICES.items() if loc.lower().split("_", 1)[0] == lang]
    if not matches:
        return None
    return sorted(matches, key=lambda n: ("(" in n, len(n), n))[0]


def _upgrade_to_hq(name: str) -> str:
    """Swap a base voice for its Premium/Enhanced sibling when one is installed."""
    if not PREFER_HQ_VARIANT or any(suffix in name for suffix in HQ_SUFFIXES):
        return name
    for suffix in HQ_SUFFIXES:
        candidate = f"{name} {suffix}"
        if candidate.lower() in VOICES:
            logger.info("voice %r -> %r (higher-quality variant)", name, candidate)
            return candidate
    return name


def _resolve_voice(requested: str) -> str:
    """Map an OpenTalking voice id onto an installed macOS voice."""
    name = (requested or "").strip()
    if not name or name.lower() in {"default", "alloy"}:
        return _upgrade_to_hq(DEFAULT_VOICE)

    if name.lower() in VOICES:
        return _upgrade_to_hq(name)

    # Edge/Azure style ids such as "ko-KR-SunHiNeural" carry the locale up front.
    locale_match = re.match(r"^([a-z]{2}[-_][A-Za-z]{2})\b", name)
    if locale_match:
        resolved = _voice_for_locale(locale_match.group(1))
        if resolved:
            logger.info("voice %r -> %r (locale match)", name, resolved)
            return _upgrade_to_hq(resolved)

    logger.info("voice %r is not installed; using %r", name, DEFAULT_VOICE)
    return _upgrade_to_hq(DEFAULT_VOICE)


async def _synthesize(text: str, voice: str, speed: float) -> bytes:
    """Run `say` and return the WAV bytes it wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        text_path = Path(tmp) / "input.txt"
        wav_path = Path(tmp) / "out.wav"
        text_path.write_text(text, encoding="utf-8")

        argv = ["say", "-v", voice, "-o", str(wav_path), "--data-format", DATA_FORMAT]
        if abs(speed - 1.0) > 1e-3:
            argv += ["-r", str(max(1, round(BASE_WPM * speed)))]
        argv += ["-f", str(text_path)]

        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=SAY_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            raise HTTPException(status_code=504, detail="say timed out") from None

        if proc.returncode != 0:
            detail = stderr.decode("utf-8", "ignore").strip() or f"say exited {proc.returncode}"
            raise HTTPException(status_code=500, detail=detail)
        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="say produced no audio")
        return wav_path.read_bytes()


def _wav_to_pcm(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.readframes(wf.getnframes())


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "default_voice": DEFAULT_VOICE,
        "installed_voices": len(VOICES),
        "prefer_hq_variant": PREFER_HQ_VARIANT,
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/v1/voices")
async def voices() -> dict[str, object]:
    """Not part of the OpenAI API; handy for checking what `say` offers."""
    return {"voices": [{"name": n, "locale": loc} for n, loc in sorted(VOICES.items())]}


@app.get("/v1/models")
async def models() -> dict[str, object]:
    return {"object": "list", "data": [{"id": "macos-say", "object": "model"}]}


class ChatMessage(BaseModel):
    role: str = ""
    content: str = ""


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest) -> Response:
    """Echo "LLM": speaks the user's text back verbatim.

    OpenTalking's realtime path always runs text through an LLM, so there is no
    built-in way to make the avatar say an exact sentence. Echoing turns
    POST /sessions/{id}/speak into a verbatim "say this" command, with no API
    key and no model download. Point OPENTALKING_LLM_BASE_URL at a real
    provider (or a local Ollama) when you want actual conversation.
    """
    text = _last_user_text(body.messages)
    created = int(__import__("time").time())
    model = body.model or "echo"

    if not body.stream:
        return JSONResponse(
            {
                "id": "chatcmpl-echo",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                ],
            }
        )

    async def event_stream():
        # Chunked so the pipeline's incremental TTS opener behaves as it would
        # with a real streaming model.
        step = 12
        pieces = [text[i : i + step] for i in range(0, len(text), step)] or [""]
        for piece in pieces:
            chunk = {
                "id": "chatcmpl-echo",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        done = {
            "id": "chatcmpl-echo",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(file: UploadFile | None = None) -> dict[str, str]:
    """Stub so OpenTalking's session key-gate accepts `openai_compatible` STT.

    We never speak into the mic on this path; sessions are driven by POST
    /sessions/{id}/speak. Returning empty text keeps the gate happy without
    pulling down a local ASR model.
    """
    if file is not None:
        await file.read()
    return {"text": ""}


@app.post("/v1/audio/speech")
async def audio_speech(body: SpeechRequest) -> Response:
    text = (body.input or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="input is required")

    voice = _resolve_voice(body.voice)
    wav = await _synthesize(text, voice, body.speed)
    logger.info("spoke %d chars as %r (requested %r)", len(text), voice, body.voice)

    fmt = (body.response_format or "wav").strip().lower()
    if fmt == "pcm":
        return Response(content=_wav_to_pcm(wav), media_type="application/octet-stream")
    if fmt not in {"wav", ""}:
        # `say` only emits WAV here; returning it labelled honestly beats guessing.
        logger.info("response_format=%r requested; returning wav", fmt)
    return Response(content=wav, media_type="audio/wav")


@app.exception_handler(HTTPException)
async def _http_error(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail}})


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(
        app,
        host=os.environ.get("SAY_SHIM_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAY_SHIM_PORT", "9999")),
        log_level="info",
    )
