"""Local orchestrator for the FeatherTalk live-conversation demo.

Serves studio_live.html, and on POST /chat: sends the user's text to OpenAI,
synthesizes the reply with macOS `say`, converts to 16kHz mono WAV, and
forwards that WAV to the live_server.py running on the rented GPU box
(reachable at BOX_URL, normally through an SSH -L tunnel to localhost:8600).

Usage:
    ./.venv_orchestrator/bin/python studio_live_server.py
"""

import os
import subprocess
import time

import requests
from flask import Flask, request, jsonify, send_file

HERE = os.path.dirname(os.path.abspath(__file__))
BOX_URL = os.environ.get("FEATHERTALK_BOX_URL", "http://116.127.115.27:51853")
OPENAI_KEY_PATH = os.path.expanduser("~/.openai_key")
VOICE = "Yuna (Premium)"

SYSTEM_PROMPT = (
    "당신은 친절하고 자연스러운 AI 면접관입니다. 사용자의 말에 한국어로, "
    "짧고 자연스럽게(1~2문장) 답하세요. 실제 사람이 대화하듯 답하고, "
    "너무 formal하거나 딱딱하지 않게 답하세요."
)

app = Flask(__name__)


def _openai_reply(user_text: str) -> str:
    key = open(OPENAI_KEY_PATH).read().strip()
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 150,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _synthesize_wav(text: str) -> str:
    ts = time.time_ns()
    txt_path = f"/tmp/live_say_{ts}.txt"
    aiff_path = f"/tmp/live_say_{ts}.aiff"
    wav_path = f"/tmp/live_say_{ts}.wav"
    with open(txt_path, "w") as f:
        f.write(text)
    subprocess.run(["say", "-v", VOICE, "-f", txt_path, "-o", aiff_path], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", aiff_path, "-ar", "16000", "-ac", "1", wav_path],
                   check=True, capture_output=True)
    os.remove(txt_path)
    os.remove(aiff_path)
    return wav_path


@app.route("/")
def index():
    return send_file(os.path.join(HERE, "studio_live.html"))


@app.route("/debug_frame", methods=["POST"])
def debug_frame():
    import base64
    data_url = request.json.get("data", "")
    tag = request.json.get("tag", "")
    b64 = data_url.split(",", 1)[1]
    name = f"debug_frame_{tag}.jpg" if tag else "debug_frame.jpg"
    with open(f"/tmp/{name}", "wb") as f:
        f.write(base64.b64decode(b64))
    return jsonify({"ok": True})


@app.route("/speak_fixed", methods=["POST"])
def speak_fixed():
    """Bypass OpenAI + TTS entirely -- send a pre-synthesized line straight to
    the box, so the measured latency is just upload + queueing + generation +
    HLS delivery, with no LLM/TTS jitter mixed in."""
    idx = request.json.get("idx", 0)
    wav_path = os.path.join(HERE, "fixed_lines", f"line_{idx}.wav")
    t0 = time.time()
    with open(wav_path, "rb") as f:
        r = requests.post(f"{BOX_URL}/speak", files={"file": ("speak.wav", f, "audio/wav")}, timeout=30)
    t1 = time.time()
    r.raise_for_status()
    print(f"[speak_fixed] idx={idx} upload_s={t1 - t0:.2f} sent_at={t0:.3f}", flush=True)
    return jsonify({"queued": r.json(), "upload_s": round(t1 - t0, 2), "sent_at": t0})


@app.route("/speak_text", methods=["POST"])
def speak_text():
    """Speak the user's own typed text verbatim -- macOS `say` -> box /speak,
    with no OpenAI call in between (unlike /chat, which replies to the text
    instead of reading it aloud)."""
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    t0 = time.time()
    wav_path = _synthesize_wav(text)
    t1 = time.time()
    with open(wav_path, "rb") as f:
        r = requests.post(f"{BOX_URL}/speak", files={"file": ("speak.wav", f, "audio/wav")}, timeout=30)
    t2 = time.time()
    os.remove(wav_path)
    r.raise_for_status()

    timing = {"tts_s": round(t1 - t0, 2), "upload_s": round(t2 - t1, 2), "total_s": round(t2 - t0, 2)}
    print(f"[speak_text] {timing}", flush=True)
    return jsonify({"queued": r.json(), "timing": timing})


@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.json.get("message", "").strip()
    if not user_text:
        return jsonify({"error": "empty message"}), 400

    t0 = time.time()
    reply = _openai_reply(user_text)
    t1 = time.time()
    wav_path = _synthesize_wav(reply)
    t2 = time.time()

    with open(wav_path, "rb") as f:
        r = requests.post(f"{BOX_URL}/speak", files={"file": ("speak.wav", f, "audio/wav")}, timeout=30)
    t3 = time.time()
    os.remove(wav_path)
    r.raise_for_status()

    timing = {"openai_s": round(t1 - t0, 2), "tts_s": round(t2 - t1, 2), "upload_s": round(t3 - t2, 2), "total_s": round(t3 - t0, 2)}
    print(f"[chat] {timing}", flush=True)
    return jsonify({"reply": reply, "queued": r.json(), "timing": timing})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
