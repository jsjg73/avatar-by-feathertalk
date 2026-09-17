#!/usr/bin/env python
"""Does a talking-head model actually distinguish vowels, or only jaw opening?

Generates isolated vowels with macOS `say`, renders each through OpenTalking's
video-creation API, then compares the mean mouth shape per vowel against the
frame-to-frame scatter within a single vowel. A pair scoring below 1.0 is not
resolved by the model at all.

    python viseme_test.py --model quicktalk --avatar <id>

Two traps this script exists to avoid:
  * Locate the mouth from the peak of (output - template) motion and LOOK at
    the saved crop. A crop that lands on the chin makes every vowel except the
    open one collapse together, which reads as a model failure.
  * Threshold globally, never per frame. A per-frame median normalises the very
    signal being measured away.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
import requests

VOWELS = {
    "a": ("아", "open"),
    "e": ("에", "mid"),
    "i": ("이", "spread"),
    "o": ("오", "rounded-mid"),
    "u": ("우", "rounded-small"),
}


def say(text: str, out: Path, voice: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(" ".join([text] * 6))
        src = f.name
    subprocess.run(
        ["say", "-v", voice, "-o", str(out), "--data-format", "LEI16@16000", "-f", src],
        check=True,
    )


def render(api: str, model: str, avatar: str, wav: Path, title: str) -> str:
    with wav.open("rb") as fh:
        r = requests.post(
            f"{api}/video-creation/jobs",
            data={"model": model, "avatar_id": avatar, "audio_source": "upload",
                  "title": title, "execution_mode": "sync"},
            files={"audio_file": (wav.name, fh, "audio/wav")},
            timeout=1800,
        )
    r.raise_for_status()
    body = r.json()
    if "export_video" not in body:
        raise SystemExit(f"render failed: {json.dumps(body, ensure_ascii=False)[:400]}")
    return body["export_video"]["path"]


def activity(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    prev = acc = None
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d = np.abs(g - prev)
            acc = d if acc is None else acc + d
            n += 1
        prev = g
    cap.release()
    return acc / max(n, 1)


def find_mouth(rendered: str, template: str | None, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Peak of (output - template) motion; falls back to raw output motion."""
    a = activity(rendered)
    m = a if template is None else np.clip(a - activity(template), 0, None)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    _, _, _, (cx, cy) = cv2.minMaxLoc(m)
    w, h = size
    return cx - w // 2, cy - h // 2, w, h


def shapes(path: str, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    cap = cv2.VideoCapture(path)
    rows = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        roi = cv2.cvtColor(fr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        rows.append(((roi - roi.mean()) / (roi.std() + 1e-6)).reshape(-1))
    cap.release()
    return np.asarray(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8210")
    p.add_argument("--model", default="quicktalk")
    p.add_argument("--avatar", required=True)
    p.add_argument("--voice", default="Yuna (Premium)")
    p.add_argument("--template", help="avatar source/template video, for a cleaner mouth-locate")
    p.add_argument("--out", type=Path, default=Path("viseme_out"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rendered = {}
    for key, (text, _) in VOWELS.items():
        wav = args.out / f"v_{key}.wav"
        say(text, wav, args.voice)
        rendered[key] = render(args.api, args.model, args.avatar, wav, f"viseme-{key}")
        print(f"  {text} ({key}) rendered")

    box = find_mouth(rendered["a"], args.template, (64, 44))
    cap = cv2.VideoCapture(rendered["a"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ok, fr = cap.read()
    cap.release()
    x, y, w, h = box
    if ok:
        cv2.imwrite(str(args.out / "crop_check.png"),
                    cv2.resize(fr[y:y + h, x:x + w], (w * 6, h * 6), interpolation=cv2.INTER_NEAREST))
    print(f"\nmouth box {box} -> {args.out/'crop_check.png'}  (LOOK AT THIS before trusting the numbers)")

    data = {}
    for key in VOWELS:
        X = shapes(rendered[key], box)
        data[key] = X[int(len(X) * 0.25):int(len(X) * 0.85)]
    cent = {k: v.mean(0) for k, v in data.items()}
    noise = float(np.mean([np.linalg.norm(v - cent[k], axis=1).mean() for k, v in data.items()]))

    keys = list(VOWELS)
    D = np.array([[float(np.linalg.norm(cent[a] - cent[b])) for b in keys] for a in keys])
    print(f"\nwithin-vowel noise floor: {noise:.2f}\n")
    print("pair                                    distance   vs noise")
    iu = np.triu_indices(len(keys), 1)
    for i, j in zip(*iu):
        a, b = keys[i], keys[j]
        ratio = D[i, j] / noise
        flag = "  UNRESOLVED" if ratio < 1.0 else ""
        print(f"  {VOWELS[a][0]} {VOWELS[a][1]:<13} vs {VOWELS[b][0]} {VOWELS[b][1]:<13} "
              f"{D[i, j]:8.2f} {ratio:9.2f}{flag}")

    no_a = [D[i, j] for i, j in zip(*iu) if "a" not in (keys[i], keys[j])]
    print(f"\njaw opening (아 vs rest) : {D[0, 1:].mean() / noise:.2f}x noise")
    print(f"lip shape (에이오우 pairs): {np.mean(no_a) / noise:.2f}x noise")
    print("\n-> below 1.0 on the lip-shape line means the model encodes jaw opening only.")

    row = [cv2.resize(np.clip(cent[k].reshape(h, w) * 40 + 128, 0, 255).astype(np.uint8),
                      (w * 5, h * 5), interpolation=cv2.INTER_NEAREST) for k in keys]
    cv2.imwrite(str(args.out / "vowel_shapes.png"), np.hstack(row))
    print(f"wrote {args.out/'vowel_shapes.png'}")


if __name__ == "__main__":
    sys.exit(main())
