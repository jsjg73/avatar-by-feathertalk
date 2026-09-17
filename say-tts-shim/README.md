# macOS `say` → OpenTalking

Connects the macOS speech synthesizer to OpenTalking's lip-sync avatars, with no
changes to the OpenTalking repo. `server.py` is a small OpenAI-compatible shim
that OpenTalking talks to as if it were a cloud provider.

## What the shim serves

| Endpoint | Why it exists |
|---|---|
| `POST /v1/audio/speech` | The actual TTS. Runs `say --data-format=LEI16@16000`, which already returns the 16-bit 16 kHz mono PCM WAV that OpenTalking's `openai_compatible` provider expects — nothing transcodes. |
| `POST /v1/chat/completions` | Echo "LLM". OpenTalking's realtime path always routes text through an LLM, so this is what makes `POST /sessions/{id}/speak` say your text *verbatim*. Point `OPENTALKING_LLM_BASE_URL` at a real provider (or a local Ollama) when you want actual conversation. |
| `POST /v1/audio/transcriptions` | Stub returning `{"text": ""}`. Session creation refuses to start unless the configured STT provider has an API key; this satisfies that gate without downloading a local ASR model. |
| `GET /health`, `GET /v1/voices` | Diagnostics. `/v1/voices` lists what `say` actually has installed. |

Voice ids are resolved leniently: an exact macOS voice name wins, otherwise an
Edge/Azure-style id like `ko-KR-SunHiNeural` is matched by locale (→ `Yuna`),
otherwise `SAY_SHIM_DEFAULT_VOICE` (default `Yuna (Premium)`).

**Premium/Enhanced upgrade.** OpenTalking normalizes voice ids before sending
them and drops the parenthesized variant, so `"Yuna (Premium)"` configured in
`.env` arrives at the shim as plain `"Yuna"` — the setting is stored correctly,
it just does not survive the trip. `SAY_SHIM_PREFER_HQ_VARIANT` (on by default)
therefore upgrades a resolved voice to its installed `(Premium)` or
`(Enhanced)` sibling. Set it to `0` to use exactly what was asked for. Every
call logs what it settled on:

```
voice 'Yuna' -> 'Yuna (Premium)' (higher-quality variant)
spoke 30 chars as 'Yuna (Premium)' (requested 'Yuna')
```

For the offline video path the shim is not involved at all — name the voice
directly in `say -v "Yuna (Premium)"`.

## Run

```bash
./shim.sh start      # also: stop | restart | status
```

Then, from the OpenTalking repo:

```bash
# Phase 1 — Light2D avatar, browser-rendered, no GPU
bash scripts/start_unified.sh --mock --api-port 8210 --web-port 5280

# Phase 2/3 — quicktalk + wav2lip, both on MPS, mock still available
bash scripts/start_unified.sh --backend local --model quicktalk \
  --api-port 8210 --web-port 5280 \
  --env ../say-tts-shim/local-mps.env
```

`local-mps.env` configures both models; `--model` only says whose backend to
override, and `.env` already pins `OPENTALKING_WAV2LIP_BACKEND=local`, so
`/models` reports `mock`, `wav2lip` and `quicktalk` all connected. Pick per
request with the `model` field, or in the WebUI's driver list.

## QuickTalk vs Wav2Lip

QuickTalk is the better model here, and the only one in the repo with explicit
Apple Silicon handling:

```
adapter.py:99      mps.is_available() -> "mps"
runtime_v2.py:204  device.type == "mps" and "CoreMLExecutionProvider" in available
                   -> ["CoreMLExecutionProvider", "CPUExecutionProvider"]
```

Its audio encoder is **chinese-hubert-large** (1.2 GB) rather than Wav2Lip's raw
mel spectrogram, and it drives a **template video** instead of pasting a patch
onto a still. Measured on the same avatar, same audio, same mouth box:

| | mouth motion | aperture range |
|---|---|---|
| quicktalk | 2.377 | 0.0642 |
| wav2lip 384px | 1.623 | 0.0462 |
| ratio | **1.47x** | **1.39x** |

Whole-frame activity also differs enormously (276.4k vs 6.3k), but almost none
of that is the model: QuickTalk composites onto a looping **template video**, so
head, torso, hands and blinks were already there. Subtracting the template's own
activity from the output leaves QuickTalk's real contribution, and it is exactly
one mouth:

| region | template | output | added |
|---|---|---|---|
| face declared in `metadata.face_box` | 0.697 | 1.900 | **+1.203** |
| the other anchor, outside `face_box` | 0.805 | 0.823 | +0.018 |
| whole frame | 0.371 | 0.400 | +0.029 |

So QuickTalk animates one declared face (2.7x over the template's own mouth
motion there) and passes everything else through. The extra liveness is real to
a viewer, but it is the avatar's template video doing that work, not the model.
Wav2Lip looks frozen because it composites onto a still image instead.

That template is also what QuickTalk requires of an avatar:
`examples/avatars/custom-*` ship a `source/source_video.mp4` and declare it, the
`model_type: wav2lip` avatars have only a still.

Cost: 4.7x realtime vs Wav2Lip's 3.0x, and 1.6 GB of weights
(`hf download datascale-ai/quicktalk --local-dir <asset_root>/checkpoints`).
One operator, `aten::linalg_svd` in the face-alignment path, falls back to CPU
with a warning; it is not on the hot path.

### MuseTalk is not an option on this Mac

Its inference core would very likely run — a single CUDA reference, and a
Stable-Diffusion-style `AutoencoderKL` + `UNet2DConditionModel` at 256x256 is
about the most MPS-proven stack there is. The blocker is
`scripts/quickstart/prepare_local_musetalk.sh`, whose preprocessing sidecar
pins `torch==2.0.1+cu118` / `torchvision==0.15.2+cu118` plus `mmcv==2.0.1`
(checked for a compiled `mmcv._ext`), `mmdet==3.1.0` and `mmpose==1.1.0`.
No cu118 wheels exist for macOS arm64.

**`--env` is not optional.** The wav2lip runtime reads its knobs with
`os.environ.get()`, while OpenTalking's `.env` is loaded by pydantic-settings,
which does not populate `os.environ`. Configure `OPENTALKING_WAV2LIP_DEVICE` in
`.env` alone and the runtime silently stays on `device=cpu` with
`jpeg_quality=85` — the startup log line is the only place this shows up:

```
Wav2Lip inference device=mps | face_detection device=mps | checkpoint=.../wav2lip384.pth | jpeg_quality=97
```

Check that line after every config change.

WebUI: http://localhost:5280 · API: http://127.0.0.1:8210

## The two avatar paths

**Realtime (Light2D).** Pick the `DOGO 博士小狗` avatar and the `轻量模式`
driver. The mouth is drawn in the browser on a canvas: 4 mouth layers chosen by
the RMS energy of the WebRTC audio, plus blink/breath/sway. No GPU, no model
weights. Energy-driven, so it tracks *when* you speak, not *which phoneme*.

**Offline (wav2lip).** Real neural lip sync, conditioned on the mel spectrogram,
so mouth shapes match the phonemes. Produces a downloadable MP4.

```bash
say -v Yuna -o narration.wav --data-format=LEI16@16000 "안녕하세요"

curl -X POST http://127.0.0.1:8210/video-creation/jobs \
  -F model=wav2lip -F avatar_id=singer -F audio_source=upload \
  -F title=demo -F execution_mode=sync \
  -F "audio_file=@narration.wav;type=audio/wav"
```

This path needs neither the shim nor an LLM — it takes the audio file directly.

## Measured on an M4 Pro

wav2lip generator forward pass, batch 8:

| variant | output | MPS | CPU |
|---|---|---|---|
| `wav2lip` (default here) | 96×96 | 311 fps | 57 fps |
| `wav2lip256` | 256×256 | 50 fps | 10 fps |
| `wav2lip384` | 384×384 | 17 fps | 3.1 fps |

End-to-end video creation of 8.1 s of narration:

| config | wall | vs realtime |
|---|---|---|
| 96px, office-woman, first run | 30 s | 3.7× |
| 96px, office-woman, warm | 10.4 s | 1.3× |
| 96px, singer (126 frames), first run | 12.7 s | 1.6× |
| **384px**, office-woman | 25 s | 3.1× |
| **384px**, singer | 30 s | 3.8× |
| **384px**, newscaster (14.4 s narration, 790×1166) | 61 s | 4.3× |
| 384px on CPU (the `.env`-only mistake) | 85 s | 10.5× |

## Quality

Three independent losses stacked on the mouth region. In order of impact:

**1. Patch upscale — the big one.** The model emits an `N×N` patch that is
`cv2.resize`d onto the detected face box. Measured box sizes:

| avatar | face box | 96px | 256px | 384px |
|---|---|---|---|---|
| office-woman | 222×304 | 2.31× upscale | 0.87× | 0.58× |
| newscaster | 222×321 | 2.31× upscale | 0.87× | 0.58× |
| singer | 164×242 | 1.71× upscale | 0.64× | 0.43× |

Measure a new avatar before assuming 384px is enough — a tighter crop or a
larger source image pushes the box up. `_detect_face_box` on the avatar's first
frame is all it takes.

That 2.31× upscale was the blur. `wav2lip384.pth` fixes it and is what this
setup uses. Note `wav2lip256` would also clear it (0.87×) at 50 fps vs 17 fps —
worth switching to if throughput ever matters more than headroom.

**2. Per-frame JPEG.** Every generated frame is JPEG-encoded before reaching the
video writer, at quality 85 by default. Now 97.

**3. The encode chain** (requires the patch below): frames went through a
`cv2.VideoWriter` `mp4v` intermediate — a lossy codec with no quality control —
and then `libx264` with no `-crf`, falling back to 23. Measured on 60 real
avatar frames: **36.50 dB vs 40.28 dB PSNR**, i.e. the chain was throwing away
3.8 dB before anyone looked at the file.

### Local patch to `opentalking/video_creation.py`

Three edits, `git diff` = 11 insertions / 5 deletions:

- `_IncrementalVideoWriter` picks `FFV1` when the target is `.mkv` (lossless),
  else keeps `mp4v`.
- The four `work_dir / "video_only.mp4"` intermediates became `.mkv`.
- `_ffmpeg_mux` passes `-crf 18`.

`pytest -k "video or creation"` passes (175 tests). Verify the CRF landed with
`strings out.mp4 | grep -o 'crf=[0-9.]*'`.

**Not attempted:** `OPENTALKING_WAV2LIP_POSTPROCESS_MODE=easy_enhanced` runs
GFPGAN face restoration on the patch. It needs `gfpgan` + `basicsr` +
`GFPGANv1.4.pth`, and since 384px already downscales into the box there is no
upscale blur left for it to recover — it would be adding synthesized detail,
which risks looking inconsistent with the untouched upper face.

## Caveats

- **wav2lip on MPS is not an upstream-supported path.** The Apple Silicon docs
  cover only `quicktalk` and call Mac validation-only; there are no Mac entries
  in the upstream benchmark table. It works because the wav2lip implementation
  has no CUDA-specific code — only `Conv2d`/`ConvTranspose2d`/`BatchNorm2d`/
  `ReLU`/`Sigmoid`/`max_pool2d`/`softmax`, all MPS-supported. Bugs here are ours.
- wav2lip regenerates the whole lower face, so cheeks and jaw stay marginally
  softer than the eyes and hair even at 384px. The patch size is not
  configurable.
- `video_creation.py` now diverges from upstream (see the patch above); rebasing
  onto a new upstream needs that reapplied.
- Light2D is undocumented upstream — it exists in code only. The avatar format
  is simple (6 layer PNGs + `avatar.json` rects and thresholds), so a custom
  character is drawings plus one JSON file.
- Pressing Return in the WebUI text box does not always submit; click 发送.
- `zh-CN-*` is OpenTalking's default voice id. Locale matching then picks a
  Chinese voice, which reads Korean text as near-silence. `.env` sets
  `OPENTALKING_TTS_OPENAI_VOICE=Yuna` to avoid that.
