#!/usr/bin/env bash
# Generate the Korean clip. Assumes setup.sh has run and inputs/ is uploaded
# to /workspace/SoulX-FlashHead/inputs/.
set -euo pipefail
cd /workspace/SoulX-FlashHead

IMAGE="${1:-inputs/newscaster.png}"
AUDIO="${2:-inputs/korean_33s.wav}"
OUT="${3:-out_$(basename "${IMAGE%.*}")_$(basename "${AUDIO%.*}").mp4}"

# --model_type lite      the realtime variant (96 FPS on a 4090)
# --audio_encode_mode    stream encodes per chunk, which is what the realtime
#                        path uses; `once` encodes the whole clip up front
# --use_face_crop True   detect and crop the face; the OpenTalking avatars are
#                        half-body shots, so this matters
# --base_seed            defaults to 42 upstream, pinned here so reruns match
python generate_video.py \
  --ckpt_dir models/SoulX-FlashHead-1_3B \
  --wav2vec_dir models/wav2vec2-base-960h \
  --model_type lite \
  --cond_image "$IMAGE" \
  --audio_path "$AUDIO" \
  --audio_encode_mode stream \
  --use_face_crop True \
  --base_seed 42 \
  --save_file "$OUT"

echo "wrote $OUT"
