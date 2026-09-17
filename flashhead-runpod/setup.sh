#!/usr/bin/env bash
# SoulX-FlashHead on a RunPod GPU pod. Paste into the pod terminal.
#
# Pod: RTX 4090 (24 GB) is what upstream benchmarks -- Model_Lite hits 96 FPS
#      there, Model_Pro 10.8 FPS. A5000/A6000/L40S also fine. Model_Pro needs
#      two RTX 5090 only for *realtime*; single-GPU offline is fine.
# Template: any "PyTorch 2.x + CUDA 12.x" image.
# Container disk: 40 GB+ (weights are 14.3 GB, or ~8 GB for Lite only).
set -euo pipefail

cd /workspace
git clone https://github.com/Soul-AILab/SoulX-FlashHead.git
cd SoulX-FlashHead

pip install -U "huggingface_hub[cli]"
pip install -r requirements.txt

# Upstream lists these as separate steps; both are CUDA-only and both are
# optional -- the model falls back to F.scaled_dot_product_attention without
# them, just slower.
pip install flash_attn==2.8.0.post2 --no-build-isolation || echo "flash_attn skipped (SDPA fallback)"
pip install sageattention==2.2.0 --no-build-isolation || echo "sageattention skipped"

mkdir -p models
# --exclude Model_Pro saves 6 GB; drop the flag if you want to compare both.
hf download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/SoulX-FlashHead-1_3B \
  --exclude "Model_Pro/*"
hf download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h \
  --exclude "*.h5" "*.msgpack" "*tf_model*"

echo
echo "Setup done. Upload inputs/ then run:  bash run_korean.sh"
