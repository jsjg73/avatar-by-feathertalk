#!/usr/bin/env bash
# Take a bare rented GPU box to "ready to measure", unattended.
#
# This is what `box.py rent vast` sends as the instance's onstart script. It
# runs at boot, as root, before anyone has logged in — so the 15 minutes of
# downloading happens while the laptop is still uploading its files, instead of
# after.
#
# It deliberately does NOT restate how the environment is built. `setup.sh` is
# the single source of that, and `box.py` splices it in here as a base64 blob
# (the SETUP_B64 line below). The Dockerfile is the third expression of the same
# steps; those two are kept in step by hand, and this one cannot drift because
# it does not contain them.
#
# What this adds around setup.sh is only what a *rented* box needs:
#   - a Python 3.11 that the image probably does not have
#   - the weights, which are 8 GB and want to start downloading immediately
#   - a marker file, so "is it ready?" is a question with an answer
#
# Run it by hand anywhere else:
#   SETUP_B64=$(base64 < setup.sh) bash provision.sh
set -uo pipefail

LOG=/root/provision.log
exec > >(tee -a "$LOG") 2>&1
echo "== provision 시작 $(date -u +%FT%TZ)"

fail() { echo "✗ $1"; echo "$1" > /root/PROVISION_FAILED; exit 1; }

export DEBIAN_FRONTEND=noninteractive
export FLASHHEAD_SRC=/root/SoulX-FlashHead
export FLASHHEAD_WEIGHTS=/root/weights
export FLASHHEAD_CACHE=/root/flashhead-cache
export HF_HOME=$FLASHHEAD_CACHE/hf
export UV_PYTHON_INSTALL_DIR=/root/.uv-python     # not /system — that vanished on Lightning
export PATH=/root/.local/bin:$PATH

echo "== apt"
apt-get update -qq || fail "apt-get update 실패"
apt-get install -y -qq --no-install-recommends \
    git curl ca-certificates ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    || fail "apt 패키지 설치 실패"

echo "== python 3.11"
# mediapipe==0.10.9 (upstream's pin, and what the face crop depends on) has no
# wheel past 3.11. Taking a newer mediapipe would change the crop and so the
# frames, which makes every number here incomparable with the earlier runs.
if python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
    PY=$(command -v python3)
    echo "이미지 python 이 3.11 입니다: $PY"
else
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv 설치 실패"
    uv venv /root/py311 --python 3.11 --seed || fail "3.11 venv 생성 실패"
    PY=/root/py311/bin/python
fi
"$PY" -V || fail "python 이 동작하지 않습니다"
echo "PYTHON=$PY" > /root/flashhead.env

echo "== 가중치 내려받기 시작 (백그라운드, 약 8 GB)"
# Started before setup.sh because both are network-bound but the weights come
# from a different host than PyPI, so the two overlap instead of queueing.
"$PY" -m pip install -q --no-cache-dir huggingface_hub || fail "huggingface_hub 설치 실패"
mkdir -p "$FLASHHEAD_WEIGHTS" "$HF_HOME"
setsid nohup "$PY" - > /root/weights.log 2>&1 <<'PYW' &
import os
from huggingface_hub import snapshot_download
w = os.environ["FLASHHEAD_WEIGHTS"]
snapshot_download("Soul-AILab/SoulX-FlashHead-1_3B", local_dir=f"{w}/SoulX-FlashHead-1_3B",
                  ignore_patterns=["*14B*", "*pro*"], max_workers=8)
snapshot_download("facebook/wav2vec2-base-960h", local_dir=f"{w}/wav2vec2-base-960h")
open("/root/WEIGHTS_READY", "w").write("ok\n")
print("weights ready", flush=True)
PYW
WPID=$!
echo "weights pid $WPID"

echo "== setup.sh (환경 구성의 유일한 출처)"
[ -n "${SETUP_B64:-}" ] || fail "SETUP_B64 가 비었습니다 — box.py 가 setup.sh 를 심어주지 않았습니다"
echo "$SETUP_B64" | base64 -d | gunzip > /root/setup.sh || fail "setup.sh 디코드 실패"
chmod +x /root/setup.sh
PYTHON="$PY" bash /root/setup.sh || fail "setup.sh 실패 — /root/provision.log 를 보세요"

echo "== 가중치 대기"
wait "$WPID" || true
[ -f /root/WEIGHTS_READY ] || fail "가중치 내려받기 실패 — /root/weights.log 를 보세요"
du -sh "$FLASHHEAD_WEIGHTS"

echo "== 확인"
"$PY" - <<'PYC' || fail "torch 가 GPU 를 보지 못합니다"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "ok", torch.cuda.is_available())
assert torch.cuda.is_available()
print("gpu:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB",
      "sm", ".".join(map(str, torch.cuda.get_device_capability(0))))
PYC

cat >> /root/flashhead.env <<ENVEOF
export FLASHHEAD_SRC=$FLASHHEAD_SRC
export FLASHHEAD_WEIGHTS=$FLASHHEAD_WEIGHTS
export FLASHHEAD_CACHE=$FLASHHEAD_CACHE
export HF_HOME=$HF_HOME
export TORCHINDUCTOR_CACHE_DIR=$FLASHHEAD_CACHE/inductor
export TRITON_CACHE_DIR=$FLASHHEAD_CACHE/triton
ENVEOF

date -u +%FT%TZ > /root/PROVISIONED
echo "== provision 완료 $(cat /root/PROVISIONED)"
