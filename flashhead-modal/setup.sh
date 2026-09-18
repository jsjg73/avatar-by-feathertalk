#!/usr/bin/env bash
# Build the renderer's environment against the Python already on this box.
#
# Use this where Docker is not available — most rented GPU boxes hand you a
# container and no way to nest one. It runs the same steps as the Dockerfile;
# keep the two in step when either changes.
#
#   FLASHHEAD_SRC=~/SoulX-FlashHead FLASHHEAD_WEIGHTS=~/weights ./setup.sh
#   python run_local.py --fetch-weights
#   python run_local.py --sessions 6 --minutes 3 --gap-s 18
#
# Assumes: an NVIDIA driver supporting CUDA 12.4 (>= 550), python >= 3.10, git,
# and ffmpeg. It checks all of those before touching anything.
set -euo pipefail

SRC="${FLASHHEAD_SRC:-$HOME/SoulX-FlashHead}"
CACHE="${FLASHHEAD_CACHE:-$HOME/flashhead-cache}"
COMPILE="${FLASHHEAD_COMPILE_BUILD:-False}"     # True to keep upstream's torch.compile
REPO=https://github.com/Soul-AILab/SoulX-FlashHead.git
PY="${PYTHON:-python3}"

say() { printf '\n== %s\n' "$1"; }

say "점검"
command -v git >/dev/null || { echo "git 이 없습니다"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg 이 없습니다 (apt-get install -y ffmpeg)"; exit 1; }
# Python 3.11 exactly. Upstream pins mediapipe==0.10.9 (face crop), whose last
# wheels are for 3.11 — on 3.12 pip offers 0.10.13+ instead, and taking a newer
# mediapipe would change the crop and so the rendered frames, which makes the
# numbers incomparable with earlier runs. Modal's image is 3.11 for the same
# reason. If the box's default is something else, build a 3.11 env and point
# PYTHON at it:  uv venv ~/py311 --python 3.11 && PYTHON=~/py311/bin/python ./setup.sh
"$PY" - <<'EOF' || exit 1
import sys
v = sys.version_info
if (v.major, v.minor) != (3, 11):
    sys.exit(f"python 3.11 이 필요합니다 (지금 {v.major}.{v.minor}). "
             "mediapipe==0.10.9 휠이 3.11 까지만 있습니다.\n"
             "  uv venv ~/py311 --python 3.11 && PYTHON=~/py311/bin/python ./setup.sh")
print("python", sys.version.split()[0], "OK")
EOF
# `uv venv` does not seed pip, and every install step below goes through it.
"$PY" -m pip --version >/dev/null 2>&1 || {
  echo "pip 이 없어 ensurepip 로 넣습니다"
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 \
    || { echo "pip 설치 실패 — 'uv venv --seed' 로 환경을 다시 만드세요"; exit 1; }
}

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "nvidia-smi 가 없습니다 — GPU 박스가 맞는지 확인하세요"; exit 1
fi

say "torch (cu124)"
# index_url, not extra-index-url: plain `torch` on PyPI resolves to a CUDA 13
# build today, whose nvidia-* stack then fights requirements.txt.
"$PY" -m pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

say "upstream"
[ -d "$SRC/.git" ] || git clone --depth 1 "$REPO" "$SRC"

say "requirements (일부 제외)"
"$PY" - "$SRC" <<'EOF'
import sys
drop = ("xformers", "nvidia-nccl-cu12", "decord", "flask", "gradio")
src = sys.argv[1] + "/requirements.txt"
keep = [l for l in open(src) if l.strip() and not any(l.lower().startswith(d) for d in drop)]
open(sys.argv[1] + "/requirements.portable.txt", "w").writelines(keep)
print("kept", len(keep), "requirement lines")
EOF
"$PY" -m pip install --no-cache-dir -r "$SRC/requirements.portable.txt"
"$PY" -m pip install --no-cache-dir "huggingface_hub[cli]"

say "flash-attn (있으면 빠르고, 없으면 SDPA 로 떨어진다)"
"$PY" -m pip install --no-cache-dir flash_attn==2.8.0.post2 --no-build-isolation \
    || echo "flash_attn 설치 실패 — SDPA 로 동작합니다"
# Installing is not the same as working. A wheel built against a different
# libtorch loads as an ImportError ("undefined symbol: ...") — and upstream
# guards its import with `except ModuleNotFoundError`, which does not catch that,
# so a half-installed flash_attn takes the whole pipeline down at import. If it
# cannot be imported, remove it so the guard sees a clean absence and falls back.
if "$PY" -m pip show flash_attn >/dev/null 2>&1; then
  if "$PY" -c 'import flash_attn' 2>/dev/null; then
    echo "flash_attn import OK"
  else
    echo "flash_attn 이 설치됐지만 import 되지 않습니다 (torch 와 ABI 불일치) — 제거하고 SDPA 로 갑니다"
    "$PY" -m pip uninstall -y flash_attn >/dev/null 2>&1
  fi
fi

PIPE="$SRC/flash_head/src/pipeline/flash_head_pipeline.py"

say "T1: torch.compile = $COMPILE"
sed -i.bak "s/^COMPILE_MODEL = .*/COMPILE_MODEL = ${COMPILE}/; s/^COMPILE_VAE = .*/COMPILE_VAE = ${COMPILE}/" "$PIPE"
grep -q "^COMPILE_MODEL = ${COMPILE}" "$PIPE" && grep -q "^COMPILE_VAE = ${COMPILE}" "$PIPE" \
    || { echo "compile 플래그 패치 실패"; exit 1; }

say "T2: 청크당 강제 동기화 8회 제거"
# Purely upstream's per-stage timing. On L4 a denoise step is ~105 ms and the
# stalls hide in it; on H100 a step is ~29 ms and they do not.
sed -i.bak '/torch\.cuda\.synchronize()/d; s/^\( *\)print(f.\[generate\].*/\1pass/' "$PIPE"
! grep -q 'torch\.cuda\.synchronize()' "$PIPE" || { echo "동기화 제거 실패"; exit 1; }
"$PY" -c "import ast; ast.parse(open('$PIPE').read())" || { echo "패치 후 파싱 실패"; exit 1; }

mkdir -p "$CACHE/inductor" "$CACHE/triton" "$CACHE/hf"
cat <<EOF

준비됐습니다. 이 값들을 셸에 두고 쓰세요:

  export FLASHHEAD_SRC=$SRC
  export FLASHHEAD_WEIGHTS=\${FLASHHEAD_WEIGHTS:-\$HOME/weights}
  export FLASHHEAD_CACHE=$CACHE
  export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
  export TRITON_CACHE_DIR=$CACHE/triton
  export HF_HOME=$CACHE/hf
  export FLASHHEAD_WARMUP=\$PWD/inputs/newscaster.png

  python run_local.py --fetch-weights          # 한 번, 약 8 GB
  python run_local.py --sessions 6 --minutes 3 --gap-s 18
EOF
