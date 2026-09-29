# FeatherTalk 동시성 벤치마킹 — 새 박스 셋업

`git clone` 만으로는 재현이 안 되는 부분(업스트림 클론, 체크포인트, 원본 영상, GPU 박스에서만 만들 수 있는 파생 캐시)을 메우기 위한 단계별 절차. 벤치마크 도구 자체(`bench_feathertalk_concurrent*.py`, `build_frame_cache.py`, `sweep_concurrent.py`)는 이 저장소에 커밋되어 있으므로 별도 준비가 필요 없다.

동시성 벤치마킹의 일반 원칙(용어, 착수 요청 템플릿)은 [`../docs/gpu-serving/README.md`](../docs/gpu-serving/README.md) 를 먼저 참고. 이 문서는 FeatherTalk 에 한정된 구체적 절차다.

## 0. 필요한 것

- 원본 아바타 영상 (`train.mp4`, 25fps 필수) — git에 없다. 로컬에 보관된 원본을 박스로 올려야 한다.
- 체크포인트 `last.pth`/`59.pth` — 이 저장소의 `flashhead-modal/checkpoints_featherhead_kjs/` 에 커밋되어 있다.
- FeatherTalk 코드 — 업스트림 클론이라 이 저장소에는 없다 (`.gitignore` 로 `/model-repos/` 전체 제외).

## 1. FeatherTalk 클론 + 패치

```bash
git clone https://github.com/anliyuan/FeatherTalk.git
cd FeatherTalk
# 맥에서 로컬 테스트할 때만: MPS 폴백 패치 (Linux GPU 박스에는 불필요)
git apply /path/to/opentalk/patches/featherhead-mps-fallback.patch
```

## 2. 박스에 클린 업로드

로컬 macOS 에서 만든 `.venv`/`.git`/`__pycache__`/`.DS_Store` 를 **절대 같이 올리지 않는다** — 특히 `uv` 로 만든 `.venv` 는 `bin/python` 심볼릭 링크가 macOS 전용 절대경로를 가리켜서, 나중에 `python -m venv` 로 새로 venv 를 만들어도 그 심볼릭 링크 슬롯을 강제로 덮어쓰지 않아 조용히 깨진다 ([`../docs/gpu-serving/README.md`](../docs/gpu-serving/README.md) §1 clean deploy 참고).

```bash
COPYFILE_DISABLE=1 tar --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=.DS_Store \
    -czf feathertalk.tar.gz FeatherTalk/
scp feathertalk.tar.gz <box>:/root/
# 박스에서
tar xzf feathertalk.tar.gz
```

체크포인트도 같이 올린다 (이 저장소에서):

```bash
scp flashhead-modal/checkpoints_featherhead_kjs/{last.pth,59.pth} <box>:/root/FeatherTalk/ckpt_full/
```

## 3. venv + 의존성 설치

**반드시 두 번에 나눠서 설치한다** — `--index-url` 을 다른 패키지와 한 명령에 섞으면 그 명령 전체가 그 인덱스로만 제한되어 `opencv-python` 등이 "no version found" 로 실패한다.

```bash
cd /root/FeatherTalk
python3.11 -m venv .venv
source .venv/bin/activate

# 1) torch — CUDA 인덱스 지정, 이번에 검증된 조합
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2) 나머지 — 인덱스 없이 별도 명령
pip install opencv-python librosa soundfile onnx onnxruntime numpy tqdm transformers
```

설치 후 `ls -la .venv/bin/python*` 로 심볼릭 링크가 `python3.11 -> /opt/conda/bin/python3.11` (또는 박스의 실제 시스템 파이썬) 을 가리키는지 확인 — macOS 잔재가 섞였으면 여기서 깨진 채로 나온다.

## 4. 데이터셋 전처리 (프레임 25fps 필수)

```bash
python data_utils/process.py data/kjs/train.mp4 --feather_hubert_checkpoint feather_hubert.pth
```

`data/kjs/full_body_img/*.jpg`, `data/kjs/landmarks/*.lms`, `data/kjs/aud_hu.npy` 가 생성된다.

## 5. 공유 이미지 캐시 빌드 (동시성 확보의 핵심)

```bash
python build_frame_cache.py --dataset data/kjs
```

`data/kjs/full_body_img_cache.raw` (프레임 수 × 1620×1080×3 바이트, 이 데이터셋 기준 43GB) + `.meta.json` 생성. 이게 없으면 `bench_feathertalk_concurrent_cached.py` 가 안 돈다 — 매 프레임 `cv2.imread` 로 돌아가 N≤8 수준으로 떨어진다.

## 6. 동시성 스윕 실행 (반복 가능한 검증)

```bash
python sweep_concurrent.py \
    --dataset data/kjs --checkpoint ckpt_full/last.pth --audio_feat data/kjs/aud_hu.npy \
    --worker bench_feathertalk_concurrent_cached.py \
    --ns 8,16,24,32 --seconds 8 --out sweep_cached_report.json
```

**기대값 (2026-09-29, RTX 4090/28코어 기준)**: N=24 까지 안전(min fps ≥ 25), N=32 에서 붕괴. 이 숫자가 안 나오면 5번(캐시 빌드)이 빠졌거나 다른 프로세스가 GPU/CPU 를 같이 쓰고 있는지 확인.

## 되짚어볼 것

이 절차 전체를 1~6 순서대로 실행하는 셋업 스크립트(`setup_box.sh`)로 묶으면 사람이 순서를 기억할 필요가 없어진다 — 아직 안 만들었다.
