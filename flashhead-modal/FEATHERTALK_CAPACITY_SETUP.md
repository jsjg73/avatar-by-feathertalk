# FeatherTalk 동시성 벤치마킹 — 새 박스 셋업

`git clone` 만으로는 재현이 안 되는 부분(업스트림 클론, 체크포인트, 원본 영상, GPU 박스에서만 만들 수 있는 파생 캐시)을 메우기 위한 단계별 절차. 벤치마크 도구 자체(`bench_feathertalk_concurrent*.py`, `build_frame_cache.py`, `sweep_concurrent.py`)는 이 저장소에 커밋되어 있으므로 별도 준비가 필요 없다.

동시성 벤치마킹의 일반 원칙(용어, 착수 요청 템플릿)은 [`../docs/gpu-serving/README.md`](../docs/gpu-serving/README.md) 를 먼저 참고. 이 문서는 FeatherTalk 에 한정된 구체적 절차다.

## 0. 필요한 것

- 원본 아바타 영상 (`train.mp4`, 25fps 필수) — git에 없다. 로컬에 보관된 원본을 박스로 올려야 한다.
- 체크포인트 `last.pth`/`59.pth` — 이 저장소의 `flashhead-modal/checkpoints_featherhead_kjs/` 에 커밋되어 있다.
- FeatherTalk 코드 — 업스트림 클론이라 이 저장소에는 없다 (`.gitignore` 로 `/model-repos/` 전체 제외).
- **Vast.ai 자격증명 — 저장소에 없고, 각자 새로 만들어야 한다.** 아래 셋:
  1. [vast.ai](https://vast.ai) 계정 + API 키 → `~/.vast_key` 에 저장(`chmod 600`) 하거나 `$VAST_API_KEY` 환경변수로 설정.
  2. 이 박스 대여 전용 SSH 키페어. 없으면 `python box.py rent vast ...` 실행 시 `box.py` 가 만들 명령어를 직접 알려준다 (내부적으로 `~/.ssh/vast_flashhead` 를 찾는다):
     ```bash
     ssh-keygen -t ed25519 -N '' -C flashhead-rented-gpu -f ~/.ssh/vast_flashhead
     ```
  3. 위 둘 다 **레포 밖(홈 디렉터리)에 두는 게 의도된 설계** — `box.py` 코드에도 커밋되지 않고, git에 실수로 딸려갈 일이 없다.

  Modal/Lightning 벤더를 쓸 경우엔 각각 `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, `LIGHTNING_API_KEY` 가 같은 방식(환경변수)으로 필요하다 — FeatherTalk 벤치마킹 자체는 Vast만 쓰므로 해당 없음.

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

# 3) 실시간 서빙(live_server.py, gateway.py)까지 쓸 경우 추가로 필요
pip install fastapi uvicorn python-multipart httpx websockets
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

## 7. 실시간 서빙 게이트웨이 (multi-session)

벤치마크(1~6)가 끝나 N=24 가 검증된 이후, 실제 세션을 처리하려면 `gateway.py`가 `live_server.py` 워커 N개를 띄우고 그 앞에서 세션을 배정한다. 설계 배경은 [`docs/live-serving/DESIGN.md`](../docs/live-serving/DESIGN.md).

**알려진 문제 (2026-09-29, 미해결) — `--workers`를 24로 올리지 말 것:**
- **워커 24개 동시 실행 시 영상이 깨진다.** 메모리 문제는 아님(격리 테스트로 기각됨) — CPU/ffmpeg 자원 경합으로 추정되나 정확한 메커니즘 미확인.
- **`/speak` 호출마다 메모리 누수가 있다** (재생 끝나도 안 줄어듦, 호출당 ~1.5MB). 오래 켜두면 워커가 결국 OOM으로 죽는다. 이미 워커 3개가 이 방식으로 죽은 적 있음.
- 이 두 문제 때문에 `ai-interview-avatar-proto` 연동 테스트 기간에는 `--workers 1`로 낮춰서 운영 중이다. 원인 규명·수정 전에는 1~2개 이상으로 올리지 말 것.

**`--port`는 그 박스가 실제로 공인 노출한 포트로 맞춰야 한다** — Vast 인스턴스는 기본적으로 컨테이너 포트 중 미리 매핑된 것 하나(보통 `8000/tcp`)만 공인 IP:포트로 열려 있다. 다른 포트로 띄우면 게이트웨이 자체는 멀쩡히 돌아도 박스 밖에서는 아무도 못 붙는다 — 처음 이걸 8080으로 띄웠다가 겪은 실수다. 실제 매핑은:

```bash
python3 -c "
import box, json
p = box.VastProvider()
for i in p._instances():
    if i.get('actual_status') == 'running':
        print(i['id'], i.get('public_ipaddr'), i.get('ports'))
"
```

```bash
python gateway.py \
    --dataset data/kjs --checkpoint ckpt_full/last.pth --fh_checkpoint feather_hubert.pth \
    --workers 24 --base_port 9000 --port 8000
```

워커 24개가 순차로 뜨고(모델 로드 시간만큼), 전부 `/status` 응답이 올 때까지 게이트웨이가 기다린 뒤에야 `8000`에서 요청을 받기 시작한다. 세션 생성은:

```bash
curl -X POST http://localhost:8000/sessions          # -> {"session_id": "..."}
curl -X POST http://localhost:8000/sessions/<id>/speak -F file=@line.wav
# WebSocket ws://localhost:8000/sessions/<id>/ws 로 fMP4 스트림 수신 (studio_live.html의 클라이언트 패턴 재사용)
curl -X POST http://localhost:8000/sessions/<id>/end
```

24슬롯이 다 찬 상태에서 `POST /sessions`는 큐잉 없이 즉시 `503 {"error": "no_capacity"}`.

**2026-09-29 실제 박스(RTX 4090)에서 엔드투엔드 검증 완료**: 워커 24개 전체 기동, 세션 생성/발화 큐잉/용량 초과 거절/WS 미디어 릴레이(실제 립싱크 프레임 수신 확인)까지 전부 통과. 공인 IP:포트(위 `ports` 조회로 확인)로 박스 밖에서 직접 호출해서 확인했다.

## 되짚어볼 것

이 절차 전체를 1~6 순서대로 실행하는 셋업 스크립트(`setup_box.sh`)로 묶으면 사람이 순서를 기억할 필요가 없어진다 — 아직 안 만들었다.
