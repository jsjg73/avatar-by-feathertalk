# FlashHead on Modal

[SoulX-FlashHead-1_3B](https://github.com/Soul-AILab/SoulX-FlashHead) (Model_Lite)를 Modal L4에 **상주 클래스**로 올린 것.
Mac에서는 `say`로 만든 16 kHz WAV와 아바타 PNG를 보내고 MP4를 받는다. 스튜디오 페이지는 [`studio/`](studio/README.md).

```bash
M=../opentalking/.venv/bin/modal
$M deploy app.py                                   # 배포 (이미지 캐시 시 ~5s). 앱 이름 "flashhead"
$M run app.py --audio inputs/korean_short.wav      # 배포된 Renderer에 프로브, 진행 로그를 Queue로 받아 출력
$M run app.py::smoke                               # 환경 점검 (GPU, import)
$M run app.py::fetch_weights                       # 가중치 Volume 채우기 (최초 1회, 이미 완료)
$M container list                                  # 과금 중인 컨테이너 — 유휴 상태면 비어 있어야 함
```

## 구조 (`app.py`)

- **이미지**: `nvidia/cuda:12.4.1` + torch cu124(`index_url`, PyPI torch는 CUDA 13으로 풀림) + 상류 requirements에서 `xformers, nvidia-nccl-cu12, decord, flask, gradio` 제거(추론 경로 미사용, 의존성 충돌). `flash_attn` 빌드 실패는 SDPA 폴백이라 무시.
  **`COMPILE_MODEL/COMPILE_VAE`를 sed로 False** — 짧은 클립에선 컴파일이 430 s를 먹고 4 s 생성함(`once` 모드는 텐서 모양이 달라 inductor 캐시도 안 맞음).
- **Volume** `flashhead-weights`(Model_Lite, VAE, wav2vec2), `flashhead-compile-cache`(현재는 거의 안 씀).
- **`Renderer` (`@app.cls`, L4, `max_containers=1`, `scaledown_window=600`, `model_type: modal.parameter` = `lite`|`pro` — 값마다 별도 컨테이너 풀/오토스케일러)**
  - `@modal.enter() load()`: `os.chdir(SRC)` 후 `get_pipeline()` → 번들된 `inputs/newscaster.png`로 1청크 워밍업 → **librosa/imageio 워밍업**(numba JIT + ffmpeg 플러그인; 이걸 안 하면 컨테이너의 첫 호출만 27 s 추가) → `state["ready"]` 플래그.
  - `@modal.method() render(image_bytes, audio_bytes, job_id, seed, tail_seconds=0.5)`: 상류 `generate_video.py`의 `once` 분기를 in-process로 — wav2vec2 한 번에 인코딩, 33프레임 청크(24프레임씩 전진), imageio h264 + ffmpeg aac 먹스. 오디오는 **영상이 음성 끝 + `tail_seconds`를 덮도록** 무음 패딩하고, 먹스에도 그 패딩된 오디오를 쓴다(상류 규칙대로면 영상이 0.96 s 짧아 마지막 말이 잘림). 진행 줄은 `modal.Queue("flashhead-progress")`의 `partition=job_id`로 push(배포된 메서드의 stdout은 호출자에게 안 오므로).
  - `@modal.method() render_stream(...)` (제너레이터): 같은 생성 루프지만 프레임을 컨테이너 안의 `ffmpeg -f hls`(fMP4, 세그먼트 = 청크 0.96 s)로 흘려 넣고, 닫힌 세그먼트를 즉시 `yield`. 호출자는 `remote_gen()`으로 받아 재생 중인 플레이어에 공급한다(스튜디오의 스트리밍 재생 경로).
  - `@modal.method() live(session_id, image_bytes, ...)`: 라이브 세션 — stream 모드를 벽시계에 맞춰 계속 생성(무음이면 대기 동작), 오디오는 큐 파티션 `<sid>:audio`, HLS는 `<sid>`로(`_HlsShipper`).
  - `@modal.exit()`: ready 플래그 제거.
- **`LivePro` (`@app.cls`, `H100:2`, `region=["jp"]`, `scaledown_window=300`)**: 이미지 = `base_image` + FA2 휠 + `xfuser==0.4.5` + `live_worker.py`. `@modal.enter()`에서 `torchrun --nproc_per_node 2 -m live_worker`를 상주시켜 Pro를 로드(워커 준비까지 ~40 s), `live()`는 감독만 한다(스튜디오 오디오 → 제어 FIFO, 상태 파일 → 로그, HLS 파일 → 큐). 실측 0.68 s/1.12 s 청크(1.65×), 스케줄 지연 0.
- **`tail_progress(job_id, call)`**: Queue를 비우고 `call.get(timeout=0)`을 폴링하는 제너레이터. CLI와 스튜디오가 공유.

## 예열(keep-warm)

```python
Renderer = modal.Cls.from_name("flashhead", "Renderer")
Renderer().update_autoscaler(min_containers=1)   # 켜기: 컨테이너 1대 상주, L4 $0.799/hr
Renderer().update_autoscaler(min_containers=0)   # 끄기: 마지막 호출 후 600 s 뒤 종료
```

- **`modal deploy`마다 이 오버라이드는 초기화된다**(SDK 문서). 그래서 스튜디오가 `warm.json`을 진실로 삼고 기동 시 다시 적용한다.
- `modal container list`는 로딩 중인 컨테이너도 보여주므로 "준비됨" 판단은 `modal.Dict("flashhead-state")["ready"]`로 한다.

## 다중 GPU·대형 GPU 점검 (`probe_gpu.py`, 2026-09-14)

`modal run probe_gpu.py`(구성별 스케줄·P2P·CUDA 호환) / `modal run probe_gpu.py::regions`(리전 고정). 결과: H100:2·H100:4·H200:2·A100-80GB:2·B200 모두 수 초 내 스케줄, P2P 가능. **B200은 현재 이미지(cu124, torch 2.6)에서 커널 없음** — Blackwell(sm_100/sm_120: B200, RTX-PRO-6000)은 cu128 이미지로 재빌드가 필요. Pro 실시간(≥2× 여유)은 2×H100/H200 + xfuser 시퀀스 병렬(Ulysses 2) + compile + flash-attn 구성이 권장안.

## Pro 실시간 벤치 (`bench_multi.py`)

`modal run bench_multi.py` — 상류의 멀티 GPU 경로(`torchrun` + xfuser Ulysses 시퀀스 병렬, stream 모드)로 Pro 청크 시간을 측정한다. jp 리전의 2×H100(FA2 compile off/on, SageAttention compile on)과 1×H100(참조)을 병렬로 띄우고 `bench/h100_jp_pro.json`에 저장. USP 어텐션은 FA2 또는 SageAttention이 필수(SDPA 폴백 없음)라 벤치 이미지는 `base_image` 위에 FA2 프리빌트 휠(+sageattention)을 얹는다. 결과는 아래 "실측" 절 참조.

### 실측 (2026-09-14, jp 리전, Pro, stream 모드, FA2, 30.9 s 오디오 = 28청크 × 1.12 s)

| 구성 | 청크당 | 실시간 배율 | 비고 |
|---|---|---|---|
| 1×H100, compile OFF | 1.072 s (1.070–1.075) | 1.05× (26 FPS) | 경계 |
| **2×H100, compile OFF** | **0.660 s (0.658–0.666)** | **1.70× (42 FPS)** | denoise 스텝 0.087 s. **권장 설정** |
| 2×H100, compile ON | 1.13 s 중앙값 (0.71–1.99, 첫 청크 310 s 컴파일) | 0.99× | USP 경로에서 compile은 그래프 단절·재컴파일로 오히려 느리고 흔들림 → 끄기 |
| 2×H100, SageAttention + compile ON | 실패 | — | yunchang `SAGE_AUTO` 경로 `ValueError: not enough values to unpack` (xfuser 0.4.5 + sageattention 1.0.6 비호환). FA2로 충분해 미추적 |

2장 병렬 효율 1.61×. 결론: **2×H100 @jp, FA2, compile OFF**로 Pro 1세션 실시간이 42 FPS(여유 70%)로 안정적. 지연 하한은 청크 1.12 s + 생성 0.66 s + 전송 ≈ 2 s.

## 리전 배치 (`probe_gpu.py::l4kr`, 2026-09-15)

| 지정 | 실제 배치(GCP) | L4 배정 시간 | 비고 |
|---|---|---|---|
| `region=["ap-northeast"]` | **asia-northeast3 = 서울** | 3대 동시 ~26 s | 한국 사용자에 최적. 단 6대 동시 요청은 한 번 240 s 내 미배정 |
| `region=["jp"]` | asia-northeast1 = 도쿄 | 90 s (H100은 13 s) | |
| `region=["ap"]` | 싱가포르/서울/도쿄 중 가용한 곳 | 19–125 s | 순간 용량에 따라 다름 |
| 미지정 | us-central1 | 12 s | 한국까지 RTT 150 ms+ |
| `ap-melbourne` | — | 즉시 거절 | L4 워커 타입 없음 |

예약형 세션 배치 정책: T−3~5분에 `ap-northeast` → 실패 시 `ap` → `any` 순으로 시도. 컨테이너 안에서 `MODAL_REGION`으로 실제 배치를 확인할 수 있다.

## Lite vs Pro (2026-09-14 A/B, `modal run app.py::ab`, 결과 `ab/`)

같은 아바타·음성(13.8 s)·시드. Pro는 같은 1.3B DiT에 Wan2.1 VAE(잠재 64×64, Lite는 16×16), 청크는 28프레임 전진(겹침 5).

| | Lite | Pro |
|---|---|---|
| 청크 시간 (L4, compile off) | 0.77 s / 24 f → **1.24× 실시간** | 8.63 s / 28 f → **0.13× 실시간** |
| denoise 스텝 | 0.10 s | 1.12 s |
| 14.8 s 클립 렌더 | 12 s | 112 s |
| 피크 VRAM | 5.5 GB | 6.7 GB |
| 선명도 (입 영역 라플라시안 분산) | 71 | 128 (×1.8) — 치아·입술 윤곽·혀가 보임 |

Pro는 스트리밍 재생이 불가(재생보다 7–8배 느림). 쓸 곳은 "최종 렌더" 옵션. Model_Pro 가중치는 Volume에 있음(`fetch_weights --include-pro` 완료).

## PoC 실제 비용 (`modal billing report`, 2026-09-10 → 09-14)

계량 합계 **$20.46**, 크레딧 차감 후 **청구 $0**. 내역: L4 ≈ $3.8(≈4.7 GPU-h: 스튜디오·Lite/Pro 렌더·Lite 라이브), H100 ≈ $15.2(≈3.9 GPU-h: 2×H100 벤치 6회(실패 3회 포함)·1×H100·Pro 라이브 세션 4회와 각 세션 뒤 5분 유휴), H200/B200/A100 프로브 ≈ $0.8, CPU·메모리 ≈ $0.3. 실측 세션 단가: Lite 라이브 L4 **$0.80/hr·세션**, Pro 라이브 2×H100 **$7.90/hr·세션**(+CPU/메모리 ~$0.1) — 아바타가 말하든 대기하든 세션이 GPU를 점유하는 동안 계속 과금.

## 문제가 생기면

- `ConflictError: The app is stopped or disabled` — Modal 콘솔에서 앱을 Stop 했거나 `modal app stop flashhead`를 한 경우. **`modal deploy app.py`** 로 되살린다(새 앱 id가 생기고 `stopped` 항목은 목록에 남지만 무해). 스튜디오는 이 오류를 보면 핸들을 다시 잡는다.
- 콘솔에서 컨테이너만 죽인 경우는 다음 호출에서 자동으로 새 컨테이너가 뜬다(콜드 스타트 비용만).

## 실측 (2026-09-11, L4)

| 구간 | 시간 |
|---|---|
| 컨테이너 기동 → `ready` (import 20–70 s + 가중치 34–99 s + 워밍업 ~31 s) | **85–125 s** |
| 준비된 컨테이너에서 렌더: 참조 준비 0.3 s, 오디오 인코딩 ~0.2 s, 청크당 0.8 s(=0.96 s 분량), 먹스 1–2 s | 오디오 길이의 **약 1.0–1.1배** |
| 스튜디오 종단(TTS + 전송 포함), 11.6 s 멘트 | **19.6 s** |
| 상주 없이 콜드 호출 | 위 두 줄의 합 |

가중치 로딩 시간이 호스트에 따라 3배 차이 나는 것은 Volume 캐시 상태 때문. 생성 자체는 전체의 몇 %라서 GPU를 올려도 의미 없다.

## 호스트 갈아타기 (2026-09-18)

Modal 이 어댑터로 밀려났다. 렌더러 본체는 클라우드 SDK 를 import 하지 않는다.

```
renderer.py    RendererCore — 파이프라인·스케줄러·대기 루프·HLS. modal 없음
               Host          호스트가 제공하는 것 전부: put / get_many /
                             publish_ready / commit — 네 개뿐이다
               LocalHost     한 박스용 구현. 드라이버가 같은 프로세스에 있으니
                             "전송"은 queue.Queue 다
app.py         Modal 어댑터 — ModalHost 가 저 네 개를 Queue/Dict/Volume 에 얹고,
               @app.cls 는 합성으로 코어에 위임한다
run_local.py   평범한 프로세스. Lightning·RunPod·vast.ai·Lambda·책상 밑 박스 공통
Dockerfile     이식 가능한 환경 정의 (Modal 이미지 체인과 나란히 유지)
setup.sh       Docker 를 못 쓰는 박스용 — 같은 단계를 호스트 파이썬에 적용
```

Modal 만 예외이고 나머지 호스트는 전부 "GPU 달린 리눅스 박스"라 같은 타깃이다.
그래서 다음 클라우드로 옮기는 비용이 하루 반이 아니라 한 시간이다.

```bash
./setup.sh                                          # 또는 docker build -t flashhead .
python run_local.py --fetch-weights                 # 한 번, 약 8 GB
python run_local.py --sessions 6 --minutes 3 --gap-s 18
```

경로는 전부 환경변수로 뺐다 — `FLASHHEAD_SRC` / `_WEIGHTS` / `_CACHE` / `_WARMUP` /
`_MAX_SESSIONS` / `_COMPILE` / `_GPU` / `_APP`.

**아직 실측으로 확인 못 한 것**: 크레딧이 없어 Modal 배포도, GPU 박스 실행도 돌려보지
못했다. 확인한 것은 네 모듈이 모두 import 되는 것, 코어에 modal 이 없는 것,
LocalHost 왕복, 그리고 Dockerfile·setup.sh 의 sed 패치가 실제 상류 파일에서
의도대로 도는 것(compile 플래그 양방향, 동기화 8개 제거 후 파싱 통과)까지다.

## H100 실험 (2026-09-17) — 동시 발화는 되지만 경제성은 L4가 낫다

`FLASHHEAD_APP=flashhead-h100 FLASHHEAD_GPU=H100 FLASHHEAD_MAX_SESSIONS=16 modal deploy app.py`
로 L4 배포 옆에 따로 세워 측정했다. 스튜디오는 `studio/studio.env`로 대상을 고른다.

| | L4 | H100 |
|---|---|---|
| GPU-시간 단가(청구 내역 역산) | $0.81 | $3.90 (4.82×) |
| 청크 시간 (Lite, 단일 세션) | 0.746 s | **0.206 s** (3.62× 빠름) |
| 상태교체 / 모델 / 호스트전송 | 0.000 / 0.736 / 0.010 | 0.000 / 0.208 / 0.005 |
| 슬롯당 예산 `floor(0.96/청크)` | 1 | **4** |
| 실측 최대 동시 발화 | 1 (구조적 상한) | **4** — 발화 슬롯의 84~86% |
| 영상 1분당 비용 | **$0.0105** | $0.0139 (1.33× 비쌈) |
| 세션-시간당 비용(발화 17%) | **$0.137** | $0.166 (1.21× 비쌈) |

- 8세션/간격 6초는 예산의 96%라 대기가 발산했다(최대 28.7 s, 영상도 119/134청크).
  12세션/간격 18초는 영상 실시간을 지켰고(behind 끝 0.0) GPU 32%였지만 대기는
  최대 17 s였다. 발화 슬롯 79개 중 40개가 이미 4개로 꽉 찬 상태였다.
- **처리량 3.6배에 가격 4.8배라 순수 연산 경제성은 H100이 불리하다.** H100이
  사는 경우는 동시 발화가 기능 요구사항일 때(L4는 예산 1이라 원리적으로 불가),
  대기 시간이 중요할 때, 그리고 컨테이너 수를 줄여야 할 때다.

### 예산이 정수라 카드 값어치가 계단식으로만 생긴다

`예산 1 → 0.960 s 이하 · 2 → 0.480 · 3 → 0.320 · 4 → 0.240`

L4(0.746 s)보다 1.5배 빠른 카드를 사도 예산은 그대로 1이다. 버려지는 몫이
L4 22%, H100 14%. 과부하 런에서 GPU 점유율 73%인데 대기가 발산한 것이 그
증거다 — 병목은 카드가 아니라 예산이었다. 슬롯마다 남는 소수점을 이월하는
크레딧 방식으로 바꿔야 중간 카드(A10G·L40S·A100)를 비교하는 의미가 생긴다.

### 남은 최적화 (크레딧 소진으로 미검증)

1. **강제 동기화 제거** — 상류 `generate()`가 단계별 시간을 찍으려고 청크당
   `torch.cuda.synchronize()`를 8회 부른다. 걷어내는 패치는 넣고 배포까지 했으나
   **측정 전에 멈췄다**. L4는 스텝 105 ms라 묻히지만 H100은 29 ms다.
2. **`torch.compile`** — `FLASHHEAD_COMPILE=1`. 상류 기본값이 켜짐이고 그 근거가
   지금의 라이브 경로와 같다(상주·고정 shape·수천 청크). 껐던 이유는 오프라인
   짧은 클립이라 더 이상 해당하지 않는다.
3. **배치** — `generate()`가 `x=noise.unsqueeze(0)`으로 항상 배치 1이다. 슬롯마다
   4세션을 순차로 4번 돌린다. 작은 텐서를 따로 던지는 건 H100에 가장 불리한
   사용법이고, 세션들을 한 forward로 묶으면 4배보다 훨씬 덜 걸린다. 다만
   세션별 `ref_img_latent`·`latent_motion_frames`를 쌓아야 해서 파이프라인
   구조 변경이 필요하다. **셋 중 잠재 이득이 가장 크다.**
