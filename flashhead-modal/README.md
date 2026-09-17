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
