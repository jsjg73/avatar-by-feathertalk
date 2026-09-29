# FeatherTalk 실시간 서빙 API — 설계안

[issue #1](https://github.com/jsjg73/avatar-by-feathertalk/issues/1) 요청에 대한 답. 구현 전 검토용 — 코드는 아직 없다.

## 0. 먼저 바로잡을 것 — 이슈의 전제 하나가 틀렸다

이슈는 "이미 증명된 concurrency 용량(N=24)을 그대로 살리는 producer-consumer 구조를 그대로 따르면 된다"고 적었는데, **이 둘은 서로 다른 아키텍처다.**

실측된 N=24는 `sweep_concurrent.py` + `bench_feathertalk_concurrent_cached.py`로 나온 값이고, 이건 **N개의 별도 OS 프로세스**(각자 CUDA context, 공유하는 건 memmap 이미지 캐시뿐)다. `docs/gpu-serving/README.md`가 원칙으로 권하는 **producer-consumer / continuous batching**(GPU를 프로세스 하나가 소유하고 여러 세션 요청을 배치로 묶는 것)은 이 프로젝트에서 딱 한 번 시도했다가 **실패한 방식**이다 — naive 버전이 N=1에서 -58%, N=8에서 -69% 역행했다(같은 문서 §4, `feedback-gpu-multi-request-serving` 메모리).

즉 지금 유일하게 실측으로 뒷받침되는 건 **N-프로세스 구조**다. 아래 설계는 이 위에 짓는다. Producer-consumer 배치는 24 세션으로도 부족해질 때 검토할 **미검증된 차기 옵션**으로 남겨둔다 (§5).

## 1. 세션 프로토콜 — control-line 설계는 재사용, 실행 모델은 아니다

`live_worker.py`(SoulX-FlashHead, 레거시)의 아이디어 중 **재사용할 것**과 **버릴 것**을 나눈다.

| `live_worker.py`의 요소 | 재사용? |
|---|---|
| JSON control-line 프로토콜 (`start`/`audio`/`end`), PCM int16 청크 전송 | **재사용** — 세션 라이프사이클 형태로 그대로 맞다 |
| `idle_timeout` / `max_seconds` / `lead_chunks`로 세션 종료·페이싱 관리 | **재사용** — 검증된 상태 머신 |
| 멀티 GPU torchrun + xfuser 시퀀스 병렬 (세션 1개를 여러 GPU에) | **버림** — FeatherTalk은 세션 1개가 GPU 1개도 안 채운다(66MB 체크포인트, 144×144). 여기 쓸 자원 배분 문제가 다르다 |
| ffmpeg → HLS 출력 | **버림** — §2에서 이유 설명 |

## 2. 영상 전달 — 자체 WS+fMP4, LiveKit 아니었다 (재정정, §8)

이슈는 전달 방식을 열어뒀지만, 이 프로젝트 자체 문서(`docs/capacity/README.md`)가 이미 "발화 대기 SLA 0.5s, 700ms 넘으면 사람이 머뭇거림으로 지각"이라고 못박아뒀다. HLS는 세그먼트 기반이라 아무리 튜닝해도 보통 수 초 지연이고, 이 SLA와 근본적으로 안 맞는다.

**처음엔 WebRTC(LiveKit)로 간다고 썼다가(§6b에서 그 근거를 한 번 정정했다가) §8에서 다시 뒤집었다.** 이유: 설계 도중 이 repo에 **이미 검증된 저지연 커스텀 전송 방식이 있다는 걸 발견했다** — `flashhead-modal/live_server.py`가 fMP4를 WebSocket으로 실시간 스트리밍하고(HLS 세그먼트 파일 폴링이 아니라 fragment 하나씩 push), 브라우저는 MediaSource Extensions로 받아 재생한다. `studio_live.html` + `studio_live_server.py`로 텍스트 입력 → TTS → 이 서버 → 브라우저 재생까지 **엔드투엔드로 이미 동작 중**이었다(2026-09-23 작성, 단일 세션 데모).

새 WebRTC 스택을 도입하는 대신 **이 방식을 multi-session으로 확장**한다 — 자세한 이유와 필요한 변경은 §8.

## 3. API 표면

제어 평면(세션 시작/종료/발화)과 미디어 평면(fMP4 WS 스트림)을 분리한다. §8의 최종 확인(문장 단위 청크로 충분, 서브-문장 스트리밍 불필요)을 반영한 **최종 계약**:

```
POST /sessions
  { "avatar": "kjs" }   # 파일럿은 아바타 1개 고정이라 생략 가능해도 됨
  -> 200 { "session_id": "..." }
  -> 503 { "error": "no_capacity" }   # 24슬롯 다 찼을 때, 큐잉 없이 즉시 거절 (§6 확인 결과)

POST /sessions/{id}/speak
  발화 조각(part) 하나 = 완성된 WAV 파일 하나 (multipart, live_server.py의 /speak와 동일).
  한 턴 안에서 여러 조각을 순서대로 여러 번 호출 — 워커 안에서 큐에 쌓여 순서대로 재생된다.
  -> { "queued_frames": int, "seconds": float }

POST /sessions/{id}/interrupt
  바지인 인터럽트 — 배정된 워커의 대기 큐를 비운다(재생 중인 조각은 자연스럽게 끝까지 재생하거나, 즉시 끊을지는
  §9에서 정할 구현 디테일). 서브-문장 단위로 자를 필요는 없다는 게 issue #1 최종 확인.

GET /sessions/{id}/ws  (WebSocket)
  서버 -> 클라이언트: fMP4 fragment 바이너리 스트림, 게이트웨이가 배정된 워커의 /ws에 붙어 그대로 릴레이
    (init segment 먼저, 이후 fragment마다 — live_server.py의 /ws와 동일 프로토콜)

POST /sessions/{id}/end
  -> 세션 종료, 워커를 정지 상태로 되돌려 워커 풀에 반납

GET /sessions/{id}
  -> { "status": "running"|"ended", "speaking": bool, "pending": bool }  (워커의 /status를 그대로 프록시)
```

게이트웨이가 HTTP(`/speak`,`/interrupt`,`/end`,`GET`)와 WS(`/ws`) 둘 다 프록시하는 이유: Vast.ai 박스는 보통 공인 포트 하나(또는 소수)만 매핑돼 있어서, 워커 24개의 포트를 전부 외부에 열어야 하는 구조보다 게이트웨이 포트 하나만 열면 되는 쪽이 실제 배포에 맞다.

## 4. 프로세스 구조 — 게이트웨이 + N-프로세스 워커 풀

```
                       ┌─────────────────────────┐
  HTTP/WS 요청  ─────▶ │   Gateway (asyncio)      │
                       │  - 세션 수명주기 관리     │
                       │  - 워커 풀에서 빈 슬롯 배정│
                       │  - 슬롯 없으면 즉시 거절   │  ← 24는 "권장"이 아니라 하드 캡
                       │  - 클라이언트 WS ↔ 배정된 │
                       │    워커의 /ws 를 프록시   │
                       └───────────┬─────────────┘
                                   │ 세션마다 워커 1개 배정
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              Worker #1      Worker #2   ...  Worker #24
          (live_server.py 1개 프로세스, 포트 하나씩)
              (자체 CUDA      (자체 CUDA         (자체 CUDA
               context,        context,           context,
               memmap 캐시     memmap 캐시        memmap 캐시
               read-only 공유) read-only 공유)    read-only 공유)
                    │              │              │
                    ▼              ▼              ▼
              자체 fMP4/WS    자체 fMP4/WS    자체 fMP4/WS
              (세션 1개 전용)  (세션 1개 전용)  (세션 1개 전용)
```

- 각 워커는 **`live_server.py`를 거의 그대로 쓴다** — 모델 로드, 25fps 생성 루프, ffmpeg fMP4 인코딩, WS 브로드캐스트가 이미 다 있다. 바꿀 건 §8 참고(전역 `STATE` 하나 → 프로세스당 세션 하나는 그대로 유지, 오디오 입력만 "완성 WAV 업로드"에서 "스트리밍 PCM"으로 교체).
- 게이트웨이는 새 코드다: HTTP 세션 관리, 워커 풀(빈 슬롯 배정/회수), 워커 프로세스 기동·재시작, 클라이언트의 오디오/WS 연결을 배정된 워커로 프록시.
- 워커당 아바타 고정이 아니라 **요청마다 어떤 아바타(체크포인트+캐시)를 로드할지 알려줘야** 한다 — 지금 `kjs` 외에 `yuna` 체크포인트(`flashhead-modal/checkpoints_featherhead_yuna/`)도 이미 학습돼 있다. 아바타별 43GB 캐시를 다 상주시킬지, 워커가 세션 시작 시 해당 아바타 캐시를 그때 매핑할지는 아바타 수와 박스 메모리에 달렸다 — `feedback-gpu-multi-request-serving`의 "스케일 전에 메모리 예산부터 계산" 원칙 그대로, 아바타 종류가 몇 개인지부터 확인해야 한다(§6).

## 5. 나중에: 24 이상이 필요해지면

Producer-consumer 배치 서버는 여전히 이론상 맞는 다음 단계지만, **아직 아무도 이걸 제대로(진짜 병렬 전처리 워커 + 배치 GPU 호출을 둘 다 갖춘 형태로) 구현/검증한 적이 없다.** 필요해지면 별도 스파이크로 붙이고, `sweep_concurrent.py`로 반드시 재검증한다 — "될 것 같다"로 넘어가지 않는다.

## 6. 확인 결과 (issue #1 코멘트, jsjg73 답변)

- **LiveKit — 저희가 새로 정하는 결정이었다.** `ai-interview-avatar-proto`는 HeyGen/LiveAvatar의 `/v1/sessions/start` 응답(`livekit_url`/`livekit_client_token`)을 그대로 통과시켜 쓰던 순수 소비자였고, LiveKit을 직접 운영한 적이 없다. **LiveKit Cloud로 제안한다** — self-host SFU/TURN 운영은 아바타 1개짜리 파일럿엔 과한 오퍼레이션 부담이고(이 프로젝트가 Vast.ai를 고른 이유와 같은 논리 — `gpu-vendor-survey-2026-09` 메모리), Cloud는 무료 티어 + REST API로 room 생성·토큰 발급이 바로 된다. 게이트웨이가 LiveKit Cloud 프로젝트의 API 키/시크릿으로 `livekit-server-sdk`(Python)를 통해 세션마다 room+토큰을 만들고, 그 값을 §3의 `POST /sessions` 응답에 그대로 실어 돌려준다.
- **오디오 청크 단위 — 실시간 스트리밍 PCM 확정.** `ai-interview-avatar-proto`의 TTS 엔진 3종(OpenAI/Gemini/Realtime) 전부 `AsyncIterator[bytes]`로 청크 스트리밍하고, 문장 전체를 기다리는 경로는 없다. 버퍼링은 "오디오가 끊김 없이 점진적으로 들어온다"는 전제로 설계.
- **아바타 개수 — 1개로 파일럿 확정** (`kjs`, 유일하게 N=24까지 실측된 아바타). `yuna` 체크포인트는 존재하지만 이번 범위 밖 — 캐시 상주 전략은 단일 아바타 기준으로 단순화.
- **거절 정책 — 즉시 거절 확정.** 24슬롯 초과 시 큐잉 없이 바로 에러.

## 6b. 정정 — "프론트 변경 없음"은 틀린 전제였다 (LiveKit 검토 당시)

§2에서 LiveKit을 고른 이유로 든 "그쪽 클라이언트가 이미 LiveKit 필드만 보고 동작해서 백엔드만 바꾸면 된다"는 확인해보니 틀렸다. 실제로는:

- 프론트는 `livekit-client` 직접 사용도, 단순 `<video>` 재생도 아니고 **`@heygen/liveavatar-web-sdk`의 `LiveAvatarStreamingSession`을 상속**해서 쓴다 (`web/src/avatar/useAvatarConnection.ts:115`). `agent.speak`/`agent.interrupt`/`avatar.speak_started` 같은 **HeyGen 벤더 전용 WebSocket 프로토콜**이 이 SDK 안에 있다.
- 이 SDK가 내부적으로 LiveKit을 쓰긴 하지만(영상/음성 트랙을 `self._remoteVideoTrack` 같은 private 필드에서 타입 캐스팅으로 꺼내 씀 — `ai-interview-avatar-proto` 쪽도 리버스 엔지니어링해서 쓰는 수준), 이 SDK 클래스 자체는 HeyGen 전용이라 **백엔드를 뭘로 바꾸든(LiveKit/WHIP/직접 WebRTC 무엇이든) 어차피 못 쓴다.**
- 즉 **"프론트 변경 최소화"는 전송 방식을 고르는 기준이 될 수 없다** — 어느 쪽을 골라도 프론트는 새로 짜야 한다. 오히려 HeyGen SDK가 강제하던 speak/interrupt 확인 로직이 없어지니 **더 단순해질 여지**가 있다 (jsjg73 답변, issue #1).
- 이 시점까지는 "LiveKit을 계속 추천하는 이유는 검증된 오픈소스 WebRTC 스택 재사용"이라고 결론 내렸었다 — **§8에서 다시 뒤집힌다.**

## 7. 남은 것

~~LiveKit Cloud 프로젝트 자체는 아직 없다~~ → §8에서 LiveKit 자체를 안 쓰기로 했으므로 무관해짐. (계정은 만들어졌으나 안 쓰임 — 무료 티어라 비용 문제는 없음.)

## 8. 재정정 — 이미 있는 걸 발견했다: `live_server.py`

LiveKit 결정을 내릴 때 **이 repo에 이미 단일 세션용 실시간 서버가 동작 중이라는 걸 몰랐다.** 설계를 이어가기 전에 이걸 반영해야 했다.

### 뭐가 있었나

`flashhead-modal/live_server.py`(2026-09-23, 아직 git 미추적) + `studio_live.html` + `studio_live_server.py`:

- FastAPI 상주 프로세스, 모델 1회 로드, 25fps 생성 스레드.
- `POST /speak`: WAV 파일 하나(발화 전체)를 받아 feather_hubert 특징 추출 → 프레임 큐에 적재.
- 영상 출력: ffmpeg가 fMP4 fragment를 stdout으로 뱉고, `_stdout_reader`가 fragment 단위로 잘라 `GET /ws`에 붙은 모든 클라이언트에 브로드캐스트. **HLS 세그먼트 파일도, WebRTC도 아닌 제3의 방식** — 세그먼트 폴링이 없어 HLS보다 훨씬 저지연이고, WebRTC의 SFU/ICE 복잡도도 없다.
- 브라우저(`studio_live.html`)는 MediaSource Extensions로 이 WS 스트림을 받아 재생 — 텍스트 입력 → OpenAI → macOS TTS(`studio_live_server.py`가 오케스트레이션) → 이 서버 → 재생까지 **이미 엔드투엔드로 동작한다.**

### 결정 — LiveKit 도입 안 함, 이 방식을 multi-session으로 확장

이유:
1. **이미 저지연으로 검증됨** — SLA(0.5s)를 만족하는지 실측까지는 아직 안 했지만, HLS류의 세그먼트 지연 문제 자체가 구조적으로 없다.
2. **프론트는 §6b에서 확인했듯 어차피 새로 짠다** — LiveKit 클라이언트 SDK를 새로 배우나, 이 repo의 WS+MSE 프로토콜을 새로 배우나 저쪽 비용은 비슷하다. 그렇다면 우리 쪽에 **이미 동작하는 코드가 있는 쪽**이 훨씬 작은 작업이다.
3. LiveKit(Cloud든 self-host든)을 붙이는 건 새 외부 서비스 의존성 + 새 코드(§4의 room/토큰 발급, `livekit-rtc` 퍼블리시 코드)를 통째로 더하는 것 — 반면 이 방식은 **기존 파일을 multi-session으로 확장**하는 훨씬 작은 델타다.

### 확장에 필요한 변경 (multi-session화)

- `STATE`가 전역 dict 하나 — 이건 **그대로 둬도 된다.** 프로세스 하나가 세션 하나를 담당하는 §4의 N-프로세스 구조와 이미 맞다(전역 STATE = "이 프로세스의 세션 상태"). 바꿀 필요 없음.
- 게이트웨이가 필요 — 지금은 사람이 직접 포트 하나로 `live_server.py`를 띄우고 URL을 하드코딩(`studio_live.html`의 `116.127.115.27:51853`)해서 씀. 세션마다 빈 워커(포트)를 배정하고 클라이언트를 그리로 연결해주는 게 §4의 게이트웨이 역할.
- **오디오 입력 — 정정, 훨씬 작은 변경이다 (jsjg73 지적, issue #1).** 처음엔 "PCM을 바이트 스트림으로 받아야 한다"고 가정했는데 틀렸다. `/speak`가 파일 하나를 통째로 받는 건 API 설계가 아니라 **`_extract_features`가 클립 하나를 통째로 특징 추출하기 때문**이고(`PAD_MS` 패딩도 "클립 가장자리엔 CNN이 볼 context가 없다"는 이유 — 즉 클립 경계마다 패딩이 필요하다는 뜻), `ai-interview-avatar-proto`의 TTS는 바이트 단위 연속 스트림이 아니라 **문장이 완성될 때마다 그 문장의 PCM을 순차적으로** 내놓는다. 그렇다면 진짜 필요한 변경은:
  - `/speak`를 문장 하나마다 여러 번 호출 — 각 문장을 지금처럼 "완성된 짧은 클립"으로 취급(패딩 로직 그대로, feather_hubert 내부 안 건드림).
  - `STATE["next"]`(슬롯 하나, 새 `/speak`가 오면 덮어씀 — 지금은 "새 턴이 이전 턴을 바로 끊는" 바지인 용도로 의도된 동작)를 **큐**로 바꿔서, 한 턴 안의 문장들이 순서대로 재생되게 한다. 턴이 바뀔 때(사용자가 새로 말할 때)의 바지인 인터럽트는 큐를 비우는 것으로 유지 가능.
  - `get_feather_hubert_from_16k_speech`나 다른 모델 코드는 안 건드린다.

  즉 "실시간 스트리밍 아키텍처를 새로 설계"가 아니라 **"단일 슬롯 → 큐" 정도의 작은 변경**이다. 진짜 서브-문장 단위(단어 중간에 잘리는) 스트리밍이 필요한 게 맞는지는 이슈에서 재확인 중.
- fMP4 WS 출력 프로토콜(`/ws`)은 세션 프록시 경로만 붙이면 그대로 재사용.

### 재사용 범위 확인 (jsjg73 질문, issue #1)

- **재사용 대상은 `live_server.py`(렌더링·스트리밍 엔진)뿐이다.** `studio_live_server.py`(macOS `say` TTS + 논스트리밍 OpenAI 챗)는 로컬 데모 전용 오케스트레이터로, `ai-interview-avatar-proto`의 실제 TTS/LLM과 무관하니 참고만 하고 재사용하지 않는다.
- **하드코딩된 박스 IP는 세션별로 동적 교체된다.** `studio_live.html`의 WS URL, `studio_live_server.py`의 `BOX_URL`은 데모의 고정값이고, 실제로는 §3의 게이트웨이가 `POST /sessions` 응답의 `media_ws_url`에 세션에 배정된 워커 주소를 실어 돌려준다.

### 이슈에 알려야 할 것 (완료)

지난 코멘트에서 "LiveKit로 갑니다"라고 답한 걸 정정했고, 파일 3개(+데모용 `fixed_lines/`)를 커밋·푸시해서 프론트가 실제 코드를 보고 확인했다.

## 11. 2026-09-29 종료 시점 상태 (내일 이어서 볼 것)

구현·실전 배포·`ai-interview-avatar-proto` 연동 테스트까지 오늘 하루에 다 진행했다. 이 문서 위쪽(§0~10)은 구현 *전* 설계 논의라 지금 코드 상태와 살짝 어긋난다 — 실제로 어떻게 됐는지는 아래가 최신이다.

### 됐다

- `gateway.py`(신규) + `live_server.py`(큐 기반으로 수정) 구현·커밋 완료. 절차는 `flashhead-modal/FEATHERTALK_CAPACITY_SETUP.md` §7.
- 실제 Vast 박스에서 24워커 엔드투엔드 검증 (세션 생성/발화 큐잉/용량 초과 거절/WS 스트림).
- **영상 깨짐 버그 근본 원인 발견 + 수정 + 검증 완료** (커밋 `3d8a90b`) — 원인은 워커들이 FIFO 경로(`hls_dir`)를 공유해서 서로의 파이프를 지워버린 것, 메모리와 무관했다. `_make_fifos` 경로에 포트를 넣어 고유하게 만들어 해결.
- 모니터링 1단계(외부 감시 스크립트 `monitor_watch.sh`, 커밋 `52df7fe`) 배포 — `docs/live-serving`이 아니라 별도 계획 파일 `/Users/kjs0703/.claude/plans/parsed-beaming-frost.md` 참고 (게이트웨이 자체 헬스체크·`/health`·워커 자동재기동은 W2~W6, 아직 미배포).

### 안 됐다 — 내일 이어갈 것

1. **`/speak` 메모리 누수 (미해결, 최우선)**. 호출마다 ~1.5MB 안 줄어드는 채로 쌓임 — 이미 워커 3개가 이걸로 OOM 사망한 전적 있음. 오늘 확인한 것:
   - 오디오 처리(`_pad_wav_with_silence`+`_extract_features`)만 단독 실행 → 워밍업 후 평평함, **누수 아님**.
   - 영상 프레임 처리(`prepare_model_input`+모델+`paste_prediction`)만 단독 실행 → 역시 평평함, **누수 아님**.
   - → 남은 용의선: **FastAPI `/speak` 엔드포인트의 HTTP/UploadFile 레이어**, 또는 **큐(`STATE["queue"]`)↔`_gen_loop` 스레드 연동** 자체. 아직 직접 재현·격리 못 함.
   - 내일 시도할 것: 실제 HTTP 서버(우버콘/FastAPI TestClient 등)를 통해 호출하되 위 두 계산 부분은 mock으로 스킵 — HTTP 레이어만 남기고 반복 호출해서 누수 재현되는지 확인. 재현되면 그게 원인.
2. **박스 컨테이너 프로세스 상한 발견** — `cgroup pids.max=3328`, 24워커만으로 이미 3327/3328 사용 중. 확장(워커 수를 더 늘리거나, 모니터링/디버깅 프로세스를 추가로 띄우는 것) 계획 세울 때 이 한도를 먼저 확인할 것. 새 박스를 빌리면 이 값도 다시 확인해야 한다(호스트마다 다를 수 있음).
3. **박스는 오늘 밤 정지(=삭제)했다.** 내일은 새 인스턴스를 빌려야 한다 — `FEATHERTALK_CAPACITY_SETUP.md`대로 셋업하되, 전처리 결과는 로컬 백업(`model-repos/FeatherTalk/data/kjs/backup_2026-09-29/preprocessed_backup.tar.gz`, 2.4GB)에서 복원하면 `process.py`(15~20분) 다시 안 돌려도 된다 — `build_frame_cache.py`만 다시 돌리면 됨.
4. 모니터링 W2~W6(워커 로그 캡처, 게이트웨이 헬스체크, `/health` 엔드포인트, 게이트웨이 자체 감독, dead 워커 자동재기동)은 아직 미배포 — 새 박스 세팅할 때 같이 넣을지 결정할 것.
