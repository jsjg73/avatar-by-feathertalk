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

## 2. 영상 전달 — WebRTC(LiveKit), HLS 아님

이슈는 전달 방식을 열어뒀지만, 이 프로젝트 자체 문서(`docs/capacity/README.md`)가 이미 "발화 대기 SLA 0.5s, 700ms 넘으면 사람이 머뭇거림으로 지각"이라고 못박아뒀다. HLS는 세그먼트 기반이라 아무리 튜닝해도 보통 수 초 지연이고, 이 SLA와 근본적으로 안 맞는다. **WebRTC로 간다.**

`ai-interview-avatar-proto`가 이미 LiveKit을 쓴다고 했으니, 새 WebRTC 스택을 만들지 않고 **LiveKit Python SDK(`livekit-rtc`)로 세션마다 방(room)에 참가자로 직접 퍼블리시**하는 걸 제안한다:

- 프레임 생성 워커가 매 틱 `LocalVideoTrack`/`LocalAudioTrack`에 프레임을 바로 push — ffmpeg 인코딩·세그먼트 단계가 아예 없어서 그만큼 지연이 준다.
- `ai-interview-avatar-proto` 쪽은 이미 연결된 LiveKit room에서 "아바타"라는 이름의 참가자가 하나 더 들어온 것처럼 구독하면 되고, 새 클라이언트 프로토콜을 안 만들어도 된다.
- **확인 필요(§6)**: 그쪽 LiveKit이 자체 호스팅인지 Cloud인지, room 발급/토큰 방식.

## 3. API 표면

제어 평면(HTTP/WS)과 미디어 평면(LiveKit room)을 분리한다 — 오디오/비디오 바이트가 이 API를 통과하지 않는다.

```
POST /sessions
  { "avatar": "kjs" | "yuna", "livekit_room": "...", "livekit_token": "..." }
  -> { "session_id": "...", "status": "starting" }

WS /sessions/{id}/audio
  클라이언트 -> 서버: 바이너리 프레임, PCM16 mono 16kHz 청크 (live_worker.py의 "audio" 메시지와 동일 포맷)
  서버 -> 클라이언트: {"event": "chunk", "k": int, "speech": bool, "gen_s": float, "behind_s": float}  (관측용, live_worker.py의 status 이벤트 재사용)

POST /sessions/{id}/end
  -> 남은 오디오를 마지막 청크로 밀어넣고 세션 종료 (live_worker.py의 "final" 플래그와 동일)

GET /sessions/{id}
  -> { "status": "running"|"ended", "chunks": int, ... }
```

## 4. 프로세스 구조 — 게이트웨이 + N-프로세스 워커 풀

```
                       ┌─────────────────────────┐
  HTTP/WS 요청  ─────▶ │   Gateway (asyncio)      │
                       │  - 세션 수명주기 관리     │
                       │  - 워커 풀에서 빈 슬롯 배정│
                       │  - 슬롯 없으면 즉시 거절   │  ← 24는 "권장"이 아니라 하드 캡
                       └───────────┬─────────────┘
                                   │ IPC (JSON-line, control_fifo 방식 재사용)
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              Worker #1      Worker #2   ...  Worker #24
              (자체 CUDA      (자체 CUDA         (자체 CUDA
               context,        context,           context,
               memmap 캐시     memmap 캐시        memmap 캐시
               read-only 공유) read-only 공유)    read-only 공유)
                    │              │              │
                    ▼              ▼              ▼
              LiveKit room    LiveKit room    LiveKit room
              (세션별 참가자)  (세션별 참가자)   (세션별 참가자)
```

- 각 워커는 `bench_feathertalk_concurrent_cached.py`의 코어 루프(모델 로드, memmap 캐시 열기, `prepare_model_input`→`model()`→`paste_prediction`)를 그대로 쓰되, 벤치마크의 "합성 프레임 순환" 대신 **실제 세션의 오디오 청크**를 소비하고 **실제 프레임을 LiveKit에 push**하도록 바꾼다 — 새로 만드는 부분이 아니라 기존 루프의 입출력만 실사용으로 교체.
- 게이트웨이는 새 코드다: HTTP/WS 서버, 워커 풀 관리(빈 슬롯 배정/회수), 워커 프로세스 기동·재시작.
- 워커당 아바타 고정이 아니라 **요청마다 어떤 아바타(체크포인트+캐시)를 로드할지 알려줘야** 한다 — 지금 `kjs` 외에 `yuna` 체크포인트(`flashhead-modal/checkpoints_featherhead_yuna/`)도 이미 학습돼 있다. 아바타별 43GB 캐시를 다 상주시킬지, 워커가 세션 시작 시 해당 아바타 캐시를 그때 매핑할지는 아바타 수와 박스 메모리에 달렸다 — `feedback-gpu-multi-request-serving`의 "스케일 전에 메모리 예산부터 계산" 원칙 그대로, 아바타 종류가 몇 개인지부터 확인해야 한다(§6).

## 5. 나중에: 24 이상이 필요해지면

Producer-consumer 배치 서버는 여전히 이론상 맞는 다음 단계지만, **아직 아무도 이걸 제대로(진짜 병렬 전처리 워커 + 배치 GPU 호출을 둘 다 갖춘 형태로) 구현/검증한 적이 없다.** 필요해지면 별도 스파이크로 붙이고, `sweep_concurrent.py`로 반드시 재검증한다 — "될 것 같다"로 넘어가지 않는다.

## 6. 구현 전에 확인해야 할 것

- **LiveKit**: self-host인지 Cloud인지, room/token 발급을 게이트웨이가 직접 하는지 `ai-interview-avatar-proto`가 넘겨주는지.
- **오디오 청크 단위**: `ai-interview-avatar-proto`의 TTS가 실시간 스트리밍 PCM을 주는지, 문장 단위 통짜 WAV를 주는지 — 버퍼링 설계가 달라진다.
- **아바타 개수**: 지금 학습된 건 `kjs`, `yuna` 둘. 동시에 몇 개 아바타가 라이브로 떠 있어야 하는지(면접관마다 다른 아바타?)에 따라 캐시 상주 전략이 바뀐다.
- **거절 정책**: 24슬롯이 다 찼을 때 큐잉할지, 즉시 에러로 거절할지 — 면접이라는 도메인 특성상 "대기시켜서 늦게 시작" 보다 "지금은 자리 없음"이 나을 가능성이 높다는 게 내 의견이나, 제품 쪽 판단 필요.
