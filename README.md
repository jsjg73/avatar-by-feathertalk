# opentalk — 라이브 아바타 PoC

맥북 TTS로 시작해 **SoulX-FlashHead를 Modal GPU 위에서 라이브 아바타 서비스로** 굴리기까지의 실험 기록과 코드.

## 구성

| 경로 | 무엇 |
|---|---|
| [`flashhead-modal/`](flashhead-modal/README.md) | **현재 주력.** Modal에 배포하는 상주 렌더러(`app.py`), Pro 라이브용 2랭크 워커(`live_worker.py`), 벤치·프로브 스크립트 |
| [`flashhead-modal/studio/`](flashhead-modal/studio/README.md) | 로컬 데모 웹앱 — 멘트/음성 파일 → 클립, 라이브 세션(Lite·Pro), 아바타 선택, 예열 토글 |
| [`say-tts-shim/`](say-tts-shim/README.md) | macOS `say`를 OpenAI 호환 TTS로 노출하는 shim (초기 단계에서 사용) |
| [`docs/pm-briefing/`](docs/pm-briefing/BRIEFING.md) | 1000 세션 규모 검토 브리핑과 역할극 기록 |
| [`patches/`](patches/README.md) | 상류 저장소(`opentalking`)에 적용한 로컬 변경분 |
| `flashhead-runpod/` | RunPod 경로 (Modal로 대체됨, 참고용) |

## 추적하지 않는 것

상류 클론(`opentalking/`, `model-repos/*` — 각자 `.git`을 가짐), 모델 가중치(`models/`, 수 GB),
렌더 산출물(`studio/jobs/`, `studio/live/`, `*.mp4`), 런타임 상태(`studio.log`, `warm.json`).
가중치는 Hugging Face와 Modal Volume(`flashhead-weights`)에서 받는다.

## 빠른 시작

```bash
cd flashhead-modal && ../opentalking/.venv/bin/modal deploy app.py   # GPU 렌더러 배포
cd studio && STUDIO_HOST=0.0.0.0 ./studio.sh start                  # http://127.0.0.1:8300
```

측정값·구조·주의점은 각 디렉터리 README에 있다.
