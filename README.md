# avatar-by-feathertalk — 라이브 아바타 면접 PoC

**현재 주력: FeatherTalk 기반 아바타 생성 + GPU 동시 세션 용량 확보.** "opentalk"·"FlashHead"는 이전 단계의 이름/모델이고 더 이상 이 프로젝트를 대표하지 않는다 — 아래 "레거시" 항목으로만 남아 있다.

## 구성

| 경로 | 무엇 |
|---|---|
| `model-repos/FeatherTalk/` | 아바타 생성 모델 (업스트림 클론, git 추적 안 함). 로컬 수정분은 [`patches/`](patches/README.md) 참고 |
| [`flashhead-modal/FEATHERTALK_CAPACITY_SETUP.md`](flashhead-modal/FEATHERTALK_CAPACITY_SETUP.md) | **새 GPU 박스에서 0부터 시작하는 절차** — 클론·패치·클린 업로드·의존성 설치·전처리·용량 검증까지 |
| `flashhead-modal/bench_feathertalk_concurrent*.py`, `build_frame_cache.py`, `sweep_concurrent.py` | 동시 세션 용량을 재는 반복 가능한 벤치마크 도구 |
| `flashhead-modal/checkpoints_featherhead_kjs/` | 학습된 체크포인트 (재학습 없이는 복구 불가 — git 추적) |
| [`docs/gpu-serving/`](docs/gpu-serving/README.md) | GPU 위에서 N개 동시 요청을 처리하는 작업 일반에 적용할 용어·착수 요청 템플릿 (에이전트/도구 무관) |
| [`patches/`](patches/README.md) | 업스트림 클론(`model-repos/*`)에 적용한 로컬 변경분 |

## 지금까지 확보된 것 (2026-09-29)

Vast.ai 4090(28코어) 박스, FeatherTalk, kjs 아바타 데이터셋 기준: 공유 memmap 이미지 캐시 적용 후 **동시 24세션까지 실시간(25fps) 유지, 32세션에서 붕괴**. 절차는 [`FEATHERTALK_CAPACITY_SETUP.md`](flashhead-modal/FEATHERTALK_CAPACITY_SETUP.md), 반복 검증은 `sweep_concurrent.py`.

## 레거시 — SoulX-FlashHead / Modal (더 이상 주력 아님)

`flashhead-modal/` 안에는 FeatherTalk 도구 옆에 **이전 단계**의 코드가 같이 남아 있다: SoulX-FlashHead 모델을 Modal GPU에 상주 렌더러로 올렸던 `app.py`/`renderer.py`/`studio/` 및 그 실측 문서([`flashhead-modal/README.md`](flashhead-modal/README.md)). `docs/capacity/`, `docs/pm-briefing/` 도 이 시절 분석(FlashHead 렌더러의 슬롯 스케줄러, 1000세션 브리핑)이다. 지금 새로 작업을 시작한다면 위 "구성" 표의 FeatherTalk 경로를 따라가면 되고, 이 레거시 영역은 참고용으로만 남겨둔 상태다.

## 추적하지 않는 것

업스트림 클론(`model-repos/*`, 각자 `.git`을 가짐), 렌더 산출물, 런타임 상태. 로컬 수정분은 `patches/`에 diff로만 보관한다.
