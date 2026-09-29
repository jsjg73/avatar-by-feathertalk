# 상류 저장소에 적용한 로컬 패치

상류 클론(`opentalking/`, `model-repos/*`)은 각자 `.git`을 가진 남의 저장소라
이 저장소에서 추적하지 않는다. 대신 우리가 만든 변경분만 패치로 남긴다.

| 패치 | 대상 | 내용 |
|---|---|---|
| `opentalking-ffv1-intermediate.patch` | `opentalking/opentalking/video_creation.py` | 중간 산출물을 손실 압축 mp4 대신 **FFV1 무손실 `.mkv`** 로 저장하고 먹스에 `-crf 18`. 확장자로 fourcc를 고르게 해 재인코딩 열화를 없앰. 적용 후 테스트 175건 통과. |
| `featherhead-mps-fallback.patch` | `model-repos/FeatherTalk/inference.py` | `run()` 의 device 선택에 `mps` 폴백 추가 (`cuda` 없으면 `mps`, 그것도 없으면 `cpu`). 맥에서 로컬 테스트할 때만 필요 — GPU 박스(Linux, CUDA)에는 적용하지 않아도 무방. |

적용:

```bash
cd opentalking && git apply ../patches/opentalking-ffv1-intermediate.patch
cd model-repos/FeatherTalk && git apply ../../patches/featherhead-mps-fallback.patch
```
