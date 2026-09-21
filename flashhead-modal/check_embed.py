#!/usr/bin/env python3
"""Does `_audio_embed_batch` give each session its own embedding?

The scheduler now runs wav2vec once per slot instead of once per speaker — it
was 58% of what a live session costs the scheduler outside generation. The
saving is only real if row j of the batch is session j's audio, and the failure
mode worth fearing is the quiet one: the batch path was written by removing an
`unsqueeze(0)`/`squeeze(0)` pair, and the analogous edit in the denoise path
(`rope_apply`) silently handed item 0 to everyone with no error at all.

So this compares, for distinct audio per row, the batched result against
upstream's own `get_audio_embedding` on that row alone.

    python check_embed.py --batch 8

Judgement is relative, not absolute. Batching changes kernel selection and so
the last bits, which means "equal" is the wrong question; "closer to its own
single run than to anyone else's" is the right one, and it is exactly what the
silent failure would break.
"""

import argparse
import os
import sys

SRC = os.environ.get("FLASHHEAD_SRC", os.path.expanduser("~/SoulX-FlashHead"))
os.chdir(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image", default=None)
    a = p.parse_args()

    import numpy as np
    import torch

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from renderer import CKPT, WAV2VEC, RendererCore

    from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, get_pipeline

    pipe = get_pipeline(world_size=1, ckpt_dir=CKPT, model_type="lite", wav2vec_dir=WAV2VEC)
    get_base_data(pipe, cond_image_path_or_dir=a.image or os.path.join(here, "inputs/newscaster.png"),
                  base_seed=0, use_face_crop=True)

    pr = get_infer_params()
    sr, fps = pr["sample_rate"], pr["tgt_fps"]
    n = pr["cached_audio_duration"] * sr
    start_idx = pr["cached_audio_duration"] * fps - pr["frame_num"]
    end_idx = pr["cached_audio_duration"] * fps

    rng = np.random.default_rng(0)
    # Real speech, not noise: wav2vec normalises its input, and white noise would
    # make every row's statistics identical — which is the one case where a mixed
    # up row could still look right.
    t = np.arange(n) / sr
    windows = [(0.4 * np.sin(2 * np.pi * (110 + 37 * i) * t)
                * (0.5 + 0.5 * np.sin(2 * np.pi * (2.1 + 0.3 * i) * t))
                + 0.02 * rng.standard_normal(n)).astype(np.float32)
               for i in range(a.batch)]

    shim = type("Shim", (), {"pipeline": pipe})()
    with torch.no_grad():
        got = RendererCore._audio_embed_batch(shim, windows, start_idx, end_idx)
        singles = torch.cat([get_audio_embedding(pipe, w, start_idx, end_idx) for w in windows])

    print(f"\n배치 {tuple(got.shape)}  vs  개별 {tuple(singles.shape)}")
    if got.shape != singles.shape:
        sys.exit(f"모양이 다릅니다 — 배치 경로가 upstream 과 같은 것을 만들지 않습니다")

    g, s = got.float(), singles.float()
    scale = s.abs().max().item()
    own = [(g[i] - s[i]).abs().max().item() for i in range(a.batch)]
    other = [min((g[i] - s[k]).abs().max().item() for k in range(a.batch) if k != i)
             for i in range(a.batch)]
    print(f"  값의 크기        : {scale:.3f}")
    print(f"  자기 짝과의 오차 : {[f'{v:.5f}' for v in own]}")
    print(f"  가장 가까운 남과 : {[f'{v:.3f}' for v in other]}")
    print(f"  비율             : {[f'{o/max(x,1e-9):.2%}' for o, x in zip(own, other)]}  ← 작을수록 확실")

    ok = all(o < x / 3 for o, x in zip(own, other)) and min(other) > 1e-3
    exact = max(own) < 1e-4 * max(scale, 1e-6)
    print(f"  판정: {'통과' if ok else '실패'}" + ("  (사실상 동일)" if exact else ""))
    if not ok:
        sys.exit("배치 임베딩이 개별 임베딩과 다릅니다 — 슬롯당 1회 최적화를 되돌려야 합니다.")


if __name__ == "__main__":
    main()
