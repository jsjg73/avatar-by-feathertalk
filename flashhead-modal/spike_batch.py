"""Spike: does batching the denoise step pay?

One question — is C(B), the time to generate B chunks together, meaningfully
below B × C(1)? If it is, a slot can serve far more speakers than floor(T/C(1))
and the card's cost per session drops. If it is not, the model is already
compute-bound at batch 1 and the three days of integration buy nothing.

Everything here is throwaway. It monkeypatches upstream in place, keeps no
abstractions, and is meant to be deleted once the number is known.

    python spike_batch.py --batches 1,2,4,8

Two upstream functions assume batch 1 and must be fixed before any of this runs:

  rope_apply        takes x[0] and returns unsqueeze(0) — for B>1 it silently
                    computes item 0 and hands it to everyone. No error.
  DiTBlock.forward  context.squeeze(0) / .unsqueeze(0) around the cross-attention.

Because the first failure is silent, correctness is checked before timing: every
item in a batch is given its own noise, and each row of the batched result must
match a separate batch-1 run with that same noise.
"""

import argparse
import os
import statistics
import sys
import time

SRC = os.environ.get("FLASHHEAD_SRC", os.path.expanduser("~/SoulX-FlashHead"))
os.chdir(SRC)
sys.path.insert(0, SRC)

import torch  # noqa: E402
from einops import rearrange  # noqa: E402

from flash_head.src.modules import flash_head_model as M  # noqa: E402


# --------------------------------------------------------------------------
# patch 1: rope_apply over the batch. Every session in a slot has the same
# resolution and frame count, so grid_sizes — and therefore the rotary table —
# is shared; only x carries a batch.
def rope_apply_batched(x, freqs, grid_sizes, use_usp=False, sp_size=1, sp_rank=0):
    b, s, n, c2 = x.shape
    c = c2 // 2
    fq = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes
    seq_len = f * h * w
    freqs_i = torch.cat([
        fq[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fq[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fq[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(1, seq_len, 1, -1)
    xc = torch.view_as_complex(x[:, :seq_len].to(torch.float64).reshape(b, seq_len, n, -1, 2))
    out = torch.view_as_real(xc * freqs_i).flatten(3)
    if s > seq_len:
        out = torch.cat([out, x[:, seq_len:]], dim=1)
    return out.to(x.dtype)


# patch 2: the block's cross-attention. `(b f)` folding is already there; only
# the context and the un-folding assumed a single item.
def block_forward_batched(self, x, context, t_mod, freqs, grid_sizes):
    b = x.shape[0]
    e = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], freqs, grid_sizes)
    x = x + y * e[2]
    f = context.shape[1]
    x_1 = rearrange(self.norm3(x), "b (f l) c -> (b f) l c", f=f)
    context_1 = rearrange(context, "b f m c -> (b f) m c")
    x = x + rearrange(self.cross_attn(x_1, context_1), "(b f) l c -> b (f l) c", b=b)
    y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
    return x + y * e[5]


def patch() -> None:
    M.rope_apply = rope_apply_batched
    blk = next(c for c in (getattr(M, n) for n in dir(M) if isinstance(getattr(M, n), type))
               if hasattr(c, "cross_attn") or "Block" in c.__name__)
    blk.forward = block_forward_batched
    print(f"patched: rope_apply, {blk.__name__}.forward", flush=True)


# --------------------------------------------------------------------------
def decode_batched(vae, z):
    """LtxVAE.decode unsqueezes a batch dim of its own; feed it one instead."""
    return vae.model.decode(vae.un_normalize_latents(z), return_dict=False, target_shape=z.shape)[0]


def denoise(pipe, x0, motion, ref, audio, step_noise):
    """The denoise loop of FlashHeadPipeline.generate, with a batch dim kept.

    Noise is supplied rather than drawn so a batched run and the single runs it
    is compared against see exactly the same numbers.
    """
    ts, n_ts = pipe.timesteps, pipe.num_timesteps
    x = x0.clone()
    for i in range(len(ts) - 1):
        x[:, :, :motion.shape[2]] = motion
        # the head reads its batch from t_mod's rows (docstring: [B*21, C]),
        # so a single shared timestep has to be widened to one row per item
        flow = pipe.model(x=x, timestep=ts[i].expand(x.shape[0]), context=audio, y=ref)
        flow = flow[0] if isinstance(flow, (tuple, list)) else flow
        t_i = (ts[i][:, None, None, None] / n_ts).to(x.dtype)
        t_i1 = (ts[i + 1][:, None, None, None] / n_ts).to(x.dtype)
        x = (1 - t_i1) * (x - flow * t_i) + t_i1 * step_noise[i]
    x[:, :, :motion.shape[2]] = motion
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--reps", type=int, default=6, help="timed repetitions per batch size")
    p.add_argument("--image", default=os.path.expanduser("~/flashhead/inputs/newscaster.png"))
    p.add_argument("--skip-check", action="store_true")
    a = p.parse_args()
    sizes = [int(s) for s in a.batches.split(",")]

    patch()
    from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, get_pipeline
    import numpy as np

    W = os.environ.get("FLASHHEAD_WEIGHTS", os.path.expanduser("~/weights"))
    pipe = get_pipeline(world_size=1, ckpt_dir=f"{W}/SoulX-FlashHead-1_3B",
                        model_type="lite", wav2vec_dir=f"{W}/wav2vec2-base-960h")
    get_base_data(pipe, cond_image_path_or_dir=a.image, base_seed=0, use_face_crop=True)
    pr = get_infer_params()
    n = pr["frame_num"] * pr["sample_rate"] // pr["tgt_fps"]
    audio1 = get_audio_embedding(pipe, np.zeros(n, dtype=np.float32))[:, : pr["frame_num"]].contiguous()

    dev, dt = pipe.device, pipe.param_dtype
    ref1 = pipe.ref_img_latent.unsqueeze(0)
    motion1 = pipe.latent_motion_frames.unsqueeze(0)
    shape = (pipe.config.out_dim, (pr["frame_num"] - 1) // pipe.config.vae_stride[0] + 1,
             pipe.lat_h, pipe.lat_w)
    steps = len(pipe.timesteps) - 1
    print(f"shapes · x {(1, *shape)} · ref {tuple(ref1.shape)} · motion {tuple(motion1.shape)} "
          f"· audio {tuple(audio1.shape)} · steps {steps} · dtype {dt}", flush=True)

    def noise_for(seed, k):
        g = torch.Generator(device=dev).manual_seed(seed)
        return [torch.randn(shape, dtype=dt, device=dev, generator=g) for _ in range(k)]

    def stack(seeds, k):
        per = [noise_for(s, k) for s in seeds]
        return (torch.stack([p[0] for p in per]),
                [torch.stack([p[1 + i] for p in per]) for i in range(k - 1)])

    def run(B, seeds):
        x0, sn = stack(seeds, steps + 1)
        return denoise(pipe, x0, motion1.expand(B, -1, -1, -1, -1),
                       ref1.expand(B, -1, -1, -1, -1),
                       audio1.expand(B, *audio1.shape[1:]), sn)

    # ---- correctness before speed: rope_apply fails silently, so a batch that
    # ---- quietly rendered item 0 four times would otherwise look like a win.
    if not a.skip_check:
        B = max(sizes)
        seeds = [100 + i for i in range(B)]
        with torch.no_grad():
            got = run(B, seeds)
            singles = torch.cat([run(1, [s]) for s in seeds])
        # bfloat16 has an 8-bit mantissa and batching changes GEMM reduction
        # order, so an exact match is not the criterion. What matters is that row
        # j of the batch is row j's computation: it must sit far closer to its own
        # single run than to any other row's. The failure this guards against —
        # rope_apply handing item 0 to everyone — would make these distances equal.
        g, sg = got.float(), singles.float()
        own = [(g[i] - sg[i]).abs().max().item() for i in range(B)]
        other = [min((g[i] - sg[k]).abs().max().item() for k in range(B) if k != i) for i in range(B)]
        print(f"\n정합성 (배치 {B} vs 개별 {B}회, 같은 노이즈)")
        print(f"  자기 짝과의 오차 : {[f'{v:.3f}' for v in own]}")
        print(f"  가장 가까운 남과 : {[f'{v:.3f}' for v in other]}")
        print(f"  비율             : {[f'{o/max(n,1e-9):.1%}' for o, n in zip(own, other)]}  ← 작을수록 확실")
        # 3x closer to its own pair than to any other row. bf16 noise varies run to
        # run; what is being excluded is a row carrying someone else's computation,
        # which would put these distances at parity.
        ok = all(o < n / 3 for o, n in zip(own, other)) and min(other) > 1e-3
        print(f"  판정: {'통과' if ok else '실패'}", flush=True)
        if not ok:
            sys.exit("배치 결과가 개별 결과와 다릅니다. 타이밍은 의미 없습니다.")

    # ---- timing
    print(f"\n타이밍 (각 {a.reps}회, 중앙값)")
    base = None
    for B in sizes:
        seeds = list(range(B))
        with torch.no_grad():
            for _ in range(2):                       # warm up this shape
                run(B, seeds)
            torch.cuda.synchronize()
            ts_den, ts_dec = [], []
            for _ in range(a.reps):
                t0 = time.time()
                z = run(B, seeds)
                torch.cuda.synchronize()
                t1 = time.time()
                v = decode_batched(pipe.vae, z)
                torch.cuda.synchronize()
                ts_den.append(t1 - t0); ts_dec.append(time.time() - t1)
                del v
        den, dec = statistics.median(ts_den), statistics.median(ts_dec)
        tot = den + dec
        base = base or tot
        print(f"  B={B:<2} denoise {den:.3f}s + decode {dec:.3f}s = {tot:.3f}s"
              f" · 청크당 {tot/B:.3f}s · 선형대비 {B*base/tot:.2f}배 이득"
              f" · 0.96s 슬롯이면 {int(0.96//(tot/B))}개", flush=True)


if __name__ == "__main__":
    main()
