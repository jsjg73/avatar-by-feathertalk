# JoyVASA + LivePortrait on Apple Silicon

Audit and smoke test of the audio-driven path that OmniRT calls
`fasterliveportrait`. Verdict: **it runs on this Mac**, but only after fixing a
silent PyTorch MPS bug, and only because JoyVASA supplies the audio side —
LivePortrait alone has no audio input at all.

## The architecture, precisely

```
audio -> [JoyVASA: chinese-hubert-base -> DiT -> motion params] -> [LivePortrait: warp -> pixels]
```

- `KlingAIResearch/LivePortrait` is **video-driven**. Its `audio_priority` flag
  only picks which input video's audio track to copy into the output file.
- `jdh-algo/JoyVASA` vendors LivePortrait and adds `DitTalkingHead`, a diffusion
  transformer that turns audio into LivePortrait motion parameters.
- OpenTalking's `mouth_open_multiplier` / `mouth_corner_multiplier` /
  `cheek_jaw_multiplier` exist in **neither** upstream (0 occurrences). OmniRT
  adds them, presumably scaling the motion params. A local adapter would have to
  implement them.
- Only the chinese-hubert motion generator is released; the README says the
  wav2vec2 variant "will be supported later".

## The bug that matters

`torch.nn.functional.grid_sample` returns **silently wrong values on MPS when
the grid argument is non-contiguous**. No exception; the face renders as vertical
smears while the pasted-back background looks fine. `DenseMotionNetwork`
produces a non-contiguous deformation field, so every LivePortrait warp hits it.

Measured against CPU with byte-identical inputs:

| grid | max relative error |
|---|---|
| the real deformation (non-contiguous) | **1.2678** |
| the same tensor `.contiguous()` | **0.0000** |

Isolating it took bisecting the pipeline, because everything else agrees between
CPU and MPS:

| stage | rel error cpu vs mps |
|---|---|
| appearance_feature_extractor | 0.0000 |
| motion_extractor | 0.0000 |
| spade_generator | 0.0003 |
| warping_module `occlusion_map` | 0.0000 |
| warping_module `deformation` | 0.0000 |
| warping_module `out` | **0.5421** |

Synthetic `grid_sample` tests pass on MPS — in-bounds, 17% out-of-bounds, 67%
out-of-bounds, 4D and 5D — because `torch.rand()` returns contiguous tensors.
Contiguity is the whole story.

The fix is one call, applied in `run_mps.py` in each repo so upstream stays
untouched:

```python
def _grid_sample_mps_safe(input, grid, *a, **kw):
    if grid.device.type == "mps" and not grid.is_contiguous():
        grid = grid.contiguous()
    return _grid_sample(input, grid, *a, **kw)
```

## The other three version-drift fixes

All in `run_mps.py`, none requiring upstream edits:

1. `torch.load` defaults to `weights_only=True` since 2.6; the official
   checkpoints pickle `argparse.Namespace` and `PosixPath`. Restored the old
   default for these trusted local files.
2. `src/modules/common.py` declares `enc_dec_mask(..., device='cuda')` and
   `DenoisingNetwork.__init__` calls it without forwarding its own device. The
   result is `register_buffer`'d, so building it on CPU is equivalent.
3. `helper.load_model` constructs `DitTalkingHead` without `device=`, so the
   class default `'cuda'` drives its internal `self.to(device)`. Injected the
   real device.
4. JoyVASA's HubertModel sets `config.output_attentions = True`, which newer
   transformers rejects unless loaded with `attn_implementation="eager"`.

## Does it fix the lip shapes? No — and the premise was wrong

**This whole investigation was launched on a false premise.** The claim that
lip-shape accuracy was a dead end in the built-in models came from never having
run the vowel test on wav2lip, plus ranking the models by mouth-shape
*diversity* (SVD effective rank), which measures jaw-opening variation rather
than accuracy. Once every model is measured with the same test on the **same**
avatar and audio:

| avatar | model | lip pairs | 이 vs 우 | resolved | raw lip |
|---|---|---|---|---|---|
| duo anchor | **wav2lip 384px** | **3.13** | **2.67** | 10/10 | 74.92 |
| duo anchor | quicktalk | 0.61 | 0.58 | 4/10 | 22.14 |
| newscaster | **wav2lip 384px** | 2.29 | 2.32 | 9/10 | 45.05 |
| newscaster | JoyVASA | 2.19 | 1.76 | 10/10 | 89.10 |

wav2lip 384px wins on both avatars, and not by noise-floor accident — its raw
inter-vowel distances are 3.4x QuickTalk's. It is also already integrated,
needs no adapter, and is the fastest of the three (3.0x realtime).

QuickTalk genuinely cannot separate lip rounding from spreading — it fails
이 vs 우 in Chinese too (0.49x), so that is a capability limit rather than the
Korean/chinese-hubert mismatch. Its advantage over wav2lip is motion volume
(1.47x). JoyVASA's advantage is head and expression movement, which wav2lip
cannot do at all. Neither buys lip accuracy.

### Lip shape IS controllable — but only after two fixes

The first sweep concluded the parameters were inert. That conclusion was wrong
for two reasons, both mine:

**1. The diffusion sampling was never seeded.** `DitTalkingHead` draws fresh
noise per call (`dit_talking_head.py:258` and `:280`) and exposes no seed, so
every render was a different draw. Three repeats of the *same* setting gave
이 vs 우 of 0.69 / 2.03 / 2.42 — a 3.5x spread, larger than any difference
between settings. `torch.manual_seed()` before each `execute()` makes it exactly
reproducible (three runs, all 2.09 / 1.25).

**2. `driving_multiplier` scales all 21 keypoints**, so signal and noise grow
together. Only 6 of them are lips (`[6, 12, 14, 17, 19, 20]`, the indices the
`animation_region` branches write). Scaling just those — OmniRT's
`mouth_open_multiplier` in effect — is a real lever. Added as
`JOYVASA_LIP_GAIN` in `src/live_portrait_wmg_pipeline.py`.

Seeded, on the newscaster avatar, the knob is cleanly monotonic in mouth motion:

| `JOYVASA_LIP_GAIN` | noise (mouth motion) | lip pairs | 이 vs 우 | resolved |
|---|---|---|---|---|
| 0.30 | **31.5** | **2.22** | **2.01** | 10/10 |
| 0.45 | 38.2 | 2.19 | 1.73 | 10/10 |
| 0.60 | 42.6 | 2.16 | 1.61 | 10/10 |
| 1.00 (default) | 50.3 | 2.09 | 1.26 | 9/10 |
| 1.40 | 57.4 | 1.98 | 1.10 | 8/10 |
| 1.80 | 64.5 | 1.98 | 1.20 | 7/10 |
| 2.40 | 68.9 | 2.05 | 1.30 | 7/10 |
| 0.45 + `cfg_scale=1.8` | 37.6 | **2.29** | 1.83 | 10/10 |
| 0.45 + `cfg_scale=4.0` | 40.2 | 1.86 | 1.13 | 10/10 |

So the model is over-articulating by default, and **damping improves accuracy**:
gain 0.30 cuts mouth motion 37% while raising 이 vs 우 by 60%. Amplifying does
the opposite. `cfg_scale` above ~2.8 also hurts, which means OpenTalking's
recommended 3.5-4.5 range is backwards for lip clarity.

Best JoyVASA config found (2.22 / 2.01, noise 31.5) is close to wav2lip 384px
(2.29 / 2.32, noise 19.7) and keeps audio-reactive head motion, which wav2lip
cannot produce at all. wav2lip is still ahead on pure lip accuracy, and it is
deterministic — byte-identical output across runs, which JoyVASA only achieves
with an explicit seed.

### The first (unseeded, therefore worthless) sweep, for the record

Sweeping the levers that already exist upstream, scored the same way:

| setting | noise | lip | 이/우 | resolved |
|---|---|---|---|---|
| baseline (Lady Agnew painting) | 12.76 | 1.30 | 1.31 | 8/10 |
| `animation_region=lip` | 5.64 | 2.57 | 2.26 | 10/10 |
| `cfg_scale=4.5` | 10.91 | 1.43 | **0.72** | 9/10 |
| `driving_multiplier=2.0` | 16.53 | 1.25 | 1.32 | 7/10 |
| baseline (newscaster photo) | 55.33 | 2.19 | 1.77 | 10/10 |
| `animation_region=lip` (newscaster) | 67.18 | 2.10 | 1.79 | 10/10 |

Every row above is a single unseeded render, so the ordering between them means
nothing; they are kept only to show what the numbers looked like before the seed
was fixed. The one structural effect that does survive is `animation_region=lip`
collapsing the noise floor on the painting (12.76 → 5.64) by freezing head pose,
while its raw inter-vowel distances actually *fell* (16.56 → 14.51).

## Cost

- `git clone` LivePortrait (76 MB) + JoyVASA (25 MB)
- weights: LivePortrait 2.0 GB, JoyVASA 395 MB, chinese-hubert-base 360 MB
  (LivePortrait's weights are symlinked into JoyVASA's `pretrained_weights/`,
  and the checkpoint wants the literal directory name
  `TencentGameMate:chinese-hubert-base`)
- 5 extra packages in the OpenTalking venv: `tyro`, `pykalman`,
  `imageio-ffmpeg`, `einops`, `omegaconf`. Nothing in the existing stack broke —
  numpy stayed 1.26.4, torch 2.10.0, onnxruntime 1.29.0.
- ~50 s for 78 frames of LivePortrait; ~65 s for a 4 s Korean clip through
  JoyVASA, on an M4 Pro.

## Run it

```bash
cd model-repos/LivePortrait && python run_mps.py                       # video-driven
cd model-repos/JoyVASA && python run_mps.py --animation_mode human \
  -r <portrait> -a <audio.wav>                                          # audio-driven
```
