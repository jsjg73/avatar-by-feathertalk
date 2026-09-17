# SoulX-FlashHead on RunPod

Everything needed to render the Korean clip on a rented GPU, since FlashHead's
upstream is CUDA-native (`flash_attn` / `sageattention` / `xfuser`) and its
published numbers are all on an RTX 4090.

## Pod

| | |
|---|---|
| GPU | RTX 4090 24 GB (what upstream benchmarks). A5000 / A6000 / L40S also work. |
| Template | any PyTorch 2.x + CUDA 12.x image |
| Container disk | 40 GB+ (weights 14.3 GB, or ~8 GB with `Model_Pro` excluded) |

Model_Lite runs 96 FPS on a single 4090; Model_Pro 10.8 FPS. The "two RTX 5090"
in upstream's README is only for *realtime* Pro — offline single-GPU is fine.

## Steps

1. Start the pod, open its terminal.
2. `bash setup.sh` — clones upstream, installs deps, downloads weights (~9 GB).
3. Upload `inputs/` to `/workspace/SoulX-FlashHead/inputs/`.
4. `bash run_korean.sh` — or pick inputs:
   `bash run_korean.sh inputs/woman.png inputs/korean_32s.wav out.mp4`
5. Download the resulting `.mp4`, then stop the pod.

`flash_attn` and `sageattention` are installed on a best-effort basis; if their
builds fail the model falls back to `F.scaled_dot_product_attention`, which is
slower but correct.

## inputs/

| file | what |
|---|---|
| `korean_33s.wav` | 33 s, macOS `say -v "Yuna (Premium)"`, 16 kHz mono — a spoken demo notice, not a news report |
| `korean_32s.wav` | 32 s, same voice, describes the technical setup |
| `newscaster.png` | 790×1166, OpenTalking's bundled synthetic anchor |
| `woman.png` | 858×1072, another bundled synthetic portrait |

Audio is already 16 kHz mono, which is what upstream's own example
(`podcast_sichuan_16k.wav`) uses. To make more:

```bash
say -v "Yuna (Premium)" -o out.wav --data-format=LEI16@16000 "문장"
```

## Why not on the Mac

The port got further than expected — `xfuser` imports with one small stub, the
attention falls back to SDPA, and the pipeline imports cleanly (shims in
`model-repos/SoulX-FlashHead/mps_compat.py`: stub
`xfuser.core.sparge_attention.block_mask`, redirect `torch.cuda.synchronize`,
override `usp_device.get_device`, neutralise `torch.cuda.amp.autocast`). It was
never run end to end, so treat it as unverified. Even working, a 1.3B video
diffusion model on an M4 Pro would be far off the 4090 numbers.
