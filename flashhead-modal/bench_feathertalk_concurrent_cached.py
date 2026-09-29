"""Same as bench_feathertalk_concurrent.py, but every resource frame comes from
a shared memory-mapped cache (see build_frame_cache.py) instead of cv2.imread
+ JPEG-decode on every single tick. That decode was measured at ~5.67ms/frame
in isolation -- the single largest non-GPU cost in the per-frame pipeline
(bigger than the face crop/mask prep, ~1ms).

An EARLIER version of this script cached frames in a per-process Python list
-- it worked at N=1 (+44% fps) but mass-OOM-killed every session past N=8,
because 8193 frames at 1620x1080x3 bytes is ~43GB, and N processes each
holding their own copy needs 43GB * N. This version np.memmap()s one shared
file instead: the OS page cache backs every process's mapping with the SAME
physical pages, so N sessions cost ~43GB total (once), not 43GB * N. Run
build_frame_cache.py once before using this script.

paste_prediction() writes into its `image` argument in place, so each tick
uses a fresh .copy() out of the mmap -- writing into the mmap itself would
both corrupt the shared cache for every other process AND let the pasted-in
mouth region accumulate across cycles through the frame set.

Usage: identical to bench_feathertalk_concurrent.py.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from face_utils import gather_audio_window, reshape_audio_feat
from inference import FramePicker, prepare_model_input, paste_prediction, load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--audio_feat", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = np.load(args.audio_feat).astype(np.float32)
    landmark_dir = os.path.join(args.dataset, "landmarks")

    with open(os.path.join(args.dataset, "full_body_img_cache.meta.json")) as f:
        meta = json.load(f)
    frame_count = meta["frame_count"]
    cache_shape = (frame_count, meta["height"], meta["width"], meta["channels"])
    cached_images = np.memmap(
        os.path.join(args.dataset, "full_body_img_cache.raw"), dtype=np.uint8, mode="r", shape=cache_shape
    )

    model = load_model(args.checkpoint, device)
    picker = FramePicker(frame_count)

    landmark_paths = [os.path.join(landmark_dir, f"{i}.lms") for i in range(frame_count)]

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.time()
    n_generated = 0
    audio_idx = 0

    with torch.no_grad():
        while time.time() - t_start < args.seconds:
            resource_index = picker.next()
            image = cached_images[resource_index].copy()
            model_input, face_crop, bbox, original_size = prepare_model_input(image, landmark_paths[resource_index], device)

            audio = reshape_audio_feat(gather_audio_window(features, audio_idx)).unsqueeze(0).to(device)
            audio_idx = (audio_idx + 1) % features.shape[0]

            prediction = model(model_input, audio)[0]
            prediction = (prediction.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            paste_prediction(image, prediction, face_crop, bbox, original_size)

            n_generated += 1

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - t_start

    with open(args.out, "w") as f:
        json.dump({"frames": n_generated, "seconds": elapsed, "fps": n_generated / elapsed}, f)

    print(f"[bench-cached] frames={n_generated} seconds={elapsed:.2f} fps={n_generated / elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
