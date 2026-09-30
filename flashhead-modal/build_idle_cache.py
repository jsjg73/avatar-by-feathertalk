"""Pre-decode the natural idle-loop frame range into ONE shared flat file,
the same np.memmap-sharing trick as build_frame_cache.py (see that file for
why: N live_server.py workers each holding a private decoded copy would cost
frame_count * H*W*3 bytes * N; memmap-ing one shared file lets the OS page
cache back every worker's mapping with the SAME physical pages instead).

This one is deliberately a SEPARATE, much smaller file from
full_body_img_cache.raw (the full 8193-frame benchmark cache) -- it holds
only the idle-loop range (a few dozen frames, picked for minimal motion +
a clean loop-back point; see docs/live-serving/DESIGN.md SS12 for how that
range was chosen), and live_server.py is the only consumer.

Usage:
    python build_idle_cache.py --dataset data/kjs --start 4565 --end 4643
Writes data/kjs/idle_cache.raw and .meta.json alongside it.
"""

import argparse
import json
import os

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--start", type=int, required=True, help="first resource frame index (inclusive)")
    p.add_argument("--end", type=int, required=True, help="last resource frame index (inclusive)")
    args = p.parse_args()

    image_dir = os.path.join(args.dataset, "full_body_img")
    indices = list(range(args.start, args.end + 1))
    frame_count = len(indices)
    first = cv2.imread(os.path.join(image_dir, f"{indices[0]}.jpg"))
    h, w, c = first.shape

    out_path = os.path.join(args.dataset, "idle_cache.raw")
    meta_path = os.path.join(args.dataset, "idle_cache.meta.json")

    mmap_out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(frame_count, h, w, c))
    for i, idx in enumerate(indices):
        img = cv2.imread(os.path.join(image_dir, f"{idx}.jpg"))
        assert img.shape == (h, w, c), f"frame {idx} shape {img.shape} != {(h, w, c)}"
        mmap_out[i] = img
    mmap_out.flush()

    with open(meta_path, "w") as f:
        json.dump({
            "frame_count": frame_count, "height": h, "width": w, "channels": c,
            "resource_start": args.start, "resource_end": args.end,
        }, f)

    print(f"wrote {out_path} ({frame_count} frames, resource {args.start}-{args.end}, "
          f"{frame_count*h*w*c/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
