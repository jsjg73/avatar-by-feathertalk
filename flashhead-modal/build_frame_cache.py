"""Pre-decode every full_body_img/*.jpg into ONE flat raw file, once.

A naive "cache all frames in memory" per process needs ~43GB for this dataset
(8193 frames at 1620x1080x3 bytes) -- fine for one process, but N processes
each holding their own private 43GB copy exhausts RAM at N=2 on a 62GB box
(confirmed: mass SIGKILL/OOM at N=8). Writing the decoded frames to one flat
file lets every session process np.memmap() it read-only instead -- the OS
page cache backs all of their mappings with the SAME physical pages, so N
sessions cost ~43GB total (once), not 43GB * N.

Usage:
    python build_frame_cache.py --dataset data/kjs
Writes data/kjs/full_body_img_cache.raw and .meta.json alongside it.
"""

import argparse
import json
import os

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()

    image_dir = os.path.join(args.dataset, "full_body_img")
    frame_count = sum(name.endswith(".jpg") for name in os.listdir(image_dir))
    first = cv2.imread(os.path.join(image_dir, "0.jpg"))
    h, w, c = first.shape

    out_path = os.path.join(args.dataset, "full_body_img_cache.raw")
    meta_path = os.path.join(args.dataset, "full_body_img_cache.meta.json")

    mmap_out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(frame_count, h, w, c))
    for i in range(frame_count):
        img = cv2.imread(os.path.join(image_dir, f"{i}.jpg"))
        assert img.shape == (h, w, c), f"frame {i} shape {img.shape} != {(h, w, c)}"
        mmap_out[i] = img
    mmap_out.flush()

    with open(meta_path, "w") as f:
        json.dump({"frame_count": frame_count, "height": h, "width": w, "channels": c}, f)

    print(f"wrote {out_path} ({frame_count} frames, {frame_count*h*w*c/1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
