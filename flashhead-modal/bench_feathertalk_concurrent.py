"""One "session" worth of FeatherTalk real-time throughput, for N-way concurrency benchmarking.

Replicates inference.py's actual per-frame pipeline (image read, face crop, mask,
model forward, paste back) so the measured fps reflects the whole per-frame cost,
not just the tiny UNet's raw compute -- CPU-side crop/landmark work is exactly the
kind of thing that was flagged as the likely real bottleneck once the network
itself got this cheap. No video is written; we only care about frames/sec.

Launched N times in parallel (separate processes, one per "session") against the
same GPU. Each run loads the model once, then loops over the audio feature file
for `--seconds` wall-clock time (looping back to frame 0 if it runs out), counting
how many frames it produced. Writes {frames, seconds} as JSON to --out so the
launcher can aggregate across processes.

Usage:
    python bench_feathertalk_concurrent.py \
        --dataset data/interviewer --audio_feat sync_test_hu.npy \
        --checkpoint ckpt_full/last.pth --seconds 30 --out /tmp/sess_0.json
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from face_utils import (
    FACE_INNER_SIZE,
    compute_face_bbox,
    crop_face,
    extract_inner,
    gather_audio_window,
    hwc_to_chw_tensor,
    mask_mouth,
    read_landmarks,
    reshape_audio_feat,
)
from model import Model, load_checkpoint_state
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
    image_dir = os.path.join(args.dataset, "full_body_img")
    landmark_dir = os.path.join(args.dataset, "landmarks")
    frame_count = sum(name.endswith(".jpg") for name in os.listdir(image_dir))

    model = load_model(args.checkpoint, device)
    picker = FramePicker(frame_count)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.time()
    n_generated = 0
    audio_idx = 0

    with torch.no_grad():
        while time.time() - t_start < args.seconds:
            resource_index = picker.next()
            image = cv2.imread(os.path.join(image_dir, f"{resource_index}.jpg"))
            landmark_path = os.path.join(landmark_dir, f"{resource_index}.lms")
            model_input, face_crop, bbox, original_size = prepare_model_input(image, landmark_path, device)

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

    print(f"[bench] frames={n_generated} seconds={elapsed:.2f} fps={n_generated / elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
