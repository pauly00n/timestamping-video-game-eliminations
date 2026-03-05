#!/usr/bin/env python3
"""
Kill Banner Detection and Clip Extraction Pipeline

A single-file pipeline that trains a YOLOv8‑nano model to detect kill‑banner HUD
events in gameplay footage, runs inference to timestamp eliminations, extracts
highlight clips, evaluates accuracy, and provides a simple template‑matching
baseline for comparison.

Usage:
----------------
❯ python kill_banner_pipeline.py train \
        --data data/annotations.yaml \
        --weights yolov8n.pt \
        --epochs 50

❯ python kill_banner_pipeline.py infer \
        --video vods/match1.mp4 \
        --weights runs/detect/train/weights/best.pt \
        --out detections.json

❯ python kill_banner_pipeline.py extract-clips \
        --video vods/match1.mp4 \
        --detections detections.json \
        --output_dir clips/

❯ python kill_banner_pipeline.py baseline \
        --video vods/match1.mp4 \
        --template assets/banner_template.png \
        --out detections_baseline.json

❯ python kill_banner_pipeline.py evaluate \
        --gt ground_truth.json \
        --pred detections.json

Dependencies:
------------
- ultralytics>=8.0.0  # YOLOv8 wrapper
- opencv-python
- numpy
- tqdm
- ffmpeg (system binary)

Install them via:
    pip install ultralytics opencv-python numpy tqdm

The annotations.yaml should follow Ultralytics dataset format, e.g.:

```
path: data
train: images/train
val: images/val
nc: 1
names: [kill_banner]
```

Make sure the directory tree under `path` contains `images/` and `labels/`
sub‑folders with YOLO TXT annotations.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO



# Utility helpers

def ffmpeg_cut(input_path: Path, start_s: float, duration_s: float, out_path: Path):
    # Cut a clip using ffmpeg without re‑encoding (fast).
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def temporal_nms(detections: List[Tuple[float, float]], window: float = 3.0) -> List[float]:
    # Merge detections within `window` seconds, keeping the highest-confidence timestamp.
    detections.sort(key=lambda x: x[0])
    clusters: List[List[Tuple[float, float]]] = []
    for t, conf in detections:
        if not clusters or t - clusters[-1][0][0] > window:
            clusters.append([(t, conf)])
        else:
            clusters[-1].append((t, conf))
    return [max(cluster, key=lambda x: x[1])[0] for cluster in clusters]


def frame_to_time(frame_idx: int, fps: float) -> float:
    return frame_idx / fps


def time_to_frame(time_s: float, fps: float) -> int:
    return int(round(time_s * fps))

# Commands

def train_yolov8(args):
    model = YOLO(args.weights)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    print("Training complete. Best weights:", results.best)


def infer_yolov8(args):
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, int(round(fps / args.rate)))  # sample at ~rate FPS

    detections = []  # (time_s, confidence)
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Inferring")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            # Ultralytics models accept numpy arrays
            results = model.predict(frame, conf=args.conf, iou=0.5, verbose=False)
            for r in results:
                for box in r.boxes:
                    time_s = frame_to_time(frame_idx, fps)
                    detections.append((time_s, float(box.conf)))
        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    # Temporal NMS
    clean_ts = temporal_nms(detections, window=args.merge_window)
    with open(args.out, "w") as f:
        json.dump(clean_ts, f, indent=2)
    print(f"Wrote {len(clean_ts)} timestamps to {args.out}")


def extract_clips(args):
    with open(args.detections) as f:
        timestamps = json.load(f)

    in_path = Path(args.video)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, t in enumerate(timestamps):
        start = max(0.0, t - args.pre)
        duration = args.pre + args.post
        out_file = out_dir / f"clip_{idx:03d}_{t:.2f}.mp4"
        ffmpeg_cut(in_path, start, duration, out_file)
    print(f"Extracted {len(timestamps)} clips ➜ {out_dir}")


# Baseline template matching compared to YOLOv8
def baseline_template(args):
    template = cv2.imread(str(args.template), cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rate_stride = max(1, int(round(fps / args.rate)))

    detections = []
    frame_idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Baseline")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % rate_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val >= args.thresh:
                detections.append(frame_to_time(frame_idx, fps))
        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    clean_ts = temporal_nms(detections, window=args.merge_window)
    with open(args.out, "w") as f:
        json.dump(clean_ts, f, indent=2)
    print(f"Baseline detected {len(clean_ts)} events ➜ {args.out}")


def evaluate_predictions(args):
    with open(args.gt) as f:
        gt = sorted(json.load(f))
    with open(args.pred) as f:
        pred = sorted(json.load(f))

    # Match predictions to ground truth within tolerance
    tol = args.tolerance
    matched_pred = [False] * len(pred)
    tp = 0
    abs_errors: List[float] = []
    for g in gt:
        best_err = tol + 1
        best_idx = -1
        for i, p in enumerate(pred):
            err = abs(p - g)
            if err <= tol and err < best_err and not matched_pred[i]:
                best_err, best_idx = err, i
        if best_idx >= 0:
            tp += 1
            matched_pred[best_idx] = True
            abs_errors.append(best_err)

    fp = matched_pred.count(False)
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else None

    metrics = {
        "ground_truth": len(gt),
        "predicted": len(pred),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "mae_seconds": mae,
    }
    print(json.dumps(metrics, indent=2))

# CLI running everything

def main():
    parser = argparse.ArgumentParser(description="Kill‑banner detection pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p_train = sub.add_parser("train", help="Train YOLOv8 model")
    p_train.add_argument("--data", required=True, help="annotations YAML path")
    p_train.add_argument("--weights", default="yolov8n.pt", help="initial weights")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.set_defaults(func=train_yolov8)

    # Inference
    p_infer = sub.add_parser("infer", help="Run model inference on video")
    p_infer.add_argument("--video", required=True, type=Path)
    p_infer.add_argument("--weights", required=True)
    p_infer.add_argument("--out", default="detections.json")
    p_infer.add_argument("--rate", type=float, default=10.0, help="FPS to sample")
    p_infer.add_argument("--conf", type=float, default=0.2)
    p_infer.add_argument("--merge_window", type=float, default=5.0)
    p_infer.set_defaults(func=infer_yolov8)

    # Extract clips (NEEDS WORK)
    p_clip = sub.add_parser("extract-clips", help="Extract highlight clips around detections")
    p_clip.add_argument("--video", required=True, type=Path)
    p_clip.add_argument("--detections", required=True, type=Path)
    p_clip.add_argument("--output_dir", default="clips/")
    p_clip.add_argument("--pre", type=float, default=2.0, help="seconds before event")
    p_clip.add_argument("--post", type=float, default=3.0, help="seconds after event")
    p_clip.set_defaults(func=extract_clips)

    # Baseline template
    p_base = sub.add_parser("baseline", help="Simple template‑matching baseline")
    p_base.add_argument("--video", required=True, type=Path)
    p_base.add_argument("--template", required=True, type=Path)
    p_base.add_argument("--out", default="detections_baseline.json")
    p_base.add_argument("--rate", type=float, default=10.0)
    p_base.add_argument("--thresh", type=float, default=0.8, help="match threshold")
    p_base.add_argument("--merge_window", type=float, default=3.0)
    p_base.set_defaults(func=baseline_template)

    # Evaluate
    p_eval = sub.add_parser("evaluate", help="Compare prediction JSON to ground truth")
    p_eval.add_argument("--gt", required=True, type=Path)
    p_eval.add_argument("--pred", required=True, type=Path)
    p_eval.add_argument("--tolerance", type=float, default=1.0, help="seconds")
    p_eval.set_defaults(func=evaluate_predictions)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
