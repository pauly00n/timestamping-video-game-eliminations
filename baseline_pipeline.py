
#!/usr/bin/env python3
"""
Baseline pipeline: raw grayscale template matcher for kill-banner detection.
Author: Paul Yoon, 2025-06-04
"""

from __future__ import annotations
import argparse, csv, os, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np


# ───────────────────────────── utilities ───────────────────────────── #

def read_ground_truth(csv_path: str | Path) -> list[float]:
    """Read a CSV with one timestamp (seconds) per line."""
    if csv_path is None:
        return []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        gt = [float(row[0]) for row in reader if row]
    return sorted(gt)


def temporal_nms(times: list[float], gate: float = 3.0) -> list[float]:
    """Merge detections closer than `gate` seconds (first one wins)."""
    times.sort()
    merged = []
    for t in times:
        if not merged or t - merged[-1] > gate:
            merged.append(t)
    return merged


def precision_recall(pred: list[float], gt: list[float], tol: float = 1.5):
    """
    Match each prediction to the nearest GT within ±tol seconds (greedy).
    Returns precision, recall, F1.
    """
    if not gt:
        return 0, 0, 0
    pred = sorted(pred)
    gt   = sorted(gt)
    matched = set()
    tp = 0
    for p in pred:
        # nearest gt index (binary search would be nicer but n is tiny)
        idx  = min(range(len(gt)), key=lambda i: abs(gt[i] - p))
        if abs(gt[idx] - p) <= tol and idx not in matched:
            matched.add(idx)
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1

def mean_abs_error(pred: list[float], gt: list[float], tol: float = 1.5) -> float:
    """
    Mean absolute error (seconds) for the *matched* true positives.
    Same greedy matching logic as precision_recall.
    """
    if not gt:
        return 0.0
    pred = sorted(pred)
    gt   = sorted(gt)
    matched = set()
    errors  = []
    for p in pred:
        idx = min(range(len(gt)), key=lambda i: abs(gt[i] - p))
        err = abs(gt[idx] - p)
        if err <= tol and idx not in matched:
            matched.add(idx)
            errors.append(err)
    return float(np.mean(errors)) if errors else 0.0


def ffmpeg_extract_clip(src: Path, centre: float, out_path: Path,
                        clip_len: float = 5.0, pre_sec: float = 2.0):
    """
    Call FFmpeg to copy-stream‐extract a fixed-length clip.
    """
    start = max(centre - pre_sec, 0.0)
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{clip_len:.3f}",
        "-c", "copy",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)


# ───────────────────────────── main pipeline ───────────────────────────── #

def run_matching(video_path: Path,
                 template_path: Path,
                 fps_sample: float = 6.0,
                 thr: float = 0.80,
                 nms_gate: float = 3.0):
    """
    Return list of detection times (seconds) in the video.
    """
    tmpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        sys.exit(f"ERROR: could not read template {template_path}")
    th, tw = tmpl.shape[:2]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video {video_path}")

    vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step      = int(round(vid_fps / fps_sample))
    if step < 1: step = 1

    detections = []
    t0 = time.perf_counter()

    for fi in range(frame_cnt):
        ret, frame = cap.read()
        if not ret:
            break
        if fi % step:
            continue

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resp = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)
        max_val = resp.max()
        if max_val >= thr:
            detections.append(fi / vid_fps)

        max_val = resp.max()
        if fi < 300:          # print the first 300 sampled frames
            print(f"{fi:06d}  max_corr={max_val:.3f}")

    cap.release()
    runtime = time.perf_counter() - t0
    vid_len = frame_cnt / vid_fps
    throughput = vid_len / runtime if runtime else 0
    return detections, throughput


def main():
    p = argparse.ArgumentParser(
        description="Raw template-matching baseline for kill-banner detection"
    )
    p.add_argument("--video",       required=True, type=Path)
    p.add_argument("--template",    required=True, type=Path)
    p.add_argument("--out_dir",     required=True, type=Path)
    p.add_argument("--gt_csv",      type=Path, help="optional ground-truth CSV")
    p.add_argument("--fps",         default=6.0,  type=float)
    p.add_argument("--thr",         default=0.80, type=float)
    p.add_argument("--gate",        default=3.0,  type=float, help="NMS window (s)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1 run matcher
    det_raw, throughput = run_matching(args.video,
                                       args.template,
                                       args.fps,
                                       args.thr,
                                       args.gate)

    # 2 temporal NMS
    det = temporal_nms(det_raw, gate=args.gate)

    # 3 clip extraction
    print(f"▶ Extracting {len(det)} clips to {args.out_dir}")
    for idx, centre in enumerate(det, 1):
        out_file = args.out_dir / f"clip_{idx:03d}.mp4"
        ffmpeg_extract_clip(args.video, centre, out_file)

    # 4 evaluation (optional)
    if args.gt_csv is not None:
        gt = read_ground_truth(args.gt_csv)
        tol = args.gate / 2          # same window you used for P/R
        prec, rec, f1 = precision_recall(det, gt, tol=tol)
        mae = mean_abs_error(det, gt, tol=tol)
        print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
        print(f"MAE (matched TPs): {mae:.3f} s")

    # 5 speed report
    print(f"Throughput: {throughput:.1f}× real-time "
          f"({throughput*30:.0f} fps equiv on 30 fps source)")

if __name__ == "__main__":
    main()
