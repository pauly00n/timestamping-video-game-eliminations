# Kill Banner Detection Pipeline

Automated elimination timestamping for Brawl Stars gameplay footage using YOLOv8-nano. Detects kill-banner HUD events in video, outputs timestamps, and extracts highlight clips. Final Project for Stanford CS 231N.

## Overview

The pipeline samples video frames, runs a fine-tuned YOLOv8-nano detector (or a template matching baseline), applies temporal NMS to merge same-event detections, and outputs a JSON list of elimination timestamps. Those timestamps can then be used to cut highlight clips with FFmpeg.

## Setup

**Requirements:** Python 3.9+, FFmpeg (system binary)

```bash
python -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python numpy tqdm
```

## Usage

### Train
Fine-tune YOLOv8-nano on your annotated dataset:
```bash
python pipeline.py train \
  --data data/annotations.yaml \
  --weights yolov8n.pt \
  --epochs 50
```

The dataset must follow [Ultralytics YOLO format](https://docs.ultralytics.com/datasets/):
```yaml
# annotations.yaml
path: data
train: images/train
val: images/val
nc: 1
names: [kill_banner]
```

### Infer
Run inference on a video and output a JSON array of timestamps (seconds):
```bash
python pipeline.py infer \
  --video match.mp4 \
  --weights best.pt \
  --out detections.json
```

Key options:
- `--conf 0.2` — confidence threshold (keep low; model trained on limited data)
- `--merge_window 5.0` — NMS window in seconds; merges same-event detections while preserving close consecutive kills
- `--rate 10.0` — frames per second to sample

### Extract Clips
Cut highlight clips around each detected timestamp:
```bash
python pipeline.py extract-clips \
  --video match.mp4 \
  --detections detections.json \
  --output_dir clips/
```

Options: `--pre 2.0` (seconds before event), `--post 3.0` (seconds after event). Uses `ffmpeg -c copy` (no re-encoding).

### Baseline (Template Matching)
Run the grayscale cross-correlation baseline for comparison:
```bash
python baseline_pipeline.py \
  --video match.mp4 \
  --template banner.png \
  --out_dir clips/ \
  --gt_csv gt_sans.csv
```

### Evaluate
Compare predicted timestamps to ground truth:
```bash
python pipeline.py evaluate \
  --gt gt_sans.json \
  --pred detections.json \
  --tolerance 1.0
```

Outputs precision, recall, TP/FP/FN, and MAE for matched true positives.

## How It Works

**YOLOv8 pipeline (`pipeline.py`):**
1. Sample video at `--rate` FPS by striding through frames with OpenCV
2. Run YOLOv8-nano detector on each sampled frame
3. Apply confidence-aware temporal NMS: cluster detections within `--merge_window` seconds, keep the highest-confidence timestamp per cluster
4. Write timestamps to JSON; optionally cut clips with FFmpeg

**Template matching baseline (`baseline_pipeline.py`):**
Same flow but uses `cv2.matchTemplate` with `TM_CCOEFF_NORMED` against `banner.png` instead of a neural detector.

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Main CLI — train, infer, extract-clips, evaluate |
| `baseline_pipeline.py` | Template matching baseline |
| `best.pt` | Trained YOLOv8-nano weights |
| `banner.png` | Kill banner template for baseline |
| `gt_sans.json` / `gt_sans.csv` | Ground truth elimination timestamps (seconds) |
| `det.json` | Latest inference output |
| `metrics.json` | Latest evaluation results |
| `frames/` | Sample extracted frames |
| `paper/` | CS 231N final paper (LaTeX + PDF) |
