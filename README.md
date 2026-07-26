# Orthodontic Bracket Detection (YOLO)

## 1. Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended for training (CPU works but is slow). Ultralytics auto-detects CUDA if available.

## 2. Prepare your dataset

Arrange your images/labels like this:

```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each image needs a matching `.txt` label file (same filename, `.txt` extension) with one line per bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates normalized 0–1 relative to image width/height.

**If your current labels are in a different format** (COCO JSON, Pascal VOC XML, CSV, LabelImg, CVAT export, etc.), tell me the format and I'll write a converter — don't hand-convert these, it's easy to get the normalization or (x,y) reference point wrong.

Edit `data.yaml` to point at your `dataset/` folder and list your class names.

### Two ways to encode "correct vs. misplaced"

- **Option A — separate classes** (recommended if your labelers can judge placement while annotating): label boxes directly as `bracket_correct` / `bracket_misplaced`. The model learns correctness end-to-end. This is usually more accurate than post-hoc geometry rules, since bracket placement correctness in orthodontics depends on things like position relative to the tooth's clinical crown height — which a CNN can learn from visual context, but a simple line-fit can't fully capture.
- **Option B — single class + geometric heuristic**: label everything as `bracket`, and use `detect_and_check.py`'s built-in heuristic (fits a curve through detected bracket centers and flags outliers). This is a rough approximation and mainly useful when you don't have placement-quality ground truth labels yet.

You can switch between these by changing `nc`/`names` in `data.yaml`.

## 3. Train

```bash
python train.py --data data.yaml --model yolo11s.pt --epochs 150 --imgsz 960
```

- `yolo11n.pt` (nano) → fastest, least accurate, good for quick iteration.
- `yolo11s.pt` / `yolo11m.pt` → good accuracy/speed balance, recommended starting point.
- `--imgsz 960` (or higher, e.g. 1280) matters here — brackets are small relative to a full intraoral photo, and YOLO struggles with tiny objects at the default 640px.

Training outputs (weights, curves, confusion matrix, sample predictions) land in `runs/bracket_detect/exp1/`. Best weights: `runs/bracket_detect/exp1/weights/best.pt`.

## 4. Evaluate / run inference

```bash
python detect_and_check.py --weights runs/bracket_detect/exp1/weights/best.pt --source path/to/test_images --conf 0.35
```

This draws boxes on each image (green = correct, red = misplaced, orange = unverified) and prints a per-image summary, saved to `runs/bracket_check/`.

## Tips specific to this task

- **Small object detection**: brackets are tiny relative to the image. Besides raising `imgsz`, consider tiling/cropping each quadrant of the mouth as a separate training image if recall is poor.
- **Class imbalance**: misplaced brackets are usually much rarer than correct ones in real data. Watch the per-class recall in the training output, not just overall mAP — a model can get high mAP while still missing most of the rare "misplaced" class. If needed, oversample misplaced examples or use `--batch` with `cls` loss weighting (`model.train(..., cls=1.0)` adjustable in Ultralytics).
- **Labeling consistency**: if multiple people labeled correctness, get an inter-rater agreement check before trusting the "misplaced" labels — noisy labels here will directly cap model accuracy.
- **Confidence threshold**: tune `--conf` on your validation set (via `detect_and_check.py`) rather than trusting the default — for a clinical-adjacent task you likely want higher recall (lower threshold) and to flag borderline cases for human review rather than silently dropping them.
