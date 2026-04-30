# Smart Crowd Density Monitoring System

A complete, modular Python + OpenCV + Ultralytics YOLOv8 project to:

- Detect **ONLY people** in a video stream
- Build a **grid-based crowd density map** (configurable, e.g. 20x20)
- Map each detected person to a grid cell (bbox center → cell)
- Render a **realistic, smoothed heatmap overlay** (Blue → Green → Yellow → Red)
- Show bounding boxes, total people count, max cell density, FPS, and alerts
- Write logs to CSV: `timestamp,total_count,max_cell_count,density,fps,alert`
- (Bonus) Optional **Flask dashboard** for live metrics

## Setup (Windows)

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Download YOLO weights (first run)

On the first run, Ultralytics will automatically download `yolov8n.pt` if it is not present in the folder.

## How to run

### Webcam (default)

```bash
python main.py
```

### Video file

```bash
python main.py --source "path/video.mp4"
```

### Grid size + heatmap smoothing

```bash
python main.py --grid 20x20 --accum 0.65 --alpha 0.55 --blur 0
```

Notes:
- `--grid RxC`: grid rows x columns
- `--accum`: temporal smoothing (higher = smoother heatmap)
- `--alpha`: heatmap overlay intensity
- `--blur`: Gaussian blur kernel size (odd). `0` picks an automatic value.

### Optional Flask dashboard

```bash
python main.py --dashboard
```

Then open `http://127.0.0.1:5000`.

### Headless mode (no OpenCV window)

```bash
python main.py --no-display
```

## Controls

- Press **q** to quit the OpenCV window.

## Logs

- CSV logs are written to `logs/crowd_log.csv` by default (configurable with `--csv`).

## Expected output (what you’ll see)

- A **semi-transparent heatmap overlay** indicating crowd density:
  - Blue/Green = low density
  - Yellow = medium density
  - Red = high density
- Bounding boxes around people
- On-screen text:
  - Total people count
  - Grid size and max cell count + density label (**SAFE / MODERATE / CROWDED**)
  - FPS
  - Alert banner if any grid cell becomes high density

