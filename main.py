from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import cv2

from detector import PersonDetector
from utils import CSVLogger, FPSTracker, SharedState, draw_text_with_bg, now_iso
from heatmap import HeatmapConfig, HeatmapEngine, draw_heatmap_legend, frame_density_label, people_to_grid_counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart Crowd Density Monitoring System (YOLOv8 + OpenCV)")
    p.add_argument(
        "--source",
        default="video.mp4",
        help="Video source: webcam index (default 0) or path to a video file.",
    )
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLOv8 model path (e.g., yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--device", default=None, help="Device for inference: 'cpu', '0' (GPU), etc. (optional)")
    p.add_argument("--csv", default="logs/crowd_log.csv", help="CSV path for logs")
    p.add_argument("--no-display", action="store_true", help="Run without showing the OpenCV window")
    p.add_argument("--dashboard", action="store_true", help="Start optional Flask dashboard on http://127.0.0.1:5000")
    p.add_argument("--grid", default="20x20", help="Grid size as RxC, e.g. 20x20, 30x40")
    p.add_argument("--accum", type=float, default=0.65, help="Heatmap EMA accumulation factor (0..1). Higher = smoother.")
    p.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay alpha (0..1)")
    p.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size (odd). 0 = auto")
    return p.parse_args()


def _source_to_capture(source: str) -> cv2.VideoCapture:
    # If it’s a digit, treat as webcam index; otherwise treat as file path/URL.
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def zone_color(density: str) -> Tuple[int, int, int]:
    # BGR colors for OpenCV
    if density == "SAFE":
        return (0, 180, 0)
    if density == "MODERATE":
        return (0, 215, 255)  # yellow-ish
    return (0, 0, 220)

def _parse_grid(s: str) -> Tuple[int, int]:
    try:
        s = s.lower().replace(" ", "")
        if "x" in s:
            a, b = s.split("x", 1)
            r = int(a)
            c = int(b)
            return max(1, r), max(1, c)
    except Exception:
        pass
    return 20, 20


def main() -> int:
    args = parse_args()

    cap = _source_to_capture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {args.source}", file=sys.stderr)
        return 2

    detector = PersonDetector(model_path=args.model, conf=args.conf, iou=args.iou, device=args.device)
    fps_tracker = FPSTracker()
    csv_logger = CSVLogger(args.csv)
    state = SharedState()

    grid_rows, grid_cols = _parse_grid(args.grid)
    heat = HeatmapEngine(
        HeatmapConfig(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            accumulation=float(args.accum),
            overlay_alpha=float(args.alpha),
            blur_ksize=int(args.blur),
        )
    )

    if args.dashboard:
        try:
            from dashboard import run_dashboard_in_thread

            run_dashboard_in_thread(state)
            print("[INFO] Dashboard running at http://127.0.0.1:5000")
        except Exception as e:
            print(f"[WARN] Dashboard failed to start: {e}", file=sys.stderr)

    win_name = "Smart Crowd Density Monitoring System"
    last_frame_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[INFO] End of stream or camera read failure.")
                break

            h, w = frame.shape[:2]
            dets = detector.detect(frame)
            bboxes = [d.bbox_xyxy for d in dets]

            grid_counts = people_to_grid_counts(
                bboxes,
                (h, w),
                grid_rows=heat.config.grid_rows,
                grid_cols=heat.config.grid_cols,
            )
            grid_acc = heat.update_counts(grid_counts)
            density = frame_density_label(grid_acc)
            max_cell = float(grid_acc.max()) if grid_acc.size else 0.0

            alert = ""
            if density == "CROWDED":
                alert = f"ALERT: High density detected (max cell count: {max_cell:.1f})"

            # Render realistic heatmap overlay (transparent where zero)
            frame, _heatmap_bgr = heat.render_overlay(frame, grid_acc)

            # Draw bounding boxes
            for d in dets:
                x1, y1, x2, y2 = d.bbox_xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Text overlays
            total_n = int(len(dets))
            draw_text_with_bg(frame, f"Total People: {total_n}", (10, 30), bg_color=(20, 20, 20))
            draw_text_with_bg(frame, f"Grid: {heat.config.grid_rows}x{heat.config.grid_cols}", (10, 65), bg_color=(20, 20, 20))
            draw_text_with_bg(frame, f"Max Cell: {max_cell:.1f}  ({density})", (10, 100), bg_color=zone_color(density))

            fps = fps_tracker.update()
            draw_text_with_bg(frame, f"FPS: {fps:.1f}", (10, 135), bg_color=(20, 20, 20))

            if alert:
                draw_text_with_bg(frame, alert, (10, 175), bg_color=(0, 0, 220))

            draw_heatmap_legend(frame, origin_xy=(10, 210), size_wh=(240, 16), title="Crowd density (per cell)")

            ts = now_iso()
            csv_logger.log(ts, total_n, max_cell, density, float(fps), alert)

            # Update dashboard state (if running)
            state.update(
                timestamp=ts,
                total_count=total_n,
                max_cell_count=max_cell,
                density=density,
                fps=float(fps),
                alert=alert,
            )

            if not args.no_display:
                cv2.imshow(win_name, frame)
                # Press 'q' to quit.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # If running headless, prevent spinning too fast on some sources
            if args.no_display:
                now = time.time()
                if now - last_frame_t < 0.001:
                    time.sleep(0.001)
                last_frame_t = now

    finally:
        cap.release()
        csv_logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

