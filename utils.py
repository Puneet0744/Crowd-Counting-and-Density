from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import cv2


def now_iso() -> str:
    """Return current local time as ISO string (seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


def ensure_parent_dir(file_path: str) -> None:
    """Create parent directory for a file path if missing."""
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass
class FPSTracker:
    """Lightweight FPS tracker using exponential moving average."""

    smoothing: float = 0.9
    _last_t: float = time.time()
    _fps: float = 0.0

    def update(self) -> float:
        t = time.time()
        dt = max(1e-9, t - self._last_t)
        inst = 1.0 / dt
        self._fps = inst if self._fps == 0.0 else (self.smoothing * self._fps + (1 - self.smoothing) * inst)
        self._last_t = t
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class CSVLogger:
    """Append-only CSV logger for per-frame or periodic metrics."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        ensure_parent_dir(csv_path)
        self._file = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._wrote_header = os.path.getsize(csv_path) > 0
        if not self._wrote_header:
            self._writer.writerow(["timestamp", "total_count", "max_cell_count", "density", "fps", "alert"])
            self._file.flush()
            self._wrote_header = True

    def log(self, timestamp: str, total_count: int, max_cell_count: float, density: str, fps: float, alert: str) -> None:
        self._writer.writerow([timestamp, int(total_count), float(max_cell_count), str(density), float(fps), str(alert)])
        # Keep I/O small but safe for power loss: flush periodically outside if needed.
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def draw_text_with_bg(
    frame,
    text: str,
    org: Tuple[int, int],
    *,
    font_scale: float = 0.7,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 6,
) -> None:
    """Draw readable text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x1, y1 = x - padding, y - th - padding
    x2, y2 = x + tw + padding, y + baseline + padding
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def overlay_filled_rect(frame, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], alpha: float) -> None:
    """Draw a semi-transparent filled rectangle overlay."""
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class SharedState:
    """Thread-safe state for optional dashboard."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: Dict[str, Any] = {
            "timestamp": now_iso(),
            "total_count": 0,
            "max_cell_count": 0.0,
            "density": "SAFE",
            "fps": 0.0,
            "alert": "",
        }

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._data.update(kwargs)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

