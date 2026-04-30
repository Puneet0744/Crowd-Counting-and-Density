from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class HeatmapConfig:
    grid_rows: int = 20
    grid_cols: int = 20
    # Exponential moving average of per-cell counts (history smoothing)
    accumulation: float = 0.65
    # Blur parameters for a more realistic surveillance-style heatmap
    blur_ksize: int = 0  # 0 => auto from frame size
    blur_sigma: float = 0  # 0 => OpenCV chooses based on ksize
    # Visualization
    overlay_alpha: float = 0.55
    colormap: int = cv2.COLORMAP_TURBO  # blue->green->yellow->red


def bbox_center(bbox_xyxy: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) // 2, (y1 + y2) // 2


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def people_to_grid_counts(
    bboxes_xyxy: Iterable[Tuple[int, int, int, int]],
    frame_shape_hw: Tuple[int, int],
    *,
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    """
    Return a (grid_rows x grid_cols) int array with people counts per cell.
    Mapping uses bbox center -> cell index.
    """
    h, w = frame_shape_hw
    grid_rows = max(1, int(grid_rows))
    grid_cols = max(1, int(grid_cols))

    counts = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cell_w = w / float(grid_cols)
    cell_h = h / float(grid_rows)

    for bbox in bboxes_xyxy:
        cx, cy = bbox_center(bbox)
        c = int(cx / cell_w) if cell_w > 0 else 0
        r = int(cy / cell_h) if cell_h > 0 else 0
        r = _clamp_int(r, 0, grid_rows - 1)
        c = _clamp_int(c, 0, grid_cols - 1)
        counts[r, c] += 1.0

    return counts


def density_bucket(cell_count: float) -> str:
    """
    Cell density threshold per spec:
    0 -> NONE
    1-3 -> LOW
    4-7 -> MED
    8+ -> HIGH
    """
    n = float(cell_count)
    if n <= 0:
        return "NONE"
    if n <= 3:
        return "LOW"
    if n <= 7:
        return "MED"
    return "HIGH"


def frame_density_label(grid_counts: np.ndarray) -> str:
    """
    Overall density label derived from max cell count.
    SAFE: no cells >= 4
    MODERATE: any cell in [4,7]
    CROWDED: any cell >= 8
    """
    if grid_counts.size == 0:
        return "SAFE"
    mx = float(np.max(grid_counts))
    if mx >= 8:
        return "CROWDED"
    if mx >= 4:
        return "MODERATE"
    return "SAFE"


class HeatmapEngine:
    """
    Grid-based crowd density heatmap with optional temporal accumulation.

    Pipeline:
    - bboxes -> per-cell counts
    - exponential accumulation for stability
    - normalize -> upsample -> Gaussian blur -> apply colormap
    - blend heatmap onto frame, transparent where intensity=0
    """

    def __init__(self, config: HeatmapConfig | None = None) -> None:
        self.config = config or HeatmapConfig()
        self._accum: np.ndarray | None = None

    def reset(self) -> None:
        self._accum = None

    def update_counts(self, grid_counts: np.ndarray) -> np.ndarray:
        grid_counts = grid_counts.astype(np.float32, copy=False)
        if self._accum is None or self._accum.shape != grid_counts.shape:
            self._accum = grid_counts.copy()
            return self._accum

        a = float(self.config.accumulation)
        a = 0.0 if a < 0.0 else 0.999 if a >= 1.0 else a
        self._accum = a * self._accum + (1.0 - a) * grid_counts
        return self._accum

    def render_overlay(
        self,
        frame_bgr: np.ndarray,
        grid_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (blended_frame_bgr, heatmap_bgr).

        Heatmap is transparent in regions with zero intensity.
        """
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return frame_bgr, np.zeros_like(frame_bgr)

        # Normalize to 0..255 (visual only). Use max of current grid for adaptiveness.
        mx = float(np.max(grid_values)) if grid_values.size else 0.0
        if mx <= 1e-6:
            heatmap = np.zeros((h, w, 3), dtype=np.uint8)
            return frame_bgr, heatmap

        g = (grid_values / mx) * 255.0
        g8 = np.clip(g, 0, 255).astype(np.uint8)

        # Upsample grid to full frame size for smooth interpolation
        up = cv2.resize(g8, (w, h), interpolation=cv2.INTER_LINEAR)

        # Blur to make hotspots spread naturally
        ksize = int(self.config.blur_ksize)
        if ksize <= 0:
            # Choose an odd kernel size proportional to frame size and grid density
            base = max(9, (min(h, w) // 50) | 1)  # ensure odd
            ksize = base
        if ksize % 2 == 0:
            ksize += 1

        blur = cv2.GaussianBlur(up, (ksize, ksize), float(self.config.blur_sigma))
        heatmap = cv2.applyColorMap(blur, int(self.config.colormap))

        # Blend with original; keep true transparency where intensity is zero.
        alpha = float(self.config.overlay_alpha)
        alpha = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
        blended = cv2.addWeighted(frame_bgr, 1.0, heatmap, alpha, 0.0)

        mask = blur > 0
        out = frame_bgr.copy()
        out[mask] = blended[mask]
        return out, heatmap


def draw_heatmap_legend(
    frame_bgr: np.ndarray,
    *,
    origin_xy: Tuple[int, int] = (10, 10),
    size_wh: Tuple[int, int] = (220, 16),
    title: str = "Density",
) -> None:
    """
    Small legend matching the colormap direction (low->high).
    Uses a generic gradient strip and the specified thresholds.
    """
    x, y = origin_xy
    w, h = size_wh
    x2, y2 = x + w, y + h
    x = max(0, x)
    y = max(0, y)
    x2 = min(frame_bgr.shape[1] - 1, x2)
    y2 = min(frame_bgr.shape[0] - 1, y2)
    if x2 <= x or y2 <= y:
        return

    strip_w = x2 - x
    strip_h = y2 - y
    grad = np.tile(np.linspace(0, 255, strip_w, dtype=np.uint8), (strip_h, 1))
    strip = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)

    # Background container
    pad = 6
    box_x1 = x - pad
    box_y1 = y - 26
    box_x2 = x2 + pad
    box_y2 = y2 + 26
    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(frame_bgr.shape[1] - 1, box_x2)
    box_y2 = min(frame_bgr.shape[0] - 1, box_y2)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0, frame_bgr)

    # Paste gradient strip
    frame_bgr[y:y2, x:x2] = strip[:, : (x2 - x)]
    cv2.rectangle(frame_bgr, (x, y), (x2, y2), (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_bgr, title, (x, y - 8), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, "0", (x, y2 + 18), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "1-3", (x + int(strip_w * 0.22), y2 + 18), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "4-7", (x + int(strip_w * 0.52), y2 + 18), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "8+", (x + int(strip_w * 0.82), y2 + 18), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

