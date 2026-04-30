from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float


class PersonDetector:
    """
    YOLOv8 person-only detector.

    Notes:
    - YOLO class id for "person" in COCO is 0.
    - We filter to person class at inference time for speed.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        *,
        conf: float = 0.35,
        iou: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR frame.
        Returns list of bounding boxes and confidences.
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        xyxy = r0.boxes.xyxy
        confs = r0.boxes.conf
        if xyxy is None or confs is None:
            return []

        xyxy_np = xyxy.detach().cpu().numpy().astype(int)
        confs_np = confs.detach().cpu().numpy().astype(float)

        dets: List[Detection] = []
        for (x1, y1, x2, y2), c in zip(xyxy_np, confs_np):
            dets.append(Detection(bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)), confidence=float(c)))
        return dets

