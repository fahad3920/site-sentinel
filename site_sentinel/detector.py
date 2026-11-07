"""PPE + person detection with tiling, class-specific thresholds and face blurring."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import config


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: List[float]  # x1, y1, x2, y2 in full-image pixel coords

    @property
    def center_bottom(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    machinery: List[Detection] = field(default_factory=list)
    image_shape: tuple = (0, 0)
    model: str = ""
    tiles_used: int = 1
    inference_ms: float = 0.0


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: List[Detection] = []
    for d in ordered:
        if all(_iou(d.xyxy, k.xyxy) < iou_thresh or d.cls != k.cls for k in kept):
            kept.append(d)
    return kept


class SafetyDetector:
    """Wraps the Apache-2.0 YOLO11 safety model (+ optional COCO machinery proxy)."""

    def __init__(self, model_path: Optional[Path] = None, machinery: bool = True):
        from ultralytics import YOLO

        path = Path(model_path) if model_path else config.PPE_MODEL_ACCURATE
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path}. Run scripts/download_model.py first."
            )
        self.model = YOLO(str(path))
        self.model_path = path
        self.names = {int(k): v for k, v in self.model.names.items()}

        self.machinery_model = None
        if machinery and config.MACHINERY_PROXY_MODEL and config.MACHINERY_PROXY_MODEL.exists():
            self.machinery_model = YOLO(str(config.MACHINERY_PROXY_MODEL))
            self.machinery_names = {int(k): v for k, v in self.machinery_model.names.items()}

        # Face blur needs OpenCV's Haar cascades (OpenCV 4.x). Degrade gracefully if missing.
        try:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except AttributeError:
            self._face_cascade = None

    def _predict_tiles(self, image: np.ndarray):
        h, w = image.shape[:2]
        raw: List[Detection] = []
        tiles = 0
        if max(h, w) <= config.TILE_MAX_SIDE:
            tiles = 1
            raw.extend(self._predict_region(image, 0, 0))
        else:
            step = int(config.TILE_MAX_SIDE * (1 - config.TILE_OVERLAP))
            y = 0
            while y < h:
                x = 0
                y2 = min(h, y + config.TILE_MAX_SIDE)
                while x < w:
                    x2 = min(w, x + config.TILE_MAX_SIDE)
                    tile = image[y:y2, x:x2]
                    if tile.size:
                        tiles += 1
                        raw.extend(self._predict_region(tile, x, y))
                    if x2 == w:
                        break
                    x = x2
                if y2 == h:
                    break
                y = y2
        return _nms(raw, config.MERGE_IOU), tiles

    def _predict_region(self, region: np.ndarray, off_x: int, off_y: int) -> List[Detection]:
        out: List[Detection] = []
        results = self.model.predict(region, verbose=False, device="cpu")
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                name = self.names.get(cls_id, str(cls_id))
                conf = float(boxes.conf[i])
                thresh = config.CONF_THRESHOLDS.get(name, config.DEFAULT_CONF)
                if conf < thresh:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                out.append(
                    Detection(
                        cls=name,
                        conf=conf,
                        xyxy=[x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y],
                    )
                )
        return out

    def _predict_machinery(self, image: np.ndarray) -> List[Detection]:
        if self.machinery_model is None:
            return []
        out: List[Detection] = []
        results = self.machinery_model.predict(image, verbose=False, device="cpu")
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                name = self.machinery_names.get(int(boxes.cls[i]), "")
                conf = float(boxes.conf[i])
                if name not in config.MACHINERY_CLASSES or conf < config.MACHINERY_CONF:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                out.append(Detection(cls=name, conf=conf, xyxy=[x1, y1, x2, y2]))
        return out

    def detect(self, image: np.ndarray) -> DetectionResult:
        import time

        t0 = time.perf_counter()
        dets, tiles = self._predict_tiles(image)
        machinery = self._predict_machinery(image)
        ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(
            detections=dets,
            machinery=machinery,
            image_shape=image.shape[:2],
            model=self.model_path.name,
            tiles_used=tiles,
            inference_ms=round(ms, 1),
        )

    def blur_faces(self, image: np.ndarray) -> np.ndarray:
        """Privacy default: blur detected faces so reports show compliance, not people."""
        if self._face_cascade is None:
            return image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        out = image.copy()
        for (x, y, w, h) in faces:
            pad = int(0.25 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(out.shape[1], x + w + pad), min(out.shape[0], y + h + pad)
            roi = out[y1:y2, x1:x2]
            if roi.size:
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
        return out
