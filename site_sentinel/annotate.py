"""Draw compliance boxes, machinery boxes and restricted-zone polygons."""
from typing import List, Sequence

import cv2
import numpy as np

from . import config
from .detector import Detection, DetectionResult
from .zones import Violation, Zone

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(det: Detection) -> str:
    pretty = {
        "hat": "Helmet OK",
        "vest": "Vest OK",
        "nohat": "NO HELMET",
        "novest": "NO VEST",
        "person": "Worker",
    }.get(det.cls, det.cls)
    return f"{pretty} {det.conf:.2f}"


def _color_for(cls: str):
    if cls in config.VIOLATION_CLASSES:
        return config.COLORS["violation"]
    if cls in config.COMPLIANT_CLASSES:
        return config.COLORS["compliant"]
    if cls == "person":
        return config.COLORS["person"]
    return config.COLORS["machinery"]


def _box(img, xyxy: Sequence[float], color, label: str, thickness: int = 2):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.5, 1)
    ty = max(y1 - 6, th + 6)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty + baseline), color, -1)
    cv2.putText(img, label, (x1 + 3, ty - 4), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_zones(img: np.ndarray, zones: List[Zone]):
    h, w = img.shape[:2]
    for zone in zones:
        pts = np.array([[int(x * w), int(y * h)] for (x, y) in zone.polygon], dtype=np.int32)
        color = config.COLORS["zone"]
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        if len(pts):
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.putText(img, zone.name, (cx - 20, cy), _FONT, 0.55, color, 2, cv2.LINE_AA)


def annotate(result: DetectionResult, zones: List[Zone] = None,
             violations: List[Violation] = None) -> np.ndarray:
    """Return a BGR canvas with all detections and violations drawn."""
    h, w = result.image_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 30)

    if zones:
        draw_zones(canvas, zones)

    for d in result.machinery:
        _box(canvas, d.xyxy, config.COLORS["machinery"], f"Machinery {d.conf:.2f}")
    for d in result.detections:
        _box(canvas, d.xyxy, _color_for(d.cls), _label(d))

    # Highlight violating people with a thicker red box.
    if violations:
        for v in violations:
            if v.detection is not None:
                _box(canvas, v.detection.xyxy, config.COLORS["violation"],
                     v.type.replace("_", " ").upper(), thickness=3)
    return canvas


def annotate_photo(image: np.ndarray, result: DetectionResult,
                   zones: List[Zone] = None, violations: List[Violation] = None) -> np.ndarray:
    """Same as annotate() but drawn over the original photo."""
    canvas = image.copy()
    if zones:
        draw_zones(canvas, zones)
    for d in result.machinery:
        _box(canvas, d.xyxy, config.COLORS["machinery"], f"Machinery {d.conf:.2f}")
    for d in result.detections:
        _box(canvas, d.xyxy, _color_for(d.cls), _label(d))
    if violations:
        for v in violations:
            if v.detection is not None:
                _box(canvas, v.detection.xyxy, config.COLORS["violation"],
                     v.type.replace("_", " ").upper(), thickness=3)
    return canvas
