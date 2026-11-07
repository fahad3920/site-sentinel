"""One-call pipeline: photo in -> detections, violations, score, annotated image out."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import config
from .annotate import annotate_photo
from .detector import SafetyDetector
from .scoring import SafetyScore, compute_safety
from .zones import Zone, evaluate_zones

_DETECTORS: dict = {}


def model_path_for(fast: bool) -> Path:
    """Model file for the requested speed. Falls back to the other model if one is missing."""
    want = config.PPE_MODEL_FAST if fast else config.PPE_MODEL_ACCURATE
    other = config.PPE_MODEL_ACCURATE if fast else config.PPE_MODEL_FAST
    if want.exists() or not other.exists():
        return want
    return other


def get_detector(fast: bool = False, machinery: bool = True) -> SafetyDetector:
    """Cached detector per model file (model load is the slow part).

    Previously a single detector was cached, so switching to the fast model after the
    first check had no effect. Now each model file gets its own cached detector.
    """
    path = model_path_for(fast)
    key = (str(path), machinery)
    if key not in _DETECTORS:
        _DETECTORS[key] = SafetyDetector(model_path=path, machinery=machinery)
    return _DETECTORS[key]


@dataclass
class InspectionResult:
    score: SafetyScore
    annotated: np.ndarray
    detections_count: int
    machinery_count: int
    inference_ms: float
    tiles_used: int
    image_shape: tuple
    detections: list = None
    machinery: list = None
    raw: Optional[object] = None          # DetectionResult (reused by the video module)
    violations: Optional[list] = None     # zone / machinery violations for this frame
    model: str = ""


def inspect_image(image: np.ndarray, zones: Optional[List[Zone]] = None,
                  blur: bool = True, fast: bool = False) -> InspectionResult:
    det = get_detector(fast=fast)
    result = det.detect(image)
    zones = zones or []
    violations = evaluate_zones(result.detections, result.machinery, zones, result.image_shape)
    score = compute_safety(result, violations)

    canvas = image.copy()
    if blur:
        canvas = det.blur_faces(canvas)
    annotated = annotate_photo(canvas, result, zones, violations)

    return InspectionResult(
        score=score,
        annotated=annotated,
        detections_count=len(result.detections),
        machinery_count=len(result.machinery),
        inference_ms=result.inference_ms,
        tiles_used=result.tiles_used,
        image_shape=tuple(result.image_shape),
        detections=result.detections,
        machinery=result.machinery,
        raw=result,
        violations=violations,
        model=result.model,
    )


def inspect_file(path: Path, zones: Optional[List[Zone]] = None,
                 blur: bool = True, fast: bool = False) -> InspectionResult:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return inspect_image(image, zones=zones, blur=blur, fast=fast)
