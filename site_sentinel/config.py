"""Site Sentinel central configuration.

Class semantics come from the Apache-2.0 YOLO11 safety model
(wesjos/Yolo-hard-hat-safety-vest): ['hat','nohat','novest','person','vest'].
'hat'/'vest' are compliant PPE; 'nohat'/'novest' are violations.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = PROJECT_ROOT / "datasets" / "ppe2286"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

# Primary PPE model (Apache-2.0). 'm' = best accuracy, 'n' = fastest on CPU.
PPE_MODEL_ACCURATE = MODELS_DIR / "yolo11m_safety.pt"
PPE_MODEL_FAST = MODELS_DIR / "yolo11n_safety.pt"

# Optional secondary COCO model used only as a heavy-machinery proxy
# (vehicle classes). Set to None to disable proximity alerts.
MACHINERY_PROXY_MODEL = MODELS_DIR / "yolov8s.pt"
MACHINERY_CLASSES = {"truck", "bus", "car", "motorcycle"}

PPE_CLASSES = ["hat", "nohat", "novest", "person", "vest"]
VIOLATION_CLASSES = {"nohat": "missing_helmet", "novest": "missing_vest"}
COMPLIANT_CLASSES = {"hat": "helmet_ok", "vest": "vest_ok"}

# Class-specific confidence thresholds: heads/vests are small, people are easy.
CONF_THRESHOLDS = {
    "hat": 0.35,
    "nohat": 0.35,
    "vest": 0.30,
    "novest": 0.30,
    "person": 0.40,
}
DEFAULT_CONF = 0.35
MACHINERY_CONF = 0.35

# Tiling: images larger than this get split so distant helmets stay detectable.
TILE_MAX_SIDE = 1280
TILE_OVERLAP = 0.15
MERGE_IOU = 0.5

# Safety-score penalties (points deducted per violation instance).
PENALTIES = {
    "missing_helmet": 10,
    "missing_vest": 6,
    "restricted_zone": 12,
    "machinery_proximity": 8,
}
SCORE_FLOOR = 0
SCORE_CEIL = 100

# Weighted milestone checklist used by the progress module (sums to 1.0).
STAGE_WEIGHTS = {
    "excavation": 0.15,
    "foundation": 0.15,
    "columns": 0.20,
    "slab": 0.20,
    "brickwork": 0.15,
    "plaster": 0.15,
}

COLORS = {
    "compliant": (22, 163, 74),     # green
    "violation": (220, 38, 38),     # red
    "person": (37, 99, 235),        # blue
    "machinery": (234, 88, 12),     # safety orange
    "zone": (234, 88, 12),
}
