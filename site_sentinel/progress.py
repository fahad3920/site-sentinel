"""Construction progress tracking: weighted milestone checklist + photo-timeline
change hints. Progress is an *estimate* that a supervisor confirms; the photo
diff only suggests "something changed here", never a hard percentage on its own.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import config


@dataclass
class Milestone:
    stage: str
    weight: float
    done: bool = False
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[str] = None

    @property
    def label(self) -> str:
        return self.stage.replace("_", " ").title()


@dataclass
class ProgressReport:
    percent: float
    milestones: List[Milestone] = field(default_factory=list)
    confirmed: bool = False
    note: str = ""


def build_checklist(state: Optional[Dict[str, bool]] = None) -> List[Milestone]:
    state = state or {}
    return [
        Milestone(stage=s, weight=w, done=bool(state.get(s, False)))
        for s, w in config.STAGE_WEIGHTS.items()
    ]


def compute_progress(milestones: List[Milestone]) -> ProgressReport:
    pct = sum(m.weight for m in milestones if m.done)
    total_w = sum(m.weight for m in milestones) or 1.0
    pct = round(100.0 * pct / total_w, 1)
    confirmed = any(m.confirmed_by for m in milestones)
    note = (
        "Supervisor-confirmed progress."
        if confirmed
        else "Unconfirmed estimate - ask a supervisor to verify the checklist."
    )
    return ProgressReport(percent=pct, milestones=milestones, confirmed=confirmed, note=note)


def confirm(milestones: List[Milestone], stage: str, supervisor: str,
            done: bool = True) -> List[Milestone]:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for m in milestones:
        if m.stage == stage:
            m.done = done
            m.confirmed_by = supervisor
            m.confirmed_at = now
    return milestones


def _signature(image: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)


def photo_change(a: np.ndarray, b: np.ndarray) -> float:
    """Return a 0..1 'how different' score between two site photos.

    This is a visual-change hint only. It cannot tell you *what* progress was
    made, so it is surfaced as "change detected" and never as a percentage.
    """
    sa, sb = _signature(a), _signature(b)
    diff = cv2.absdiff(sa, sb)
    return round(float(diff.mean()) / 255.0, 3)


@dataclass
class TimelineEntry:
    captured_at: str
    image_path: str
    progress_pct: Optional[float] = None
    change_from_prev: Optional[float] = None
    note: str = ""


def build_timeline(entries: List[TimelineEntry], images: Optional[Dict[str, np.ndarray]] = None) -> List[TimelineEntry]:
    """Sort entries chronologically and attach change-from-previous hints."""
    entries = sorted(entries, key=lambda e: e.captured_at)
    images = images or {}
    prev_img = None
    for e in entries:
        cur_img = images.get(e.image_path)
        if prev_img is not None and cur_img is not None:
            e.change_from_prev = photo_change(prev_img, cur_img)
        if cur_img is not None:
            prev_img = cur_img
    return entries
