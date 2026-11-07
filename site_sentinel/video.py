"""Video safety analysis: upload a site video, get an annotated video + a timeline.

How it works (CPU friendly):
  * Only `sample_fps` frames per second are run through the detector (default 2).
  * The latest detections are drawn on EVERY frame, so the output video plays smoothly.
  * Each analysed frame gets the same safety score as a photo (scoring.py).
  * Consecutive frames with violations are grouped into incidents with a snapshot
    of the worst moment.
  * The output is H.264 MP4 (plays in browsers) when ffmpeg is available through
    imageio-ffmpeg; otherwise an MPEG-4 file that still downloads and plays in VLC.

Honest note: counts are per analysed frame. The same worker seen in 10 frames is
10 frame-detections, so the summary reports "seconds with a violation" and the
"most at once", never a head count of different people.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from . import config
from .annotate import annotate_photo
from .pipeline import get_detector, inspect_image
from .zones import Zone

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")

ProgressFn = Callable[[float, str], None]


@dataclass
class FrameRecord:
    t: float                      # seconds from the start
    frame_index: int
    score: int
    grade: str
    workers: int
    helmets_ok: int
    vests_ok: int
    counts: Dict[str, int] = field(default_factory=dict)
    detections: int = 0


@dataclass
class Incident:
    start: float
    end: float
    types: List[str]
    worst_score: int
    worst_t: float
    snapshot: str = ""            # path to the annotated JPEG of the worst moment

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 1)


@dataclass
class VideoReport:
    source: str
    duration_s: float
    fps: float
    width: int
    height: int
    frames_total: int
    frames_analysed: int
    sample_fps: float
    model: str
    processing_s: float
    average_score: float
    min_score: int
    worst_t: float
    safe_time_pct: float          # share of analysed frames without any violation
    violation_seconds: Dict[str, float] = field(default_factory=dict)
    most_at_once: Dict[str, int] = field(default_factory=dict)
    max_workers: int = 0
    incidents: List[Incident] = field(default_factory=list)
    frames: List[FrameRecord] = field(default_factory=list)
    output_video: str = ""
    worst_frame: str = ""
    report_json: str = ""
    timeline_csv: str = ""
    browser_playable: bool = False

    @property
    def grade(self) -> str:
        s = self.average_score
        return "Excellent" if s >= 90 else "Good" if s >= 75 else "Needs attention" if s >= 55 else "Critical"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grade"] = self.grade
        for inc, raw in zip(self.incidents, d["incidents"]):
            raw["duration"] = inc.duration
        return d


# ---------------------------------------------------------------- drawing ----

def _score_color(score: int):
    if score >= 75:
        return (94, 197, 34)      # green (BGR)
    if score >= 55:
        return (11, 158, 245)     # amber
    return (68, 68, 239)          # red


def draw_hud(frame: np.ndarray, score: int, t: float, counts: Dict[str, int], workers: int) -> np.ndarray:
    """Score badge, time stamp and live violation chips in the top-left corner."""
    h, w = frame.shape[:2]
    scale = max(0.5, min(1.4, w / 1280))
    pad = int(14 * scale)
    color = _score_color(score)

    # translucent panel
    panel_w, panel_h = int(430 * scale), int(96 * scale)
    col = pad + int(150 * scale)   # text column, right of the big score number
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (20, 14, 7), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.rectangle(frame, (pad, pad), (pad + panel_w, pad + panel_h), color, max(1, int(2 * scale)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{score}", (pad + int(12 * scale), pad + int(68 * scale)), font,
                1.7 * scale, color, max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.putText(frame, "SAFETY SCORE", (col, pad + int(34 * scale)), font,
                0.55 * scale, (200, 200, 200), max(1, int(1 * scale)), cv2.LINE_AA)
    mm, ss = divmod(int(t), 60)
    cv2.putText(frame, f"{mm:02d}:{ss:02d}  |  {workers} worker(s)", (col, pad + int(62 * scale)),
                font, 0.55 * scale, (240, 240, 240), max(1, int(1 * scale)), cv2.LINE_AA)
    chips = ", ".join(f"{k.replace('_', ' ')} x{v}" for k, v in counts.items()) or "no violations"
    cv2.putText(frame, chips[:34], (col, pad + int(86 * scale)), font,
                0.45 * scale, (120, 120, 255) if counts else (140, 220, 140), max(1, int(1 * scale)), cv2.LINE_AA)

    # brand tag top-right
    tag = "SITE SENTINEL  AI"
    (tw, th), _ = cv2.getTextSize(tag, font, 0.5 * scale, 1)
    cv2.rectangle(frame, (w - tw - 3 * pad, pad), (w - pad, pad + th + pad), (12, 88, 234), -1)
    cv2.putText(frame, tag, (w - tw - 2 * pad, pad + th + pad // 2), font, 0.5 * scale,
                (255, 255, 255), max(1, int(1 * scale)), cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------- writing ----

class _VideoWriter:
    """H.264 through imageio-ffmpeg when installed (browser playable), else OpenCV mp4v."""

    def __init__(self, path: Path, size, fps: float):
        self.path = path
        self.browser_playable = False
        self._gen = None
        self._cv = None
        try:
            import imageio_ffmpeg

            self._gen = imageio_ffmpeg.write_frames(
                str(path), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                macro_block_size=2, quality=7, ffmpeg_log_level="error",
                output_params=["-movflags", "+faststart"])
            self._gen.send(None)
            self.browser_playable = True
        except Exception:
            self._gen = None
            self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame_bgr: np.ndarray):
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        else:
            self._cv.write(frame_bgr)

    def close(self):
        if self._gen is not None:
            self._gen.close()
        if self._cv is not None:
            self._cv.release()


# ---------------------------------------------------------------- analysis ---

def _group_incidents(records: List[FrameRecord], gap_s: float) -> List[List[FrameRecord]]:
    groups: List[List[FrameRecord]] = []
    for r in records:
        if not r.counts:
            continue
        if groups and r.t - groups[-1][-1].t <= gap_s:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def analyze_video(
    path: Path,
    out_dir: Path,
    zones: Optional[List[Zone]] = None,
    blur: bool = True,
    fast: bool = True,
    sample_fps: float = 2.0,
    max_seconds: Optional[float] = None,
    max_width: int = 1280,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    detect_fn=None,
) -> VideoReport:
    """Analyse a video file. `detect_fn(frame)` can replace inspect_image (used in tests)."""
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zones = zones or []
    started = time.perf_counter()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path.name}. Try MP4 (H.264).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    limit_frames = int(max_seconds * fps) if max_seconds else None
    planned = min(total, limit_frames) if (total and limit_frames) else (total or limit_frames or 0)
    step = max(1, int(round(fps / max(0.1, sample_fps))))

    scale = min(1.0, max_width / src_w) if src_w else 1.0
    out_w = int(src_w * scale) // 2 * 2 or 2
    out_h = int(src_h * scale) // 2 * 2 or 2

    stem = path.stem.replace(" ", "_")[:40] or "video"
    out_video = out_dir / f"{stem}_annotated.mp4"
    writer = _VideoWriter(out_video, (out_w, out_h), fps)

    blur_det = None
    if blur and detect_fn is None:
        blur_det = get_detector(fast=fast)

    def run_detect(frame):
        if detect_fn is not None:
            return detect_fn(frame)
        return inspect_image(frame, zones=zones, blur=blur, fast=fast)

    records: List[FrameRecord] = []
    snapshots: Dict[int, np.ndarray] = {}     # frame_index -> annotated (only violation frames)
    last = None
    model_name = ""
    idx = 0
    worst_img, worst_score, worst_t = None, 101, 0.0

    try:
        while True:
            if limit_frames is not None and idx >= limit_frames:
                break
            if should_stop and should_stop():
                break
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            elif frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))
            t = idx / fps

            if idx % step == 0 or last is None:
                last = run_detect(frame)
                model_name = getattr(last, "model", "") or model_name
                s = last.score
                rec = FrameRecord(t=round(t, 2), frame_index=idx, score=s.score, grade=s.grade,
                                  workers=s.workers, helmets_ok=s.compliant_helmets,
                                  vests_ok=s.compliant_vests, counts=dict(s.counts),
                                  detections=last.detections_count)
                records.append(rec)
                annotated = last.annotated
                if s.counts:
                    snapshots[idx] = annotated
                if s.score < worst_score:
                    worst_score, worst_t, worst_img = s.score, t, annotated
                if progress:
                    frac = (idx + 1) / planned if planned else 0.0
                    progress(min(0.99, frac), f"Analysed {len(records)} frame(s) - {mm_ss(t)}")
            else:
                # Re-draw the latest detections on this in-between frame (no new inference).
                canvas = frame
                if blur_det is not None:
                    canvas = blur_det.blur_faces(canvas)
                raw = getattr(last, "raw", None)
                annotated = (annotate_photo(canvas, raw, zones, last.violations or [])
                             if raw is not None else canvas)

            s = last.score
            writer.write(draw_hud(annotated, s.score, t, s.counts, s.workers))
            idx += 1
    finally:
        cap.release()
        writer.close()

    if not records:
        raise ValueError("No frames could be read from this video.")

    analysed = len(records)
    scores = [r.score for r in records]
    seconds_per_sample = step / fps
    violation_seconds: Dict[str, float] = {}
    most_at_once: Dict[str, int] = {}
    for r in records:
        for k, v in r.counts.items():
            violation_seconds[k] = round(violation_seconds.get(k, 0.0) + seconds_per_sample, 1)
            most_at_once[k] = max(most_at_once.get(k, 0), v)

    incidents: List[Incident] = []
    for n, group in enumerate(_group_incidents(records, gap_s=max(1.5, 2 * seconds_per_sample)), 1):
        worst = min(group, key=lambda r: r.score)
        types = sorted({k for r in group for k in r.counts})
        snap_path = out_dir / f"{stem}_incident_{n:02d}.jpg"
        img = snapshots.get(worst.frame_index)
        if img is not None:
            cv2.imwrite(str(snap_path), draw_hud(img, worst.score, worst.t, worst.counts, worst.workers),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        incidents.append(Incident(start=group[0].t, end=round(group[-1].t + seconds_per_sample, 2),
                                  types=types, worst_score=worst.score, worst_t=worst.t,
                                  snapshot=str(snap_path) if img is not None else ""))

    worst_path = ""
    if worst_img is not None:
        worst_path = str(out_dir / f"{stem}_worst.jpg")
        cv2.imwrite(worst_path, worst_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    report = VideoReport(
        source=path.name,
        duration_s=round(idx / fps, 2),
        fps=round(fps, 2),
        width=out_w,
        height=out_h,
        frames_total=idx,
        frames_analysed=analysed,
        sample_fps=round(fps / step, 2),
        model=model_name or Path(config.PPE_MODEL_FAST if fast else config.PPE_MODEL_ACCURATE).name,
        processing_s=round(time.perf_counter() - started, 1),
        average_score=round(sum(scores) / analysed, 1),
        min_score=min(scores),
        worst_t=round(worst_t, 2),
        safe_time_pct=round(100.0 * sum(1 for r in records if not r.counts) / analysed, 1),
        violation_seconds=violation_seconds,
        most_at_once=most_at_once,
        max_workers=max(r.workers for r in records),
        incidents=incidents,
        frames=records,
        output_video=str(out_video),
        worst_frame=worst_path,
        browser_playable=writer.browser_playable,
    )

    # Machine-readable outputs next to the video
    report.timeline_csv = str(out_dir / f"{stem}_timeline.csv")
    with open(report.timeline_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "score", "grade", "workers", "helmets_ok", "vests_ok",
                    "missing_helmet", "missing_vest", "restricted_zone", "machinery_proximity"])
        for r in records:
            w.writerow([r.t, r.score, r.grade, r.workers, r.helmets_ok, r.vests_ok,
                        r.counts.get("missing_helmet", 0), r.counts.get("missing_vest", 0),
                        r.counts.get("restricted_zone", 0), r.counts.get("machinery_proximity", 0)])
    report.report_json = str(out_dir / f"{stem}_report.json")
    Path(report.report_json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if progress:
        progress(1.0, "Done")
    return report


def mm_ss(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"
