"""Offline tests for the video module (no model needed: a fake detector is injected)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from site_sentinel.detector import Detection, DetectionResult
from site_sentinel.pipeline import InspectionResult
from site_sentinel.scoring import compute_safety
from site_sentinel.video import analyze_video, draw_hud


def _make_video(path: Path, seconds: float = 4.0, fps: int = 10, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.putText(frame, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        w.write(frame)
    w.release()


def _fake_detect_factory(sample_period: float, bad_from: float, bad_to: float):
    """Analysed frame k is at k * sample_period seconds. Between bad_from and bad_to one worker
    has no helmet; otherwise everyone wears one."""
    calls = {"n": 0}

    def run(frame):
        sec = calls["n"] * sample_period
        calls["n"] += 1
        head = "nohat" if bad_from <= sec <= bad_to else "hat"
        dets = [Detection("person", 0.9, [100, 50, 160, 220]), Detection(head, 0.8, [110, 50, 150, 90])]
        raw = DetectionResult(detections=dets, machinery=[], image_shape=frame.shape[:2], model="fake.pt")
        score = compute_safety(raw, [])
        return InspectionResult(score=score, annotated=frame.copy(), detections_count=len(dets),
                                machinery_count=0, inference_ms=1.0, tiles_used=1,
                                image_shape=frame.shape[:2], detections=dets, machinery=[],
                                raw=raw, violations=[], model="fake.pt")

    return run


def test_video_report_timeline_and_incident():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "clip.mp4"
        _make_video(src, seconds=4.0, fps=10)
        progress = []
        rep = analyze_video(src, td / "out", blur=False, sample_fps=2,
                            detect_fn=_fake_detect_factory(0.5, 1.0, 2.0),
                            progress=lambda f, m: progress.append(f))
        assert rep.frames_total == 40
        assert rep.frames_analysed == 8                     # 2 per second for 4 s
        assert rep.min_score == 90                          # one missing helmet = -10
        assert len(rep.incidents) == 1
        inc = rep.incidents[0]
        assert inc.types == ["missing_helmet"] and inc.start == 1.0
        assert rep.most_at_once["missing_helmet"] == 1
        assert 0 < rep.safe_time_pct < 100
        assert Path(rep.output_video).exists() and Path(rep.output_video).stat().st_size > 0
        assert Path(rep.timeline_csv).read_text().count("\n") == 9   # header + 8 rows
        data = json.loads(Path(rep.report_json).read_text())
        assert data["grade"] in ("Excellent", "Good") and data["incidents"][0]["duration"] > 0
        assert progress[-1] == 1.0


def test_video_max_seconds_limit():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "clip.mp4"
        _make_video(src, seconds=4.0, fps=10)
        rep = analyze_video(src, td / "out", blur=False, sample_fps=2, max_seconds=1.0,
                            detect_fn=_fake_detect_factory(0.5, 99, 99))
        assert rep.frames_total == 10 and rep.incidents == [] and rep.safe_time_pct == 100.0


def test_unreadable_video_raises():
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "not_a_video.mp4"
        bad.write_bytes(b"hello")
        try:
            analyze_video(bad, Path(td) / "out", detect_fn=lambda f: None)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_hud_draws_without_error():
    img = np.zeros((360, 640, 3), np.uint8)
    out = draw_hud(img, 72, 65.0, {"missing_helmet": 2}, 5)
    assert out.shape == img.shape and out.any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
