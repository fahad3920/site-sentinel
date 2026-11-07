"""Offline unit tests - no model download needed (pure-logic modules only)."""
import numpy as np

from site_sentinel import config
from site_sentinel.detector import Detection
from site_sentinel.progress import (build_checklist, compute_progress, confirm,
                                    photo_change)
from site_sentinel.scoring import compute_safety
from site_sentinel.zones import Zone, evaluate_zones
from site_sentinel.report import WeekSummary, build_texts


def _det(cls, x1, y1, x2, y2, conf=0.9):
    return Detection(cls=cls, conf=conf, xyxy=[x1, y1, x2, y2])


def test_scoring_clean_frame_is_100():
    class R:
        detections = [_det("person", 0, 0, 50, 100), _det("hat", 10, 0, 30, 20),
                      _det("vest", 5, 40, 45, 100)]
        machinery = []
        image_shape = (200, 200)
    s = compute_safety(R(), [])
    assert s.score == 100, s.score
    assert s.workers == 1


def test_scoring_missing_ppe_deducts():
    class R:
        detections = [_det("person", 0, 0, 50, 100), _det("nohat", 10, 0, 30, 20),
                      _det("novest", 5, 40, 45, 100)]
        machinery = []
        image_shape = (200, 200)
    s = compute_safety(R(), [])
    # 10 (helmet) + 6 (vest)
    assert s.score == 84, s.score
    assert s.counts["missing_helmet"] == 1


def test_score_floor_zero():
    class R:
        detections = [_det("nohat", 0, 0, 10, 10)] * 20
        machinery = []
        image_shape = (100, 100)
    assert compute_safety(R(), []).score == 0


def test_restricted_zone_fires():
    person = _det("person", 40, 40, 60, 90)  # center-bottom ~ (50, 90) -> (0.5, 0.9)
    zone = Zone(name="z", polygon=[(0, 0.5), (1, 0.5), (1, 1), (0, 1)])
    v = evaluate_zones([person], [], [zone], (100, 100))
    assert len(v) == 1 and v[0].type == "restricted_zone"


def test_restricted_zone_outside_no_fire():
    person = _det("person", 40, 10, 60, 40)  # center-bottom (50,40) -> (0.5,0.4)
    zone = Zone(name="z", polygon=[(0, 0.5), (1, 0.5), (1, 1), (0, 1)])
    assert evaluate_zones([person], [], [zone], (100, 100)) == []


def test_machinery_proximity():
    person = _det("person", 10, 10, 30, 110)   # height 100
    truck = _det("truck", 35, 40, 90, 110)      # gap ~5px < 0.75*100
    v = evaluate_zones([person], [truck], [], (200, 200))
    assert any(x.type == "machinery_proximity" for x in v)


def test_progress_weights_sum_and_confirm():
    assert abs(sum(config.STAGE_WEIGHTS.values()) - 1.0) < 1e-9
    ms = build_checklist()
    assert compute_progress(ms).percent == 0.0
    ms = confirm(ms, "foundation", "Sup", done=True)
    rep = compute_progress(ms)
    assert rep.percent == 15.0 and rep.confirmed


def test_photo_change_detects_difference():
    a = np.zeros((64, 64, 3), np.uint8)
    b = np.full((64, 64, 3), 255, np.uint8)
    assert photo_change(a, a) < 0.01
    assert photo_change(a, b) > 0.5


def test_report_text_payloads():
    s = WeekSummary(site_name="X", period_start="2026-01-01", period_end="2026-01-07",
                    inspections=3, average_score=80, worst_score=64, total_violations=4,
                    top_issues=["missing vest: 3"], progress_percent=15.0,
                    progress_confirmed=True)
    t = build_texts(s)
    assert "Site Sentinel" in t["whatsapp"] and "80/100" in t["whatsapp"]
    assert t["email"].startswith("Subject:")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
