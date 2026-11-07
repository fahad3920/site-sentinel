"""Daily safety score + plain-language compliance summary.

The score is intentionally simple and honest: start at 100 and deduct a fixed
pen per violation instance (see config.PENALTIES), clamped to [0, 100]. It is a
supervisor aid, not a legal compliance measurement.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from . import config
from .detector import DetectionResult
from .zones import Violation


@dataclass
class SafetyScore:
    score: int
    grade: str
    workers: int
    compliant_helmets: int
    compliant_vests: int
    violations: List[Violation] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    headline: str = ""
    advice: List[str] = field(default_factory=list)


def _grade(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Needs attention"
    return "Critical"


def count_ppe(result: DetectionResult) -> Dict[str, int]:
    counts = {"helmet_ok": 0, "vest_ok": 0, "missing_helmet": 0, "missing_vest": 0, "person": 0}
    for d in result.detections:
        if d.cls in config.COMPLIANT_CLASSES:
            counts[config.COMPLIANT_CLASSES[d.cls]] += 1
        elif d.cls in config.VIOLATION_CLASSES:
            counts[config.VIOLATION_CLASSES[d.cls]] += 1
        elif d.cls == "person":
            counts["person"] += 1
    return counts


def compute_safety(result: DetectionResult, violations: List[Violation]) -> SafetyScore:
    counts = count_ppe(result)

    tally: Dict[str, int] = {}
    for d in result.detections:
        if d.cls in config.VIOLATION_CLASSES:
            key = config.VIOLATION_CLASSES[d.cls]
            tally[key] = tally.get(key, 0) + 1
    for v in violations:
        tally[v.type] = tally.get(v.type, 0) + 1

    penalty = sum(config.PENALTIES.get(k, 0) * n for k, n in tally.items())
    score = int(max(config.SCORE_FLOOR, min(config.SCORE_CEIL, config.SCORE_CEIL - penalty)))

    workers = counts["person"] or (counts["helmet_ok"] + counts["missing_helmet"])
    headline = f"{score}/100 - {_grade(score)}"

    advice: List[str] = []
    if tally.get("missing_helmet"):
        advice.append(f"{tally['missing_helmet']} worker(s) without a hard hat detected.")
    if tally.get("missing_vest"):
        advice.append(f"{tally['missing_vest']} worker(s) without a high-visibility vest.")
    if tally.get("restricted_zone"):
        advice.append(f"{tally['restricted_zone']} person(s) inside a restricted area.")
    if tally.get("machinery_proximity"):
        advice.append(f"{tally['machinery_proximity']} person(s) too close to machinery.")
    if not advice:
        advice.append("No violations detected in this frame. Keep it up.")
    advice.append("Estimate only - please confirm with a site supervisor.")

    return SafetyScore(
        score=score,
        grade=_grade(score),
        workers=workers,
        compliant_helmets=counts["helmet_ok"],
        compliant_vests=counts["vest_ok"],
        violations=violations,
        counts=tally,
        headline=headline,
        advice=advice,
    )
