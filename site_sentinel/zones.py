"""Restricted-zone and machinery-proximity violation logic."""
from dataclasses import dataclass
from typing import List, Tuple

from .detector import Detection

Point = Tuple[float, float]


@dataclass
class Zone:
    name: str
    polygon: List[Point]  # normalized (x, y) in [0,1] relative to image size
    kind: str = "restricted"  # 'restricted' | 'machinery'

    def contains(self, point: Point, image_shape: Tuple[int, int]) -> bool:
        h, w = image_shape[:2]
        px, py = point[0] * w, point[1] * h
        poly = [(x * w, y * h) for (x, y) in self.polygon]
        return _point_in_polygon((px, py), poly)


def _point_in_polygon(pt: Point, poly: List[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class Violation:
    type: str
    detail: str
    detection: Detection = None
    zone: str = ""


def _box_gap(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return (dx ** 2 + dy ** 2) ** 0.5


def evaluate_zones(detections: List[Detection], machinery: List[Detection],
                   zones: List[Zone], image_shape, proximity_ratio: float = 0.75) -> List[Violation]:
    """Flag people inside restricted zones and people standing near machinery."""
    violations: List[Violation] = []
    people = [d for d in detections if d.cls == "person"]
    h, w = shape = tuple(image_shape[:2])

    for zone in zones:
        if zone.kind != "restricted":
            continue
        for person in people:
            cx, cy = person.center_bottom
            normalized = (cx / w if w else 0.0, cy / h if h else 0.0)
            if zone.contains(normalized, shape):
                violations.append(
                    Violation(
                        type="restricted_zone",
                        detail=f"Person inside restricted zone '{zone.name}'",
                        detection=person,
                        zone=zone.name,
                    )
                )

    for person in people:
        person_h = person.xyxy[3] - person.xyxy[1]
        for mach in machinery:
            if _box_gap(person, mach) < proximity_ratio * person_h:
                violations.append(
                    Violation(
                        type="machinery_proximity",
                        detail=f"Person within close range of {mach.cls} (machinery proxy)",
                        detection=person,
                    )
                )
                break

    return violations


# Convenience: attach normalized center-bottom to Detection via property
def _center_bottom_normalized(self):
    return (self.center_bottom[0], self.center_bottom[1])


Detection.center_bottom_normalized = property(
    lambda self: (self.center_bottom[0], self.center_bottom[1])
)
