"""Measured-accuracy evaluation on the bundled ground-truth PPE dataset.

Dataset: datasets/ppe2286 (Roboflow 20250731-PPE2286, CC BY 4.0),
classes ['Helmet','NoHelmet','NoVest','Vest'] -> mapped to model
classes ['hat','nohat','novest','vest'].

Reports TWO honest numbers:
  1. mAP@0.5  - proper per-class average precision over the full
     confidence range (raw low-conf predictions, IoU 0.5 matching).
  2. Production precision/recall/F1 - what Site Sentinel actually outputs
     at the thresholds in site_sentinel/config.py.

Run:  python scripts/evaluate.py [--split valid] [--limit N]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from site_sentinel import config
from site_sentinel.detector import Detection, _nms

DATASET = ROOT / "datasets" / "ppe2286"
# dataset class id -> model class name
GT_TO_MODEL = {0: "hat", 1: "nohat", 2: "novest", 3: "vest"}
EVAL_CLASSES = ["hat", "nohat", "novest", "vest"]
IOU_MATCH = 0.5


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path: Path, shape):
    h, w = shape
    boxes = defaultdict(list)
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        if cid not in GT_TO_MODEL:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes[GT_TO_MODEL[cid]].append([x1, y1, x2, y2])
    return boxes


def predict_raw(model, image, conf=0.001):
    """Low-conf raw predictions (no production thresholds) for mAP curves."""
    out = defaultdict(list)
    results = model.predict(image, verbose=False, device="cpu", conf=conf, iou=0.7)
    names = {int(k): v for k, v in model.names.items()}
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            cname = names.get(int(r.boxes.cls[i]), "")
            if cname not in EVAL_CLASSES:
                continue
            out[cname].append((float(r.boxes.conf[i]), r.boxes.xyxy[i].tolist()))
    return out


def predict_production(detector, image):
    """Predictions exactly as the app produces them (thresholds + tiling)."""
    res = detector.detect(image)
    out = defaultdict(list)
    for d in res.detections:
        if d.cls in EVAL_CLASSES:
            out[d.cls].append((d.conf, d.xyxy))
    return out


def average_precision(preds, gt_all, iou=IOU_MATCH):
    """preds: list of (conf, box, img_id). gt_all: {img_id: [boxes]}."""
    n_gt = sum(len(v) for v in gt_all.values())
    if n_gt == 0:
        return None, 0, 0
    preds = sorted(preds, key=lambda p: p[0], reverse=True)
    matched = {img: [False] * len(b) for img, b in gt_all.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for idx, (conf, box, img) in enumerate(preds):
        gts = gt_all.get(img, [])
        best_i, best_iou = -1, 0.0
        for gi, gb in enumerate(gts):
            v = _iou(box, gb)
            if v > best_iou:
                best_iou, best_i = v, gi
        if best_i >= 0 and best_iou >= iou and not matched[img][best_i]:
            matched[img][best_i] = True
            tp[idx] = 1
        else:
            fp[idx] = 1
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / (tp_c + fp_c)
    # AP via all-point interpolation
    mrec = np.concatenate(([0.0], rec, [rec[-1]]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return ap, float(tp_c[-1]), float(fp_c[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--limit", type=int, default=0, help="cap images (0 = all)")
    ap.add_argument("--fast", action="store_true", help="use yolo11n")
    args = ap.parse_args()

    from ultralytics import YOLO
    from site_sentinel.detector import SafetyDetector

    img_dir = DATASET / args.split / "images"
    lbl_dir = DATASET / args.split / "labels"
    files = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        files = files[:args.limit]
    print(f"Evaluating {len(files)} {args.split} images "
          f"(IoU>={IOU_MATCH}, classes={EVAL_CLASSES})")

    path = config.PPE_MODEL_FAST if args.fast else config.PPE_MODEL_ACCURATE
    model = YOLO(str(path))
    detector = SafetyDetector(model_path=path, machinery=False)

    raw_preds = defaultdict(list)      # cls -> [(conf, box, img_id)]
    prod_preds = defaultdict(list)
    gt_raw = defaultdict(lambda: defaultdict(list))   # cls -> {img: [boxes]}
    gt_prod = defaultdict(lambda: defaultdict(list))
    t0 = time.perf_counter()

    for n, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        shape = img.shape[:2]
        gt = load_gt(lbl_dir / (f.stem + ".txt"), shape)
        for cls in EVAL_CLASSES:
            gt_raw[cls][f.name] = gt.get(cls, [])
            gt_prod[cls][f.name] = gt.get(cls, [])
        for cls, items in predict_raw(model, img).items():
            for conf, box in items:
                raw_preds[cls].append((conf, box, f.name))
        for cls, items in predict_production(detector, img).items():
            for conf, box in items:
                prod_preds[cls].append((conf, box, f.name))
        if (n + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {n+1}/{len(files)}  ({el:.0f}s, {el/(n+1)*1000:.0f} ms/img)")

    el = time.perf_counter() - t0
    print(f"\nInference total {el:.0f}s  ({el/len(files)*1000:.0f} ms/img on CPU)\n")

    print("=== mAP@0.5 (full confidence range) ===")
    aps = []
    for cls in EVAL_CLASSES:
        apv, tp, fp = average_precision(raw_preds[cls], gt_raw[cls])
        if apv is None:
            print(f"  {cls:8s} no ground truth")
            continue
        aps.append(apv)
        print(f"  {cls:8s} AP={apv:.3f}  (gt={sum(len(v) for v in gt_raw[cls].values())})")
    if aps:
        print(f"  mAP@0.5 = {sum(aps)/len(aps):.3f}")

    print("\n=== Production thresholds (what the app reports) ===")
    tot_tp = tot_fp = tot_fn = 0
    for cls in EVAL_CLASSES:
        apv, tp, fp = average_precision(prod_preds[cls], gt_prod[cls])
        if apv is None:
            continue
        n_gt = sum(len(v) for v in gt_prod[cls].values())
        fn = n_gt - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        tot_tp += tp; tot_fp += fp; tot_fn += fn
        print(f"  {cls:8s} P={prec:.3f} R={rec:.3f} F1={f1:.3f}  tp={int(tp)} fp={int(fp)} fn={int(fn)}")
    P = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    R = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    print(f"  OVERALL  P={P:.3f} R={R:.3f} F1={F:.3f}")
    print("\nThese are MEASURED on the bundled dataset, not model-card claims.")

    metrics = {
        "dataset": DATASET.name,
        "split": args.split,
        "images": len(files),
        "model": Path(path).name,
        "map50": round(sum(aps) / len(aps), 3) if aps else None,
        "per_class_ap": {c: round(a, 3) for c, a in zip(EVAL_CLASSES, aps)},
        "production": {"precision": round(P, 3), "recall": round(R, 3), "f1": round(F, 3)},
        "ms_per_image": round(el / len(files) * 1000, 0),
    }
    out = ROOT / "outputs" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"metrics written to {out}")


if __name__ == "__main__":
    main()
