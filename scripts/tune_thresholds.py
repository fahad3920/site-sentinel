"""Tune per-class confidence thresholds on the bundled validation split.

Caches ONE raw low-conf prediction pass to outputs/raw_preds.json so repeated
sweeps are instant. Then, per class, picks the confidence threshold that
maximises F1 and reports a higher-precision alternative (spec favours
precision over recall to avoid alarm fatigue).

Run:  python scripts/tune_thresholds.py            # infer + sweep
      python scripts/tune_thresholds.py --sweep    # reuse cache, sweep only
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
sys.path.insert(0, str(ROOT / "scripts"))

from site_sentinel import config
from evaluate import DATASET, EVAL_CLASSES, GT_TO_MODEL, IOU_MATCH, _iou, load_gt

CACHE = ROOT / "outputs" / "raw_preds.json"


def infer(split="valid"):
    from ultralytics import YOLO
    model = YOLO(str(config.PPE_MODEL_ACCURATE))
    names = {int(k): v for k, v in model.names.items()}
    img_dir = DATASET / split / "images"
    lbl_dir = DATASET / split / "labels"
    store = {"preds": defaultdict(list), "gt": defaultdict(dict)}
    t0 = time.perf_counter()
    files = sorted(img_dir.glob("*.jpg"))
    for n, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        shape = img.shape[:2]
        gt = load_gt(lbl_dir / (f.stem + ".txt"), shape)
        for cls in EVAL_CLASSES:
            store["gt"][cls][f.name] = gt.get(cls, [])
        res = model.predict(img, verbose=False, device="cpu", conf=0.001, iou=0.7)
        for r in res:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                cname = names.get(int(r.boxes.cls[i]), "")
                if cname not in EVAL_CLASSES:
                    continue
                store["preds"][cname].append(
                    [float(r.boxes.conf[i]), r.boxes.xyxy[i].tolist(), f.name])
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(files)} ({time.perf_counter()-t0:.0f}s)")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(store), encoding="utf-8")
    print(f"cached {CACHE} in {time.perf_counter()-t0:.0f}s")
    return store


def pr_at(preds, gt, thresh):
    """precision/recall at a single confidence threshold (greedy IoU match)."""
    sel = [p for p in preds if p[0] >= thresh]
    sel.sort(key=lambda p: p[0], reverse=True)
    n_gt = sum(len(v) for v in gt.values())
    matched = {img: [False] * len(b) for img, b in gt.items()}
    tp = fp = 0
    for conf, box, img in sel:
        gts = gt.get(img, [])
        bi, bv = -1, 0.0
        for gi, gb in enumerate(gts):
            v = _iou(box, gb)
            if v > bv:
                bv, bi = v, gi
        if bi >= 0 and bv >= IOU_MATCH and not matched[img][bi]:
            matched[img][bi] = True
            tp += 1
        else:
            fp += 1
    fn = n_gt - tp
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / n_gt if n_gt else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F, tp, fp, fn


def sweep(store):
    print("\nper-class threshold sweep (F1-max, plus precision>=0.7 option):")
    rec = {}
    for cls in EVAL_CLASSES:
        preds = store["preds"].get(cls, [])
        gt = store["gt"].get(cls, {})
        n_gt = sum(len(v) for v in gt.values())
        if n_gt == 0 or not preds:
            rec[cls] = config.CONF_THRESHOLDS.get(cls, 0.35)
            print(f"  {cls:8s} skip (no gt/preds)")
            continue
        best = None
        prec_opt = None
        for t in np.arange(0.05, 0.95, 0.025):
            P, R, F, tp, fp, fn = pr_at(preds, gt, float(t))
            if best is None or F > best[1]:
                best = (float(t), F, P, R)
            if prec_opt is None and P >= 0.7 and R > 0.3:
                prec_opt = (float(t), P, R, F)
        t_f1, F, P, R = best
        line = (f"  {cls:8s} gt={n_gt:4d}  F1-max t={t_f1:.3f} P={P:.2f} R={R:.2f} F1={F:.2f}")
        if prec_opt:
            line += f"  |  P>=0.7 t={prec_opt[0]:.3f} P={prec_opt[1]:.2f} R={prec_opt[2]:.2f}"
        print(line)
        rec[cls] = round(t_f1, 3)
    print("\nrecommended CONF_THRESHOLDS (F1-max):")
    print("  " + json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="reuse cache, no inference")
    args = ap.parse_args()
    if args.sweep and CACHE.exists():
        store = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        store = infer()
    sweep(store)


if __name__ == "__main__":
    main()
