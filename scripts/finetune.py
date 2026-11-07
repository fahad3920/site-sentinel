"""Fine-tune the Apache-2.0 YOLO11 safety weights on the bundled PPE dataset.

CPU-only transfer learning: backbone warm-starts from yolo11n_safety.pt; the
detection head is re-initialised for this dataset's 4 classes. The result stays
Apache-2.0 (fine-tuning an Apache-2.0 base does not add AGPL).

Run (probe speed first):  python scripts/finetune.py --epochs 1 --imgsz 416
Run (real):               python scripts/finetune.py --epochs 20 --imgsz 640
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from site_sentinel import config

DATA_YAML = config.DATASET_DIR / "data_ss.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--model", default=str(config.PPE_MODEL_FAST))
    ap.add_argument("--name", default="ss_finetune")
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml_path = config.OUTPUT_DIR / "data_finetune.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {config.DATASET_DIR.as_posix()}\n"
        f"train: {(config.DATASET_DIR / 'train' / 'images').as_posix()}\n"
        f"val: {(config.DATASET_DIR / 'valid' / 'images').as_posix()}\n"
        "nc: 4\n"
        "names: ['hat', 'nohat', 'novest', 'vest']\n",
        encoding="utf-8")

    model = YOLO(args.model)
    t0 = time.perf_counter()
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        patience=5,
        name=args.name,
        project=str(ROOT / "outputs" / "runs"),
        pretrained=True,
        verbose=False,
        plots=False,
    )
    el = time.perf_counter() - t0
    print(f"\nFine-tune finished in {el/60:.1f} min "
          f"({el/max(1,args.epochs)/60:.1f} min/epoch)")
    best = ROOT / "outputs" / "runs" / args.name / "weights" / "best.pt"
    print("best weights:", best, best.exists())


if __name__ == "__main__":
    main()
