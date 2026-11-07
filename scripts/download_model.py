"""Download the Apache-2.0 YOLO11 safety weights (and optional COCO machinery proxy).

Run:  python scripts/download_model.py
Weights go to models/. The PPE model is from HuggingFace
(wesjos/Yolo-hard-hat-safety-vest, apache-2.0). The machinery proxy is the
standard Ultralytics COCO yolov8s - only used for vehicle proximity hints.
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

PPE = {
    "yolo11m_safety.pt": "https://huggingface.co/wesjos/Yolo-hard-hat-safety-vest/resolve/main/yolo11m.pt",
    "yolo11n_safety.pt": "https://huggingface.co/wesjos/Yolo-hard-hat-safety-vest/resolve/main/yolo11n.pt",
}


def _hook(blocks, bs, total):
    done = blocks * bs
    if total > 0:
        pct = min(100, done * 100 // total)
        sys.stdout.write(f"\r  {pct}% ({done//1024//1024}MB)")
        sys.stdout.flush()


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip] {dest.name} already present")
        return
    print(f"[get ] {dest.name}\n  {url}")
    urlretrieve(url, dest, reporthook=_hook)
    print(f"\n[ok  ] {dest.name} ({dest.stat().st_size//1024//1024}MB)")


def main():
    for name, url in PPE.items():
        fetch(url, MODELS / name)
    print("\nDone. PPE model classes: hat, nohat, novel, person, vest")
    print("(hat/vest = compliant, nohat/novest = violations)")


if __name__ == "__main__":
    main()
