"""Site Sentinel CLI - inspect one image or a whole folder from the terminal.

Examples:
  python -m app.cli samples/photo.jpg
  python -m app.cli samples --save-annotated outputs/annotated
  python -m app.cli site_walk.mp4 --fast            (video: annotated MP4 + timeline + incidents)
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from site_sentinel import config
from site_sentinel.pipeline import inspect_file
from site_sentinel.video import VIDEO_EXTENSIONS, analyze_video, mm_ss
from site_sentinel.zones import Zone


def _print_result(path: str, res):
    s = res.score
    print(f"\n== {path} ==")
    print(f"  score      : {s.score}/100 ({s.grade})")
    print(f"  workers    : {s.workers}  helmets_ok: {s.compliant_helmets}  vests_ok: {s.compliant_vests}")
    if s.counts:
        print(f"  violations : {s.counts}")
    print(f"  detections : {res.detections_count}  machinery: {res.machinery_count}")
    print(f"  timing     : {res.inference_ms:.0f} ms  tiles: {res.tiles_used}  shape: {res.image_shape}")
    for a in s.advice:
        print(f"  - {a}")


def main():
    ap = argparse.ArgumentParser(description="Site Sentinel safety inspector")
    ap.add_argument("path", help="image file or folder")
    ap.add_argument("--fast", action="store_true", help="use yolo11n (faster, less accurate)")
    ap.add_argument("--no-blur", action="store_true", help="disable face blurring")
    ap.add_argument("--save-annotated", metavar="DIR", help="write annotated images here")
    ap.add_argument("--sample-fps", type=float, default=2.0, help="video: frames analysed per second")
    ap.add_argument("--max-seconds", type=float, default=None, help="video: only analyse the first N seconds")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
        _run_video(p, args)
        return
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.jpg")) + glob.glob(str(p / "*.png")))
    else:
        files = [str(p)]

    out_dir = Path(args.save_annotated) if args.save_annotated else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_dets = 0
    for f in files:
        try:
            res = inspect_file(Path(f), blur=not args.no_blur, fast=args.fast)
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        _print_result(f, res)
        total_dets += res.detections_count
        if out_dir:
            dest = out_dir / (Path(f).stem + "_annotated.jpg")
            cv2.imwrite(str(dest), res.annotated)
            print(f"  saved      : {dest}")

    print(f"\nProcessed {len(files)} image(s), {total_dets} total detections.")
    print("Site Sentinel provides estimates to support supervisors, not legal compliance.")


def _run_video(p: Path, args):
    out_dir = Path(args.save_annotated) if args.save_annotated else config.OUTPUT_DIR / "videos" / p.stem

    def progress(frac, msg):
        bar = "#" * int(frac * 30)
        print(f"\r  [{bar:<30}] {frac * 100:5.1f}%  {msg}", end="", flush=True)

    rep = analyze_video(p, out_dir, blur=not args.no_blur, fast=args.fast, sample_fps=args.sample_fps,
                        max_seconds=args.max_seconds, progress=progress)
    print(f"\n\n== {p.name} ==")
    print(f"  length      : {mm_ss(rep.duration_s)}  ({rep.frames_analysed} frames analysed at {rep.sample_fps} fps)")
    print(f"  safety      : average {rep.average_score}/100 ({rep.grade}), lowest {rep.min_score} at {mm_ss(rep.worst_t)}")
    print(f"  safe time   : {rep.safe_time_pct}%   most workers seen: {rep.max_workers}")
    for k, secs in rep.violation_seconds.items():
        print(f"  {k:<20}: {secs:.0f} s  (most at once: {rep.most_at_once.get(k, 0)})")
    for n, inc in enumerate(rep.incidents, 1):
        print(f"  incident #{n}: {mm_ss(inc.start)}-{mm_ss(inc.end)}  {', '.join(inc.types)}  score {inc.worst_score}")
    print(f"  video       : {rep.output_video}")
    print(f"  timeline    : {rep.timeline_csv}")
    print(f"  processing  : {rep.processing_s} s")
    print("Site Sentinel provides estimates to support supervisors, not legal compliance.")


if __name__ == "__main__":
    main()
