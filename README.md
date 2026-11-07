# 🦺 Site Sentinel — Construction Safety & Progress Tracking

CPU-only computer-vision tool that reads a **single site photo** and answers two
questions a supervisor actually asks:

1. **Is anyone working unsafely right now?** (missing helmet/vest, inside a
   restricted zone, too close to machinery) → a plain-language **safety score 0–100**.
2. **How far along is the build?** → a **weighted milestone checklist** the
   supervisor confirms, with photo-timeline "change detected" hints.

Built for sites with **no GPU, no internet at the edge, and workers who may
prefer Urdu**. Everything runs on a normal laptop.

---

## Why this model (licensing + accuracy)

PPE detection uses **`wesjos/Yolo-hard-hat-safety-vest`** (YOLO11, **Apache-2.0**)
from HuggingFace — *not* Ultralytics' AGPL weights — so the project stays
**commercially usable without the AGPL obligation**.

Classes: `hat`, `nohat`, `novest`, `person`, `vest`
(`hat`/`vest` = compliant, `nohat`/`novest` = violations).

Model-card **mAP50** (reported by the model author, not measured here):
| variant | mAP50 |
|---|---|
| yolo11m | **0.809** |
| yolo11n | 0.708 |

We ship **both**: `m` for accuracy, `n` for speed on weak CPUs.

### Honest accuracy statement (read this)

- The mAP50 figures above are **from the model card**, not measured on your site.
- **Measured on the bundled ground-truth dataset** (`datasets/ppe2286`, Roboflow
  `20250731-PPE2286`, **CC BY 4.0**, 457 validation images), the *base* weights
  score **mAP@0.5 ≈ 0.50** with high recall but low precision — i.e. the base
  model over-flags on this annotation domain. Run
  `python scripts/evaluate.py` to reproduce; results land in
  `outputs/metrics.json` and appear in the dashboard **Accuracy** tab.
- Because of that gap we **fine-tune on the bundled train split**
  (`scripts/finetune.py`, CPU transfer learning from the Apache-2.0 base). The
  fine-tuned weights stay **Apache-2.0**. See `outputs/runs/` for the trained
  model and its validation metrics.
- We deliberately tune for **precision over recall**: a missed violation is bad,
  but a *false alarm* erodes trust faster. Class-specific confidence thresholds
  plus tiling keep small distant helmets detectable without flooding false
  positives. Thresholds were swept on the validation split
  (`scripts/tune_thresholds.py`).
- **Site Sentinel gives an estimate to support a supervisor. It is not a legal
  compliance measurement.** The UI and reports say this explicitly.

---

## Features (MVP + v2)

**Safety (MVP)**
- Missing **helmet / vest** detection with green (OK) / red (violation) boxes.
- **Restricted-zone polygons** — person center-bottom inside polygon = violation.
- **Machinery proximity** — person box within 0.75× their height of a vehicle
  (COCO truck/bus/car/motorcycle proxy) = violation.
- **Daily safety score** = 100 − Σ penalties, clamped 0–100
  (helmet −10, vest −6, restricted zone −12, machinery −8).
- **Tiling** for large photos (1280 px tiles, 15 % overlap, NMS merge) so distant
  workers are not missed.
- **Face blur by default** (Haar cascade) — reports show compliance, not people.

**Progress (v2)**
- Weighted milestone checklist: excavation .15, foundation .15, columns .20,
  slab .20, brickwork .15, plaster .15 (sums to 1.0) → **% complete**.
- **Supervisor confirmation** stamps who/when; unconfirmed shows as "estimate".
- **Photo timeline** with change-from-previous hint (visual diff only — never
  presented as a percentage on its own).

**Video (v0.2)**
- **Upload a site video** (MP4/MOV/AVI/MKV/WEBM) in the dashboard **Video check** tab,
  the CLI or the API. The detector checks `sample_fps` frames per second (default 2)
  and draws the latest boxes on **every** frame, so the output video plays smoothly.
- Output: **annotated H.264 MP4** with a live score badge, a **score-over-time chart**
  with incidents shaded red, **incident cards** (time range, violation types,
  snapshot of the worst moment, "play from here"), CSV timeline and JSON report.
- Summary: average and lowest score, **safe time %**, seconds with each violation and
  the **most at once** (frame counts are never presented as a head count of people).
- Videos feed the weekly PDF (incident snapshots included) and WhatsApp/email text.

**Delivery**
- **FastAPI** backend (`app/api.py`) + **dark Streamlit dashboard**
  (`app/dashboard.py`) with safety-orange accent, big plain-language cards,
  green/red boxes, timeline slider, zone map.
- **Urdu / English toggle** in the dashboard.
- **Weekly PDF report** (ReportLab) + copy-paste **WhatsApp / email** text.
- **CLI** for one-off or folder batch inspection.

---

## Install (Windows)

```bat
:: 1. create venv + CPU torch
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

:: 2. fetch Apache-2.0 weights (already bundled in models/ if you got the ZIP)
python scripts\download_model.py
```

> `models/yolo11m_safety.pt` and `yolo11n_safety.pt` ship inside the ZIP so the
> tool works **offline** after install.

## Run

```bat
run_api.bat          :: FastAPI on http://127.0.0.1:8100
run_dashboard.bat    :: Streamlit on http://127.0.0.1:8501
```

Or CLI (photos, folders and videos):

```bat
.venv\Scripts\python.exe -m app.cli samples\photo.jpg --save-annotated outputs\annotated
.venv\Scripts\python.exe -m app.cli samples --fast      :: whole folder, fast model
.venv\Scripts\python.exe -m app.cli site_walk.mp4 --fast --sample-fps 2   :: video
```

## API quick reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model + inspection count |
| POST | `/inspect` | upload photo → score + annotated JPEG (base64) |
| GET/POST | `/zones` | read / save restricted-zone polygons |
| GET | `/progress` | weighted % complete |
| POST | `/progress/confirm` | supervisor confirms a stage |
| POST | `/report` | build weekly PDF + WhatsApp/email text |
| GET | `/report/latest/pdf` | download latest PDF |
| POST | `/inspect/video` | upload a video → starts a background job (`job_id`) |
| GET | `/jobs/{job_id}` | progress 0–1, message, and the full video report when done |
| GET | `/jobs/{job_id}/video` | download the annotated MP4 |

Example:

```bash
curl -X POST http://127.0.0.1:8100/inspect -F "file=@samples/photo.jpg" -F "blur=true"
curl -X POST http://127.0.0.1:8100/inspect/video -F "file=@site_walk.mp4" -F "sample_fps=2"
curl http://127.0.0.1:8100/jobs/<job_id>
```

## Tests

```bat
set PYTHONPATH=%CD%
.venv\Scripts\python.exe tests\test_core.py
.venv\Scripts\python.exe tests\test_video.py
```

9/9 offline unit tests cover scoring math, zone point-in-polygon, machinery
proximity, progress weights/confirmation, photo-diff, and report payloads.

---

## Project layout

```
site-sentinel/
├─ site_sentinel/          # core library (no web deps)
│  ├─ config.py            # thresholds, penalties, stage weights, colors
│  ├─ detector.py          # YOLO11 PPE + tiling + NMS + face blur
│  ├─ zones.py             # restricted-zone + machinery-proximity violations
│  ├─ scoring.py           # safety score + plain-language advice
│  ├─ annotate.py          # green/red boxes + zone polygons
│  ├─ progress.py          # weighted checklist + photo-timeline diff
│  ├─ report.py            # weekly PDF + WhatsApp/email text
│  ├─ video.py             # video analysis: annotated MP4, timeline, incidents
│  └─ pipeline.py          # one-call photo → result
├─ app/
│  ├─ api.py               # FastAPI backend
│  ├─ dashboard.py         # dark Streamlit UI (Urdu/English)
│  └─ cli.py               # terminal inspector
├─ scripts/download_model.py
├─ scripts/evaluate.py        # measured mAP/P/R/F1 on bundled dataset -> outputs/metrics.json
├─ scripts/tune_thresholds.py # sweep conf thresholds on valid split
├─ scripts/finetune.py        # CPU transfer-learning on bundled train split
├─ tests/test_core.py
├─ models/                 # Apache-2.0 YOLO11 weights + LICENSE_model.txt
├─ datasets/ppe2286/       # YOUR ground-truth PPE dataset (CC BY 4.0): train 1829 / valid 457
├─ samples/                # real construction photos for verification
└─ outputs/                # annotated images, metrics.json, weekly PDFs, runs/
```

## Using your own dataset (the one you supplied)

The bundled `datasets/ppe2286` is the **Dataset of Personal Protective Equipment
(PPE)** you provided (Roboflow `20250731-PPE2286`, **CC BY 4.0**), YOLO format,
classes `Helmet/NoHelmet/NoVest/Vest` mapped to model classes
`hat/nohat/novest/vest`.

```bat
:: measure current accuracy on its validation split
.venv\Scripts\python.exe scripts\evaluate.py

:: fine-tune on its training split (CPU, ~8 min/epoch at imgsz 416)
.venv\Scripts\python.exe scripts\finetune.py --epochs 12 --imgsz 416 --batch 8
```

Fine-tuned weights land in `outputs/runs/<name>/weights/best.pt`. To use one as
the production model, copy it over `models/yolo11n_safety.pt` (or point
`config.PPE_MODEL_FAST` at it).

## Roadmap (not built yet — be honest in demos)

- Real heavy-machinery classes (excavator/crane) instead of the COCO vehicle proxy.
- Hard-negative mining for false-alarm reduction on your own site photos.
- Persistent storage (SQLite) + multi-site + auth.
- On-device (ONNX/quantized) build for phones.

## License note

- **Code**: MIT-style, free to use.
- **PPE weights**: Apache-2.0 (see `models/LICENSE_model.txt`) — commercially safe.
- **Machinery proxy** (`yolov8s.pt`, optional): Ultralytics AGPL — used only as a
  proximity hint. If you ship commercially, drop it or swap for an Apache-2.0 model.
