"""Site Sentinel FastAPI backend.

Endpoints:
  GET  /health                -> service + model status
  POST /inspect               -> upload a photo, get score + annotated image
  POST /zones                 -> persist restricted-zone polygons (per site)
  GET  /zones                 -> read saved zones
  POST /progress/confirm      -> supervisor confirms a milestone
  GET  /progress              -> current weighted progress estimate
  POST /report                -> build weekly PDF + WhatsApp/email text
  POST /inspect/video         -> upload a video, starts a background job
  GET  /jobs/{job_id}         -> job progress and, when done, the video report
  GET  /jobs/{job_id}/video   -> download the annotated MP4
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from site_sentinel import config
from site_sentinel.pipeline import inspect_image, model_path_for
from site_sentinel.video import VIDEO_EXTENSIONS, analyze_video
from site_sentinel.progress import build_checklist, compute_progress, confirm
from site_sentinel.report import WeekSummary, build_pdf, build_texts
from site_sentinel.zones import Zone

app = FastAPI(title="Site Sentinel", version="0.2.0",
              description="Construction safety & progress tracking (CPU-only).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

STATE_DIR = config.OUTPUT_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
ZONE_FILE = STATE_DIR / "zones.json"

_milestones = build_checklist()
_history: List[dict] = []


def _load_zones() -> List[Zone]:
    import json
    if not ZONE_FILE.exists():
        return []
    try:
        raw = json.loads(ZONE_FILE.read_text(encoding="utf-8"))
        return [Zone(name=z["name"], polygon=[tuple(p) for p in z["polygon"]],
                     kind=z.get("kind", "restricted")) for z in raw]
    except Exception:
        return []


def _save_zones(zones: List[Zone]):
    import json
    ZONE_FILE.write_text(json.dumps(
        [{"name": z.name, "polygon": [list(p) for p in z.polygon], "kind": z.kind}
         for z in zones]), encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model_path_for(False).name,
        "model_exists": model_path_for(False).exists(),
        "fast_model": model_path_for(True).name,
        "video": True,
        "classes": config.PPE_CLASSES,
        "inspections": len(_history),
    }


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")
    return img


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), blur: bool = Form(True),
                  fast: bool = Form(False)):
    data = await file.read()
    img = _decode_upload(data)
    zones = _load_zones()
    result = inspect_image(img, zones=zones, blur=blur, fast=fast)

    ok, buf = cv2.imencode(".jpg", result.annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    rec = {
        "at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "score": result.score.score,
        "violations": result.score.counts,
    }
    _history.append(rec)

    return JSONResponse({
        "score": result.score.score,
        "grade": result.score.grade,
        "headline": result.score.headline,
        "advice": result.score.advice,
        "workers": result.score.workers,
        "compliant_helmets": result.score.compliant_helmets,
        "compliant_vests": result.score.compliant_vests,
        "violation_counts": result.score.counts,
        "detections": result.detections_count,
        "machinery": result.machinery_count,
        "inference_ms": result.inference_ms,
        "tiles_used": result.tiles_used,
        "image_shape": list(result.image_shape),
        "annotated_jpeg_b64": b64,
    })


@app.post("/zones")
async def set_zones(payload: dict):
    zones_in = payload.get("zones", [])
    zones = []
    for z in zones_in:
        poly = z.get("polygon")
        if not poly or len(poly) < 3:
            raise HTTPException(400, "Each zone needs a polygon with >= 3 points.")
        zones.append(Zone(name=str(z.get("name", "Zone")),
                          polygon=[(float(p[0]), float(p[1])) for p in poly],
                          kind=str(z.get("kind", "restricted"))))
    _save_zones(zones)
    return {"saved": len(zones)}


@app.get("/zones")
def get_zones():
    return {"zones": [{"name": z.name, "polygon": z.polygon, "kind": z.kind}
                      for z in _load_zones()]}


@app.get("/progress")
def get_progress():
    rep = compute_progress(_milestones)
    return {
        "percent": rep.percent,
        "confirmed": rep.confirmed,
        "note": rep.note,
        "milestones": [{"stage": m.stage, "label": m.label, "weight": m.weight,
                        "done": m.done, "confirmed_by": m.confirmed_by}
                       for m in rep.milestones],
    }


@app.post("/progress/confirm")
async def confirm_stage(payload: dict):
    stage = payload.get("stage")
    supervisor = payload.get("supervisor")
    if not stage or not supervisor:
        raise HTTPException(400, "Both 'stage' and 'supervisor' are required.")
    if stage not in config.STAGE_WEIGHTS:
        raise HTTPException(400, f"Unknown stage '{stage}'.")
    global _milestones
    _milestones = confirm(_milestones, stage, str(supervisor),
                          done=bool(payload.get("done", True)))
    return get_progress()


@app.post("/report")
async def make_report(payload: dict):
    site = str(payload.get("site_name", "Site"))
    if not _history:
        raise HTTPException(400, "No inspections recorded yet - upload a photo first.")
    scores = [h["score"] for h in _history]
    tally: Dict[str, int] = {}
    for h in _history:
        for k, v in h["violations"].items():
            tally[k] = tally.get(k, 0) + v
    top = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
    top_issues = [f"{k.replace('_',' ')}: {v}" for k, v in top[:4]] or ["None recorded"]

    rep = compute_progress(_milestones)
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    summary = WeekSummary(
        site_name=site,
        period_start=start.strftime("%Y-%m-%d"),
        period_end=end.strftime("%Y-%m-%d"),
        inspections=len(_history),
        average_score=sum(scores) / len(scores),
        worst_score=min(scores),
        total_violations=sum(tally.values()),
        top_issues=top_issues,
        progress_percent=rep.percent,
        progress_confirmed=rep.confirmed,
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = config.OUTPUT_DIR / f"weekly_{end.strftime('%Y%m%d_%H%M%S')}.pdf"
    build_pdf(summary, pdf_path)
    texts = build_texts(summary)
    return {"pdf_path": str(pdf_path), "whatsapp": texts["whatsapp"],
            "email": texts["email"]}


@app.get("/report/latest/pdf")
def latest_pdf():
    pdfs = sorted(config.OUTPUT_DIR.glob("weekly_*.pdf")) if config.OUTPUT_DIR.exists() else []
    if not pdfs:
        raise HTTPException(404, "No report generated yet.")
    return FileResponse(str(pdfs[-1]), media_type="application/pdf",
                        filename=pdfs[-1].name)


# ---------------------------------------------------------------- video jobs ----
import threading
import uuid

_JOBS: Dict[str, dict] = {}
_JOB_LOCK = threading.Lock()
MAX_VIDEO_MB = 500


def _run_video_job(job_id: str, src: Path, work: Path, blur: bool, fast: bool,
                   sample_fps: float, max_seconds: Optional[float]):
    def progress(frac: float, msg: str):
        with _JOB_LOCK:
            _JOBS[job_id].update(progress=round(frac, 3), message=msg)

    try:
        rep = analyze_video(src, work, zones=_load_zones(), blur=blur, fast=fast,
                            sample_fps=sample_fps, max_seconds=max_seconds, progress=progress)
        _history.append({"at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                         "score": round(rep.average_score),
                         "violations": {k: len([i for i in rep.incidents if k in i.types])
                                        for k in rep.violation_seconds}})
        with _JOB_LOCK:
            _JOBS[job_id].update(status="done", progress=1.0, message="Done", report=rep.to_dict())
    except Exception as e:  # noqa: BLE001 - report any failure to the caller
        with _JOB_LOCK:
            _JOBS[job_id].update(status="failed", message=str(e))


@app.post("/inspect/video")
async def inspect_video(file: UploadFile = File(...), blur: bool = Form(True), fast: bool = Form(True),
                        sample_fps: float = Form(2.0), max_seconds: Optional[float] = Form(180.0)):
    """Start analysing a video. Poll GET /jobs/{job_id} until status is 'done'."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported video type '{suffix}'. Use one of {', '.join(VIDEO_EXTENSIONS)}.")
    if not 0.1 <= sample_fps <= 10:
        raise HTTPException(400, "sample_fps must be between 0.1 and 10.")
    job_id = uuid.uuid4().hex[:12]
    work = config.OUTPUT_DIR / "videos" / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"upload{suffix}"
    size = 0
    with open(src, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_MB * 1024 * 1024:
                fh.close()
                src.unlink(missing_ok=True)
                raise HTTPException(413, f"Video is larger than {MAX_VIDEO_MB} MB.")
            fh.write(chunk)
    with _JOB_LOCK:
        _JOBS[job_id] = {"job_id": job_id, "status": "running", "progress": 0.0,
                         "message": "Queued", "file": file.filename, "report": None}
    threading.Thread(target=_run_video_job, daemon=True,
                     args=(job_id, src, work, blur, fast, sample_fps, max_seconds)).start()
    return {"job_id": job_id, "status": "running", "poll": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job id.")
        return dict(job)


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Video not ready.")
    path = Path(job["report"]["output_video"])
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)
