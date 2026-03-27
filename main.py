"""
Media Forensics & Authenticity Analyzer — FastAPI Backend
Run: uvicorn main:app --reload
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKey, APIKeyHeader
from pydantic import BaseModel

from analyzers.audio import analyze_audio
from analyzers.image import analyze_image
from analyzers.pdf import analyze_pdf
from analyzers.video import analyze_video
from services.classifier import classify_file
from services.explanation import build_report
from services.scoring import fuse_signals
from services.storage import cleanup_upload, save_upload


def _split_csv_env(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


API_KEY = os.getenv("DEEPFAKE_API_KEY", "drishyam_admin_2026")
API_KEY_NAME = "X-API-KEY"
MAX_SIZE_MB = int(os.getenv("MAX_SIZE_MB", "100"))
SYNC_MAX_SIZE_MB = int(os.getenv("SYNC_MAX_SIZE_MB", str(min(MAX_SIZE_MB, 30))))
JOB_RETENTION_MINUTES = int(os.getenv("JOB_RETENTION_MINUTES", "60"))
DEFAULT_PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

cors_origins_value = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
cors_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX")
allow_origins = ["*"] if cors_origins_value == "*" else _split_csv_env(cors_origins_value)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
app = FastAPI(title="Media Forensics API", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_origin_regex=cors_origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. This is adequate for a single Railway service instance.
# For multi-instance or long-lived retention, move this to Redis/PostgreSQL.
jobs: dict[str, dict] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    media_type: Optional[str] = None
    storage_url: Optional[str] = None
    metrics: Optional[dict] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    expires_at: Optional[str] = None


async def get_api_key(header_api_key: str = Security(api_key_header)):
    if header_api_key == API_KEY:
        return header_api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API Key")


def _safe_filename(upload: UploadFile) -> str:
    return Path(upload.filename or "upload.bin").name


def _cleanup_expired_jobs() -> None:
    now = _utc_now()
    expired_ids = []
    for job_id, job in jobs.items():
        expires_at = job.get("expires_at")
        if expires_at and expires_at <= now:
            expired_ids.append(job_id)

    for job_id in expired_ids:
        cleanup_upload(job_id)
        jobs.pop(job_id, None)


def _job_response(job_id: str, job: dict) -> JobStatus:
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        media_type=job.get("file_type"),
        storage_url=job.get("storage_url"),
        metrics=job.get("metrics"),
        created_at=_isoformat(job.get("created_at")),
        completed_at=_isoformat(job.get("completed_at")),
        expires_at=_isoformat(job.get("expires_at")),
    )


async def _run_analyzer(file_path: Path, file_type: str) -> dict:
    if file_type == "image":
        return await analyze_image(file_path)
    if file_type == "audio":
        return await analyze_audio(file_path)
    if file_type == "video":
        return await analyze_video(file_path)
    if file_type == "pdf":
        return await analyze_pdf(file_path)
    raise ValueError(f"Unsupported file type: {file_type}")


async def _build_pipeline_result(file_path: Path, file_type: str) -> tuple[dict, dict]:
    signals = await _run_analyzer(file_path, file_type)
    score, band, metrics = fuse_signals(signals, file_type)
    report = build_report(file_type, score, band, {**signals, "metrics": metrics})
    return report, metrics


async def _prepare_upload(
    file: UploadFile,
    *,
    size_limit_mb: int,
) -> tuple[str, str, Path, Optional[str]]:
    _cleanup_expired_jobs()

    content = await file.read()
    if len(content) > size_limit_mb * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {size_limit_mb}MB limit")

    filename = _safe_filename(file)
    suffix = Path(filename).suffix.lower()
    file_type = classify_file(suffix)
    if not file_type:
        raise HTTPException(415, f"Unsupported file type: {suffix}")

    job_id = str(uuid.uuid4())
    storage_res = await save_upload(job_id, filename, content, file.content_type)
    created_at = _utc_now()
    jobs[job_id] = {
        "status": "queued",
        "result": None,
        "metrics": None,
        "file_type": file_type,
        "storage_url": storage_res["public_url"],
        "created_at": created_at,
        "completed_at": None,
        "expires_at": None,
    }
    return job_id, file_type, storage_res["local_path"], storage_res["public_url"]


async def run_pipeline(job_id: str, file_path: Path, file_type: str):
    """Background task: run full analysis and persist result in memory."""
    jobs[job_id]["status"] = "processing"
    try:
        report, metrics = await _build_pipeline_result(file_path, file_type)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = report
        jobs[job_id]["metrics"] = metrics
        jobs[job_id]["completed_at"] = _utc_now()
        jobs[job_id]["expires_at"] = _utc_now() + timedelta(minutes=JOB_RETENTION_MINUTES)
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = {"error": str(exc)}
        jobs[job_id]["completed_at"] = _utc_now()
        jobs[job_id]["expires_at"] = _utc_now() + timedelta(minutes=JOB_RETENTION_MINUTES)
    finally:
        cleanup_upload(job_id)


@app.get("/")
async def root():
    return {
        "service": "DRISHYAM Forensics API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "sync_endpoint": "/analyze-sync",
        "async_endpoint": "/analyze",
    }


@app.post("/analyze", response_model=JobStatus)
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Queue analysis and return a job ID immediately.
    Best for larger video or slower forensic runs.
    """
    del api_key
    job_id, file_type, file_path, public_url = await _prepare_upload(file, size_limit_mb=MAX_SIZE_MB)
    background_tasks.add_task(run_pipeline, job_id, file_path, file_type)
    return _job_response(job_id, jobs[job_id])


@app.post("/analyze-sync", response_model=JobStatus)
async def analyze_sync(
    file: UploadFile = File(...),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Run analysis in-request and return the final verdict directly.
    Best for images, audio, PDFs, and smaller files from your main app.
    """
    del api_key
    job_id, file_type, file_path, _public_url = await _prepare_upload(file, size_limit_mb=SYNC_MAX_SIZE_MB)
    jobs[job_id]["status"] = "processing"

    try:
        report, metrics = await _build_pipeline_result(file_path, file_type)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = report
        jobs[job_id]["metrics"] = metrics
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = {"error": str(exc)}
    finally:
        jobs[job_id]["completed_at"] = _utc_now()
        jobs[job_id]["expires_at"] = _utc_now() + timedelta(minutes=JOB_RETENTION_MINUTES)
        cleanup_upload(job_id)

    return _job_response(job_id, jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str, api_key: APIKey = Depends(get_api_key)):
    """Poll analysis status for async jobs."""
    del api_key
    _cleanup_expired_jobs()
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return _job_response(job_id, jobs[job_id])


@app.get("/stats")
async def get_model_stats(api_key: APIKey = Depends(get_api_key)):
    """
    Return engine capabilities and deployment runtime settings.
    """
    del api_key
    from services.model_runtime import get_external_model_status
    from services.scoring import ENGINE_NAME, ENGINE_UPDATED_AT, ENGINE_VERSION, get_model_metrics

    return {
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "service_version": app.version,
        "status": "Healthy",
        "updated_at": ENGINE_UPDATED_AT,
        "calibration_status": "heuristic_ensemble",
        "deployment": {
            "max_size_mb": MAX_SIZE_MB,
            "sync_max_size_mb": SYNC_MAX_SIZE_MB,
            "job_retention_minutes": JOB_RETENTION_MINUTES,
            "public_base_url": DEFAULT_PUBLIC_BASE_URL,
        },
        "supported_modalities": {
            "image": get_model_metrics("image"),
            "audio": get_model_metrics("audio"),
            "video": get_model_metrics("video"),
            "pdf": get_model_metrics("pdf"),
        },
        "external_model_runtime": get_external_model_status(),
        "forensic_capabilities": [
            "Image metadata, pixel, cloning, resampling, and synthetic-pattern analysis",
            "Voice smoothness, spoof, and splice detection",
            "Video frame, temporal, and audio-video alignment analysis",
            "PDF/DOCX/TXT metadata, hidden-content, and text-pattern screening",
            "Optional local Transformers or Hugging Face inference-backed learned detectors",
            "Synchronous and asynchronous API workflows",
        ],
        "note": (
            "This endpoint reports runtime capability and configuration, not a third-party benchmark claim. "
            "Use labeled evaluation data to calibrate production thresholds."
        ),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "sync_max_size_mb": SYNC_MAX_SIZE_MB,
        "job_retention_minutes": JOB_RETENTION_MINUTES,
    }
