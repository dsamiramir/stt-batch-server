"""HTTP API for the local STT batch server."""

from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from . import envfile
from . import gpu as gpu_mod
from . import schedule as sched
from . import store as store_mod
from .config import ENV_FILE, settings
from .engine import Engine
from .models import (
    LIVE_FIELDS,
    CleanupOut,
    GateStatus,
    HealthOut,
    JobList,
    JobOut,
    SettingsOut,
    SettingsPatch,
    TranscribeOptions,
)
from .schedule import WindowSpecError
from .store import JobStore
from .worker import Worker

VERSION = "0.1.0"
CHUNK = 1024 * 1024
STATIC_DIR = Path(__file__).resolve().parent / "static"
_SAFE_NAME = re.compile(r"[^0-9A-Za-z가-힣._-]+")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("stt.api")


class _State:
    store: JobStore
    engine: Engine
    worker: Worker


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    state.store = JobStore(settings.db_path)
    recovered = state.store.requeue_orphans()
    if recovered:
        log.info("requeued %d job(s) left running by a previous run", recovered)
    state.engine = Engine(settings)
    state.worker = Worker(settings, state.store, state.engine)
    state.worker.start()
    try:
        yield
    finally:
        state.worker.shutdown()
        state.worker.join(timeout=30)
        state.engine.unload()
        state.store.close()


app = FastAPI(
    title="Local STT Batch Server",
    version=VERSION,
    summary="Multi-channel speech-to-text on local GPU, scheduled for off-peak hours.",
    lifespan=lifespan,
)


# --- auth ------------------------------------------------------------------
async def require_api_key(request: Request) -> None:
    if not settings.api_key:
        return
    supplied = request.headers.get("x-api-key")
    if not supplied:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if supplied != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


Auth = Depends(require_api_key)


# --- helpers ---------------------------------------------------------------
def _safe_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", Path(name or "audio").name).strip("._")
    return cleaned[:120] or "audio"


def _eta_seconds(
    row: dict[str, Any], position: int | None, average: float | None
) -> int | None:
    """Rough finish estimate. Assumes the worker keeps running, so it does not
    include time spent waiting for an off-peak window or for the GPU to free up.
    """
    if average is None:
        return None
    if row["status"] == store_mod.QUEUED and position:
        return int(position * average)
    if row["status"] == store_mod.RUNNING:
        remaining = max(0.0, 1.0 - float(row.get("progress") or 0.0))
        return int(remaining * average)
    return None


def _job_view(row: dict[str, Any], *, reused: bool = False) -> JobOut:
    """One job with its live place in line, as the single-job endpoints show it."""
    position = None
    if row["status"] == store_mod.QUEUED:
        position = state.store.queue_positions().get(row["id"])
    average = state.store.average_job_seconds()
    return JobOut.from_row(
        row,
        queue_position=position,
        eta_seconds=_eta_seconds(row, position, average),
        reused=reused,
    )


def _reuse(row: dict[str, Any], response: Response) -> JobOut:
    """Answer a resubmission with the job that already covers it."""
    response.status_code = 200
    log.info(
        "duplicate submission for client_ref=%s -> existing job %s (%s)",
        row.get("client_ref"), row["id"], row["status"],
    )
    return _job_view(row, reused=True)


def _get_job_or_404(job_id: str) -> dict[str, Any]:
    job = state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return job


# --- endpoints -------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Status dashboard. The page itself is public; its data calls carry the key."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/v1/health", response_model=HealthOut, tags=["status"])
def health() -> HealthOut:
    stat = gpu_mod.get_gpu(settings.gpu_index)
    average = state.store.average_job_seconds()
    waiting = state.store.counts().get(store_mod.QUEUED, 0)
    backlog = int(waiting * average) if average and waiting else None
    return HealthOut(
        status="ok",
        version=VERSION,
        gate=GateStatus(**state.worker.gate_status()),
        gpu=(
            {
                "index": stat.index,
                "utilization_pct": stat.util_pct,
                "memory_used_mb": stat.used_mb,
                "memory_free_mb": stat.free_mb,
                "memory_total_mb": stat.total_mb,
            }
            if stat
            else None
        ),
        model=state.engine.describe(),
        queue=state.store.counts(),
        backlog_eta_seconds=backlog,
        current_job=state.worker.current_job_id,
    )


@app.post(
    "/v1/jobs",
    response_model=JobOut,
    status_code=202,
    tags=["jobs"],
    responses={200: {
        "model": JobOut,
        "description": "Resubmission: a job for this client_ref is already "
                       "queued, running or done, and is returned as-is.",
    }},
)
async def submit_job(
    response: Response,
    file: UploadFile = File(..., description="Audio file; multi-channel is fine."),
    options: str = Form("{}", description="JSON object of TranscribeOptions."),
    callback_url: str = Form("", description="POSTed a JSON summary when finished."),
    client_ref: str = Form("", description="Your own id, echoed back everywhere."),
    priority: int = Form(0, description="Higher runs first."),
    force_new: bool = Form(
        False,
        description="Queue a fresh job even when one for this client_ref is "
                    "already queued, running or done.",
    ),
    _: None = Auth,
) -> JobOut:
    """Queue audio for transcription.

    Submitting a ``client_ref`` that already has a queued, running or done job
    returns that job with HTTP 200 instead of transcribing the same audio again.
    """
    try:
        parsed = TranscribeOptions.model_validate(json.loads(options or "{}"))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"bad options: {exc}") from exc

    dedupe = settings.dedupe_client_ref and bool(client_ref) and not force_new
    if dedupe:
        # Cheap pre-check, so a resubmission costs no disk write at all. The
        # authoritative check is the atomic one once the upload has landed.
        existing = state.store.find_reusable(client_ref)
        if existing is not None:
            await file.close()
            return _reuse(existing, response)

    limit = settings.max_upload_mb * 1024 * 1024
    stem = _safe_name(file.filename or "audio")
    dest = settings.upload_dir / f"{uuid.uuid4().hex}_{stem}"

    written = 0
    try:
        with dest.open("wb") as sink:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds STT_MAX_UPLOAD_MB={settings.max_upload_mb}",
                    )
                sink.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="uploaded file is empty")

    fields: dict[str, Any] = dict(
        filename=file.filename or stem,
        audio_path=dest,
        options=parsed.as_overrides(),
        callback_url=callback_url or None,
        client_ref=client_ref or None,
        priority=priority,
    )
    if dedupe:
        job, created = state.store.create_or_reuse(**fields)
        if not created:
            # Lost a race to an identical submission; theirs stands.
            dest.unlink(missing_ok=True)
            return _reuse(job, response)
    else:
        job = state.store.create(**fields)
    state.worker.nudge()
    log.info("queued job %s (%s, %.1f MB)", job["id"], job["filename"], written / 1e6)
    return JobOut.from_row(job)


@app.get("/v1/jobs", response_model=JobList, tags=["jobs"])
async def list_jobs(
    status: str | None = Query(None, description="queued|running|done|failed|canceled"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    client_ref: str | None = Query(
        None, description="Only jobs submitted with this client_ref."
    ),
    _: None = Auth,
) -> JobList:
    rows = state.store.list(
        status=status, limit=limit, offset=offset, client_ref=client_ref
    )
    positions = state.store.queue_positions()
    average = state.store.average_job_seconds()
    backlog = int(len(positions) * average) if average and positions else None
    return JobList(
        total_by_status=state.store.counts(),
        jobs=[
            JobOut.from_row(
                r,
                queue_position=positions.get(r["id"]),
                eta_seconds=_eta_seconds(r, positions.get(r["id"]), average),
            )
            for r in rows
        ],
        backlog_eta_seconds=backlog,
    )


@app.get("/v1/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
async def get_job(job_id: str, _: None = Auth) -> JobOut:
    return _job_view(_get_job_or_404(job_id))


@app.get("/v1/jobs/{job_id}/result", tags=["jobs"])
async def get_result(job_id: str, _: None = Auth):
    """The transcript. It outlives the recording it came from - the upload is
    reclaimed once this exists."""
    job = _get_job_or_404(job_id)
    if job["status"] != store_mod.DONE or not job.get("result_path"):
        raise HTTPException(
            status_code=409,
            detail=f"job is {job['status']}; result not available yet",
        )
    path = Path(job["result_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="result file was removed")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.delete("/v1/jobs/{job_id}", tags=["jobs"])
async def cancel_job(job_id: str, _: None = Auth) -> JSONResponse:
    _get_job_or_404(job_id)
    status = state.store.request_cancel(job_id)
    return JSONResponse({"job_id": job_id, "status": status})


def _settings_payload() -> SettingsOut:
    stat = gpu_mod.get_gpu(settings.gpu_index)
    return SettingsOut(
        **{name: getattr(settings, name) for name in LIVE_FIELDS},
        model_size=settings.model_size,
        gpu_index=settings.gpu_index,
        gpu_total_mb=stat.total_mb if stat else None,
        restart_required_for=[
            "model_size", "device", "compute_type", "port", "api_key", "data_dir",
        ],
        env_file=str(ENV_FILE),
        gate=GateStatus(**state.worker.gate_status()),
    )


@app.get("/v1/settings", response_model=SettingsOut, tags=["settings"])
def get_settings(_: None = Auth) -> SettingsOut:
    """Settings that can be changed while the server runs."""
    return _settings_payload()


@app.put("/v1/settings", response_model=SettingsOut, tags=["settings"])
def put_settings(body: SettingsPatch, request: Request, _: None = Auth) -> SettingsOut:
    """Apply a partial update, then persist it to the .env file.

    Changes take effect immediately; the file write is what makes them survive a
    restart. A bad window spec is rejected before anything is touched.
    """
    changes = body.changes()
    if not changes:
        return _settings_payload()

    if "windows" in changes:
        try:
            sched.parse_windows(changes["windows"])
        except WindowSpecError as exc:
            raise HTTPException(status_code=422, detail=f"bad window spec: {exc}") from exc

    previous = {name: getattr(settings, name) for name in changes}
    try:
        for name, value in changes.items():
            setattr(settings, name, value)
        state.worker.apply_settings()
    except Exception as exc:  # noqa: BLE001 - never leave settings half-applied
        for name, value in previous.items():
            setattr(settings, name, value)
        state.worker.apply_settings()
        raise HTTPException(status_code=422, detail=f"rejected: {exc}") from exc

    envfile.update_env_file(
        ENV_FILE, {f"STT_{name.upper()}": value for name, value in changes.items()}
    )
    who = request.client.host if request.client else "unknown"
    log.info("settings updated by %s: %s", who, changes)
    return _settings_payload()


@app.post("/v1/maintenance/cleanup", response_model=CleanupOut, tags=["status"])
async def cleanup(request: Request, _: None = Auth) -> CleanupOut:
    """Run the audio retention sweep now instead of waiting for its interval.

    Same pass the worker runs on a timer: uploads whose transcript exists,
    uploads past ``STT_AUDIO_RETENTION_DAYS``, and stray files no job claims.
    """
    report = await run_in_threadpool(state.worker.clean_now)
    who = request.client.host if request.client else "unknown"
    log.info("manual cleanup by %s removed %s", who, report.summary())
    return CleanupOut(**report.as_dict())


class ControlIn(BaseModel):
    pause: bool | None = None
    override_minutes: int | None = None


@app.post("/v1/control", response_model=GateStatus, tags=["status"])
async def control(body: ControlIn, request: Request, _: None = Auth) -> GateStatus:
    """Pause/resume the worker, or ignore the schedule for a while.

    ``override_minutes: 0`` clears an override and restores normal scheduling.
    """
    who = request.client.host if request.client else "unknown"
    if body.pause is not None:
        state.worker.paused = body.pause
        state.worker.nudge()
        log.info("worker %s by %s", "paused" if body.pause else "resumed", who)
    if body.override_minutes is not None:
        until = state.worker.set_override(body.override_minutes)
        log.info(
            "schedule override %s by %s",
            f"set until {until:%Y-%m-%d %H:%M}" if until else "cleared",
            who,
        )
    return GateStatus(**state.worker.gate_status())


def run() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    run()
