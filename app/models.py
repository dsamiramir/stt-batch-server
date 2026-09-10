"""Request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TranscribeOptions(BaseModel):
    """Per-job overrides for the server-wide transcription defaults."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = Field(
        default=None, description="ISO code such as 'ko'; null or '' auto-detects."
    )
    beam_size: int | None = Field(default=None, ge=1, le=10)
    vad_filter: bool | None = None
    word_timestamps: bool | None = None
    condition_on_previous_text: bool | None = None
    initial_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    channel_names: list[str] | None = Field(
        default=None, description="Positional labels, e.g. ['agent', 'customer']."
    )

    def as_overrides(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


# Filled in by the endpoint rather than read straight off the DB row.
_DERIVED_FIELDS = ("result_url", "queue_position", "eta_seconds", "reused")


class JobOut(BaseModel):
    id: str
    status: str
    priority: int = 0
    filename: str
    client_ref: str | None = None
    channels: int | None = None
    duration_sec: float | None = None
    progress: float = 0.0
    stage: str | None = None
    error: str | None = None
    attempts: int = 0
    rtf: float | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    audio_deleted_at: str | None = Field(
        default=None,
        description="When the source upload was reclaimed. Set once the "
                    "transcript exists, or once retention expired it.",
    )
    result_url: str | None = None
    queue_position: int | None = Field(
        default=None, description="1-based place in line; null unless queued."
    )
    eta_seconds: int | None = Field(
        default=None,
        description="Rough seconds until this job finishes, from the recent "
                    "average. Null when there is nothing to estimate from.",
    )
    reused: bool = Field(
        default=False,
        description="True when a submission was answered with an existing job "
                    "for the same client_ref instead of queuing a new one.",
    )

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        queue_position: int | None = None,
        eta_seconds: int | None = None,
        reused: bool = False,
    ) -> "JobOut":
        data = {k: row.get(k) for k in cls.model_fields if k not in _DERIVED_FIELDS}
        data["result_url"] = (
            f"/v1/jobs/{row['id']}/result" if row.get("result_path") else None
        )
        data["queue_position"] = queue_position
        data["eta_seconds"] = eta_seconds
        data["reused"] = reused
        return cls(**data)


class JobList(BaseModel):
    total_by_status: dict[str, int]
    jobs: list[JobOut]
    backlog_eta_seconds: int | None = Field(
        default=None, description="Rough seconds to drain the whole queue."
    )


class GateStatus(BaseModel):
    accepting_work: bool
    reason: str
    schedule_enabled: bool
    windows: list[str]
    window_open: bool
    window_closes_at: str | None = None
    next_window_at: str | None = None
    override_until: str | None = None
    paused: bool = False


class HealthOut(BaseModel):
    status: str
    version: str
    gate: GateStatus
    gpu: dict[str, Any] | None = None
    model: dict[str, Any]
    queue: dict[str, int]
    backlog_eta_seconds: int | None = None
    current_job: str | None = None


# --- runtime settings ------------------------------------------------------
# Only fields that can be changed without restarting appear here. Model size,
# device, port and api key are deliberately excluded: they are read once at
# startup, so letting the dashboard edit them would show a lie.
LIVE_FIELDS = (
    "schedule_enabled",
    "windows",
    "gpu_idle_check",
    "gpu_max_util",
    "gpu_min_free_mb",
    "gpu_probe_samples",
    "gpu_probe_interval_sec",
    "poll_interval_sec",
    "finish_job_outside_window",
    "delete_audio_on_success",
    "audio_retention_days",
    "cleanup_interval_sec",
)


class SettingsPatch(BaseModel):
    """Partial update. Omitted fields keep their current value."""

    model_config = ConfigDict(extra="forbid")

    schedule_enabled: bool | None = Field(
        default=None, description="Off means the time window is ignored entirely."
    )
    windows: str | None = Field(
        default=None,
        description="';'-separated '<days> <HH:MM>-<HH:MM>' entries.",
        max_length=2000,
    )
    gpu_idle_check: bool | None = Field(
        default=None, description="Off means jobs start regardless of GPU load."
    )
    gpu_max_util: int | None = Field(default=None, ge=0, le=100)
    gpu_min_free_mb: int | None = Field(default=None, ge=0, le=1_048_576)
    gpu_probe_samples: int | None = Field(default=None, ge=1, le=10)
    gpu_probe_interval_sec: float | None = Field(default=None, ge=0.1, le=10.0)
    poll_interval_sec: float | None = Field(default=None, ge=1.0, le=300.0)
    finish_job_outside_window: bool | None = None
    delete_audio_on_success: bool | None = Field(
        default=None, description="Drop the upload as soon as its transcript exists."
    )
    audio_retention_days: float | None = Field(
        default=None, ge=0.0, le=3650.0,
        description="How long audio survives for jobs that never finished. 0 keeps it.",
    )
    cleanup_interval_sec: int | None = Field(
        default=None, ge=0, le=86_400,
        description="How often the sweep runs. 0 turns the periodic sweep off.",
    )

    def changes(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SettingsOut(BaseModel):
    """Effective settings, plus the read-only values needed to explain them."""

    schedule_enabled: bool
    windows: str
    gpu_idle_check: bool
    gpu_max_util: int
    gpu_min_free_mb: int
    gpu_probe_samples: int
    gpu_probe_interval_sec: float
    poll_interval_sec: float
    finish_job_outside_window: bool
    delete_audio_on_success: bool
    audio_retention_days: float
    cleanup_interval_sec: int

    # read-only, shown for context
    model_size: str
    gpu_index: int
    gpu_total_mb: int | None = None
    restart_required_for: list[str] = Field(default_factory=list)
    env_file: str = ""
    gate: GateStatus | None = None


class CleanupOut(BaseModel):
    """What a retention sweep removed."""

    deleted: int = Field(description="Upload files removed, all reasons together.")
    transcribed: int = Field(description="Removed because their transcript exists.")
    expired: int = Field(description="Removed because retention ran out.")
    orphans: int = Field(description="Files on disk that no job row pointed at.")
    failed_jobs: int = Field(
        description="Jobs marked failed because their audio expired before they ran."
    )
    freed_bytes: int
