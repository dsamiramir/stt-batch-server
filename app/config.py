"""Runtime configuration, loaded from environment / .env with the STT_ prefix."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="STT_",
        extra="ignore",
        # The dashboard writes to this object at runtime; validate those writes.
        validate_assignment=True,
    )

    # --- storage -----------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"

    # --- http --------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8760
    api_key: str = ""
    max_upload_mb: int = 4096
    # Answer a resubmission of the same client_ref with the job that already
    # covers it (queued, running or done) instead of transcribing the audio
    # again. Failed and canceled jobs never block a retry, and a client can
    # still insist on a fresh job with force_new=true.
    dedupe_client_ref: bool = True

    # --- model -------------------------------------------------------------
    model_size: str = "large-v3"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"  # auto | float16 | int8_float16 | int8 | ...
    cpu_threads: int = 0
    model_download_root: str = ""
    # Free VRAM by dropping the model when the queue has been empty this long.
    model_idle_unload_sec: int = 900

    # --- transcription defaults (overridable per job) ----------------------
    language: str = "ko"  # "" means auto-detect
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    condition_on_previous_text: bool = False
    initial_prompt: str = ""
    temperature: float = 0.0

    # --- multi-channel handling -------------------------------------------
    max_channels: int = 8
    # Comma-separated, positional: "agent,customer" -> ch0=agent, ch1=customer
    channel_names: str = ""
    skip_silent_channels: bool = True
    silence_rms_threshold: float = 1e-4

    # --- off-peak scheduling ----------------------------------------------
    schedule_enabled: bool = True
    # ";"-separated entries: "<days> <HH:MM>-<HH:MM>". End <= start wraps past
    # midnight; "24:00" means end of day. Days accept Mon-Fri / Sat,Sun / *.
    windows: str = "Mon-Fri 22:00-08:00; Sat 00:00-24:00; Sun 00:00-24:00"

    gpu_idle_check: bool = True
    gpu_index: int = 0
    gpu_max_util: int = 25  # percent
    gpu_min_free_mb: int = 6000
    gpu_probe_samples: int = 3
    gpu_probe_interval_sec: float = 1.0

    # --- retention ---------------------------------------------------------
    # The upload is the bulk of the disk cost and the sensitive half of a job;
    # the transcript is the part worth keeping. So a job that produced one has
    # no further use for its audio, and a job that never will is not going to
    # be rescued days later - two clocks, one sweep.
    delete_audio_on_success: bool = True
    # How long audio survives for jobs that never finished (queued, failed,
    # canceled). 0 keeps it forever.
    audio_retention_days: float = 3.0
    # How often the worker runs the sweep. 0 turns the periodic sweep off,
    # leaving only the delete-on-success path.
    cleanup_interval_sec: int = 3600

    # --- worker ------------------------------------------------------------
    poll_interval_sec: float = 5.0
    max_attempts: int = 2
    callback_timeout_sec: float = 15.0
    callback_retries: int = 3
    # Keep transcribing a job that started inside the window even after it
    # closes. Turning this off requeues the job at the next channel boundary.
    finish_job_outside_window: bool = True

    @field_validator("data_dir", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else (BASE_DIR / v).resolve()

    # --- derived -----------------------------------------------------------
    @property
    def channel_name_list(self) -> list[str]:
        return [p.strip() for p in self.channel_names.split(",") if p.strip()]

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def result_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
