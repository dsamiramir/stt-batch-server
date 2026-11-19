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
    # still insist on a fresh job with force=true.
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
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    # 모델은 이 프롬프트의 문체를 모방한다. 구두점이 든 짧은 한 줄이 마침표와
    # 쉼표를 되살리는 가장 싼 수단이지만, 그 문장이 발화로 베껴지기도 한다.
    # 켜기 전에 scripts/transcript_metrics.py로 A/B를 돌려 볼 것.
    initial_prompt: str = ""
    # 실제로 자꾸 오인식되는 고유명사만 넣는다. 이 목록은 디코더의 "직전 텍스트"
    # 자리(initial_prompt와 같은 슬롯)에 들어가므로 길어질수록 그대로 발화로
    # 베껴진다. 잡이 보낸 목록과 합집합으로 쓰이니 중복은 적어도 된다.
    hotwords: str = "당근알바"
    # 단어 시각이 켜져 있을 때, 이 초보다 긴 무음에 걸친 루프를 버린다.
    hallucination_silence_threshold: float = 2.0
    temperature: float = 0.0

    # --- VAD (faster-whisper가 무음을 걷어낼 때 쓰는 값) -------------------
    # 이 값을 내리면 한국어 대화가 잘게 쪼개지고, 조각마다 문맥이 끊겨
    # 구두점이 사라진다. faster-whisper 기본값과 같은 2000을 쓴다.
    vad_min_silence_ms: int = 2000
    vad_speech_pad_ms: int = 400
    # 한 조각의 상한. Whisper의 디코딩 윈도와 같은 30초로 둔다.
    vad_max_speech_sec: float = 30.0

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
