from __future__ import annotations

import os
import tempfile
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# The settings object is built at import time, so the environment has to be in
# place before anything under app/ is imported. Point the server at a scratch
# data dir and at an off-peak window that deliberately excludes "now", so the
# worker thread never claims a job during the API tests.
_start = (datetime.now() + timedelta(hours=2)).strftime("%H:%M")
_end = (datetime.now() + timedelta(hours=3)).strftime("%H:%M")
_TMP = Path(tempfile.mkdtemp(prefix="stt-test-"))

os.environ.update(
    STT_DATA_DIR=str(_TMP),
    STT_WINDOWS=f"* {_start}-{_end}",
    STT_SCHEDULE_ENABLED="true",
    STT_GPU_IDLE_CHECK="false",
    STT_API_KEY="",
    STT_POLL_INTERVAL_SEC="1",
)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return _TMP


def write_wav(path: Path, channels: list[np.ndarray], sample_rate: int = 16000) -> Path:
    stacked = np.stack(channels, axis=1)
    pcm = np.clip(stacked, -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(len(channels))
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    sample_rate = 16000
    t = np.linspace(0, 2.0, sample_rate * 2, endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = np.zeros_like(t)  # silent second channel
    return write_wav(tmp_path / "stereo.wav", [left, right], sample_rate)
