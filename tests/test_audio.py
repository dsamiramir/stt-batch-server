from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio import AudioError, load_channels, probe, rms
from tests.conftest import write_wav


def test_probe_reports_stereo(stereo_wav: Path) -> None:
    info = probe(stereo_wav)
    assert info.channels == 2
    assert info.source_sample_rate == 16000
    assert info.duration_sec == pytest.approx(2.0, abs=0.05)


def test_channels_are_kept_separate(stereo_wav: Path) -> None:
    channels, info = load_channels(stereo_wav)
    assert len(channels) == 2
    assert info.duration_sec == pytest.approx(2.0, abs=0.05)
    for channel in channels:
        assert channel.dtype == np.float32
        assert channel.size == pytest.approx(32000, abs=400)
    assert rms(channels[0]) > 0.1
    assert rms(channels[1]) < 1e-6


def test_resamples_to_16k(tmp_path: Path) -> None:
    sample_rate = 44100
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    path = write_wav(tmp_path / "44k.wav", [tone, tone], sample_rate)

    channels, info = load_channels(path)
    assert info.source_sample_rate == 44100
    assert channels[0].size == pytest.approx(16000, abs=200)


def test_four_channels(tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tracks = [0.1 * (i + 1) * np.sin(2 * np.pi * (200 + 100 * i) * t) for i in range(4)]
    path = write_wav(tmp_path / "quad.wav", tracks, sample_rate)

    channels, info = load_channels(path)
    assert info.channels == 4
    assert len(channels) == 4
    levels = [rms(c) for c in channels]
    assert levels == sorted(levels), "channel order must be preserved"


def test_max_channels_truncates(tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0, 0.5, sample_rate // 2, endpoint=False)
    tracks = [0.1 * np.sin(2 * np.pi * 300 * t) for _ in range(4)]
    path = write_wav(tmp_path / "quad2.wav", tracks, sample_rate)

    channels, _ = load_channels(path, max_channels=2)
    assert len(channels) == 2


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        probe(tmp_path / "nope.wav")


def test_non_audio_file_raises(tmp_path: Path) -> None:
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(AudioError):
        load_channels(junk)
