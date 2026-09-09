"""Audio decoding: any container/codec PyAV can read -> one float32 mono track
per source channel, resampled to 16 kHz. No external ffmpeg binary required."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

TARGET_SR = 16_000


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    channels: int
    source_sample_rate: int
    duration_sec: float
    codec: str
    container: str


def probe(path: Path) -> AudioInfo:
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioError(f"no audio stream in {path.name}")
            stream = container.streams.audio[0]
            duration = 0.0
            if stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = container.duration / av.time_base
            return AudioInfo(
                channels=_channel_count(stream),
                source_sample_rate=int(stream.rate or 0),
                duration_sec=round(duration, 3),
                codec=stream.codec_context.name,
                container=container.format.name,
            )
    except (av.FFmpegError, OSError) as exc:
        raise AudioError(f"cannot open {path.name}: {exc}") from exc


def _channel_count(stream) -> int:
    layout = getattr(stream, "layout", None)
    if layout is not None and getattr(layout, "nb_channels", 0):
        return int(layout.nb_channels)
    return int(getattr(stream, "channels", 0) or 1)


def load_channels(
    path: Path, max_channels: int = 8, target_sr: int = TARGET_SR
) -> tuple[list[np.ndarray], AudioInfo]:
    """Decode ``path`` into a list of mono float32 arrays, one per channel."""
    info = probe(path)
    n_channels = min(info.channels, max_channels)
    if n_channels < 1:
        raise AudioError(f"{path.name} reports {info.channels} channels")

    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            resampler = _make_resampler(stream, target_sr)
            buckets: list[list[np.ndarray]] = [[] for _ in range(info.channels)]

            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    _append(buckets, out.to_ndarray(), info.channels)
            for out in resampler.resample(None):  # flush
                _append(buckets, out.to_ndarray(), info.channels)
    except (av.FFmpegError, OSError) as exc:
        raise AudioError(f"decode failed for {path.name}: {exc}") from exc

    channels: list[np.ndarray] = []
    for chunks in buckets[:n_channels]:
        if chunks:
            channels.append(np.concatenate(chunks).astype(np.float32, copy=False))
        else:
            channels.append(np.zeros(0, dtype=np.float32))

    if all(c.size == 0 for c in channels):
        raise AudioError(f"decoded no samples from {path.name}")

    if info.duration_sec <= 0:
        measured = max(c.size for c in channels) / float(target_sr)
        info = AudioInfo(
            channels=info.channels,
            source_sample_rate=info.source_sample_rate,
            duration_sec=round(measured, 3),
            codec=info.codec,
            container=info.container,
        )
    return channels, info


def _make_resampler(stream, target_sr: int):
    """Planar float output keeps each source channel in its own numpy row."""
    layout = getattr(stream, "layout", None)
    for candidate in (layout, _channel_count(stream)):
        if candidate is None:
            continue
        try:
            return av.AudioResampler(format="fltp", layout=candidate, rate=target_sr)
        except (ValueError, av.FFmpegError, TypeError):
            continue
    raise AudioError("could not build a resampler for this stream layout")


def _append(buckets: list[list[np.ndarray]], arr: np.ndarray, n_channels: int) -> None:
    # Planar float frames arrive as (channels, samples); guard against a packed
    # (1, channels*samples) frame in case the resampler ignored our format.
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 1 and n_channels > 1 and arr.shape[1] % n_channels == 0:
        arr = arr.reshape(-1, n_channels).T
    for ch in range(min(arr.shape[0], len(buckets))):
        buckets[ch].append(arr[ch])


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
