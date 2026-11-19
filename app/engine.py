"""faster-whisper wrapper: lazy model load, CPU fallback, cancellable decode."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from .config import Settings
from .cuda_runtime import configure_cuda_runtime
from .quality import (
    echo_phrases,
    merge_hotwords,
    prepare_segments,
    usable_initial_prompt,
)

log = logging.getLogger("stt.engine")


class Cancelled(Exception):
    """Raised inside a decode loop when the caller asked it to stop."""


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None
        self._lock = threading.RLock()
        self._last_used = 0.0
        self._info: dict[str, Any] = {}
        self._load_error: str | None = None

    # --- lifecycle ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        return {
            "model_size": self.settings.model_size,
            "loaded": self.is_loaded,
            "last_used_ago_sec": (
                round(time.monotonic() - self._last_used, 1) if self._last_used else None
            ),
            "load_error": self._load_error,
            **self._info,
        }

    def ensure_loaded(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model

            configure_cuda_runtime()
            from faster_whisper import WhisperModel

            device = resolve_device(self.settings.device)
            compute_type = resolve_compute_type(self.settings.compute_type, device)
            kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
            if self.settings.cpu_threads > 0:
                kwargs["cpu_threads"] = self.settings.cpu_threads
            if self.settings.model_download_root:
                kwargs["download_root"] = self.settings.model_download_root

            started = time.monotonic()
            try:
                model = WhisperModel(self.settings.model_size, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if device != "cuda":
                    self._load_error = str(exc)
                    raise
                log.warning("CUDA model load failed (%s); falling back to CPU", exc)
                device, compute_type = "cpu", "int8"
                kwargs.update(device=device, compute_type=compute_type)
                model = WhisperModel(self.settings.model_size, **kwargs)
                self._load_error = f"CUDA unavailable, running on CPU: {exc}"
            else:
                self._load_error = None

            self._model = model
            self._info = {
                "device": device,
                "compute_type": compute_type,
                "load_seconds": round(time.monotonic() - started, 2),
            }
            self._last_used = time.monotonic()
            log.info(
                "loaded %s on %s/%s in %.1fs",
                self.settings.model_size, device, compute_type,
                self._info["load_seconds"],
            )
            return self._model

    def unload(self) -> bool:
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            self._info = {}
            import gc

            gc.collect()
            log.info("model unloaded, VRAM released")
            return True

    def maybe_unload_idle(self) -> bool:
        timeout = self.settings.model_idle_unload_sec
        if timeout <= 0 or not self.is_loaded or not self._last_used:
            return False
        if time.monotonic() - self._last_used < timeout:
            return False
        return self.unload()

    # --- inference ---------------------------------------------------------
    def transcribe(
        self,
        audio: np.ndarray,
        opts: dict[str, Any],
        on_segment: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Decode one channel in a single call, then clean up what comes back.

        One call, not one per speech slice. faster-whisper strips the silence
        itself, decodes the remaining speech in 30s windows - so a sentence is
        punctuated with its own context in view - and adds the removed silence
        back into the timestamps on the way out.
        """
        model = self.ensure_loaded()
        decode = self._decode_options(opts)
        use_vad = bool(opts.get("vad_filter", self.settings.vad_filter))

        with self._lock:
            segments_iter, info = model.transcribe(
                audio,
                vad_filter=use_vad,
                vad_parameters=self._vad_parameters(opts) if use_vad else None,
                **decode,
            )

            collected: list[dict[str, Any]] = []
            for seg in segments_iter:
                if should_stop is not None and should_stop():
                    raise Cancelled()
                item = _segment_to_dict(seg)
                collected.append(item)
                # Progress only reads the segment's end, so report as they
                # arrive rather than waiting for the whole channel to finish.
                if on_segment is not None:
                    on_segment(item)

            cleaned = prepare_segments(
                collected,
                echo_terms=echo_phrases(decode["initial_prompt"], decode["hotwords"]),
            )
            self._last_used = time.monotonic()

        return cleaned, {
            "language": info.language,
            "language_probability": round(float(info.language_probability or 0.0), 4),
            "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
        }

    def _decode_options(self, opts: dict[str, Any]) -> dict[str, Any]:
        """Per-job overrides folded onto the server defaults, as decode kwargs."""
        s = self.settings
        return {
            "language": opts.get("language", s.language) or None,
            "beam_size": int(opts.get("beam_size", s.beam_size)),
            "word_timestamps": bool(opts.get("word_timestamps", s.word_timestamps)),
            "condition_on_previous_text": bool(
                opts.get("condition_on_previous_text", s.condition_on_previous_text)
            ),
            "initial_prompt": usable_initial_prompt(
                opts.get("initial_prompt", s.initial_prompt)
            ),
            "temperature": opts.get("temperature", s.temperature),
            "hotwords": merge_hotwords(s.hotwords, opts.get("hotwords")),
            "hallucination_silence_threshold": opts.get(
                "hallucination_silence_threshold", s.hallucination_silence_threshold
            ),
        }

    def _vad_parameters(self, opts: dict[str, Any]) -> Any:
        """Silero settings. Shorter silences cut Korean mid-thought, so 2s."""
        from faster_whisper.vad import VadOptions

        s = self.settings
        return VadOptions(
            min_silence_duration_ms=int(
                opts.get("vad_min_silence_ms", s.vad_min_silence_ms)
            ),
            speech_pad_ms=int(opts.get("vad_speech_pad_ms", s.vad_speech_pad_ms)),
            max_speech_duration_s=float(
                opts.get("vad_max_speech_sec", s.vad_max_speech_sec)
            ),
        )


def _segment_to_dict(seg: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "start": round(float(seg.start), 3),
        "end": round(float(seg.end), 3),
        "text": (seg.text or "").strip(),
        "avg_logprob": round(float(seg.avg_logprob), 4),
        "no_speech_prob": round(float(seg.no_speech_prob), 4),
        "compression_ratio": round(float(getattr(seg, "compression_ratio", 0.0) or 0.0), 4),
    }
    words = getattr(seg, "words", None)
    if words:
        item["words"] = [
            {
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "word": w.word,
                "probability": round(float(w.probability), 4),
            }
            for w in words
        ]
    return item
