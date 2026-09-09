"""Background worker: one job at a time so the GPU is never oversubscribed."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from . import audio as audio_mod
from . import gpu as gpu_mod
from . import retention
from . import schedule as sched
from . import store as store_mod
from .config import Settings
from .engine import Cancelled, Engine
from .store import JobStore

log = logging.getLogger("stt.worker")


class Deferred(Exception):
    """The off-peak window closed; put the job back on the queue."""


class Worker(threading.Thread):
    def __init__(self, settings: Settings, store: JobStore, engine: Engine) -> None:
        super().__init__(name="stt-worker", daemon=True)
        self.settings = settings
        self.store = store
        self.engine = engine
        self.windows = sched.parse_windows(settings.windows)

        self._stop_event = threading.Event()
        self._wake = threading.Event()
        self.paused = False
        self.override_until: datetime | None = None
        self.current_job_id: str | None = None
        self.last_gate_reason = "starting up"
        # None means "never swept", so the first pass of the loop cleans up
        # whatever the previous run left behind.
        self._last_cleanup: float | None = None

    # --- control -----------------------------------------------------------
    def shutdown(self) -> None:
        self._stop_event.set()
        self._wake.set()

    def nudge(self) -> None:
        """Wake the poll loop early, e.g. right after a job is submitted."""
        self._wake.set()

    def set_override(self, minutes: int | None) -> datetime | None:
        if not minutes or minutes <= 0:
            self.override_until = None
        else:
            self.override_until = datetime.now().astimezone() + timedelta(minutes=minutes)
        self.nudge()
        return self.override_until

    # --- gating ------------------------------------------------------------
    def _overridden(self, now: datetime) -> bool:
        return self.override_until is not None and now.astimezone() < self.override_until

    def evaluate_gate(self, now: datetime | None = None) -> tuple[bool, str]:
        """Full check, including the debounced multi-sample GPU probe.

        This sleeps between samples, so only the worker thread may call it -
        running it inside a request would block the event loop for seconds.
        """
        now = now or datetime.now()
        if self.paused:
            return False, "worker paused"

        overridden = self._overridden(now)
        if self.settings.schedule_enabled and not overridden:
            if not sched.is_open(self.windows, now):
                nxt = sched.next_open_at(self.windows, now)
                when = nxt.strftime("%Y-%m-%d %H:%M") if nxt else "never"
                return False, f"outside off-peak window; next opens {when}"

        if self.settings.gpu_idle_check and not overridden:
            ok, reason = gpu_mod.check_idle(
                index=self.settings.gpu_index,
                max_util=self.settings.gpu_max_util,
                min_free_mb=self.settings.gpu_min_free_mb,
                samples=self.settings.gpu_probe_samples,
                interval_sec=self.settings.gpu_probe_interval_sec,
                require_free_vram=not self.engine.is_loaded,
            )
            return (True, reason) if ok else (False, f"GPU busy: {reason}")

        if overridden:
            return True, "manual override active"
        if not self.settings.schedule_enabled:
            return True, "scheduling disabled; running whenever work arrives"
        return True, "inside off-peak window"

    def _gpu_glance(self) -> tuple[bool, str]:
        """One cached nvidia-smi read for display: no sampling, no sleeping."""
        stat = gpu_mod.get_gpu(self.settings.gpu_index)
        if stat is None:
            return True, "nvidia-smi unavailable; GPU gate skipped"
        if stat.util_pct > self.settings.gpu_max_util:
            return False, (
                f"GPU busy: GPU util {stat.util_pct}% > {self.settings.gpu_max_util}%"
            )
        if not self.engine.is_loaded and stat.free_mb < self.settings.gpu_min_free_mb:
            return False, (
                f"GPU busy: free VRAM {stat.free_mb}MB "
                f"< {self.settings.gpu_min_free_mb}MB"
            )
        return True, f"GPU idle (util {stat.util_pct}%, free {stat.free_mb}MB)"

    def gate_status(self) -> dict[str, Any]:
        """Cheap snapshot for the API. Mirrors evaluate_gate's decision order,
        but reads the GPU once from cache instead of probing."""
        now = datetime.now()
        overridden = self._overridden(now)

        if self.paused:
            allowed, reason = False, "worker paused"
        elif (
            self.settings.schedule_enabled
            and not overridden
            and not sched.is_open(self.windows, now)
        ):
            nxt = sched.next_open_at(self.windows, now)
            when = nxt.strftime("%Y-%m-%d %H:%M") if nxt else "never"
            allowed, reason = False, f"outside off-peak window; next opens {when}"
        elif self.settings.gpu_idle_check and not overridden:
            allowed, reason = self._gpu_glance()
        elif overridden:
            allowed, reason = True, "manual override active"
        elif not self.settings.schedule_enabled:
            allowed, reason = True, "scheduling disabled; running whenever work arrives"
        else:
            allowed, reason = True, "inside off-peak window"

        closes = sched.window_closes_at(self.windows, now)
        nxt = sched.next_open_at(self.windows, now)
        return {
            "accepting_work": allowed,
            "reason": reason,
            "schedule_enabled": self.settings.schedule_enabled,
            "windows": [w.raw for w in self.windows],
            "window_open": sched.is_open(self.windows, now),
            "window_closes_at": closes.isoformat(timespec="seconds") if closes else None,
            "next_window_at": nxt.isoformat(timespec="seconds") if nxt else None,
            "override_until": (
                self.override_until.isoformat(timespec="seconds")
                if self.override_until
                else None
            ),
            "paused": self.paused,
        }

    def apply_settings(self) -> None:
        """Pick up settings the worker cached at start-up. Raises WindowSpecError
        if the new window spec is unparseable, leaving the old one in place."""
        self.windows = sched.parse_windows(self.settings.windows)
        self.nudge()

    # --- main loop ---------------------------------------------------------
    def run(self) -> None:
        log.info("worker started; windows=%s", [w.raw for w in self.windows])
        while not self._stop_event.is_set():
            self._maybe_clean()

            # Look for work before touching the GPU. With the schedule gate off
            # this loop runs around the clock, and probing an empty queue would
            # spawn nvidia-smi every few seconds for nothing.
            if self.paused or not self.store.has_queued():
                self.last_gate_reason = "worker paused" if self.paused else "queue empty"
                self.engine.maybe_unload_idle()
                self._sleep(self.settings.poll_interval_sec)
                continue

            allowed, reason = self.evaluate_gate()
            self.last_gate_reason = reason
            if not allowed:
                self.engine.maybe_unload_idle()
                self._sleep(self.settings.poll_interval_sec)
                continue

            job = self.store.claim_next()
            if job is None:
                self._sleep(self.settings.poll_interval_sec)
                continue

            self.current_job_id = job["id"]
            try:
                self._process(job)
            except Exception:  # noqa: BLE001 - loop must survive any job
                log.exception("unhandled error on job %s", job["id"])
            finally:
                self.current_job_id = None
        log.info("worker stopped")

    def _sleep(self, seconds: float) -> None:
        self._wake.wait(timeout=seconds)
        self._wake.clear()

    # --- retention ---------------------------------------------------------
    def _maybe_clean(self) -> None:
        """Run the audio retention sweep on its own slow clock.

        It lives on the worker thread because that thread is the only one that
        may block, and because nothing else is allowed to touch an upload while
        a job is running.
        """
        interval = float(self.settings.cleanup_interval_sec)
        if interval <= 0:
            return
        now = time.monotonic()
        if self._last_cleanup is not None and now - self._last_cleanup < max(60.0, interval):
            return
        self._last_cleanup = now
        try:
            report = self.clean_now()
        except Exception:  # noqa: BLE001 - a sweep must never kill the worker
            log.exception("retention sweep failed")
            return
        if report.deleted:
            log.info("retention sweep removed %s", report.summary())

    def clean_now(self) -> retention.SweepReport:
        """Sweep on demand, for the maintenance endpoint."""
        return retention.sweep(self.settings, self.store)

    def _reclaim_audio(self, job_id: str, audio_path: str) -> None:
        if not self.settings.delete_audio_on_success:
            return
        freed = retention.reclaim_one(self.settings, self.store, job_id, audio_path)
        if freed is None:
            log.warning("job %s: audio still locked; leaving it for the sweep", job_id)
        else:
            log.info("job %s: audio removed (%.1f MB freed)", job_id, freed / 1e6)

    # --- job execution -----------------------------------------------------
    def _process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        opts: dict[str, Any] = json.loads(job["options"] or "{}")
        path = Path(job["audio_path"])
        started = time.monotonic()
        log.info("job %s start: %s", job_id, job["filename"])

        try:
            if not path.exists():
                raise FileNotFoundError(f"audio file missing: {path}")

            self.store.update(job_id, stage="decoding audio", progress=0.0)
            channels, info = audio_mod.load_channels(
                path, max_channels=self.settings.max_channels
            )
            self.store.update(
                job_id, channels=len(channels), duration_sec=info.duration_sec
            )

            names = opts.get("channel_names") or self.settings.channel_name_list
            total = len(channels)
            duration = max(info.duration_sec, 0.001)
            channel_results: list[dict[str, Any]] = []
            detected_language: str | None = None

            for index, samples in enumerate(channels):
                self._checkpoint(job_id, between_channels=True)
                name = names[index] if index < len(names) else f"ch{index}"

                if (
                    self.settings.skip_silent_channels
                    and audio_mod.rms(samples) < self.settings.silence_rms_threshold
                ):
                    log.info("job %s: channel %s is silent, skipped", job_id, name)
                    channel_results.append(
                        {"index": index, "name": name, "skipped": "silent", "segments": []}
                    )
                    self.store.update(job_id, progress=round((index + 1) / total, 4))
                    continue

                self.store.update(
                    job_id, stage=f"transcribing {name} ({index + 1}/{total})"
                )
                base = index / total

                def on_segment(seg: dict[str, Any], _base: float = base) -> None:
                    frac = min(seg["end"] / duration, 1.0)
                    self.store.update(
                        job_id, progress=round(_base + frac / total, 4)
                    )

                segments, meta = self.engine.transcribe(
                    samples,
                    opts,
                    on_segment=on_segment,
                    should_stop=lambda: self._stop_requested(job_id),
                )
                detected_language = detected_language or meta.get("language")
                channel_results.append(
                    {
                        "index": index,
                        "name": name,
                        "language": meta.get("language"),
                        "language_probability": meta.get("language_probability"),
                        "segments": segments,
                    }
                )
                self.store.update(job_id, progress=round((index + 1) / total, 4))

            elapsed = time.monotonic() - started
            result = self._build_result(
                job, info, channel_results, detected_language, elapsed
            )
            result_path = self.settings.result_dir / f"{job_id}.json"
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.store.update(
                job_id,
                status=store_mod.DONE,
                progress=1.0,
                stage=None,
                result_path=str(result_path),
                finished_at=store_mod.utcnow(),
                rtf=round(elapsed / duration, 3),
            )
            log.info(
                "job %s done in %.1fs (%.2fx realtime)",
                job_id, elapsed, duration / elapsed if elapsed else 0.0,
            )
            self._reclaim_audio(job_id, job["audio_path"])
            self._fire_callback(job_id)

        except Deferred:
            self.store.update(
                job_id,
                status=store_mod.QUEUED,
                stage="deferred until next off-peak window",
                progress=0.0,
                attempts=max(0, int(job.get("attempts") or 1) - 1),
            )
            log.info("job %s deferred; window closed", job_id)

        except Cancelled:
            if self._stop_event.is_set():
                self.store.update(
                    job_id,
                    status=store_mod.QUEUED,
                    stage="requeued on shutdown",
                    progress=0.0,
                    attempts=max(0, int(job.get("attempts") or 1) - 1),
                )
                log.info("job %s requeued for shutdown", job_id)
            else:
                self.store.update(
                    job_id,
                    status=store_mod.CANCELED,
                    stage=None,
                    finished_at=store_mod.utcnow(),
                )
                log.info("job %s canceled", job_id)
                self._fire_callback(job_id)

        except Exception as exc:  # noqa: BLE001
            attempts = int(job.get("attempts") or 1)
            message = f"{type(exc).__name__}: {exc}"
            log.exception("job %s failed (attempt %d)", job_id, attempts)
            if attempts < self.settings.max_attempts:
                self.store.update(
                    job_id, status=store_mod.QUEUED, stage="retrying", error=message,
                    progress=0.0,
                )
            else:
                self.store.update(
                    job_id,
                    status=store_mod.FAILED,
                    stage=None,
                    error=message,
                    finished_at=store_mod.utcnow(),
                )
                self._fire_callback(job_id)

    # --- helpers -----------------------------------------------------------
    def _stop_requested(self, job_id: str) -> bool:
        if self._stop_event.is_set():
            return True
        return self.store.status_of(job_id) == store_mod.CANCELING

    def _checkpoint(self, job_id: str, between_channels: bool = False) -> None:
        if self._stop_requested(job_id):
            raise Cancelled()
        if (
            between_channels
            and not self.settings.finish_job_outside_window
            and self.settings.schedule_enabled
            and not sched.is_open(self.windows, datetime.now())
            and not (
                self.override_until
                and datetime.now().astimezone() < self.override_until
            )
        ):
            raise Deferred()

    def _build_result(
        self,
        job: dict[str, Any],
        info: audio_mod.AudioInfo,
        channel_results: list[dict[str, Any]],
        language: str | None,
        elapsed: float,
    ) -> dict[str, Any]:
        merged: list[dict[str, Any]] = []
        for channel in channel_results:
            for seg in channel["segments"]:
                merged.append({"channel": channel["index"], "speaker": channel["name"], **seg})
        merged.sort(key=lambda s: (s["start"], s["channel"]))

        transcript = "\n".join(
            f"[{_ts(s['start'])}] {s['speaker']}: {s['text']}" for s in merged if s["text"]
        )
        return {
            "job_id": job["id"],
            "filename": job["filename"],
            "client_ref": job.get("client_ref"),
            "model": self.settings.model_size,
            "engine": self.engine.describe(),
            "language": language,
            "audio": {
                "channels_total": info.channels,
                "channels_transcribed": len(channel_results),
                "duration_sec": info.duration_sec,
                "source_sample_rate": info.source_sample_rate,
                "codec": info.codec,
                "container": info.container,
            },
            "processing_sec": round(elapsed, 2),
            "created_at": store_mod.utcnow(),
            "channels": channel_results,
            "segments": merged,
            "text": transcript,
        }

    def _fire_callback(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job or not job.get("callback_url"):
            return
        payload = {
            "job_id": job_id,
            "status": job["status"],
            "client_ref": job.get("client_ref"),
            "filename": job.get("filename"),
            "error": job.get("error"),
            "result_url": f"/v1/jobs/{job_id}/result" if job.get("result_path") else None,
        }
        url = job["callback_url"]
        headers = {"content-type": "application/json"}
        if self.settings.api_key:
            headers["authorization"] = f"Bearer {self.settings.api_key}"

        for attempt in range(1, self.settings.callback_retries + 1):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.settings.callback_timeout_sec,
                )
                if response.status_code < 400:
                    log.info("callback for %s delivered (%s)", job_id, response.status_code)
                    return
                log.warning(
                    "callback for %s got HTTP %s (attempt %d)",
                    job_id, response.status_code, attempt,
                )
            except httpx.HTTPError as exc:
                log.warning("callback for %s failed: %s (attempt %d)", job_id, exc, attempt)
            if attempt < self.settings.callback_retries:
                time.sleep(2 * attempt)
        log.error("callback for %s gave up after %d attempts", job_id, self.settings.callback_retries)


def _ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
