"""Reclaiming upload disk space once a recording has served its purpose.

Two clocks, one sweep. A job that produced a transcript has no further use for
its audio, so that upload goes as soon as the result lands. A job that never
produced one - failed, canceled, or still queued - is not going to be rescued
days later, so its audio goes once it is old enough. The transcript is small
and stays either way; the recording is the bulk of the disk cost and the
sensitive half of a call, so it is the part with an expiry date.

Deletions are idempotent and failure-tolerant: a file locked by another process
is simply left for the next sweep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import store as store_mod
from .config import Settings
from .store import JobStore

log = logging.getLogger("stt.retention")

# Cap on how much one sweep does, so a first run against a long-neglected
# directory cannot hold the worker loop for minutes.
BATCH = 500


@dataclass
class SweepReport:
    """What one pass actually removed."""

    transcribed: int = 0  # uploads whose job finished
    expired: int = 0      # uploads past the retention window
    orphans: int = 0      # files on disk no job row points at
    failed_jobs: int = 0  # jobs marked failed because their audio expired
    freed_bytes: int = 0

    @property
    def deleted(self) -> int:
        return self.transcribed + self.expired + self.orphans

    def as_dict(self) -> dict[str, int]:
        return {
            "deleted": self.deleted,
            "transcribed": self.transcribed,
            "expired": self.expired,
            "orphans": self.orphans,
            "failed_jobs": self.failed_jobs,
            "freed_bytes": self.freed_bytes,
        }

    def summary(self) -> str:
        return (
            f"{self.deleted} file(s), {self.freed_bytes / 1e6:.1f} MB "
            f"(transcribed {self.transcribed}, expired {self.expired}, "
            f"orphans {self.orphans})"
        )


def cutoff_iso(days: float, now: datetime | None = None) -> str:
    """The timestamp a job must predate to have its audio expire."""
    now = now or datetime.now(timezone.utc)
    return (now - timedelta(days=days)).isoformat(timespec="seconds")


def _unlink(path_text: str) -> int | None:
    """Delete one upload. Returns bytes freed, or None if the file is still
    there and the caller should try again on the next sweep."""
    if not path_text:
        return 0
    path = Path(path_text)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    try:
        path.unlink()
    except FileNotFoundError:
        return 0  # already gone; still counts as reclaimed
    except OSError as exc:
        # Windows keeps a handle on a file another process has open. Leave it.
        log.warning("could not delete %s: %s", path, exc)
        return None
    return size


def reclaim_one(store: JobStore, job_id: str, audio_path: str) -> int | None:
    """Delete a job's upload and record it. Returns bytes freed, or None if the
    file survived and should be retried later."""
    freed = _unlink(audio_path)
    if freed is None:
        return None
    store.mark_audio_deleted(job_id)
    return freed


def sweep(
    settings: Settings, store: JobStore, now: datetime | None = None
) -> SweepReport:
    """One full pass: finished uploads, expired uploads, then stray files."""
    report = SweepReport()
    now = now or datetime.now(timezone.utc)

    if settings.delete_audio_on_success:
        for row in store.transcribed_audio(BATCH):
            freed = reclaim_one(store, row["id"], row["audio_path"])
            if freed is not None:
                report.transcribed += 1
                report.freed_bytes += freed

    days = settings.audio_retention_days
    if days > 0:
        cutoff = cutoff_iso(days, now)
        for row in store.expired_audio(cutoff, BATCH):
            freed = reclaim_one(store, row["id"], row["audio_path"])
            if freed is None:
                continue
            report.expired += 1
            report.freed_bytes += freed
            # A queued job whose audio just expired can never run. Failing it
            # here says so plainly, instead of letting the worker claim it and
            # trip over a missing file once per retry.
            if row["status"] in store_mod.ACTIVE_STATES:
                store.update(
                    row["id"],
                    status=store_mod.FAILED,
                    stage=None,
                    progress=0.0,
                    error=f"audio expired after {days:g} day(s) of retention",
                    finished_at=store_mod.utcnow(),
                )
                report.failed_jobs += 1
        _sweep_orphans(settings, store, now, days, report)

    return report


def _sweep_orphans(
    settings: Settings,
    store: JobStore,
    now: datetime,
    days: float,
    report: SweepReport,
) -> None:
    """Remove upload files no job row points at.

    These come from job rows deleted by hand and from uploads interrupted
    between writing the file and inserting the row. The mtime check keeps an
    upload still streaming in from being mistaken for a leftover.
    """
    upload_dir = settings.upload_dir
    if not upload_dir.is_dir():
        return

    # Match on file name, not full path: names carry a uuid prefix so they are
    # unique, and a row written before the data dir moved still protects its
    # file instead of looking like a leftover.
    known = {Path(p).name for p in store.referenced_audio_paths()}
    stale_before = (now - timedelta(days=days)).timestamp()

    for entry in upload_dir.iterdir():
        if report.orphans >= BATCH:
            break
        if not entry.is_file() or entry.name in known:
            continue
        try:
            if entry.stat().st_mtime >= stale_before:
                continue
        except OSError:
            continue
        freed = _unlink(str(entry))
        if freed is not None:
            log.info("removed orphan upload %s", entry.name)
            report.orphans += 1
            report.freed_bytes += freed
