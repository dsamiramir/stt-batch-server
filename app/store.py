"""SQLite-backed job store. Single writer lock keeps it safe across the API
threads and the worker thread; WAL keeps readers from blocking the worker."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELING = "canceling"
CANCELED = "canceled"

ACTIVE_STATES = (QUEUED, RUNNING, CANCELING)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    priority      INTEGER NOT NULL DEFAULT 0,
    filename      TEXT NOT NULL,
    audio_path    TEXT NOT NULL,
    result_path   TEXT,
    options       TEXT NOT NULL DEFAULT '{}',
    callback_url  TEXT,
    client_ref    TEXT,
    channels      INTEGER,
    duration_sec  REAL,
    progress      REAL NOT NULL DEFAULT 0,
    stage         TEXT,
    error         TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    rtf           REAL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    audio_deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""

_UPDATABLE = {
    "status", "priority", "result_path", "channels", "duration_sec", "progress",
    "stage", "error", "attempts", "rtf", "started_at", "finished_at", "options",
    "audio_deleted_at",
}

# Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an
# existing table alone, so a database from an older build needs them bolted on.
_ADDED_COLUMNS = {
    "audio_deleted_at": "TEXT",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        present = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for column, decl in _ADDED_COLUMNS.items():
            if column not in present:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- writes ------------------------------------------------------------
    def create(
        self,
        *,
        filename: str,
        audio_path: Path,
        options: dict[str, Any],
        callback_url: str | None = None,
        client_ref: str | None = None,
        priority: int = 0,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, status, priority, filename, audio_path, "
                "options, callback_url, client_ref, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    job_id, QUEUED, priority, filename, str(audio_path),
                    json.dumps(options, ensure_ascii=False), callback_url,
                    client_ref, utcnow(),
                ),
            )
            self._conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def update(self, job_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*fields.values(), job_id),
            )
            self._conn.commit()

    def claim_next(self) -> dict[str, Any] | None:
        """Atomically move the highest-priority queued job to RUNNING."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE status = ? "
                "ORDER BY priority DESC, created_at ASC LIMIT 1",
                (QUEUED,),
            ).fetchone()
            if row is None:
                return None
            cur = self._conn.execute(
                "UPDATE jobs SET status = ?, started_at = ?, progress = 0, "
                "attempts = attempts + 1, error = NULL WHERE id = ? AND status = ?",
                (RUNNING, utcnow(), row["id"], QUEUED),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
            return self.get(row["id"])

    def request_cancel(self, job_id: str) -> str | None:
        """Cancel outright if queued, or flag a running job to stop."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            status = row["status"]
            if status == QUEUED:
                self._conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, stage = NULL "
                    "WHERE id = ? AND status = ?",
                    (CANCELED, utcnow(), job_id, QUEUED),
                )
                self._conn.commit()
                return CANCELED
            if status == RUNNING:
                self._conn.execute(
                    "UPDATE jobs SET status = ? WHERE id = ? AND status = ?",
                    (CANCELING, job_id, RUNNING),
                )
                self._conn.commit()
                return CANCELING
            return status

    def requeue_orphans(self) -> int:
        """After a crash or restart, running jobs are no longer running."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = ?, stage = NULL, progress = 0 "
                "WHERE status IN (?, ?)",
                (QUEUED, RUNNING, CANCELING),
            )
            self._conn.commit()
            return cur.rowcount

    def mark_audio_deleted(self, job_id: str) -> None:
        """Record that the upload is gone, so the sweep stops reconsidering it
        and the API can tell a client the audio is no longer downloadable."""
        self.update(job_id, audio_deleted_at=utcnow())

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # --- reads -------------------------------------------------------------
    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def status_of(self, job_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return row["status"] if row else None

    def list(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def has_queued(self) -> bool:
        """Cheap 'is there anything to do' check, so the worker can skip the
        expensive gate evaluation while the queue is empty."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM jobs WHERE status = ? LIMIT 1", (QUEUED,)
            ).fetchone()
        return row is not None

    def queue_positions(self) -> dict[str, int]:
        """1-based rank of every queued job in the order the worker will claim
        them. One query for the whole queue, so listing stays cheap."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status = ? "
                "ORDER BY priority DESC, created_at ASC",
                (QUEUED,),
            ).fetchall()
        return {row["id"]: index + 1 for index, row in enumerate(rows)}

    def average_job_seconds(self, sample: int = 20) -> float | None:
        """Wall-clock seconds per job over the most recent completions.

        Used for rough ETAs, so a rolling sample beats an all-time average:
        it tracks the length of whatever the caller is currently sending.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT AVG(secs) AS avg_secs FROM ("
                "  SELECT (julianday(finished_at) - julianday(started_at)) * 86400.0"
                "         AS secs"
                "  FROM jobs"
                "  WHERE status = ? AND started_at IS NOT NULL"
                "        AND finished_at IS NOT NULL"
                "  ORDER BY finished_at DESC LIMIT ?"
                ")",
                (DONE, sample),
            ).fetchone()
        if row is None or row["avg_secs"] is None:
            return None
        return float(row["avg_secs"])

    # --- retention ---------------------------------------------------------
    def transcribed_audio(self, limit: int = 500) -> list[dict[str, Any]]:
        """Finished jobs still holding on to their upload."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, status, audio_path FROM jobs "
                "WHERE status = ? AND audio_deleted_at IS NULL AND audio_path <> '' "
                "ORDER BY finished_at ASC LIMIT ?",
                (DONE, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def expired_audio(self, cutoff: str, limit: int = 500) -> list[dict[str, Any]]:
        """Jobs that never produced a transcript and have sat past ``cutoff``.

        Running and canceling jobs are excluded on purpose: the worker has the
        file open, and a sweep must never pull it out from under a transcription
        in flight. Timestamps are all UTC ISO strings with the same offset, so
        comparing them as text sorts the same way as comparing the instants.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, status, audio_path FROM jobs "
                "WHERE status NOT IN (?, ?, ?) AND audio_deleted_at IS NULL "
                "      AND audio_path <> '' "
                "      AND COALESCE(finished_at, started_at, created_at) < ? "
                "ORDER BY created_at ASC LIMIT ?",
                (DONE, RUNNING, CANCELING, cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def referenced_audio_paths(self) -> set[str]:
        """Every upload a job row still lays claim to, for orphan detection.

        A job that has already released its audio claims nothing, so its row
        does not protect a file of that name from the orphan sweep. That is
        what lets a leftover copy be cleaned up after the data dir moves.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT audio_path FROM jobs "
                "WHERE audio_path <> '' AND audio_deleted_at IS NULL"
            ).fetchall()
        return {r["audio_path"] for r in rows}

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}
