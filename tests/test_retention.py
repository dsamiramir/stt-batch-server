"""Audio retention: what gets deleted, what is left alone, and when.

The dangerous half of a cleanup is the file it must *not* touch, so most of
these assert survival rather than deletion.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import retention
from app import store as store_mod
from app.config import Settings
from app.store import JobStore


@pytest.fixture
def env(tmp_path: Path):
    settings = Settings()
    settings.data_dir = tmp_path
    settings.delete_audio_on_success = True
    settings.audio_retention_days = 3.0
    settings.ensure_dirs()
    store = JobStore(settings.db_path)
    yield settings, store
    store.close()


def _upload(settings: Settings, name: str, mtime_days_ago: float = 0.0) -> Path:
    path = settings.upload_dir / name
    path.write_bytes(b"x" * 1024)
    if mtime_days_ago:
        when = (datetime.now() - timedelta(days=mtime_days_ago)).timestamp()
        import os

        os.utime(path, (when, when))
    return path


def _job(
    settings: Settings,
    store: JobStore,
    name: str,
    *,
    status: str = store_mod.QUEUED,
    age_days: float = 0.0,
) -> dict:
    path = _upload(settings, name, age_days)
    job = store.create(filename=name, audio_path=path, options={})
    fields: dict = {}
    if status != store_mod.QUEUED:
        fields["status"] = status
    if age_days:
        stamp = (
            datetime.now(timezone.utc) - timedelta(days=age_days)
        ).isoformat(timespec="seconds")
        # created_at is not updatable through the store's guarded writer; this
        # is the one place that has to age a row by hand.
        store._conn.execute(  # noqa: SLF001
            "UPDATE jobs SET created_at = ?, finished_at = ? WHERE id = ?",
            (stamp, stamp if status != store_mod.QUEUED else None, job["id"]),
        )
        store._conn.commit()  # noqa: SLF001
    if fields:
        store.update(job["id"], **fields)
    return store.get(job["id"])


def test_transcribed_audio_goes_immediately(env) -> None:
    settings, store = env
    job = _job(settings, store, "done.wav", status=store_mod.DONE)
    path = Path(job["audio_path"])

    report = retention.sweep(settings, store)

    assert not path.exists()
    assert report.transcribed == 1
    assert report.freed_bytes == 1024
    assert store.get(job["id"])["audio_deleted_at"] is not None


def test_a_second_sweep_does_not_re_count(env) -> None:
    settings, store = env
    _job(settings, store, "done.wav", status=store_mod.DONE)
    retention.sweep(settings, store)
    assert retention.sweep(settings, store).deleted == 0


def test_running_audio_is_never_touched(env) -> None:
    settings, store = env
    job = _job(settings, store, "live.wav", status=store_mod.RUNNING, age_days=30)

    retention.sweep(settings, store)

    assert Path(job["audio_path"]).exists(), "the worker still has this file open"
    assert store.get(job["id"])["audio_deleted_at"] is None


def test_fresh_unfinished_audio_survives(env) -> None:
    settings, store = env
    queued = _job(settings, store, "waiting.wav", age_days=1)
    failed = _job(settings, store, "broken.wav", status=store_mod.FAILED, age_days=2)

    assert retention.sweep(settings, store).expired == 0
    assert Path(queued["audio_path"]).exists()
    assert Path(failed["audio_path"]).exists()


@pytest.mark.parametrize(
    "status", [store_mod.QUEUED, store_mod.FAILED, store_mod.CANCELED]
)
def test_unfinished_audio_expires_past_the_window(env, status) -> None:
    settings, store = env
    job = _job(settings, store, f"old-{status}.wav", status=status, age_days=4)

    report = retention.sweep(settings, store)

    assert not Path(job["audio_path"]).exists()
    assert report.expired == 1
    assert store.get(job["id"])["audio_deleted_at"] is not None


def test_an_expired_queued_job_is_failed_not_left_to_trip(env) -> None:
    settings, store = env
    job = _job(settings, store, "stale.wav", age_days=5)

    report = retention.sweep(settings, store)

    row = store.get(job["id"])
    assert row["status"] == store_mod.FAILED
    assert "expired" in row["error"]
    assert report.failed_jobs == 1
    assert store.claim_next() is None, "a job with no audio must not be claimed"


def test_an_expired_failed_job_keeps_its_own_error(env) -> None:
    settings, store = env
    job = _job(settings, store, "old.wav", status=store_mod.FAILED, age_days=9)
    store.update(job["id"], error="RuntimeError: CUDA out of memory")

    retention.sweep(settings, store)

    assert store.get(job["id"])["error"] == "RuntimeError: CUDA out of memory"


def test_zero_days_keeps_unfinished_audio_forever(env) -> None:
    settings, store = env
    settings.audio_retention_days = 0
    job = _job(settings, store, "ancient.wav", status=store_mod.FAILED, age_days=400)

    assert retention.sweep(settings, store).expired == 0
    assert Path(job["audio_path"]).exists()


def test_success_deletion_can_be_turned_off(env) -> None:
    settings, store = env
    settings.delete_audio_on_success = False
    job = _job(settings, store, "keep.wav", status=store_mod.DONE)

    assert retention.sweep(settings, store).transcribed == 0
    assert Path(job["audio_path"]).exists()


def test_results_are_never_swept(env) -> None:
    settings, store = env
    job = _job(settings, store, "done.wav", status=store_mod.DONE)
    result = settings.result_dir / f"{job['id']}.json"
    result.write_text("{}", encoding="utf-8")

    retention.sweep(settings, store)

    assert result.exists(), "the transcript outlives the recording"


def test_orphan_files_are_swept_once_stale(env) -> None:
    settings, store = env
    fresh = _upload(settings, "fresh_orphan.wav")
    stale = _upload(settings, "stale_orphan.wav", mtime_days_ago=10)
    claimed = _job(settings, store, "claimed.wav", age_days=1)

    report = retention.sweep(settings, store)

    assert not stale.exists()
    assert fresh.exists(), "a fresh file may still be an upload in flight"
    assert Path(claimed["audio_path"]).exists()
    assert report.orphans == 1


def test_a_missing_file_still_marks_the_job(env) -> None:
    settings, store = env
    job = _job(settings, store, "vanished.wav", status=store_mod.DONE)
    Path(job["audio_path"]).unlink()

    report = retention.sweep(settings, store)

    assert report.transcribed == 1
    assert report.freed_bytes == 0
    assert store.get(job["id"])["audio_deleted_at"] is not None


def test_a_locked_file_is_left_for_the_next_sweep(env, monkeypatch) -> None:
    settings, store = env
    job = _job(settings, store, "locked.wav", status=store_mod.DONE)

    monkeypatch.setattr(retention, "_unlink", lambda _p: None)
    assert retention.sweep(settings, store).transcribed == 0
    assert store.get(job["id"])["audio_deleted_at"] is None

    monkeypatch.undo()
    assert retention.sweep(settings, store).transcribed == 1
    assert not Path(job["audio_path"]).exists()


def test_migration_adds_the_column_to_an_older_database(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "jobs.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, "
            "priority INTEGER NOT NULL DEFAULT 0, filename TEXT NOT NULL, "
            "audio_path TEXT NOT NULL, options TEXT NOT NULL DEFAULT '{}', "
            "created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO jobs (id, status, filename, audio_path, created_at) "
            "VALUES ('old', 'done', 'a.wav', '/tmp/a.wav', '2020-01-01T00:00:00+00:00')"
        )

    store = JobStore(db)
    try:
        assert "audio_deleted_at" in store.get("old")
        store.mark_audio_deleted("old")
        assert store.get("old")["audio_deleted_at"] is not None
    finally:
        store.close()


# --- a moved data directory --------------------------------------------------
# audio_path is absolute and written at submission time, so it goes stale the
# moment the data dir moves. Retention must follow the current dir, never the
# recorded one - otherwise it deletes files out of the old location.
def test_a_stale_path_never_deletes_outside_the_upload_dir(env, tmp_path) -> None:
    settings, store = env
    old_dir = tmp_path / "old-home" / "uploads"
    old_dir.mkdir(parents=True)
    leftover = old_dir / "abc_call.wav"
    leftover.write_bytes(b"x" * 2048)

    here = _upload(settings, "abc_call.wav")
    job = store.create(filename="call.wav", audio_path=leftover, options={})
    store.update(job["id"], status=store_mod.DONE)

    report = retention.sweep(settings, store)

    assert leftover.exists(), "a path outside the upload dir is not ours to delete"
    assert not here.exists(), "the copy in the live upload dir is the one to reclaim"
    assert report.transcribed == 1


def test_a_stale_path_with_nothing_on_disk_still_settles(env, tmp_path) -> None:
    settings, store = env
    job = store.create(
        filename="gone.wav", audio_path=tmp_path / "elsewhere" / "xyz_gone.wav",
        options={},
    )
    store.update(job["id"], status=store_mod.DONE)

    assert retention.sweep(settings, store).transcribed == 1
    assert store.get(job["id"])["audio_deleted_at"] is not None
    assert retention.sweep(settings, store).deleted == 0, "and does not recur"


def test_a_reclaimed_row_stops_shielding_its_file_from_the_orphan_sweep(env) -> None:
    """After a move, the live copy is a file no *claiming* row points at."""
    settings, store = env
    job = _job(settings, store, "left_behind.wav", status=store_mod.DONE, age_days=9)
    path = Path(job["audio_path"])
    store.mark_audio_deleted(job["id"])  # marked, but the file is still here

    report = retention.sweep(settings, store)

    assert not path.exists()
    assert report.orphans == 1
