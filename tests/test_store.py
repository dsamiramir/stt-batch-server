from __future__ import annotations

from pathlib import Path

import pytest

from app import store as store_mod
from app.store import JobStore


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    s = JobStore(tmp_path / "jobs.db")
    yield s
    s.close()


def _make(store: JobStore, name: str = "a.wav", priority: int = 0) -> dict:
    return store.create(
        filename=name,
        audio_path=Path("/tmp") / name,
        options={"language": "ko"},
        priority=priority,
    )


def test_create_and_get(store: JobStore) -> None:
    job = _make(store)
    assert job["status"] == store_mod.QUEUED
    assert store.get(job["id"])["filename"] == "a.wav"
    assert store.get("missing") is None


def test_claim_is_priority_then_fifo(store: JobStore) -> None:
    first = _make(store, "first.wav")
    _make(store, "second.wav")
    urgent = _make(store, "urgent.wav", priority=5)

    assert store.claim_next()["id"] == urgent["id"]
    assert store.claim_next()["id"] == first["id"]


def test_claim_marks_running_and_counts_attempts(store: JobStore) -> None:
    job = _make(store)
    claimed = store.claim_next()
    assert claimed["status"] == store_mod.RUNNING
    assert claimed["attempts"] == 1
    assert claimed["started_at"] is not None
    assert store.claim_next() is None, "a running job must not be claimed twice"
    assert store.status_of(job["id"]) == store_mod.RUNNING


def test_cancel_queued_is_immediate(store: JobStore) -> None:
    job = _make(store)
    assert store.request_cancel(job["id"]) == store_mod.CANCELED
    assert store.claim_next() is None


def test_cancel_running_flags_for_the_worker(store: JobStore) -> None:
    _make(store)
    claimed = store.claim_next()
    assert store.request_cancel(claimed["id"]) == store_mod.CANCELING
    assert store.status_of(claimed["id"]) == store_mod.CANCELING


def test_requeue_orphans_after_restart(store: JobStore) -> None:
    _make(store)
    _make(store, "b.wav")
    store.claim_next()
    assert store.requeue_orphans() == 1
    assert store.counts()[store_mod.QUEUED] == 2


def test_update_rejects_unknown_columns(store: JobStore) -> None:
    job = _make(store)
    store.update(job["id"], progress=0.5, stage="working")
    assert store.get(job["id"])["progress"] == 0.5
    with pytest.raises(ValueError):
        store.update(job["id"], nonsense="x")


def test_list_filters_by_status(store: JobStore) -> None:
    _make(store)
    _make(store, "b.wav")
    store.claim_next()
    assert len(store.list(status=store_mod.RUNNING)) == 1
    assert len(store.list()) == 2
