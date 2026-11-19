"""실행 버전은 현재 health나 재사용 시점이 아니라 실제 처리 결과에 고정한다."""
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from app import store as store_mod, worker as worker_mod
from app.config import Settings
from app.models import JobOut
from app.store import JobStore
from app.versioning import SERVER_VERSION, build_version
from app.worker import Worker


def test_build_fingerprint_ignores_location_crlf_and_secrets(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        (root / "app").mkdir(parents=True)
        (root / "pyproject.toml").write_text('[project]\nversion="0.1.0"\n')
    (roots[0] / "app" / "main.py").write_bytes(b"value = 1\n")
    (roots[1] / "app" / "main.py").write_bytes(b"value = 1\r\n")
    (roots[1] / ".env").write_text("SECRET=never-read")
    assert build_version(roots[0]) == build_version(roots[1])
    (roots[1] / "app" / "main.py").write_text("value = 2\n")
    assert build_version(roots[0]) != build_version(roots[1])


def test_existing_jobs_have_unknown_version_after_schema_upgrade(tmp_path):
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(store_mod._SCHEMA)
        conn.execute("insert into jobs(id,status,filename,audio_path,created_at) values ('old','done','old.wav','old.wav','2026-11-01')")
    store = JobStore(db)
    try:
        row = store.get("old")
        assert row["submitted_server_version"] is None
        assert row["processing_server_version"] is None
        assert JobOut.from_row(row).processing_server_version is None
    finally:
        store.close()


def test_result_and_callback_keep_actual_processing_version(tmp_path, stereo_wav, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", schedule_enabled=False,
                        delete_audio_on_success=False, callback_retries=1)
    settings.ensure_dirs()
    store = JobStore(settings.db_path)
    monkeypatch.setattr(store_mod, "SERVER_VERSION", "0.1.0+submitted")
    submitted = store.create(filename="a.wav", audio_path=stereo_wav, options={}, client_ref="version-test", callback_url="http://example.invalid/callback")
    job = store.claim_next()
    callbacks = []
    monkeypatch.setattr(worker_mod.httpx, "post", lambda *a, **kw: callbacks.append(kw["json"]) or SimpleNamespace(status_code=200))
    class FakeEngine:
        def describe(self): return {"model": "fake"}
        def transcribe(self, *args, **kwargs): return [], {"language": "ko"}
    monkeypatch.setattr(worker_mod, "SERVER_VERSION", "0.1.0+processed")
    worker = Worker(settings, store, FakeEngine())
    worker._process(job)
    row = store.get(job["id"])
    assert row["status"] == "done"
    assert row["submitted_server_version"] == "0.1.0+submitted"
    assert row["processing_server_version"] == "0.1.0+processed"
    result_path = Path(row["result_path"])
    assert json.loads(result_path.read_text(encoding="utf-8"))["server_version"] == "0.1.0+processed"
    assert callbacks[0]["server_version"] == "0.1.0+processed"
    monkeypatch.setattr(worker_mod, "SERVER_VERSION", "0.1.0+next-build")
    reused = store.find_reusable("version-test")
    assert JobOut.from_row(reused).processing_server_version == "0.1.0+processed"
    assert json.loads(result_path.read_text(encoding="utf-8"))["server_version"] == "0.1.0+processed"
    store.close()


def test_health_version_uses_build_fingerprint():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        assert client.get("/v1/health").json()["version"] == SERVER_VERSION
    assert app.version == SERVER_VERSION
    assert len(SERVER_VERSION.split("+")[1]) == 16
