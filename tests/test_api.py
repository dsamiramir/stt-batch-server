"""API-level tests. conftest pins the off-peak window to a slot that excludes
"now", so the worker thread stays gated and jobs sit in `queued`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _submit(client: TestClient, wav: Path, **data) -> dict:
    with wav.open("rb") as fh:
        response = client.post(
            "/v1/jobs",
            files={"file": (wav.name, fh, "audio/wav")},
            data=data,
        )
    assert response.status_code == 202, response.text
    return response.json()


def test_health_reports_a_closed_gate(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["gate"]["accepting_work"] is False
    assert "next opens" in body["gate"]["reason"]
    assert body["gate"]["window_open"] is False
    assert body["model"]["loaded"] is False


def test_submit_queues_a_job(client: TestClient, stereo_wav: Path) -> None:
    job = _submit(
        client,
        stereo_wav,
        options=json.dumps({"language": "ko", "channel_names": ["agent", "customer"]}),
        client_ref="ticket-42",
    )
    assert job["status"] == "queued"
    assert job["client_ref"] == "ticket-42"
    assert job["result_url"] is None

    fetched = client.get(f"/v1/jobs/{job['id']}").json()
    assert fetched["id"] == job["id"]
    assert fetched["filename"] == "stereo.wav"


def test_result_is_409_until_finished(client: TestClient, stereo_wav: Path) -> None:
    job = _submit(client, stereo_wav)
    response = client.get(f"/v1/jobs/{job['id']}/result")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/does-not-exist").status_code == 404
    assert client.get("/v1/jobs/does-not-exist/result").status_code == 404


def test_bad_options_are_rejected(client: TestClient, stereo_wav: Path) -> None:
    with stereo_wav.open("rb") as fh:
        response = client.post(
            "/v1/jobs",
            files={"file": (stereo_wav.name, fh, "audio/wav")},
            data={"options": json.dumps({"beam_size": 99})},
        )
    assert response.status_code == 422

    with stereo_wav.open("rb") as fh:
        response = client.post(
            "/v1/jobs",
            files={"file": (stereo_wav.name, fh, "audio/wav")},
            data={"options": "{not json"},
        )
    assert response.status_code == 422


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs", files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert response.status_code == 422


def test_cancel_a_queued_job(client: TestClient, stereo_wav: Path) -> None:
    job = _submit(client, stereo_wav)
    response = client.delete(f"/v1/jobs/{job['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "canceled"
    assert client.get(f"/v1/jobs/{job['id']}").json()["status"] == "canceled"


def test_listing_and_counts(client: TestClient, stereo_wav: Path) -> None:
    _submit(client, stereo_wav)
    body = client.get("/v1/jobs", params={"status": "queued", "limit": 5}).json()
    assert all(j["status"] == "queued" for j in body["jobs"])
    assert body["total_by_status"]["queued"] >= 1


def test_control_pause_and_override(client: TestClient) -> None:
    paused = client.post("/v1/control", json={"pause": True}).json()
    assert paused["paused"] is True
    assert paused["accepting_work"] is False

    resumed = client.post("/v1/control", json={"pause": False}).json()
    assert resumed["paused"] is False
    assert resumed["accepting_work"] is False  # still outside the window

    overridden = client.post("/v1/control", json={"override_minutes": 30}).json()
    assert overridden["override_until"] is not None
    assert overridden["accepting_work"] is True

    cleared = client.post("/v1/control", json={"override_minutes": 0}).json()
    assert cleared["override_until"] is None
    assert cleared["accepting_work"] is False


# --- settings ---------------------------------------------------------------
@pytest.fixture
def settings_sandbox(tmp_path, monkeypatch):
    """Keep settings edits from leaking into other tests, and never let a test
    write to the project's real .env."""
    from app import main as main_mod
    from app.config import settings as live
    from app.models import LIVE_FIELDS

    monkeypatch.setattr(main_mod, "ENV_FILE", tmp_path / ".env")
    # 워커가 이 테스트 중에 잡을 집어가면 모델을 실제로 로드해 버린다.
    main_mod.state.worker.paused = True

    before = {name: getattr(live, name) for name in LIVE_FIELDS}
    try:
        yield tmp_path / ".env"
    finally:
        for name, value in before.items():
            setattr(live, name, value)
        main_mod.state.worker.apply_settings()
        main_mod.state.worker.paused = False


def test_get_settings_reports_live_fields(client: TestClient, settings_sandbox) -> None:
    body = client.get("/v1/settings").json()
    assert body["schedule_enabled"] is True
    assert body["gpu_idle_check"] is False       # conftest 가 꺼둔 값
    assert isinstance(body["gpu_max_util"], int)
    assert "model_size" in body["restart_required_for"]
    assert body["gate"]["paused"] is True


def test_put_settings_applies_and_persists(client: TestClient, settings_sandbox) -> None:
    from dotenv import dotenv_values

    body = client.put("/v1/settings", json={
        "schedule_enabled": False,
        "gpu_idle_check": True,
        "gpu_max_util": 40,
        "gpu_min_free_mb": 5000,
    }).json()

    assert body["schedule_enabled"] is False
    assert body["gpu_max_util"] == 40

    saved = dotenv_values(settings_sandbox)
    assert saved["STT_SCHEDULE_ENABLED"] == "false"
    assert saved["STT_GPU_IDLE_CHECK"] == "true"
    assert saved["STT_GPU_MAX_UTIL"] == "40"
    assert saved["STT_GPU_MIN_FREE_MB"] == "5000"


def test_put_settings_changes_the_reported_gate(client: TestClient, settings_sandbox) -> None:
    from app import main as main_mod

    # 게이트 판정을 보려면 일시정지를 풀어야 하지만, 큐를 비워둬야 안전하다.
    for job in client.get("/v1/jobs", params={"status": "queued", "limit": 500}).json()["jobs"]:
        client.delete(f"/v1/jobs/{job['id']}")
    main_mod.state.worker.paused = False

    closed = client.get("/v1/health").json()["gate"]
    assert closed["accepting_work"] is False
    assert "outside off-peak window" in closed["reason"]

    body = client.put("/v1/settings", json={
        "schedule_enabled": False, "gpu_idle_check": False,
    }).json()
    assert body["gate"]["accepting_work"] is True
    assert "scheduling disabled" in body["gate"]["reason"]


def test_put_settings_updates_the_window_spec(client: TestClient, settings_sandbox) -> None:
    body = client.put("/v1/settings", json={"windows": "Sat,Sun 09:00-18:00"}).json()
    assert body["windows"] == "Sat,Sun 09:00-18:00"
    assert body["gate"]["windows"] == ["Sat,Sun 09:00-18:00"]


def test_put_settings_rejects_a_bad_window_spec(client: TestClient, settings_sandbox) -> None:
    from app.config import settings as live

    before = live.windows
    response = client.put("/v1/settings", json={"windows": "Xyz 99:99-00:00"})
    assert response.status_code == 422
    assert "window spec" in response.json()["detail"]
    assert live.windows == before, "거부된 값이 반영되면 안 된다"
    assert not settings_sandbox.exists(), "거부된 요청은 파일을 건드리면 안 된다"


@pytest.mark.parametrize("payload", [
    {"gpu_max_util": 250},
    {"gpu_max_util": -1},
    {"poll_interval_sec": 0.1},
    {"gpu_probe_samples": 99},
    {"model_size": "tiny"},        # 재시작이 필요한 값은 받지 않는다
    {"port": 9999},
])
def test_put_settings_validation(client: TestClient, settings_sandbox, payload) -> None:
    assert client.put("/v1/settings", json=payload).status_code == 422


def test_put_settings_empty_body_is_a_noop(client: TestClient, settings_sandbox) -> None:
    response = client.put("/v1/settings", json={})
    assert response.status_code == 200
    assert not settings_sandbox.exists()


# --- retention ---------------------------------------------------------------
def test_retention_settings_round_trip(client: TestClient, settings_sandbox) -> None:
    from dotenv import dotenv_values

    body = client.put("/v1/settings", json={
        "delete_audio_on_success": False,
        "audio_retention_days": 7,
        "cleanup_interval_sec": 0,
    }).json()

    assert body["delete_audio_on_success"] is False
    assert body["audio_retention_days"] == 7
    assert body["cleanup_interval_sec"] == 0, "0 은 생략이 아니라 '주기 정리 끔' 이다"

    saved = dotenv_values(settings_sandbox)
    assert saved["STT_DELETE_AUDIO_ON_SUCCESS"] == "false"
    assert saved["STT_AUDIO_RETENTION_DAYS"] == "7.0"
    assert saved["STT_CLEANUP_INTERVAL_SEC"] == "0"


@pytest.mark.parametrize("payload", [
    {"audio_retention_days": -1},
    {"cleanup_interval_sec": -1},
    {"cleanup_interval_sec": 99999999},
])
def test_retention_settings_validation(client: TestClient, settings_sandbox, payload) -> None:
    assert client.put("/v1/settings", json=payload).status_code == 422


def test_cleanup_leaves_queued_uploads_alone(client: TestClient, stereo_wav: Path) -> None:
    """큐에 막 들어온 잡의 오디오까지 쓸어가면 안 된다."""
    job = _submit(client, stereo_wav)

    report = client.post("/v1/maintenance/cleanup")
    assert report.status_code == 200
    assert report.json()["expired"] == 0

    body = client.get(f"/v1/jobs/{job['id']}").json()
    assert body["status"] == "queued"
    assert body["audio_deleted_at"] is None


# --- queue position / eta ----------------------------------------------------
def test_queue_position_is_reported_in_order(client: TestClient, stereo_wav: Path) -> None:
    first = _submit(client, stereo_wav, client_ref="pos-1")
    second = _submit(client, stereo_wav, client_ref="pos-2")

    p1 = client.get(f"/v1/jobs/{first['id']}").json()["queue_position"]
    p2 = client.get(f"/v1/jobs/{second['id']}").json()["queue_position"]
    assert p1 is not None and p2 == p1 + 1, "먼저 넣은 잡이 앞 순번이어야 한다"


def test_priority_jumps_the_queue(client: TestClient, stereo_wav: Path) -> None:
    normal = _submit(client, stereo_wav, client_ref="prio-normal")
    urgent = _submit(client, stereo_wav, client_ref="prio-urgent", priority="5")

    pn = client.get(f"/v1/jobs/{normal['id']}").json()["queue_position"]
    pu = client.get(f"/v1/jobs/{urgent['id']}").json()["queue_position"]
    assert pu < pn


def test_terminal_jobs_have_no_position(client: TestClient, stereo_wav: Path) -> None:
    job = _submit(client, stereo_wav)
    client.delete(f"/v1/jobs/{job['id']}")
    body = client.get(f"/v1/jobs/{job['id']}").json()
    assert body["queue_position"] is None
    assert body["eta_seconds"] is None


def test_eta_is_null_without_history(client: TestClient, stereo_wav: Path) -> None:
    """이 테스트 DB 에는 완료된 잡이 없으므로 추정 근거가 없다."""
    job = _submit(client, stereo_wav)
    assert client.get(f"/v1/jobs/{job['id']}").json()["eta_seconds"] is None


def test_eta_uses_recent_average(client: TestClient, stereo_wav: Path) -> None:
    from app import main as main_mod

    job = _submit(client, stereo_wav)
    position = client.get(f"/v1/jobs/{job['id']}").json()["queue_position"]

    # 완료 이력을 흉내내서 추정치가 순번에 비례하는지 본다.
    real = main_mod.state.store.average_job_seconds
    main_mod.state.store.average_job_seconds = lambda sample=20: 100.0
    try:
        body = client.get(f"/v1/jobs/{job['id']}").json()
        assert body["eta_seconds"] == position * 100
        listing = client.get("/v1/jobs", params={"limit": 100}).json()
        assert listing["backlog_eta_seconds"] is not None
    finally:
        main_mod.state.store.average_job_seconds = real


def test_listing_reports_positions_without_n_queries(client: TestClient, stereo_wav: Path) -> None:
    _submit(client, stereo_wav)
    body = client.get("/v1/jobs", params={"status": "queued", "limit": 100}).json()
    positions = [j["queue_position"] for j in body["jobs"]]
    assert all(p is not None for p in positions)
    assert sorted(positions) == list(range(1, len(positions) + 1))


# --- duplicate submissions ---------------------------------------------------
def _post(client: TestClient, wav: Path, **data):
    with wav.open("rb") as fh:
        return client.post(
            "/v1/jobs", files={"file": (wav.name, fh, "audio/wav")}, data=data
        )


def test_resubmitting_a_client_ref_returns_the_existing_job(
    client: TestClient, stereo_wav: Path, data_dir: Path
) -> None:
    first = _submit(client, stereo_wav, client_ref="conv-dup-1")
    uploads_before = len(list((data_dir / "uploads").iterdir()))

    again = _post(client, stereo_wav, client_ref="conv-dup-1")

    assert again.status_code == 200, "새 잡이 아니므로 202 가 아니다"
    body = again.json()
    assert body["id"] == first["id"]
    assert body["reused"] is True
    assert body["queue_position"] is not None
    assert len(list((data_dir / "uploads").iterdir())) == uploads_before, \
        "중복 제출은 업로드 파일을 새로 쓰지 않는다"


@pytest.mark.parametrize("flag", ["force", "force_new"])
def test_force_queues_a_second_job_beside_the_first(
    client: TestClient, stereo_wav: Path, flag: str
) -> None:
    ref = f"conv-dup-{flag}"
    first = _submit(client, stereo_wav, client_ref=ref)
    second = _submit(client, stereo_wav, client_ref=ref, **{flag: "true"})

    assert second["id"] != first["id"]
    assert second["reused"] is False
    assert second["client_ref"] == ref, "재처리도 원래 client_ref 를 그대로 쓴다"
    assert client.get(f"/v1/jobs/{first['id']}").json()["status"] == "queued", \
        "force 는 기존 잡을 건드리지 않는다"


def test_a_canceled_job_can_be_resubmitted(client: TestClient, stereo_wav: Path) -> None:
    first = _submit(client, stereo_wav, client_ref="conv-dup-3")
    client.delete(f"/v1/jobs/{first['id']}")
    second = _submit(client, stereo_wav, client_ref="conv-dup-3")
    assert second["id"] != first["id"]


def test_jobs_without_client_ref_are_never_merged(client: TestClient, stereo_wav: Path) -> None:
    a = _submit(client, stereo_wav)
    b = _submit(client, stereo_wav)
    assert a["id"] != b["id"]


def test_listing_filters_by_client_ref(client: TestClient, stereo_wav: Path) -> None:
    job = _submit(client, stereo_wav, client_ref="conv-list-1")
    _submit(client, stereo_wav, client_ref="conv-list-2")

    body = client.get("/v1/jobs", params={"client_ref": "conv-list-1"}).json()

    assert [j["id"] for j in body["jobs"]] == [job["id"]]
