"""The gate decision matrix, exercised without starting a worker thread.

gate_status() is what the dashboard reads on every poll, so it must agree with
evaluate_gate() (what the worker actually acts on) while never sleeping.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app import gpu as gpu_mod
from app import worker as worker_mod
from app.config import Settings
from app.gpu import GpuStat
from app.worker import Worker


class _FakeStore:
    def __init__(self, queued: bool = True) -> None:
        self._queued = queued

    def has_queued(self) -> bool:
        return self._queued


class _FakeEngine:
    def __init__(self, loaded: bool = False) -> None:
        self.is_loaded = loaded


def make_worker(**overrides) -> Worker:
    settings = Settings()
    settings.windows = "Mon-Fri 22:00-08:00; Sat 00:00-24:00; Sun 00:00-24:00"
    settings.schedule_enabled = True
    settings.gpu_idle_check = True
    settings.gpu_max_util = 25
    settings.gpu_min_free_mb = 6000
    for key, value in overrides.items():
        setattr(settings, key, value)
    return Worker(settings, _FakeStore(), _FakeEngine())  # type: ignore[arg-type]


@pytest.fixture
def idle_gpu(monkeypatch: pytest.MonkeyPatch):
    def _set(util: int = 3, used: int = 2000, total: int = 16000):
        monkeypatch.setattr(
            gpu_mod, "get_gpu", lambda index: GpuStat(index, util, used, total)
        )
        monkeypatch.setattr(
            worker_mod.gpu_mod, "get_gpu", lambda index: GpuStat(index, util, used, total)
        )
    return _set


# 2026-09-01 is a Tuesday.
IN_WINDOW = datetime(2026, 9, 1, 23, 30)
OUT_OF_WINDOW = datetime(2026, 9, 1, 14, 0)


def test_both_gates_off_always_accepts(idle_gpu) -> None:
    idle_gpu(util=99, used=15900, total=16000)  # 아무리 바빠도 무시해야 한다
    w = make_worker(schedule_enabled=False, gpu_idle_check=False)
    status = w.gate_status()
    assert status["accepting_work"] is True
    assert "scheduling disabled" in status["reason"]
    assert w.evaluate_gate(OUT_OF_WINDOW)[0] is True


def test_gpu_only_runs_whenever_the_card_is_free(idle_gpu) -> None:
    w = make_worker(schedule_enabled=False, gpu_idle_check=True)
    idle_gpu(util=3)
    assert w.gate_status()["accepting_work"] is True

    idle_gpu(util=80)
    status = w.gate_status()
    assert status["accepting_work"] is False
    assert "GPU util 80% > 25%" in status["reason"]


def test_gpu_only_blocks_on_low_vram_until_the_model_is_resident(idle_gpu) -> None:
    w = make_worker(schedule_enabled=False)
    idle_gpu(util=2, used=14000, total=16000)  # 여유 2000MB < 6000MB
    assert w.gate_status()["accepting_work"] is False

    # 모델이 이미 올라가 있으면 그 VRAM 은 우리 것이므로 요건에서 빼야 한다.
    w.engine.is_loaded = True
    assert w.gate_status()["accepting_work"] is True


def test_schedule_only_ignores_a_busy_gpu(idle_gpu) -> None:
    idle_gpu(util=99, used=15900, total=16000)
    w = make_worker(schedule_enabled=True, gpu_idle_check=False)
    assert w.evaluate_gate(IN_WINDOW)[0] is True
    assert w.evaluate_gate(OUT_OF_WINDOW)[0] is False


def test_both_gates_on_requires_both(idle_gpu) -> None:
    w = make_worker()
    idle_gpu(util=3)
    assert w.evaluate_gate(IN_WINDOW)[0] is True
    assert w.evaluate_gate(OUT_OF_WINDOW)[0] is False

    idle_gpu(util=80)
    assert w.evaluate_gate(IN_WINDOW)[0] is False


def test_pause_beats_everything(idle_gpu) -> None:
    idle_gpu(util=1)
    w = make_worker(schedule_enabled=False, gpu_idle_check=False)
    w.paused = True
    assert w.gate_status()["accepting_work"] is False
    assert w.gate_status()["reason"] == "worker paused"
    assert w.evaluate_gate()[0] is False


def test_override_beats_both_gates(idle_gpu) -> None:
    idle_gpu(util=99, used=15900, total=16000)
    w = make_worker()
    assert w.gate_status()["accepting_work"] is False

    w.set_override(30)
    status = w.gate_status()
    assert status["accepting_work"] is True
    assert status["reason"] == "manual override active"
    assert status["override_until"] is not None

    w.set_override(0)
    assert w.gate_status()["accepting_work"] is False


def test_expired_override_stops_applying(idle_gpu) -> None:
    idle_gpu(util=99, used=15900, total=16000)
    w = make_worker()
    w.override_until = datetime.now().astimezone() - timedelta(minutes=1)
    assert w.gate_status()["accepting_work"] is False


def test_missing_nvidia_smi_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_mod.gpu_mod, "get_gpu", lambda index: None)
    w = make_worker(schedule_enabled=False, gpu_idle_check=True)
    status = w.gate_status()
    assert status["accepting_work"] is True
    assert "nvidia-smi unavailable" in status["reason"]


def test_gate_status_never_probes(monkeypatch: pytest.MonkeyPatch, idle_gpu) -> None:
    """A sleeping probe here would stall every /v1/health request."""
    idle_gpu(util=3)
    calls: list[int] = []
    monkeypatch.setattr(
        worker_mod.gpu_mod, "check_idle", lambda **kw: calls.append(1) or (True, "x")
    )
    w = make_worker(schedule_enabled=False)
    w.gate_status()
    assert calls == []


def test_apply_settings_reparses_windows() -> None:
    w = make_worker()
    w.settings.windows = "Sat,Sun 09:00-18:00"
    w.apply_settings()
    assert [win.raw for win in w.windows] == ["Sat,Sun 09:00-18:00"]


def test_apply_settings_rejects_a_bad_spec_and_keeps_the_old_one() -> None:
    from app.schedule import WindowSpecError

    w = make_worker()
    before = list(w.windows)
    w.settings.windows = "Xyz 99:99-00:00"
    with pytest.raises(WindowSpecError):
        w.apply_settings()
    assert w.windows == before
