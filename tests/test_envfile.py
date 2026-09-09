from __future__ import annotations

from pathlib import Path

from app.envfile import format_value, update_env_file

SAMPLE = """# 주석은 그대로 남아야 한다
STT_HOST=0.0.0.0
STT_PORT=8760

# STT_INITIAL_PROMPT=상담 녹취입니다
STT_SCHEDULE_ENABLED=true
STT_WINDOWS=Mon-Fri 22:00-08:00; Sat 00:00-24:00
"""


def test_updates_in_place_and_keeps_everything_else(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(SAMPLE, encoding="utf-8")

    update_env_file(env, {"STT_SCHEDULE_ENABLED": False, "STT_GPU_MAX_UTIL": 40})
    lines = env.read_text(encoding="utf-8").splitlines()

    assert "STT_SCHEDULE_ENABLED=false" in lines
    assert "STT_GPU_MAX_UTIL=40" in lines
    assert "# 주석은 그대로 남아야 한다" in lines
    assert "STT_HOST=0.0.0.0" in lines
    assert "STT_PORT=8760" in lines
    # 원래 키는 제자리에, 새 키만 뒤에 붙는다
    assert lines.index("STT_SCHEDULE_ENABLED=false") < lines.index("STT_GPU_MAX_UTIL=40")


def test_commented_out_keys_are_left_alone(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(SAMPLE, encoding="utf-8")

    update_env_file(env, {"STT_INITIAL_PROMPT": "새 프롬프트"})
    text = env.read_text(encoding="utf-8")

    assert "# STT_INITIAL_PROMPT=상담 녹취입니다" in text
    assert 'STT_INITIAL_PROMPT="새 프롬프트"' in text


def test_creates_file_when_missing(tmp_path: Path) -> None:
    env = tmp_path / "nested" / ".env"
    update_env_file(env, {"STT_GPU_IDLE_CHECK": True})
    assert env.read_text(encoding="utf-8").splitlines()[-1] == "STT_GPU_IDLE_CHECK=true"


def test_no_updates_is_a_noop(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(SAMPLE, encoding="utf-8")
    update_env_file(env, {})
    assert env.read_text(encoding="utf-8") == SAMPLE


def test_values_survive_a_round_trip_through_dotenv(tmp_path: Path) -> None:
    from dotenv import dotenv_values

    env = tmp_path / ".env"
    spec = "Mon-Fri 22:00-08:00; Sat 00:00-24:00; Sun 00:00-24:00"
    update_env_file(
        env,
        {
            "STT_WINDOWS": spec,
            "STT_SCHEDULE_ENABLED": False,
            "STT_GPU_MIN_FREE_MB": 6000,
            "STT_GPU_PROBE_INTERVAL_SEC": 1.5,
            "STT_INITIAL_PROMPT": '따옴표 " 포함',
        },
    )
    values = dotenv_values(env)

    assert values["STT_WINDOWS"] == spec
    assert values["STT_SCHEDULE_ENABLED"] == "false"
    assert values["STT_GPU_MIN_FREE_MB"] == "6000"
    assert values["STT_GPU_PROBE_INTERVAL_SEC"] == "1.5"
    assert values["STT_INITIAL_PROMPT"] == '따옴표 " 포함'


def test_format_value() -> None:
    assert format_value(True) == "true"
    assert format_value(False) == "false"
    assert format_value(12) == "12"
    assert format_value(1.5) == "1.5"
    assert format_value("ko") == "ko"
    assert format_value("a b") == '"a b"'
    assert format_value("") == '""'
