from __future__ import annotations

from app.config import Settings
from app.engine import Engine


def _engine(**overrides: object) -> Engine:
    return Engine(Settings(**overrides))


def test_a_jobs_hotwords_are_added_to_the_servers_list() -> None:
    """The client sends its own terms on every job, so a server-side list that
    lost to the override would never reach the decoder."""
    engine = _engine(hotwords="당근알바")
    assert engine._decode_options({"hotwords": "당근 당근페이"})["hotwords"] == (
        "당근알바 당근 당근페이"
    )


def test_the_default_silence_window_is_the_library_default() -> None:
    """500ms cut Korean conversation into ~2.5s slices and halved the share of
    segments that ended in punctuation."""
    assert Settings.model_fields["vad_min_silence_ms"].default == 2000


def test_vad_parameters_come_from_settings() -> None:
    params = _engine(
        vad_min_silence_ms=1800, vad_speech_pad_ms=300, vad_max_speech_sec=25.0
    )._vad_parameters({})
    assert params.min_silence_duration_ms == 1800
    assert params.speech_pad_ms == 300
    assert params.max_speech_duration_s == 25.0


def test_a_job_can_override_the_vad_window_for_an_ab_run() -> None:
    params = _engine(vad_min_silence_ms=2000)._vad_parameters({"vad_min_silence_ms": 500})
    assert params.min_silence_duration_ms == 500


def test_a_blank_prompt_reaches_the_decoder_as_none() -> None:
    assert _engine(initial_prompt="")._decode_options({})["initial_prompt"] is None
