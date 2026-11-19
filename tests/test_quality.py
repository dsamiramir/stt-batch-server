from __future__ import annotations

import pytest

from app.quality import (
    echo_phrases,
    merge_hotwords,
    prepare_segments,
    usable_initial_prompt,
)

SAMPLE_RATE = 16_000


def test_faster_whisper_puts_the_removed_silence_back() -> None:
    """The engine leans on this instead of slicing the audio itself.

    Decoding each VAD slice separately was introduced because the restore was
    thought to be broken, and it cost the model the context it punctuates
    with. If a version bump ever does break the restore, fail here rather than
    in a month of transcripts.
    """
    from faster_whisper.transcribe import restore_speech_timestamps

    # 20s of silence, speech 20-25s, 10s of silence, speech 35-40s.
    chunks = [
        {"start": 20 * SAMPLE_RATE, "end": 25 * SAMPLE_RATE},
        {"start": 35 * SAMPLE_RATE, "end": 40 * SAMPLE_RATE},
    ]

    class Segment:
        def __init__(self, start: float, end: float) -> None:
            self.start, self.end, self.words = start, end, None

    first, second = restore_speech_timestamps(
        [Segment(0.0, 4.5), Segment(5.0, 9.5)], chunks, SAMPLE_RATE
    )
    assert first.start == pytest.approx(20, abs=0.05)
    assert second.start == pytest.approx(35, abs=0.05)


def test_long_segment_splits_at_the_first_word_of_each_piece() -> None:
    words = [
        {"start": float(i), "end": float(i) + 0.4, "word": "가" if i < 12 else "나"}
        for i in range(16)
    ]
    [first, second] = prepare_segments(
        [{"start": 0.0, "end": 15.4, "text": "가" * 16, "compression_ratio": 1.1, "words": words}]
    )
    assert first["start"] == 0.0
    assert second["start"] >= 10
    assert second["start"] == second["words"][0]["start"]


def test_loops_echo_and_compressed_text_are_dropped() -> None:
    kept = prepare_segments(
        [
            {"start": 0, "end": 1, "text": "네, 고객님", "compression_ratio": 1.0},
            {"start": 1, "end": 2, "text": "네네네네네네", "compression_ratio": 1.0},
            {"start": 2, "end": 3, "text": "상담원과 고객의 통화 녹취입니다.", "compression_ratio": 1.0},
            {"start": 3, "end": 4, "text": "같은말" * 30, "compression_ratio": 3.5},
        ]
    )
    assert [s["text"] for s in kept] == ["네, 고객님"]


def test_a_segment_that_is_only_a_hotword_is_dropped() -> None:
    """Measured 49 of these in 300 recent calls: the decoder reading its own
    prompt back. A hotword inside a sentence is a customer talking."""
    terms = echo_phrases("당근알바 당근페이 비즈니스", "네, 고객님. 확인해 드리겠습니다.")
    kept = prepare_segments(
        [
            {"start": 0, "end": 1, "text": "당근페이", "compression_ratio": 1.0},
            {"start": 1, "end": 2, "text": "비즈니스 비즈니스", "compression_ratio": 1.0},
            {"start": 2, "end": 3, "text": "네, 고객님. 확인해 드리겠습니다.", "compression_ratio": 1.0},
            {"start": 3, "end": 4, "text": "당근페이 충전하려고 하는데요", "compression_ratio": 1.0},
        ],
        echo_terms=terms,
    )
    assert [s["text"] for s in kept] == ["당근페이 충전하려고 하는데요"]


def test_a_single_syllable_is_never_treated_as_echo() -> None:
    """"네" on its own is an answer far more often than it is an echo."""
    terms = echo_phrases("네 당근알바")
    kept = prepare_segments(
        [{"start": 0, "end": 1, "text": "네", "compression_ratio": 1.0}],
        echo_terms=terms,
    )
    assert [s["text"] for s in kept] == ["네"]


def test_a_repeated_tail_fragment_is_dropped_only_when_it_is_adjacent() -> None:
    kept = prepare_segments(
        [
            {"start": 0.0, "end": 5.0, "text": "전화가 가게 되세요", "compression_ratio": 1.0},
            # The seam: the boundary syllable handed back a second time.
            {"start": 5.1, "end": 5.4, "text": "요", "compression_ratio": 1.0},
            # Seven seconds later it is someone speaking, not a seam.
            {"start": 12.0, "end": 12.4, "text": "요", "compression_ratio": 1.0},
        ]
    )
    assert [s["text"] for s in kept] == ["전화가 가게 되세요", "요"]
    assert kept[1]["start"] == 12.0


def test_a_blank_initial_prompt_is_none_and_a_sentence_is_kept() -> None:
    """The blanket block on sentence prompts went too far: a punctuated line is
    the strongest lever on output that reads as unspaced. Echo is handled by
    filtering a segment that is only the prompt."""
    assert usable_initial_prompt("  ") is None
    assert usable_initial_prompt(None) is None
    assert usable_initial_prompt(" 네, 고객님. ") == "네, 고객님."


def test_merge_hotwords_unions_server_and_job_terms() -> None:
    assert merge_hotwords("당근알바", "당근 당근페이") == "당근알바 당근 당근페이"
    assert merge_hotwords("당근알바", "당근알바 당근") == "당근알바 당근"
    assert merge_hotwords("", None) is None
