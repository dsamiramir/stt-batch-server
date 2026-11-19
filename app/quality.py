"""Prompt handling, hallucination filters and segment splits.

faster-whisper already restores timestamps when it does the VAD itself: it
concatenates speech, decodes that stream, then adds back the silence it removed
(``SpeechTimestampsMap``), so a 20s lead-in leaves the first segment at t≈20.
Slicing the audio ourselves and decoding each slice on its own bought nothing
and cost the model its context: a 0.6s slice gives it nothing to punctuate
with, which is where the run-on, comma-less Korean came from. So the engine
hands the whole channel to one ``transcribe`` call and the helpers here only
clean up what comes back.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

MAX_SEGMENT_SEC = 10.0
# Whisper's default. Above this the text is mostly a loop.
COMPRESSION_RATIO_LIMIT = 2.4
# Never a real line in these calls, and the old transcription prompt said it.
PROMPT_ECHO = "통화 녹취"
# A fragment this short that only repeats the previous line is a decode seam.
MAX_TAIL_CHARS = 3
MAX_TAIL_GAP_SEC = 0.2
_AFFIRMATION = re.compile(r"^[네예에요아응어]+$")
_SENTENCE_END = re.compile(r"[.?!。…]$")
_PUNCT_OR_SPACE = re.compile(r"[\s,.!?~…·、。]+")


def _compact(text: str | None) -> str:
    """Text with spacing and punctuation removed, for comparing wording only."""
    return _PUNCT_OR_SPACE.sub("", text or "")


def usable_initial_prompt(value: str | None) -> str | None:
    """Blank prompts become None so faster-whisper leaves the slot empty.

    The prompt is a style anchor: the model imitates its punctuation, which is
    the cheapest lever on Korean that reads as unspaced. It is also copied back
    as speech when a decode window holds little of it, so a segment that is
    only the prompt is dropped below rather than blocked here.
    """
    return (value or "").strip() or None


def merge_hotwords(*sources: str | None) -> str | None:
    """Union of the server's terms and the job's, server first, deduped.

    hotwords land in the decoder's "previous text" slot, the same place as the
    prompt, so they are a bias hint and not a mode a client switches: a job
    that sends its own terms still gets the ones the server knows misfire.
    Order is kept so the prompt tokens are reproducible across runs.
    """
    merged: list[str] = []
    for source in sources:
        for token in (source or "").split():
            if token not in merged:
                merged.append(token)
    return " ".join(merged) or None


def echo_phrases(*sources: str | None) -> frozenset[str]:
    """What counts as prompt echo: each hotword, and each whole prompt.

    Single syllables are left out. "네" as its own segment is a real answer far
    more often than it is an echo, and dropping it would cost more than it saves.
    """
    phrases: set[str] = set()
    for source in sources:
        text = (source or "").strip()
        if not text:
            continue
        whole = _compact(text)
        if len(whole) >= 2:
            phrases.add(whole)
        for token in text.split():
            compact = _compact(token)
            if len(compact) >= 2:
                phrases.add(compact)
    return frozenset(phrases)


def is_prompt_echo(text: str) -> bool:
    return PROMPT_ECHO in (text or "")


def is_echo(text: str, phrases: Iterable[str]) -> bool:
    """True when the whole segment is one prompt term, or that term repeated.

    Matching the whole segment and not a substring is deliberate: "당근알바
    공고를 올리려고요" is a customer talking, while a segment that is exactly
    "당근페이" is the decoder reading its own prompt back.
    """
    compact = _compact(text)
    if not compact:
        return False
    for phrase in phrases:
        if compact == phrase:
            return True
        if len(compact) > len(phrase) and len(compact) % len(phrase) == 0:
            if phrase * (len(compact) // len(phrase)) == compact:
                return True
    return False


def is_affirmation_loop(text: str) -> bool:
    """네/예 repeated until the segment is only that syllable."""
    compact = _compact(text)
    return len(compact) >= 4 and _AFFIRMATION.fullmatch(compact) is not None


def is_high_compression(ratio: float | None, limit: float = COMPRESSION_RATIO_LIMIT) -> bool:
    return ratio is not None and ratio > limit


def repeats_previous_tail(
    segment: dict[str, Any],
    text: str,
    previous: dict[str, Any],
    max_gap_sec: float = MAX_TAIL_GAP_SEC,
) -> bool:
    """A couple of syllables that only repeat the line just before them.

    Two decodes that meet at a word hand the boundary syllable back twice:
    "...전화가 가게 되세요" and then "요" on its own. A fragment that arrives
    later than the gap is someone speaking, not a seam, so it stays.
    """
    compact = _compact(text)
    if not compact or len(compact) > MAX_TAIL_CHARS or _SENTENCE_END.search(text):
        return False
    gap = float(segment.get("start") or 0.0) - float(previous.get("end") or 0.0)
    if gap > max_gap_sec:
        return False
    return _compact(str(previous.get("text") or "")).endswith(compact)


def _segment_from_words(segment: dict[str, Any], words: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(str(w.get("word") or "") for w in words).strip()
    out = dict(segment)
    out["start"] = round(float(words[0]["start"]), 3)
    out["end"] = round(float(words[-1]["end"]), 3)
    out["text"] = text
    out["words"] = words
    return out


def split_long_segment(segment: dict[str, Any], max_sec: float = MAX_SEGMENT_SEC) -> list[dict[str, Any]]:
    """Break a segment longer than ``max_sec`` on word boundaries.

    The piece's start is its first word, so a click seeks to that word rather
    than the start of a turn that runs on for tens of seconds.
    """
    words = list(segment.get("words") or [])
    if len(words) < 2:
        return [segment]
    start = float(words[0]["start"])
    end = float(words[-1]["end"])
    if end - start <= max_sec:
        return [_segment_from_words(segment, words)]

    pieces: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []
    bucket_start = float(words[0]["start"])
    for word in words:
        if not bucket:
            bucket_start = float(word["start"])
        word_end = float(word["end"])
        if bucket and word_end - bucket_start > max_sec:
            pieces.append(_segment_from_words(segment, bucket))
            bucket = []
            bucket_start = float(word["start"])
        bucket.append(word)
        tail = str(word.get("word") or "").strip()
        if (
            bucket
            and _SENTENCE_END.search(tail)
            and float(bucket[-1]["end"]) - bucket_start >= 4
        ):
            pieces.append(_segment_from_words(segment, bucket))
            bucket = []
    if bucket:
        pieces.append(_segment_from_words(segment, bucket))
    return [p for p in pieces if p.get("text")]


def prepare_segments(
    segments: list[dict[str, Any]],
    echo_terms: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Drop loops, prompt echo and seam fragments, then split what is still long.

    ``echo_terms`` comes from the prompt and hotwords the decode actually ran
    with, so the filter follows whatever is configured instead of a phrase
    frozen into this module.
    """
    phrases = frozenset(echo_terms)
    prepared: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        if is_prompt_echo(text) or is_echo(text, phrases) or is_affirmation_loop(text):
            continue
        ratio = segment.get("compression_ratio")
        if is_high_compression(float(ratio) if ratio is not None else None):
            continue
        if prepared and repeats_previous_tail(segment, text, prepared[-1]):
            continue
        prepared.extend(split_long_segment({**segment, "text": text}))
    return prepared
