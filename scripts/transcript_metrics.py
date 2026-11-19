"""Read-only quality metrics over the transcripts already on disk.

The 9/28 regression - per-slice decoding, hotwords in the prompt slot - showed
up in five numbers before anyone read a transcript: segments that end in
punctuation halved, two-syllable fragments went from 3% to 24%, sub-second
segments from 5% to 40%, mean segment length from 8.9s to 2.5s, and segments
that were nothing but a hotword appeared at all. Run this after any change to
the decode path, and before/after an A/B on the VAD window or the prompt.

    python scripts/transcript_metrics.py --by week
    python scripts/transcript_metrics.py --days 2 --sample 200
    python scripts/transcript_metrics.py --by all --json

Nothing here touches the server, the GPU or the database: it only reads
data/results/*.json, so it is safe to run while jobs are processing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent

HANGUL = re.compile(r"[가-힣]")
SENTENCE_END = re.compile(r"[.?!。…]\s*$")
PUNCT_OR_SPACE = re.compile(r"[\s,.!?~…·、。]+")
# Longer than this and a segment is no longer a fragment, whatever it says.
FRAGMENT_HANGUL = 2


def compact(text: str) -> str:
    return PUNCT_OR_SPACE.sub("", text or "")


def echo_terms(raw: str | None) -> frozenset[str]:
    """Hotwords to look for as whole segments; the server's list by default."""
    if raw is None:
        try:
            sys.path.insert(0, str(BASE_DIR))
            from app.config import settings  # noqa: PLC0415

            raw = settings.hotwords
        except Exception:  # noqa: BLE001 - metrics must run without the app
            return frozenset()
    return frozenset(t for t in (compact(p) for p in raw.split()) if len(t) >= 2)


class Bucket:
    def __init__(self) -> None:
        self.files = 0
        self.segments = 0
        self.punct_end = 0
        self.fragments = 0
        self.sub1s = 0
        self.over10s = 0
        self.seconds = 0.0
        self.chars = 0
        self.spaces = 0
        self.hotword_only = 0

    def add_segment(self, segment: dict[str, Any], terms: frozenset[str]) -> None:
        text = str(segment.get("text") or "").strip()
        if not text or not HANGUL.search(text):
            return
        self.segments += 1
        duration = float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)
        self.seconds += duration
        self.chars += len(text)
        self.spaces += text.count(" ")
        ends_sentence = bool(SENTENCE_END.search(text))
        if ends_sentence:
            self.punct_end += 1
        if len(HANGUL.findall(text)) <= FRAGMENT_HANGUL and not ends_sentence:
            self.fragments += 1
        if duration < 1.0:
            self.sub1s += 1
        if duration >= 10.0:
            self.over10s += 1
        bare = compact(text)
        if bare in terms or any(
            len(bare) > len(t) and len(bare) % len(t) == 0
            and t * (len(bare) // len(t)) == bare
            for t in terms
        ):
            self.hotword_only += 1

    def report(self) -> dict[str, Any]:
        n = max(self.segments, 1)
        return {
            "files": self.files,
            "segments": self.segments,
            "punct_end_pct": round(100.0 * self.punct_end / n, 1),
            "fragment_pct": round(100.0 * self.fragments / n, 1),
            "sub1s_pct": round(100.0 * self.sub1s / n, 1),
            "over10s_pct": round(100.0 * self.over10s / n, 1),
            "mean_segment_sec": round(self.seconds / n, 2),
            "chars_per_space": round(self.chars / self.spaces, 2) if self.spaces else None,
            "hotword_only": self.hotword_only,
        }


def bucket_key(when: datetime, by: str) -> str:
    if by == "day":
        return when.strftime("%Y-%m-%d")
    if by == "week":
        return when.strftime("%G-W%V")
    if by == "month":
        return when.strftime("%Y-%m")
    return "all"


def collect(paths: list[Path], by: str, terms: frozenset[str]) -> dict[str, Bucket]:
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
            continue
        bucket = buckets[bucket_key(datetime.fromtimestamp(path.stat().st_mtime), by)]
        bucket.files += 1
        for segment in result.get("segments", []):
            bucket.add_segment(segment, terms)
    return buckets


COLUMNS = (
    ("files", "files", "{:>6}"),
    ("segments", "segs", "{:>7}"),
    ("punct_end_pct", "punct%", "{:>7}"),
    ("fragment_pct", "frag%", "{:>6}"),
    ("sub1s_pct", "<1s%", "{:>6}"),
    ("over10s_pct", ">=10s%", "{:>7}"),
    ("mean_segment_sec", "mean s", "{:>7}"),
    ("chars_per_space", "ch/sp", "{:>6}"),
    ("hotword_only", "echo", "{:>5}"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=BASE_DIR / "data" / "results")
    parser.add_argument("--days", type=float, help="Only files modified this recently.")
    parser.add_argument("--sample", type=int, help="Read at most this many files.")
    parser.add_argument("--by", choices=("day", "week", "month", "all"), default="day")
    parser.add_argument(
        "--terms",
        help="Hotwords to count as echo when a segment is only that term. "
             "Defaults to the server's configured list.",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args()

    paths = sorted(args.results.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        print(f"no transcripts under {args.results}", file=sys.stderr)
        return 1
    if args.days:
        cutoff = (datetime.now() - timedelta(days=args.days)).timestamp()
        paths = [p for p in paths if p.stat().st_mtime >= cutoff]
    if args.sample and len(paths) > args.sample:
        random.seed(7)  # same sample across runs, so two runs are comparable
        paths = sorted(random.sample(paths, args.sample), key=lambda p: p.stat().st_mtime)

    terms = echo_terms(args.terms)
    buckets = collect(paths, args.by, terms)
    reports = {key: buckets[key].report() for key in sorted(buckets)}

    if args.json:
        print(json.dumps({"terms": sorted(terms), "buckets": reports}, ensure_ascii=False, indent=2))
        return 0

    label_width = max(len(k) for k in reports) if reports else 10
    header = f"{'bucket':{label_width}} " + " ".join(
        fmt.format(title) for _, title, fmt in COLUMNS
    )
    print(f"{len(paths)} transcripts, echo terms: {' '.join(sorted(terms)) or '(none)'}")
    print(header)
    print("-" * len(header))
    for key, report in reports.items():
        cells = " ".join(
            fmt.format("-" if report[field] is None else report[field])
            for field, _, fmt in COLUMNS
        )
        print(f"{key:{label_width}} {cells}")
    print(
        "\npunct% = segments ending in . ? !   frag% = <=2 hangul chars, no sentence end"
        "\necho = segments that were nothing but a hotword"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
