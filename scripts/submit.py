"""Reference client: upload an audio file, poll until done, save the transcript.

    python scripts/submit.py call.wav --server http://192.168.0.10:8760 \
        --channel-names agent,customer --out call.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="audio file to transcribe")
    parser.add_argument("--server", default="http://127.0.0.1:8760")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--language", default=None, help="e.g. ko; omit to auto-detect")
    parser.add_argument("--channel-names", default=None, help="comma separated")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--client-ref", default="")
    parser.add_argument("--callback-url", default="")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON")
    parser.add_argument("--poll", type=float, default=10.0, help="seconds between polls")
    parser.add_argument(
        "--no-wait", action="store_true", help="submit and print the job id, then exit"
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"no such file: {args.audio}", file=sys.stderr)
        return 2

    headers = {"x-api-key": args.api_key} if args.api_key else {}
    options: dict[str, object] = {}
    if args.language:
        options["language"] = args.language
    if args.channel_names:
        options["channel_names"] = [c.strip() for c in args.channel_names.split(",")]
    if args.word_timestamps:
        options["word_timestamps"] = True

    base = args.server.rstrip("/")
    with httpx.Client(base_url=base, headers=headers, timeout=None) as client:
        with args.audio.open("rb") as fh:
            response = client.post(
                "/v1/jobs",
                files={"file": (args.audio.name, fh, "application/octet-stream")},
                data={
                    "options": json.dumps(options, ensure_ascii=False),
                    "callback_url": args.callback_url,
                    "client_ref": args.client_ref,
                    "priority": str(args.priority),
                },
            )
        response.raise_for_status()
        job = response.json()
        job_id = job["id"]
        print(f"queued {job_id}")

        gate = client.get("/v1/health").json()["gate"]
        if not gate["accepting_work"]:
            print(f"note: worker is holding off — {gate['reason']}")
        if args.no_wait:
            return 0

        last = ""
        while True:
            time.sleep(args.poll)
            job = client.get(f"/v1/jobs/{job_id}").json()
            line = f"{job['status']} {job['progress'] * 100:5.1f}% {job.get('stage') or ''}"
            if line != last:
                print(line.rstrip())
                last = line
            if job["status"] in ("done", "failed", "canceled"):
                break

        if job["status"] != "done":
            print(f"job {job['status']}: {job.get('error')}", file=sys.stderr)
            return 1

        result = client.get(f"/v1/jobs/{job_id}/result").json()

    out = args.out or args.audio.with_suffix(".stt.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{result['text']}\n")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
