"""Rewriting .env in place, preserving comments and key order.

The dashboard saves settings here so they survive a restart. Only the keys we
are handed get touched; everything else in the file is copied through verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

# Matches an active assignment only - a commented-out line never starts with an
# identifier, so "# STT_INITIAL_PROMPT=..." is left alone.
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")

_NEEDS_QUOTING = re.compile(r"[\s#'\"]")


def format_value(value: Any) -> str:
    """Render a Python value the way python-dotenv will read it back."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = "" if value is None else str(value)
    if text == "" or _NEEDS_QUOTING.search(text):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def update_env_file(path: Path, updates: Mapping[str, Any]) -> None:
    """Apply ``updates`` to the env file at ``path``, creating it if needed."""
    if not updates:
        return

    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = {key: format_value(val) for key, val in updates.items()}

    out: list[str] = []
    for line in existing:
        match = _ASSIGNMENT.match(line)
        key = match.group(1) if match else None
        if key is not None and key in pending:
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)

    if pending:
        if out and out[-1].strip():
            out.append("")
        out.append("# --- 대시보드에서 저장한 값 ---")
        out.extend(f"{key}={value}" for key, value in pending.items())

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target then swap, so a crash cannot leave a half-file.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(path)
