"""GPU utilisation probing via nvidia-smi.

Used to hold jobs back while the machine is busy with something else. If
nvidia-smi is unavailable the probe reports "unknown" and the caller decides;
the worker treats unknown as "allowed" so a CPU-only box still drains its queue.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

_QUERY = "index,utilization.gpu,memory.used,memory.total"
_CACHE_TTL = 2.0
_cache: tuple[float, list["GpuStat"] | None] = (0.0, None)


@dataclass(frozen=True)
class GpuStat:
    index: int
    util_pct: int
    used_mb: int
    total_mb: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


def _creationflags() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def query_gpus(use_cache: bool = True) -> list[GpuStat] | None:
    """Snapshot of every visible GPU, or ``None`` when nvidia-smi is missing."""
    global _cache
    now = time.monotonic()
    if use_cache and _cache[1] is not None and now - _cache[0] < _CACHE_TTL:
        return _cache[1]

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_creationflags(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    stats: list[GpuStat] = []
    for line in proc.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            stats.append(
                GpuStat(
                    index=int(fields[0]),
                    util_pct=int(float(fields[1])),
                    used_mb=int(float(fields[2])),
                    total_mb=int(float(fields[3])),
                )
            )
        except ValueError:
            continue
    _cache = (now, stats)
    return stats


def get_gpu(index: int) -> GpuStat | None:
    stats = query_gpus()
    if not stats:
        return None
    for stat in stats:
        if stat.index == index:
            return stat
    return stats[0]


def check_idle(
    index: int,
    max_util: int,
    min_free_mb: int,
    samples: int = 3,
    interval_sec: float = 1.0,
    require_free_vram: bool = True,
) -> tuple[bool, str]:
    """Sample the GPU a few times; idle only if *every* sample is under budget.

    ``require_free_vram`` is turned off once our own model is resident, since
    that memory is already ours and would otherwise permanently fail the check.
    """
    observed: list[GpuStat] = []
    for i in range(max(1, samples)):
        stat = get_gpu(index)
        if stat is None:
            return True, "nvidia-smi unavailable; GPU gate skipped"
        observed.append(stat)
        if stat.util_pct > max_util:
            return False, f"GPU util {stat.util_pct}% > {max_util}%"
        if require_free_vram and stat.free_mb < min_free_mb:
            return False, f"free VRAM {stat.free_mb}MB < {min_free_mb}MB"
        if i + 1 < samples:
            time.sleep(interval_sec)
            query_gpus(use_cache=False)

    peak = max(s.util_pct for s in observed)
    free = min(s.free_mb for s in observed)
    return True, f"GPU idle (peak util {peak}%, free {free}MB)"
