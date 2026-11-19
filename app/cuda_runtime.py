"""Expose project-local NVIDIA wheels to Windows' DLL loader."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

# Closing these handles removes the directories from the DLL search path.
_dll_handles: dict[str, Any] = {}


def configure_cuda_runtime() -> None:
    if sys.platform != "win32":
        return

    for package, component in (
        ("nvidia-cublas-cu12", "cublas"),
        ("nvidia-cudnn-cu12", "cudnn"),
    ):
        try:
            dist = distribution(package)
        except PackageNotFoundError:
            continue  # CPU-only installations can still run.
        directory = Path(dist.locate_file(f"nvidia/{component}/bin"))
        if not directory.is_dir():
            continue
        path = str(directory.resolve())
        if path in _dll_handles:
            continue
        _dll_handles[path] = os.add_dll_directory(path)
        # CTranslate2 also loads libraries with LoadLibrary, which uses PATH.
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
