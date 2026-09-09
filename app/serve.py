"""Entry point for unattended runs: ``python -m app.serve``.

Owning the log file here rather than shell-redirecting into it avoids two traps
on Windows: PowerShell writes redirected output as UTF-16 and wraps native
stderr in error records, and `cmd /c` mangles quotes in paths that contain
spaces. It also gives us real size-based rotation instead of a manual rename.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
MAX_BYTES = 20 * 1024 * 1024
BACKUPS = 3


def configure_logging(log_path: Path, level: int = logging.INFO) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [file_handler, console]

    # uvicorn installs its own handlers; strip them so everything lands in ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    # The dashboard polls a few times a second; access lines would drown the
    # log. Job submissions, gate overrides and settings changes are logged by
    # the application itself, so the audit trail survives.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main() -> None:
    import uvicorn

    from .config import settings
    from .main import app

    settings.ensure_dirs()
    configure_logging(settings.data_dir / "server.log")

    log = logging.getLogger("stt.serve")
    log.info("=== starting on %s:%s ===", settings.host, settings.port)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        access_log=False,
        # Keep uvicorn from replacing the logging we just set up.
        log_config=None,
    )


if __name__ == "__main__":
    main()
