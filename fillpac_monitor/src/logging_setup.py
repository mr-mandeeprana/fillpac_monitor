"""
Central logging configuration.

- Rotating file logs under logs/ (size-based rotation, kept N backups).
- Console output for interactive/dev runs.
- Optional DbLogHandler that mirrors WARNING+ records into dbo.SystemLogs,
  wired up separately in app.py once the DbManager is available (so logging
  never depends on the DB being reachable at import time).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class DbLogHandler(logging.Handler):
    """Forwards log records to the database via a DbManager instance.

    Only attached for WARNING and above by default, to keep DB write volume
    low; full detail always exists in the rotating file logs regardless.
    """

    def __init__(self, db_manager, station_id: str, level=logging.WARNING):
        super().__init__(level=level)
        self.db_manager = db_manager
        self.station_id = station_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.db_manager.enqueue_log(
                station_id=self.station_id,
                level=record.levelname,
                component=record.name,
                message=msg,
            )
        except Exception:
            # Never let logging itself crash the app.
            self.handleError(record)


def setup_logging(
    log_dir: str | Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 10,
    logger_name: str = "fillpac",
) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        # Already configured (e.g. re-entrant call) - don't duplicate handlers.
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "fillpac_monitor.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "fillpac_monitor.errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(fmt)
    logger.addHandler(error_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def attach_db_handler(logger: logging.Logger, db_manager, station_id: str) -> Optional[DbLogHandler]:
    if db_manager is None:
        return None
    handler = DbLogHandler(db_manager, station_id)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return handler
