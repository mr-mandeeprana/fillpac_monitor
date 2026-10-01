"""
DbManager: talks to SQL Server (SSMS-managed database) via pyodbc.

Design goals:
- Never block the video/inference loop. All writes go through an in-memory
  queue drained by a dedicated background thread that batches inserts.
- Survive DB being temporarily unreachable: queue keeps growing (bounded)
  and the writer thread retries with backoff; nothing crashes the main app.
- Two kinds of rows: status events (dbo.StatusEvents) and system logs
  (dbo.SystemLogs), plus periodic camera health pings (dbo.CameraHealth).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import pyodbc
except ImportError:  # pragma: no cover - allows running without DB installed
    pyodbc = None


@dataclass
class _QueuedRow:
    kind: str  # "status_event" | "log" | "health"
    payload: dict = field(default_factory=dict)


class DbManager:
    def __init__(self, settings: dict, logger):
        self.settings = settings
        self.logger = logger
        self.enabled = settings.get("enabled", True) and pyodbc is not None

        if settings.get("enabled", True) and pyodbc is None:
            self.logger.warning(
                "pyodbc is not installed - database logging disabled. "
                "Install with: pip install pyodbc"
            )

        self._queue: "queue.Queue[_QueuedRow]" = queue.Queue(
            maxsize=settings.get("write_queue_max_size", 5000)
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._conn = None
        self._connect_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="DbWriter")
        self._thread.start()
        self.logger.info("DbManager writer thread started")

    def stop(self, timeout: float = 5.0) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._close_conn()
        self.logger.info("DbManager stopped")

    # ------------------------------------------------------------------ #
    # Public enqueue API (non-blocking, called from the hot path)
    # ------------------------------------------------------------------ #
    def enqueue_status_event(
        self,
        station_id: str,
        status: str,
        event_type: str = "CHANGE",
        track_id: Optional[int] = None,
        prev_status: Optional[str] = None,
        prev_duration_sec: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if not self.enabled:
            return
        row = _QueuedRow(
            kind="status_event",
            payload=dict(
                station_id=station_id,
                track_id=track_id,
                status=status,
                event_type=event_type,
                event_time_utc=datetime.now(timezone.utc),
                prev_status=prev_status,
                prev_duration_sec=prev_duration_sec,
                metadata=json.dumps(metadata) if metadata else None,
            ),
        )
        self._safe_put(row)

    def enqueue_log(self, station_id: Optional[str], level: str, component: str, message: str) -> None:
        if not self.enabled:
            return
        row = _QueuedRow(
            kind="log",
            payload=dict(
                station_id=station_id,
                level=level,
                component=component,
                message=message,
                logged_at_utc=datetime.now(timezone.utc),
            ),
        )
        self._safe_put(row)

    def enqueue_health(
        self, station_id: str, is_connected: bool, consecutive_failures: int,
        reconnect_count: int, note: str = "",
    ) -> None:
        if not self.enabled:
            return
        row = _QueuedRow(
            kind="health",
            payload=dict(
                station_id=station_id,
                is_connected=is_connected,
                consecutive_failures=consecutive_failures,
                reconnect_count=reconnect_count,
                note=note,
                checked_at_utc=datetime.now(timezone.utc),
            ),
        )
        self._safe_put(row)

    def _safe_put(self, row: _QueuedRow) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Drop oldest-style backpressure: log once in a while, don't spam.
            self.logger.warning("DB write queue full - dropping row (kind=%s)", row.kind)

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    def _build_conn_str(self) -> str:
        s = self.settings
        parts = [
            f"DRIVER={s['driver']}",
            f"SERVER={s['server']}",
            f"DATABASE={s['database']}",
        ]
        if s.get("trusted_connection"):
            parts.append("Trusted_Connection=yes")
        else:
            parts.append(f"UID={s['user']}")
            parts.append(f"PWD={s['password']}")
        parts.append(f"Encrypt={'yes' if s.get('encrypt', True) else 'no'}")
        parts.append(f"TrustServerCertificate={'yes' if s.get('trust_server_certificate', True) else 'no'}")
        parts.append(f"Connection Timeout={s.get('connect_timeout_sec', 5)}")
        return ";".join(parts)

    def _get_conn(self):
        with self._connect_lock:
            if self._conn is not None:
                return self._conn
            conn_str = self._build_conn_str()
            self._conn = pyodbc.connect(conn_str, autocommit=False)
            self.logger.info("Connected to SQL Server database %s", self.settings.get("database"))
            return self._conn

    def _close_conn(self) -> None:
        with self._connect_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # ------------------------------------------------------------------ #
    # Background writer
    # ------------------------------------------------------------------ #
    def _writer_loop(self) -> None:
        flush_interval = self.settings.get("batch_flush_interval_sec", 2.0)
        max_rows = self.settings.get("batch_flush_max_rows", 200)
        backoffs = self.settings.get("retry_backoff_sec", [1, 2, 5, 10, 30])
        backoff_idx = 0

        buffer: list[_QueuedRow] = []
        last_flush = time.time()

        while not self._stop_event.is_set() or not self._queue.empty() or buffer:
            try:
                timeout = max(0.05, flush_interval - (time.time() - last_flush))
                row = self._queue.get(timeout=timeout)
                buffer.append(row)
            except queue.Empty:
                pass

            should_flush = (
                len(buffer) >= max_rows
                or (buffer and (time.time() - last_flush) >= flush_interval)
            )
            if should_flush and buffer:
                try:
                    self._flush(buffer)
                    buffer.clear()
                    last_flush = time.time()
                    backoff_idx = 0
                except Exception as exc:
                    self.logger.error("DB flush failed (%d rows buffered): %s", len(buffer), exc)
                    self._close_conn()
                    wait = backoffs[min(backoff_idx, len(backoffs) - 1)]
                    backoff_idx += 1
                    time.sleep(wait)

    def _flush(self, rows: list[_QueuedRow]) -> None:
        conn = self._get_conn()
        cur = conn.cursor()

        for row in rows:
            p = row.payload
            if row.kind == "status_event":
                cur.execute(
                    """
                    INSERT INTO dbo.StatusEvents
                        (StationID, TrackID, Status, EventType, EventTimeUtc,
                         PrevStatus, PrevDurationSec, Metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    p["station_id"], p["track_id"], p["status"], p["event_type"],
                    p["event_time_utc"], p["prev_status"], p["prev_duration_sec"], p["metadata"],
                )
            elif row.kind == "log":
                cur.execute(
                    """
                    INSERT INTO dbo.SystemLogs (StationID, LogLevel, Component, Message, LoggedAtUtc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    p["station_id"], p["level"], p["component"], p["message"], p["logged_at_utc"],
                )
            elif row.kind == "health":
                cur.execute(
                    """
                    INSERT INTO dbo.CameraHealth
                        (StationID, IsConnected, ConsecutiveFailures, ReconnectCount, Note, CheckedAtUtc)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    p["station_id"], p["is_connected"], p["consecutive_failures"],
                    p["reconnect_count"], p["note"], p["checked_at_utc"],
                )

        conn.commit()