"""
StatusStateMachine turns noisy, per-frame status observations
(NO_OPERATOR / OPERATOR_PRESENT / NOT_WORKING / WORKING) into confirmed,
debounced state changes, and is responsible for emitting DB events:

- A "CHANGE" row is written only once a new status has been observed
  continuously for `debounce_sec` (avoids flooding the DB / flapping on
  single noisy frames).
- A "HEARTBEAT" row is written periodically while in a stable state, so
  a long unbroken WORKING period is still visible in the DB without
  waiting for the next transition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class _CandidateState:
    status: str
    track_id: Optional[int]
    since: float


class StatusStateMachine:
    def __init__(self, station_id: str, db_manager, logger, debounce_sec: float, heartbeat_sec: float):
        self.station_id = station_id
        self.db = db_manager
        self.logger = logger
        self.debounce_sec = debounce_sec
        self.heartbeat_sec = heartbeat_sec

        self.confirmed_status = "NO_OPERATOR"
        self.confirmed_track_id: Optional[int] = None
        self.confirmed_since = time.time()
        self._candidate: Optional[_CandidateState] = None
        self._last_heartbeat = time.time()

    def observe(self, status: str, track_id: Optional[int], metadata: Optional[dict] = None) -> str:
        """Feed one frame's raw observation. Returns the current *confirmed*
        status (which may lag `status` until it's been stable long enough)."""
        now = time.time()

        if status == self.confirmed_status:
            # Observation matches current confirmed state - reset any pending candidate.
            self._candidate = None
            self._maybe_heartbeat(now, metadata)
            return self.confirmed_status

        if self._candidate is None or self._candidate.status != status:
            self._candidate = _CandidateState(status=status, track_id=track_id, since=now)
            return self.confirmed_status

        # Same candidate persisting - check if it's been stable long enough to confirm.
        if now - self._candidate.since >= self.debounce_sec:
            self._confirm_transition(new_status=status, track_id=track_id, at=now, metadata=metadata)

        return self.confirmed_status

    def _confirm_transition(self, new_status: str, track_id: Optional[int], at: float, metadata: Optional[dict]):
        prev_status = self.confirmed_status
        prev_duration = at - self.confirmed_since

        self.logger.info(
            "Status change: %s -> %s (station=%s, track=%s, prev lasted %.1fs)",
            prev_status, new_status, self.station_id, track_id, prev_duration,
        )

        if self.db is not None:
            self.db.enqueue_status_event(
                station_id=self.station_id,
                status=new_status,
                event_type="CHANGE",
                track_id=track_id,
                prev_status=prev_status,
                prev_duration_sec=round(prev_duration, 2),
                metadata=metadata,
            )

        self.confirmed_status = new_status
        self.confirmed_track_id = track_id
        self.confirmed_since = at
        self._candidate = None
        self._last_heartbeat = at

    def _maybe_heartbeat(self, now: float, metadata: Optional[dict]) -> None:
        if now - self._last_heartbeat < self.heartbeat_sec:
            return
        self._last_heartbeat = now
        if self.db is not None:
            self.db.enqueue_status_event(
                station_id=self.station_id,
                status=self.confirmed_status,
                event_type="HEARTBEAT",
                track_id=self.confirmed_track_id,
                prev_status=None,
                prev_duration_sec=round(now - self.confirmed_since, 2),
                metadata=metadata,
            )
