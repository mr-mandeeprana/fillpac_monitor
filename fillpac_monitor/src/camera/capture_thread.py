"""
Camera capture: opens RTSP or USB sources, warms up, and (optionally) reads
frames on a dedicated background thread so the main loop always processes
the *latest* frame instead of an ever-growing backlog.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np


def is_rtsp_source(source: str) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


class FrameCaptureThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name="FrameCapture")
        self.cap = cap
        self.current_frame: Optional[np.ndarray] = None
        self.current_ret = False
        self.lock = threading.Lock()
        self.running = True
        self.frame_count = 0

    def run(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.current_frame = frame
                    self.current_ret = True
                    self.frame_count += 1
            else:
                with self.lock:
                    self.current_ret = False

    def get_frame(self):
        with self.lock:
            return self.current_ret, self.current_frame

    def stop(self) -> None:
        self.running = False
        self.join(timeout=2.0)


class CameraManager:
    """Wraps open/warmup/reconnect logic for RTSP or USB cameras."""

    def __init__(self, source: str, cfg: dict, logger):
        self.source = source
        self.cfg = cfg
        self.logger = logger
        self.rtsp_mode = is_rtsp_source(source)
        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[FrameCaptureThread] = None
        self.reconnect_count = 0

    def open(self):
        """Opens the camera and returns (cap, first_frame). Raises RuntimeError on failure."""
        if self.rtsp_mode:
            self.logger.info("Setting FFMPEG RTSP options")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.cfg.get(
                "ffmpeg_options",
                "rtsp_transport;tcp|stimeout;5000000|max_delay;200000|reorder_queue_size;64",
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self._safe_source_repr()}")

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.get("target_fps", 30))
        except Exception:
            pass

        self.logger.info("Warming up camera...")
        attempts = self.cfg.get("warmup_attempts", 15)
        delay = self.cfg.get("warmup_delay_sec", 0.15)
        for attempt in range(attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                self.logger.info("Camera ready after %d attempt(s)", attempt + 1)
                self.cap = cap
                return cap, frame
            time.sleep(delay)

        cap.release()
        raise RuntimeError("Camera warmed up but no frames were received")

    def start_threaded_capture(self) -> FrameCaptureThread:
        assert self.cap is not None, "call open() first"
        self.capture_thread = FrameCaptureThread(self.cap)
        self.capture_thread.start()
        time.sleep(0.5)
        self.logger.info("Threaded capture started")
        return self.capture_thread

    def reconnect(self, max_attempts: int = 5):
        """Attempt to reconnect; returns (cap, frame) or (None, None)."""
        delay = self.cfg.get("reconnect_delay_sec", 2.0)
        for attempt in range(1, max_attempts + 1):
            self.logger.warning("Reconnecting to camera (attempt %d/%d)...", attempt, max_attempts)
            try:
                cap, frame = self.open()
                self.reconnect_count += 1
                return cap, frame
            except RuntimeError as exc:
                self.logger.error("Reconnect attempt failed: %s", exc)
                time.sleep(delay)
        return None, None

    def release(self) -> None:
        if self.capture_thread is not None:
            self.capture_thread.stop()
        if self.cap is not None:
            self.cap.release()

    def _safe_source_repr(self) -> str:
        # Avoid leaking credentials embedded in rtsp:// URLs into logs.
        if "@" in self.source and self.rtsp_mode:
            scheme, rest = self.source.split("://", 1)
            _, host_part = rest.split("@", 1)
            return f"{scheme}://***:***@{host_part}"
        return self.source
