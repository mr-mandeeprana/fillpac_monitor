"""
Camera capture manager for RTSP and USB cameras.

Features:
- Explicit OpenCV FFmpeg backend for RTSP.
- Configurable FFmpeg RTSP options.
- Latest-frame background capture to minimize latency.
- Camera warm-up before starting inference.
- RTSP retry logic.
- Reconnect support.
- Safe RTSP logging without exposing credentials.
- Diagnostic logging of the actual RTSP source configuration.
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
    """Return True when source is an RTSP URL."""
    return isinstance(source, str) and source.lower().startswith("rtsp://")


class FrameCaptureThread(threading.Thread):
    """
    Background camera reader.

    Only the newest frame is retained. If inference is slower than the
    camera FPS, old frames are discarded instead of building a backlog.
    """

    def __init__(self, cap: cv2.VideoCapture, logger=None):
        super().__init__(daemon=True, name="FrameCapture")

        self.cap = cap
        self.logger = logger

        self.current_frame: Optional[np.ndarray] = None
        self.current_ret = False

        self.lock = threading.Lock()
        self.running = True

        self.frame_count = 0
        self.last_frame_time = 0.0

    def run(self) -> None:
        """Continuously read frames and keep only the newest frame."""

        while self.running:
            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.lock:
                        self.current_frame = frame
                        self.current_ret = True
                        self.frame_count += 1
                        self.last_frame_time = time.time()
                else:
                    with self.lock:
                        self.current_ret = False

                    # Prevent a tight CPU loop if the camera temporarily fails.
                    time.sleep(0.005)

            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "Camera capture thread read exception: %s",
                        exc,
                    )

                with self.lock:
                    self.current_ret = False

                time.sleep(0.01)

    def get_frame(self):
        """
        Return the newest available frame.

        Returns:
            tuple: (ret, frame)
        """

        with self.lock:
            return self.current_ret, self.current_frame

    def stop(self) -> None:
        """Stop the background capture thread."""

        self.running = False

        try:
            self.join(timeout=2.0)
        except RuntimeError:
            pass


class CameraManager:
    """
    Camera manager for RTSP and USB/local cameras.

    RTSP cameras are always opened explicitly through OpenCV FFmpeg.
    """

    def __init__(self, source: str, cfg: dict, logger):
        self.source = source
        self.cfg = cfg
        self.logger = logger

        self.rtsp_mode = is_rtsp_source(source)

        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[FrameCaptureThread] = None

        self.reconnect_count = 0

    # ==================================================================
    # CAMERA OPEN
    # ==================================================================

    def open(self):
        """
        Open the camera and wait until a real frame is received.

        Returns:
            (cap, first_frame)

        Raises:
            RuntimeError: if the camera cannot be opened or does not
                          provide frames after warmup attempts.
        """

        if self.rtsp_mode:
            cap = self._open_rtsp()
        else:
            cap = self._open_usb()

        if cap is None or not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {self._safe_source_repr()}"
            )

        # --------------------------------------------------------------
        # Camera properties
        # --------------------------------------------------------------

        try:
            # This is only a hint.
            # FFmpeg/network streams may ignore this property.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        target_fps = self.cfg.get("target_fps")

        if target_fps:
            try:
                cap.set(cv2.CAP_PROP_FPS, float(target_fps))
            except Exception:
                pass

        # --------------------------------------------------------------
        # Warm-up
        # --------------------------------------------------------------

        self.logger.info("Warming up camera...")

        attempts = int(
            self.cfg.get(
                "warmup_attempts",
                15,
            )
        )

        delay = float(
            self.cfg.get(
                "warmup_delay_sec",
                0.15,
            )
        )

        first_frame = None

        for attempt in range(1, attempts + 1):

            try:
                ret, frame = cap.read()

            except Exception as exc:

                ret = False
                frame = None

                self.logger.warning(
                    "Camera read exception during warmup "
                    "(attempt %d/%d): %s",
                    attempt,
                    attempts,
                    exc,
                )

            if ret and frame is not None:

                first_frame = frame

                h, w = frame.shape[:2]

                self.logger.info(
                    "Camera ready after %d attempt(s)",
                    attempt,
                )

                self.logger.info(
                    "Stream resolution: %dx%d",
                    w,
                    h,
                )

                self.cap = cap

                return cap, first_frame

            if attempt < attempts:
                time.sleep(delay)

        # --------------------------------------------------------------
        # No frame received
        # --------------------------------------------------------------

        try:
            cap.release()
        except Exception:
            pass

        raise RuntimeError(
            "Camera opened but no frames were received after "
            f"{attempts} warmup attempts: {self._safe_source_repr()}"
        )

    # ==================================================================
    # RTSP
    # ==================================================================

    def _open_rtsp(self) -> cv2.VideoCapture:
        """
        Open RTSP explicitly through OpenCV's FFmpeg backend.

        The standalone packaged RTSP test has already confirmed that
        OpenCV + FFmpeg + the RTSP stream work on the client machine.

        Therefore this method keeps the application RTSP opening path
        as close as possible to that successful test.
        """

        self.logger.info("Setting FFMPEG RTSP options")

        options = self.cfg.get(
            "ffmpeg_options",
            "rtsp_transport;tcp|stimeout;5000000|max_delay;200000|reorder_queue_size;64",
        )

        options = str(options).strip()

        # --------------------------------------------------------------
        # Configure OpenCV FFmpeg
        # --------------------------------------------------------------

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options

        self.logger.info(
            "FFmpeg RTSP options configured: %s",
            options,
        )

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------

        self.logger.info(
            "RTSP URL length: %d",
            len(self.source),
        )

        self.logger.info(
            "RTSP source: %s",
            self._safe_source_repr(),
        )

        self.logger.info(
            "RTSP credentials detected: %s",
            "yes" if "@" in self.source else "no",
        )

        if "@" in self.source:
            try:
                host_part = self.source.split("@", 1)[1]

                self.logger.info(
                    "RTSP host/path: %s",
                    host_part,
                )

            except Exception:
                pass

        # --------------------------------------------------------------
        # IMPORTANT:
        # Explicit CAP_FFMPEG is required.
        # --------------------------------------------------------------

        self.logger.info(
            "Opening RTSP using OpenCV FFmpeg backend..."
        )

        cap = cv2.VideoCapture(
            self.source,
            cv2.CAP_FFMPEG,
        )

        self.logger.info(
            "OpenCV VideoCapture isOpened=%s",
            cap.isOpened(),
        )

        if cap.isOpened():

            self.logger.info(
                "RTSP stream opened successfully with FFmpeg"
            )

            return cap

        # --------------------------------------------------------------
        # First attempt failed.
        # --------------------------------------------------------------

        self.logger.warning(
            "RTSP FFmpeg open failed. Releasing and retrying..."
        )

        try:
            cap.release()
        except Exception:
            pass

        time.sleep(0.25)

        # --------------------------------------------------------------
        # Retry
        # --------------------------------------------------------------

        self.logger.info(
            "Retrying RTSP using OpenCV FFmpeg backend..."
        )

        cap = cv2.VideoCapture(
            self.source,
            cv2.CAP_FFMPEG,
        )

        self.logger.info(
            "RTSP retry isOpened=%s",
            cap.isOpened(),
        )

        if cap.isOpened():

            self.logger.info(
                "RTSP stream opened successfully on retry"
            )

            return cap

        # --------------------------------------------------------------
        # Final failure
        # --------------------------------------------------------------

        try:
            cap.release()
        except Exception:
            pass

        self.logger.error(
            "RTSP stream could not be opened with OpenCV FFmpeg: %s",
            self._safe_source_repr(),
        )

        # IMPORTANT:
        # Return None rather than an empty VideoCapture object.
        # This allows open() to raise the correct RuntimeError.
        return None

    # ==================================================================
    # USB / LOCAL CAMERA
    # ==================================================================

    def _open_usb(self) -> cv2.VideoCapture:
        """Open a local USB/webcam source."""

        backend = (
            cv2.CAP_DSHOW
            if sys.platform == "win32"
            else cv2.CAP_ANY
        )

        self.logger.info(
            "Opening local camera using backend=%s",
            backend,
        )

        cap = cv2.VideoCapture(
            self.source,
            backend,
        )

        if cap.isOpened():
            return cap

        try:
            cap.release()
        except Exception:
            pass

        self.logger.warning(
            "Backend-specific camera open failed; "
            "trying OpenCV default backend"
        )

        cap = cv2.VideoCapture(self.source)

        return cap

    # ==================================================================
    # THREADED CAPTURE
    # ==================================================================

    def start_threaded_capture(self) -> FrameCaptureThread:
        """
        Start the latest-frame background capture thread.

        CameraManager.open() must be called first.
        """

        if self.cap is None:
            raise RuntimeError(
                "Camera is not open. Call open() before "
                "start_threaded_capture()."
            )

        # Avoid accidentally creating multiple reader threads.
        if (
            self.capture_thread is not None
            and self.capture_thread.is_alive()
        ):
            self.logger.warning(
                "Threaded capture is already running"
            )

            return self.capture_thread

        self.capture_thread = FrameCaptureThread(
            self.cap,
            logger=self.logger,
        )

        self.capture_thread.start()

        # Give the background reader a moment to obtain frames.
        time.sleep(0.1)

        self.logger.info(
            "Threaded capture started"
        )

        return self.capture_thread

    # ==================================================================
    # RECONNECT
    # ==================================================================

    def reconnect(self, max_attempts: int = 5):
        """
        Attempt to reconnect to the camera.

        Returns:
            (cap, frame)

        or:

            (None, None)
        """

        # --------------------------------------------------------------
        # Stop existing capture thread
        # --------------------------------------------------------------

        if self.capture_thread is not None:

            try:
                self.capture_thread.stop()
            except Exception:
                pass

            self.capture_thread = None

        # --------------------------------------------------------------
        # Release existing camera
        # --------------------------------------------------------------

        if self.cap is not None:

            try:
                self.cap.release()
            except Exception:
                pass

            self.cap = None

        delay = float(
            self.cfg.get(
                "reconnect_delay_sec",
                2.0,
            )
        )

        # --------------------------------------------------------------
        # Reconnect attempts
        # --------------------------------------------------------------

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            self.logger.warning(
                "Reconnecting to camera "
                "(attempt %d/%d)...",
                attempt,
                max_attempts,
            )

            try:

                cap, frame = self.open()

                self.reconnect_count += 1

                self.logger.info(
                    "Camera reconnect successful"
                )

                return cap, frame

            except RuntimeError as exc:

                self.logger.error(
                    "Reconnect attempt failed: %s",
                    exc,
                )

                if attempt < max_attempts:
                    time.sleep(delay)

        self.logger.error(
            "Camera reconnect failed after %d attempts",
            max_attempts,
        )

        return None, None

    # ==================================================================
    # RELEASE
    # ==================================================================

    def release(self) -> None:
        """Stop capture thread and release the camera."""

        if self.capture_thread is not None:

            try:
                self.capture_thread.stop()
            except Exception:
                pass

            self.capture_thread = None

        if self.cap is not None:

            try:
                self.cap.release()
            except Exception:
                pass

            self.cap = None

        self.logger.info(
            "Camera released"
        )

    # ==================================================================
    # SAFE LOGGING
    # ==================================================================

    def _safe_source_repr(self) -> str:
        """
        Return the camera URL without exposing credentials.

        Example:

            rtsp://***:***@172.20.45.131:554/video/live?channel=1&subtype=1
        """

        if (
            self.rtsp_mode
            and isinstance(self.source, str)
            and "@" in self.source
        ):

            try:

                scheme, rest = self.source.split(
                    "://",
                    1,
                )

                _, host_part = rest.split(
                    "@",
                    1,
                )

                return (
                    f"{scheme}://***:***@"
                    f"{host_part}"
                )

            except Exception:

                return "rtsp://***:***@<hidden>"

        return str(self.source)