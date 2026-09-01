"""
Handles loading/saving operator & work zone polygons per station, and the
interactive mouse-click zone-drawing UI. Zones are saved as pixel coordinates
alongside the resolution they were drawn at, and rescaled automatically if
the camera resolution later changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ZoneManager:
    def __init__(self, zones_dir: str | Path, station_id: str, logger):
        self.zones_dir = Path(zones_dir)
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        self.station_id = station_id
        self.logger = logger
        self.file_path = self.zones_dir / f"{station_id}.json"

    def load(self, frame_shape: tuple[int, int]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Returns (operator_zone, work_zone) as int32 arrays, rescaled to
        frame_shape (h, w) if the saved resolution differs. Returns None if
        no saved zones exist."""
        if not self.file_path.exists():
            return None

        try:
            data = json.loads(self.file_path.read_text())
        except Exception as exc:
            self.logger.error("Failed to read zone file %s: %s", self.file_path, exc)
            return None

        saved_h, saved_w = data["frame_h"], data["frame_w"]
        cur_h, cur_w = frame_shape

        operator_zone = np.array(data["operator_zone"], dtype=np.float32)
        work_zone = np.array(data["work_zone"], dtype=np.float32)

        if (saved_h, saved_w) != (cur_h, cur_w):
            sx, sy = cur_w / saved_w, cur_h / saved_h
            operator_zone[:, 0] *= sx
            operator_zone[:, 1] *= sy
            work_zone[:, 0] *= sx
            work_zone[:, 1] *= sy
            self.logger.info(
                "Rescaled saved zones from %dx%d to %dx%d", saved_w, saved_h, cur_w, cur_h
            )

        return operator_zone.astype(np.int32), work_zone.astype(np.int32)

    def save(self, operator_zone: np.ndarray, work_zone: np.ndarray, frame_shape: tuple[int, int]) -> None:
        h, w = frame_shape
        data = {
            "station_id": self.station_id,
            "frame_h": h,
            "frame_w": w,
            "operator_zone": np.asarray(operator_zone).tolist(),
            "work_zone": np.asarray(work_zone).tolist(),
        }
        self.file_path.write_text(json.dumps(data, indent=2))
        self.logger.info("Saved zones to %s", self.file_path)

    def defaults_from_fractions(
        self, operator_fractions: list, work_fractions: list, frame_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame_shape
        operator_zone = (np.array(operator_fractions, dtype=np.float32) * np.array([w, h])).astype(np.int32)
        work_zone = (np.array(work_fractions, dtype=np.float32) * np.array([w, h])).astype(np.int32)
        return operator_zone, work_zone

    @staticmethod
    def draw_interactive(frame: np.ndarray, window_name: str, color: tuple) -> np.ndarray:
        """Blocking interactive polygon draw. Returns int32 Nx2 array."""
        points: list[list[int]] = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_click)

        try:
            while True:
                display = frame.copy()
                for p in points:
                    cv2.circle(display, tuple(p), 6, color, -1)
                if len(points) >= 2:
                    cv2.polylines(display, [np.asarray(points, dtype=np.int32)], False, color, 2)
                cv2.putText(
                    display,
                    f"{window_name}: click points | ENTER=finish | R=reset | ESC=cancel",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA,
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in (13, 32) and len(points) >= 3:
                    break
                if key == ord("r"):
                    points.clear()
                if key == 27:
                    raise RuntimeError(f"{window_name} cancelled by user")
        finally:
            cv2.destroyWindow(window_name)

        return np.asarray(points, dtype=np.int32)
