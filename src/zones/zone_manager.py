"""
Handles loading/saving the working-zone polygon for each station.

Only ONE configurable zone is used:

    WORK ZONE

The complete camera frame is automatically considered the
operator-presence area. No operator zone needs to be configured.

Zones are saved as pixel coordinates together with the resolution
at which they were drawn and are automatically rescaled if the
camera resolution changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ZoneManager:

    def __init__(
        self,
        zones_dir: str | Path,
        station_id: str,
        logger
    ):
        self.zones_dir = Path(zones_dir)
        self.zones_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.station_id = station_id
        self.logger = logger

        self.file_path = (
            self.zones_dir / f"{station_id}.json"
        )

    # =========================================================
    # LOAD WORK ZONE
    # =========================================================

    def load(
        self,
        frame_shape: tuple[int, int]
    ) -> Optional[np.ndarray]:

        """
        Returns:

            work_zone -> int32 Nx2 array

        Returns None if no saved zone exists.
        """

        if not self.file_path.exists():
            return None

        try:
            data = json.loads(
                self.file_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as exc:

            self.logger.error(
                "Failed to read zone file %s: %s",
                self.file_path,
                exc
            )

            return None

        saved_h = data["frame_h"]
        saved_w = data["frame_w"]

        cur_h, cur_w = frame_shape

        work_zone = np.array(
            data["work_zone"],
            dtype=np.float32
        )

        # -----------------------------------------------------
        # RESCALE IF CAMERA RESOLUTION CHANGED
        # -----------------------------------------------------

        if (
            saved_h,
            saved_w
        ) != (
            cur_h,
            cur_w
        ):

            sx = cur_w / saved_w
            sy = cur_h / saved_h

            work_zone[:, 0] *= sx
            work_zone[:, 1] *= sy

            self.logger.info(
                "Rescaled working zone "
                "from %dx%d to %dx%d",
                saved_w,
                saved_h,
                cur_w,
                cur_h
            )

        return work_zone.astype(
            np.int32
        )

    # =========================================================
    # SAVE WORK ZONE
    # =========================================================

    def save(
        self,
        work_zone: np.ndarray,
        frame_shape: tuple[int, int]
    ) -> None:

        """
        Save only the working-zone polygon.
        """

        h, w = frame_shape

        data = {
            "station_id": self.station_id,
            "frame_h": h,
            "frame_w": w,
            "work_zone": np.asarray(
                work_zone
            ).tolist(),
        }

        self.file_path.write_text(
            json.dumps(
                data,
                indent=2
            ),
            encoding="utf-8"
        )

        self.logger.info(
            "Saved working zone to %s",
            self.file_path
        )

    # =========================================================
    # DEFAULT WORK ZONE
    # =========================================================

    def defaults_from_fractions(
        self,
        work_fractions: list,
        frame_shape: tuple[int, int]
    ) -> np.ndarray:

        """
        Create a default working zone from normalized
        coordinates.

        Example:

            [
                [0.35, 0.25],
                [0.75, 0.25],
                [0.75, 0.75],
                [0.35, 0.75]
            ]
        """

        h, w = frame_shape

        work_zone = (
            np.array(
                work_fractions,
                dtype=np.float32
            )
            * np.array(
                [w, h]
            )
        ).astype(np.int32)

        return work_zone

    # =========================================================
    # INTERACTIVE WORK-ZONE DRAWING
    # =========================================================

    @staticmethod
    def draw_interactive(
        frame: np.ndarray,
        window_name: str,
        color: tuple
    ) -> np.ndarray:

        """
        Draw ONLY the working-zone polygon.

        Controls:

            Left mouse click -> add point
            ENTER / SPACE    -> finish
            R                -> reset
            ESC              -> cancel
        """

        points: list[list[int]] = []

        def on_click(
            event,
            x,
            y,
            flags,
            param
        ):

            if event == cv2.EVENT_LBUTTONDOWN:

                points.append(
                    [x, y]
                )

        cv2.namedWindow(
            window_name
        )

        cv2.setMouseCallback(
            window_name,
            on_click
        )

        try:

            while True:

                display = frame.copy()

                # Draw points
                for p in points:

                    cv2.circle(
                        display,
                        tuple(p),
                        6,
                        color,
                        -1
                    )

                # Draw polygon
                if len(points) >= 2:

                    cv2.polylines(
                        display,
                        [
                            np.asarray(
                                points,
                                dtype=np.int32
                            )
                        ],
                        False,
                        color,
                        2
                    )

                cv2.putText(
                    display,
                    (
                        f"{window_name}: "
                        "click points | "
                        "ENTER=finish | "
                        "R=reset | "
                        "ESC=cancel"
                    ),
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    color,
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(
                    window_name,
                    display
                )

                key = cv2.waitKey(20) & 0xFF

                # Finish
                if (
                    key in (13, 32)
                    and len(points) >= 3
                ):
                    break

                # Reset
                if key == ord("r"):
                    points.clear()

                # Cancel
                if key == 27:

                    raise RuntimeError(
                        f"{window_name} "
                        "cancelled by user"
                    )

        finally:

            cv2.destroyWindow(
                window_name
            )

        return np.asarray(
            points,
            dtype=np.int32
        )