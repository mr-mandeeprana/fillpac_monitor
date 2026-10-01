"""Thin wrapper around MediaPipe HandLandmarker for finger-level detail."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

# Standard 21-point hand landmark connections (thumb/index/middle/ring/pinky + palm).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
)


def ensure_hand_landmarker_model(model_path: str | Path, model_url: str, logger) -> None:
    model_path = Path(model_path)
    if model_path.exists():
        return
    logger.info("HandLandmarker model not found, downloading to %s", model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(model_url, str(model_path))
        logger.info("HandLandmarker model download complete")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download hand_landmarker.task ({exc}). If this machine has no "
            f"internet access, download it manually from {model_url} and place it at "
            f"{model_path}."
        ) from exc


class HandModel:
    def __init__(
        self,
        model_path: str | Path,
        model_url: str,
        max_hands: int,
        detection_conf: float,
        tracking_conf: float,
        logger,
    ):
        self.logger = logger
        ensure_hand_landmarker_model(model_path, model_url, logger)
        base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=tracking_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.logger.info("MediaPipe HandLandmarker loaded")

    def detect_in_crop(self, crop_bgr: np.ndarray, offset_xy: tuple[int, int]):
        """Detects hands in a cropped BGR image, returns list of dicts with
        21 (x, y) points already translated back to full-frame pixel coords,
        plus the handedness label."""
        if crop_bgr.size == 0:
            return []

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        crop_h, crop_w = crop_bgr.shape[:2]
        ox, oy = offset_xy
        handedness_list = result.handedness or []

        entries = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            if i < len(handedness_list) and handedness_list[i]:
                label = handedness_list[i][0].category_name
            else:
                label = f"hand{i}"

            pts = [
                (int(lm.x * crop_w) + ox, int(lm.y * crop_h) + oy)
                for lm in hand_landmarks
            ]
            entries.append({"pts": pts, "label": label})

        return entries

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass
