"""Thin wrapper around Ultralytics YOLO pose for person tracking + keypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

# COCO-17 keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10


class PoseModel:
    def __init__(self, model_path: str | Path, device: str, conf: float, iou: float, logger):
        self.logger = logger
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Place the .pt weights there "
                f"or update models.pose_model_path in config.yaml."
            )
        self.logger.info("Loading YOLO pose model from %s", model_path)
        self.model = YOLO(str(model_path))
        self.model.overrides["conf"] = conf
        self.model.overrides["iou"] = iou
        self.model.overrides["device"] = device
        self.model.overrides["verbose"] = False
        self.logger.info("YOLO pose model loaded (device=%s)", device)

    def track(self, inference_frame: np.ndarray):
        """Returns the raw ultralytics Results object for a single frame."""
        return self.model.track(inference_frame, persist=True, classes=[0], verbose=False)[0]

    @staticmethod
    def valid_keypoint(kxy, kconf, index, conf_threshold):
        if index >= len(kxy) or kconf[index] < conf_threshold:
            return None
        return int(kxy[index][0]), int(kxy[index][1])
