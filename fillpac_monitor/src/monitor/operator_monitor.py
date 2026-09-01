"""
OperatorMonitor: runs YOLO pose tracking + MediaPipe hand detail on each
frame, determines raw per-frame status, feeds it through the debounced
StatusStateMachine, and draws the annotated overlay frame.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from src.detection.hand_model import HAND_CONNECTIONS, HandModel
from src.detection.pose_model import (
    LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, NOSE, PoseModel, RIGHT_ELBOW,
    RIGHT_SHOULDER, RIGHT_WRIST,
)
from src.monitor.state_machine import StatusStateMachine

# BGR colors
NO_OPERATOR_COLOR = (0, 0, 255)
NOT_WORKING_COLOR = (0, 165, 255)
WORKING_COLOR = (0, 255, 0)
PRESENT_COLOR = (0, 255, 255)
OPERATOR_ZONE_COLOR = (255, 0, 255)
WORK_ZONE_COLOR = (255, 0, 0)

STATUS_COLORS = {
    "NO_OPERATOR": NO_OPERATOR_COLOR,
    "OPERATOR_PRESENT": PRESENT_COLOR,
    "NOT_WORKING": NOT_WORKING_COLOR,
    "WORKING": WORKING_COLOR,
}


def point_in_polygon(point, polygon) -> bool:
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def avg_movement_points(history: deque) -> Optional[float]:
    """Average per-frame landmark displacement across a window of 21-point sets."""
    if len(history) < 2:
        return None
    pts_list = list(history)
    diffs = []
    for i in range(1, len(pts_list)):
        prev = np.asarray(pts_list[i - 1], dtype=np.float32)
        curr = np.asarray(pts_list[i], dtype=np.float32)
        diffs.append(float(np.linalg.norm(curr - prev, axis=1).mean()))
    return float(np.mean(diffs)) if diffs else None


def downscale_for_inference(frame: np.ndarray, target_width: int):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0
    scale = target_width / w
    new_w, new_h = target_width, int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


class OperatorMonitor:
    def __init__(
        self,
        pose_model: PoseModel,
        hand_model: HandModel,
        operator_zone: np.ndarray,
        work_zone: np.ndarray,
        monitor_cfg: dict,
        display_cfg: dict,
        inference_width: int,
        keypoint_conf_threshold: int,
        hand_crop_padding: int,
        station_id: str,
        db_manager,
        logger,
    ):
        self.pose_model = pose_model
        self.hand_model = hand_model
        self.operator_zone = operator_zone
        self.work_zone = work_zone
        self.monitor_cfg = monitor_cfg
        self.display_cfg = display_cfg
        self.inference_width = inference_width
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.hand_crop_padding = hand_crop_padding
        self.logger = logger

        self.hand_movement_threshold = monitor_cfg.get("hand_movement_threshold_px", 4.0)
        self.history_window = monitor_cfg.get("history_window_frames", 8)
        self.min_history = monitor_cfg.get("min_history_for_judgement", 2)
        self.absence_timeout = monitor_cfg.get("operator_absence_timeout_sec", 2.0)

        self.hand_history = defaultdict(lambda: deque(maxlen=self.history_window))
        self.last_seen_operator: Optional[float] = None

        self.state_machine = StatusStateMachine(
            station_id=station_id,
            db_manager=db_manager,
            logger=logger,
            debounce_sec=monitor_cfg.get("status_debounce_sec", 1.5),
            heartbeat_sec=monitor_cfg.get("heartbeat_interval_sec", 60),
        )

        self.frame_times: deque = deque(maxlen=30)
        self.inference_times: deque = deque(maxlen=30)

    def update_zones(self, operator_zone: np.ndarray, work_zone: np.ndarray) -> None:
        self.operator_zone = operator_zone
        self.work_zone = work_zone

    def process_frame(self, frame: np.ndarray):
        """Returns (display_frame, confirmed_status_str)."""
        t_start = time.time()
        display_frame = frame.copy() if self.display_cfg.get("show_window", True) else frame
        orig_h, orig_w = frame.shape[:2]

        inference_frame, scale_factor = downscale_for_inference(frame, self.inference_width)

        t_inf_start = time.time()
        result = self.pose_model.track(inference_frame)
        self.inference_times.append((time.time() - t_inf_start) * 1000)

        now = time.time()
        operator_present = False
        raw_status = "NO_OPERATOR"
        track_id: Optional[int] = None
        metadata: dict = {}

        has_ids = result.boxes is not None and result.boxes.id is not None
        has_kpts = result.keypoints is not None

        if has_ids and has_kpts:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            kxy_all = result.keypoints.xy.cpu().numpy()
            kconf_all = result.keypoints.conf.cpu().numpy()

            for box, tid, kxy, kconf in zip(boxes, ids, kxy_all, kconf_all):
                x1, y1, x2, y2 = box / scale_factor
                kxy_scaled = kxy / scale_factor
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if self.display_cfg.get("show_window", True):
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (130, 130, 130), 1)

                if not point_in_polygon(center, self.operator_zone):
                    continue

                operator_present = True
                track_id = int(tid)
                self.last_seen_operator = now

                hand_entries = self._detect_hands_for_track(
                    frame, x1, y1, x2, y2, orig_w, orig_h, tid
                )
                hand_in_zone = any(h["in_zone"] for h in hand_entries)
                hand_moving = any(h["moving"] for h in hand_entries)

                raw_status = "WORKING" if (hand_in_zone and hand_moving) else "NOT_WORKING"
                metadata = {
                    "hands_detected": len(hand_entries),
                    "hand_in_zone": hand_in_zone,
                    "hand_moving": hand_moving,
                    "movements_px": [
                        round(h["movement"], 2) if h["movement"] is not None else None
                        for h in hand_entries
                    ],
                }

                if self.display_cfg.get("show_window", True):
                    color = STATUS_COLORS[raw_status]
                    self._draw_overlay_for_track(
                        display_frame, x1, y1, x2, y2, center, kxy_scaled, kconf,
                        hand_entries, raw_status, color, tid, hand_in_zone, hand_moving,
                    )
                break  # only track one operator per zone (first found)

        if not operator_present:
            if self.last_seen_operator is not None and now - self.last_seen_operator < self.absence_timeout:
                raw_status = "OPERATOR_PRESENT"
            else:
                raw_status = "NO_OPERATOR"

        confirmed_status = self.state_machine.observe(raw_status, track_id, metadata)

        if self.display_cfg.get("show_window", True):
            self._draw_zones_and_hud(display_frame, confirmed_status, orig_w, orig_h, scale_factor)

        self.frame_times.append((time.time() - t_start) * 1000)
        return display_frame, confirmed_status

    # ------------------------------------------------------------------ #
    def _detect_hands_for_track(self, frame, x1, y1, x2, y2, orig_w, orig_h, track_id):
        pad = self.hand_crop_padding
        cx1 = max(0, int(x1) - pad)
        cy1 = max(0, int(y1) - pad)
        cx2 = min(orig_w, int(x2) + pad)
        cy2 = min(orig_h, int(y2) + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        raw_entries = self.hand_model.detect_in_crop(crop, (cx1, cy1))

        hand_entries = []
        for entry in raw_entries:
            pts = entry["pts"]
            label = entry["label"]
            hand_key = (track_id, label)

            in_zone = any(point_in_polygon(pt, self.work_zone) for pt in pts)
            if in_zone:
                self.hand_history[hand_key].append(pts)
            else:
                self.hand_history[hand_key].clear()

            movement = avg_movement_points(self.hand_history[hand_key])
            moving = (
                in_zone
                and movement is not None
                and len(self.hand_history[hand_key]) >= self.min_history
                and movement > self.hand_movement_threshold
            )
            hand_entries.append({"pts": pts, "in_zone": in_zone, "moving": moving, "movement": movement, "label": label})

        return hand_entries

    # ------------------------------------------------------------------ #
    def _draw_overlay_for_track(
        self, display_frame, x1, y1, x2, y2, center, kxy_scaled, kconf,
        hand_entries, raw_status, color, track_id, hand_in_zone, hand_moving,
    ):
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(display_frame, center, 5, color, -1)

        if self.display_cfg.get("draw_skeleton", True):
            self._draw_skeleton(display_frame, kxy_scaled, kconf, color)

        if self.display_cfg.get("draw_hand_landmarks", True):
            for h in hand_entries:
                self._draw_hand(display_frame, h["pts"], color)

        label_txt = f"ID-{track_id}: {raw_status}"
        ly = max(25, int(y1) - 10)
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display_frame, (int(x1), ly - th - 8), (int(x1) + tw + 8, ly + 5), (0, 0, 0), -1)
        cv2.putText(display_frame, label_txt, (int(x1) + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        move_vals = ", ".join(
            f"{h['label'][0]}={h['movement']:.1f}px" if h["movement"] is not None else f"{h['label'][0]}=-"
            for h in hand_entries
        ) or "-"
        debug = (
            f"Hands={len(hand_entries)} InZone={'YES' if hand_in_zone else 'NO'} "
            f"Moving={'YES' if hand_moving else 'NO'} [{move_vals}]"
        )
        cv2.putText(display_frame, debug, (int(x1), int(y2) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_skeleton(frame, kxy_scaled, kconf, color):
        idx_map = {
            "nose": NOSE, "left_shoulder": LEFT_SHOULDER, "right_shoulder": RIGHT_SHOULDER,
            "left_elbow": LEFT_ELBOW, "right_elbow": RIGHT_ELBOW,
            "left_wrist": LEFT_WRIST, "right_wrist": RIGHT_WRIST,
        }
        pts = {}
        for name, idx in idx_map.items():
            pts[name] = PoseModel.valid_keypoint(kxy_scaled, kconf, idx, 0.40)

        connections = [
            ("nose", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("nose", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "right_shoulder"),
        ]
        for a, b in connections:
            if pts.get(a) is not None and pts.get(b) is not None:
                cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)

        radii = {"left_wrist": 10, "right_wrist": 10, "left_elbow": 8, "right_elbow": 8,
                 "left_shoulder": 8, "right_shoulder": 8, "nose": 6}
        for name, pt in pts.items():
            if pt is not None:
                r = radii.get(name, 6)
                cv2.circle(frame, pt, r, color, -1)
                cv2.circle(frame, pt, r, (255, 255, 255), 1)

    @staticmethod
    def _draw_hand(frame, points, color):
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)
        for i, pt in enumerate(points):
            r = 7 if i == 0 else 5
            cv2.circle(frame, pt, r, color, -1)
            cv2.circle(frame, pt, r, (255, 255, 255), 1)

    def _draw_zones_and_hud(self, display_frame, confirmed_status, orig_w, orig_h, scale_factor):
        cv2.polylines(display_frame, [self.operator_zone], True, OPERATOR_ZONE_COLOR, 2)
        cv2.polylines(display_frame, [self.work_zone], True, WORK_ZONE_COLOR, 3)

        cv2.putText(display_frame, "OPERATOR ZONE", tuple(self.operator_zone[0] + np.array([5, 22])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, OPERATOR_ZONE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(display_frame, "HAND WORK ZONE", tuple(self.work_zone[0] + np.array([5, 22])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WORK_ZONE_COLOR, 2, cv2.LINE_AA)

        color = STATUS_COLORS.get(confirmed_status, (255, 255, 255))
        cv2.rectangle(display_frame, (10, 10), (470, 68), (20, 20, 20), -1)
        cv2.putText(display_frame, confirmed_status.replace("_", " "), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)

        avg_inf = np.mean(self.inference_times) if self.inference_times else 0
        avg_frame = np.mean(self.frame_times) if self.frame_times else 0
        timing_str = (
            f"Frame: {avg_frame:.0f}ms | Inf: {avg_inf:.0f}ms | "
            f"Res: {int(orig_w/scale_factor)}x{int(orig_h/scale_factor)}"
        )
        cv2.putText(display_frame, timing_str, (20, display_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "Q=quit  R=redraw zones", (20, display_frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self) -> None:
        self.hand_model.close()
