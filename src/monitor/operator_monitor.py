"""
FillPac Operator Monitor

YOLO Pose:
    - Person detection/tracking
    - Left/right shoulder
    - Left/right elbow
    - Left/right wrist

MediaPipe:
    - Full 21-point left/right hand landmarks

Detection logic:
    FULL CAMERA FRAME
        -> PERSON DETECTED
        -> OPERATOR PRESENT
        -> SHOULDER -> ELBOW -> WRIST -> HAND
        -> HAND/WRIST IN WORK ZONE?
        -> MOVEMENT?
        -> WORKING / NOT WORKING

Status timing:
    WORKING:
        Immediate when valid hand/wrist movement is detected.

    NOT_WORKING:
        Only after no valid working movement for
        monitor.not_working_timeout_sec.

    NO_OPERATOR:
        After operator_absence_timeout_sec.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from src.detection.hand_model import HAND_CONNECTIONS, HandModel
from src.detection.pose_model import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    PoseModel,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from src.monitor.state_machine import StatusStateMachine


# =============================================================================
# COLORS - BGR
# =============================================================================

NO_OPERATOR_COLOR = (0, 0, 255)
NOT_WORKING_COLOR = (0, 165, 255)
WORKING_COLOR = (0, 255, 0)
PRESENT_COLOR = (0, 255, 255)

# Work zone
WORK_ZONE_COLOR = (255, 0, 0)

# Arm / pose colors
SHOULDER_COLOR = (255, 255, 0)
ELBOW_COLOR = (255, 200, 0)
WRIST_COLOR = (0, 255, 255)

# Hand
HAND_COLOR = (255, 0, 255)


STATUS_COLORS = {
    "NO_OPERATOR": NO_OPERATOR_COLOR,
    "OPERATOR_PRESENT": PRESENT_COLOR,
    "NOT_WORKING": NOT_WORKING_COLOR,
    "WORKING": WORKING_COLOR,
}


# =============================================================================
# HELPERS
# =============================================================================

def point_in_polygon(point, polygon) -> bool:
    """
    Return True if a point is inside or on the work-zone polygon.
    """
    if polygon is None or len(polygon) < 3:
        return False

    return (
        cv2.pointPolygonTest(
            polygon,
            (float(point[0]), float(point[1])),
            False,
        )
        >= 0
    )


def avg_movement_points(history: deque) -> Optional[float]:
    """
    Calculate average per-frame displacement across a sequence of
    21-point hand landmarks.
    """
    if len(history) < 2:
        return None

    pts_list = list(history)
    diffs = []

    for i in range(1, len(pts_list)):
        prev = np.asarray(pts_list[i - 1], dtype=np.float32)
        curr = np.asarray(pts_list[i], dtype=np.float32)

        if prev.shape != curr.shape:
            continue

        diffs.append(
            float(
                np.linalg.norm(
                    curr - prev,
                    axis=1,
                ).mean()
            )
        )

    return float(np.mean(diffs)) if diffs else None


def avg_movement_points_2d(history: deque) -> Optional[float]:
    """
    Calculate average movement for a sequence of single (x, y) points.
    Used as a wrist-movement fallback when MediaPipe hand detection
    temporarily fails.
    """
    if len(history) < 2:
        return None

    pts_list = list(history)
    diffs = []

    for i in range(1, len(pts_list)):
        prev = np.asarray(pts_list[i - 1], dtype=np.float32)
        curr = np.asarray(pts_list[i], dtype=np.float32)

        diffs.append(
            float(
                np.linalg.norm(curr - prev)
            )
        )

    return float(np.mean(diffs)) if diffs else None


def downscale_for_inference(
    frame: np.ndarray,
    target_width: int,
):
    """
    Downscale frame for YOLO inference while keeping the original
    frame for display and MediaPipe hand detection.
    """
    h, w = frame.shape[:2]

    if target_width <= 0 or w <= target_width:
        return frame, 1.0

    scale = target_width / w

    new_w = target_width
    new_h = int(h * scale)

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    return resized, scale


# =============================================================================
# OPERATOR MONITOR
# =============================================================================

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
        keypoint_conf_threshold: float,
        hand_crop_padding: int,
        station_id: str,
        db_manager,
        logger,
    ):
        self.pose_model = pose_model
        self.hand_model = hand_model

        # Kept for compatibility with current app.py.
        #
        # IMPORTANT:
        # Operator presence is now the FULL CAMERA FRAME.
        # This zone is NOT used for operator detection.
        self.operator_zone = operator_zone

        # Only manually configured zone.
        self.work_zone = work_zone

        self.monitor_cfg = monitor_cfg
        self.display_cfg = display_cfg

        self.inference_width = inference_width
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.hand_crop_padding = hand_crop_padding

        self.logger = logger

        # ---------------------------------------------------------------------
        # Movement settings
        # ---------------------------------------------------------------------

        self.hand_movement_threshold = float(
            monitor_cfg.get(
                "hand_movement_threshold_px",
                4.0,
            )
        )

        self.history_window = int(
            monitor_cfg.get(
                "history_window_frames",
                8,
            )
        )

        self.min_history = int(
            monitor_cfg.get(
                "min_history_for_judgement",
                2,
            )
        )

        # ---------------------------------------------------------------------
        # Operator absence
        # ---------------------------------------------------------------------

        self.absence_timeout = float(
            monitor_cfg.get(
                "operator_absence_timeout_sec",
                2.0,
            )
        )

        # ---------------------------------------------------------------------
        # NOT WORKING delay
        # ---------------------------------------------------------------------

        self.not_working_timeout = float(
            monitor_cfg.get(
                "not_working_timeout_sec",
                5.0,
            )
        )

        # ---------------------------------------------------------------------
        # Movement history
        # ---------------------------------------------------------------------

        # MediaPipe hand history:
        # (track_id, handedness) -> deque of 21-point sets
        self.hand_history = defaultdict(
            lambda: deque(
                maxlen=self.history_window
            )
        )

        # YOLO wrist history:
        # (track_id, left/right) -> deque of (x, y)
        self.wrist_history = defaultdict(
            lambda: deque(
                maxlen=self.history_window
            )
        )

        # Last time a valid operator was seen.
        self.last_seen_operator: Optional[float] = None

        # Last time the operator was confirmed to be working.
        #
        # This is what creates:
        #
        # WORKING
        #    |
        #    | no movement
        #    |
        #    | 5 seconds
        #    v
        # NOT_WORKING
        #
        self.last_working_activity: Optional[float] = None

        # Current working state used for timeout logic.
        self.current_working = False

        # ---------------------------------------------------------------------
        # Status state machine
        # ---------------------------------------------------------------------

        self.state_machine = StatusStateMachine(
            station_id=station_id,
            db_manager=db_manager,
            logger=logger,
            debounce_sec=monitor_cfg.get(
                "status_debounce_sec",
                0.0,
            ),
            heartbeat_sec=monitor_cfg.get(
                "heartbeat_interval_sec",
                60,
            ),
        )

        # ---------------------------------------------------------------------
        # Performance
        # ---------------------------------------------------------------------

        self.frame_times: deque = deque(maxlen=30)
        self.inference_times: deque = deque(maxlen=30)

    # =========================================================================
    # ZONES
    # =========================================================================

    def update_zones(
        self,
        operator_zone: np.ndarray,
        work_zone: np.ndarray,
    ) -> None:
        """
        Kept compatible with existing app.py.

        Operator zone is ignored for actual detection because the complete
        camera frame is the operator-presence area.
        """
        self.operator_zone = operator_zone
        self.work_zone = work_zone

    # =========================================================================
    # MAIN FRAME PROCESSING
    # =========================================================================

    def process_frame(self, frame: np.ndarray):
        """
        Process one frame.

        Returns:
            display_frame,
            confirmed_status
        """

        t_start = time.time()

        show_window = self.display_cfg.get(
            "show_window",
            True,
        )

        display_frame = (
            frame.copy()
            if show_window
            else frame
        )

        orig_h, orig_w = frame.shape[:2]

        # ---------------------------------------------------------------------
        # YOLO inference
        # ---------------------------------------------------------------------

        inference_frame, scale_factor = downscale_for_inference(
            frame,
            self.inference_width,
        )

        t_inf_start = time.time()

        result = self.pose_model.track(
            inference_frame
        )

        self.inference_times.append(
            (time.time() - t_inf_start) * 1000
        )

        now = time.time()

        # ---------------------------------------------------------------------
        # Default status
        # ---------------------------------------------------------------------

        operator_present = False

        raw_status = "NO_OPERATOR"

        track_id: Optional[int] = None

        metadata: dict = {}

        # ---------------------------------------------------------------------
        # YOLO result availability
        # ---------------------------------------------------------------------

        has_ids = (
            result.boxes is not None
            and result.boxes.id is not None
        )

        has_kpts = (
            result.keypoints is not None
        )

        # ---------------------------------------------------------------------
        # PERSON DETECTION
        # ---------------------------------------------------------------------

        if has_ids and has_kpts:

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            kxy_all = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            kconf_all = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

            # -------------------------------------------------------------
            # Select person
            # -------------------------------------------------------------

            for (
                box,
                tid,
                kxy,
                kconf,
            ) in zip(
                boxes,
                ids,
                kxy_all,
                kconf_all,
            ):

                # Convert YOLO inference coordinates back to original frame.
                x1, y1, x2, y2 = (
                    box / scale_factor
                )

                kxy_scaled = (
                    kxy / scale_factor
                )

                center = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                )

                # ---------------------------------------------------------
                # Person = operator.
                #
                # NO OPERATOR ZONE CHECK.
                #
                # Entire camera frame is operator area.
                # ---------------------------------------------------------

                operator_present = True

                track_id = int(tid)

                self.last_seen_operator = now

                # ---------------------------------------------------------
                # Draw person bounding box
                # ---------------------------------------------------------

                if show_window:
                    cv2.rectangle(
                        display_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (130, 130, 130),
                        1,
                    )

                # ---------------------------------------------------------
                # Extract YOLO arm keypoints
                # ---------------------------------------------------------

                arm_points = self._get_arm_points(
                    kxy_scaled,
                    kconf,
                )

                # ---------------------------------------------------------
                # Update wrist movement
                # ---------------------------------------------------------

                wrist_info = self._update_wrist_movement(
                    track_id,
                    arm_points,
                )

                # ---------------------------------------------------------
                # MediaPipe hands
                # ---------------------------------------------------------

                hand_entries = self._detect_hands_for_track(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    orig_w,
                    orig_h,
                    tid,
                )

                # ---------------------------------------------------------
                # Determine work-zone state
                # ---------------------------------------------------------

                hand_in_zone = any(
                    h["in_zone"]
                    for h in hand_entries
                )

                # Wrist can also establish work-zone presence.
                wrist_in_zone = any(
                    w["in_zone"]
                    for w in wrist_info
                )

                # Combined:
                # hand OR wrist inside work zone.
                arm_in_work_zone = (
                    hand_in_zone
                    or wrist_in_zone
                )

                # ---------------------------------------------------------
                # Movement
                # ---------------------------------------------------------

                hand_moving = any(
                    h["moving"]
                    for h in hand_entries
                )

                wrist_moving = any(
                    w["moving"]
                    for w in wrist_info
                )

                arm_moving = (
                    hand_moving
                    or wrist_moving
                )

                # ---------------------------------------------------------
                # Working condition
                # ---------------------------------------------------------

                valid_working_activity = (
                    arm_in_work_zone
                    and arm_moving
                )

                # ---------------------------------------------------------
                # Immediate WORKING
                # ---------------------------------------------------------

                if valid_working_activity:

                    self.last_working_activity = now
                    self.current_working = True

                    raw_status = "WORKING"

                else:

                    # -----------------------------------------------------
                    # No valid movement.
                    #
                    # Do NOT immediately say NOT_WORKING.
                    #
                    # Wait for not_working_timeout_sec.
                    # -----------------------------------------------------

                    if (
                        self.last_working_activity is not None
                        and (
                            now
                            - self.last_working_activity
                            < self.not_working_timeout
                        )
                    ):

                        # Keep previous working state during grace period.
                        if self.current_working:
                            raw_status = "WORKING"
                        else:
                            raw_status = "NOT_WORKING"

                    else:

                        self.current_working = False

                        raw_status = "NOT_WORKING"

                # ---------------------------------------------------------
                # Metadata
                # ---------------------------------------------------------

                metadata = {
                    "hands_detected": len(
                        hand_entries
                    ),

                    "hand_in_zone": hand_in_zone,

                    "wrist_in_zone": wrist_in_zone,

                    "arm_in_work_zone": arm_in_work_zone,

                    "hand_moving": hand_moving,

                    "wrist_moving": wrist_moving,

                    "arm_moving": arm_moving,

                    "movements_px": [
                        (
                            round(
                                h["movement"],
                                2,
                            )
                            if h["movement"] is not None
                            else None
                        )
                        for h in hand_entries
                    ],

                    "wrist_movements_px": [
                        (
                            round(
                                w["movement"],
                                2,
                            )
                            if w["movement"] is not None
                            else None
                        )
                        for w in wrist_info
                    ],

                    "not_working_timeout_sec": (
                        self.not_working_timeout
                    ),
                }

                # ---------------------------------------------------------
                # Draw everything
                # ---------------------------------------------------------

                if show_window:

                    color = STATUS_COLORS.get(
                        raw_status,
                        PRESENT_COLOR,
                    )

                    self._draw_overlay_for_track(
                        display_frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        center,
                        kxy_scaled,
                        kconf,
                        arm_points,
                        hand_entries,
                        wrist_info,
                        raw_status,
                        color,
                        tid,
                        arm_in_work_zone,
                        arm_moving,
                    )

                # ---------------------------------------------------------
                # Current system is designed for one operator.
                # ---------------------------------------------------------

                break

        # =========================================================================
        # NO OPERATOR / TEMPORARY LOSS
        # =========================================================================

        if not operator_present:

            if (
                self.last_seen_operator is not None
                and (
                    now - self.last_seen_operator
                    < self.absence_timeout
                )
            ):

                raw_status = "OPERATOR_PRESENT"

            else:

                raw_status = "NO_OPERATOR"

                # Reset working state after operator is gone.
                self.current_working = False
                self.last_working_activity = None

        # =========================================================================
        # STATUS STATE MACHINE
        # =========================================================================

        confirmed_status = self.state_machine.observe(
            raw_status,
            track_id,
            metadata,
        )

        # =========================================================================
        # ZONE + HUD
        # =========================================================================

        if show_window:

            self._draw_zones_and_hud(
                display_frame,
                confirmed_status,
                orig_w,
                orig_h,
                scale_factor,
            )

        # =========================================================================
        # PERFORMANCE
        # =========================================================================

        self.frame_times.append(
            (time.time() - t_start) * 1000
        )

        return display_frame, confirmed_status

    # =========================================================================
    # YOLO ARM KEYPOINTS
    # =========================================================================

    def _get_arm_points(
        self,
        kxy_scaled,
        kconf,
    ):
        """
        Extract complete left/right arm.

        Returns:
            {
                "left": {
                    "shoulder": (x,y) or None,
                    "elbow": (x,y) or None,
                    "wrist": (x,y) or None,
                },
                "right": {
                    ...
                }
            }
        """

        return {
            "left": {
                "shoulder": PoseModel.valid_keypoint(
                    kxy_scaled,
                    kconf,
                    LEFT_SHOULDER,
                    self.keypoint_conf_threshold,
                ),
                "elbow": PoseModel.valid_keypoint(
                    kxy_scaled,
                    kconf,
                    LEFT_ELBOW,
                    self.keypoint_conf_threshold,
                ),
                "wrist": PoseModel.valid_keypoint(
                    kxy_scaled,
                    kconf,
                    LEFT_WRIST,
                    self.keypoint_conf_threshold,
                ),
            },

            "right": {
                "shoulder": PoseModel.valid_keypoint(
                    kxy_scaled,
                    kconf,
                    RIGHT_SHOULDER,
                    self.keypoint_conf_threshold,
                ),
                "elbow": PoseModel.valid_keypoint(
                    kxy_scaled,
                    kconf,
                    RIGHT_ELBOW,
                    self.keypoint_conf_threshold,
                ),
                "wrist": PoseModel.valid_keypoint(
                    kxy_scaled,
                    kconf,
                    RIGHT_WRIST,
                    self.keypoint_conf_threshold,
                ),
            },
        }

    # =========================================================================
    # WRIST MOVEMENT
    # =========================================================================

    def _update_wrist_movement(
        self,
        track_id,
        arm_points,
    ):
        """
        Track left/right wrist movement.

        This provides a fallback when MediaPipe does not detect
        the hand for a frame.
        """

        results = []

        for side in ("left", "right"):

            wrist = arm_points[side]["wrist"]

            history_key = (
                int(track_id),
                side,
            )

            if wrist is not None:

                self.wrist_history[
                    history_key
                ].append(wrist)

                movement = avg_movement_points_2d(
                    self.wrist_history[
                        history_key
                    ]
                )

                moving = (
                    movement is not None
                    and len(
                        self.wrist_history[
                            history_key
                        ]
                    ) >= self.min_history
                    and movement
                    > self.hand_movement_threshold
                )

                in_zone = point_in_polygon(
                    wrist,
                    self.work_zone,
                )

                results.append(
                    {
                        "side": side,
                        "wrist": wrist,
                        "in_zone": in_zone,
                        "movement": movement,
                        "moving": (
                            in_zone
                            and moving
                        ),
                    }
                )

            else:

                # Do not immediately clear history.
                # A temporary missed keypoint should not destroy movement
                # detection.
                results.append(
                    {
                        "side": side,
                        "wrist": None,
                        "in_zone": False,
                        "movement": None,
                        "moving": False,
                    }
                )

        return results

    # =========================================================================
    # HAND DETECTION
    # =========================================================================

    def _detect_hands_for_track(
        self,
        frame,
        x1,
        y1,
        x2,
        y2,
        orig_w,
        orig_h,
        track_id,
    ):

        pad = self.hand_crop_padding

        cx1 = max(
            0,
            int(x1) - pad,
        )

        cy1 = max(
            0,
            int(y1) - pad,
        )

        cx2 = min(
            orig_w,
            int(x2) + pad,
        )

        cy2 = min(
            orig_h,
            int(y2) + pad,
        )

        crop = frame[
            cy1:cy2,
            cx1:cx2,
        ]

        raw_entries = self.hand_model.detect_in_crop(
            crop,
            (cx1, cy1),
        )

        hand_entries = []

        for entry in raw_entries:

            pts = entry["pts"]
            label = entry["label"]

            hand_key = (
                int(track_id),
                label,
            )

            # -------------------------------------------------------------
            # Work zone
            # -------------------------------------------------------------

            in_zone = any(
                point_in_polygon(
                    pt,
                    self.work_zone,
                )
                for pt in pts
            )

            # -------------------------------------------------------------
            # Movement history
            # -------------------------------------------------------------

            if in_zone:

                self.hand_history[
                    hand_key
                ].append(pts)

            else:

                # Keep a little history rather than immediately deleting
                # movement information. This helps prevent flickering.
                #
                # However, don't let an old hand position create WORKING
                # indefinitely.
                if len(
                    self.hand_history[
                        hand_key
                    ]
                ) > 2:

                    self.hand_history[
                        hand_key
                    ].popleft()

            movement = avg_movement_points(
                self.hand_history[
                    hand_key
                ]
            )

            moving = (
                in_zone
                and movement is not None
                and len(
                    self.hand_history[
                        hand_key
                    ]
                ) >= self.min_history
                and movement
                > self.hand_movement_threshold
            )

            hand_entries.append(
                {
                    "pts": pts,
                    "in_zone": in_zone,
                    "moving": moving,
                    "movement": movement,
                    "label": label,
                }
            )

        return hand_entries

    # =========================================================================
    # DRAW PERSON / ARM / HAND
    # =========================================================================

    def _draw_overlay_for_track(
        self,
        display_frame,
        x1,
        y1,
        x2,
        y2,
        center,
        kxy_scaled,
        kconf,
        arm_points,
        hand_entries,
        wrist_info,
        raw_status,
        color,
        track_id,
        arm_in_work_zone,
        arm_moving,
    ):

        # ---------------------------------------------------------------------
        # Person box
        # ---------------------------------------------------------------------

        cv2.rectangle(
            display_frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )

        cv2.circle(
            display_frame,
            center,
            5,
            color,
            -1,
        )

        # ---------------------------------------------------------------------
        # Full arm skeleton
        # ---------------------------------------------------------------------

        if self.display_cfg.get(
            "draw_skeleton",
            True,
        ):

            self._draw_full_arm_skeleton(
                display_frame,
                arm_points,
                color,
            )

        # ---------------------------------------------------------------------
        # Full hands
        # ---------------------------------------------------------------------

        if self.display_cfg.get(
            "draw_hand_landmarks",
            True,
        ):

            for hand in hand_entries:

                hand_color = (
                    WORKING_COLOR
                    if hand["in_zone"]
                    else HAND_COLOR
                )

                self._draw_hand(
                    display_frame,
                    hand["pts"],
                    hand_color,
                )

        # ---------------------------------------------------------------------
        # Status label
        # ---------------------------------------------------------------------

        label_txt = (
            f"ID-{track_id}: {raw_status}"
        )

        ly = max(
            25,
            int(y1) - 10,
        )

        (
            tw,
            th,
        ), _ = cv2.getTextSize(
            label_txt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        cv2.rectangle(
            display_frame,
            (
                int(x1),
                ly - th - 8,
            ),
            (
                int(x1) + tw + 8,
                ly + 5,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            display_frame,
            label_txt,
            (
                int(x1) + 4,
                ly,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------------------
        # Debug information
        # ---------------------------------------------------------------------

        move_vals = ", ".join(
            (
                f"{h['label'][0]}="
                f"{h['movement']:.1f}px"
            )
            if h["movement"] is not None
            else f"{h['label'][0]}=-"
            for h in hand_entries
        ) or "-"

        wrist_vals = ", ".join(
            (
                f"{w['side'][0].upper()}="
                f"{w['movement']:.1f}px"
            )
            if w["movement"] is not None
            else f"{w['side'][0].upper()}=-"
            for w in wrist_info
        ) or "-"

        debug_1 = (
            f"Hands={len(hand_entries)} "
            f"ArmZone={'YES' if arm_in_work_zone else 'NO'} "
            f"Moving={'YES' if arm_moving else 'NO'}"
        )

        debug_2 = (
            f"HandMove=[{move_vals}] "
            f"WristMove=[{wrist_vals}]"
        )

        debug_y = int(y2) + 22

        cv2.putText(
            display_frame,
            debug_1,
            (
                int(x1),
                debug_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            display_frame,
            debug_2,
            (
                int(x1),
                debug_y + 18,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )

    # =========================================================================
    # FULL ARM SKELETON
    # =========================================================================

    @staticmethod
    def _draw_full_arm_skeleton(
        frame,
        arm_points,
        status_color,
    ):
        """
        Draw:

            LEFT SHOULDER -> LEFT ELBOW -> LEFT WRIST

            RIGHT SHOULDER -> RIGHT ELBOW -> RIGHT WRIST

        Also draws the shoulder-to-shoulder connection.
        """

        left = arm_points["left"]
        right = arm_points["right"]

        # ---------------------------------------------------------------------
        # Helper
        # ---------------------------------------------------------------------

        def draw_segment(
            p1,
            p2,
            color,
            thickness=4,
        ):
            if p1 is not None and p2 is not None:

                cv2.line(
                    frame,
                    p1,
                    p2,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        # ---------------------------------------------------------------------
        # LEFT ARM
        # ---------------------------------------------------------------------

        draw_segment(
            left["shoulder"],
            left["elbow"],
            SHOULDER_COLOR,
            4,
        )

        draw_segment(
            left["elbow"],
            left["wrist"],
            ELBOW_COLOR,
            4,
        )

        # ---------------------------------------------------------------------
        # RIGHT ARM
        # ---------------------------------------------------------------------

        draw_segment(
            right["shoulder"],
            right["elbow"],
            SHOULDER_COLOR,
            4,
        )

        draw_segment(
            right["elbow"],
            right["wrist"],
            ELBOW_COLOR,
            4,
        )

        # ---------------------------------------------------------------------
        # SHOULDERS
        # ---------------------------------------------------------------------

        if (
            left["shoulder"] is not None
            and right["shoulder"] is not None
        ):

            cv2.line(
                frame,
                left["shoulder"],
                right["shoulder"],
                status_color,
                3,
                cv2.LINE_AA,
            )

        # ---------------------------------------------------------------------
        # Points
        # ---------------------------------------------------------------------

        points = [
            (
                "L-SHOULDER",
                left["shoulder"],
                SHOULDER_COLOR,
                9,
            ),
            (
                "L-ELBOW",
                left["elbow"],
                ELBOW_COLOR,
                9,
            ),
            (
                "L-WRIST",
                left["wrist"],
                WRIST_COLOR,
                10,
            ),
            (
                "R-SHOULDER",
                right["shoulder"],
                SHOULDER_COLOR,
                9,
            ),
            (
                "R-ELBOW",
                right["elbow"],
                ELBOW_COLOR,
                9,
            ),
            (
                "R-WRIST",
                right["wrist"],
                WRIST_COLOR,
                10,
            ),
        ]

        for (
            name,
            point,
            color,
            radius,
        ) in points:

            if point is None:
                continue

            cv2.circle(
                frame,
                point,
                radius,
                color,
                -1,
            )

            cv2.circle(
                frame,
                point,
                radius,
                (255, 255, 255),
                1,
            )

            # Small keypoint label
            label_offset_x = 8
            label_offset_y = -8

            cv2.putText(
                frame,
                name,
                (
                    point[0] + label_offset_x,
                    point[1] + label_offset_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    # =========================================================================
    # HAND DRAWING
    # =========================================================================

    @staticmethod
    def _draw_hand(
        frame,
        points,
        color,
    ):
        """
        Draw complete 21-point MediaPipe hand.
        """

        if points is None or len(points) < 21:
            return

        # Connections
        for a, b in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[a],
                points[b],
                color,
                2,
                cv2.LINE_AA,
            )

        # Landmark points
        for i, pt in enumerate(points):

            radius = (
                7
                if i == 0
                else 5
            )

            cv2.circle(
                frame,
                pt,
                radius,
                color,
                -1,
            )

            cv2.circle(
                frame,
                pt,
                radius,
                (255, 255, 255),
                1,
            )

    # =========================================================================
    # ZONES + HUD
    # =========================================================================

    def _draw_zones_and_hud(
        self,
        display_frame,
        confirmed_status,
        orig_w,
        orig_h,
        scale_factor,
    ):

        # ---------------------------------------------------------------------
        # ONLY WORK ZONE
        # ---------------------------------------------------------------------

        if (
            self.work_zone is not None
            and len(self.work_zone) >= 3
        ):

            cv2.polylines(
                display_frame,
                [self.work_zone],
                True,
                WORK_ZONE_COLOR,
                3,
            )

            first_point = self.work_zone[0]

            cv2.putText(
                display_frame,
                "HAND WORK ZONE",
                (
                    int(first_point[0]) + 5,
                    int(first_point[1]) + 22,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WORK_ZONE_COLOR,
                2,
                cv2.LINE_AA,
            )

        # ---------------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------------

        color = STATUS_COLORS.get(
            confirmed_status,
            (255, 255, 255),
        )

        cv2.rectangle(
            display_frame,
            (10, 10),
            (500, 72),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            display_frame,
            confirmed_status.replace(
                "_",
                " ",
            ),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            color,
            2,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------------------
        # Timing
        # ---------------------------------------------------------------------

        avg_inf = (
            np.mean(self.inference_times)
            if self.inference_times
            else 0
        )

        avg_frame = (
            np.mean(self.frame_times)
            if self.frame_times
            else 0
        )

        actual_width = (
            int(orig_w / scale_factor)
            if scale_factor
            else orig_w
        )

        actual_height = (
            int(orig_h / scale_factor)
            if scale_factor
            else orig_h
        )

        timing_str = (
            f"Frame: {avg_frame:.0f}ms | "
            f"Inf: {avg_inf:.0f}ms | "
            f"Res: {actual_width}x{actual_height}"
        )

        cv2.putText(
            display_frame,
            timing_str,
            (
                20,
                display_frame.shape[0] - 50,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            display_frame,
            "Q=quit  R=redraw work zone",
            (
                20,
                display_frame.shape[0] - 18,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # =========================================================================
    # CLOSE
    # =========================================================================

    def close(self) -> None:
        try:
            self.hand_model.close()
        except Exception:
            pass