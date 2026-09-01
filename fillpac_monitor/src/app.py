"""
Application orchestrator. Wires together config, logging, DB, camera,
detection models, zones, and the main processing loop. This is the only
module that knows about all the pieces at once - everything else is a
focused, independently testable component.
"""

from __future__ import annotations

import signal
import time

import cv2

from src.camera.capture_thread import CameraManager
from src.config_loader import AppConfig, load_config
from src.db.db_manager import DbManager
from src.detection.hand_model import HandModel
from src.detection.pose_model import PoseModel
from src.logging_setup import attach_db_handler, setup_logging
from src.monitor.operator_monitor import OperatorMonitor
from src.zones.zone_manager import ZoneManager


class FillPacApp:
    def __init__(self, config_path: str | None = None, env_path: str | None = None):
        self.cfg: AppConfig = load_config(config_path, env_path)
        self.logger = setup_logging(
            log_dir=self.cfg.path(self.cfg.get("logging.log_dir", "logs")),
            level=self.cfg.get("logging.log_level", "INFO"),
            max_bytes=self.cfg.get("logging.max_bytes", 10 * 1024 * 1024),
            backup_count=self.cfg.get("logging.backup_count", 10),
        )
        self.station_id = self.cfg.station_id

        self.db: DbManager | None = None
        self.camera_manager: CameraManager | None = None
        self.zone_manager: ZoneManager | None = None
        self.monitor: OperatorMonitor | None = None
        self._shutdown_requested = False

    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("FillPac Operator Monitor - starting up (station=%s)", self.station_id)
        self.logger.info("=" * 70)

        # ---- Database ----
        db_settings = self.cfg.db_settings()
        self.db = DbManager(db_settings, self.logger)
        self.db.start()
        if db_settings.get("enabled"):
            attach_db_handler(self.logger, self.db, self.station_id)

        # ---- Camera ----
        self.camera_manager = CameraManager(
            source=self.cfg.camera_source,
            cfg=self.cfg.get("camera", {}),
            logger=self.logger,
        )
        cap, first_frame = self.camera_manager.open()
        h, w = first_frame.shape[:2]
        self.logger.info("Stream resolution: %dx%d", w, h)

        if self.cfg.get("camera.use_threaded_capture", True):
            self.camera_manager.start_threaded_capture()

        # ---- Zones ----
        self.zone_manager = ZoneManager(
            zones_dir=self.cfg.path("zones_config"),
            station_id=self.station_id,
            logger=self.logger,
        )
        loaded = self.zone_manager.load(frame_shape=(h, w))
        if loaded is not None:
            operator_zone, work_zone = loaded
            self.logger.info("Loaded saved zones for station %s", self.station_id)
        elif self.cfg.get("zones.interactive_draw_on_first_run", True):
            self.logger.info("No saved zones found - launching interactive zone drawing")
            operator_zone = self.zone_manager.draw_interactive(
                first_frame, "1 - Draw OPERATOR ZONE", (255, 0, 255)
            )
            work_zone = self.zone_manager.draw_interactive(
                first_frame, "2 - Draw HAND WORK ZONE", (255, 0, 0)
            )
            self.zone_manager.save(operator_zone, work_zone, (h, w))
        else:
            operator_zone, work_zone = self.zone_manager.defaults_from_fractions(
                self.cfg.get("zones.operator_zone_fractions"),
                self.cfg.get("zones.work_zone_fractions"),
                (h, w),
            )
            self.logger.info("Using default fractional zones (no interactive draw)")

        # ---- Detection models ----
        models_cfg = self.cfg.get("models", {})
        pose_model = PoseModel(
            model_path=self.cfg.path(models_cfg.get("pose_model_path")),
            device=models_cfg.get("device", "cpu"),
            conf=models_cfg.get("yolo_conf_threshold", 0.25),
            iou=models_cfg.get("yolo_iou_threshold", 0.25),
            logger=self.logger,
        )
        hand_model = HandModel(
            model_path=self.cfg.path(models_cfg.get("hand_landmarker_path")),
            model_url=models_cfg.get("hand_landmarker_url"),
            max_hands=models_cfg.get("hand_max_num", 2),
            detection_conf=models_cfg.get("hand_detection_conf", 0.5),
            tracking_conf=models_cfg.get("hand_tracking_conf", 0.5),
            logger=self.logger,
        )

        # ---- Monitor ----
        self.monitor = OperatorMonitor(
            pose_model=pose_model,
            hand_model=hand_model,
            operator_zone=operator_zone,
            work_zone=work_zone,
            monitor_cfg=self.cfg.get("monitor", {}),
            display_cfg=self.cfg.get("display", {}),
            inference_width=models_cfg.get("inference_width", 416),
            keypoint_conf_threshold=models_cfg.get("keypoint_conf_threshold", 0.40),
            hand_crop_padding=models_cfg.get("hand_crop_padding", 60),
            station_id=self.station_id,
            db_manager=self.db,
            logger=self.logger,
        )

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.logger.info("Setup complete - entering main loop")

    def _handle_signal(self, signum, frame) -> None:
        self.logger.info("Received shutdown signal (%s)", signum)
        self._shutdown_requested = True

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        cam = self.camera_manager
        cfg = self.cfg
        rtsp_mode = cam.rtsp_mode
        use_threaded = cfg.get("camera.use_threaded_capture", True)
        max_failures = cfg.get("camera.max_consecutive_failures", 30)
        show_window = cfg.get("display.show_window", True)
        consecutive_failures = 0

        try:
            while not self._shutdown_requested:
                if use_threaded:
                    ret, frame = cam.capture_thread.get_frame()
                else:
                    ret, frame = cam.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures % 10 == 0:
                        self.logger.warning("No frames received (%d consecutive)", consecutive_failures)
                        if self.db:
                            self.db.enqueue_health(
                                self.station_id, is_connected=False,
                                consecutive_failures=consecutive_failures,
                                reconnect_count=cam.reconnect_count,
                                note="No frames received",
                            )

                    if not rtsp_mode:
                        self.logger.error("Camera stopped responding (non-RTSP source) - exiting")
                        break

                    if consecutive_failures >= max_failures:
                        self.logger.error("Too many consecutive failures - attempting reconnect")
                        cam.release()
                        cap, _ = cam.reconnect()
                        if cap is None:
                            self.logger.critical("Could not reconnect to camera - shutting down")
                            break
                        if use_threaded:
                            cam.start_threaded_capture()
                        consecutive_failures = 0
                    else:
                        time.sleep(0.001)
                    continue

                consecutive_failures = 0
                display_frame, status = self.monitor.process_frame(frame)

                if show_window:
                    cv2.imshow("FillPac - Operator Monitor", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.logger.info("Quit requested by user")
                        break
                    if key == ord("r"):
                        self._redraw_zones(display_frame)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.shutdown()

    def _redraw_zones(self, frame) -> None:
        self.logger.info("Redrawing zones (user requested)")
        operator_zone = self.zone_manager.draw_interactive(frame, "1 - Draw OPERATOR ZONE", (255, 0, 255))
        work_zone = self.zone_manager.draw_interactive(frame, "2 - Draw HAND WORK ZONE", (255, 0, 0))
        h, w = frame.shape[:2]
        self.zone_manager.save(operator_zone, work_zone, (h, w))
        self.monitor.update_zones(operator_zone, work_zone)

    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        self.logger.info("Shutting down...")
        if self.monitor is not None:
            self.monitor.close()
        if self.camera_manager is not None:
            self.camera_manager.release()
        cv2.destroyAllWindows()
        if self.db is not None:
            self.db.stop()
        self.logger.info("Shutdown complete")
