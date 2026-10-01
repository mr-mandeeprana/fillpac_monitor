from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from elasticsearch import Elasticsearch


class ELKLogger:
    """
    Non-blocking Elasticsearch logger.

    Main application:
        monitor -> ELKLogger.log()

    Background worker:
        queue -> Elasticsearch

    This prevents Elasticsearch/network delays from slowing
    down the camera/YOLO processing loop.
    """

    def __init__(
        self,
        host: str,
        index_name: str,
        logger: logging.Logger,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        verify_certs: bool = False,
        queue_size: int = 5000,
    ):
        self.logger = logger
        self.index_name = index_name

        self.queue = queue.Queue(
            maxsize=queue_size
        )

        self.running = True

        # -------------------------------------------------------------
        # Elasticsearch connection
        # -------------------------------------------------------------

        if api_key:
            self.client = Elasticsearch(
                host,
                api_key=api_key,
                verify_certs=verify_certs,
            )

        elif username and password:
            self.client = Elasticsearch(
                host,
                basic_auth=(
                    username,
                    password,
                ),
                verify_certs=verify_certs,
            )

        else:
            self.client = Elasticsearch(
                host,
                verify_certs=verify_certs,
            )

        # -------------------------------------------------------------
        # Test connection
        # -------------------------------------------------------------

        try:
            info = self.client.info()

            self.logger.info(
                "Elasticsearch connected: %s",
                info.get("version", {}).get(
                    "number",
                    "unknown",
                ),
            )

        except Exception as exc:

            self.logger.error(
                "Elasticsearch connection failed: %s",
                exc,
            )

        # -------------------------------------------------------------
        # Background worker
        # -------------------------------------------------------------

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="elk-worker",
        )

        self.worker.start()

    # =================================================================
    # PUBLIC LOG METHOD
    # =================================================================

    def log(
        self,
        data: dict,
    ) -> bool:
        """
        Add document to background queue.

        Never blocks the camera pipeline.
        """

        if not self.running:
            return False

        try:
            self.queue.put_nowait(data)
            return True

        except queue.Full:

            self.logger.warning(
                "Elasticsearch queue full; dropping event"
            )

            return False

    # =================================================================
    # WORKER
    # =================================================================

    def _worker_loop(self):

        while self.running:

            try:

                document = self.queue.get(
                    timeout=1.0
                )

            except queue.Empty:
                continue

            try:

                self.client.index(
                    index=self.index_name,
                    document=document,
                )

            except Exception as exc:

                self.logger.error(
                    "Failed to send document to Elasticsearch: %s",
                    exc,
                )

            finally:

                self.queue.task_done()

    # =================================================================
    # STATUS EVENT
    # =================================================================

    def log_status(
        self,
        station_id: str,
        station_name: str,
        status: str,
        metadata: Optional[dict] = None,
    ):

        document = {
            "@timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "event_type": "operator_status",

            "station_id": station_id,
            "station_name": station_name,

            "status": status,

            "metadata": metadata or {},
        }

        self.log(document)

    # =================================================================
    # HEARTBEAT
    # =================================================================

    def log_heartbeat(
        self,
        station_id: str,
        station_name: str,
        data: Optional[dict] = None,
    ):

        document = {
            "@timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "event_type": "heartbeat",

            "station_id": station_id,
            "station_name": station_name,

            "status": "ONLINE",

            "metadata": data or {},
        }

        self.log(document)

    # =================================================================
    # CLOSE
    # =================================================================

    def close(self):

        self.running = False

        try:
            self.worker.join(
                timeout=3
            )
        except Exception:
            pass

        try:
            self.client.close()
        except Exception:
            pass