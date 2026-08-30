"""
Event Logger Framework
----------------------
Handles structured logging across SQLite database, rotating file logs, and
real-time subscriber callbacks (for UI dashboard updates).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
from logging.handlers import RotatingFileHandler

from config import (
    LOG_FILE_PATH,
    CONSOLE_LOG_LEVEL,
    FILE_LOG_LEVEL,
    LOG_ROTATION_MAX_BYTES,
    LOG_ROTATION_BACKUP_COUNT,
)
from src.database.db_manager import DatabaseManager
from src.events.event_types import EventType, RiskLevel


class EventLogger:
    """Centralized event logger for WatchGuard Vision."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        log_file_path: Optional[Path] = LOG_FILE_PATH,
        attach_handlers: bool = True,
    ):
        self.db_manager = db_manager or DatabaseManager()
        self.log_file_path = log_file_path
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        if attach_handlers:
            self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        """Sets up Python logging to rotating file and stdout."""
        self.logger = logging.getLogger("WatchGuardVision")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            # Rotating File Handler
            if self.log_file_path:
                file_level = getattr(logging, str(FILE_LOG_LEVEL).upper(), logging.INFO)
                file_handler = RotatingFileHandler(
                    str(self.log_file_path),
                    maxBytes=LOG_ROTATION_MAX_BYTES,
                    backupCount=LOG_ROTATION_BACKUP_COUNT,
                    encoding="utf-8",
                )
                file_handler.setLevel(file_level)
                file_formatter = logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)

            # Stream Handler (console)
            console_level = getattr(logging, str(CONSOLE_LOG_LEVEL).upper(), logging.INFO)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(console_level)
            stream_formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler)

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback listener that receives new events in real time."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a callback listener."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def log_event(
        self,
        event_type: EventType | str,
        risk_level: RiskLevel | str,
        description: str,
        evidence_frame_path: Optional[str] = None,
        person_name: Optional[str] = None,
        object_summary: Optional[str] = None,
        metadata_json: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Records an event to database, log file, and notifies all UI subscribers.
        """
        # Normalize enums
        e_type = event_type.value if isinstance(event_type, EventType) else str(event_type)
        r_level = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Database insertion
        event_id = self.db_manager.log_security_event(
            event_type=e_type,
            risk_level=r_level,
            description=description,
            evidence_frame_path=evidence_frame_path,
            person_name=person_name,
            object_summary=object_summary,
            metadata_json=metadata_json,
            timestamp=ts,
        )

        # 2. File log
        log_msg = f"[{r_level}] ({e_type}) {description}"
        if r_level in ("RED", "ORANGE"):
            self.logger.warning(log_msg)
        elif r_level == "YELLOW":
            self.logger.info(log_msg)
        else:
            self.logger.info(log_msg)

        event_payload = {
            "id": event_id,
            "event_type": e_type,
            "risk_level": r_level,
            "description": description,
            "evidence_frame_path": evidence_frame_path,
            "person_name": person_name,
            "object_summary": object_summary,
            "metadata_json": metadata_json,
            "timestamp": timestamp,
        }

        # 3. Notify subscribers (UI updates)
        for subscriber in self._subscribers:
            try:
                subscriber(event_payload)
            except Exception as e:
                self.logger.error(f"Error executing event subscriber callback: {e}")

        return event_payload
