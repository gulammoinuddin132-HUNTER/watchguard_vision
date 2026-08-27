"""
Response Engine (Modular Interface)
-----------------------------------
Handles visual alerts, voice responses, evidence snapshot saving,
and security event dispatching.
"""

from typing import Any, Dict, Optional
import numpy as np

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel


class ResponseEngine:
    """
    Response Engine for incident mitigation, notifications, and evidence preservation.
    """

    def __init__(self, db_manager: DatabaseManager, event_logger: EventLogger):
        self.db = db_manager
        self.logger = event_logger

    def trigger_alert(
        self,
        risk_level: RiskLevel,
        title: str,
        description: str,
        frame: Optional[np.ndarray] = None,
        evidence_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes security alert protocols: logs event, captures evidence frame,
        and prepares audio/visual feedback.
        """
        event = self.logger.log_event(
            event_type=EventType.SECURITY_ALERT,
            risk_level=risk_level,
            description=f"{title}: {description}",
            evidence_frame_path=evidence_path,
        )
        return event
