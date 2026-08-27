"""
Unit tests for EventLogger and subscriber notification system
"""

import tempfile
import unittest
from pathlib import Path

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel


class TestEventLogger(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_watchguard.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_event_and_subscriber_callback(self):
        """Verify subscribers receive event payloads accurately."""
        received_events = []

        def callback(event):
            received_events.append(event)

        self.logger.subscribe(callback)

        event = self.logger.log_event(
            event_type=EventType.CAMERA_STARTED,
            risk_level=RiskLevel.GREEN,
            description="Webcam streaming online.",
        )

        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["event_type"], "CAMERA_STARTED")
        self.assertEqual(received_events[0]["risk_level"], "GREEN")
        self.assertEqual(received_events[0]["description"], "Webcam streaming online.")

    def test_risk_level_properties(self):
        """Verify risk level labels and color codes."""
        self.assertEqual(RiskLevel.GREEN.label, "NORMAL")
        self.assertEqual(RiskLevel.YELLOW.label, "ATTENTION")
        self.assertEqual(RiskLevel.ORANGE.label, "HIGH RISK")
        self.assertEqual(RiskLevel.RED.label, "SECURITY INCIDENT")


if __name__ == "__main__":
    unittest.main()
