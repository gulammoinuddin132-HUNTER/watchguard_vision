"""
Automated Integration Tests for Real-Time Security Event Log Feed
------------------------------------------------------------------
Tests:
- Events inserted into SQLite database
- Events retrieved via get_recent_security_events and get_total_event_count
- Dashboard table population from SQLite records
- New live events appearing at the top of the event table (newest first)
- Risk formatting with visual emojis and labels
- Real-time subscriber queue processing in GUI event pump
"""

import tempfile
import unittest
import tkinter as tk
from pathlib import Path

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine
from src.ui.dashboard import WatchGuardDashboard


class TestEventFeedIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_event_feed.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.camera = CameraManager()
        self.person_engine = PersonRecognitionEngine(db_manager=self.db)
        self.object_engine = ObjectDetectionEngine(db_manager=self.db)
        self.context_engine = ContextEngine()

        self.root = tk.Tk()
        self.root.withdraw()  # Hide window during headless tests
        self.dashboard = WatchGuardDashboard(
            root=self.root,
            camera_manager=self.camera,
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.person_engine,
            object_engine=self.object_engine,
            context_engine=self.context_engine,
        )

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        self.temp_dir.cleanup()

    def test_database_event_insertion_and_count(self):
        """Verify events are properly inserted and counted in SQLite."""
        initial_count = self.db.get_total_event_count()
        self.logger.log_event(
            event_type=EventType.SECURITY_ALERT,
            risk_level=RiskLevel.RED,
            description="Test breach incident.",
        )
        new_count = self.db.get_total_event_count()
        self.assertEqual(new_count, initial_count + 1)

        events = self.db.get_recent_security_events(limit=5)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["risk_level"], "RED")
        self.assertEqual(events[0]["description"], "Test breach incident.")

    def test_table_populated_with_retrieved_rows(self):
        """Verify Treeview table contains rows corresponding to database events."""
        # Insert 3 distinct test events
        self.logger.log_event(EventType.CAMERA_STARTED, RiskLevel.GREEN, "Camera operational.")
        self.logger.log_event(EventType.UNKNOWN_PERSON, RiskLevel.YELLOW, "Unknown visitor detected.")
        self.logger.log_event(EventType.POLICY_VIOLATION, RiskLevel.ORANGE, "Restricted zone accessed.")

        # Reload table
        self.dashboard._load_recent_events()

        children = self.dashboard.tree.get_children()
        self.assertGreaterEqual(len(children), 3)

        # Inspect top row (newest event: ORANGE)
        top_row = self.dashboard.tree.item(children[0])
        values = top_row["values"]
        self.assertIn("ORANGE", str(values[2]))
        self.assertEqual(values[3], "POLICY_VIOLATION")
        self.assertEqual(values[4], "Restricted zone accessed.")

    def test_latest_event_prepended_at_top(self):
        """Verify real-time events appear at index 0 (the top of the table)."""
        self.dashboard._load_recent_events()

        # Log a new critical event
        self.logger.log_event(
            event_type=EventType.UNAUTHORIZED_ACCESS,
            risk_level=RiskLevel.RED,
            description="High priority perimeter breach!",
        )

        # Process UI event queue
        self.dashboard._schedule_event_pump()

        children = self.dashboard.tree.get_children()
        top_row = self.dashboard.tree.item(children[0])
        values = top_row["values"]

        self.assertIn("RED", str(values[2]))
        self.assertEqual(values[3], "UNAUTHORIZED_ACCESS")
        self.assertEqual(values[4], "High priority perimeter breach!")

    def test_risk_level_formatting(self):
        """Verify clear visual badges and labels for each risk level."""
        self.assertEqual(self.dashboard._format_risk_display("GREEN"), "🟢 NORMAL (GREEN)")
        self.assertEqual(self.dashboard._format_risk_display("YELLOW"), "🟡 ATTENTION (YELLOW)")
        self.assertEqual(self.dashboard._format_risk_display("ORANGE"), "🟠 HIGH RISK (ORANGE)")
        self.assertEqual(self.dashboard._format_risk_display("RED"), "🔴 INCIDENT (RED)")

    def test_treeview_columns_configured(self):
        """Verify all 5 required columns exist in the Treeview."""
        expected_cols = ("id", "timestamp", "risk", "type", "description")
        tree_cols = self.dashboard.tree["columns"]
        for col in expected_cols:
            self.assertIn(col, tree_cols)


if __name__ == "__main__":
    unittest.main()
