"""
Automated Unit Tests for Context & Security Intelligence Engine (Phase 2.3)
---------------------------------------------------------------------------
Tests:
- Rule A: Known/Authorized person + normal activity -> GREEN
- Rule B: Unknown person detected -> YELLOW
- Rule C: Unknown person + protected object in proximity -> RED
- Rule D: Authorized person + protected object outside authorized hours -> ORANGE
- Rule E: Person detected inside restricted zone -> ORANGE
- Rule F: Unknown person inside restricted zone -> RED
- Spatial bounding box distance & proximity calculation
- Time context evaluation (authorized vs. after-hours)
- Explainable human-readable reason string generation
- Cooldown throttling and event debouncing
- SQLite persistence of security decisions and snapshots
"""

from datetime import datetime, time as dtime
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.core.context_engine import ContextEngine, SecurityDecision, compute_box_distance
from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel


class TestContextEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_context.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.engine = ContextEngine(proximity_threshold=150.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_box_distance(self):
        """Verify bounding box edge-to-edge distance calculations."""
        # Overlapping boxes -> distance 0.0
        box1 = (100, 100, 100, 100)  # x: 100..200, y: 100..200
        box2 = (150, 150, 100, 100)  # x: 150..250, y: 150..250
        self.assertEqual(compute_box_distance(box1, box2), 0.0)

        # Touching horizontally -> distance 0.0
        box3 = (200, 100, 50, 100)   # x: 200..250, y: 100..200
        self.assertEqual(compute_box_distance(box1, box3), 0.0)

        # Separated horizontally by 50px
        box4 = (250, 100, 50, 100)   # x: 250..300, y: 100..200
        self.assertAlmostEqual(compute_box_distance(box1, box4), 50.0)

        # Separated diagonally (dx=30, dy=40 -> dist=50)
        box5 = (230, 240, 50, 50)
        self.assertAlmostEqual(compute_box_distance(box1, box5), 50.0)

    def test_time_window_evaluation(self):
        """Verify operational hours check."""
        engine = ContextEngine(hours_start="09:00", hours_end="18:00")

        # 14:30 is during authorized hours
        dt_day = datetime(2026, 8, 23, 14, 30)
        is_auth, time_str = engine.is_authorized_time(dt_day)
        self.assertTrue(is_auth)
        self.assertEqual(time_str, "14:30")

        # 21:45 is outside authorized hours
        dt_night = datetime(2026, 8, 23, 21, 45)
        is_auth, time_str = engine.is_authorized_time(dt_night)
        self.assertFalse(is_auth)
        self.assertEqual(time_str, "21:45")

    def test_rule_a_normal_activity_green(self):
        """Verify Rule A: Authorized Person + Normal Object -> GREEN."""
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Officer Alice", "bbox": (100, 100, 100, 150), "is_known": True}],
        }
        object_data = {
            "detected": True,
            "count": 1,
            "objects": [{"label": "bottle", "confidence": 0.90, "bbox": (500, 400, 50, 80)}],
            "summary": "bottle",
        }
        dec = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(dec.risk_level, RiskLevel.GREEN)
        self.assertEqual(dec.decision, "NORMAL_ACTIVITY")
        self.assertFalse(dec.evidence_required)
        self.assertIn("Authorized personnel verified: Officer Alice", dec.reason)

    def test_rule_b_unknown_person_yellow(self):
        """Verify Rule B: Unknown Person in Standard Area -> YELLOW."""
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": True,
            "people": [{"name": "Unknown Person", "bbox": (100, 100, 100, 150), "is_known": False}],
        }
        object_data = {"detected": False, "count": 0, "objects": [], "summary": "None"}
        dec = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(dec.risk_level, RiskLevel.YELLOW)
        self.assertEqual(dec.decision, "UNKNOWN_PERSON")
        self.assertTrue(dec.evidence_required)
        self.assertIn("Unregistered individual detected", dec.reason)

    def test_rule_c_unknown_person_protected_object_red(self):
        """Verify Rule C: Unknown Person + Protected Object in Proximity -> RED."""
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": True,
            "people": [{"name": "Unknown Person", "bbox": (100, 100, 100, 150), "is_known": False}],
        }
        # Laptop within 50px of person (proximity threshold is 150px)
        object_data = {
            "detected": True,
            "count": 1,
            "objects": [{"label": "laptop", "confidence": 0.94, "bbox": (180, 120, 120, 90)}],
            "summary": "laptop",
        }
        dec = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(dec.risk_level, RiskLevel.RED)
        self.assertEqual(dec.decision, "POTENTIAL_UNAUTHORIZED_INTERACTION")
        self.assertTrue(dec.evidence_required)
        self.assertIn("LAPTOP", dec.reason)
        self.assertIn("visual proximity", dec.reason)

    def test_rule_d_authorized_person_after_hours_orange(self):
        """Verify Rule D: Authorized User + Protected Object Outside Hours -> ORANGE."""
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Officer Alice", "bbox": (100, 100, 100, 150), "is_known": True}],
        }
        object_data = {
            "detected": True,
            "count": 1,
            "objects": [{"label": "laptop", "confidence": 0.92, "bbox": (150, 120, 100, 80)}],
            "summary": "laptop",
        }
        # Force after-hours time (23:15)
        self.engine.set_time_override(dtime(23, 15))
        dec = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(dec.risk_level, RiskLevel.ORANGE)
        self.assertEqual(dec.decision, "SUSPICIOUS_AUTHORIZED_USER_ACTIVITY")
        self.assertTrue(dec.evidence_required)
        self.assertIn("outside authorized hours", dec.reason)
        self.assertIn("Officer Alice", dec.reason)

    def test_rule_e_restricted_zone_activity_orange(self):
        """Verify Rule: Authorized Person in Restricted Zone -> ORANGE (Outside Hours) / GREEN (Inside Hours)."""
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Officer Alice", "bbox": (500, 100, 100, 150), "is_known": True}],
        }
        object_data = {"detected": False, "count": 0, "objects": [], "summary": "None"}
        self.engine.set_zone_override("RESTRICTED")

        # 1. After-hours in restricted zone -> ORANGE (AFTER_HOURS_RESTRICTED_ACCESS_DENIED)
        self.engine.set_time_override(dtime(22, 30))
        dec = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(dec.risk_level, RiskLevel.ORANGE)
        self.assertEqual(dec.decision, "AFTER_HOURS_RESTRICTED_ACCESS_DENIED")
        self.assertFalse(dec.is_access_granted)
        self.assertTrue(dec.evidence_required)
        self.assertIn("outside permitted operating hours", dec.reason)

        # 2. Permitted operating hours in restricted zone -> GREEN (AUTHORIZED_RESTRICTED_ZONE_ACCESS)
        self.engine.set_time_override(dtime(14, 0))
        dec_day = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(dec_day.risk_level, RiskLevel.GREEN)
        self.assertEqual(dec_day.decision, "AUTHORIZED_RESTRICTED_ZONE_ACCESS")
        self.assertTrue(dec_day.is_access_granted)

    def test_rule_f_unknown_person_restricted_zone_red(self):
        """Verify Rule F: Unknown Person in Restricted Zone -> RED."""
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": True,
            "people": [{"name": "Unknown Person", "bbox": (500, 100, 100, 150), "is_known": False}],
        }
        object_data = {"detected": False, "count": 0, "objects": [], "summary": "None"}
        self.engine.set_zone_override("RESTRICTED")
        dec = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(dec.risk_level, RiskLevel.RED)
        self.assertEqual(dec.decision, "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS")
        self.assertTrue(dec.evidence_required)
        self.assertIn("Unauthorized access", dec.reason)

    def test_alert_debouncing_and_cooldown(self):
        """Verify context engine throttles rapid repetitive alerts."""
        rule_key = "test_rule_alert"
        # First alert passes
        self.assertTrue(self.engine.should_trigger_alert(rule_key, cooldown_seconds=1.5))
        # Immediate second alert throttled
        self.assertFalse(self.engine.should_trigger_alert(rule_key, cooldown_seconds=1.5))
        # After cooldown passes, alert fires again
        time.sleep(1.6)
        self.assertTrue(self.engine.should_trigger_alert(rule_key, cooldown_seconds=1.5))

    def test_sqlite_event_integration(self):
        """Verify security decisions are logged cleanly into SQLite database."""
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": True,
            "people": [{"name": "Unknown Person", "bbox": (100, 100, 100, 150), "is_known": False}],
        }
        object_data = {
            "detected": True,
            "count": 1,
            "objects": [{"label": "cell phone", "confidence": 0.91, "bbox": (120, 120, 50, 80)}],
            "summary": "cell phone",
        }
        dec = self.engine.evaluate_context(person_data, object_data)

        # Log decision into event logger
        event = self.logger.log_event(
            event_type=dec.event_type,
            risk_level=dec.risk_level,
            description=dec.reason,
            person_name="Unknown Person",
            object_summary="cell phone",
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["risk_level"], "RED")
        self.assertIn("cell phone", event["object_summary"])

        # Check in DB
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT risk_level, event_type, description FROM security_events WHERE id = ?", (event["id"],))
            row = cursor.fetchone()
            self.assertEqual(row["risk_level"], "RED")
            self.assertEqual(row["event_type"], "SECURITY_ALERT")

    def test_hud_rendering(self):
        """Verify draw_context_hud renders zones and proximity vectors without corrupting frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Officer Alice", "bbox": (100, 100, 100, 150), "is_known": True}],
        }
        object_data = {
            "detected": True,
            "count": 1,
            "objects": [{"label": "laptop", "confidence": 0.95, "bbox": (180, 120, 120, 90)}],
            "summary": "laptop",
        }
        dec = self.engine.evaluate_context(person_data, object_data)
        annotated = self.engine.draw_context_hud(frame, dec)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertIsInstance(annotated, np.ndarray)


if __name__ == "__main__":
    unittest.main()
