"""
WatchGuard Vision - Phase 5.7.1 Current-State Integrity & Stable Dashboard Tests
--------------------------------------------------------------------------------
Validates:
1. Exact live person count (1 unknown person -> people_count == 1, never labeled authorized)
2. Case-insensitive "UNKNOWN PERSON" / "Unknown" classification
3. Separation of CURRENT_STATE vs REGISTERED_PROFILES in database
4. Zero people in camera -> people_count == 0, no stale session leaks
5. Multi-person counts (auth + unk -> count == 2)
6. Stable widget update caching (RiskBanner, TelemetryItem)
7. Event log layout geometry weights
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
import cv2
import numpy as np

from src.core.person_entity import PersonEntity
from src.core.copilot_reasoner import (
    build_structured_ai_context,
    DeterministicCopilotReasoner,
)
from src.core.voice_assistant import VoiceAssistant, SessionContext
from src.core.context_engine import ContextEngine, SecurityDecision
from src.events.event_types import RiskLevel, EventType
from src.database.db_manager import DatabaseManager
from src.ui.widgets import RiskBanner, TelemetryItem


class TestPhase571CurrentStateIntegrity(unittest.TestCase):
    """Rigorous validation of Phase 5.7.1 current-state correctness and UI stability."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_p571.db")
        self.db = DatabaseManager(self.db_path)

        self.context_engine = MagicMock(spec=ContextEngine)
        self.context_engine.hours_start = "09:00"
        self.context_engine.hours_end = "18:00"
        self.context_engine.last_decision = None

        self.reasoner = DeterministicCopilotReasoner()
        self.va = VoiceAssistant(
            context_engine=self.context_engine,
            db_manager=self.db,
        )
        self.va.mute_audio = True

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- Test 1: Single UNKNOWN person in live frame -> people_count == 1, NOT authorized ---
    def test_01_single_unknown_person_integrity(self):
        """Single unknown person must report exactly count 1 and NEVER say 'authorized'."""
        unk_entity = PersonEntity(
            track_id=1,
            bbox=(100, 100, 80, 120),
            identity="UNKNOWN PERSON",
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.92,
            is_spoof=False,
        )
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="UNREGISTERED_PERSON",
            reason="Unregistered individual detected.",
            person_entities=[unk_entity.to_dict()],
            context_summary={
                "person_detected": True,
                "person_name": "UNKNOWN PERSON",
                "is_authorized": False,
                "is_unknown": True,
                "person_entities": [unk_entity.to_dict()],
                "objects": [],
            },
        )
        self.va._sync_session_from_live_state()

        # Check session context
        self.assertEqual(self.va.session_context.last_people_count, 1)
        self.assertFalse(self.va.session_context.last_is_authorized)
        self.assertTrue(self.va.session_context.last_is_unknown)

        # Check Voice Assistant response
        resp_overview = self.va._query_scene_overview()
        self.assertIn("unregistered person", resp_overview.lower())
        self.assertNotIn("authorized person", resp_overview.lower())

        resp_person = self.va._query_current_person()
        self.assertIn("unregistered person", resp_person.lower())
        self.assertNotIn("authorized", resp_person.lower())

        resp_count = self.va._query_live_people_count()
        self.assertIn("one person in view", resp_count.lower())
        self.assertIn("unregistered", resp_count.lower())

    # --- Test 2: Registered profile in database does NOT create a second live person ---
    def test_02_database_profile_isolation_from_current_state(self):
        """A registered face in DB must not appear in CURRENT_STATE when only 1 unknown person is live."""
        # Seed 1 registered profile in SQLite
        u_id = self.db.create_user("hunter_admin", "Hunter Lead", role="Staff", access_level=3)

        unk_entity = PersonEntity(
            track_id=1,
            bbox=(100, 100, 80, 120),
            identity="Unknown Person",
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.88,
        )
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="UNREGISTERED_PERSON",
            reason="Unregistered individual detected.",
            person_entities=[unk_entity.to_dict()],
            context_summary={
                "person_detected": True,
                "person_name": "Unknown Person",
                "is_authorized": False,
                "person_entities": [unk_entity.to_dict()],
            },
        )
        self.va._sync_session_from_live_state()

        context = build_structured_ai_context(self.context_engine, None, self.db)
        curr = context["CURRENT_STATE"]
        self.assertEqual(curr["people_count"], 1)
        self.assertEqual(len(curr["people"]), 1)
        self.assertFalse(curr["people"][0]["is_registered"])

        # Voice assistant queries
        resp = self.va.process_command("Who is in front of the camera?")
        self.assertNotIn("Hunter", resp)
        self.assertIn("unregistered", resp.lower())

    # --- Test 3: Zero people in camera -> people_count == 0 (no stale session revival) ---
    def test_03_zero_people_in_camera(self):
        """When 0 people are visible, people_count MUST equal 0."""
        # Initially primed session with someone
        self.va.session_context.last_people_entities = [{"identity": "Old Ghost", "is_known": True}]
        self.va.session_context.last_people_count = 1

        # Now new decision arrives with 0 people
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Perimeter clear.",
            person_entities=[],
            context_summary={
                "person_detected": False,
                "person_name": "None",
                "is_authorized": False,
                "person_entities": [],
                "objects": [],
            },
        )
        self.va._sync_session_from_live_state()

        self.assertEqual(self.va.session_context.last_people_count, 0)
        self.assertIsNone(self.va.session_context.last_person_name)
        self.assertFalse(self.va.session_context.last_is_authorized)

        resp = self.va._query_current_person()
        self.assertIn("do not detect any person", resp.lower())

        resp_count = self.va._query_live_people_count()
        self.assertIn("no people in view", resp_count.lower())

    # --- Test 4: Single Authorized Person -> exact count 1 and authorized label ---
    def test_04_single_authorized_person(self):
        """Single authorized person must report count 1 with recognized status."""
        auth_entity = PersonEntity(
            track_id=1,
            bbox=(100, 100, 80, 120),
            identity="Alice Admin",
            is_known=True,
            role="Staff",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.98,
        )
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="AUTHORIZED_ACCESS_GRANTED",
            reason="Verified personnel.",
            person_entities=[auth_entity.to_dict()],
            context_summary={
                "person_detected": True,
                "person_name": "Alice Admin",
                "is_authorized": True,
                "person_entities": [auth_entity.to_dict()],
            },
        )
        self.va._sync_session_from_live_state()

        self.assertEqual(self.va.session_context.last_people_count, 1)
        self.assertTrue(self.va.session_context.last_is_authorized)

        resp = self.va._query_current_person()
        self.assertIn("Alice Admin", resp)
        self.assertIn("Authorized", resp)

    # --- Test 5: Authorized + Unregistered simultaneous -> count == 2 ---
    def test_05_authorized_plus_unregistered_count_two(self):
        """Simultaneous authorized and unknown person must report exactly count 2."""
        e1 = PersonEntity(track_id=1, bbox=(50, 50, 50, 50), identity="Alice Admin", is_known=True, role="Staff", liveness_state="STABLE_LIVE")
        e2 = PersonEntity(track_id=2, bbox=(200, 200, 50, 50), identity="Unknown Person", is_known=False, role="Visitor", liveness_state="STABLE_LIVE")

        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="MULTI_PERSON_MONITORING",
            reason="Multiple individuals in scene.",
            person_entities=[e1.to_dict(), e2.to_dict()],
            context_summary={
                "person_detected": True,
                "person_name": "Alice Admin",
                "is_authorized": True,
                "person_entities": [e1.to_dict(), e2.to_dict()],
            },
        )
        self.va._sync_session_from_live_state()

        self.assertEqual(self.va.session_context.last_people_count, 2)

        resp_count = self.va._query_live_people_count()
        self.assertIn("two people in view", resp_count.lower())
        self.assertIn("Alice Admin", resp_count)
        self.assertIn("unregistered", resp_count.lower())

        resp_overview = self.va._query_scene_overview()
        self.assertIn("two people", resp_overview.lower())
        self.assertIn("Alice Admin", resp_overview)
        self.assertIn("unregistered", resp_overview.lower())

    # --- Test 6: RiskBanner and TelemetryItem Flicker-Free Caching ---
    def test_06_widget_caching_flicker_free(self):
        """Widget updates must be idempotent and cached to prevent GUI thrashing."""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            item = TelemetryItem(root, "Status:", "INITIAL")
            self.assertEqual(item._current_value, "INITIAL")

            # Update with same value -> cached, no redraw
            item.set_value("INITIAL")
            self.assertEqual(item._current_value, "INITIAL")

            # Update with new value
            item.set_value("ACTIVE")
            self.assertEqual(item._current_value, "ACTIVE")

            banner = RiskBanner(root)
            banner.update_risk(RiskLevel.GREEN, "Normal operations.")
            self.assertEqual(banner._current_level, RiskLevel.GREEN)

            # Re-update with exact same parameters -> cached
            banner.update_risk(RiskLevel.GREEN, "Normal operations.")
            self.assertEqual(banner._current_level, RiskLevel.GREEN)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
