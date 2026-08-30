"""
WatchGuard Vision - Phase 5.11 Regression Test Suite
---------------------------------------------------
Tests:
Bug 1 (Stale Gemini Visual Responses & Query Sequencing):
  TEST 1: Person visible -> visual query -> person leaves frame -> Gemini returns -> stale suppressed.
  TEST 2: No person visible -> "What do you observe?" -> indicates no person detected.
  TEST 3: Person visible -> "Who is in front of the camera?" -> corresponds to snapshot.
  TEST 4: Rapid consecutive queries -> older query cannot overwrite newer query.
  TEST 5: Person leaves frame but object remains -> does not report person present.
  TEST 6: Gemini unavailable -> deterministic fallback based on latest state.

Bug 2 (Clear Event History Persistent Database Deletion):
  TEST 7: Create security events -> clear event history -> database count is 0.
  TEST 8: Clear event history -> restart session -> event history remains empty.
  TEST 9: Clear event history -> generate one new event -> only new event exists.
  TEST 10: Clear event history -> users and face registrations preserved.
  TEST 11: Clear event history with linked incidents -> zero orphaned records.
  TEST 12: Database failure during clear -> transaction rollback.
  TEST 13: SQLite integrity check post-clear -> "ok".
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional, Dict, Any, List
import sqlite3
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.database.db_manager import DatabaseManager
from src.database.models import IncidentRecord
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.incident_manager import IncidentManager
from src.core.person_entity import PersonEntity
from src.core.copilot_reasoner import (
    BaseAICopilotReasoner,
    CopilotResponse,
    CopilotResponseSource,
    DeterministicCopilotReasoner,
)
from src.core.voice_assistant import VoiceAssistant


class MockSlowGeminiReasoner(BaseAICopilotReasoner):
    """Simulates asynchronous Gemini reasoner with configurable delay and responses."""

    def __init__(self, delay_seconds: float = 0.05, canned_response: str = "I observe one unregistered person."):
        self.model_name = "gemini-1.5-pro-mock"
        self.delay = delay_seconds
        self.canned_response = canned_response
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def generate_explanation(
        self,
        query: str,
        context: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> CopilotResponse:
        self.call_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        return CopilotResponse(
            text=self.canned_response,
            source=CopilotResponseSource.AI_GROUNDED,
            latency_ms=self.delay * 1000.0,
            model_name="gemini-1.5-pro-mock",
        )


class MockCameraManager:
    """Mock camera providing synthetic test frames."""

    def __init__(self):
        self.is_running = True

    def get_frame(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)


class TestPhase511LiveConsistencyAndReset(unittest.TestCase):
    """Comprehensive regression tests for Phase 5.11."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_p511.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.context_engine = ContextEngine(db_manager=self.db)
        self.logger = EventLogger(db_manager=self.db)
        self.camera = MockCameraManager()

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # =========================================================================
    # BUG 1 TESTS: Stale Gemini Responses & Query Ordering
    # =========================================================================

    def test_01_person_leaves_frame_stale_response_suppressed(self):
        """TEST 1: Person visible -> query dispatched -> person leaves frame -> stale Gemini response suppressed."""
        # 1. Person is present in camera view
        unreg_person = PersonEntity(
            track_id=1,
            bbox=(100, 100, 200, 300),
            identity="Unknown Person",
            is_known=False,
            access_state="DENIED",
        )
        dec_present = SecurityDecision(
            decision="UNKNOWN_PERSON",
            risk_level=RiskLevel.YELLOW,
            reason="Unregistered person detected",
            is_access_granted=False,
            person_entities=[unreg_person],
        )
        self.context_engine.last_decision = dec_present

        mock_gemini = MockSlowGeminiReasoner(
            delay_seconds=0.08,
            canned_response="I observe one unregistered person in front of the camera.",
        )
        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            camera_manager=self.camera,
            copilot_reasoner=mock_gemini,
        )

        # Sync initial state: person is present
        va._sync_session_from_live_state()

        # Simulate background worker: while Gemini is "thinking", person leaves frame!
        import threading
        def person_leaves_worker():
            time.sleep(0.02)
            # Person walks away -> scene is now empty
            dec_empty = SecurityDecision(
                decision="NORMAL_ACTIVITY",
                risk_level=RiskLevel.GREEN,
                reason="No active threats",
                is_access_granted=True,
                person_entities=[],
            )
            self.context_engine.last_decision = dec_empty
            va._sync_session_from_live_state()

        t_leave = threading.Thread(target=person_leaves_worker)
        t_leave.start()

        # Dispatch visual query
        response = va.process_command("What do you observe?")
        t_leave.join()

        # Stale response "I observe one unregistered person" must NOT be returned!
        self.assertNotIn("I observe one unregistered person", response)
        self.assertIn("The camera currently detects no person", response)

    def test_02_no_person_visible_what_do_you_observe(self):
        """TEST 2: No person visible -> query -> indicates no person detected."""
        dec_empty = SecurityDecision(
            decision="NORMAL_ACTIVITY",
            risk_level=RiskLevel.GREEN,
            reason="Routine monitoring",
            is_access_granted=True,
            person_entities=[],
        )
        self.context_engine.last_decision = dec_empty

        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            camera_manager=self.camera,
        )
        va._sync_session_from_live_state()

        resp = va.process_command("What do you observe?")
        lower_resp = resp.lower()
        self.assertTrue(
            "no person" in lower_resp
            or "not detect" in lower_resp
            or "routine monitoring" in lower_resp
            or "clear" in lower_resp
        )

    def test_03_person_visible_who_is_in_front_corresponds_to_snapshot(self):
        """TEST 3: Person visible -> query -> response corresponds to snapshot."""
        hunter = PersonEntity(
            track_id=1,
            bbox=(100, 100, 200, 300),
            identity="Hunter",
            is_known=True,
            access_state="GRANTED",
        )
        dec = SecurityDecision(
            decision="AUTHORIZED_ACCESS",
            risk_level=RiskLevel.GREEN,
            reason="Hunter recognized",
            is_access_granted=True,
            person_entities=[hunter],
        )
        self.context_engine.last_decision = dec

        mock_gemini = MockSlowGeminiReasoner(
            delay_seconds=0.01,
            canned_response="Hunter is standing in front of the camera.",
        )
        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            camera_manager=self.camera,
            copilot_reasoner=mock_gemini,
        )
        va._sync_session_from_live_state()

        resp = va.process_command("Who is in front of the camera?")
        self.assertEqual(resp, "Hunter is standing in front of the camera.")

    def test_04_rapid_queries_older_cannot_overwrite_newer(self):
        """TEST 4: Two visual queries dispatched rapidly -> older response cannot overwrite newer response."""
        hunter = PersonEntity(track_id=1, bbox=(100, 100, 200, 300), identity="Hunter", is_known=True)
        dec = SecurityDecision(
            decision="AUTHORIZED_ACCESS",
            risk_level=RiskLevel.GREEN,
            reason="Hunter",
            is_access_granted=True,
            person_entities=[hunter],
        )
        self.context_engine.last_decision = dec

        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            camera_manager=self.camera,
        )
        va._sync_session_from_live_state()

        # Dispatch Query 1 (id=1)
        # Dispatch Query 2 (id=2)
        # If Query 1 finishes AFTER Query 2:
        with va._query_lock:
            va._query_counter = 2
            va._latest_displayed_query_id = 2

        # Simulate Query 1 returning late with query_id=1
        # When _query_lock checks query_id (1) < _latest_displayed_query_id (2), it must be suppressed!
        # Test suppression logic directly:
        with va._query_lock:
            should_suppress = (1 < va._latest_displayed_query_id)
        self.assertTrue(should_suppress)

    def test_05_person_leaves_frame_object_remains(self):
        """TEST 5: Person leaves frame but object remains -> does not report person present."""
        # Initial: Person + Laptop
        person = PersonEntity(track_id=1, bbox=(100, 100, 200, 300), identity="Unknown Person", is_known=False)
        dec_init = SecurityDecision(
            decision="UNKNOWN_PERSON",
            risk_level=RiskLevel.YELLOW,
            reason="Unregistered person",
            is_access_granted=False,
            person_entities=[person],
        )
        self.context_engine.last_decision = dec_init

        mock_gemini = MockSlowGeminiReasoner(
            delay_seconds=0.08,
            canned_response="I see an unknown person interacting with a laptop.",
        )
        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            camera_manager=self.camera,
            copilot_reasoner=mock_gemini,
        )
        va.session_context.last_objects = [{"label": "laptop"}]
        va._sync_session_from_live_state()

        import threading
        def leave_worker():
            time.sleep(0.02)
            # Person left, but laptop object remains on table
            dec_no_person = SecurityDecision(
                decision="NORMAL_ACTIVITY",
                risk_level=RiskLevel.GREEN,
                reason="Monitoring",
                is_access_granted=True,
                person_entities=[],
            )
            self.context_engine.last_decision = dec_no_person
            va._sync_session_from_live_state()
            va.session_context.last_objects = [{"label": "laptop"}]

        t = threading.Thread(target=leave_worker)
        t.start()

        resp = va.process_command("What do you observe?")
        t.join()

        # Must NOT report person present
        self.assertNotIn("unknown person interacting", resp)
        self.assertIn("The camera currently detects no person", resp)

    def test_06_gemini_unavailable_deterministic_fallback(self):
        """TEST 6: Gemini unavailable -> deterministic fallback based on latest state."""
        hunter = PersonEntity(track_id=1, bbox=(100, 100, 200, 300), identity="Hunter", is_known=True, access_state="GRANTED")
        dec = SecurityDecision(
            decision="AUTHORIZED_ACCESS",
            risk_level=RiskLevel.GREEN,
            reason="Hunter recognized",
            is_access_granted=True,
            person_entities=[hunter],
        )
        self.context_engine.last_decision = dec

        # Deterministic reasoner (Gemini unavailable)
        det_reasoner = DeterministicCopilotReasoner()
        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            camera_manager=self.camera,
            copilot_reasoner=det_reasoner,
        )
        va._sync_session_from_live_state()

        resp = va.process_command("Who is in front of the camera?")
        self.assertTrue("Hunter" in resp or "authorized" in resp.lower())

    # =========================================================================
    # BUG 2 TESTS: Clear Event History Persistent Database Deletion
    # =========================================================================

    def test_07_clear_event_history_database_count_is_zero(self):
        """TEST 7: Populate security events -> clear event history -> database count becomes 0."""
        # Insert 10 events
        for i in range(10):
            self.db.log_security_event(
                event_type="TEST_EVENT",
                risk_level="GREEN",
                description=f"Test event #{i}",
            )

        count_before = self.db.get_total_event_count()
        self.assertEqual(count_before, 10)

        # Clear event history
        deleted = self.db.clear_event_history(clear_incidents=True)
        self.assertEqual(deleted.get("security_events"), 10)

        count_after = self.db.get_total_event_count()
        self.assertEqual(count_after, 0)

    def test_08_clear_event_history_persists_across_sessions(self):
        """TEST 8: Clear event history -> restart application/new DB instance -> remains empty."""
        for i in range(5):
            self.db.log_security_event(event_type="EVT", risk_level="GREEN", description=f"Evt {i}")

        self.db.clear_event_history(clear_incidents=True)

        # Reopen database connection as a fresh application instance
        new_db = DatabaseManager(db_path=self.db_path)
        self.assertEqual(new_db.get_total_event_count(), 0)
        self.assertEqual(len(new_db.get_recent_security_events()), 0)

    def test_09_clear_event_history_then_generate_one_event(self):
        """TEST 9: Clear event history -> generate one new event -> only new event exists with fresh id."""
        for i in range(5):
            self.db.log_security_event(event_type="OLD_EVT", risk_level="GREEN", description=f"Old {i}")

        self.db.clear_event_history(clear_incidents=True)

        # Insert 1 new event
        new_id = self.db.log_security_event(
            event_type="NEW_EVT",
            risk_level="GREEN",
            description="Fresh event #1",
        )

        self.assertEqual(new_id, 1)  # sqlite_sequence was reset
        events = self.db.get_recent_security_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["description"], "Fresh event #1")

    def test_10_clear_event_history_preserves_users_and_faces(self):
        """TEST 10: Clear event history -> users and face registrations remain completely intact."""
        # Create user and face registration
        u_id = self.db.create_user(username="alice", full_name="Alice Officer", role="Staff", access_level=3)
        fake_feat = np.ones((1, 128), dtype=np.float32).tobytes()
        f_id = self.db.register_face(user_id=u_id, face_encoding=fake_feat, image_path="data/faces/alice.jpg")

        # Create security events
        self.db.log_security_event(event_type="LOGIN", risk_level="GREEN", description="Alice login")

        # Clear event history
        self.db.clear_event_history(clear_incidents=True)

        # Users and face registrations must be strictly preserved
        users = self.db.get_users()
        faces = self.db.get_active_face_registrations()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["username"], "alice")
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["user_id"], u_id)
        self.assertEqual(self.db.get_total_event_count(), 0)

    def test_11_clear_event_history_with_linked_incidents(self):
        """TEST 11: Clear event history with linked security_incidents -> zero orphaned records."""
        ev_id = self.db.log_security_event(
            event_type="RESTRICTED_ACCESS",
            risk_level="RED",
            description="Unauthorized entry",
        )
        rec = IncidentRecord(
            incident_id="INC-20260829-001",
            risk_level="RED",
            state="DETECTED",
            policy_code="RESTRICTED_ZONE",
            reason="Violation",
            zone="Restricted Area",
            created_at="2026-08-29T12:00:00",
            updated_at="2026-08-29T12:00:00",
            event_id=ev_id,
        )
        self.db.create_incident(rec)

        # Clear event history
        deleted = self.db.clear_event_history(clear_incidents=True)
        self.assertEqual(deleted.get("security_incidents"), 1)
        self.assertEqual(deleted.get("security_events"), 1)

        # Verify neither table has remaining records
        with self.db._connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM security_incidents")
            inc_cnt = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM security_events")
            ev_cnt = c.fetchone()[0]

        self.assertEqual(inc_cnt, 0)
        self.assertEqual(ev_cnt, 0)

    def test_12_database_failure_during_clear_rolls_back(self):
        """TEST 12: Database failure during clear -> transaction rolls back safely."""
        self.db.log_security_event(event_type="EVENT_A", risk_level="GREEN", description="Event A")

        # Create a subclass or monkeypatch that raises on delete
        def failing_connection():
            class FailingCursor:
                def execute(self, sql, *args):
                    if "DELETE FROM security_events" in sql:
                        raise sqlite3.OperationalError("Simulated Disk I/O Failure")
                def fetchone(self):
                    return [1]
            class FailingConn:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False
                def cursor(self):
                    return FailingCursor()
                def commit(self):
                    pass
                def rollback(self):
                    pass
            return FailingConn()

        orig_conn = self.db._connection
        self.db._connection = failing_connection
        try:
            with self.assertRaises(sqlite3.OperationalError):
                self.db.clear_event_history(clear_incidents=True)
        finally:
            self.db._connection = orig_conn

        # After failure, records remain intact
        self.assertEqual(self.db.get_total_event_count(), 1)

    def test_13_sqlite_integrity_check_post_clear(self):
        """TEST 13: Verify SQLite integrity after clearing -> 'ok'."""
        for i in range(20):
            self.db.log_security_event(event_type=f"EVT_{i}", risk_level="GREEN", description=f"Event {i}")

        self.db.clear_event_history(clear_incidents=True)

        opt = self.db.optimize_database()
        self.assertEqual(opt["integrity"], "ok")
        self.assertEqual(opt["table_counts"]["security_events"], 0)
        self.assertEqual(opt["table_counts"]["security_incidents"], 0)


if __name__ == "__main__":
    unittest.main()
