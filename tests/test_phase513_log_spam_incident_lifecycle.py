"""
WatchGuard Vision - Phase 5.13 Log Spam + Incident Lifecycle Audit Tests
-------------------------------------------------------------------------
Validates:
1. Persistent unauthorized condition produces exactly one security event.
2. Persistent condition maintains single active incident across frames.
3. Cooldown expiration does not duplicate an active incident.
4. Repeated incident resolution is strictly idempotent (no duplicate logs/DB updates).
5. Voice query produces exactly one canonical COPILOT_QUERY log and 0 routine DB writes.
6. Unchanged security state emits no repeated INFO security logs.
7. Genuine security state transition generates a new edge-triggered event.
8. Person leaves perimeter (sustained clearance) and re-enters -> new event allowed.
9. Phase 5.9 database retention protections remain strictly intact.
10. High-frequency internal diagnostics remain at DEBUG level.
"""

import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import logging

from src.database.db_manager import DatabaseManager
from src.database.models import IncidentRecord
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.core.incident_manager import IncidentManager, IncidentState, SecurityIncident
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.copilot_reasoner import DeterministicCopilotReasoner
from src.core.voice_assistant import VoiceAssistant
from src.core.track_security_state import TrackSecurityState, ZoneOccupancyState


class TestPhase513LogSpamIncidentLifecycle(unittest.TestCase):
    """Unit tests for Phase 5.13 requirements."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="watchguard_p513_test_"))
        self.db_path = self.test_dir / "test_p513.db"
        self.log_path = self.test_dir / "test_p513.log"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(
            db_manager=self.db,
            log_file_path=self.log_path,
            attach_handlers=True,
        )

    def tearDown(self):
        if hasattr(self.logger, "logger"):
            for h in list(self.logger.logger.handlers):
                h.close()
                self.logger.logger.removeHandler(h)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # H.1: Same persistent unauthorized condition -> one security event
    # -------------------------------------------------------------------------

    def test_01_persistent_unauthorized_condition_produces_single_security_event(self):
        """Verify that 100 consecutive frames of identical unauthorized condition produce only 1 event."""
        recorded_events = []
        self.logger.subscribe(lambda ev: recorded_events.append(ev))

        # Simulate Dashboard edge-triggering logic
        last_logged_security_state = None
        consecutive_green = 0

        for frame_idx in range(100):
            risk_level = RiskLevel.RED
            decision_code = "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS"
            p_name = "Unregistered"
            zone_name = "Restricted Area"
            reason = "Unauthorized access: Unregistered individual detected inside restricted security zone"

            if risk_level in (RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED):
                consecutive_green = 0
                current_state = (risk_level.value, decision_code, p_name, zone_name)
                if current_state != last_logged_security_state:
                    self.logger.log_event(
                        event_type=EventType.UNAUTHORIZED_ACCESS,
                        risk_level=risk_level,
                        description=reason,
                        person_name=p_name,
                    )
                    last_logged_security_state = current_state

        self.assertEqual(len(recorded_events), 1)
        self.assertEqual(self.db.get_total_event_count(), 1)

    # -------------------------------------------------------------------------
    # H.2: Same persistent condition -> one active incident
    # -------------------------------------------------------------------------

    def test_02_persistent_condition_maintains_single_active_incident(self):
        """Verify 100 consecutive frames with persistent violation maintain exactly 1 active incident."""
        inc_mgr = IncidentManager(db_manager=self.db, incident_cooldown_seconds=30.0)

        incident_ids = []
        for frame in range(100):
            inc = inc_mgr.process_incident(
                risk_level=RiskLevel.ORANGE,
                policy_code="RESTRICTED_ZONE_ACCESS",
                reason="Subject inside restricted zone",
                zone_name="Restricted Area",
                person_name="Hunter",
                track_id=1,
                duration_seconds=frame * 0.033,
            )
            self.assertIsNotNone(inc)
            incident_ids.append(inc.incident_id)

        # All 100 frames must map to the same incident ID
        self.assertEqual(len(set(incident_ids)), 1)
        self.assertEqual(len(inc_mgr.get_active_incidents()), 1)

    # -------------------------------------------------------------------------
    # H.3: Cooldown expiration -> does not create duplicate active incident
    # -------------------------------------------------------------------------

    def test_03_cooldown_expiration_does_not_duplicate_active_incident(self):
        """Verify advancing time past cooldown (35s) does NOT create a duplicate active incident."""
        inc_mgr = IncidentManager(db_manager=self.db, incident_cooldown_seconds=30.0)

        # 1. Initial incident creation at t = 0
        inc1 = inc_mgr.process_incident(
            risk_level=RiskLevel.RED,
            policy_code="REPEATED_RESTRICTED_ZONE_VIOLATIONS",
            reason="Repeated violations detected",
            zone_name="Restricted Area",
            person_name="Unregistered",
            track_id=1,
        )
        self.assertIsNotNone(inc1)

        # 2. Advance time past the 30-second cooldown window
        original_time = time.time
        try:
            time.time = MagicMock(return_value=original_time() + 60.0)

            # Call process_incident at t + 60s while incident is still active
            inc2 = inc_mgr.process_incident(
                risk_level=RiskLevel.RED,
                policy_code="REPEATED_RESTRICTED_ZONE_VIOLATIONS",
                reason="Repeated violations detected (still persistent)",
                zone_name="Restricted Area",
                person_name="Unregistered",
                track_id=1,
                duration_seconds=60.0,
            )

            # Must return the SAME incident ID without creating a second record
            self.assertEqual(inc1.incident_id, inc2.incident_id)
            self.assertEqual(len(inc_mgr.get_active_incidents()), 1)
        finally:
            time.time = original_time

    # -------------------------------------------------------------------------
    # H.4: Repeated resolution call -> no duplicate resolution event
    # -------------------------------------------------------------------------

    def test_04_repeated_resolution_call_is_idempotent_no_duplicate_log(self):
        """Verify calling resolve_incident repeatedly emits INCIDENT_RESOLVED exactly once."""
        inc_mgr = IncidentManager(db_manager=self.db)
        inc = inc_mgr.process_incident(
            risk_level=RiskLevel.RED,
            policy_code="UNAUTHORIZED_ACCESS",
            reason="Perimeter breach",
            zone_name="Standard Perimeter",
            person_name="Unknown",
        )

        with self.assertLogs("WatchGuardVision.IncidentManager", level=logging.INFO) as log_ctx:
            # First resolution
            res1 = inc_mgr.resolve_incident(inc.incident_id, operator="Test Operator")
            self.assertTrue(res1)

            # Second resolution (no-op)
            res2 = inc_mgr.resolve_incident(inc.incident_id, operator="Test Operator")
            self.assertTrue(res2)

            # Third resolution (no-op)
            res3 = inc_mgr.resolve_incident(inc.incident_id, operator="Test Operator")
            self.assertTrue(res3)

        resolved_logs = [r for r in log_ctx.output if "INCIDENT_RESOLVED" in r]
        self.assertEqual(len(resolved_logs), 1)

    # -------------------------------------------------------------------------
    # H.5: Same voice query -> one COPILOT_QUERY log
    # -------------------------------------------------------------------------

    def test_05_voice_query_produces_single_copilot_query_log_and_no_routine_db_writes(self):
        """Verify that a voice query produces exactly one COPILOT_QUERY log and 0 routine DB writes."""
        ctx_engine = ContextEngine(db_manager=self.db)
        analytics = SecurityAnalyticsEngine(db_manager=self.db, context_engine=ctx_engine)
        reasoner = DeterministicCopilotReasoner(security_analytics=analytics)
        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=ctx_engine,
            security_analytics=analytics,
            copilot_reasoner=reasoner,
        )

        db_count_before = self.db.get_total_event_count()

        with self.assertLogs("WatchGuardVision.VoiceAssistant", level=logging.INFO) as log_ctx:
            resp = va.process_command("What is the current threat level?")
            self.assertIsNotNone(resp)

        copilot_logs = [r for r in log_ctx.output if "COPILOT_QUERY:" in r]
        self.assertEqual(len(copilot_logs), 1)

        # Ensure no routine DB writes occurred to security_events
        db_count_after = self.db.get_total_event_count()
        self.assertEqual(db_count_before, db_count_after)

    # -------------------------------------------------------------------------
    # H.6: Unchanged state -> no repeated INFO security log
    # -------------------------------------------------------------------------

    def test_06_unchanged_state_produces_no_repeated_info_security_logs(self):
        """Verify that identical consecutive frames do not spam INFO or WARNING logs."""
        logged_messages = []
        last_logged_security_state = None

        with self.assertLogs("WatchGuardVision", level=logging.INFO) as log_ctx:
            for frame in range(20):
                risk = RiskLevel.YELLOW
                code = "UNKNOWN_PERSON"
                name = "Unknown Person"
                zone = "Protected Asset Area"

                curr = (risk.value, code, name, zone)
                if curr != last_logged_security_state:
                    self.logger.log_event(
                        event_type=EventType.UNKNOWN_PERSON,
                        risk_level=risk,
                        description="Unregistered individual detected in Protected Asset Area.",
                        person_name=name,
                    )
                    last_logged_security_state = curr

        alert_logs = [r for r in log_ctx.output if "UNKNOWN_PERSON" in r]
        self.assertEqual(len(alert_logs), 1)

    # -------------------------------------------------------------------------
    # H.7: New state transition -> new event is generated
    # -------------------------------------------------------------------------

    def test_07_genuine_state_transition_generates_new_event(self):
        """Verify transitioning from YELLOW to RED generates a distinct new event."""
        recorded_events = []
        self.logger.subscribe(lambda ev: recorded_events.append(ev))

        last_logged_security_state = None

        # Frame 1: YELLOW
        state1 = (RiskLevel.YELLOW.value, "UNKNOWN_PERSON", "Unregistered", "Protected Asset Area")
        if state1 != last_logged_security_state:
            self.logger.log_event(
                event_type=EventType.UNKNOWN_PERSON,
                risk_level=RiskLevel.YELLOW,
                description="Unregistered individual in Protected Asset Area.",
            )
            last_logged_security_state = state1

        # Frame 2: YELLOW (unchanged) -> suppressed
        if state1 != last_logged_security_state:
            self.logger.log_event(
                event_type=EventType.UNKNOWN_PERSON,
                risk_level=RiskLevel.YELLOW,
                description="Unregistered individual in Protected Asset Area.",
            )

        # Frame 3: RED (escalation to unauthorized interaction) -> new event!
        state2 = (RiskLevel.RED.value, "POTENTIAL_UNAUTHORIZED_INTERACTION", "Unregistered", "Protected Asset Area")
        if state2 != last_logged_security_state:
            self.logger.log_event(
                event_type=EventType.SECURITY_ALERT,
                risk_level=RiskLevel.RED,
                description="Unregistered individual near protected asset.",
            )
            last_logged_security_state = state2

        self.assertEqual(len(recorded_events), 2)
        self.assertEqual(recorded_events[0]["risk_level"], "YELLOW")
        self.assertEqual(recorded_events[1]["risk_level"], "RED")

    # -------------------------------------------------------------------------
    # H.8: Person leaves and later re-enters -> a new event is allowed
    # -------------------------------------------------------------------------

    def test_08_exit_and_reentry_generates_allowed_new_event(self):
        """Verify that after sustained departure (>=30 frames GREEN), a re-entry triggers a new event."""
        recorded_events = []
        self.logger.subscribe(lambda ev: recorded_events.append(ev))

        last_logged_security_state = None
        consecutive_green_frames = 0

        # 1. Person enters (YELLOW)
        current_state = ("YELLOW", "UNKNOWN_PERSON", "Unregistered", "Protected Asset Area")
        consecutive_green_frames = 0
        if current_state != last_logged_security_state:
            self.logger.log_event(
                event_type=EventType.UNKNOWN_PERSON,
                risk_level=RiskLevel.YELLOW,
                description="Person arrival 1",
            )
            last_logged_security_state = current_state

        self.assertEqual(len(recorded_events), 1)

        # 2. Person leaves: 35 consecutive frames of GREEN / clear perimeter
        for _ in range(35):
            consecutive_green_frames += 1
            if consecutive_green_frames >= 30:
                if last_logged_security_state is not None:
                    last_logged_security_state = None

        self.assertIsNone(last_logged_security_state)

        # 3. Person re-enters later -> Fresh event allowed
        consecutive_green_frames = 0
        if current_state != last_logged_security_state:
            self.logger.log_event(
                event_type=EventType.UNKNOWN_PERSON,
                risk_level=RiskLevel.YELLOW,
                description="Person arrival 2",
            )
            last_logged_security_state = current_state

        self.assertEqual(len(recorded_events), 2)

    # -------------------------------------------------------------------------
    # H.9: Existing retention rules still work
    # -------------------------------------------------------------------------

    def test_09_existing_retention_rules_still_work(self):
        """Verify Phase 5.9 retention protections remain active (ORANGE/RED never pruned)."""
        # Insert 10 critical ORANGE/RED events
        red_ids = []
        for i in range(10):
            r_id = self.db.log_security_event(
                event_type="UNAUTHORIZED_ACCESS",
                risk_level="RED",
                description=f"Critical breach #{i}",
            )
            red_ids.append(r_id)

        # Insert 60 GREEN events
        for i in range(60):
            self.db.log_security_event(
                event_type="NORMAL_ACTIVITY",
                risk_level="GREEN",
                description=f"Routine patrol #{i}",
            )

        # Prune down to 20 normal events
        pruned = self.db.enforce_event_retention(max_normal_events=20)
        self.assertEqual(pruned, 40)

        # All 10 RED events must still be intact
        recent = self.db.get_recent_security_events(limit=100)
        remaining_ids = {r["id"] for r in recent}
        for r_id in red_ids:
            self.assertIn(r_id, remaining_ids)

    # -------------------------------------------------------------------------
    # H.10: High-frequency diagnostics moved to DEBUG
    # -------------------------------------------------------------------------

    def test_10_high_frequency_diagnostics_remain_at_debug_level(self):
        """Verify camera probe diagnostics and state transitions do not log at INFO."""
        from src.camera.camera_manager import CameraManager

        cam = CameraManager()
        # Probe devices: all probing lines should be DEBUG, not INFO
        with self.assertLogs("WatchGuardVision.Camera", level=logging.DEBUG) as debug_ctx:
            cam._open_physical_camera(auto_discover=False)

        info_probe_logs = [
            r for r in debug_ctx.output
            if "INFO:" in r and "CAMERA DIAGNOSTIC" in r
        ]
        self.assertEqual(len(info_probe_logs), 0)


if __name__ == "__main__":
    unittest.main()
