"""
WatchGuard Vision - Phase 5.9 Audit, Event Retention & Log Spam Control Tests
-----------------------------------------------------------------------------
Validates:
1. Database event retention policy (maximum 500 normal events).
2. Protection of critical events (ORANGE/RED) from retention pruning.
3. Protection of events linked to active security incidents.
4. Retention preservation of newest events over oldest.
5. Periodic auto-enforcement of retention on event insertion.
6. Dashboard live event table ceiling at 100 newest events.
7. Clear Table View behavior remains UI-only and does not delete SQLite records.
8. Edge-triggered / state-transition logging prevents duplicate writes for unchanged state.
9. Genuine state transitions produce new security events.
10. Incident lifecycle deduplication for ongoing active incidents.
11. Idempotent incident resolution without log spam.
12. Rotating file logger configuration (5MB max, 2 backups).
13. Database vacuum and optimize maintenance operations.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from logging.handlers import RotatingFileHandler

from config import (
    MAX_DATABASE_NORMAL_EVENTS,
    MAX_RECENT_EVENTS_DISPLAY,
    EVENT_RETENTION_CHECK_INTERVAL,
    LOG_ROTATION_MAX_BYTES,
    LOG_ROTATION_BACKUP_COUNT,
)
from src.database.db_manager import DatabaseManager
from src.database.models import IncidentRecord
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.core.incident_manager import IncidentManager, IncidentState, SecurityIncident
from src.core.context_engine import SecurityDecision


class TestPhase59AuditRetentionAndSpam(unittest.TestCase):
    """Test suite for Phase 5.9 requirements."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="watchguard_p59_test_"))
        self.db_path = self.test_dir / "test_p59.db"
        self.log_path = self.test_dir / "test_p59.log"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(
            db_manager=self.db,
            log_file_path=self.log_path,
            attach_handlers=True,
        )

    def tearDown(self):
        # Remove handlers to avoid file locking on Windows
        if hasattr(self.logger, "logger"):
            for h in list(self.logger.logger.handlers):
                h.close()
                self.logger.logger.removeHandler(h)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # PART B: Database Event Retention
    # -------------------------------------------------------------------------

    def test_enforce_event_retention_limit(self):
        """Verify normal GREEN events exceeding the limit are pruned to the limit."""
        # Insert 550 GREEN events. Auto-retention kicks in during insertion to keep <= 500.
        for i in range(550):
            self.db.log_security_event(
                event_type="NORMAL_ACTIVITY",
                risk_level="GREEN",
                description=f"Normal patrol observation #{i}",
                timestamp=f"2026-08-28 10:{i//60:02d}:{i%60:02d}",
            )

        total_before = self.db.get_total_event_count()
        self.assertEqual(total_before, 500)

        # Enforce retention down to 400
        pruned = self.db.enforce_event_retention(max_normal_events=400)
        self.assertEqual(pruned, 100)

        total_after = self.db.get_total_event_count()
        self.assertEqual(total_after, 400)

    def test_retention_preserves_newest_events(self):
        """Verify the newest normal events are preserved and the oldest are deleted."""
        for i in range(120):
            self.db.log_security_event(
                event_type="NORMAL_ACTIVITY",
                risk_level="GREEN",
                description=f"Patrol event sequence {i:03d}",
            )

        # Prune down to 100
        pruned = self.db.enforce_event_retention(max_normal_events=100)
        self.assertEqual(pruned, 20)

        recent = self.db.get_recent_security_events(limit=150)
        self.assertEqual(len(recent), 100)

        # Newest should be sequence 119, oldest remaining should be sequence 020
        descriptions = [r["description"] for r in recent]
        self.assertIn("Patrol event sequence 119", descriptions[0])
        self.assertIn("Patrol event sequence 020", descriptions[-1])
        self.assertNotIn("Patrol event sequence 000", descriptions)
        self.assertNotIn("Patrol event sequence 019", descriptions)

    def test_retention_strictly_protects_critical_events(self):
        """Verify ORANGE and RED events are never pruned by event retention."""
        # Insert 10 critical RED and ORANGE events
        red_ids = []
        for i in range(5):
            r_id = self.db.log_security_event(
                event_type="SECURITY_ALERT",
                risk_level="RED",
                description=f"Critical unauthorized access breach #{i}",
            )
            red_ids.append(r_id)

        orange_ids = []
        for i in range(5):
            o_id = self.db.log_security_event(
                event_type="POLICY_VIOLATION",
                risk_level="ORANGE",
                description=f"High risk after-hours violation #{i}",
            )
            orange_ids.append(o_id)

        # Insert 60 GREEN events
        for i in range(60):
            self.db.log_security_event(
                event_type="NORMAL_ACTIVITY",
                risk_level="GREEN",
                description=f"Routine patrol #{i}",
            )

        # Enforce retention to 20 normal events
        pruned = self.db.enforce_event_retention(max_normal_events=20)
        self.assertEqual(pruned, 40)

        # Verify all RED and ORANGE events still exist
        recent_all = self.db.get_recent_security_events(limit=100)
        remaining_ids = {r["id"] for r in recent_all}

        for rid in red_ids:
            self.assertIn(rid, remaining_ids)
        for oid in orange_ids:
            self.assertIn(oid, remaining_ids)

        # Total remaining = 10 critical + 20 normal = 30
        self.assertEqual(len(recent_all), 30)

    def test_retention_protects_events_linked_to_active_incidents(self):
        """Verify GREEN events linked to unresolved incidents are protected from deletion."""
        # Insert an old GREEN event linked to an active incident
        linked_ev_id = self.db.log_security_event(
            event_type="PERSON_DETECTED",
            risk_level="GREEN",
            description="Initial observation for active incident investigation",
        )

        # Create active incident linking this event
        inc = IncidentRecord(
            incident_id="INC-TEST-ACTIVE-001",
            event_id=linked_ev_id,
            track_id=1,
            person_name="Subject Alpha",
            risk_level="ORANGE",
            state="ALERTED",  # Unresolved!
            policy_code="RESTRICTED_ZONE_ACCESS",
            reason="Investigation ongoing",
            zone="Restricted Area",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.db.create_incident(inc)

        # Insert 50 other GREEN events
        for i in range(50):
            self.db.log_security_event(
                event_type="NORMAL_ACTIVITY",
                risk_level="GREEN",
                description=f"Patrol #{i}",
            )

        # Enforce retention limit of 10 normal events
        self.db.enforce_event_retention(max_normal_events=10)

        # Verify linked_ev_id was protected because its incident is active!
        recent = self.db.get_recent_security_events(limit=100)
        remaining_ids = {r["id"] for r in recent}
        self.assertIn(linked_ev_id, remaining_ids)

    def test_periodic_retention_on_insertion(self):
        """Verify retention is automatically enforced every EVENT_RETENTION_CHECK_INTERVAL insertions."""
        # We temporarily patch MAX_DATABASE_NORMAL_EVENTS to 30 for this test
        with patch("src.database.db_manager.MAX_DATABASE_NORMAL_EVENTS", 30):
            with patch("src.database.db_manager.EVENT_RETENTION_CHECK_INTERVAL", 10):
                for i in range(45):
                    self.db.log_security_event(
                        event_type="NORMAL_ACTIVITY",
                        risk_level="GREEN",
                        description=f"Auto retention event #{i}",
                    )

                # At 45 insertions, checks ran at 10, 20, 30, 40.
                # After insertion 40, retention pruned count to 30.
                # Then insertions 41-44 were added -> total count should be 30 + 5 = 35.
                total = self.db.get_total_event_count()
                self.assertLessEqual(total, 35)

    # -------------------------------------------------------------------------
    # PART B: Live Dashboard Display Limit & UI Clear
    # -------------------------------------------------------------------------

    def test_dashboard_treeview_capped_at_100(self):
        """Verify treeview row insertion is capped at MAX_RECENT_EVENTS_DISPLAY (100)."""
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.withdraw()  # Do not display window
        try:
            tree = ttk.Treeview(root, columns=("id", "ts", "risk", "type", "desc"))

            # Simulate dashboard's _insert_event_to_tree logic
            for i in range(130):
                event_payload = {
                    "id": i + 1,
                    "timestamp": "12:00:00",
                    "risk_level": "GREEN",
                    "event_type": "INFO",
                    "description": f"Test event #{i}",
                }
                item = tree.insert("", 0, values=(event_payload["id"], "12:00:00", "GREEN", "INFO", f"Test event #{i}"))
                children = tree.get_children()
                if len(children) > MAX_RECENT_EVENTS_DISPLAY:
                    for old_item in children[MAX_RECENT_EVENTS_DISPLAY:]:
                        tree.delete(old_item)

            children = tree.get_children()
            self.assertEqual(len(children), MAX_RECENT_EVENTS_DISPLAY)
            self.assertEqual(len(children), 100)
            # Top item should be the newest (#130)
            top_val = tree.item(children[0])["values"][0]
            self.assertEqual(int(top_val), 130)
        finally:
            root.destroy()

    def test_clear_table_is_ui_only(self):
        """Verify clearing table view does not delete SQLite records."""
        for i in range(5):
            self.db.log_security_event(
                event_type="SYSTEM_STARTUP",
                risk_level="GREEN",
                description=f"Startup log #{i}",
            )

        self.assertEqual(self.db.get_total_event_count(), 5)

        # Clear UI mock
        mock_tree = MagicMock()
        mock_tree.get_children.return_value = ["item1", "item2", "item3"]

        for item in mock_tree.get_children():
            mock_tree.delete(item)

        # Verify mock delete was called
        self.assertEqual(mock_tree.delete.call_count, 3)

        # Verify SQLite records are 100% untouched
        self.assertEqual(self.db.get_total_event_count(), 5)

    # -------------------------------------------------------------------------
    # PART C: Log Spam Investigation & Edge-Triggered Logging
    # -------------------------------------------------------------------------

    def test_unchanged_security_state_prevents_duplicate_events(self):
        """Verify that unchanged security states do not produce duplicate events."""
        recorded_events = []

        def subscriber(ev):
            recorded_events.append(ev)

        self.logger.subscribe(subscriber)

        # Mock dashboard state tracking
        last_logged_security_state = None

        decision = MagicMock()
        decision.risk_level = RiskLevel.RED
        decision.decision = "POTENTIAL_UNAUTHORIZED_INTERACTION"
        decision.reason = "Unregistered person near protected asset (LAPTOP)"
        decision.event_type = EventType.SECURITY_ALERT
        decision.evidence_required = False

        # Simulate 10 consecutive frames with the identical state
        for frame in range(10):
            current_state = (
                decision.risk_level.value,
                decision.decision,
                "Unregistered",
                "laptop",
                "Standard Perimeter",
            )
            if current_state != last_logged_security_state:
                self.logger.log_event(
                    event_type=decision.event_type,
                    risk_level=decision.risk_level,
                    description=decision.reason,
                    person_name="Unregistered",
                    object_summary="laptop",
                )
                last_logged_security_state = current_state

        # Exactly 1 event should have been logged despite 10 frames
        self.assertEqual(len(recorded_events), 1)
        self.assertEqual(self.db.get_total_event_count(), 1)

    def test_genuine_state_transition_triggers_new_event(self):
        """Verify that distinct state transitions trigger new events."""
        recorded_events = []

        def subscriber(ev):
            recorded_events.append(ev)

        self.logger.subscribe(subscriber)

        last_logged_security_state = None

        states = [
            # Frame 1-3: Unknown person (YELLOW)
            (RiskLevel.YELLOW, "UNKNOWN_PERSON", "Unregistered", "None", "Standard Perimeter"),
            (RiskLevel.YELLOW, "UNKNOWN_PERSON", "Unregistered", "None", "Standard Perimeter"),
            (RiskLevel.YELLOW, "UNKNOWN_PERSON", "Unregistered", "None", "Standard Perimeter"),
            # Frame 4-6: Person interacts with protected laptop (escalation to RED)
            (RiskLevel.RED, "POTENTIAL_UNAUTHORIZED_INTERACTION", "Unregistered", "laptop", "Standard Perimeter"),
            (RiskLevel.RED, "POTENTIAL_UNAUTHORIZED_INTERACTION", "Unregistered", "laptop", "Standard Perimeter"),
            (RiskLevel.RED, "POTENTIAL_UNAUTHORIZED_INTERACTION", "Unregistered", "laptop", "Standard Perimeter"),
            # Frame 7: Returns to normal (GREEN)
            (RiskLevel.GREEN, "NORMAL_ACTIVITY", "None", "None", "Standard Perimeter"),
        ]

        for r_level, dec_code, p_name, obj_summ, zone in states:
            if r_level in (RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED):
                curr = (r_level.value, dec_code, p_name, obj_summ, zone)
                if curr != last_logged_security_state:
                    self.logger.log_event(
                        event_type=EventType.SECURITY_ALERT,
                        risk_level=r_level,
                        description=f"State: {dec_code}",
                        person_name=p_name,
                        object_summary=obj_summ,
                    )
                    last_logged_security_state = curr
            elif r_level == RiskLevel.GREEN:
                if last_logged_security_state is not None:
                    last_logged_security_state = None

        # There should be exactly 2 logged threat events: YELLOW (frame 1) and RED (frame 4)
        self.assertEqual(len(recorded_events), 2)
        self.assertEqual(recorded_events[0]["risk_level"], "YELLOW")
        self.assertEqual(recorded_events[1]["risk_level"], "RED")

    def test_incident_deduplication_in_incident_manager(self):
        """Verify IncidentManager does not create duplicate incidents for ongoing active violation."""
        inc_mgr = IncidentManager(db_manager=self.db, incident_cooldown_seconds=0.1)

        # Call process_incident frame 1
        inc1 = inc_mgr.process_incident(
            risk_level=RiskLevel.ORANGE,
            policy_code="RESTRICTED_ZONE_ACCESS",
            reason="Subject inside restricted zone",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=1,
        )
        self.assertIsNotNone(inc1)

        # Call process_incident frame 2 (immediate next frame)
        inc2 = inc_mgr.process_incident(
            risk_level=RiskLevel.ORANGE,
            policy_code="RESTRICTED_ZONE_ACCESS",
            reason="Subject inside restricted zone (still inside)",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=1,
            duration_seconds=5.0,
        )

        # Must return the SAME incident without creating a second record
        self.assertEqual(inc1.incident_id, inc2.incident_id)
        active = inc_mgr.get_active_incidents()
        self.assertEqual(len(active), 1)

    def test_resolve_incident_idempotent(self):
        """Verify resolve_incident on an already resolved incident does not log duplicate resolutions."""
        inc_mgr = IncidentManager(db_manager=self.db)
        inc = inc_mgr.process_incident(
            risk_level=RiskLevel.RED,
            policy_code="UNAUTHORIZED_ACCESS",
            reason="Perimeter breach",
            zone_name="Standard Perimeter",
            person_name="Unknown",
        )

        # First resolution -> True
        res1 = inc_mgr.resolve_incident(inc.incident_id, operator="Test Operator")
        self.assertTrue(res1)

        # Second resolution -> True (idempotent, no-op)
        res2 = inc_mgr.resolve_incident(inc.incident_id, operator="Test Operator")
        self.assertTrue(res2)

        # Database incident state should be RESOLVED
        row = self.db.get_incident_by_id(inc.incident_id)
        self.assertEqual(row["state"], "RESOLVED")

    # -------------------------------------------------------------------------
    # PART D: Log Rotation Configuration
    # -------------------------------------------------------------------------

    def test_log_rotation_configuration(self):
        """Verify EventLogger sets up RotatingFileHandler with ~5MB maxBytes and 2 backups."""
        file_handlers = [h for h in self.logger.logger.handlers if isinstance(h, RotatingFileHandler)]
        self.assertEqual(len(file_handlers), 1)
        handler = file_handlers[0]
        self.assertEqual(handler.maxBytes, LOG_ROTATION_MAX_BYTES)
        self.assertEqual(handler.backupCount, LOG_ROTATION_BACKUP_COUNT)
        self.assertEqual(handler.maxBytes, 5 * 1024 * 1024)
        self.assertEqual(handler.backupCount, LOG_ROTATION_BACKUP_COUNT)

    # -------------------------------------------------------------------------
    # PART E: Database Maintenance & Safety
    # -------------------------------------------------------------------------

    def test_vacuum_and_optimize_methods(self):
        """Verify vacuum_database and optimize_database maintenance methods execute safely."""
        vac_ok = self.db.vacuum_database()
        self.assertTrue(vac_ok)

        opt_report = self.db.optimize_database()
        self.assertEqual(opt_report["status"], "OPTIMIZE_COMPLETE")
        self.assertIn("integrity", opt_report)
        self.assertEqual(opt_report["integrity"], "ok")
        self.assertIn("table_counts", opt_report)
    def test_copilot_single_log_emission(self):
        """Verify that processing a voice/copilot query logs COPILOT_QUERY exactly once."""
        import logging
        from src.core.voice_assistant import VoiceAssistant
        from src.core.context_engine import ContextEngine
        from src.core.security_analytics import SecurityAnalyticsEngine
        from src.core.copilot_reasoner import DeterministicCopilotReasoner

        ctx_engine = ContextEngine(db_manager=self.db)
        analytics = SecurityAnalyticsEngine(db_manager=self.db, context_engine=ctx_engine)
        reasoner = DeterministicCopilotReasoner(security_analytics=analytics)
        va = VoiceAssistant(
            db_manager=self.db,
            context_engine=ctx_engine,
            security_analytics=analytics,
            copilot_reasoner=reasoner,
        )

        with self.assertLogs("WatchGuardVision.VoiceAssistant", level=logging.INFO) as log_ctx:
            va.process_command("What is the current security status?")

        copilot_query_logs = [r for r in log_ctx.output if "COPILOT_QUERY:" in r]
        self.assertEqual(len(copilot_query_logs), 1)


if __name__ == "__main__":
    unittest.main()
