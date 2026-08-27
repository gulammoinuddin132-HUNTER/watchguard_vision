"""
Unit and Integration Tests for WatchGuard 2.0 Phase 4
------------------------------------------------------
Temporal Policy + Incident Intelligence Test Suite:
1. Authorized user + normal area + daytime
2. Authorized user + restricted area + daytime
3. Authorized user + restricted area + after-hours
4. Authorized user + normal/protected area + after-hours
5. Unknown + restricted area
6. Spoof + restricted area
7. Temporal PAD instability (oscillation between live and spoof)
8. Stable multi-frame authorization
9. Repeated restricted-zone violations (ORANGE -> RED escalation)
10. Repeated spoof attempts (RED + escalation)
11. Incident acknowledge operator action
12. Incident resolve operator action
13. Per-face track state isolation
14. Voice assistant queries against live incident & violation state
"""

import unittest
from datetime import datetime, time as dtime
import tempfile
from pathlib import Path

from src.events.event_types import RiskLevel, EventType
from src.database.db_manager import DatabaseManager
from src.database.models import IncidentRecord
from src.core.policy_engine import PolicyEngine, ZonePolicy, ExplainableDecisionTuple
from src.core.track_security_state import TrackSecurityStateManager, TrackSecurityState
from src.core.incident_manager import IncidentManager, IncidentState
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant, VoiceIntent


class TestTemporalPolicyPhase4(unittest.TestCase):
    """Test suite verifying all Phase 4 specifications."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_watchguard_phase4.db"
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.context_engine = ContextEngine(db_manager=self.db_manager)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Authorized user + normal area + daytime
    def test_01_authorized_user_normal_area_daytime(self):
        self.context_engine.set_time_override(dtime(14, 0))  # 14:00 daytime
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "has_verifying": False,
            "people": [{
                "track_id": 1,
                "name": "Hunter",
                "is_known": True,
                "role": "Security Engineer",
                "similarity": 0.88,
                "is_spoof": False,
                "liveness_confidence": 0.95,
                "bbox": (50, 50, 100, 100),  # In normal zone
            }],
        }
        dec = self.context_engine.evaluate_context(person_data, {"objects": [], "summary": "None"})
        self.assertTrue(dec.is_access_granted)
        self.assertEqual(dec.risk_level, RiskLevel.GREEN)
        self.assertEqual(dec.decision, "NORMAL_ACTIVITY")
        self.assertIsNotNone(dec.explainable_tuple)
        self.assertIn("Hunter", dec.explainable_tuple.who)
        self.assertIn("ACCESS GRANTED", dec.explainable_tuple.result)

    # 2. Authorized user + restricted area + daytime
    def test_02_authorized_user_restricted_area_daytime(self):
        self.context_engine.set_time_override(dtime(14, 0))
        self.context_engine.set_zone_override("RESTRICTED")
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "has_verifying": False,
            "people": [{
                "track_id": 1,
                "name": "Hunter",
                "is_known": True,
                "role": "Security Engineer",
                "similarity": 0.88,
                "is_spoof": False,
                "liveness_confidence": 0.95,
                "bbox": (400, 50, 100, 100),
            }],
        }
        dec = self.context_engine.evaluate_context(person_data, {"objects": [], "summary": "None"})
        self.assertTrue(dec.is_access_granted)
        self.assertEqual(dec.risk_level, RiskLevel.GREEN)
        self.assertEqual(dec.decision, "AUTHORIZED_RESTRICTED_ZONE_ACCESS")
        self.assertIn("Restricted Area", dec.explainable_tuple.where)

    # 3. Authorized user + restricted area + after-hours
    def test_03_authorized_user_restricted_area_after_hours(self):
        self.context_engine.set_time_override(dtime(22, 45))  # After-hours
        self.context_engine.set_zone_override("RESTRICTED")
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "has_verifying": False,
            "people": [{
                "track_id": 1,
                "name": "Hunter",
                "is_known": True,
                "role": "Security Engineer",
                "similarity": 0.88,
                "is_spoof": False,
                "liveness_confidence": 0.95,
                "bbox": (400, 50, 100, 100),
            }],
        }
        dec = self.context_engine.evaluate_context(person_data, {"objects": [], "summary": "None"})
        self.assertFalse(dec.is_access_granted)
        self.assertEqual(dec.risk_level, RiskLevel.ORANGE)
        self.assertEqual(dec.decision, "AFTER_HOURS_RESTRICTED_ACCESS_DENIED")
        self.assertIsNotNone(dec.incident)
        self.assertEqual(dec.incident.risk_level, RiskLevel.ORANGE)

    # 4. Authorized user + normal/protected area + after-hours (24/7 access granted)
    def test_04_authorized_user_normal_area_after_hours(self):
        self.context_engine.set_time_override(dtime(22, 45))  # After-hours
        self.context_engine.set_zone_override("NORMAL")
        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "has_verifying": False,
            "people": [{
                "track_id": 1,
                "name": "Hunter",
                "is_known": True,
                "role": "Security Engineer",
                "similarity": 0.88,
                "is_spoof": False,
                "liveness_confidence": 0.95,
                "bbox": (50, 50, 100, 100),
            }],
        }
        dec = self.context_engine.evaluate_context(person_data, {"objects": [], "summary": "None"})
        self.assertTrue(dec.is_access_granted)
        self.assertEqual(dec.risk_level, RiskLevel.GREEN)
        self.assertEqual(dec.decision, "NORMAL_ACTIVITY")

    # 5. Unknown + restricted area
    def test_05_unknown_person_restricted_area(self):
        self.context_engine.set_time_override(dtime(14, 0))
        self.context_engine.set_zone_override("RESTRICTED")
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": True,
            "has_spoof": False,
            "has_verifying": False,
            "people": [{
                "track_id": 5,
                "name": "Unknown Person",
                "is_known": False,
                "role": "Visitor",
                "confidence": 0.92,
                "is_spoof": False,
                "liveness_confidence": 0.90,
                "bbox": (400, 50, 100, 100),
            }],
        }
        dec = self.context_engine.evaluate_context(person_data, {"objects": [], "summary": "None"})
        self.assertFalse(dec.is_access_granted)
        self.assertEqual(dec.risk_level, RiskLevel.RED)
        self.assertEqual(dec.decision, "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS")

    # 6. Spoof + restricted area
    def test_06_spoof_attack_restricted_area(self):
        self.context_engine.set_time_override(dtime(14, 0))
        self.context_engine.set_zone_override("RESTRICTED")
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": False,
            "has_spoof": True,
            "has_verifying": False,
            "people": [{
                "track_id": 9,
                "name": "Hunter",
                "is_known": True,
                "role": "Security Engineer",
                "similarity": 0.86,
                "is_spoof": True,
                "attack_type": "PHOTO_REPLAY_ATTACK",
                "liveness_confidence": 0.12,
                "bbox": (400, 50, 100, 100),
            }],
        }
        dec = self.context_engine.evaluate_context(person_data, {"objects": [], "summary": "None"})
        self.assertFalse(dec.is_access_granted)
        self.assertEqual(dec.risk_level, RiskLevel.RED)
        self.assertEqual(dec.decision, "PRESENTATION_ATTACK_IMPERSONATION")
        self.assertIsNotNone(dec.incident)

    # 7. Temporal PAD instability (Oscillation between live and spoof)
    def test_07_temporal_pad_instability_access_pending(self):
        t_mgr = TrackSecurityStateManager()
        t_state = t_mgr.get_or_create_track(15)

        # Oscillating updates (LIVE -> SPOOF -> LIVE -> WARMUP)
        t_state.update(
            identity="Hunter", identity_conf=0.85, role="Staff", is_known=True,
            liveness_state="LIVE", liveness_conf=0.90, is_spoof=False, attack_type=None,
            zone_id="NORMAL", zone_name="Perimeter", time_str="14:00", is_auth_time=True,
            nearby_objects=[], policy_code="LIVENESS_VERIFYING", is_access_granted=False,
            risk_level=RiskLevel.YELLOW, reason="Warmup",
        )
        t_state.update(
            identity="Hunter", identity_conf=0.85, role="Staff", is_known=True,
            liveness_state="SPOOF", liveness_conf=0.40, is_spoof=True, attack_type="PHOTO",
            zone_id="NORMAL", zone_name="Perimeter", time_str="14:00", is_auth_time=True,
            nearby_objects=[], policy_code="LIVENESS_VERIFYING", is_access_granted=False,
            risk_level=RiskLevel.YELLOW, reason="Oscillating",
        )
        # Hysteresis prevents immediate confirmation of LIVE
        self.assertFalse(t_state.confirmed_live)
        self.assertEqual(t_state.access_state, "SPOOF_REJECTED")

    # 8. Stable multi-frame authorization
    def test_08_stable_multi_frame_authorization(self):
        t_mgr = TrackSecurityStateManager()
        t_state = t_mgr.get_or_create_track(20)

        # Feed 3 consecutive live frames
        for _ in range(3):
            t_state.update(
                identity="Hunter", identity_conf=0.88, role="Staff", is_known=True,
                liveness_state="LIVE", liveness_conf=0.95, is_spoof=False, attack_type=None,
                zone_id="NORMAL", zone_name="Perimeter", time_str="14:00", is_auth_time=True,
                nearby_objects=[], policy_code="NORMAL_ACTIVITY", is_access_granted=True,
                risk_level=RiskLevel.GREEN, reason="Verified",
            )

        self.assertTrue(t_state.confirmed_live)
        self.assertEqual(t_state.access_state, "GRANTED")
        self.assertEqual(t_state.risk_level, RiskLevel.GREEN)

    # 9. Repeated restricted-zone violations (ORANGE -> RED escalation)
    def test_09_repeated_restricted_zone_violations_escalation(self):
        self.context_engine.set_time_override(dtime(22, 45))

        person_data_restricted = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 7, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (400, 50, 100, 100)}],
        }
        person_data_normal = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 7, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (50, 50, 100, 100)}],
        }

        # Entry 1: After-hours violation
        self.context_engine.set_zone_override("RESTRICTED")
        dec1 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        self.assertEqual(dec1.risk_level, RiskLevel.ORANGE)

        # Exit 1
        self.context_engine.set_zone_override("NORMAL")
        self.context_engine.evaluate_context(person_data_normal, {"objects": []})

        # Entry 2: Second violation
        self.context_engine.set_zone_override("RESTRICTED")
        dec2 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        self.assertEqual(dec2.risk_level, RiskLevel.ORANGE)

        # Exit 2
        self.context_engine.set_zone_override("NORMAL")
        self.context_engine.evaluate_context(person_data_normal, {"objects": []})

        # Entry 3: Third violation within window -> ESCALATES TO RED
        self.context_engine.set_zone_override("RESTRICTED")
        dec3 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        self.assertEqual(dec3.risk_level, RiskLevel.RED)
        self.assertEqual(dec3.decision, "REPEATED_RESTRICTED_ZONE_VIOLATIONS")
        self.assertIn("Repeated security violations detected", dec3.reason)

    # 10. Repeated spoof attempts (RED + Escalation)
    def test_10_repeated_spoof_attempts_escalation(self):
        self.context_engine.set_time_override(dtime(14, 0))
        self.context_engine.set_zone_override("NORMAL")

        spoof_person = {
            "detected": True, "has_authorized": False, "has_unknown": False, "has_spoof": True, "has_verifying": False,
            "people": [{"track_id": 8, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": True, "attack_type": "PHOTO", "liveness_confidence": 0.10, "bbox": (50, 50, 100, 100)}],
        }
        live_person = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 8, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (50, 50, 100, 100)}],
        }

        # Session 1: Spoof presentation
        self.context_engine.evaluate_context(spoof_person, {"objects": []})
        # Phone lowered/cleared
        self.context_engine.evaluate_context(live_person, {"objects": []})

        # Session 2: Spoof presentation
        self.context_engine.evaluate_context(spoof_person, {"objects": []})
        # Phone lowered/cleared
        self.context_engine.evaluate_context(live_person, {"objects": []})

        # Session 3: Third spoof presentation within window -> ESCALATES TO RED
        dec3 = self.context_engine.evaluate_context(spoof_person, {"objects": []})

        self.assertEqual(dec3.risk_level, RiskLevel.RED)
        self.assertEqual(dec3.decision, "REPEATED_PRESENTATION_ATTACK_ATTEMPTS")
        self.assertIn("Repeated presentation attack attempts", dec3.reason)

    # 11. Incident acknowledge operator action
    def test_11_incident_acknowledge_action(self):
        inc_mgr = IncidentManager(db_manager=self.db_manager)
        inc = inc_mgr.process_incident(
            risk_level=RiskLevel.ORANGE,
            policy_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Outside hours",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=1,
        )
        self.assertIsNotNone(inc)
        self.assertEqual(inc.state, IncidentState.ALERTED)

        # Operator acknowledges
        success = inc_mgr.acknowledge_incident(inc.incident_id, operator="Officer Smith")
        self.assertTrue(success)
        self.assertEqual(inc.state, IncidentState.ACKNOWLEDGED)
        self.assertIsNotNone(inc.acknowledged_at)

        # Verify database record
        db_row = self.db_manager.get_incident_by_id(inc.incident_id)
        self.assertIsNotNone(db_row)
        self.assertEqual(db_row["state"], "ACKNOWLEDGED")

    # 12. Incident resolve operator action
    def test_12_incident_resolve_action(self):
        inc_mgr = IncidentManager(db_manager=self.db_manager)
        inc = inc_mgr.process_incident(
            risk_level=RiskLevel.RED,
            policy_code="PRESENTATION_ATTACK_IMPERSONATION",
            reason="Photo spoof",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=1,
        )
        # Operator resolves
        success = inc_mgr.resolve_incident(inc.incident_id, operator="Officer Smith")
        self.assertTrue(success)
        self.assertEqual(inc.state, IncidentState.RESOLVED)

        # Verify database record
        db_row = self.db_manager.get_incident_by_id(inc.incident_id)
        self.assertEqual(db_row["state"], "RESOLVED")

    # 13. Per-face track state isolation
    def test_13_per_face_track_state_isolation(self):
        t_mgr = TrackSecurityStateManager()
        t1 = t_mgr.get_or_create_track(1)  # Track 1: Live Hunter
        t2 = t_mgr.get_or_create_track(2)  # Track 2: Spoof Photo

        t1.update(
            identity="Hunter", identity_conf=0.90, role="Security Lead", is_known=True,
            liveness_state="LIVE", liveness_conf=0.96, is_spoof=False, attack_type=None,
            zone_id="NORMAL", zone_name="Perimeter", time_str="14:00", is_auth_time=True,
            nearby_objects=[], policy_code="NORMAL_ACTIVITY", is_access_granted=True,
            risk_level=RiskLevel.GREEN, reason="Live user",
        )
        t2.update(
            identity="Hunter", identity_conf=0.86, role="Security Lead", is_known=True,
            liveness_state="SPOOF", liveness_conf=0.15, is_spoof=True, attack_type="PHOTO",
            zone_id="RESTRICTED", zone_name="Restricted Area", time_str="14:00", is_auth_time=True,
            nearby_objects=[], policy_code="PRESENTATION_ATTACK_IMPERSONATION", is_access_granted=False,
            risk_level=RiskLevel.RED, reason="Photo attack",
        )

        # Verify state separation
        self.assertEqual(t1.identity, "Hunter")
        self.assertEqual(t1.liveness_state, "LIVE")
        self.assertEqual(t1.risk_level, RiskLevel.GREEN)
        self.assertEqual(t1.zone, "NORMAL")

        self.assertEqual(t2.identity, "Hunter")
        self.assertEqual(t2.liveness_state, "SPOOF")
        self.assertEqual(t2.risk_level, RiskLevel.RED)
        self.assertEqual(t2.zone, "RESTRICTED")
        self.assertEqual(t2.access_state, "SPOOF_REJECTED")

        # Track 1 must NOT have inherited Track 2's spoof or zone
        self.assertFalse(t1.is_spoof)
        self.assertFalse(t1.confirmed_spoof)

    # 14. Voice assistant queries against live incident & violation state
    def test_14_voice_assistant_phase4_queries(self):
        assistant = VoiceAssistant(
            db_manager=self.db_manager,
            context_engine=self.context_engine,
        )

        # Trigger after-hours incident on ContextEngine
        self.context_engine.set_time_override(dtime(22, 45))
        self.context_engine.set_zone_override("RESTRICTED")
        person_data = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (400, 50, 100, 100)}],
        }
        self.context_engine.evaluate_context(person_data, {"objects": []})

        # Query: "Why was Hunter denied?"
        resp1 = assistant.process_command("Why was Hunter denied?")
        self.assertIn("Hunter", resp1)
        self.assertIn("22:45", resp1)
        self.assertIn("denied", resp1.lower())

        # Query: "What is the current incident?"
        resp2 = assistant.process_command("What is the current incident?")
        self.assertIn("INC-", resp2)
        self.assertIn("AFTER_HOURS_RESTRICTED_ACCESS_DENIED", resp2)

        # Query: "When did this incident start?"
        resp3 = assistant.process_command("When did this incident start?")
        self.assertIn("Incident INC-", resp3)

        # Query: "Has this alert been acknowledged?"
        resp4 = assistant.process_command("Has this alert been acknowledged?")
        self.assertIn("not been acknowledged", resp4.lower())

        # Command: "Acknowledge incident"
        resp5 = assistant.process_command("Acknowledge incident")
        self.assertIn("has been acknowledged", resp5.lower())

        # Query again: "Has this alert been acknowledged?"
        resp6 = assistant.process_command("Has this alert been acknowledged?")
        self.assertIn("yes", resp6.lower())
        self.assertIn("has been acknowledged", resp6.lower())

        # Command: "Resolve incident"
        resp7 = assistant.process_command("Resolve incident")
        self.assertIn("has been resolved", resp7.lower())


if __name__ == "__main__":
    unittest.main()
