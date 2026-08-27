"""
Unit and Integration Tests for WatchGuard 2.0 Phase 4.1
--------------------------------------------------------
Incident Debouncing + Zone Entry/Exit Intelligence:
1. Continuous presence creates one incident
2. Zone exit closes incident
3. Re-entry creates a new incident
4. 3 separate entries trigger escalation
5. 30 frames of continuous presence do NOT create 30 violations
6. Continuous spoof presence does not create duplicate incidents
7. Multiple people maintain independent zone states
8. Voice assistant zone entry/exit & violation queries
"""

import unittest
from datetime import datetime, time as dtime
import tempfile
from pathlib import Path
import time

from src.events.event_types import RiskLevel, EventType
from src.database.db_manager import DatabaseManager
from src.core.track_security_state import TrackSecurityStateManager, TrackSecurityState, ZoneOccupancyState
from src.core.incident_manager import IncidentManager, IncidentState
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant


class TestZoneDebouncingPhase41(unittest.TestCase):
    """Test suite verifying all Phase 4.1 specifications."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_watchguard_phase41.db"
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.context_engine = ContextEngine(db_manager=self.db_manager)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Continuous presence creates ONE incident
    def test_01_continuous_presence_creates_one_incident(self):
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
                "role": "Staff",
                "similarity": 0.88,
                "is_spoof": False,
                "liveness_confidence": 0.95,
                "bbox": (400, 50, 100, 100),
            }],
        }

        # Simulate 15 consecutive frames
        first_incident_id = None
        for i in range(15):
            dec = self.context_engine.evaluate_context(person_data, {"objects": []})
            self.assertIsNotNone(dec.incident)
            if i == 0:
                first_incident_id = dec.incident.incident_id
            else:
                # Active incident ID must remain identical across frames
                self.assertEqual(dec.incident.incident_id, first_incident_id)

        # Verify only 1 active incident exists
        active_incs = self.context_engine.incident_manager.get_active_incidents()
        self.assertEqual(len(active_incs), 1)
        self.assertEqual(active_incs[0].incident_id, first_incident_id)

    # 2. Zone exit closes incident
    def test_02_zone_exit_closes_incident(self):
        self.context_engine.set_time_override(dtime(22, 45))
        self.context_engine.set_zone_override("RESTRICTED")

        # 1. Enter Restricted Zone
        person_inside = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (400, 50, 100, 100)}],
        }
        dec1 = self.context_engine.evaluate_context(person_inside, {"objects": []})
        self.assertIsNotNone(dec1.incident)
        inc_id = dec1.incident.incident_id
        self.assertEqual(dec1.incident.state, IncidentState.ALERTED)

        # 2. Move out to Normal Area (Exit)
        self.context_engine.set_zone_override("NORMAL")
        person_outside = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (50, 50, 100, 100)}],
        }
        dec2 = self.context_engine.evaluate_context(person_outside, {"objects": []})

        # Incident should be closed/resolved
        inc = self.context_engine.incident_manager.get_incident(inc_id)
        self.assertIsNotNone(inc)
        self.assertEqual(inc.state, IncidentState.RESOLVED)

    # 3. Re-entry creates a new incident
    def test_03_re_entry_creates_new_incident(self):
        self.context_engine.set_time_override(dtime(22, 45))

        person_data_restricted = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (400, 50, 100, 100)}],
        }
        person_data_normal = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (50, 50, 100, 100)}],
        }

        # Entry 1 (Restricted)
        self.context_engine.set_zone_override("RESTRICTED")
        dec1 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        inc_id1 = dec1.incident.incident_id

        # Exit (Normal)
        self.context_engine.set_zone_override("NORMAL")
        self.context_engine.evaluate_context(person_data_normal, {"objects": []})

        # Entry 2 (Re-enter Restricted)
        self.context_engine.set_zone_override("RESTRICTED")
        dec3 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        inc_id2 = dec3.incident.incident_id

        # Must be distinct incident IDs
        self.assertNotEqual(inc_id1, inc_id2)

    # 4. 3 separate entries trigger escalation
    def test_04_three_separate_entries_trigger_escalation(self):
        self.context_engine.set_time_override(dtime(22, 45))

        person_data_restricted = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (400, 50, 100, 100)}],
        }
        person_data_normal = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (50, 50, 100, 100)}],
        }

        # Entry 1: ORANGE
        self.context_engine.set_zone_override("RESTRICTED")
        dec1 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        self.assertEqual(dec1.risk_level, RiskLevel.ORANGE)

        # Exit 1
        self.context_engine.set_zone_override("NORMAL")
        self.context_engine.evaluate_context(person_data_normal, {"objects": []})

        # Entry 2: ORANGE
        self.context_engine.set_zone_override("RESTRICTED")
        dec2 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        self.assertEqual(dec2.risk_level, RiskLevel.ORANGE)

        # Exit 2
        self.context_engine.set_zone_override("NORMAL")
        self.context_engine.evaluate_context(person_data_normal, {"objects": []})

        # Entry 3: ESCALATES TO RED
        self.context_engine.set_zone_override("RESTRICTED")
        dec3 = self.context_engine.evaluate_context(person_data_restricted, {"objects": []})
        self.assertEqual(dec3.risk_level, RiskLevel.RED)
        self.assertEqual(dec3.decision, "REPEATED_RESTRICTED_ZONE_VIOLATIONS")

    # 5. 30 frames of continuous presence do NOT create 30 violations
    def test_05_thirty_continuous_frames_do_not_create_multiple_violations(self):
        self.context_engine.set_time_override(dtime(22, 45))
        self.context_engine.set_zone_override("RESTRICTED")

        person_data = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (400, 50, 100, 100)}],
        }

        # Feed 30 frames of continuous presence
        last_dec = None
        for _ in range(30):
            last_dec = self.context_engine.evaluate_context(person_data, {"objects": []})

        # Distinct violation count must remain exactly 1
        tstate = self.context_engine.track_manager.get_track(1)
        self.assertIsNotNone(tstate)
        self.assertEqual(len(tstate.distinct_violations), 1)
        self.assertEqual(last_dec.risk_level, RiskLevel.ORANGE)  # Does NOT falsely escalate to RED

    # 6. Continuous spoof presence does not create duplicate incidents
    def test_06_continuous_spoof_presence_debouncing(self):
        self.context_engine.set_time_override(dtime(14, 0))
        self.context_engine.set_zone_override("NORMAL")

        spoof_data = {
            "detected": True, "has_authorized": False, "has_unknown": False, "has_spoof": True, "has_verifying": False,
            "people": [{"track_id": 2, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": True, "attack_type": "PHOTO", "liveness_confidence": 0.10, "bbox": (50, 50, 100, 100)}],
        }

        first_inc_id = None
        for i in range(20):
            dec = self.context_engine.evaluate_context(spoof_data, {"objects": []})
            if i == 0:
                first_inc_id = dec.incident.incident_id
            else:
                self.assertEqual(dec.incident.incident_id, first_inc_id)

        tstate = self.context_engine.track_manager.get_track(2)
        self.assertEqual(len(tstate.distinct_spoof_sessions), 1)

    # 7. Multiple people maintain independent zone states
    def test_07_multiple_people_independent_zone_states(self):
        t_mgr = TrackSecurityStateManager()
        t1 = t_mgr.get_or_create_track(10)  # Hunter: INSIDE Restricted
        t2 = t_mgr.get_or_create_track(20)  # Alice: OUTSIDE in Normal

        t1.update(
            identity="Hunter", identity_conf=0.90, role="Staff", is_known=True,
            liveness_state="LIVE", liveness_conf=0.96, is_spoof=False, attack_type=None,
            zone_id="RESTRICTED", zone_name="Restricted Area", time_str="22:45", is_auth_time=False,
            nearby_objects=[], policy_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED", is_access_granted=False,
            risk_level=RiskLevel.ORANGE, reason="After-hours",
        )
        t2.update(
            identity="Alice", identity_conf=0.89, role="Staff", is_known=True,
            liveness_state="LIVE", liveness_conf=0.94, is_spoof=False, attack_type=None,
            zone_id="NORMAL", zone_name="Standard Perimeter", time_str="22:45", is_auth_time=False,
            nearby_objects=[], policy_code="NORMAL_ACTIVITY", is_access_granted=True,
            risk_level=RiskLevel.GREEN, reason="Normal area",
        )

        self.assertEqual(t1.zone_occupancy, ZoneOccupancyState.INSIDE)
        self.assertEqual(len(t1.distinct_violations), 1)
        self.assertEqual(t1.risk_level, RiskLevel.ORANGE)

        self.assertEqual(t2.zone_occupancy, ZoneOccupancyState.OUTSIDE)
        self.assertEqual(len(t2.distinct_violations), 0)
        self.assertEqual(t2.risk_level, RiskLevel.GREEN)

    # 8. Voice assistant zone entry/exit & violation queries
    def test_08_voice_assistant_zone_queries(self):
        assistant = VoiceAssistant(
            db_manager=self.db_manager,
            context_engine=self.context_engine,
        )

        self.context_engine.set_time_override(dtime(22, 45))
        self.context_engine.set_zone_override("RESTRICTED")

        person_data = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (400, 50, 100, 100)}],
        }
        self.context_engine.evaluate_context(person_data, {"objects": []})

        # Query: "Did Hunter enter the restricted zone?"
        resp1 = assistant.process_command("Did Hunter enter the restricted zone?")
        self.assertIn("Yes", resp1)
        self.assertIn("Hunter", resp1)
        self.assertIn("Restricted Area", resp1)

        # Query: "When did he enter?"
        resp2 = assistant.process_command("When did he enter?")
        self.assertIn("Hunter entered the Restricted Area at 22:45", resp2)

        # Query: "Has he left the restricted area?"
        resp3 = assistant.process_command("Has he left the restricted area?")
        self.assertIn("No", resp3)
        self.assertIn("currently inside the Restricted Area", resp3)

        # Query: "How many violations happened?"
        resp4 = assistant.process_command("How many violations happened?")
        self.assertIn("1 distinct security violation", resp4)


if __name__ == "__main__":
    unittest.main()
