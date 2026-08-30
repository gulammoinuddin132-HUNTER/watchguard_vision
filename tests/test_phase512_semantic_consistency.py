"""
tests/test_phase512_semantic_consistency.py
=============================================================================
Regression tests for WatchGuard Phase 5.12:
Access Status and Voice Response Semantic Consistency Fix.

Validates the strict semantic invariants:
1. UNREGISTERED != AUTHORIZED
2. UNKNOWN_VISITOR != ACCESS_GRANTED
3. YELLOW != ACCESS_GRANTED
4. Risk level and access authorization are strictly decoupled.
5. If access is DENIED, every voice explanation must say DENIED / NOT ALLOWED.
6. A yellow risk level must NEVER be interpreted or emitted as 'granted'.
7. ContextEngine person_entities must synchronize post-policy with track state.
=============================================================================
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.policy_engine import RiskLevel
from src.core.voice_assistant import VoiceAssistant
from src.core.copilot_reasoner import DeterministicCopilotReasoner, build_structured_ai_context
from src.database.db_manager import DatabaseManager


class TestPhase512SemanticConsistency(unittest.TestCase):

    def setUp(self):
        self.db = DatabaseManager(":memory:")
        self.context_engine = ContextEngine(zones_config={
            "STANDARD": {"name": "Standard Perimeter", "type": "standard", "rect_norm": [0.0, 0.0, 1.0, 1.0]},
            "PROTECTED": {"name": "Protected Asset Area", "type": "protected", "rect_norm": [0.0, 0.0, 1.0, 1.0]},
            "RESTRICTED": {"name": "Restricted Area", "type": "restricted", "rect_norm": [0.0, 0.0, 1.0, 1.0]},
        })
        self.va = VoiceAssistant(context_engine=self.context_engine, db_manager=self.db)
        self.va.camera = MagicMock()
        self.va.camera.is_running = True
        self.va.person_engine = MagicMock()
        self.va.person_engine.is_ready = True

    def test_case_a_registered_authorized_granted(self):
        """Case A: Registered + authorized + permitted zone/time -> GRANTED."""
        dec = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="AUTHORIZED_ACCESS",
            reason="Authorized personnel Hunter detected in Standard Perimeter during normal hours.",
            is_access_granted=True,
            context_summary={
                "person_detected": True,
                "person_name": "Hunter",
                "is_authorized": True,
                "is_unknown": False,
                "has_spoof": False,
                "zone_name": "Standard Perimeter",
                "time_str": "14:00",
                "is_authorized_hours": True,
                "person_entities": [{
                    "track_id": 1,
                    "identity": "Hunter",
                    "is_known": True,
                    "is_spoof": False,
                    "role": "Security Lead",
                    "liveness_state": "STABLE_LIVE",
                    "access_state": "GRANTED",
                    "zone_name": "Standard Perimeter",
                    "risk_level": "GREEN",
                }],
            },
            per_track_states=[{"track_id": 1, "access_state": "GRANTED", "is_known": True}],
            person_entities=[{
                "track_id": 1,
                "identity": "Hunter",
                "is_known": True,
                "is_spoof": False,
                "role": "Security Lead",
                "liveness_state": "STABLE_LIVE",
                "access_state": "GRANTED",
                "zone_name": "Standard Perimeter",
                "risk_level": "GREEN",
            }],
        )
        self.context_engine.last_decision = dec

        # Test explain access decision
        resp = self.va.process_command("why is he allowed?")
        self.assertIn("Access is granted", resp)
        self.assertIn("Hunter", resp)
        self.assertNotIn("denied", resp.lower())

        # Test person authorization query
        resp_auth = self.va.process_command("is Hunter allowed?")
        self.assertIn("Access is granted", resp_auth)

    def test_case_b1_unregistered_in_protected_asset_area_denied(self):
        """Case B1: Unregistered person + Protected Asset Area -> Strictly DENIED."""
        dec = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="UNKNOWN_PERSON",
            reason="Unregistered individual detected in Protected Asset Area.",
            is_access_granted=False,
            context_summary={
                "person_detected": True,
                "person_name": "Unknown Person",
                "is_authorized": False,
                "is_unknown": True,
                "has_spoof": False,
                "zone_name": "Protected Asset Area",
                "time_str": "14:30",
                "is_authorized_hours": True,
                "person_entities": [{
                    "track_id": 1,
                    "identity": "Unknown Person",
                    "is_known": False,
                    "is_spoof": False,
                    "role": "Visitor",
                    "liveness_state": "LIVE",
                    "access_state": "DENIED",
                    "zone_name": "Protected Asset Area",
                    "risk_level": "YELLOW",
                }],
            },
            per_track_states=[{"track_id": 1, "access_state": "DENIED", "is_known": False}],
            person_entities=[{
                "track_id": 1,
                "identity": "Unknown Person",
                "is_known": False,
                "is_spoof": False,
                "role": "Visitor",
                "liveness_state": "LIVE",
                "access_state": "DENIED",
                "zone_name": "Protected Asset Area",
                "risk_level": "YELLOW",
            }],
        )
        self.context_engine.last_decision = dec

        # Query: "why he is not allowed?"
        resp = self.va.process_command("why he is not allowed?")
        expected = "Access is denied. An unregistered individual was detected in the Protected Asset Area."
        self.assertEqual(resp, expected)

        # STRICT INVARIANT ASSERTIONS:
        self.assertNotIn("Access is granted", resp)
        self.assertNotIn("GRANTED (YELLOW)", resp)
        self.assertNotIn("GRANTED", resp)
        self.assertNotIn("authorized", resp.lower())

        # Follow-up why test
        resp_why = self.va.process_command("why was he denied?")
        self.assertIn("Access is denied", resp_why)
        self.assertIn("Protected Asset Area", resp_why)

    def test_case_b2_unregistered_in_restricted_area_denied(self):
        """Case B2: Unregistered person + Restricted Area -> Strictly DENIED."""
        dec = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="UNKNOWN_PERSON",
            reason="Unregistered individual detected in Restricted Area.",
            is_access_granted=False,
            context_summary={
                "person_detected": True,
                "person_name": "Unknown Person",
                "is_authorized": False,
                "is_unknown": True,
                "has_spoof": False,
                "zone_name": "Restricted Area",
                "time_str": "15:00",
                "is_authorized_hours": True,
                "person_entities": [{
                    "track_id": 1,
                    "identity": "Unknown Person",
                    "is_known": False,
                    "is_spoof": False,
                    "role": "Visitor",
                    "liveness_state": "LIVE",
                    "access_state": "DENIED",
                    "zone_name": "Restricted Area",
                    "risk_level": "YELLOW",
                }],
            },
            per_track_states=[{"track_id": 1, "access_state": "DENIED", "is_known": False}],
            person_entities=[{
                "track_id": 1,
                "identity": "Unknown Person",
                "is_known": False,
                "is_spoof": False,
                "role": "Visitor",
                "liveness_state": "LIVE",
                "access_state": "DENIED",
                "zone_name": "Restricted Area",
                "risk_level": "YELLOW",
            }],
        )
        self.context_engine.last_decision = dec

        resp = self.va.process_command("why is he not allowed?")
        expected = "Access is denied. An unregistered individual (not registered in the database) was detected in the Restricted Area."
        self.assertEqual(resp, expected)
        self.assertNotIn("granted", resp.lower())

    def test_case_c_unregistered_standard_perimeter(self):
        """Case C: Unregistered person + Standard Perimeter -> Do not claim authorization."""
        dec = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="UNKNOWN_PERSON",
            reason="Unregistered individual detected in Standard Perimeter.",
            is_access_granted=False,
            context_summary={
                "person_detected": True,
                "person_name": "Unknown Person",
                "is_authorized": False,
                "is_unknown": True,
                "has_spoof": False,
                "zone_name": "Standard Perimeter",
                "time_str": "12:00",
                "is_authorized_hours": True,
                "person_entities": [{
                    "track_id": 1,
                    "identity": "Unknown Person",
                    "is_known": False,
                    "is_spoof": False,
                    "role": "Visitor",
                    "liveness_state": "LIVE",
                    "access_state": "DENIED",
                    "zone_name": "Standard Perimeter",
                    "risk_level": "YELLOW",
                }],
            },
            per_track_states=[{"track_id": 1, "access_state": "DENIED", "is_known": False}],
            person_entities=[{
                "track_id": 1,
                "identity": "Unknown Person",
                "is_known": False,
                "is_spoof": False,
                "role": "Visitor",
                "liveness_state": "LIVE",
                "access_state": "DENIED",
                "zone_name": "Standard Perimeter",
                "risk_level": "YELLOW",
            }],
        )
        self.context_engine.last_decision = dec

        resp = self.va.process_command("explain access decision")
        self.assertIn("Access is denied", resp)
        self.assertIn("not registered in the facial database", resp)
        self.assertNotIn("granted", resp.lower())

    def test_case_d_after_hours_restriction(self):
        """Case D: After-hours restriction -> Explicitly state outside permitted hours."""
        dec = SecurityDecision(
            risk_level=RiskLevel.ORANGE,
            decision="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Access denied: Subject Hunter detected in Restricted Area outside authorized hours (22:15).",
            is_access_granted=False,
            context_summary={
                "person_detected": True,
                "person_name": "Hunter",
                "is_authorized": False,
                "is_unknown": False,
                "has_spoof": False,
                "zone_name": "Restricted Area",
                "time_str": "22:15",
                "is_authorized_hours": False,
                "person_entities": [{
                    "track_id": 1,
                    "identity": "Hunter",
                    "is_known": True,
                    "is_spoof": False,
                    "role": "Staff",
                    "liveness_state": "LIVE",
                    "access_state": "DENIED",
                    "zone_name": "Restricted Area",
                    "risk_level": "ORANGE",
                }],
            },
            per_track_states=[{"track_id": 1, "access_state": "DENIED", "is_known": True}],
            person_entities=[{
                "track_id": 1,
                "identity": "Hunter",
                "is_known": True,
                "is_spoof": False,
                "role": "Staff",
                "liveness_state": "LIVE",
                "access_state": "DENIED",
                "zone_name": "Restricted Area",
                "risk_level": "ORANGE",
            }],
        )
        self.context_engine.last_decision = dec

        resp = self.va.process_command("why was access denied?")
        self.assertIn("Access is denied because this is outside the permitted access hours", resp)
        self.assertNotIn("Access is granted", resp)
        self.assertNotIn("is authorized and live", resp)

    def test_case_e_yellow_risk_never_granted(self):
        """Case E: Yellow risk level with denied access must never emit GRANTED (YELLOW)."""
        dec = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="UNKNOWN_PERSON",
            reason="Unregistered individual detected in Protected Asset Area.",
            is_access_granted=False,
            context_summary={
                "person_detected": True,
                "person_name": "Unknown Person",
                "is_authorized": False,
                "is_unknown": True,
                "has_spoof": False,
                "zone_name": "Protected Asset Area",
                "time_str": "10:00",
                "is_authorized_hours": True,
                "person_entities": [{
                    "track_id": 1,
                    "identity": "Unknown Person",
                    "is_known": False,
                    "is_spoof": False,
                    "role": "Visitor",
                    "access_state": "DENIED",
                    "zone_name": "Protected Asset Area",
                    "risk_level": "YELLOW",
                }],
            },
            per_track_states=[{"track_id": 1, "access_state": "DENIED"}],
            person_entities=[{
                "track_id": 1,
                "identity": "Unknown Person",
                "is_known": False,
                "is_spoof": False,
                "role": "Visitor",
                "access_state": "DENIED",
                "zone_name": "Protected Asset Area",
                "risk_level": "YELLOW",
            }],
        )
        self.context_engine.last_decision = dec

        resp = self.va.process_command("why he is not allowed?")
        self.assertNotIn("GRANTED (YELLOW)", resp)
        self.assertNotIn("GRANTED", resp)
        self.assertTrue(resp.startswith("Access is denied"))

    def test_context_engine_synchronizes_person_entities(self):
        """Verify ContextEngine synchronizes fused entities with post-policy evaluated track state."""
        person_data = {
            "detected": True,
            "count": 1,
            "people": [{
                "track_id": 1,
                "name": "Unknown Person",
                "user_id": None,
                "is_known": False,
                "match_pct": 0,
                "is_spoof": False,
                "is_ambiguous": False,
                "liveness_state": "LIVE",
                "liveness_confidence": 0.95,
                "attack_type": "none",
                "joint_decision": MagicMock(status=MagicMock(value="UNKNOWN"), is_spoof=False),
                "bbox": (100, 100, 200, 300),
            }],
            "has_unknown": True,
            "has_authorized": False,
            "has_spoof": False,
        }
        object_data = {"detected": False, "count": 0, "objects": []}

        # Dedicated context engine configured with Protected Asset Area
        ce = ContextEngine(zones_config={
            "PROTECTED_ASSET_AREA": {"name": "Protected Asset Area", "type": "protected", "rect_norm": [0.0, 0.0, 1.0, 1.0]},
        })
        decision = ce.evaluate_context(
            person_data=person_data,
            object_data=object_data,
            camera_id=0,
            frame_shape=(480, 640, 3),
        )

        self.assertFalse(decision.is_access_granted)
        self.assertEqual(decision.access_state, "DENIED")
        self.assertEqual(decision.decision, "UNKNOWN_PERSON")

        # Crucial check: person_entities inside decision must have access_state == DENIED
        self.assertTrue(len(decision.person_entities) >= 1)
        p0 = decision.person_entities[0]
        self.assertEqual(p0["access_state"], "DENIED")
        self.assertFalse(p0["is_known"])

    def test_copilot_reasoner_denial_alignment(self):
        """Verify DeterministicCopilotReasoner produces consistent denial for Protected Asset Area."""
        ctx = {
            "CURRENT_STATE": {
                "active_zone": "Protected Asset Area",
                "master_decision_code": "UNKNOWN_PERSON",
                "master_risk_level": "YELLOW",
                "master_reason": "Unregistered individual detected in Protected Asset Area.",
                "is_access_granted": False,
                "live_people_in_view": [{
                    "identity": "Unknown Person",
                    "is_registered": False,
                    "liveness_result": "LIVE",
                    "access_decision": "DENIED",
                }],
            },
            "EVIDENCE_GROUNDING": {"who": "Unknown Person"},
        }
        reasoner = DeterministicCopilotReasoner()
        resp = reasoner.generate_explanation("why he is not allowed?", ctx)
        self.assertIn("Access is denied. An unregistered individual was detected in the Protected Asset Area.", resp.text)
        self.assertNotIn("granted", resp.text.lower())


if __name__ == "__main__":
    unittest.main()
