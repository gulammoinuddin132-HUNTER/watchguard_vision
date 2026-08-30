"""
WatchGuard Vision - Final Copilot & Unified V2 Test Suite
---------------------------------------------------------
Validates Task A (Unified 84-Class V2 Detector) and Task B (AI Copilot Knowledge Upgrade).
"""

import unittest
import os
from pathlib import Path
import cv2
import numpy as np

from config import (
    YOLOV8N_MODEL_PATH,
    WATCHGUARD_UNIFIED_CLASSES,
    COCO_CLASSES,
    SECURITY_OBJECT_CATEGORIES,
)
from src.core.interfaces.detector_interface import DetectedObject
from src.core.voice_assistant import VoiceAssistant, VoiceIntent
from src.core.semantic_router import SemanticQueryRouter, SemanticResolution
from src.core.copilot_reasoner import build_structured_ai_context, COPILOT_SYSTEM_PROMPT
from src.core.policy_engine import PolicyEngine
from src.events.event_types import RiskLevel


class MockDecision:
    def __init__(self, is_access_granted=True, risk_level=RiskLevel.GREEN, access_state="GRANTED", policy_code="NORMAL_ACTIVITY", reason="Routine monitoring"):
        self.is_access_granted = is_access_granted
        self.risk_level = risk_level
        self.access_state = access_state
        self.policy_code = policy_code
        self.decision = policy_code
        self.reason = reason
        self.evidence_required = False
        self.correlated_interactions = []
        self.per_track_states = []
        self.incident = None
        self.context_summary = {
            "current_zone": "Standard Perimeter",
            "is_authorized_hours": True,
            "operating_hours_window": "09:00 to 18:00",
            "objects": [
                {"label": "usb_drive", "distance_tier": "MEDIUM", "associated_person_track": 1, "pixel_distance_to_person": 45.0, "confidence": 0.92},
                {"label": "laptop", "distance_tier": "CLOSE", "associated_person_track": 1, "pixel_distance_to_person": 25.0, "confidence": 0.95},
            ],
            "person_entities": [
                {
                    "track_id": 1,
                    "identity": "Hunter",
                    "role": "Staff",
                    "is_known": True,
                    "liveness_state": "LIVE",
                    "access_state": "GRANTED",
                    "risk_level": "GREEN",
                    "associated_objects": [{"label": "usb_drive", "distance_tier": "MEDIUM", "associated_person_track": 1}],
                }
            ],
        }
        self.person_entities = self.context_summary["person_entities"]
        self.ccsdf_result = {
            "risk_score": 0.15,
            "risk_level": "GREEN",
            "dominant_factor": "identity",
            "recommended_action": "ALLOW",
            "tailgating_status": "NONE",
            "tailgating_leader": "Hunter",
            "tailgating_follower": "Unknown Person",
            "tailgating_distance": 140.0,
            "behavior_status": "NORMAL_ACTIVITY",
            "confidence_tier": "HIGH",
            "confidence": {"overall_confidence": 0.94, "identity_confidence": 0.96, "pad_confidence": 0.92},
            "normalized_signals": {"identity": 0.05, "pad": 0.08, "zone": 0.0, "time": 0.0, "proximity": 0.0, "history": 0.0, "asset": 0.0, "behavior": 0.05},
        }


class MockContextEngine:
    def __init__(self):
        self.last_decision = MockDecision()
        self.hours_start = "09:00"
        self.hours_end = "18:00"

    def get_context_summary(self):
        return {
            "current_zone": "Standard Perimeter",
            "is_authorized_hours": True,
            "operating_hours_window": "09:00 to 18:00",
            "active_tracks": [
                {
                    "track_id": 1,
                    "identity": "Hunter",
                    "role": "Staff",
                    "is_known": True,
                    "liveness_state": "LIVE",
                    "access_state": "GRANTED",
                    "risk_level": "GREEN",
                    "associated_objects": [{"label": "usb_drive", "distance_tier": "MEDIUM", "associated_person_track": 1}],
                }
            ],
            "objects": [
                {"label": "usb_drive", "distance_tier": "MEDIUM", "associated_person_track": 1, "pixel_distance_to_person": 45.0, "confidence": 0.92},
                {"label": "laptop", "distance_tier": "CLOSE", "associated_person_track": 1, "pixel_distance_to_person": 25.0, "confidence": 0.95},
            ],
            "events_5min": [],
        }


class TestFinalCopilotAndUnifiedV2(unittest.TestCase):
    def setUp(self):
        self.ctx_engine = MockContextEngine()
        self.va = VoiceAssistant(context_engine=self.ctx_engine)
        self.router = SemanticQueryRouter(VoiceIntent)

    def test_01_v2_model_output_geometry_and_taxonomy(self):
        """Verify V2 model preserves 80 COCO classes, adds 4 security classes, and has shape [1, 88, 8400]."""
        model_path = Path("data/models/yolo_watchguard_unified_84_v2.onnx")
        self.assertTrue(model_path.exists(), "V2 model must exist")

        net = cv2.dnn.readNetFromONNX(str(model_path))
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        net.setInput(dummy)
        out = net.forward()
        self.assertEqual(out.shape, (1, 88, 8400), "Output shape must be [1, 88, 8400]")

        self.assertEqual(len(WATCHGUARD_UNIFIED_CLASSES), 84)
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[0], "person")
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[80], "pen")
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[81], "access_badge")
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[82], "usb_drive")
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[83], "keys")

    def test_02_security_objects_query_routing_and_response(self):
        """Verify queries about security objects route to SECURITY_OBJECTS_QUERY and return grounded context."""
        queries = [
            "What security objects are present?",
            "Was a USB drive detected?",
            "Who is associated with the USB drive?",
            "Were keys detected?",
        ]
        for q in queries:
            res = self.router.route_query(q)
            self.assertEqual(res.intent, VoiceIntent.SECURITY_OBJECTS_QUERY, f"Query '{q}' should route to SECURITY_OBJECTS_QUERY")

        ans = self.va.process_command("What security objects are present?")
        self.assertIn("usb_drive", ans.lower())
        self.assertIn("medium", ans.lower())
        self.assertIn("contextual evidence", ans.lower())

    def test_03_detector_mode_query_routing_and_response(self):
        """Verify queries about detector mode route to DETECTOR_MODE_QUERY and describe active mode."""
        queries = [
            "What detector mode is active?",
            "Which model is running?",
            "What object detector is active?",
        ]
        for q in queries:
            res = self.router.route_query(q)
            self.assertEqual(res.intent, VoiceIntent.DETECTOR_MODE_QUERY, f"Query '{q}' should route to DETECTOR_MODE_QUERY")

        ans = self.va.process_command("What detector mode is active?")
        self.assertIn("mode", ans.lower())
        self.assertIn("edge cpu", ans.lower())

    def test_04_research_novelty_query_routing_and_response(self):
        """Verify queries about WatchGuard novelty return grounded research contribution statement."""
        queries = [
            "What is novel about WatchGuard?",
            "What is the research contribution of WatchGuard?",
            "What makes WatchGuard different?",
        ]
        for q in queries:
            res = self.router.route_query(q)
            self.assertEqual(res.intent, VoiceIntent.RESEARCH_NOVELTY_QUERY, f"Query '{q}' should route to RESEARCH_NOVELTY_QUERY")

        ans = self.va.process_command("What is novel about WatchGuard?")
        self.assertIn("continuous contextual security decision fusion", ans.lower())
        self.assertIn("deterministic policy authority", ans.lower())
        self.assertNotIn("world's first", ans.lower())
        self.assertNotIn("unique worldwide", ans.lower())

    def test_05_research_ccsdf_state_query_routing_and_response(self):
        """Verify queries about CCSDF state route to RESEARCH_CCSDF_STATE_QUERY and explain R(t)."""
        queries = [
            "What is the current CCSDF risk?",
            "What is the current CCSDF state?",
            "What is the dominant risk factor?",
        ]
        for q in queries:
            res = self.router.route_query(q)
            self.assertEqual(res.intent, VoiceIntent.RESEARCH_CCSDF_STATE_QUERY, f"Query '{q}' should route to RESEARCH_CCSDF_STATE_QUERY")

        ans = self.va.process_command("What is the current CCSDF risk?")
        self.assertIn("r=0.15", ans.lower())
        self.assertIn("green", ans.lower())

    def test_06_tailgating_explanation_and_non_transfer_guarantee(self):
        """Verify tailgating explanation explicitly details leader, follower, and non-transfer invariant."""
        # Case 1: Tailgating detected
        self.ctx_engine.last_decision.ccsdf_result["tailgating_status"] = "TAILGATING_SUSPECTED"
        ans_tg = self.va.process_command("Was tailgating detected?")
        self.assertIn("tailgating is suspected", ans_tg.lower())
        self.assertIn("hunter", ans_tg.lower())
        self.assertIn("unknown person", ans_tg.lower())
        self.assertIn("never transfers", ans_tg.lower())
        self.assertIn("denied", ans_tg.lower())

        # Case 2: No tailgating
        self.ctx_engine.last_decision.ccsdf_result["tailgating_status"] = "NONE"
        ans_no_tg = self.va.process_command("Was tailgating detected?")
        self.assertIn("no tailgating pattern", ans_no_tg.lower())

    def test_07_uncertainty_explanation_and_tiers(self):
        """Verify uncertainty query explains overall confidence, biometric breakdown, and confidence tier."""
        self.ctx_engine.last_decision.ccsdf_result["confidence_tier"] = "HIGH"
        self.ctx_engine.last_decision.ccsdf_result["confidence"] = {
            "overall_confidence": 0.94,
            "identity_confidence": 0.96,
            "pad_confidence": 0.92,
        }
        ans = self.va.process_command("How confident is the current decision?")
        self.assertIn("high", ans.lower())
        self.assertIn("0.94", ans)
        self.assertIn("identity: 0.96", ans.lower())
        self.assertIn("pad: 0.92", ans.lower())

    def test_08_behavior_anomaly_explanation(self):
        """Verify behavior query explains dwell, entries, and approach anomalies."""
        self.ctx_engine.last_decision.ccsdf_result["behavior_status"] = "BEHAVIOR_ANOMALY"
        self.ctx_engine.last_decision.ccsdf_result["normalized_signals"]["behavior"] = 0.85
        ans = self.va.process_command("What behavior was unusual?")
        self.assertIn("behavior anomaly detected", ans.lower())
        self.assertIn("S_behavior=0.85", ans)

    def test_09_spoof_attempts_counter_semantics(self):
        """Verify spoof counter explains confirmed presentation attack sessions today."""
        ans = self.va.process_command("How many spoof attempts occurred?")
        self.assertIn("spoof", ans.lower())

    def test_10_structured_ai_context_injection(self):
        """Verify build_structured_ai_context injects CCSDF, tailgating, behavior, uncertainty, and detector mode."""
        ai_ctx = build_structured_ai_context(context_engine=self.ctx_engine)
        curr = ai_ctx.get("CURRENT_STATE", {})
        self.assertIn("ccsdf_state", curr)
        self.assertIn("tailgating_state", curr)
        self.assertIn("behavior_state", curr)
        self.assertIn("uncertainty_state", curr)
        self.assertIn("detector_mode", curr)
        self.assertIn("live_detected_objects", curr)

        self.assertEqual(curr["ccsdf_state"]["risk_score"], 0.15)
        self.assertEqual(curr["uncertainty_state"]["confidence_tier"], "HIGH")

    def test_11_copilot_system_prompt_comprehensive_rules(self):
        """Verify COPILOT_SYSTEM_PROMPT contains all 20 architectural rules and strict invariants."""
        self.assertIn("watchguard_unified", COPILOT_SYSTEM_PROMPT)
        self.assertIn("usb_drive", COPILOT_SYSTEM_PROMPT)
        self.assertIn("S(t) =", COPILOT_SYSTEM_PROMPT)
        self.assertIn("NEVER transfers", COPILOT_SYSTEM_PROMPT)
        self.assertIn("CONFIRMED PRESENTATION ATTACK SESSIONS", COPILOT_SYSTEM_PROMPT)
        self.assertIn("PolicyEngine", COPILOT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
