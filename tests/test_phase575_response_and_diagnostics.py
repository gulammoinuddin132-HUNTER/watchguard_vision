"unittest for Phase 5.7.5"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.core.voice_assistant import VoiceAssistant, SessionContext
from src.core.copilot_reasoner import DeterministicCopilotReasoner
from src.core.object_detection import ObjectDetectionEngine


class TestPhase575ResponseAndDiagnostics(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_context_engine = MagicMock()
        self.mock_context_engine.hours_start = "09:00"
        self.mock_context_engine.hours_end = "18:00"
        self.assistant = VoiceAssistant(
            db_manager=self.mock_db,
            context_engine=self.mock_context_engine,
        )
        self.reasoner = DeterministicCopilotReasoner()

    def test_01_unregistered_repeated_violations_after_hours_restricted(self):
        self.assistant.session_context.last_person_name = "Unknown Person"
        self.assistant.session_context.last_is_authorized = False
        self.assistant.session_context.last_is_unknown = True
        self.assistant.session_context.last_role = "Unregistered Visitor"
        self.assistant.session_context.last_zone = "Restricted Area"
        self.assistant.session_context.last_decision_code = "REPEATED_RESTRICTED_ZONE_VIOLATIONS"
        self.assistant.session_context.last_time_str = "22:45"
        self.assistant.session_context.last_is_authorized_hours = False
        self.assistant.session_context.last_risk_level = "RED"
        self.assistant.session_context.last_reason = "Repeated restricted-zone violations"

        response = self.assistant._handle_explain_access_decision("Why am I being denied?")
        self.assertIn("Access is denied", response)
        self.assertIn("unregistered", response.lower())
        self.assertIn("Restricted Area", response)
        self.assertIn("repeated restricted-zone violations", response.lower())
        self.assertNotIn("authorized as", response.lower())

        auth_resp = self.assistant._handle_follow_up_person_authorization("Am I allowed here?")
        self.assertIn("No", auth_resp)
        self.assertIn("unregistered visitor", auth_resp.lower())
        self.assertNotIn("is authorized as", auth_resp.lower())

        context = {
            "CURRENT_STATE": {
                "people": [{"identity": "Unknown Person", "is_registered": False, "liveness_result": "LIVE"}],
                "people_count": 1,
                "active_zone": "Restricted Area",
                "time_status": "AFTER_HOURS",
                "master_decision_code": "REPEATED_RESTRICTED_ZONE_VIOLATIONS",
                "master_risk_level": "RED",
                "master_reason": "Repeated restricted-zone violations",
                "operating_hours_window": "09:00 to 18:00",
            },
            "EVIDENCE_GROUNDING": {"who": "Unknown Person"},
        }
        reasoner_resp = self.reasoner.generate_explanation("Why am I being denied?", context)
        self.assertIn("Access is denied", reasoner_resp.text)
        self.assertIn("unregistered", reasoner_resp.text.lower())
        self.assertIn("repeated restricted-zone violations", reasoner_resp.text.lower())
        self.assertNotIn("authorized as", reasoner_resp.text.lower())

    def test_02_strict_invariants_unregistered_is_never_authorized(self):
        for role in ["Visitor", "Unregistered Visitor", "Guest", "Unknown"]:
            self.assistant.session_context.last_person_name = "Unknown Person"
            self.assistant.session_context.last_is_authorized = False
            self.assistant.session_context.last_is_unknown = True
            self.assistant.session_context.last_role = role
            self.assistant.session_context.last_zone = "Standard Perimeter"
            self.assistant.session_context.last_decision_code = "NORMAL_ACTIVITY"
            self.assistant.session_context.last_risk_level = "GREEN"

            resp = self.assistant._handle_explain_access_decision("Why was access refused?")
            self.assertIn("Access is denied", resp)
            self.assertIn("not registered", resp.lower())
            self.assertNotIn("is authorized as", resp.lower())

    def test_03_spoof_presentation_attack_explanation(self):
        self.assistant.session_context.last_person_name = "Hunter"
        self.assistant.session_context.last_is_authorized = True
        self.assistant.session_context.last_is_unknown = False
        self.assistant.session_context.last_decision_code = "PRESENTATION_ATTACK_IMPERSONATION"
        self.assistant.session_context.last_risk_level = "RED"

        resp = self.assistant._handle_explain_access_decision("Why was I denied?")
        self.assertIn("Access is denied", resp)
        self.assertIn("presentation attack or spoof attempt", resp.lower())
        self.assertIn("Hunter", resp)


    def test_04_registered_person_after_hours_explanation(self):
        self.assistant.session_context.last_person_name = "Hunter"
        self.assistant.session_context.last_is_authorized = True
        self.assistant.session_context.last_is_unknown = False
        self.assistant.session_context.last_role = "Security Officer"
        self.assistant.session_context.last_zone = "Restricted Area"
        self.assistant.session_context.last_decision_code = "AFTER_HOURS_RESTRICTED_ACCESS_DENIED"
        self.assistant.session_context.last_time_str = "22:15"
        self.assistant.session_context.last_is_authorized_hours = False
        self.assistant.session_context.last_risk_level = "ORANGE"

        resp = self.assistant._handle_explain_access_decision("Why can't I enter?")
        self.assertIn("Access is denied", resp)
        self.assertIn("Hunter", resp)
        self.assertIn("Security Officer", resp)
        self.assertIn("22:15", resp)
        self.assertIn("09:00 to 18:00", resp)

    def test_05_vectorized_yolov8_postprocessing(self):
        engine = ObjectDetectionEngine()
        mock_net = MagicMock()
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[0, 0, 100] = 320.0
        out[0, 1, 100] = 240.0
        out[0, 2, 100] = 200.0
        out[0, 3, 100] = 150.0
        out[0, 4 + 63, 100] = 0.85

        mock_net.forward.return_value = out
        engine._net = mock_net
        engine._is_ready = True

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = engine.process_frame(dummy_frame)

        self.assertTrue(res["detected"])
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["objects"][0]["label"], "laptop")
        self.assertAlmostEqual(res["objects"][0]["confidence"], 0.85, places=2)

if __name__ == '__main__':
    unittest.main()
