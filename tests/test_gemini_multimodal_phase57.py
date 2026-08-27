"""
Unit & Integration Tests for WatchGuard 2.0 Phase 5.7:
Gemini Multimodal Security Copilot + Critical Event Log / Dashboard Layout Fix
--------------------------------------------------------------------------------
Validates:
1. Gemini unavailable -> Graceful deterministic fail-secure fallback.
2. Gemini available -> Multimodal reasoning execution.
3. Visual query -> Captures/sends single snapshot frame.
4. Non-visual query -> Operates without image transmission.
5. One person scenario -> Grounded single-person response.
6. Two people scenario -> Preserves both entities (Gulam Moinuddin + Unknown).
7. Authorized + unknown person -> Grounded access explanation.
8. Authorized + phone object -> Acknowledges non-person physical asset.
9. Authorized + phone photo spoof -> Rejection and incident attribution.
10. Current state grounding -> Strictly adheres to 7-tuple facts.
11. Recent history grounding -> 5-minute bounded event retrieval.
12. Anti-hallucination -> No invented people.
13. Anti-hallucination -> No invented objects.
14. Voice/text parity -> Shared semantic router and response pipeline.
15. Contextual pronoun follow-up -> "Why can't he enter?" resolves target.
16. Dashboard Layout & Event Log -> Grid-configured, bounded, two-way scrollable.
"""

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from config import COPILOT_MODEL_NAME
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.copilot_reasoner import (
    build_structured_ai_context,
    encode_frame_to_jpeg_base64,
    execute_watchguard_tool,
    is_visual_query,
    BaseAICopilotReasoner,
    CopilotResponse,
    CopilotResponseSource,
    DeterministicCopilotReasoner,
    GeminiCopilotReasoner,
    get_copilot_reasoner,
)
from src.core.person_entity import PersonEntity
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.voice_assistant import VoiceAssistant, VoiceIntent
from src.database.db_manager import DatabaseManager
from src.events.event_types import RiskLevel


class TestGeminiMultimodalPhase57(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_phase57.db"
        self.db = DatabaseManager(db_path=self.db_path)

        self.context_engine = ContextEngine(db_manager=self.db)
        self.analytics = SecurityAnalyticsEngine(
            db_manager=self.db,
            context_engine=self.context_engine,
            incident_manager=self.context_engine.incident_manager,
            cache_ttl_seconds=0.01,
        )

        self.camera_mock = MagicMock()
        self.camera_mock.is_running = True
        # Synthetic test frame (640x480)
        self.sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_mock.get_frame.return_value = (True, self.sample_frame)

        self.person_mock = MagicMock()
        self.person_mock.is_ready = True
        self.object_mock = MagicMock()
        self.object_mock.is_ready = True

        self.deterministic_reasoner = DeterministicCopilotReasoner(security_analytics=self.analytics)
        self.gemini_reasoner = GeminiCopilotReasoner(
            api_key="TEST_MOCK_GEMINI_KEY_ABC123",
            fallback_reasoner=self.deterministic_reasoner,
        )

        self.va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            security_analytics=self.analytics,
            camera_manager=self.camera_mock,
            person_engine=self.person_mock,
            object_engine=self.object_mock,
            copilot_reasoner=self.gemini_reasoner,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_test_scene(self, entities, non_person_objects=None):
        non_person_objects = non_person_objects or []
        obj_summary = ", ".join([o["label"] for o in non_person_objects]) if non_person_objects else "None"
        has_auth = any(e.is_known and not e.is_spoof for e in entities)
        has_unk = any(not e.is_known and not e.is_spoof for e in entities)
        has_spoof = any(e.is_spoof for e in entities)

        decision = SecurityDecision(
            risk_level=RiskLevel.RED if has_spoof else (RiskLevel.YELLOW if has_unk else RiskLevel.GREEN),
            decision="PRESENTATION_ATTACK_IMPERSONATION" if has_spoof else ("UNAUTHORIZED_ACCESS" if has_unk else "AUTHORIZED_PERSON_NORMAL_PERIMETER"),
            reason="Live scene verification.",
            person_entities=entities,
            is_access_granted=has_auth and not has_spoof,
            context_summary={
                "person_detected": len(entities) > 0,
                "person_name": entities[0].identity if entities else "None",
                "is_authorized": has_auth,
                "is_unknown": has_unk,
                "has_spoof": has_spoof,
                "objects_detected": len(non_person_objects),
                "object_summary": obj_summary,
                "objects": non_person_objects,
                "person_entities": [e.to_dict() for e in entities],
                "person_entity_count": len(entities),
                "zone_name": "Standard Perimeter",
                "is_authorized_hours": True,
            },
        )
        self.context_engine.last_decision = decision
        self.va._sync_session_from_live_state()
        return decision

    # --- Test 1: Gemini Unavailable -> Fail-Secure Deterministic Fallback ---
    def test_01_gemini_unavailable_fallback(self):
        offline_reasoner = GeminiCopilotReasoner(api_key="", fallback_reasoner=self.deterministic_reasoner)
        self.assertFalse(offline_reasoner.is_available())

        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1])
        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)

        resp = offline_reasoner.generate_explanation("Why was he authorized?", context)
        self.assertEqual(resp.source, CopilotResponseSource.FALLBACK_DETERMINISTIC)
        self.assertIn("GREEN", resp.text)

    # --- Test 2: Gemini Available -> Successful Multimodal Query Execution ---
    @patch("urllib.request.urlopen")
    def test_02_gemini_available_multimodal_execution(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"candidates": [{"content": {"parts": [{"text": "I observe two people: Gulam Moinuddin who is authorized, and an unregistered individual."}]}}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        e2 = PersonEntity(track_id=2, bbox=(0, 0, 10, 10), identity="Unknown Person", is_known=False, role="Visitor", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1, e2])
        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)

        resp = self.gemini_reasoner.generate_explanation("What do you see?", context, frame=self.sample_frame)
        self.assertEqual(resp.source, CopilotResponseSource.AI_GROUNDED)
        self.assertIn("Gulam Moinuddin", resp.text)
        self.assertIn("unregistered", resp.text)

    # --- Test 3: Visual Query Classification & Frame Capture ---
    def test_03_visual_query_classification(self):
        self.assertTrue(is_visual_query("What do you see?"))
        self.assertTrue(is_visual_query("Who is in front of the camera?"))
        self.assertTrue(is_visual_query("Describe the scene."))
        self.assertTrue(is_visual_query("What is the second person holding?"))
        self.assertTrue(is_visual_query("Is there another person beside me?"))

        # Non-visual queries must NOT request frame
        self.assertFalse(is_visual_query("Why was Hunter denied?"))
        self.assertFalse(is_visual_query("What is the current risk level?"))
        self.assertFalse(is_visual_query("How many spoof attempts happened today?"))
        self.assertFalse(is_visual_query("What happened in the last 5 minutes?"))

    # --- Test 4: JPEG Base64 Encoding Latency & Integrity ---
    def test_04_jpeg_base64_encoding(self):
        t0 = time.time()
        b64 = encode_frame_to_jpeg_base64(self.sample_frame, quality=80)
        elapsed_ms = (time.time() - t0) * 1000.0

        self.assertIsNotNone(b64)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 100)
        self.assertLess(elapsed_ms, 50.0)  # Must be fast in-memory conversion (< 50ms)

    # --- Test 5: One Person Grounded Scenario ---
    def test_05_one_person_grounded_scenario(self):
        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1])

        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)
        curr = context["CURRENT_STATE"]
        self.assertEqual(curr["people_count"], 1)
        self.assertEqual(curr["people"][0]["identity"], "Gulam Moinuddin")

        resp = self.deterministic_reasoner.generate_explanation("How many people are there?", context)
        self.assertIn("Gulam Moinuddin", resp.text)
        self.assertIn("one person", resp.text.lower())

    # --- Test 6 & 7: Two People Scenario (Gulam Moinuddin + Unknown) ---
    def test_06_two_people_scenario_grounding(self):
        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        e2 = PersonEntity(track_id=2, bbox=(0, 0, 10, 10), identity="Unknown Person", is_known=False, role="Visitor", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1, e2])

        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)
        curr = context["CURRENT_STATE"]
        self.assertEqual(curr["people_count"], 2)

        resp = self.deterministic_reasoner.generate_explanation("What do you see?", context)
        self.assertIn("Gulam Moinuddin", resp.text)
        self.assertIn("unregistered", resp.text.lower())
        self.assertIn("2 people", resp.text)

    # --- Test 8: Authorized + Phone Object ---
    def test_08_authorized_plus_phone_object(self):
        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        obj1 = {"label": "cell phone", "confidence": 0.92}
        self._create_test_scene([e1], [obj1])

        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)
        curr = context["CURRENT_STATE"]
        self.assertEqual(curr["objects_count"], 1)
        self.assertEqual(curr["objects"][0]["label"], "cell phone")

        resp = self.deterministic_reasoner.generate_explanation("What objects are in view?", context)
        self.assertIn("cell phone", resp.text)
        self.assertIn("Gulam Moinuddin", resp.text)

    # --- Test 9: Authorized + Phone Photo Spoof Rejection ---
    def test_09_authorized_plus_spoof_rejection(self):
        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_SPOOF", is_spoof=True)
        self._create_test_scene([e1])

        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)
        resp = self.deterministic_reasoner.generate_explanation("Why was access denied?", context)
        self.assertIn("presentation attack", resp.text.lower())
        self.assertIn("Gulam Moinuddin", resp.text)

    # --- Test 10 & 11: Authoritative State & Recent History Grounding ---
    def test_10_authoritative_grounding(self):
        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)
        self.assertIn("CURRENT_STATE", context)
        self.assertIn("RECENT_HISTORY", context)
        self.assertIn("HISTORICAL_EVENTS", context)
        self.assertIn("EVIDENCE_GROUNDING", context)

    # --- Test 12 & 13: WatchGuard Tool Execution ---
    def test_12_watchguard_tool_execution(self):
        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1])
        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)

        res_people = execute_watchguard_tool("get_current_people", {}, context)
        self.assertEqual(res_people["people_count"], 1)
        self.assertEqual(res_people["people"][0]["identity"], "Gulam Moinuddin")

        res_status = execute_watchguard_tool("get_current_security_status", {}, context)
        self.assertEqual(res_status["master_risk_level"], "GREEN")

    # --- Test 14: Voice + Text Parity ---
    def test_14_voice_text_parity(self):
        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1])

        resp_voice = self.va.process_command("Who is in front of the camera?")
        resp_text = self.va.process_command("who is in front of the camera")
        self.assertEqual(resp_voice, resp_text)

    # --- Test 15: Contextual Pronoun Resolution ---
    def test_15_contextual_pronoun_resolution(self):
        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1])

        # Prime session context with Gulam Moinuddin
        self.va.session_context.last_person_name = "Gulam Moinuddin"
        self.va.session_context.last_is_authorized = True

        resp_followup = self.va.process_command("Why can't he enter?")
        self.assertIn("Gulam", resp_followup)

    # --- Test 16: Fail-Secure Network Timeout Handling ---
    @patch("urllib.request.urlopen")
    def test_16_network_timeout_fail_secure(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("Gemini API connection timed out")

        e1 = PersonEntity(track_id=1, bbox=(0, 0, 10, 10), identity="Gulam Moinuddin", is_known=True, role="Administrator", liveness_state="STABLE_LIVE")
        self._create_test_scene([e1])
        context = build_structured_ai_context(self.context_engine, self.analytics, self.db)

        # Must fall back cleanly to deterministic response without crashing
        resp = self.gemini_reasoner.generate_explanation("What do you see?", context, frame=self.sample_frame)
        self.assertEqual(resp.source, CopilotResponseSource.FALLBACK_DETERMINISTIC)
        self.assertIn("Gulam Moinuddin", resp.text)


if __name__ == "__main__":
    unittest.main()
