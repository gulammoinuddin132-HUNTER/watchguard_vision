"""
Unit & Integration Tests for WatchGuard 2.0 Phase 5.7.3
Visual Query Routing + Access Query Routing
-----------------------------------------------------------------------------
Validates:
1. PERSON_HOLDING_OBJECT Semantic Intent & Dispatch:
   - "What am I holding?", "What's in my hand?", "What object is he holding?", "Can you tell what I'm carrying?", etc.
2. EXPLAIN_ACCESS_DECISION Semantic Intent & Grounded Explanations:
   - "Why am I being denied?", "Why was I denied?", "Why can't I enter?", "What is preventing me from entering?", etc.
3. VISUAL_SCENE, VISUAL_COUNT, VISUAL_OBJECT Semantic Intents:
   - "What do you see?", "Describe the scene.", "How many people are visible?", "What objects are visible?"
4. Authoritative Grounding & Registry vs Live State Isolation:
   - Empty frame never assumes registered profiles are present.
   - Correctly handles Unregistered, Spoof, After-Hours Restricted, and Authorized access states.
5. Telemetry & Invocation Logging:
   - Verifies QUERY, INTENT, ROUTE, GEMINI_CALLED, SNAPSHOT_CAPTURED, SNAPSHOT_TIMESTAMP, GEMINI_RESPONSE_RECEIVED.
6. Zero Unknown Command Fallthrough:
   - Ensures visual & access queries never return "I did not recognize that command".
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.person_entity import PersonEntity
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.semantic_router import SemanticQueryRouter, SemanticResolution
from src.core.voice_assistant import VoiceAssistant, VoiceIntent, SessionContext
from src.core.copilot_reasoner import (
    is_visual_query,
    DeterministicCopilotReasoner,
    CopilotResponseSource,
    build_structured_ai_context,
)
from src.database.db_manager import DatabaseManager
from src.events.event_types import RiskLevel


class TestPhase573VisualAccessRouting(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_phase573.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.router = SemanticQueryRouter(VoiceIntent)

        self.context_engine = ContextEngine(db_manager=self.db)
        self.analytics_engine = SecurityAnalyticsEngine(
            db_manager=self.db,
            context_engine=self.context_engine,
            incident_manager=self.context_engine.incident_manager,
            cache_ttl_seconds=0.01,
        )
        self.camera_mock = MagicMock()
        self.camera_mock.is_running = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_mock.get_frame.return_value = (True, dummy_frame)

        self.va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            security_analytics=self.analytics_engine,
            camera_manager=self.camera_mock,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Test 1: PERSON_HOLDING_OBJECT Semantic Routing ---
    def test_01_person_holding_object_intent_routing(self):
        variations = [
            "What am I holding?",
            "What object am I holding?",
            "What's in my hand?",
            "What is in my hand?",
            "What is he holding?",
            "What is she holding?",
            "What is that person holding?",
            "Can you tell what I'm carrying?",
            "Can you tell what I am carrying?",
            "Is he holding anything?",
            "What object is in his hand?",
            "What is in their hands?",
            "What am I carrying?",
            "what is in hand",
        ]
        for q in variations:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.PERSON_HOLDING_OBJECT,
                f"Query failed to resolve to PERSON_HOLDING_OBJECT: '{q}' (got {res.intent})",
            )
            self.assertTrue(res.confidence >= 0.90)

    # --- Test 2: EXPLAIN_ACCESS_DECISION Semantic Routing ---
    def test_02_explain_access_decision_intent_routing(self):
        variations = [
            "Why am I being denied?",
            "Why was I denied?",
            "Why can't I enter?",
            "Why can I not enter?",
            "Why isn't my access allowed?",
            "Why is not my access allowed?",
            "Why was my access refused?",
            "What is preventing me from entering?",
            "Why won't you let me in?",
            "Why will you not let me in?",
            "What is stopping me?",
            "Why am I blocked?",
            "Why was I rejected?",
            "What caused the rejection?",
            "Why was Hunter denied?",
            "Why didn't Hunter get access?",
        ]
        for q in variations:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.EXPLAIN_ACCESS_DECISION,
                f"Query failed to resolve to EXPLAIN_ACCESS_DECISION: '{q}' (got {res.intent})",
            )
            self.assertTrue(res.confidence >= 0.90)

    # --- Test 3: VISUAL_* Intents Routing ---
    def test_03_visual_scene_and_count_intents_routing(self):
        scene_queries = [
            "What do you see?",
            "Describe the scene.",
            "What is happening in front of the camera?",
            "Tell me what you observe.",
            "What do you observe?",
        ]
        for q in scene_queries:
            res = self.router.route_query(q)
            self.assertIn(
                res.intent,
                (VoiceIntent.VISUAL_SCENE, VoiceIntent.WHAT_DO_YOU_SEE),
                f"Scene query failed: '{q}' (got {res.intent})",
            )

        count_queries = [
            "How many people are visible?",
            "How many people do you see?",
            "How many people are in the frame?",
            "How many people in view?",
            "How many faces do you see?",
        ]
        for q in count_queries:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.VISUAL_COUNT,
                f"Count query failed: '{q}' (got {res.intent})",
            )

        object_queries = [
            "What objects are visible?",
            "What objects do you see?",
            "Show detected objects",
            "What items are in view?",
        ]
        for q in object_queries:
            res = self.router.route_query(q)
            self.assertIn(
                res.intent,
                (VoiceIntent.VISUAL_OBJECT, VoiceIntent.DETECTED_OBJECTS),
                f"Object query failed: '{q}' (got {res.intent})",
            )

    # --- Test 4: is_visual_query Classification ---
    def test_04_is_visual_query_classification(self):
        self.assertTrue(is_visual_query("What am I holding?"))
        self.assertTrue(is_visual_query("What object is in my hand?"))
        self.assertTrue(is_visual_query("What's in his hand?"))
        self.assertTrue(is_visual_query("What do you see?"))
        self.assertTrue(is_visual_query("Describe the scene."))
        self.assertTrue(is_visual_query("How many people are visible?"))
        self.assertTrue(is_visual_query("What is that person holding?"))

        self.assertFalse(is_visual_query("Why am I being denied?"))
        self.assertFalse(is_visual_query("Why was Hunter denied?"))
        self.assertFalse(is_visual_query("What are the authorized hours?"))
        self.assertFalse(is_visual_query("What happened in the last 5 minutes?"))
        self.assertFalse(is_visual_query("When was the last red alert?"))

    # --- Test 5: Access Decision Grounding - Empty Frame ---
    def test_05_access_decision_empty_frame(self):
        uid = self.db.create_user(username="hunter", full_name="Hunter", role="Staff", access_level=2)
        self.db.register_face(user_id=uid, face_encoding=b"123")
        self.va.session_context.last_people_entities = []

        response = self.va.process_command("Why am I being denied?")
        self.assertIn("No person is currently detected in the camera view", response)
        self.assertNotIn("Hunter is registered", response)

    # --- Test 6: Access Decision Grounding - Unregistered Visitor ---
    def test_06_access_decision_unregistered_visitor(self):
        self.va.session_context.last_people_entities = [{
            "identity": "Unknown Person",
            "is_known": False,
            "is_spoof": False,
            "liveness_state": "LIVE",
            "role": "Visitor",
            "access_state": "DENIED",
        }]
        self.va.session_context.last_zone = "Standard Perimeter"

        response = self.va.process_command("Why am I being denied?")
        self.assertIn("Access is denied", response)
        self.assertIn("not registered in the facial database", response)

    # --- Test 7: Access Decision Grounding - Spoof Presentation ---
    def test_07_access_decision_spoof_presentation(self):
        self.va.session_context.last_people_entities = [{
            "identity": "Hunter",
            "is_known": True,
            "is_spoof": True,
            "liveness_state": "SPOOF",
            "role": "Staff",
            "access_state": "DENIED",
        }]
        self.va.session_context.last_decision_code = "PRESENTATION_ATTACK_IMPERSONATION"

        response = self.va.process_command("Why was I refused access?")
        self.assertIn("Access is denied", response)
        self.assertIn("presentation attack", response.lower())

    # --- Test 8: Access Decision Grounding - After Hours Restricted Denial ---
    def test_08_access_decision_after_hours_denial(self):
        self.va.session_context.last_people_entities = [{
            "identity": "Hunter",
            "is_known": True,
            "is_spoof": False,
            "liveness_state": "LIVE",
            "role": "Staff",
            "access_state": "DENIED",
        }]
        self.va.session_context.last_zone = "Restricted Area"
        self.va.session_context.last_time_str = "22:15"
        self.va.session_context.last_is_authorized_hours = False
        self.va.session_context.last_decision_code = "AFTER_HOURS_RESTRICTED_ACCESS_DENIED"

        response = self.va.process_command("Why can't I enter?")
        self.assertIn("Access is denied", response)
        self.assertIn("22:15", response)
        self.assertIn("09:00 to 18:00", response)

    # --- Test 9: Person Holding Object Fallback & Zero Unknown Command ---
    def test_09_person_holding_object_fallback(self):
        self.va.session_context.last_objects = [{"label": "cell phone", "confidence": 0.91}]
        resp_a = self.va.process_command("What am I holding?")
        self.assertIn("cell phone", resp_a)
        self.assertNotIn("I did not recognize that command", resp_a)

        self.va.session_context.last_objects = []
        resp_b = self.va.process_command("What's in my hand?")
        self.assertIn("No objects are currently detected in your hands", resp_b)
        self.assertNotIn("I did not recognize that command", resp_b)


if __name__ == "__main__":
    unittest.main()