"""
Unit Tests for Voice Assistant Module (Phase 2.4)
-------------------------------------------------
Automated tests covering:
- Voice and Typed query parity (both use exact same intent & state engine)
- Recent Security Activity hard limit of 3 events (SQL LIMIT 3)
- Latest Alert hard limit of 1 non-GREEN event (SQL LIMIT 1)
- Text fallback fully functional during microphone / speech recognition failure
- Natural command parsing and intent resolution for all 10 supported security intents
- At least 5 natural language variants per intent (50+ test queries)
- Query normalization: lowercasing, punctuation stripping, contraction expansion, filler word tolerance
- "Who is in front of the camera?" (Authorized, Unregistered, Perimeter Clear)
- "What do you see?" (Integrated person & objects scene description)
- "Is there any security risk?" (GREEN, YELLOW, ORANGE, RED threat levels)
- "Why is it high risk?" (Traceable rationale extraction)
- "What objects are detected?" (Active object breakdown)
- "Who was detected most recently?" (SQLite historical lookup)
- "Read the latest alert." (High priority event retrieval)
- Unavailable state & camera offline handling
- Unknown command / fallback handling
- Audio mute toggle handling
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.core.context_engine import ContextEngine
from src.core.voice_assistant import VoiceAssistant, VoiceIntent, normalize_query


class TestVoiceAssistant(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_voice.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.context_engine = ContextEngine()

        self.mock_camera = MagicMock()
        self.mock_camera.is_running = True
        self.mock_camera.camera_index = 0

        self.mock_person_engine = MagicMock()
        self.mock_person_engine.is_ready = True

        self.mock_object_engine = MagicMock()
        self.mock_object_engine.is_ready = True

        self.va = VoiceAssistant(
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            camera_manager=self.mock_camera,
        )
        # Mute audio during automated test runs
        self.va.mute_audio = True

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Query Normalization & Intent Resolution Tests ---

    def test_query_normalization(self):
        """Test lowercase, punctuation stripping, contraction expansion, and filler word tolerance."""
        self.assertEqual(normalize_query("Who's in front of me?"), "who is in front of me")
        self.assertEqual(normalize_query("What's visible right now?!"), "what is visible right now")
        self.assertEqual(normalize_query("Hey WatchGuard, can you please tell me what you see?"), "what you see")
        self.assertEqual(normalize_query("Is there a threat???"), "is there a threat")
        self.assertEqual(normalize_query("   Tell me what happened just now...  "), "what happened just now")

    def test_intent_resolution_variants_5_per_intent(self):
        """Test at least 5 natural language query variants for all 10 supported security intents."""
        test_matrix = {
            VoiceIntent.WHO_IS_IN_FRONT: [
                "who is in front of the camera",
                "who is in front",
                "who do you see",
                "who is there",
                "who's in front of me",
                "identify the person",
                "who is that standing there",
                "who is on camera",
            ],
            VoiceIntent.WHAT_DO_YOU_SEE: [
                "what do you see",
                "what can you see",
                "what is in front of the camera",
                "what is visible",
                "what is around me",
                "describe the scene",
                "what do you observe",
            ],
            VoiceIntent.SECURITY_RISK: [
                "is there any security risk",
                "is there a security threat",
                "is anything wrong",
                "is everything safe",
                "what is the current threat level",
                "is there a threat",
                "are there any risks",
                "are we safe",
            ],
            VoiceIntent.WHY_HIGH_RISK: [
                "why is the system showing high risk",
                "why is it high risk",
                "why is the threat level orange",
                "explain the alert",
                "why is risk red",
                "why is the risk elevated",
            ],
            VoiceIntent.CURRENT_SECURITY_STATUS: [
                "what is the current security status",
                "what is the system status",
                "how is security right now",
                "system status update",
                "give me a status report",
                "what is the operational status",
            ],
            VoiceIntent.DETECTED_OBJECTS: [
                "what objects are detected",
                "what objects do you see",
                "what objects are around",
                "what are you detecting",
                "list detected objects",
                "show detected objects",
            ],
            VoiceIntent.RECENT_SECURITY_EVENTS: [
                "show recent security events",
                "what happened recently",
                "what happened just now",
                "what are the latest security events",
                "what are the latest alerts",
                "show me recent activity",
                "tell me what happened",
                "what happened in the last few minutes",
            ],
            VoiceIntent.LATEST_ALERT: [
                "what is the latest alert",
                "what was the last security alert",
                "latest incident",
                "most recent warning",
                "read the latest alert",
            ],
            VoiceIntent.RECENT_PERSON_DETECTED: [
                "who was detected most recently",
                "who was the last person seen",
                "who was here last",
                "last detected person",
                "who visited recently",
            ],
            VoiceIntent.STOP_LISTENING: [
                "stop listening",
                "cancel",
                "never mind",
                "standby",
                "stop",
                "go to sleep",
            ],
        }

        for expected_intent, queries in test_matrix.items():
            for q in queries:
                resolved_intent, norm = self.va.resolve_intent(q)
                self.assertEqual(
                    resolved_intent,
                    expected_intent,
                    f"Failed query: '{q}' -> Got {resolved_intent.name}, expected {expected_intent.name} (Normalized: '{norm}')",
                )

    # --- Voice & Typed Parity Tests (Unified Intent Engine) ---

    def test_voice_and_typed_identical_responses_what_happened_recently(self):
        """Verify Voice transcription and Typed input produce identical responses for 'what happened recently?'."""
        self.logger.log_event(EventType.AUTHORIZED_PERSON, RiskLevel.GREEN, "Hunter logged in.", person_name="Hunter")
        self.logger.log_event(EventType.SECURITY_ALERT, RiskLevel.YELLOW, "Unregistered person detected.")

        voice_transcription = "What happened recently?"
        typed_input = "what happened recently"

        voice_resp = self.va.process_command(voice_transcription)
        typed_resp = self.va.process_command(typed_input)

        self.assertEqual(voice_resp, typed_resp)
        self.assertIn("latest 2 events", voice_resp)
        self.assertIn("YELLOW", voice_resp)

    def test_voice_and_typed_identical_responses_who_is_in_front_of_me(self):
        """Verify Voice transcription and Typed input produce identical responses for 'who is in front of me?'."""
        p_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Hunter", "similarity": 0.94, "bbox": (100, 100, 100, 100)}],
        }
        self.context_engine.evaluate_context(p_data, {"detected": False, "count": 0, "objects": []})

        voice_transcription = "Who is in front of me?"
        typed_input = "who is in front of me"

        voice_resp = self.va.process_command(voice_transcription)
        typed_resp = self.va.process_command(typed_input)

        self.assertEqual(voice_resp, typed_resp)
        self.assertIn("Hunter", voice_resp)

    def test_voice_and_typed_identical_responses_is_there_any_security_risk(self):
        """Verify Voice transcription and Typed input produce identical responses for 'is there any security risk?'."""
        self.context_engine.evaluate_context({"detected": False, "people": []}, {"detected": False, "count": 0, "objects": []})

        voice_transcription = "Is there any security risk?"
        typed_input = "is there any security risk"

        voice_resp = self.va.process_command(voice_transcription)
        typed_resp = self.va.process_command(typed_input)

        self.assertEqual(voice_resp, typed_resp)
        self.assertIn("GREEN", voice_resp)

    # --- Hard Limit Tests for Recent Activity and Latest Alert ---

    def test_recent_security_events_hard_limit_of_3(self):
        """Verify Recent Activity query returns at most 3 events using direct SQL LIMIT 3."""
        # Insert 10 events into SQLite
        for i in range(1, 11):
            self.logger.log_event(EventType.SECURITY_ALERT, RiskLevel.YELLOW, f"Event number {i}")

        resp = self.va.process_command("What happened recently?")
        self.assertIn("latest 3 events", resp)
        self.assertIn("1.", resp)
        self.assertIn("2.", resp)
        self.assertIn("3.", resp)
        self.assertNotIn("4.", resp)
        self.assertIn("Event number 10", resp)  # newest first
        self.assertIn("Event number 9", resp)
        self.assertIn("Event number 8", resp)

    def test_latest_alert_hard_limit_of_1_non_green(self):
        """Verify Latest Alert query returns at most 1 non-GREEN event using direct SQL WHERE risk_level != 'GREEN' LIMIT 1."""
        self.logger.log_event(EventType.AUTHORIZED_PERSON, RiskLevel.GREEN, "Routine access 1.")
        self.logger.log_event(EventType.SECURITY_ALERT, RiskLevel.YELLOW, "First warning: Unverified face.")
        self.logger.log_event(EventType.AUTHORIZED_PERSON, RiskLevel.GREEN, "Routine access 2.")
        self.logger.log_event(EventType.POLICY_VIOLATION, RiskLevel.ORANGE, "Latest critical incident: Perimeter breach.")
        self.logger.log_event(EventType.AUTHORIZED_PERSON, RiskLevel.GREEN, "Routine access 3.")

        resp = self.va.process_command("What is the latest alert?")
        self.assertIn("Latest security alert", resp)
        self.assertIn("ORANGE", resp)
        self.assertIn("Perimeter breach", resp)
        self.assertNotIn("Unverified face", resp)

    def test_text_fallback_functional_when_microphone_fails(self):
        """Verify text fallback operates seamlessly without microphone or audio hardware."""
        p_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Hunter", "bbox": (100, 100, 100, 100)}],
        }
        self.context_engine.evaluate_context(p_data, {"detected": False, "count": 0, "objects": []})

        # Process command directly via text entry (simulating zero microphone presence)
        resp = self.va.process_command("Who is in front of the camera?")
        self.assertIn("Hunter", resp)

    # --- Live State Query Tests ---

    def test_query_current_person_authorized(self):
        """Test 'Who is in front of the camera?' with authorized user Hunter."""
        p_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Hunter", "similarity": 0.94, "bbox": (100, 100, 100, 100)}],
        }
        o_data = {"detected": False, "count": 0, "objects": []}
        self.context_engine.evaluate_context(p_data, o_data)

        resp = self.va.process_command("Who is in front of the camera?")
        self.assertIn("Hunter", resp)
        self.assertIn("Authorized", resp)

    def test_query_current_person_unregistered(self):
        """Test 'Who is in front of the camera?' with unknown visitor."""
        p_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": True,
            "people": [{"name": "Unknown", "similarity": 0.2, "bbox": (100, 100, 100, 100)}],
        }
        o_data = {"detected": False, "count": 0, "objects": []}
        self.context_engine.evaluate_context(p_data, o_data)

        resp = self.va.process_command("Who is that in front of the camera?")
        self.assertIn("unregistered", resp.lower())

    def test_query_current_person_clear(self):
        """Test 'Who is in front of the camera?' when no person is detected."""
        p_data = {"detected": False, "people": []}
        o_data = {"detected": False, "count": 0, "objects": []}
        self.context_engine.evaluate_context(p_data, o_data)

        resp = self.va.process_command("Who is in front of the camera?")
        self.assertIn("do not detect any person", resp.lower())

    def test_query_scene_overview(self):
        """Test 'What do you see?' with person and objects."""
        p_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Hunter", "bbox": (100, 100, 100, 100)}],
        }
        o_data = {
            "detected": True,
            "count": 2,
            "objects": [{"label": "laptop"}, {"label": "cell phone"}],
            "summary": "laptop, cell phone",
        }
        self.context_engine.evaluate_context(p_data, o_data)

        resp = self.va.process_command("What do you see?")
        self.assertIn("Hunter", resp)
        self.assertIn("laptop", resp)

    def test_query_security_risk_green_and_incident(self):
        """Test 'Is there any security risk?' across threat levels."""
        # GREEN state
        p_data = {"detected": False, "people": []}
        o_data = {"detected": False, "count": 0, "objects": []}
        self.context_engine.evaluate_context(p_data, o_data)

        resp_green = self.va.process_command("Is there any security risk?")
        self.assertIn("GREEN", resp_green)
        self.assertIn("Normal", resp_green)

        # RED state (Unknown interacting with laptop)
        p_data = {"detected": True, "has_unknown": True, "people": [{"name": "Unknown", "bbox": (100, 100, 100, 100)}]}
        o_data = {"detected": True, "count": 1, "objects": [{"label": "laptop", "bbox": (110, 110, 80, 80)}], "summary": "laptop"}
        self.context_engine.evaluate_context(p_data, o_data)

        resp_red = self.va.process_command("Are there any security risks?")
        self.assertIn("RED", resp_red)
        self.assertIn("Critical", resp_red)

    def test_query_why_high_risk(self):
        """Test 'Why is the system showing high risk?' for traceable explanation."""
        p_data = {"detected": True, "has_unknown": True, "people": [{"name": "Unknown", "bbox": (100, 100, 100, 100)}]}
        o_data = {"detected": True, "count": 1, "objects": [{"label": "laptop", "bbox": (110, 110, 80, 80)}], "summary": "laptop"}
        self.context_engine.evaluate_context(p_data, o_data)

        resp = self.va.process_command("Why is the system showing high risk?")
        self.assertIn("POTENTIAL_UNAUTHORIZED_INTERACTION", resp)
        self.assertIn("Traceable reason:", resp)

    def test_query_detected_objects(self):
        """Test 'What objects are detected?'."""
        p_data = {"detected": False, "people": []}
        o_data = {"detected": True, "count": 1, "objects": [{"label": "backpack"}], "summary": "backpack"}
        self.context_engine.evaluate_context(p_data, o_data)

        resp = self.va.process_command("What objects are detected?")
        self.assertIn("backpack", resp)

    def test_query_recent_person_from_db(self):
        """Test 'Who was detected most recently?'."""
        self.logger.log_event(
            event_type=EventType.AUTHORIZED_PERSON,
            risk_level=RiskLevel.GREEN,
            description="Hunter logged in.",
            person_name="Hunter",
        )

        resp = self.va.process_command("Who was detected most recently?")
        self.assertIn("Hunter", resp)

    def test_query_stop_listening(self):
        """Test 'Stop listening.'."""
        self.va._is_listening = True
        resp = self.va.process_command("Stop listening.")
        self.assertFalse(self.va.is_listening)
        self.assertIn("standby", resp.lower())

    def test_camera_offline_handling(self):
        """Test handling when camera is stopped."""
        self.mock_camera.is_running = False
        resp = self.va.process_command("Who is in front of the camera?")
        self.assertIn("stopped", resp.lower())

    def test_unknown_command_fallback(self):
        """Test unsupported command response."""
        resp = self.va.process_command("What is the weather in Tokyo?")
        self.assertIn("did not recognize", resp.lower())
        self.assertIn("Who is in front of the camera", resp)

    def test_empty_command(self):
        """Test empty query string handling."""
        resp = self.va.process_command("   ")
        self.assertIn("specify a question", resp.lower())

    def test_mute_audio_setting(self):
        """Verify mute_audio flag prevents speaking without errors."""
        self.va.mute_audio = True
        self.va.speak("Test silent response", non_blocking=False)
        self.assertFalse(self.va.is_speaking)


if __name__ == "__main__":
    unittest.main()
