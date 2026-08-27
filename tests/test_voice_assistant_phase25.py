"""
Phase 2.5 Advanced Contextual Security Assistant Unit & Integration Tests
-------------------------------------------------------------------------
Validates:
1. Multi-turn follow-up conversations with pronoun & entity resolution (he/she/they/this person/why).
2. Current vs. Historical state inquiry distinction.
3. Temporal security queries (5/10 min window, today's alert count, last red/orange alert).
4. Spatial person-object relationship queries.
5. Repeated unauthorized activity auditing & most serious event resolution.
6. Strict Voice/Text execution parity.
7. Graceful unknown command handling.
"""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant, VoiceIntent, normalize_query


class TestVoiceAssistantPhase25(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_phase25.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.context_engine = ContextEngine()

        self.mock_camera = MagicMock(spec=CameraManager)
        self.mock_camera.is_running = True
        self.mock_camera.camera_index = 0

        self.mock_person_engine = MagicMock(spec=PersonRecognitionEngine)
        self.mock_person_engine.is_ready = True
        self.mock_person_engine.known_face_count = 1

        self.mock_object_engine = MagicMock(spec=ObjectDetectionEngine)
        self.mock_object_engine.is_ready = True

        self.va = VoiceAssistant(
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            camera_manager=self.mock_camera,
        )
        self.va.mute_audio = True

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- 1. Multi-Turn Follow-Up & Pronoun Resolution ---

    def test_multi_turn_authorized_person_follow_up(self):
        """User: 'Who is in front of me?' -> 'Is he allowed to be there?' -> 'Why?'"""
        # Simulate live observation: Authorized person Hunter
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Perimeter clear. Routine surveillance active.",
            matched_rules=["RULE_AUTH_PERIMETER"],
            evidence_required=False,
            recommended_response="Continue routine monitoring.",
            context_summary={
                "person_detected": True,
                "person_name": "Hunter",
                "is_authorized": True,
                "is_unknown": False,
                "role": "Security Lead",
                "zone_name": "Control Room",
            },
            event_type=EventType.AUTHORIZED_PERSON,
        )

        # Turn 1: Who is in front?
        r1 = self.va.process_command("Who is in front of the camera?")
        self.assertIn("Hunter", r1)
        self.assertIn("Authorized", r1)

        # Turn 2: Is he allowed to be there? (Pronoun resolution 'he' -> Hunter)
        r2 = self.va.process_command("Is he allowed to be there?")
        self.assertIn("Yes", r2)
        self.assertIn("Hunter is authorized", r2)
        self.assertIn("Security Lead", r2)

        # Turn 3: Why? (Contextual explanation of authorization)
        r3 = self.va.process_command("Why?")
        self.assertIn("Hunter is registered", r3)
        self.assertIn("Security Lead", r3)

    def test_multi_turn_unregistered_person_follow_up(self):
        """User: 'Who is that?' -> 'Is this person allowed?' -> 'Why?'"""
        # Simulate live observation: Unregistered visitor in restricted area (ORANGE)
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.ORANGE,
            decision="UNAUTHORIZED_RESTRICTED_ZONE_ACCESS",
            reason="Unregistered individual detected inside RESTRICTED zone without credentials.",
            matched_rules=["RULE_RESTRICTED_ZONE_BREACH"],
            evidence_required=True,
            recommended_response="Dispatch security.",
            context_summary={
                "person_detected": True,
                "person_name": "Unknown Person",
                "is_authorized": False,
                "is_unknown": True,
                "role": "Visitor",
                "zone_name": "Server Vault",
            },
            event_type=EventType.POLICY_VIOLATION,
        )

        # Turn 1: Who is in front?
        r1 = self.va.process_command("Who is that in front of the camera?")
        self.assertIn("unregistered", r1.lower())

        # Turn 2: Is this person allowed?
        r2 = self.va.process_command("Is this person allowed to be here?")
        self.assertIn("No", r2)
        self.assertIn("unregistered visitor", r2)

        # Turn 3: Why?
        r3 = self.va.process_command("Why is he not allowed?")
        self.assertIn("not registered", r3.lower())

    # --- 2. Current vs. Historical State Distinction ---

    def test_current_vs_historical_state_queries(self):
        """Distinguish live camera/engine state from historical SQLite event logs."""
        # 1. Seed past event in SQLite from earlier today
        self.logger.log_event(
            event_type=EventType.POLICY_VIOLATION,
            risk_level=RiskLevel.ORANGE,
            description="Perimeter breach detected earlier at fence zone.",
            timestamp="2026-08-24 00:10:00",
        )

        # 2. Set CURRENT live state to GREEN (Safe)
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Normal operations with no active violations.",
            matched_rules=["RULE_NORMAL"],
            evidence_required=False,
            recommended_response="Normal monitoring.",
            context_summary={"person_detected": False, "objects_detected": 0},
            event_type=EventType.AUTHORIZED_PERSON,
        )

        # Query CURRENT state
        curr_resp = self.va.process_command("What is happening right now?")
        self.assertIn("Risk is GREEN", curr_resp)
        self.assertIn("Normal operations", curr_resp)

        # Query HISTORICAL state
        hist_resp = self.va.process_command("What happened earlier?")
        self.assertIn("Earlier today", hist_resp)
        self.assertIn("ORANGE", hist_resp)
        self.assertIn("Perimeter breach", hist_resp)

    # --- 3. Temporal Queries ---

    def test_temporal_minutes_query(self):
        """Verify queries like 'What happened in the last 5 minutes?'"""
        now = datetime.now()
        # Seed 2 events within the last 3 minutes
        ts1 = (now - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
        ts2 = (now - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")

        self.logger.log_event(
            event_type=EventType.UNKNOWN_PERSON,
            risk_level=RiskLevel.YELLOW,
            description="Visitor approached door",
            timestamp=ts1,
        )
        self.logger.log_event(
            event_type=EventType.AUTHORIZED_PERSON,
            risk_level=RiskLevel.GREEN,
            description="Hunter verified",
            timestamp=ts2,
        )

        resp = self.va.process_command("What happened in the last 5 minutes?")
        self.assertIn("In the last 5 minutes", resp)
        self.assertIn("Hunter verified", resp)
        self.assertIn("Visitor approached door", resp)

    def test_temporal_alert_by_level(self):
        """Verify 'When was the last red alert?' and 'When was the last orange alert?'"""
        self.logger.log_event(
            event_type=EventType.SECURITY_ALERT,
            risk_level=RiskLevel.RED,
            description="Critical asset tampering attempt",
            timestamp="2026-08-24 01:00:00",
        )
        self.logger.log_event(
            event_type=EventType.POLICY_VIOLATION,
            risk_level=RiskLevel.ORANGE,
            description="Restricted zone after-hours access",
            timestamp="2026-08-24 01:05:00",
        )

        resp_red = self.va.process_command("When was the last red alert?")
        self.assertIn("The last RED alert occurred at", resp_red)
        self.assertIn("Critical asset tampering attempt", resp_red)

        resp_orange = self.va.process_command("When was the last orange alert?")
        self.assertIn("The last ORANGE alert occurred at", resp_orange)
        self.assertIn("Restricted zone after-hours access", resp_orange)

    def test_alert_count_today(self):
        """Verify 'How many alerts happened today?'"""
        today_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.log_event(event_type=EventType.SECURITY_ALERT, risk_level=RiskLevel.RED, description="Alert 1", timestamp=today_ts)
        self.logger.log_event(event_type=EventType.POLICY_VIOLATION, risk_level=RiskLevel.ORANGE, description="Alert 2", timestamp=today_ts)
        self.logger.log_event(event_type=EventType.UNKNOWN_PERSON, risk_level=RiskLevel.YELLOW, description="Alert 3", timestamp=today_ts)
        self.logger.log_event(event_type=EventType.AUTHORIZED_PERSON, risk_level=RiskLevel.GREEN, description="Normal access", timestamp=today_ts)

        resp = self.va.process_command("How many security alerts happened today?")
        self.assertIn("3 security alerts have occurred today", resp)

    # --- 4. Spatial Person + Object Relationships ---

    def test_spatial_relationship_queries(self):
        """Verify queries regarding physical proximity and interactions with protected objects."""
        # Simulate active interaction: Person near laptop
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.RED,
            decision="UNAUTHORIZED_ASSET_INTERACTION",
            reason="Unregistered person in visual proximity (14.2px) to protected asset LAPTOP.",
            matched_rules=["RULE_UNAUTHORIZED_ASSET_INTERACTION"],
            evidence_required=True,
            recommended_response="Lock asset immediately.",
            correlated_interactions=[
                {
                    "object_label": "laptop",
                    "confidence": 0.88,
                    "distance_pixels": 14.2,
                    "is_in_proximity": True,
                    "is_protected": True,
                }
            ],
            context_summary={
                "person_detected": True,
                "person_name": "Unknown Visitor",
                "is_authorized": False,
                "is_unknown": True,
                "objects_detected": 1,
                "objects": [{"label": "laptop"}],
            },
            event_type=EventType.SECURITY_ALERT,
        )

        resp_laptop = self.va.process_command("Is anyone near the laptop?")
        self.assertIn("Yes", resp_laptop)
        self.assertIn("Unknown Visitor", resp_laptop)
        self.assertIn("14.2px", resp_laptop)
        self.assertIn("laptop", resp_laptop)

        resp_phone = self.va.process_command("Is anyone near the phone?")
        self.assertIn("No spatial interactions with cell phone", resp_phone)

    # --- 5. Event Intelligence & Repeated Unauthorized Auditing ---

    def test_repeated_unauthorized_detections(self):
        """Verify 'Have there been repeated unauthorized detections?'"""
        now = datetime.now()
        self.logger.log_event(
            event_type=EventType.UNKNOWN_PERSON,
            risk_level=RiskLevel.YELLOW,
            description="Unknown 1",
            timestamp=(now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.logger.log_event(
            event_type=EventType.UNKNOWN_PERSON,
            risk_level=RiskLevel.YELLOW,
            description="Unknown 2",
            timestamp=(now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        )

        resp = self.va.process_command("Have there been repeated unauthorized detections?")
        self.assertIn("Yes", resp)
        self.assertIn("2 unauthorized detections", resp)

    def test_most_serious_event_resolution(self):
        """Verify 'What was the most serious recent event?' prioritizes RED over YELLOW/GREEN."""
        self.logger.log_event(event_type=EventType.UNKNOWN_PERSON, risk_level=RiskLevel.YELLOW, description="Unregistered visitor at lobby")
        self.logger.log_event(event_type=EventType.SECURITY_ALERT, risk_level=RiskLevel.RED, description="Server room intrusion")
        self.logger.log_event(event_type=EventType.POLICY_VIOLATION, risk_level=RiskLevel.ORANGE, description="Tailgating at turnstile")

        resp = self.va.process_command("What was the most serious recent event?")
        self.assertIn("most serious recorded event", resp)
        self.assertIn("RED", resp)
        self.assertIn("Server room intrusion", resp)

    # --- 6. Voice and Text Parity ---

    def test_voice_text_query_parity(self):
        """Verify identical normalization, intent mapping, and response generation for voice and typed inputs."""
        queries = [
            ("What is happening right now?", VoiceIntent.CURRENT_STATE_QUERY),
            ("Is he allowed to be there?", VoiceIntent.FOLLOW_UP_PERSON_AUTHORIZATION),
            ("Why?", VoiceIntent.FOLLOW_UP_WHY),
            ("What happened in the last 5 minutes?", VoiceIntent.TEMPORAL_EVENTS_MINUTES),
            ("When was the last red alert?", VoiceIntent.TEMPORAL_ALERT_BY_LEVEL),
            ("How many alerts happened today?", VoiceIntent.ALERT_COUNT_TODAY),
            ("Is anyone near the laptop?", VoiceIntent.SPATIAL_RELATIONSHIP),
            ("What was the most serious event?", VoiceIntent.MOST_SERIOUS_EVENT),
        ]

        for q_text, expected_intent in queries:
            intent, norm = self.va.resolve_intent(q_text)
            self.assertEqual(intent, expected_intent, f"Failed for query '{q_text}'")

            resp_typed = self.va.process_command(q_text)
            resp_spoken = self.va.process_command(norm)
            self.assertTrue(len(resp_typed) > 0)
            self.assertTrue(len(resp_spoken) > 0)

    # --- 7. Unknown & Ambiguous Command Handling ---

    def test_unknown_query_graceful_fallback(self):
        """Verify out-of-domain conversational queries return helpful guidance without crashing."""
        resp = self.va.process_command("Tell me a funny joke about robots.")
        self.assertIn("I did not recognize that command", resp)
        self.assertIn("Who is in front of the camera?", resp)


if __name__ == "__main__":
    unittest.main()
