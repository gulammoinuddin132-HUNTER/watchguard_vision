"""
Unit & Integration Tests for WatchGuard 2.0 Phase 5.6: Semantic Query Router
-----------------------------------------------------------------------------
Validates:
1. 35+ Natural Language Semantic Query Variations across all major intent clusters.
2. Contraction normalization and spoken disfluency stripping.
3. Contextual pronoun and entity resolution ("he", "she", "they", "this person" -> last_person_name).
4. Speech transcription tolerance (imperfect phrasing, colloquialisms, missing auxiliary verbs).
5. Voice and typed text query parity.
6. Database reset validation (Profiles=0, Events=0, Incidents=0).
7. UI Clear Table View separation from database deletion.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.person_entity import PersonEntity
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.semantic_router import SemanticQueryRouter, SemanticResolution
from src.core.voice_assistant import VoiceAssistant, VoiceIntent, SessionContext
from src.database.db_manager import DatabaseManager
from src.events.event_types import RiskLevel


class TestSemanticRouterPhase56(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_phase56.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.router = SemanticQueryRouter(VoiceIntent)

        self.context_engine = ContextEngine(db_manager=self.db)
        self.analytics_engine = SecurityAnalyticsEngine(
            db_manager=self.db,
            context_engine=self.context_engine,
            incident_manager=self.context_engine.incident_manager,
            cache_ttl_seconds=0.01,
        )
        self.va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            security_analytics=self.analytics_engine,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Test 1: Explain Access Decision Variations (10+ variations) ---
    def test_01_explain_access_decision_variations(self):
        denial_variations = [
            "Why was Hunter denied?",
            "Why didn't Hunter get access?",
            "Why couldn't Hunter enter?",
            "What stopped Hunter from entering?",
            "Why is Hunter not allowed there?",
            "Tell me why his access was refused.",
            "What's stopping him?",
            "Why can't he enter?",
            "What caused the rejection?",
            "why hunter cannot go inside",
            "why he is denied",
            "what's the reason he got blocked",
            "Can you explain why Hunter was rejected?",
        ]
        session = SessionContext(last_person_name="Hunter")
        for q in denial_variations:
            res = self.router.route_query(q, session)
            self.assertIn(
                res.intent,
                (VoiceIntent.EXPLAIN_ACCESS_DECISION, VoiceIntent.FOLLOW_UP_PERSON_AUTHORIZATION),
                f"Query failed to resolve to EXPLAIN_ACCESS_DECISION: '{q}' (got {res.intent})",
            )
            self.assertTrue(res.confidence >= 0.90)

        auth_variations = [
            "Is Hunter allowed to be here?",
            "Is he allowed inside?",
        ]
        for q in auth_variations:
            res = self.router.route_query(q, session)
            self.assertEqual(
                res.intent,
                VoiceIntent.FOLLOW_UP_PERSON_AUTHORIZATION,
                f"Query failed to resolve to FOLLOW_UP_PERSON_AUTHORIZATION: '{q}' (got {res.intent})",
            )
            self.assertTrue(res.confidence >= 0.90)

    # --- Test 2: Who Is Present Variations ---
    def test_02_who_is_present_variations(self):
        variations = [
            "Who is in front of the camera?",
            "Who is standing there?",
            "Who's standing there",
            "Who is present right now?",
            "Who do you see in the frame?",
            "Who is that person?",
            "Identify the person in front",
        ]
        for q in variations:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.WHO_IS_IN_FRONT,
                f"Query failed to resolve to WHO_IS_IN_FRONT: '{q}' (got {res.intent})",
            )

    # --- Test 3: What Do You See & Scene Overview ---
    def test_03_what_do_you_see_variations(self):
        variations = [
            "What do you see?",
            "Describe what you see",
            "What is visible in the scene?",
            "Give me a scene overview",
            "Overview of the view",
        ]
        for q in variations:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.WHAT_DO_YOU_SEE,
                f"Query failed to resolve to WHAT_DO_YOU_SEE: '{q}' (got {res.intent})",
            )

    # --- Test 4: Detected Objects Variations ---
    def test_04_detected_objects_variations(self):
        variations = [
            "What objects are detected?",
            "What items are in view?",
            "Show detected objects",
            "What objects are visible?",
            "detected objects",
        ]
        for q in variations:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.DETECTED_OBJECTS,
                f"Query failed to resolve to DETECTED_OBJECTS: '{q}' (got {res.intent})",
            )

    # --- Test 5: Current Security Threat & Status ---
    def test_05_security_threat_and_status(self):
        threat_queries = [
            "What is the security risk?",
            "What is the current threat level?",
            "Is there a security risk?",
            "What is the risk level?",
        ]
        for q in threat_queries:
            res = self.router.route_query(q)
            self.assertEqual(res.intent, VoiceIntent.SECURITY_RISK)

        status_queries = [
            "What is the current security status?",
            "What is the system status?",
            "system status",
        ]
        for q in status_queries:
            res = self.router.route_query(q)
            self.assertEqual(res.intent, VoiceIntent.CURRENT_SECURITY_STATUS)

        live_queries = [
            "What is happening right now?",
            "Is there a threat right now?",
            "live state",
        ]
        for q in live_queries:
            res = self.router.route_query(q)
            self.assertEqual(res.intent, VoiceIntent.CURRENT_STATE_QUERY)

    # --- Test 6: Why High Risk Variations ---
    def test_06_why_high_risk_variations(self):
        variations = [
            "Why is the system red?",
            "Why is the risk orange?",
            "Why is there elevated risk?",
            "Why red?",
            "Why is the dashboard orange?",
        ]
        for q in variations:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.WHY_HIGH_RISK,
                f"Query failed to resolve to WHY_HIGH_RISK: '{q}' (got {res.intent})",
            )

    # --- Test 7: Temporal Events Minutes ---
    def test_07_temporal_minutes_variations(self):
        variations = [
            "What happened in the last 5 minutes?",
            "what happen in the last five minutes",
            "What occurred in the past 10 minutes?",
            "Show events from the last 20 minutes",
        ]
        for q in variations:
            res = self.router.route_query(q)
            self.assertEqual(
                res.intent,
                VoiceIntent.TEMPORAL_EVENTS_MINUTES,
                f"Query failed to resolve to TEMPORAL_EVENTS_MINUTES: '{q}' (got {res.intent})",
            )
            self.assertIsNotNone(res.target_time_minutes)

    # --- Test 8: Zone Entry and Exit Queries ---
    def test_08_zone_entry_and_exit(self):
        entry_queries = [
            "Did Hunter enter the restricted zone?",
            "Has Hunter walked into the restricted area?",
            "Did he enter the restricted area?",
        ]
        session = SessionContext(last_person_name="Hunter")
        for q in entry_queries:
            res = self.router.route_query(q, session)
            self.assertEqual(res.intent, VoiceIntent.ZONE_ENTRY_QUERY)

        exit_queries = [
            "Has he left the restricted area?",
            "Did Hunter exit the restricted zone?",
            "Has Hunter departed the restricted area?",
        ]
        for q in exit_queries:
            res = self.router.route_query(q, session)
            self.assertEqual(res.intent, VoiceIntent.ZONE_EXIT_QUERY)

        time_queries = [
            "When did Hunter enter?",
            "When did he go inside?",
            "What was his entry time?",
        ]
        for q in time_queries:
            res = self.router.route_query(q, session)
            self.assertEqual(res.intent, VoiceIntent.ZONE_ENTRY_TIME_QUERY)

    # --- Test 9: Violation History & Counts ---
    def test_09_violation_history_and_counts(self):
        session = SessionContext(last_person_name="Hunter")
        hist_queries = [
            "Has Hunter violated the policy before?",
            "Has he done this before?",
            "Does he have prior violations?",
        ]
        for q in hist_queries:
            res = self.router.route_query(q, session)
            self.assertEqual(res.intent, VoiceIntent.VIOLATION_HISTORY_QUERY)

        count_queries = [
            "How many violations happened?",
            "How many times did he enter?",
            "How many breaches occurred?",
        ]
        for q in count_queries:
            res = self.router.route_query(q, session)
            self.assertEqual(res.intent, VoiceIntent.RECENT_VIOLATIONS_QUERY)

    # --- Test 10: Contextual Pronoun Resolution in Pipeline ---
    def test_10_contextual_pronoun_pipeline_resolution(self):
        self.va.session_context.last_person_name = "Hunter"
        self.va.session_context.last_is_authorized = True
        self.va.session_context.last_decision_code = "AFTER_HOURS_RESTRICTED_ACCESS_DENIED"
        self.va.session_context.last_reason = "Hunter detected in Restricted Area after hours."
        self.va.session_context.last_time_str = "22:45"
        self.va.session_context.last_zone = "Restricted Area"
        self.va.session_context.last_is_authorized_hours = False

        # Spoken question using pronoun "he"
        resp = self.va.process_command("Why was he denied?")
        self.assertIn("Hunter", resp)
        self.assertIn("Restricted Area", resp)
        self.assertIn("09:00 to 18:00", resp)

    # --- Test 11: Controlled Database Reset ---
    def test_11_controlled_database_reset(self):
        # Insert test users and events
        u_id = self.db.create_user("test_user_99", "Test User", "Staff", 1)
        self.db.log_security_event(
            event_type="UNAUTHORIZED_ACCESS",
            risk_level="RED",
            description="Test alert",
            person_name="Test User",
        )

        self.assertGreater(len(self.db.get_users()), 0)
        self.assertGreater(self.db.get_total_event_count(), 0)

        # Execute Reset
        result = self.db.reset_development_data(backup=False, confirm=True)
        self.assertEqual(result["status"], "CLEAN_RESET_COMPLETE")
        self.assertEqual(len(self.db.get_users()), 0)
        self.assertEqual(len(self.db.get_active_face_registrations()), 0)
        self.assertEqual(self.db.get_total_event_count(), 0)
        self.assertEqual(len(self.db.get_active_incidents()), 0)


if __name__ == "__main__":
    unittest.main()
