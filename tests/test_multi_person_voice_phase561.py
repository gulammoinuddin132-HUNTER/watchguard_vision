"""
Unit & Regression Tests for WatchGuard 2.0 Phase 5.6.1: Multi-Person Voice Context Correction
---------------------------------------------------------------------------------------------
Validates:
1. One authorized person.
2. One unknown person.
3. Authorized + unknown simultaneously.
4. Two authorized people.
5. Two unknown people.
6. Authorized + spoofed face entity.
7. Authorized + unknown + cell phone object.
8. Three distinct people (1 authorized + 2 unknown).

Verifies across all cases:
- people_count
- individual identities preserved independently (no collapse / suppression)
- Voice responses for "Who is in front of the camera?", "What do you see?",
  "How many people are there?", and "What objects are detected?"
- AI-grounded & deterministic reasoner response parity.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.copilot_reasoner import (
    build_structured_ai_context,
    DeterministicCopilotReasoner,
    CopilotResponseSource,
)
from src.core.person_entity import PersonEntity
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.voice_assistant import VoiceAssistant, VoiceIntent
from src.database.db_manager import DatabaseManager
from src.events.event_types import RiskLevel


class TestMultiPersonVoicePhase561(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_phase561.db"
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
        self.person_mock = MagicMock()
        self.person_mock.is_ready = True
        self.object_mock = MagicMock()
        self.object_mock.is_ready = True

        self.va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            security_analytics=self.analytics,
            camera_manager=self.camera_mock,
            person_engine=self.person_mock,
            object_engine=self.object_mock,
        )
        self.reasoner = DeterministicCopilotReasoner(security_analytics=self.analytics)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _set_scene(self, entities, non_person_objects=None):
        non_person_objects = non_person_objects or []
        obj_summary = ", ".join([o["label"] for o in non_person_objects]) if non_person_objects else "None"

        has_auth = any(e.is_known and not e.is_spoof for e in entities)
        has_unk = any(not e.is_known and not e.is_spoof for e in entities)
        has_spoof = any(e.is_spoof for e in entities)

        primary_name = entities[0].identity if entities else "None"

        decision = SecurityDecision(
            risk_level=RiskLevel.RED if has_spoof else (RiskLevel.YELLOW if has_unk else RiskLevel.GREEN),
            decision="PRESENTATION_ATTACK_IMPERSONATION" if has_spoof else ("UNAUTHORIZED_ACCESS" if has_unk else "AUTHORIZED_PERSON_NORMAL_PERIMETER"),
            reason="Scene observation",
            person_entities=entities,
            is_access_granted=has_auth and not has_spoof,
            context_summary={
                "person_detected": len(entities) > 0,
                "person_name": primary_name,
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

    # --- Test Case 1: One Authorized Person ---
    def test_case_01_one_authorized_person(self):
        e1 = PersonEntity(
            track_id=1,
            bbox=(100, 100, 100, 200),
            identity="Gulam Moinuddin",
            identity_confidence=0.96,
            is_known=True,
            role="Administrator",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
        )
        self._set_scene([e1])

        self.assertEqual(self.va.session_context.last_people_count, 1)

        resp_who = self.va.process_command("Who is in front of the camera?")
        self.assertIn("Gulam Moinuddin", resp_who)
        self.assertIn("Authorized personnel", resp_who)

        resp_see = self.va.process_command("What do you see?")
        self.assertIn("Gulam Moinuddin", resp_see)

        resp_count = self.va.process_command("How many people are there?")
        self.assertIn("one person", resp_count.lower())
        self.assertIn("Gulam Moinuddin", resp_count)

        resp_obj = self.va.process_command("What objects are detected?")
        self.assertIn("No non-person objects", resp_obj)
        self.assertIn("Gulam Moinuddin", resp_obj)

    # --- Test Case 2: One Unknown Person ---
    def test_case_02_one_unknown_person(self):
        e1 = PersonEntity(
            track_id=2,
            bbox=(100, 100, 100, 200),
            identity="Unknown Person",
            identity_confidence=0.0,
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.92,
        )
        self._set_scene([e1])

        self.assertEqual(self.va.session_context.last_people_count, 1)

        resp_who = self.va.process_command("Who is in front of the camera?")
        self.assertIn("unregistered person", resp_who.lower())

        resp_see = self.va.process_command("What do you see?")
        self.assertIn("unregistered person", resp_see.lower())

        resp_count = self.va.process_command("How many people are there?")
        self.assertIn("one person", resp_count.lower())

        resp_obj = self.va.process_command("What objects are detected?")
        self.assertIn("No non-person objects", resp_obj)
        self.assertIn("unregistered person", resp_obj.lower())

    # --- Test Case 3: Authorized + Unknown Simultaneously (The Core Bug Scenario) ---
    def test_case_03_authorized_plus_unknown_simultaneously(self):
        e1 = PersonEntity(
            track_id=11,
            bbox=(50, 100, 100, 200),
            identity="Gulam Moinuddin",
            identity_confidence=0.96,
            is_known=True,
            role="Administrator",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
        )
        e2 = PersonEntity(
            track_id=26,
            bbox=(250, 100, 100, 200),
            identity="Unknown Person",
            identity_confidence=0.0,
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.91,
        )
        self._set_scene([e1, e2])

        self.assertEqual(self.va.session_context.last_people_count, 2)

        # "Who is in front of the camera?" MUST report BOTH Gulam and unregistered person
        resp_who = self.va.process_command("Who is in front of the camera?")
        self.assertIn("two people", resp_who.lower())
        self.assertIn("Gulam Moinuddin", resp_who)
        self.assertIn("unregistered person", resp_who.lower())

        # "What do you see?" MUST report BOTH
        resp_see = self.va.process_command("What do you see?")
        self.assertIn("two people", resp_see.lower())
        self.assertIn("Gulam Moinuddin", resp_see)
        self.assertIn("unregistered", resp_see.lower())

        # "How many people are there?"
        resp_count = self.va.process_command("How many people are there?")
        self.assertIn("two people", resp_count.lower())
        self.assertIn("Gulam Moinuddin", resp_count)

        # "What objects are detected?" MUST NOT say "one person: Gulam Moinuddin"
        resp_obj = self.va.process_command("What objects are detected?")
        self.assertIn("No non-person objects", resp_obj)
        self.assertIn("two people", resp_obj.lower())
        self.assertIn("Gulam Moinuddin", resp_obj)
        self.assertIn("unregistered person", resp_obj.lower())

        # AI Structured Context Parity
        ctx = build_structured_ai_context(self.context_engine, self.analytics, self.db)
        curr = ctx["CURRENT_STATE"]
        self.assertEqual(curr["people_count"], 2)
        self.assertEqual(len(curr["people"]), 2)

        ai_resp = self.reasoner.generate_explanation("What do you see?", ctx)
        self.assertIn("2 people", ai_resp.text)
        self.assertIn("Gulam Moinuddin", ai_resp.text)
        self.assertIn("unregistered", ai_resp.text.lower())

    # --- Test Case 4: Two Authorized People ---
    def test_case_04_two_authorized_people(self):
        e1 = PersonEntity(
            track_id=1,
            bbox=(50, 100, 100, 200),
            identity="Gulam Moinuddin",
            identity_confidence=0.96,
            is_known=True,
            role="Administrator",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
        )
        e2 = PersonEntity(
            track_id=2,
            bbox=(250, 100, 100, 200),
            identity="Alice Smith",
            identity_confidence=0.94,
            is_known=True,
            role="Staff",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.93,
        )
        self._set_scene([e1, e2])

        self.assertEqual(self.va.session_context.last_people_count, 2)

        resp_who = self.va.process_command("Who is in front of the camera?")
        self.assertIn("Gulam Moinuddin", resp_who)
        self.assertIn("Alice Smith", resp_who)
        self.assertIn("authorized", resp_who.lower())

        resp_see = self.va.process_command("What do you see?")
        self.assertIn("Gulam Moinuddin", resp_see)
        self.assertIn("Alice Smith", resp_see)

    # --- Test Case 5: Two Unknown People ---
    def test_case_05_two_unknown_people(self):
        e1 = PersonEntity(
            track_id=1,
            bbox=(50, 100, 100, 200),
            identity="Unknown Person",
            identity_confidence=0.0,
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.91,
        )
        e2 = PersonEntity(
            track_id=2,
            bbox=(250, 100, 100, 200),
            identity="Unknown Person",
            identity_confidence=0.0,
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.89,
        )
        self._set_scene([e1, e2])

        self.assertEqual(self.va.session_context.last_people_count, 2)

        resp_who = self.va.process_command("Who is in front of the camera?")
        self.assertIn("two", resp_who.lower())
        self.assertIn("unregistered", resp_who.lower())

        resp_see = self.va.process_command("What do you see?")
        self.assertIn("two unregistered people", resp_see.lower())

    # --- Test Case 6: Authorized + Spoofed Face Entity ---
    def test_case_06_authorized_plus_spoofed_face(self):
        e1 = PersonEntity(
            track_id=1,
            bbox=(50, 100, 100, 200),
            identity="Gulam Moinuddin",
            identity_confidence=0.96,
            is_known=True,
            role="Administrator",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
        )
        e2 = PersonEntity(
            track_id=2,
            bbox=(250, 100, 100, 200),
            identity="Alice Smith",
            identity_confidence=0.92,
            is_known=True,
            role="Staff",
            liveness_state="STABLE_SPOOF",
            liveness_confidence=0.88,
            is_spoof=True,
            attack_type="PHOTO_REPLAY_ATTACK",
        )
        self._set_scene([e1, e2])

        self.assertEqual(self.va.session_context.last_people_count, 2)

        resp_who = self.va.process_command("Who is in front of the camera?")
        self.assertIn("Gulam Moinuddin", resp_who)
        self.assertIn("spoof presentation", resp_who.lower())

    # --- Test Case 7: Authorized + Unknown + Cell Phone Object ---
    def test_case_07_authorized_plus_unknown_plus_phone(self):
        e1 = PersonEntity(
            track_id=1,
            bbox=(50, 100, 100, 200),
            identity="Gulam Moinuddin",
            identity_confidence=0.96,
            is_known=True,
            role="Administrator",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
        )
        e2 = PersonEntity(
            track_id=2,
            bbox=(250, 100, 100, 200),
            identity="Unknown Person",
            identity_confidence=0.0,
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.90,
        )
        obj1 = {"label": "cell phone", "confidence": 0.91}
        self._set_scene([e1, e2], [obj1])

        resp_obj = self.va.process_command("What objects are detected?")
        self.assertIn("cell phone", resp_obj.lower())
        self.assertIn("two people", resp_obj.lower())
        self.assertIn("Gulam Moinuddin", resp_obj)
        self.assertIn("unregistered person", resp_obj.lower())

        resp_see = self.va.process_command("What do you see?")
        self.assertIn("two people", resp_see.lower())
        self.assertIn("Gulam Moinuddin", resp_see)
        self.assertIn("cell phone", resp_see.lower())

    # --- Test Case 8: Three Distinct People ---
    def test_case_08_three_distinct_people(self):
        e1 = PersonEntity(
            track_id=1,
            bbox=(50, 100, 80, 160),
            identity="Gulam Moinuddin",
            identity_confidence=0.96,
            is_known=True,
            role="Administrator",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
        )
        e2 = PersonEntity(
            track_id=2,
            bbox=(180, 100, 80, 160),
            identity="Unknown Person",
            identity_confidence=0.0,
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.90,
        )
        e3 = PersonEntity(
            track_id=3,
            bbox=(300, 100, 80, 160),
            identity="Unknown Person",
            identity_confidence=0.0,
            is_known=False,
            role="Visitor",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.88,
        )
        self._set_scene([e1, e2, e3])

        self.assertEqual(self.va.session_context.last_people_count, 3)

        resp_who = self.va.process_command("Who is in front of the camera?")
        self.assertIn("3 people", resp_who)
        self.assertIn("Gulam Moinuddin", resp_who)
        self.assertIn("2 unregistered people", resp_who)

        resp_count = self.va.process_command("How many people are there?")
        self.assertIn("3 people", resp_count)


if __name__ == "__main__":
    unittest.main()
