"""
Tests for Zone-Specific Time-Based Access Control in WatchGuard Vision.
Verifies the exact policy matrix:
- Authorized hours: 09:00–18:00
- Operating hours restriction applies ONLY to the RESTRICTED AREA.
- Authorized personnel in Normal/Protected areas have 24/7 access (GRANTED, GREEN).
- Authorized personnel in Restricted Area after hours are DENIED (ORANGE, AFTER_HOURS_RESTRICTED_ACCESS_DENIED).
- Presentation attack is DENIED (RED, PRESENTATION_ATTACK_IMPERSONATION) regardless of time/zone.
- Unknown in Restricted Area is DENIED (RED, UNAUTHORIZED_RESTRICTED_ZONE_ACCESS).
- Standard Monitoring uses real local time, while simulations use overrides without bleeding state.
- Voice assistant correctly answers time/zone questions.
"""

import unittest
from datetime import datetime, time as dtime
from unittest.mock import MagicMock

import numpy as np

from src.core.context_engine import ContextEngine, SecurityDecision
from src.events.event_types import RiskLevel, EventType
from src.core.voice_assistant import VoiceAssistant, VoiceIntent


class TestZoneSpecificTimeAccessControl(unittest.TestCase):

    def setUp(self):
        self.engine = ContextEngine()

    def test_hunter_protected_area_daytime_granted(self):
        """1. Hunter + Protected Area + daytime (14:00) -> ACCESS GRANTED, GREEN."""
        self.engine.set_time_override(dtime(14, 0))
        self.engine.set_zone_override("PROTECTED")

        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "people": [
                {
                    "name": "Hunter",
                    "is_known": True,
                    "is_spoof": False,
                    "confidence": 0.92,
                    "similarity": 0.88,
                    "role": "Security Engineer",
                    "bbox": (100, 100, 150, 150),
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.GREEN)
        self.assertEqual(decision.decision, "NORMAL_ACTIVITY")
        self.assertTrue(decision.is_access_granted)
        self.assertIn("Hunter", decision.reason)

    def test_hunter_protected_area_after_hours_granted(self):
        """2. Hunter + Protected Area + after-hours (22:49) -> ACCESS GRANTED, GREEN."""
        self.engine.set_time_override(dtime(22, 49))
        self.engine.set_zone_override("PROTECTED")

        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "people": [
                {
                    "name": "Hunter",
                    "is_known": True,
                    "is_spoof": False,
                    "confidence": 0.92,
                    "similarity": 0.88,
                    "role": "Security Engineer",
                    "bbox": (100, 100, 150, 150),
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.GREEN)
        self.assertEqual(decision.decision, "NORMAL_ACTIVITY")
        self.assertTrue(decision.is_access_granted)
        self.assertIn("does not apply outside the Restricted Area", decision.reason)

    def test_hunter_restricted_area_daytime_granted(self):
        """3. Hunter + Restricted Area + daytime (11:30) -> ACCESS GRANTED, GREEN."""
        self.engine.set_time_override(dtime(11, 30))
        self.engine.set_zone_override("RESTRICTED")

        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "people": [
                {
                    "name": "Hunter",
                    "is_known": True,
                    "is_spoof": False,
                    "confidence": 0.92,
                    "similarity": 0.88,
                    "role": "Security Engineer",
                    "bbox": (100, 100, 150, 150),
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.GREEN)
        self.assertEqual(decision.decision, "AUTHORIZED_RESTRICTED_ZONE_ACCESS")
        self.assertTrue(decision.is_access_granted)
        self.assertIn("Hunter", decision.reason)
        self.assertIn("during permitted operating hours", decision.reason)

    def test_hunter_restricted_area_after_hours_denied(self):
        """4. Hunter + Restricted Area + after-hours (22:49) -> ACCESS DENIED, ORANGE."""
        self.engine.set_time_override(dtime(22, 49))
        self.engine.set_zone_override("RESTRICTED")

        person_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "has_spoof": False,
            "people": [
                {
                    "name": "Hunter",
                    "is_known": True,
                    "is_spoof": False,
                    "confidence": 0.92,
                    "similarity": 0.88,
                    "role": "Security Engineer",
                    "bbox": (100, 100, 150, 150),
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.ORANGE)
        self.assertEqual(decision.decision, "AFTER_HOURS_RESTRICTED_ACCESS_DENIED")
        self.assertFalse(decision.is_access_granted)
        self.assertEqual(decision.context_summary["person_name"], "Hunter")
        self.assertTrue(decision.context_summary["is_authorized"])
        self.assertIn("outside permitted operating hours", decision.reason)

    def test_unknown_restricted_area_denied_red(self):
        """5. Unknown + Restricted Area -> ACCESS DENIED, RED."""
        self.engine.set_time_override(dtime(14, 0))
        self.engine.set_zone_override("RESTRICTED")

        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": True,
            "has_spoof": False,
            "people": [
                {
                    "name": "Unknown Person",
                    "is_known": False,
                    "is_spoof": False,
                    "confidence": 0.85,
                    "similarity": 0.35,
                    "role": "Visitor",
                    "bbox": (100, 100, 150, 150),
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.RED)
        self.assertEqual(decision.decision, "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS")
        self.assertFalse(decision.is_access_granted)

    def test_hunter_photo_spoof_restricted_area_denied_red(self):
        """6. Hunter photo spoof + Restricted Area -> SPOOF REJECTED, RED."""
        self.engine.set_time_override(dtime(14, 0))
        self.engine.set_zone_override("RESTRICTED")

        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_unknown": False,
            "has_spoof": True,
            "people": [
                {
                    "name": "Hunter",
                    "is_known": True,
                    "is_spoof": True,
                    "attack_type": "PHOTO_REPLAY_ATTACK",
                    "confidence": 0.90,
                    "similarity": 0.86,
                    "role": "Security Engineer",
                    "bbox": (100, 100, 150, 150),
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = self.engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.RED)
        self.assertEqual(decision.decision, "PRESENTATION_ATTACK_IMPERSONATION")
        self.assertFalse(decision.is_access_granted)
        self.assertIn("Hunter", decision.reason)

    def test_standard_monitoring_uses_actual_local_time(self):
        """7 & 8. Standard Monitoring uses actual system clock and overrides clear cleanly."""
        self.engine.set_time_override(None)
        self.engine.set_zone_override(None)

        now = datetime.now()
        is_auth, time_str = self.engine.is_authorized_time()
        self.assertEqual(time_str, now.strftime("%H:%M"))

        expected_is_auth = dtime(9, 0) <= now.time() <= dtime(18, 0)
        self.assertEqual(is_auth, expected_is_auth)

        # Apply override
        self.engine.set_time_override(dtime(23, 0))
        is_auth_over, time_str_over = self.engine.is_authorized_time()
        self.assertFalse(is_auth_over)
        self.assertEqual(time_str_over, "23:00")

        # Clear override
        self.engine.set_time_override(None)
        is_auth_cleared, _ = self.engine.is_authorized_time()
        self.assertEqual(is_auth_cleared, expected_is_auth)


class TestVoiceAssistantZoneTimeQueries(unittest.TestCase):

    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_engine.hours_start = "09:00"
        self.mock_engine.hours_end = "18:00"
        self.mock_db = MagicMock()
        self.mock_db.get_total_event_count.return_value = 0
        self.mock_db.get_events_count_by_date.return_value = 0
        self.mock_db.get_repeated_unauthorized_events.return_value = []
        self.mock_db.get_events_in_timespan.return_value = []
        self.mock_db.get_latest_alert_by_risk.return_value = None
        self.mock_db.get_most_serious_event.return_value = None

        self.va = VoiceAssistant(db_manager=self.mock_db, context_engine=self.mock_engine)

    def test_query_authorized_hours(self):
        """Voice Assistant answers 'What are the authorized hours?' correctly."""
        res = self.va.process_command("What are the authorized hours?")
        self.assertIn("09:00 to 18:00", res)
        self.assertIn("Restricted Area", res)

    def test_query_hunter_after_hours_restricted_denied(self):
        """Voice Assistant explains why Hunter was denied in Restricted Area after hours."""
        mock_decision = SecurityDecision(
            risk_level=RiskLevel.ORANGE,
            decision="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Authorized personnel detected inside Restricted Area outside permitted operating hours (22:49).",
            matched_rules=["RULE_AFTER_HOURS_RESTRICTED_ZONE_DENIED"],
            evidence_required=True,
            recommended_response="DENY ACCESS.",
            is_access_granted=False,
            context_summary={
                "person_detected": True,
                "person_name": "Hunter",
                "is_authorized": True,
                "is_unknown": False,
                "zone_name": "Restricted Area",
                "time_str": "22:49",
                "is_authorized_hours": False,
            },
        )
        self.mock_engine.last_decision = mock_decision

        res = self.va.process_command("Is Hunter allowed here?")
        self.assertIn("No", res)
        self.assertIn("Hunter", res)
        self.assertIn("22:49", res)
        self.assertIn("Restricted Area", res)

        res_why = self.va.process_command("Why was Hunter denied?")
        self.assertIn("outside the permitted", res_why)
        self.assertIn("Restricted Area", res_why)

    def test_query_hunter_protected_area_granted(self):
        """Voice Assistant answers that Hunter is authorized in Protected Area."""
        mock_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Authorized personnel verified in Protected Asset Area.",
            matched_rules=["RULE_A_NORMAL_ACTIVITY"],
            evidence_required=False,
            recommended_response="Continue monitoring.",
            is_access_granted=True,
            context_summary={
                "person_detected": True,
                "person_name": "Hunter",
                "is_authorized": True,
                "is_unknown": False,
                "zone_name": "Protected Asset Area",
                "time_str": "22:49",
                "is_authorized_hours": False,
            },
        )
        self.mock_engine.last_decision = mock_decision

        res = self.va.process_command("Is Hunter allowed here?")
        self.assertIn("Yes", res)
        self.assertIn("Hunter", res)
        self.assertIn("Protected Asset Area", res)


if __name__ == "__main__":
    unittest.main()
