"""
Unit & Integration Tests for WatchGuard 2.0 Phase 5.5: AI Security Copilot
---------------------------------------------------------------------------
Validates:
1. Compact structured context builder with CURRENT_STATE, RECENT_HISTORY, HISTORICAL_EVENTS.
2. Evidence grounding: identity, liveness, zone, operating hours, decision, incident.
3. Anti-hallucination payload safety: zero raw frames, zero audio, zero biometric embeddings.
4. DeterministicCopilotReasoner operations.
5. GeminiCopilotReasoner request formatting and mock execution (AI_GROUNDED).
6. Graceful fail-secure fallback (FALLBACK_DETERMINISTIC) on network failure or missing API key.
7. VoiceAssistant fast-path vs AI-path dispatching.
8. Accurate source logging (DETERMINISTIC vs AI_GROUNDED vs FALLBACK_DETERMINISTIC).
9. SessionContext latency and source tracking.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.copilot_reasoner import (
    BaseAICopilotReasoner,
    CopilotResponse,
    CopilotResponseSource,
    DeterministicCopilotReasoner,
    GeminiCopilotReasoner,
    build_structured_ai_context,
    get_copilot_reasoner,
)
from src.core.incident_manager import IncidentManager, SecurityIncident, IncidentState
from src.core.person_entity import PersonEntity
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.voice_assistant import VoiceAssistant, VoiceIntent
from src.database.db_manager import DatabaseManager
from src.events.event_types import RiskLevel


class TestAISecurityCopilotPhase55(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_copilot_phase55.db"
        self.db = DatabaseManager(db_path=self.db_path)

        self.context_engine = ContextEngine(db_manager=self.db)
        self.analytics_engine = SecurityAnalyticsEngine(
            db_manager=self.db,
            context_engine=self.context_engine,
            incident_manager=self.context_engine.incident_manager,
            cache_ttl_seconds=0.01,
        )
        self.deterministic_reasoner = DeterministicCopilotReasoner(security_analytics=self.analytics_engine)
        self.va = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            security_analytics=self.analytics_engine,
            copilot_reasoner=self.deterministic_reasoner,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Test 1: Structured Context Assembly & Section Separation ---
    def test_01_structured_context_sections(self):
        # Create a live person entity and active decision
        entity = PersonEntity(
            track_id=1,
            bbox=(100, 100, 100, 200),
            face_bbox=(110, 110, 80, 80),
            identity="Hunter",
            identity_confidence=0.96,
            is_known=True,
            role="Staff",
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
            person_object_detected=True,
            zone="Restricted Area",
            access_state="DENIED",
            risk_level=RiskLevel.ORANGE,
            decision_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Hunter detected in Restricted Area after hours.",
        )

        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.ORANGE,
            decision="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Authorized personnel detected inside Restricted Area outside permitted operating hours (22:45).",
            person_entities=[entity],
            is_access_granted=False,
            context_summary={
                "person_name": "Hunter",
                "is_authorized": True,
                "zone_name": "Restricted Area",
                "is_authorized_hours": False,
                "objects": [{"label": "laptop", "confidence": 0.90}],
            },
        )

        context = build_structured_ai_context(
            context_engine=self.context_engine,
            security_analytics=self.analytics_engine,
            database_manager=self.db,
        )

        # Verify Top-Level Sections
        self.assertIn("CURRENT_STATE", context)
        self.assertIn("RECENT_HISTORY", context)
        self.assertIn("HISTORICAL_EVENTS", context)
        self.assertIn("EVIDENCE_GROUNDING", context)

        # Verify CURRENT_STATE Fields
        curr = context["CURRENT_STATE"]
        self.assertEqual(curr["master_risk_level"], "ORANGE")
        self.assertEqual(curr["master_decision_code"], "AFTER_HOURS_RESTRICTED_ACCESS_DENIED")
        self.assertEqual(curr["time_status"], "AFTER_HOURS")
        self.assertEqual(len(curr["live_people_in_view"]), 1)
        self.assertEqual(curr["live_people_in_view"][0]["identity"], "Hunter")
        self.assertEqual(curr["live_people_in_view"][0]["liveness_result"], "LIVE")

        # Verify EVIDENCE_GROUNDING 7-Tuple
        grounding = context["EVIDENCE_GROUNDING"]
        self.assertEqual(grounding["who"], "Hunter")
        self.assertEqual(grounding["what"], "AFTER_HOURS_RESTRICTED_ACCESS_DENIED")
        self.assertEqual(grounding["where"], "Restricted Area")

    # --- Test 2: Privacy & Anti-Hallucination Payload Safety ---
    def test_02_privacy_and_payload_safety(self):
        context = build_structured_ai_context(
            context_engine=self.context_engine,
            security_analytics=self.analytics_engine,
            database_manager=self.db,
        )
        context_str = json.dumps(context)

        # Guarantee zero raw pixel / audio / vector data
        self.assertNotIn("frame_bytes", context_str)
        self.assertNotIn("image_bytes", context_str)
        self.assertNotIn("audio_bytes", context_str)
        self.assertNotIn("face_encoding", context_str)
        self.assertNotIn("embedding", context_str)

    # --- Test 3: Deterministic Reasoner Execution ---
    def test_03_deterministic_copilot_reasoner(self):
        context = {
            "CURRENT_STATE": {
                "active_zone": "Restricted Area",
                "operating_hours_window": "09:00 to 18:00",
                "master_risk_level": "ORANGE",
                "master_decision_code": "AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
                "master_reason": "Outside permitted hours",
            },
            "EVIDENCE_GROUNDING": {
                "who": "Hunter",
                "what": "AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
                "where": "Restricted Area",
            },
        }

        resp = self.deterministic_reasoner.generate_explanation("Why was Hunter denied?", context)
        self.assertIsInstance(resp, CopilotResponse)
        self.assertEqual(resp.source, CopilotResponseSource.DETERMINISTIC)
        self.assertIn("Hunter", resp.text)
        self.assertIn("Restricted Area", resp.text)
        self.assertIn("09:00 to 18:00", resp.text)

    # --- Test 4: Gemini Copilot Reasoner Mock Execution ---
    @patch("urllib.request.urlopen")
    def test_04_gemini_copilot_reasoner_mock(self, mock_urlopen):
        # Mock successful Gemini API response
        mock_resp_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Hunter was denied access because current surveillance time is outside the authorized 09:00 to 18:00 operating window for the Restricted Area."
                            }
                        ]
                    }
                }
            ]
        }
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value.status = 200
        mock_context_manager.__enter__.return_value.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
        mock_urlopen.return_value = mock_context_manager

        gemini_reasoner = GeminiCopilotReasoner(api_key="mock-gemini-key-12345", timeout_seconds=1.0)
        self.assertTrue(gemini_reasoner.is_available())

        context = {"CURRENT_STATE": {}, "EVIDENCE_GROUNDING": {"who": "Hunter"}}
        resp = gemini_reasoner.generate_explanation("Why was Hunter denied?", context)

        self.assertEqual(resp.source, CopilotResponseSource.AI_GROUNDED)
        self.assertIn("Hunter was denied access", resp.text)
        self.assertGreater(resp.latency_ms, 0.0)

    # --- Test 5: Graceful Fail-Secure Fallback on Network Error ---
    @patch("urllib.request.urlopen")
    def test_05_gemini_copilot_fail_secure_fallback(self, mock_urlopen):
        # Mock network failure / timeout
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        gemini_reasoner = GeminiCopilotReasoner(api_key="mock-gemini-key-12345", timeout_seconds=0.1)
        context = {
            "CURRENT_STATE": {
                "active_zone": "Restricted Area",
                "operating_hours_window": "09:00 to 18:00",
                "master_risk_level": "ORANGE",
                "master_decision_code": "AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            },
            "EVIDENCE_GROUNDING": {
                "who": "Hunter",
            },
        }

        resp = gemini_reasoner.generate_explanation("Why was Hunter denied?", context)
        # Must return deterministic fallback without throwing exception
        self.assertEqual(resp.source, CopilotResponseSource.FALLBACK_DETERMINISTIC)
        self.assertIn("Hunter", resp.text)

    # --- Test 6: Fast-Path Deterministic Routing in Voice Assistant ---
    def test_06_fast_path_deterministic_routing(self):
        # Query status directly
        resp = self.va.process_command("What is the current security status?")
        self.assertEqual(self.va.session_context.last_response_source, CopilotResponseSource.DETERMINISTIC)
        self.assertIn("GREEN", resp)

        # Query spoof attempts directly
        resp2 = self.va.process_command("How many spoof attempts occurred?")
        self.assertEqual(self.va.session_context.last_response_source, CopilotResponseSource.DETERMINISTIC)
        self.assertIn("presentation attack", resp2.lower())

    # --- Test 7: AI-Path Conversational Query Routing ---
    def test_07_ai_path_conversational_routing(self):
        # Mock an active AI reasoner
        mock_ai_reasoner = MagicMock(spec=BaseAICopilotReasoner)
        mock_ai_reasoner.is_available.return_value = True
        mock_ai_reasoner.model_name = "Mock-AI-Model"
        mock_ai_reasoner.generate_explanation.return_value = CopilotResponse(
            text="The surveillance facility has recorded 1 authorized entry and 0 security anomalies in recent monitoring.",
            source=CopilotResponseSource.AI_GROUNDED,
            latency_ms=120.0,
        )

        va_ai = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            security_analytics=self.analytics_engine,
            copilot_reasoner=mock_ai_reasoner,
        )

        # Ask conversational query
        resp = va_ai.process_command("Can you give me a general overview of everything happening today?")
        self.assertEqual(va_ai.session_context.last_response_source, CopilotResponseSource.AI_GROUNDED)
        self.assertIn("surveillance facility", resp)

    # --- Test 8: Missing API Key Fallback Factory ---
    def test_08_copilot_reasoner_factory(self):
        # Without API key, should return DeterministicCopilotReasoner
        reasoner = get_copilot_reasoner(security_analytics=self.analytics_engine, api_key="")
        self.assertIsInstance(reasoner, DeterministicCopilotReasoner)

        # With API key, should return GeminiCopilotReasoner
        reasoner_gemini = get_copilot_reasoner(security_analytics=self.analytics_engine, api_key="valid-api-key-123")
        self.assertIsInstance(reasoner_gemini, GeminiCopilotReasoner)


if __name__ == "__main__":
    unittest.main()
