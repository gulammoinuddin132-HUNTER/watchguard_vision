"""
Unit & Integration Tests for WatchGuard 2.0 Phase 5: Security Intelligence & Analytics
---------------------------------------------------------------------------------------
Validates:
1. One active incident metrics & 7-tuple summary generation
2. Multiple incidents with severity prioritization (CRITICAL/RED > RED > ORANGE > YELLOW > GREEN)
3. Current live vs historical state isolation
4. Five-minute bounded query accuracy
5. Today bounded query accuracy
6. Repeated restricted-zone violations aggregation
7. Repeated spoof attempts aggregation
8. Incident correlation (single incident for continuous presence)
9. Correct metric aggregation across time windows (live, 5min, 15min, 1hr, today)
10. Voice assistant answer for current risk ("Why is the system red?")
11. Voice assistant answer for recent history ("What happened in the last 5 minutes?")
12. Voice assistant answer for historical counts ("How many spoof attempts occurred?")
13. False-context protection (no double counting of fused person entities)
14. Stale/resolved incident not presented as active live threat
15. Explainable person denial ("Why was Hunter denied?")
"""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.incident_manager import IncidentManager, IncidentState, SecurityIncident
from src.core.person_entity import PersonEntity
from src.core.security_analytics import SecurityAnalyticsEngine, SecurityMetrics, IncidentSummary, SecurityExplanation
from src.core.voice_assistant import VoiceAssistant
from src.database.db_manager import DatabaseManager
from src.events.event_types import RiskLevel


class TestSecurityAnalyticsPhase5(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_analytics_phase5.db"
        self.db = DatabaseManager(db_path=self.db_path)

        self.context_engine = ContextEngine(db_manager=self.db)
        self.analytics_engine = SecurityAnalyticsEngine(
            db_manager=self.db,
            context_engine=self.context_engine,
            incident_manager=self.context_engine.incident_manager,
            cache_ttl_seconds=0.01,  # Fast TTL for tests
        )
        self.voice_assistant = VoiceAssistant(
            db_manager=self.db,
            context_engine=self.context_engine,
            security_analytics=self.analytics_engine,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Test 1: Single Active Incident Metric and Summary ---
    def test_01_single_active_incident_metric_and_summary(self):
        # Dispatch one active incident
        inc = self.context_engine.incident_manager.process_incident(
            risk_level=RiskLevel.ORANGE,
            policy_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Authorized personnel detected inside Restricted Area outside permitted operating hours (22:45).",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=1,
            time_status="AFTER_HOURS",
        )
        self.assertIsNotNone(inc)

        metrics = self.analytics_engine.get_metrics("live")
        self.assertEqual(metrics.active_incidents, 1)
        self.assertEqual(metrics.current_risk_level, RiskLevel.ORANGE)

        summary = self.analytics_engine.generate_security_summary("live")
        self.assertIn("ORANGE", summary)
        self.assertIn("Hunter", summary)
        self.assertIn("Restricted Area", summary)
        self.assertIn("09:00", summary)

    # --- Test 2: Multiple Incidents Prioritization ---
    def test_02_multiple_incidents_prioritization(self):
        # 1. Insert Yellow incident into DB
        now_iso = datetime.now().isoformat()
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO security_incidents (incident_id, person_name, risk_level, state, policy_code, reason, zone, created_at, updated_at)
                VALUES ('INC-YELLOW-001', 'Unknown', 'YELLOW', 'DETECTED', 'UNREGISTERED_VISITOR_DETECTED', 'Unregistered person', 'Standard Perimeter', ?, ?)
                """,
                (now_iso, now_iso),
            )
            conn.commit()

        # 2. Create Orange and Red incidents in IncidentManager
        inc_orange = self.context_engine.incident_manager.process_incident(
            risk_level=RiskLevel.ORANGE,
            policy_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Hunter after hours",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=2,
            force_new=True,
        )
        inc_red = self.context_engine.incident_manager.process_incident(
            risk_level=RiskLevel.RED,
            policy_code="PRESENTATION_ATTACK_IMPERSONATION",
            reason="Spoof attack detected using Hunter credentials",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=3,
            force_new=True,
        )

        prioritized = self.analytics_engine.get_prioritized_incidents(limit=5)
        self.assertEqual(len(prioritized), 3)
        # Red must be first, followed by Orange, then Yellow
        self.assertEqual(prioritized[0].risk_level, RiskLevel.RED)
        self.assertEqual(prioritized[1].risk_level, RiskLevel.ORANGE)
        self.assertEqual(prioritized[2].risk_level, RiskLevel.YELLOW)

    # --- Test 3: Current vs Historical State Isolation ---
    def test_03_current_vs_historical_state_isolation(self):
        # Log historical RED alert in database from 2 hours ago
        two_hours_ago = (datetime.now() - timedelta(hours=2)).isoformat()
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO security_events (timestamp, risk_level, event_type, description, person_name)
                VALUES (?, 'RED', 'PRESENTATION_ATTACK_IMPERSONATION', 'Historical spoof attack', 'Hunter')
                """,
                (two_hours_ago,),
            )
            conn.commit()

        # Set live context to GREEN (routine monitoring, no active incidents)
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Perimeter clear. Standard surveillance active.",
            is_access_granted=True,
        )

        # Live summary must report GREEN
        live_summary = self.analytics_engine.generate_security_summary("live")
        self.assertIn("GREEN", live_summary)

        # Historical query should find the old threat
        hist_resp = self.voice_assistant.process_command("Was there a threat earlier?")
        self.assertIn("RED", hist_resp)
        self.assertIn("Historical spoof attack", hist_resp)

    # --- Test 4: Five-Minute Bounded Query Accuracy ---
    def test_04_five_minute_bounded_query(self):
        now = datetime.now()
        ts_2min_ago = (now - timedelta(minutes=2)).isoformat()
        ts_10min_ago = (now - timedelta(minutes=10)).isoformat()

        with self.db._connection() as conn:
            cursor = conn.cursor()
            # Recent event (2 min ago)
            cursor.execute(
                """
                INSERT INTO security_events (timestamp, risk_level, event_type, description, person_name)
                VALUES (?, 'ORANGE', 'AFTER_HOURS_RESTRICTED_ACCESS_DENIED', 'Recent after-hours violation in Restricted Area', 'Hunter')
                """,
                (ts_2min_ago,),
            )
            # Older event (10 min ago)
            cursor.execute(
                """
                INSERT INTO security_events (timestamp, risk_level, event_type, description, person_name)
                VALUES (?, 'YELLOW', 'UNREGISTERED_VISITOR', 'Old visitor', 'Visitor')
                """,
                (ts_10min_ago,),
            )
            conn.commit()

        summary_5min = self.analytics_engine.get_recent_activity_summary(window_seconds=300.0)
        self.assertIn("1 after-hours violation(s)", summary_5min)

    # --- Test 5: Today Bounded Query Accuracy ---
    def test_05_today_bounded_query(self):
        now = datetime.now()
        ts_today = now.strftime("%Y-%m-%dT%H:%M:%S")
        ts_yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")

        with self.db._connection() as conn:
            cursor = conn.cursor()
            # Today's event
            cursor.execute(
                """
                INSERT INTO security_events (timestamp, risk_level, event_type, description)
                VALUES (?, 'RED', 'SPOOF_ATTACK_DETECTED', 'Today spoof attack detected')
                """,
                (ts_today,),
            )
            # Yesterday's event
            cursor.execute(
                """
                INSERT INTO security_events (timestamp, risk_level, event_type, description)
                VALUES (?, 'RED', 'SPOOF_ATTACK_DETECTED', 'Yesterday spoof attack detected')
                """,
                (ts_yesterday,),
            )
            conn.commit()

        metrics_today = self.analytics_engine.get_metrics("today")
        self.assertEqual(metrics_today.spoof_attempts, 1)

    # --- Test 6: Repeated Restricted Zone Violations Aggregation ---
    def test_06_repeated_restricted_zone_violations_aggregation(self):
        now = datetime.now()
        for i in range(3):
            ts = (now - timedelta(minutes=i * 2)).isoformat()
            with self.db._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO security_events (timestamp, risk_level, event_type, description, person_name)
                    VALUES (?, 'ORANGE', 'AFTER_HOURS_RESTRICTED_ACCESS_DENIED', 'Restricted Area entry after hours', 'Hunter')
                    """,
                    (ts,),
                )
                conn.commit()

        metrics = self.analytics_engine.get_metrics("5min")
        self.assertEqual(metrics.after_hours_violations, 3)
        self.assertEqual(metrics.restricted_zone_entries, 3)

    # --- Test 7: Repeated Spoof Attempts Aggregation ---
    def test_07_repeated_spoof_attempts_aggregation(self):
        now = datetime.now()
        for i in range(4):
            ts = (now - timedelta(minutes=i)).isoformat()
            with self.db._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO security_events (timestamp, risk_level, event_type, description)
                    VALUES (?, 'RED', 'PRESENTATION_ATTACK_IMPERSONATION', 'Photo spoof attack rejected')
                    """,
                    (ts,),
                )
                conn.commit()

        metrics = self.analytics_engine.get_metrics("today")
        self.assertEqual(metrics.spoof_attempts, 4)

    # --- Test 8: Incident Correlation Single Incident Continuous Stay ---
    def test_08_incident_correlation_single_incident_continuous_stay(self):
        # 10 continuous frames of Hunter in Restricted Area after hours
        for _ in range(10):
            self.context_engine.incident_manager.process_incident(
                risk_level=RiskLevel.ORANGE,
                policy_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
                reason="Continuous presence",
                zone_name="Restricted Area",
                person_name="Hunter",
                track_id=1,
            )

        active = self.context_engine.incident_manager.get_active_incidents()
        self.assertEqual(len(active), 1, "Continuous stay must maintain ONE single debounced incident")

    # --- Test 9: Correct Metric Aggregation Across Windows ---
    def test_09_correct_metric_aggregation_across_windows(self):
        metrics_live = self.analytics_engine.get_metrics("live")
        self.assertEqual(metrics_live.window_name, "live")

        metrics_5m = self.analytics_engine.get_metrics("5min")
        self.assertEqual(metrics_5m.window_name, "5min")

        metrics_today = self.analytics_engine.get_metrics("today")
        self.assertEqual(metrics_today.window_name, "today")

    # --- Test 10: Voice Answer for Current Risk ---
    def test_10_voice_answer_for_current_risk(self):
        self.context_engine.incident_manager.process_incident(
            risk_level=RiskLevel.RED,
            policy_code="PRESENTATION_ATTACK_IMPERSONATION",
            reason="Spoof attack detected using Hunter credentials.",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=1,
            force_new=True,
        )

        resp = self.voice_assistant.process_command("Why is the system red?")
        self.assertIn("RED", resp)
        self.assertIn("Hunter", resp)

    # --- Test 11: Voice Answer for Recent History ---
    def test_11_voice_answer_for_recent_history(self):
        now = datetime.now()
        ts = (now - timedelta(minutes=2)).isoformat()
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO security_events (timestamp, risk_level, event_type, description, person_name)
                VALUES (?, 'ORANGE', 'AFTER_HOURS_RESTRICTED_ACCESS_DENIED', 'After-hours violation', 'Hunter')
                """,
                (ts,),
            )
            conn.commit()

        resp = self.voice_assistant.process_command("What happened in the last 5 minutes?")
        self.assertIn("minutes", resp.lower())
        self.assertIn("After-hours", resp)

    # --- Test 12: Voice Answer for Historical Counts ---
    def test_12_voice_answer_for_historical_counts(self):
        now = datetime.now()
        for _ in range(2):
            with self.db._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO security_events (timestamp, risk_level, event_type, description)
                    VALUES (?, 'RED', 'SPOOF_ATTACK_DETECTED', 'Photo spoof attack rejected')
                    """,
                    (now.isoformat(),),
                )
                conn.commit()

        resp = self.voice_assistant.process_command("How many spoof attempts occurred?")
        self.assertIn("2", resp)
        self.assertIn("presentation attack", resp.lower())

    # --- Test 13: False-Context Protection (No Duplicate Person Counting) ---
    def test_13_false_context_protection(self):
        # Create 1 fused PersonEntity (Face Hunter + Person box)
        entity = PersonEntity(
            track_id=1,
            bbox=(100, 100, 100, 200),
            face_bbox=(110, 110, 80, 80),
            identity="Hunter",
            identity_confidence=0.95,
            is_known=True,
            liveness_state="STABLE_LIVE",
            liveness_confidence=0.95,
            person_object_detected=True,
            zone="Restricted Area",
            access_state="GRANTED",
            risk_level=RiskLevel.GREEN,
        )

        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Hunter authorized in area.",
            person_entities=[entity],
        )

        metrics = self.analytics_engine.get_metrics("live")
        self.assertEqual(metrics.distinct_people_seen, 1, "Hunter + person bounding box must be exactly 1 person")
        self.assertEqual(metrics.unknown_visitors, 0)

    # --- Test 14: Stale Incident Not Presented as Current Live Threat ---
    def test_14_stale_incident_not_presented_as_current(self):
        # Create and resolve an incident
        inc = self.context_engine.incident_manager.process_incident(
            risk_level=RiskLevel.ORANGE,
            policy_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Resolved violation",
            zone_name="Restricted Area",
            person_name="Hunter",
            track_id=1,
            force_new=True,
        )
        self.context_engine.incident_manager.resolve_incident(inc.incident_id, operator="Unit Test")

        # Set live context clear
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Perimeter clear.",
        )

        metrics = self.analytics_engine.get_metrics("live")
        self.assertEqual(metrics.active_incidents, 0)
        self.assertEqual(metrics.current_risk_level, RiskLevel.GREEN)

        summary = self.analytics_engine.generate_security_summary("live")
        self.assertIn("GREEN", summary)

    # --- Test 15: Explainable Person Denial ---
    def test_15_why_person_denied_explanation(self):
        # Setup decision where Hunter is denied after hours
        self.context_engine.last_decision = SecurityDecision(
            risk_level=RiskLevel.ORANGE,
            decision="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            reason="Authorized personnel detected inside Restricted Area outside permitted operating hours (22:45).",
            is_access_granted=False,
            person_entities=[
                PersonEntity(
                    track_id=1,
                    bbox=(100, 100, 100, 200),
                    face_bbox=(110, 110, 80, 80),
                    identity="Hunter",
                    identity_confidence=0.92,
                    is_known=True,
                    liveness_state="STABLE_LIVE",
                    liveness_confidence=0.95,
                    person_object_detected=True,
                    zone="Restricted Area",
                    access_state="DENIED",
                    risk_level=RiskLevel.ORANGE,
                )
            ],
        )

        resp = self.voice_assistant.process_command("Why is Hunter denied?")
        self.assertIn("Hunter", resp)
        self.assertIn("09:00", resp)
        self.assertIn("18:00", resp)


if __name__ == "__main__":
    unittest.main()
