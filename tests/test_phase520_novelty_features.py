"""
WatchGuard Vision - Phase 5.20 Novelty Features Test Suite
----------------------------------------------------------
Comprehensive verification for:
  1. Tailgating / Multi-Person Access Anomaly Detector
  2. Behavioral & Temporal Anomaly Engine
  3. Uncertainty-Aware Decision Layer
  4. CCSDF Multi-Dimensional Fusion Extension (S_behavior, S_asset, Confidence)
  5. Critical Safety Invariants & Multi-Track Isolation
  6. 11 Deterministic Research Scenarios (A through K)
"""

import unittest
import time
from typing import Dict, Any, List

from config import (
    TAILGATING_TIME_WINDOW_SECONDS,
    TAILGATING_SPATIAL_THRESHOLD_PIXELS,
    TAILGATING_CONFIRMATION_FRAMES,
    BEHAVIOR_DWELL_THRESHOLD_SECONDS,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
)
from src.research.tailgating_engine import (
    TailgatingDetector,
    TailgatingEvent,
    compute_centroid_distance,
)
from src.research.behavior_engine import (
    BehaviorAnomalyEngine,
    TrackBehaviorMetrics,
)
from src.research.uncertainty_engine import (
    UncertaintyAggregator,
    DecisionConfidence,
)
from src.research.ccsdf_config import (
    CCSDFWeights,
    CCSDFThresholds,
    CCSDFAction,
    WEIGHT_SET_A_BALANCED,
)
from src.research.ccsdf_engine import (
    CCSDFEngine,
    CCSDFContext,
    IdentityState,
    LivenessState,
    ZoneState,
    TemporalState,
    ProximityState,
    HistoryState,
    SmallAssetState,
    BehaviorState,
    TailgatingState,
)
from src.core.semantic_router import SemanticQueryRouter
from src.core.voice_assistant import VoiceIntent, VoiceAssistant


class TestPhase520NoveltyFeatures(unittest.TestCase):
    """Full 20-point test suite for Phase 5.20 Research Novelties."""

    def setUp(self):
        self.tailgating_detector = TailgatingDetector(confirmation_frames=2, cooldown_seconds=2.0)
        self.behavior_engine = BehaviorAnomalyEngine(dwell_threshold=10.0, approach_threshold=3, restricted_retry_threshold=2)
        self.uncertainty_aggregator = UncertaintyAggregator(high_threshold=0.75, low_threshold=0.45)
        self.ccsdf_engine = CCSDFEngine(weights=WEIGHT_SET_A_BALANCED)

    # -------------------------------------------------------------------------
    # PART 1: TAILGATING DETECTION (Tests 1 - 5)
    # -------------------------------------------------------------------------

    def test_01_authorized_and_unknown_tailgating(self):
        """Authorized leader + unknown follower within spatial threshold -> TAILGATING_SUSPECTED."""
        p_auth = {"track_id": 1, "identity": "Hunter", "is_known": True, "access_state": "GRANTED", "liveness_state": "LIVE", "bbox": (100, 100, 100, 200), "zone": "RESTRICTED"}
        p_foll = {"track_id": 2, "identity": "Unknown Person", "is_known": False, "access_state": "PENDING", "liveness_state": "LIVE", "bbox": (150, 120, 100, 200), "zone": "RESTRICTED"}

        # Observation 1: candidate
        events, s_risk = self.tailgating_detector.evaluate_tracks([p_auth, p_foll])
        self.assertFalse(events[0].is_confirmed)

        # Observation 2: confirmed
        events, s_risk = self.tailgating_detector.evaluate_tracks([p_auth, p_foll])
        self.assertTrue(len(events) > 0)
        self.assertTrue(events[0].is_confirmed)
        self.assertEqual(events[0].authorized_identity, "Hunter")
        self.assertEqual(events[0].follower_identity, "Unknown Person")
        self.assertGreaterEqual(s_risk, 0.60)

    def test_02_authorized_and_authorized_no_tailgating(self):
        """Two authorized personnel in close proximity must NOT trigger tailgating."""
        p1 = {"track_id": 1, "identity": "Hunter", "is_known": True, "access_state": "GRANTED", "liveness_state": "LIVE", "bbox": (100, 100, 100, 200), "zone": "NORMAL"}
        p2 = {"track_id": 2, "identity": "Alice", "is_known": True, "access_state": "GRANTED", "liveness_state": "LIVE", "bbox": (150, 120, 100, 200), "zone": "NORMAL"}

        events, s_risk = self.tailgating_detector.evaluate_tracks([p1, p2])
        self.assertEqual(len(events), 0)
        self.assertEqual(s_risk, 0.0)

    def test_03_unknown_and_unknown_no_authorized_leader(self):
        """Two unknown visitors without an authorized leader must NOT trigger tailgating."""
        p1 = {"track_id": 1, "identity": "Unknown 1", "is_known": False, "access_state": "PENDING", "liveness_state": "LIVE", "bbox": (100, 100, 100, 200), "zone": "NORMAL"}
        p2 = {"track_id": 2, "identity": "Unknown 2", "is_known": False, "access_state": "PENDING", "liveness_state": "LIVE", "bbox": (150, 120, 100, 200), "zone": "NORMAL"}

        events, s_risk = self.tailgating_detector.evaluate_tracks([p1, p2])
        self.assertEqual(len(events), 0)
        self.assertEqual(s_risk, 0.0)

    def test_04_temporal_tailgating_confirmation(self):
        """Single-frame transient proximity does not confirm tailgating."""
        p_auth = {"track_id": 1, "identity": "Hunter", "is_known": True, "access_state": "GRANTED", "liveness_state": "LIVE", "bbox": (100, 100, 100, 200), "zone": "NORMAL"}
        p_foll = {"track_id": 2, "identity": "Unknown", "is_known": False, "access_state": "PENDING", "liveness_state": "LIVE", "bbox": (120, 110, 100, 200), "zone": "NORMAL"}

        # 1 frame only
        events, _ = self.tailgating_detector.evaluate_tracks([p_auth, p_foll])
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].is_confirmed)

    def test_05_tailgating_resolution(self):
        """Separation or exit resolves the tailgating event cleanly."""
        p_auth = {"track_id": 1, "identity": "Hunter", "is_known": True, "access_state": "GRANTED", "liveness_state": "LIVE", "bbox": (100, 100, 100, 200), "zone": "NORMAL"}
        p_foll_close = {"track_id": 2, "identity": "Unknown", "is_known": False, "access_state": "PENDING", "liveness_state": "LIVE", "bbox": (120, 110, 100, 200), "zone": "NORMAL"}
        p_foll_far = {"track_id": 2, "identity": "Unknown", "is_known": False, "access_state": "PENDING", "liveness_state": "LIVE", "bbox": (500, 300, 100, 200), "zone": "NORMAL"}

        # Confirm tailgating
        self.tailgating_detector.evaluate_tracks([p_auth, p_foll_close])
        self.tailgating_detector.evaluate_tracks([p_auth, p_foll_close])

        # Follower separates far away (> spatial threshold)
        events, s_risk = self.tailgating_detector.evaluate_tracks([p_auth, p_foll_far])
        self.assertEqual(len(events), 0)
        self.assertEqual(s_risk, 0.0)

    # -------------------------------------------------------------------------
    # PART 2: BEHAVIORAL & TEMPORAL ANOMALY (Tests 6 - 9)
    # -------------------------------------------------------------------------

    def test_06_dwell_anomaly(self):
        """Prolonged dwell in restricted area elevates S_behavior risk."""
        t0 = 1000.0
        # Frame 1: enter restricted zone
        s1, status1, _ = self.behavior_engine.update_track_observation(
            track_id=10, identity="Hunter", is_known=True, zone="RESTRICTED", current_time=t0
        )
        self.assertEqual(status1, "NORMAL_ACTIVITY")

        # Frame 2: 12 seconds dwell in restricted area (> dwell threshold of 10s)
        s2, status2, _ = self.behavior_engine.update_track_observation(
            track_id=10, identity="Hunter", is_known=True, zone="RESTRICTED", current_time=t0 + 12.0
        )
        self.assertGreaterEqual(s2, 0.35)
        self.assertIn(status2, ("UNUSUAL_PATTERN", "BEHAVIOR_ANOMALY"))

    def test_07_repeated_restricted_access(self):
        """Repeated restricted zone entries elevate retry count and S_behavior risk."""
        t0 = 2000.0
        # Entry 1
        self.behavior_engine.update_track_observation(track_id=11, identity="Unknown", is_known=False, zone="RESTRICTED", current_time=t0)
        # Exit to normal
        self.behavior_engine.update_track_observation(track_id=11, identity="Unknown", is_known=False, zone="NORMAL", current_time=t0 + 2.0)
        # Entry 2
        self.behavior_engine.update_track_observation(track_id=11, identity="Unknown", is_known=False, zone="RESTRICTED", current_time=t0 + 4.0)
        # Exit to normal
        self.behavior_engine.update_track_observation(track_id=11, identity="Unknown", is_known=False, zone="NORMAL", current_time=t0 + 6.0)
        # Entry 3 (exceeds retry threshold of 2)
        s_risk, status, metrics = self.behavior_engine.update_track_observation(
            track_id=11, identity="Unknown", is_known=False, zone="RESTRICTED", current_time=t0 + 8.0
        )
        self.assertGreaterEqual(metrics["restricted_entries"], 3)
        self.assertGreaterEqual(s_risk, 0.40)

    def test_08_repeated_asset_approach(self):
        """Repeated approaches and retreats from protected asset elevate risk."""
        t0 = 3000.0
        # Approach 1
        self.behavior_engine.update_track_observation(track_id=12, identity="Visitor", is_known=False, zone="NORMAL", nearest_asset_dist=80.0, current_time=t0)
        # Retreat 1
        self.behavior_engine.update_track_observation(track_id=12, identity="Visitor", is_known=False, zone="NORMAL", nearest_asset_dist=250.0, current_time=t0 + 2.0)
        # Approach 2
        self.behavior_engine.update_track_observation(track_id=12, identity="Visitor", is_known=False, zone="NORMAL", nearest_asset_dist=80.0, current_time=t0 + 4.0)
        # Retreat 2
        self.behavior_engine.update_track_observation(track_id=12, identity="Visitor", is_known=False, zone="NORMAL", nearest_asset_dist=250.0, current_time=t0 + 6.0)
        # Approach 3
        s_risk, status, metrics = self.behavior_engine.update_track_observation(
            track_id=12, identity="Visitor", is_known=False, zone="NORMAL", nearest_asset_dist=80.0, current_time=t0 + 8.0
        )
        self.assertGreaterEqual(metrics["protected_asset_approaches"], 3)
        self.assertGreaterEqual(s_risk, 0.35)

    def test_09_normal_behavior(self):
        """Standard stationary or non-anomalous movement maintains NORMAL_ACTIVITY."""
        t0 = 4000.0
        s_risk, status, _ = self.behavior_engine.update_track_observation(
            track_id=13, identity="Hunter", is_known=True, zone="NORMAL", nearest_asset_dist=400.0, current_time=t0
        )
        self.assertLess(s_risk, 0.30)
        self.assertEqual(status, "NORMAL_ACTIVITY")

    # -------------------------------------------------------------------------
    # PART 3: UNCERTAINTY-AWARE DECISIONS & SAFETY GUARDS (Tests 10 - 14)
    # -------------------------------------------------------------------------

    def test_10_low_confidence_uncertainty_guard(self):
        """Low overall confidence must guard ALLOW into PENDING_VERIFICATION."""
        conf = self.uncertainty_aggregator.evaluate_confidence(
            identity_confidence=0.30, pad_confidence=0.40, object_confidence=0.50
        )
        self.assertEqual(conf.confidence_tier, "LOW")
        guarded_act, was_overridden, _ = self.uncertainty_aggregator.apply_uncertainty_action_guards(
            recommended_action="ALLOW",
            confidence=conf,
            is_known_identity=True,
            is_live_pad=True,
            authoritative_action="GRANTED",
        )
        self.assertTrue(was_overridden)
        self.assertEqual(guarded_act, "PENDING_VERIFICATION")

    def test_11_medium_confidence_caution_guard(self):
        """Medium overall confidence must guard ALLOW into ATTENTION."""
        conf = self.uncertainty_aggregator.evaluate_confidence(
            identity_confidence=0.65, pad_confidence=0.70, object_confidence=0.80
        )
        self.assertEqual(conf.confidence_tier, "MEDIUM")
        guarded_act, was_overridden, _ = self.uncertainty_aggregator.apply_uncertainty_action_guards(
            recommended_action="ALLOW",
            confidence=conf,
            is_known_identity=True,
            is_live_pad=True,
            authoritative_action="GRANTED",
        )
        self.assertTrue(was_overridden)
        self.assertEqual(guarded_act, "ATTENTION")

    def test_12_high_confidence(self):
        """High confidence preserves normal policy recommendation."""
        conf = self.uncertainty_aggregator.evaluate_confidence(
            identity_confidence=0.95, pad_confidence=0.92, object_confidence=0.90
        )
        self.assertEqual(conf.confidence_tier, "HIGH")
        guarded_act, was_overridden, _ = self.uncertainty_aggregator.apply_uncertainty_action_guards(
            recommended_action="ALLOW",
            confidence=conf,
            is_known_identity=True,
            is_live_pad=True,
            authoritative_action="GRANTED",
        )
        self.assertFalse(was_overridden)
        self.assertEqual(guarded_act, "ALLOW")

    def test_13_pad_failure_safety_guard(self):
        """PAD presentation attack / spoof failure must strictly output DENY."""
        conf = self.uncertainty_aggregator.evaluate_confidence(identity_confidence=0.98, pad_confidence=0.05)
        guarded_act, was_overridden, reason = self.uncertainty_aggregator.apply_uncertainty_action_guards(
            recommended_action="ALLOW",
            confidence=conf,
            is_known_identity=True,
            is_live_pad=False,  # Spoof
            authoritative_action="DENIED",
        )
        self.assertTrue(was_overridden)
        self.assertEqual(guarded_act, "DENY")
        self.assertIn("PAD_LIVENESS_FAILURE_GUARD", reason)

    def test_14_unknown_person_authorization_safety_guard(self):
        """Unregistered visitor must NEVER receive ALLOW recommendation."""
        conf = self.uncertainty_aggregator.evaluate_confidence(identity_confidence=0.90, pad_confidence=0.90)
        guarded_act, was_overridden, reason = self.uncertainty_aggregator.apply_uncertainty_action_guards(
            recommended_action="ALLOW",
            confidence=conf,
            is_known_identity=False,
            is_live_pad=True,
            authoritative_action="DENIED",
        )
        self.assertTrue(was_overridden)
        self.assertEqual(guarded_act, "DENY")
        self.assertIn("UNREGISTERED_IDENTITY_GUARD", reason)

    # -------------------------------------------------------------------------
    # PART 4: MULTI-TRACK ISOLATION & CCSDF FUSION (Tests 15 - 18)
    # -------------------------------------------------------------------------

    def test_15_multi_track_isolation(self):
        """Verify behavior and authorization states are strictly isolated across tracks."""
        # Track 101: Hunter (Authorized in Normal Area)
        s1, stat1, m1 = self.behavior_engine.update_track_observation(track_id=101, identity="Hunter", is_known=True, zone="NORMAL")
        # Track 102: Unknown (Restricted Area)
        s2, stat2, m2 = self.behavior_engine.update_track_observation(track_id=102, identity="Unknown", is_known=False, zone="RESTRICTED")

        self.assertEqual(m1["identity"], "Hunter")
        self.assertEqual(m2["identity"], "Unknown")
        self.assertNotEqual(m1["current_zone"], m2["current_zone"])
        self.assertEqual(m1["restricted_entries"], 0)
        self.assertEqual(m2["restricted_entries"], 1)

    def test_16_ccsdf_behavior_signal_fusion(self):
        """Verify S_behavior is included in CCSDF normalized signals and fused with w_behavior."""
        ctx = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL", occupancy="OUTSIDE"),
            temporal=TemporalState(is_authorized_hours=True),
            behavior=BehaviorState(s_behavior=0.80, status="BEHAVIOR_ANOMALY"),
        )
        res = self.ccsdf_engine.evaluate_context(ctx, authoritative_action="GRANTED")
        self.assertIn("behavior", res.normalized_signals)
        self.assertEqual(res.normalized_signals["behavior"], 0.80)
        self.assertEqual(res.behavior_status, "BEHAVIOR_ANOMALY")

    def test_17_edge_triggered_event_generation(self):
        """First tailgating detection triggers alert; subsequent identical observation is throttled."""
        event = TailgatingEvent(
            authorized_track_id=1,
            follower_track_id=2,
            time_delta=0.5,
            spatial_distance=50.0,
            authorized_identity="Hunter",
            follower_identity="Unknown",
            confidence=0.85,
            zone="NORMAL",
            is_confirmed=True,
        )
        t0 = 5000.0
        should_alert_1 = self.tailgating_detector.should_emit_alert_event(event, current_time=t0)
        self.assertTrue(should_alert_1)

        # Immediate follow-up frame must be throttled
        should_alert_2 = self.tailgating_detector.should_emit_alert_event(event, current_time=t0 + 0.1)
        self.assertFalse(should_alert_2)

    def test_18_no_event_spam(self):
        """After cooldown elapsed, event can alert again."""
        event = TailgatingEvent(
            authorized_track_id=1,
            follower_track_id=2,
            time_delta=0.5,
            spatial_distance=50.0,
            authorized_identity="Hunter",
            follower_identity="Unknown",
            confidence=0.85,
            zone="NORMAL",
            is_confirmed=True,
        )
        t0 = 6000.0
        self.tailgating_detector.should_emit_alert_event(event, current_time=t0)
        should_alert_after = self.tailgating_detector.should_emit_alert_event(event, current_time=t0 + 3.0)
        self.assertTrue(should_alert_after)

    # -------------------------------------------------------------------------
    # PART 5: GEMINI GROUNDING & PRODUCTION AUTHORITY (Tests 19 - 20)
    # -------------------------------------------------------------------------

    def test_19_gemini_query_grounding(self):
        """Copilot query router accurately matches tailgating, behavior, and confidence queries."""
        router = SemanticQueryRouter(VoiceIntent)

        res_tail = router.route_query("Was tailgating detected?")
        self.assertEqual(res_tail.intent, VoiceIntent.RESEARCH_TAILGATING_QUERY)

        res_beh = router.route_query("What behavior was unusual?")
        self.assertEqual(res_beh.intent, VoiceIntent.RESEARCH_BEHAVIOR_QUERY)

        res_conf = router.route_query("How confident is the current decision?")
        self.assertEqual(res_conf.intent, VoiceIntent.RESEARCH_CONFIDENCE_QUERY)

    def test_20_production_security_authority_unchanged(self):
        """PolicyEngine authoritative action remains completely unmutated by research fusion."""
        ctx = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL", occupancy="OUTSIDE"),
            temporal=TemporalState(is_authorized_hours=True),
            behavior=BehaviorState(s_behavior=0.0, status="NORMAL_ACTIVITY"),
        )
        eval_res = self.ccsdf_engine.evaluate_context(ctx, authoritative_action="GRANTED")
        self.assertEqual(eval_res.authoritative_action, "GRANTED")

    # -------------------------------------------------------------------------
    # PART 6: 11 DETERMINISTIC RESEARCH SCENARIOS (A through K)
    # -------------------------------------------------------------------------

    def test_21_scenarios_a_through_k(self):
        """Evaluates all 11 deterministic research scenarios."""
        scenarios = {
            "A_Auth_Normal": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True, similarity=0.95),
                liveness=LivenessState("LIVE", 0.95),
                zone=ZoneState("NORMAL", "Standard Perimeter"),
                temporal=TemporalState(is_authorized_hours=True),
            ),
            "B_Unk_Normal": CCSDFContext(
                identity=IdentityState("Unknown", is_known=False, similarity=0.20),
                liveness=LivenessState("LIVE", 0.90),
                zone=ZoneState("NORMAL", "Standard Perimeter"),
                temporal=TemporalState(is_authorized_hours=True),
            ),
            "C_Auth_Restricted": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True, similarity=0.95),
                liveness=LivenessState("LIVE", 0.95),
                zone=ZoneState("RESTRICTED", "Restricted Area", occupancy="INSIDE"),
                temporal=TemporalState(is_authorized_hours=True),
            ),
            "D_Auth_AfterHours": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True, similarity=0.95),
                liveness=LivenessState("LIVE", 0.95),
                zone=ZoneState("NORMAL", "Standard Perimeter"),
                temporal=TemporalState(is_authorized_hours=False),
            ),
            "E_Tailgating": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True, similarity=0.95),
                liveness=LivenessState("LIVE", 0.95),
                zone=ZoneState("RESTRICTED", "Restricted Area"),
                tailgating=TailgatingState(is_tailgating_suspected=True, s_tailgating=1.0, status="TAILGATING_SUSPECTED"),
            ),
            "F_Repeated_Restricted": CCSDFContext(
                identity=IdentityState("Unknown", is_known=False),
                history=HistoryState(violations_count=3, repeated_entries_count=3),
            ),
            "G_Long_Dwell": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True),
                behavior=BehaviorState(s_behavior=0.75, status="BEHAVIOR_ANOMALY", dwell_seconds=30.0),
            ),
            "H_Asset_Approach": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True),
                proximity=ProximityState(has_interaction=True, nearest_distance_pixels=20.0, asset_label="laptop"),
                behavior=BehaviorState(s_behavior=0.60, approach_count=4),
            ),
            "I_Low_Conf_Identity": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True, similarity=0.55, confidence=0.30),
            ),
            "J_Low_Conf_PAD": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True, similarity=0.95),
                liveness=LivenessState("VERIFYING", confidence=0.40),
            ),
            "K_Multi_Person_Entry": CCSDFContext(
                identity=IdentityState("Hunter", is_known=True, similarity=0.95),
                zone=ZoneState("NORMAL", occupancy="INSIDE"),
                tailgating=TailgatingState(is_tailgating_suspected=False, status="NONE"),
            ),
        }

        results = {}
        for sc_name, ctx in scenarios.items():
            res = self.ccsdf_engine.evaluate_context(ctx, authoritative_action="EVALUATION")
            self.assertIsNotNone(res)
            self.assertIsInstance(res.risk_score, float)
            self.assertIn(res.confidence_tier, ("HIGH", "MEDIUM", "LOW"))
            results[sc_name] = res

        self.assertEqual(len(results), 11)


if __name__ == "__main__":
    unittest.main()
