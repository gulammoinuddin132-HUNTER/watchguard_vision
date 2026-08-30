"""
WatchGuard Vision - Phase 5.16 Continuous Contextual Security Decision Fusion (CCSDF) Tests
-------------------------------------------------------------------------------------------
Validates:
  1. Normalized signal generation in [0.0, 1.0]
  2. Weighted risk calculation formula R(t) = sum(w_i * S_i)
  3. Parameterized weight configuration loading (Sets A, B, C, D)
  4. Track-persistent CCSDF state & context change detection
  5. Zone transition continuous authorization
  6. Time transition after-hours continuous authorization
  7. PAD failure override safety guard (Spoof -> DENY)
  8. Unknown-person override safety guard (Unregistered cannot be ALLOW)
  9. History contribution & saturation at 3 violations
  10. Proximity contribution & distance inverse scaling
  11. Baseline comparative evaluation (Baselines 1–4 vs CCSDF)
  12. Feature ablation configurations
  13. Explainable decision trace generation
  14. No database spam on unchanged context
  15. Zero impact / non-interference on authoritative baseline security decision
  16. Full integration with ContextEngine and Voice Assistant queries
"""

import unittest
from datetime import time as dtime

from src.research.ccsdf_config import (
    CCSDFWeights,
    CCSDFThresholds,
    CCSDFAction,
    WEIGHT_SET_A_BALANCED,
    WEIGHT_SET_B_PAD_HEAVY,
    WEIGHT_SET_C_ZONE_TIME_HEAVY,
    WEIGHT_SET_D_HISTORY_HEAVY,
    WEIGHT_PROFILES,
    ABLATION_CONFIGS,
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
    TrackCCSDFState,
)
from src.research.scenarios import get_all_scenarios, SecurityScenario
from src.research.scenario_runner import (
    run_baseline_comparison,
    run_weight_sensitivity_study,
    run_ablation_study,
)
from src.core.context_engine import ContextEngine, SecurityDecision, RiskLevel
from src.core.voice_assistant import VoiceAssistant, VoiceIntent


class TestPhase516CCSDF(unittest.TestCase):
    """Phase 5.16 CCSDF Research Engine Test Suite."""

    def setUp(self):
        self.engine = CCSDFEngine(weights=WEIGHT_SET_A_BALANCED)

    # -------------------------------------------------------------------------
    # 1. Normalized Signal Generation
    # -------------------------------------------------------------------------
    def test_01_normalized_signal_bounds(self):
        """Signals must be bounded strictly within [0.0, 1.0]."""
        id_known = IdentityState(is_known=True, similarity=0.95)
        id_unknown = IdentityState(is_known=False, similarity=0.0)
        self.assertEqual(id_known.get_normalized_risk(), 0.0)
        self.assertEqual(id_unknown.get_normalized_risk(), 1.0)

        live_ok = LivenessState(liveness_state="LIVE", confidence=1.0)
        live_warmup = LivenessState(liveness_state="WARMUP", confidence=0.5)
        live_spoof = LivenessState(liveness_state="SPOOF", confidence=0.1)
        self.assertEqual(live_ok.get_normalized_risk(), 0.0)
        self.assertEqual(live_warmup.get_normalized_risk(), 0.5)
        self.assertEqual(live_spoof.get_normalized_risk(), 1.0)

        zone_norm = ZoneState(zone_id="NORMAL")
        zone_prot = ZoneState(zone_id="PROTECTED_ASSET_AREA")
        zone_rest = ZoneState(zone_id="RESTRICTED")
        self.assertEqual(zone_norm.get_normalized_risk(), 0.0)
        self.assertEqual(zone_prot.get_normalized_risk(), 0.5)
        self.assertEqual(zone_rest.get_normalized_risk(), 1.0)

        time_ok = TemporalState(is_authorized_hours=True)
        time_ah = TemporalState(is_authorized_hours=False)
        self.assertEqual(time_ok.get_normalized_risk(), 0.0)
        self.assertEqual(time_ah.get_normalized_risk(), 1.0)

        prox_clear = ProximityState(has_interaction=False)
        prox_touch = ProximityState(has_interaction=True, is_holding_or_touching=True)
        prox_mid = ProximityState(has_interaction=True, nearest_distance_pixels=60.0, proximity_threshold_pixels=120.0)
        self.assertEqual(prox_clear.get_normalized_risk(), 0.0)
        self.assertEqual(prox_touch.get_normalized_risk(), 1.0)
        self.assertAlmostEqual(prox_mid.get_normalized_risk(), 0.5, places=2)

    # -------------------------------------------------------------------------
    # 2. Weighted Risk Calculation
    # -------------------------------------------------------------------------
    def test_02_weighted_risk_calculation(self):
        """Verifies mathematical weighted sum formula: R(t) = sum(w_i * S_i)."""
        # All signals 0.0 -> Risk 0.00
        ctx_zero = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=1.0),
            liveness=LivenessState(liveness_state="LIVE", confidence=1.0),
            zone=ZoneState(zone_id="NORMAL"),
            temporal=TemporalState(is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False),
            history=HistoryState(violations_count=0),
        )
        res_zero = self.engine.evaluate_context(ctx_zero, authoritative_action="GRANTED")
        self.assertEqual(res_zero.risk_score, 0.0)
        self.assertEqual(res_zero.recommended_action, CCSDFAction.ALLOW)

        # All signals 1.0 -> Risk 1.00
        ctx_max = CCSDFContext(
            identity=IdentityState(is_known=False),
            liveness=LivenessState(liveness_state="SPOOF"),
            zone=ZoneState(zone_id="RESTRICTED"),
            temporal=TemporalState(is_authorized_hours=False),
            proximity=ProximityState(has_interaction=True, is_holding_or_touching=True),
            history=HistoryState(violations_count=5),
        )
        res_max = self.engine.evaluate_context(ctx_max, authoritative_action="DENIED")
        self.assertEqual(res_max.risk_score, 1.0)
        self.assertEqual(res_max.recommended_action, CCSDFAction.DENY)

    # -------------------------------------------------------------------------
    # 3. Parameterized Configuration Loading
    # -------------------------------------------------------------------------
    def test_03_configuration_profiles(self):
        """Verifies weight profiles sum to 1.0 and load correctly."""
        for name, profile in WEIGHT_PROFILES.items():
            total = sum(profile.to_dict().values())
            self.assertAlmostEqual(total, 1.0, places=3, msg=f"Profile {name} does not sum to 1.0")

    # -------------------------------------------------------------------------
    # 4. Track-Persistent CCSDF State & Change Detection
    # -------------------------------------------------------------------------
    def test_04_track_persistent_state(self):
        """Track state must re-evaluate only when context hash changes."""
        track = TrackCCSDFState(track_id=101)
        ctx = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.92),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL"),
            temporal=TemporalState(is_authorized_hours=True),
        )

        # First evaluation
        has_changed, res1 = track.update(self.engine, ctx, authoritative_action="GRANTED")
        self.assertTrue(has_changed)
        self.assertEqual(track.re_evaluation_count, 1)

        # Identical second frame -> no re-evaluation
        has_changed, res2 = track.update(self.engine, ctx, authoritative_action="GRANTED")
        self.assertFalse(has_changed)
        self.assertEqual(track.re_evaluation_count, 1)

        # Context change (moves to restricted) -> re-evaluates
        ctx_changed = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.92),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="RESTRICTED"),
            temporal=TemporalState(is_authorized_hours=True),
        )
        has_changed, res3 = track.update(self.engine, ctx_changed, authoritative_action="GRANTED")
        self.assertTrue(has_changed)
        self.assertEqual(track.re_evaluation_count, 2)

    # -------------------------------------------------------------------------
    # 5. Zone Transition Continuous Authorization
    # -------------------------------------------------------------------------
    def test_05_zone_transition_evaluation(self):
        """Hunter in Standard (ALLOW) moves into Restricted (Risk escalates)."""
        ctx_std = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL"),
            temporal=TemporalState(is_authorized_hours=True),
        )
        res_std = self.engine.evaluate_context(ctx_std, authoritative_action="GRANTED")
        self.assertEqual(res_std.recommended_action, CCSDFAction.ALLOW)

        ctx_rest = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="RESTRICTED"),
            temporal=TemporalState(is_authorized_hours=True),
        )
        res_rest = self.engine.evaluate_context(ctx_rest, authoritative_action="GRANTED")
        self.assertGreater(res_rest.risk_score, res_std.risk_score)
        self.assertEqual(res_rest.dominant_factor, "zone")

    # -------------------------------------------------------------------------
    # 6. Time Transition After-Hours Authorization
    # -------------------------------------------------------------------------
    def test_06_time_transition_after_hours(self):
        """Crossing 18:00 into after-hours must elevate risk and recommend DENY."""
        ctx_ah = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="RESTRICTED"),
            temporal=TemporalState(current_time_str="22:45", is_authorized_hours=False),
        )
        res_ah = self.engine.evaluate_context(ctx_ah, authoritative_action="DENIED")
        self.assertGreaterEqual(res_ah.risk_score, 0.35)
        self.assertEqual(res_ah.recommended_action, CCSDFAction.DENY)

    # -------------------------------------------------------------------------
    # 7. PAD Failure Override Safety Guard
    # -------------------------------------------------------------------------
    def test_07_spoof_override_safety_guard(self):
        """Spoof presentation attack must strictly output DENY regardless of high face similarity."""
        ctx_spoof = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.99),
            liveness=LivenessState(liveness_state="SPOOF", confidence=0.10),
            zone=ZoneState(zone_id="NORMAL"),
            temporal=TemporalState(is_authorized_hours=True),
        )
        res = self.engine.evaluate_context(ctx_spoof, authoritative_action="DENIED")
        self.assertTrue(res.is_conflict_overridden)
        self.assertEqual(res.recommended_action, CCSDFAction.DENY)
        self.assertEqual(res.risk_level, "RED")

    # -------------------------------------------------------------------------
    # 8. Unknown-Person Safety Guard
    # -------------------------------------------------------------------------
    def test_08_unknown_person_safety_guard(self):
        """Unregistered visitor must NEVER receive ALLOW recommendation."""
        ctx_unreg = CCSDFContext(
            identity=IdentityState(name="Unknown Person", is_known=False, similarity=0.0),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL"),
            temporal=TemporalState(is_authorized_hours=True),
        )
        res = self.engine.evaluate_context(ctx_unreg, authoritative_action="DENIED")
        self.assertNotEqual(res.recommended_action, CCSDFAction.ALLOW)
        self.assertIn(res.recommended_action, (CCSDFAction.ATTENTION, CCSDFAction.DENY))

    # -------------------------------------------------------------------------
    # 9. History Contribution & Saturation
    # -------------------------------------------------------------------------
    def test_09_history_saturation(self):
        """History risk scales linearly and saturates at 3 violations."""
        h0 = HistoryState(violations_count=0)
        h1 = HistoryState(violations_count=1)
        h2 = HistoryState(violations_count=2)
        h3 = HistoryState(violations_count=3)
        h5 = HistoryState(violations_count=5)

        self.assertEqual(h0.get_normalized_risk(), 0.0)
        self.assertAlmostEqual(h1.get_normalized_risk(), 0.3333, places=3)
        self.assertAlmostEqual(h2.get_normalized_risk(), 0.6667, places=3)
        self.assertEqual(h3.get_normalized_risk(), 1.0)
        self.assertEqual(h5.get_normalized_risk(), 1.0)

    # -------------------------------------------------------------------------
    # 10. Proximity Contribution & Scaling
    # -------------------------------------------------------------------------
    def test_10_proximity_distance_scaling(self):
        """Proximity risk increases as distance to asset approaches zero."""
        p_far = ProximityState(has_interaction=True, nearest_distance_pixels=120.0, proximity_threshold_pixels=120.0)
        p_near = ProximityState(has_interaction=True, nearest_distance_pixels=30.0, proximity_threshold_pixels=120.0)
        p_touch = ProximityState(has_interaction=True, is_holding_or_touching=True)

        self.assertEqual(p_far.get_normalized_risk(), 0.0)
        self.assertAlmostEqual(p_near.get_normalized_risk(), 0.75, places=2)
        self.assertEqual(p_touch.get_normalized_risk(), 1.0)

    # -------------------------------------------------------------------------
    # 11. Baseline Comparative Evaluation
    # -------------------------------------------------------------------------
    def test_11_baseline_comparative_runner(self):
        """Executes scenario runner across all 4 baselines and CCSDF."""
        metrics = run_baseline_comparison()
        self.assertIn("Baseline 1 (Face Only)", metrics)
        self.assertIn("Baseline 2 (Face + PAD)", metrics)
        self.assertIn("Baseline 3 (Face+PAD+Zone+Time)", metrics)
        self.assertIn("Baseline 4 (Face+PAD+Zone+Time+Prox)", metrics)
        self.assertIn("WatchGuard CCSDF (Proposed)", metrics)

        # Baseline 1 must fail spoof (FAUS = 1.0)
        self.assertEqual(metrics["Baseline 1 (Face Only)"].faus, 1.0)
        # Baseline 2 must block spoof (FAUS = 0.0), but fail after-hours context (FAPC > 0)
        self.assertEqual(metrics["Baseline 2 (Face + PAD)"].faus, 0.0)
        self.assertGreater(metrics["Baseline 2 (Face + PAD)"].fapc, 0.0)
        # WatchGuard CCSDF must have 0% FAUS and 0% FAPC
        self.assertEqual(metrics["WatchGuard CCSDF (Proposed)"].faus, 0.0)
        self.assertEqual(metrics["WatchGuard CCSDF (Proposed)"].fapc, 0.0)

    # -------------------------------------------------------------------------
    # 12. Feature Ablation Study
    # -------------------------------------------------------------------------
    def test_12_ablation_study_runner(self):
        """Runs ablation study over all feature masks."""
        ablation_res = run_ablation_study()
        for mask_name in ABLATION_CONFIGS.keys():
            self.assertIn(mask_name, ablation_res)
            self.assertGreaterEqual(ablation_res[mask_name].total_evaluations, 10)

    # -------------------------------------------------------------------------
    # 13. Decision Trace Generation
    # -------------------------------------------------------------------------
    def test_13_decision_trace_generation(self):
        """Trace must include WHO, IDENTITY, LIVENESS, ZONE, TIME, RISK, and RECOMMENDATION."""
        ctx = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.94),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="RESTRICTED"),
            temporal=TemporalState(current_time_str="14:00", is_authorized_hours=True),
        )
        eval_res = self.engine.evaluate_context(ctx, authoritative_action="GRANTED")
        trace_str = eval_res.trace.to_string()
        trace_dict = eval_res.trace.to_dict()

        self.assertIn("WHO: Hunter", trace_str)
        self.assertIn("ZONE: Restricted Area", trace_str)
        self.assertIn("CCSDF RISK:", trace_str)
        self.assertIn("RECOMMENDED:", trace_str)
        self.assertIn("AUTHORITATIVE: GRANTED", trace_str)
        self.assertEqual(trace_dict["who"], "Hunter")

    # -------------------------------------------------------------------------
    # 14. ContextEngine Integration & Zero Authoritative Impact
    # -------------------------------------------------------------------------
    def test_14_context_engine_authoritative_preservation(self):
        """ContextEngine must populate decision.ccsdf_result without modifying is_access_granted."""
        ce = ContextEngine(db_manager=None)
        person_data = {
            "detected": True,
            "people": [{
                "name": "Unknown Person",
                "is_known": False,
                "similarity": 0.0,
                "bbox": [50, 50, 100, 150],
                "track_id": 1,
            }],
        }
        dec = ce.evaluate_context(person_data=person_data, object_data={"objects": []})

        # Authoritative decision is strictly preserved
        self.assertFalse(dec.is_access_granted)
        self.assertEqual(dec.access_state, "DENIED")
        self.assertEqual(dec.decision, "UNKNOWN_PERSON")

        # CCSDF research result is attached
        self.assertIsNotNone(dec.ccsdf_result)
        self.assertIn("risk_score", dec.ccsdf_result)
        self.assertIn("recommended_action", dec.ccsdf_result)

    # -------------------------------------------------------------------------
    # 15. Voice Assistant Research Queries
    # -------------------------------------------------------------------------
    def test_15_voice_assistant_research_queries(self):
        """Voice Assistant handles research queries with mandatory [RESEARCH SIMULATION] disclaimer."""
        ce = ContextEngine(db_manager=None)
        va = VoiceAssistant(context_engine=ce, db_manager=None, camera_manager=None)

        # Baseline query
        resp_base = va.process_command("What would the baseline face-only system have decided?")
        self.assertIn("[RESEARCH SIMULATION / BASELINE]", resp_base)

        # Contributing factor query
        resp_factors = va.process_command("What factors caused the risk increase?")
        self.assertIn("CCSDF contextual risk", resp_factors)

        # Authorization change query
        resp_auth = va.process_command("Why did authorization change?")
        self.assertIn("Authorization is currently", resp_auth)


if __name__ == "__main__":
    unittest.main()
