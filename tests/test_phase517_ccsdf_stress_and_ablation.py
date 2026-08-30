"""
WatchGuard Vision - Phase 5.17 Automated Test Suite
---------------------------------------------------
Tests for CCSDF Stress Testing, Baseline Comparisons, Feature Ablations,
Weight Sensitivity, and Continuous Authorization Sequences.
"""

import os
import unittest
from src.research.ccsdf_config import (
    WEIGHT_PROFILES,
    ABLATION_CONFIGS,
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
)
from src.research.ccsdf_stress_evaluator import (
    CCSDFStressHarness,
    build_stress_scenarios,
    evaluate_baseline_1_face_only,
    evaluate_baseline_2_face_pad,
    evaluate_baseline_3_face_pad_zone_time,
    evaluate_baseline_4_face_pad_zone_time_prox,
    evaluate_watchguard_ccsdf,
    run_continuous_authorization_sequence,
    run_multi_person_tracking_simulation,
)


class TestPhase517CCSDFStressAndAblation(unittest.TestCase):
    """Automated test cases for Phase 5.17 research evaluations."""

    @classmethod
    def setUpClass(cls):
        cls.harness = CCSDFStressHarness(output_dir="data/research")
        cls.results = cls.harness.run_full_evaluation()

    def test_01_all_12_stress_scenarios_execute(self):
        """Verify all 12 stress scenarios (A through L) are defined and evaluated."""
        scenarios = build_stress_scenarios()
        self.assertEqual(len(scenarios), 12, "Must contain exactly 12 stress scenarios (A through L).")
        scenario_ids = [s.scenario_id for s in scenarios]
        expected_ids = [f"SCENARIO_{c}" for c in "ABCDEFGHIJKL"]
        self.assertEqual(scenario_ids, expected_ids)

    def test_02_baseline_progression_metrics(self):
        """Verify baseline progression: adding PAD eliminates Spoof False Access."""
        m1 = self.results["baseline_metrics"]["BASELINE_1_FACE_ONLY"]
        m2 = self.results["baseline_metrics"]["BASELINE_2_FACE_PAD"]
        m_ccsdf = self.results["baseline_metrics"]["WATCHGUARD_CCSDF_FULL"]

        # Baseline 1 has spoof false access
        self.assertEqual(m1["spoof_false_access_count"], 1, "Baseline 1 must fail on spoof presentation attack.")
        # Baseline 2 and CCSDF have ZERO spoof false access
        self.assertEqual(m2["spoof_false_access_count"], 0, "Baseline 2 must prevent spoof false access.")
        self.assertEqual(m_ccsdf["spoof_false_access_count"], 0, "Full CCSDF must prevent spoof false access.")
        # Baseline 1 has higher FAR than Baseline 2
        self.assertGreater(m1["far"], m2["far"])

    def test_03_ablation_study_execution(self):
        """Verify all 6 ablation configurations produce valid results."""
        abl_metrics = self.results["ablation_metrics"]
        self.assertEqual(len(abl_metrics), 6)
        self.assertIn("FULL_CCSDF", abl_metrics)
        self.assertIn("MINUS_HISTORY", abl_metrics)
        self.assertIn("MINUS_PROXIMITY", abl_metrics)
        self.assertIn("MINUS_TIME", abl_metrics)
        self.assertIn("MINUS_ZONE", abl_metrics)
        self.assertIn("MINUS_PAD", abl_metrics)

    def test_04_weight_sensitivity_profiles(self):
        """Verify all 4 weight profiles are evaluated."""
        sens_metrics = self.results["weight_sensitivity_metrics"]
        self.assertEqual(len(sens_metrics), 4)
        self.assertIn("SET_A_BALANCED", sens_metrics)
        self.assertIn("SET_B_PAD_HEAVY", sens_metrics)
        self.assertIn("SET_C_ZONE_TIME_HEAVY", sens_metrics)
        self.assertIn("SET_D_HISTORY_HEAVY", sens_metrics)

    def test_05_continuous_authorization_sequence(self):
        """Verify continuous authorization tracks transitions across 5 sequence steps."""
        engine = CCSDFEngine(weights=WEIGHT_SET_A_BALANCED)
        trace = run_continuous_authorization_sequence(engine)
        self.assertEqual(len(trace), 5)
        # Step 1 is low risk (Permitted)
        self.assertLess(trace[0]["risk_score"], 0.25)
        # Step 4 (near asset tampering after hours) has higher risk than Step 1
        self.assertGreater(trace[3]["risk_score"], trace[0]["risk_score"])

    def test_06_multi_person_tracking_isolation(self):
        """Verify multi-person simultaneous evaluation isolates tracks."""
        engine = CCSDFEngine(weights=WEIGHT_SET_A_BALANCED)
        trace = run_multi_person_tracking_simulation(engine)
        self.assertEqual(len(trace), 2)
        t_auth = trace[0]
        t_unknown = trace[1]
        self.assertEqual(t_auth["track_id"], 101)
        self.assertEqual(t_auth["recommended_action"], "ALLOW")
        self.assertEqual(t_unknown["track_id"], 102)
        self.assertEqual(t_unknown["recommended_action"], "ATTENTION")
        self.assertTrue(t_auth["state_isolated"])
        self.assertTrue(t_unknown["state_isolated"])

    def test_07_machine_readable_exports_present(self):
        """Verify JSON and CSV export files exist and are non-empty."""
        json_path = "data/research/ccsdf_stress_results.json"
        csv_path = "data/research/ccsdf_stress_results.csv"
        self.assertTrue(os.path.exists(json_path), f"Missing {json_path}")
        self.assertTrue(os.path.exists(csv_path), f"Missing {csv_path}")
        self.assertGreater(os.path.getsize(json_path), 500)
        self.assertGreater(os.path.getsize(csv_path), 500)


if __name__ == "__main__":
    unittest.main()
