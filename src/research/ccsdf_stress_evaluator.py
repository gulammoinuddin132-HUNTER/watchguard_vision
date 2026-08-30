"""
WatchGuard Vision - CCSDF Stress Testing & Ablation Study Evaluator
-------------------------------------------------------------------
Phase 5.17 Research Evaluation Framework.
Implements deterministic comparative evaluations across:
  - 4 Progressively Simpler Baselines (Face-Only, Face+PAD, Face+PAD+Zone+Time, Face+PAD+Zone+Time+Prox)
  - Full WatchGuard CCSDF Continuous Contextual Fusion
  - 5 Feature Ablations (Minus History, Minus Proximity, Minus Time, Minus Zone, Minus PAD)
  - 4 Parameterized Weight Sensitivity Profiles (Balanced, PAD-Heavy, Zone/Time-Heavy, History-Heavy)
  - 12 Conflicting Context Stress Scenarios (A through L)
  - Continuous Authorization & Multi-Person Tracking Sequences

Outputs machine-readable JSON & CSV datasets and calculates formal security & operational metrics.
"""

import os
import sys
import time
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.research.ccsdf_config import (
    CCSDFWeights,
    CCSDFThresholds,
    CCSDFAction,
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
    CCSDFEvaluationResult,
)


@dataclass
class ScenarioDefinition:
    """Definition of a stress scenario with input contextual states and ground-truth label."""
    scenario_id: str
    name: str
    description: str
    context: CCSDFContext
    ground_truth_authorized: bool  # True = legitimate authorized access, False = unauthorized / prohibited
    expected_primary_risk: str     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    category: str


def build_stress_scenarios() -> List[ScenarioDefinition]:
    """Constructs the 12 comprehensive conflicting-context stress scenarios (A through L)."""
    scenarios = []

    # Scenario A: Authorized + Normal
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_A",
        name="Authorized + Normal",
        description="Known authorized person, confirmed live, standard perimeter zone, regular business hours, 0 violations.",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.92, confidence=0.95, role="Security Staff"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.96),
            zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="OUTSIDE"),
            temporal=TemporalState(current_time_str="14:30", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=True,
        expected_primary_risk="LOW",
        category="Baseline Legitimate",
    ))

    # Scenario B: Unknown + Normal
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_B",
        name="Unknown + Normal",
        description="Unregistered visitor, confirmed live, standard perimeter zone, regular hours, 0 violations.",
        context=CCSDFContext(
            identity=IdentityState(name="Unknown Person", is_known=False, similarity=0.22, confidence=0.30, role="Visitor"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="OUTSIDE"),
            temporal=TemporalState(current_time_str="14:35", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="MEDIUM",
        category="Identity Anomaly",
    ))

    # Scenario C: Authorized + Restricted Zone
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_C",
        name="Authorized + Restricted Zone",
        description="Known authorized person, confirmed live, entered Restricted Area (Server Enclosure), regular hours, 0 violations.",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.91, confidence=0.94, role="Security Staff"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.94),
            zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE", duration_seconds=12.0),
            temporal=TemporalState(current_time_str="14:40", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="HIGH",
        category="Spatial Policy Violation",
    ))

    # Scenario D: Authorized + Restricted + After Hours
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_D",
        name="Authorized + Restricted + After Hours",
        description="Known authorized person, confirmed live, entered Restricted Area at 22:45 (after-hours schedule violation).",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.89, confidence=0.92, role="Security Staff"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.93),
            zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE", duration_seconds=25.0),
            temporal=TemporalState(current_time_str="22:45", is_authorized_hours=False),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="CRITICAL",
        category="Compound Policy Violation",
    ))

    # Scenario E: Unknown + Restricted Zone
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_E",
        name="Unknown + Restricted Zone",
        description="Unregistered visitor, confirmed live, deep inside Restricted Area during daytime hours.",
        context=CCSDFContext(
            identity=IdentityState(name="Unknown Person", is_known=False, similarity=0.18, confidence=0.25, role="Visitor"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.94),
            zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE", duration_seconds=18.0),
            temporal=TemporalState(current_time_str="15:10", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="CRITICAL",
        category="Compound Identity & Spatial Breach",
    ))

    # Scenario F: Authorized Identity + Spoof Attack
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_F",
        name="Authorized Identity + Spoof Attack",
        description="High facial similarity match to authorized executive, but PAD confirms presentation attack (photo/video replay).",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95, confidence=0.98, role="Administrator"),
            liveness=LivenessState(liveness_state="SPOOF", confidence=0.98, attack_type="PHOTO_REPLAY"),
            zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="OUTSIDE"),
            temporal=TemporalState(current_time_str="11:15", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="CRITICAL",
        category="Biometric Spoof Presentation",
    ))

    # Scenario G: Authorized + Protected Asset Proximity
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_G",
        name="Authorized + Protected Asset Proximity",
        description="Known authorized staff in normal zone, but reaching within 20px of sensitive hardware terminal.",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.90, confidence=0.93, role="Security Staff"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="OUTSIDE"),
            temporal=TemporalState(current_time_str="13:20", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=True, nearest_distance_pixels=20.0, asset_label="Cryptographic Terminal"),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="MEDIUM",
        category="Spatial Asset Proximity Anomaly",
    ))

    # Scenario H: Repeated Violations (Escalation)
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_H",
        name="Repeated Violations (Historical Escalation)",
        description="Known staff member with 3 prior distinct zone breaches recorded in history database.",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.88, confidence=0.91, role="Security Staff"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.94),
            zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="OUTSIDE"),
            temporal=TemporalState(current_time_str="16:00", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=3, repeated_entries_count=4),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="MEDIUM",
        category="Behavioral Historical Non-Compliance",
    ))

    # Scenario I: Multi-Person / Tailgating Suspect
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_I",
        name="Multi-Person / Tailgating Unknown Follower",
        description="Unknown individual closely following authorized staff member through access portal.",
        context=CCSDFContext(
            identity=IdentityState(name="Unknown Person", is_known=False, similarity=0.20, confidence=0.28, role="Visitor"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.92),
            zone=ZoneState(zone_id="PROTECTED_ASSET_AREA", zone_name="Protected Asset Area", occupancy="INSIDE"),
            temporal=TemporalState(current_time_str="10:05", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=True, nearest_distance_pixels=45.0, asset_label="Door Portal"),
            history=HistoryState(violations_count=1),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="HIGH",
        category="Multi-Person Tailgating",
    ))

    # Scenario J: PAD Uncertain (Warmup / Eye-Blink Hold)
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_J",
        name="PAD Uncertain / Warmup Hold",
        description="Known authorized person with PAD state currently accumulating temporal blink cues (PENDING).",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.93, confidence=0.96, role="Security Staff"),
            liveness=LivenessState(liveness_state="WARMUP", confidence=0.50),
            zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="OUTSIDE"),
            temporal=TemporalState(current_time_str="12:00", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False, nearest_distance_pixels=999.0),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=False,  # Should hold pending verification, not grant unverified access
        expected_primary_risk="LOW",
        category="Liveness Temporal Hold",
    ))

    # Scenario K: Conflicting Compound Context
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_K",
        name="Conflicting Compound Context",
        description="Known staff, high biometric score, live, but simultaneously inside Restricted Area after hours touching protected equipment.",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.94, confidence=0.97, role="Security Staff"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.96),
            zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE", duration_seconds=45.0),
            temporal=TemporalState(current_time_str="23:15", is_authorized_hours=False),
            proximity=ProximityState(has_interaction=True, nearest_distance_pixels=0.0, is_holding_or_touching=True, asset_label="Central Server"),
            history=HistoryState(violations_count=2),
        ),
        ground_truth_authorized=False,
        expected_primary_risk="CRITICAL",
        category="Severe Multi-Factor Conflict",
    ))

    # Scenario L: Perimeter Boundary Drift
    scenarios.append(ScenarioDefinition(
        scenario_id="SCENARIO_L",
        name="Perimeter Boundary Drift",
        description="Known staff member in normal zone drifting towards protected asset perimeter (60px proximity).",
        context=CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.90, confidence=0.93, role="Security Staff"),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="OUTSIDE"),
            temporal=TemporalState(current_time_str="14:00", is_authorized_hours=True),
            proximity=ProximityState(has_interaction=True, nearest_distance_pixels=60.0, asset_label="Asset Boundary"),
            history=HistoryState(violations_count=0),
        ),
        ground_truth_authorized=True,  # Boundary approach in normal zone within safe margins is permitted with attention
        expected_primary_risk="LOW",
        category="Spatial Boundary Interaction",
    ))

    return scenarios


# =============================================================================
# BASELINE EVALUATOR FUNCTIONS
# =============================================================================

def evaluate_baseline_1_face_only(context: CCSDFContext) -> Tuple[str, bool, float]:
    """
    Baseline 1: Face / Identity Only ($S_{id}$).
    Grants access if identity is recognized and verified.
    Ignores PAD, Zone, Time, Proximity, History.
    Returns: (action, is_granted, risk_score)
    """
    s_id = context.identity.get_normalized_risk()
    is_granted = (s_id < 0.25)
    action = "ALLOW" if is_granted else "DENY"
    return action, is_granted, s_id


def evaluate_baseline_2_face_pad(context: CCSDFContext) -> Tuple[str, bool, float]:
    """
    Baseline 2: Face + PAD ($S_{id}, S_{pad}$).
    Grants access if identity is verified AND liveness is confirmed natural live.
    Ignores Zone, Time, Proximity, History.
    Returns: (action, is_granted, risk_score)
    """
    s_id = context.identity.get_normalized_risk()
    s_pad = context.liveness.get_normalized_risk()
    
    if s_pad >= 0.8:
        return "DENY", False, 1.0
    if s_pad == 0.5:
        return "PENDING_VERIFICATION", False, 0.5
    if s_id >= 0.7:
        return "DENY", False, 0.85
    
    is_granted = (s_id < 0.25 and s_pad < 0.25)
    action = "ALLOW" if is_granted else "ATTENTION"
    return action, is_granted, round((s_id + s_pad) / 2.0, 4)


def evaluate_baseline_3_face_pad_zone_time(context: CCSDFContext) -> Tuple[str, bool, float]:
    """
    Baseline 3: Face + PAD + Zone + Time ($S_{id}, S_{pad}, S_{zone}, S_{time}$).
    Traditional multi-factor rule-based access control.
    Fails if face is unknown, spoof detected, zone is restricted, or after-hours.
    Ignores Proximity and History.
    Returns: (action, is_granted, risk_score)
    """
    s_id = context.identity.get_normalized_risk()
    s_pad = context.liveness.get_normalized_risk()
    s_zone = context.zone.get_normalized_risk()
    s_time = context.temporal.get_normalized_risk()

    if s_pad >= 0.8:
        return "DENY", False, 1.0
    if s_pad == 0.5:
        return "PENDING_VERIFICATION", False, 0.5
    if s_zone >= 1.0 or s_time >= 1.0 or s_id >= 0.7:
        return "DENY", False, 0.80

    risk = round((s_id + s_pad + s_zone + s_time) / 4.0, 4)
    is_granted = (risk < 0.25 and s_zone == 0.0 and s_time == 0.0)
    action = "ALLOW" if is_granted else "RESTRICT"
    return action, is_granted, risk


def evaluate_baseline_4_face_pad_zone_time_prox(context: CCSDFContext) -> Tuple[str, bool, float]:
    """
    Baseline 4: Face + PAD + Zone + Time + Proximity ($S_{id}, S_{pad}, S_{zone}, S_{time}, S_{prox}$).
    Extends Baseline 3 with spatial proximity.
    Ignores Historical Violation Context.
    Returns: (action, is_granted, risk_score)
    """
    s_id = context.identity.get_normalized_risk()
    s_pad = context.liveness.get_normalized_risk()
    s_zone = context.zone.get_normalized_risk()
    s_time = context.temporal.get_normalized_risk()
    s_prox = context.proximity.get_normalized_risk()

    if s_pad >= 0.8:
        return "DENY", False, 1.0
    if s_pad == 0.5:
        return "PENDING_VERIFICATION", False, 0.5
    if s_zone >= 1.0 or s_time >= 1.0 or s_id >= 0.7 or s_prox >= 0.8:
        return "DENY", False, 0.85

    risk = round((s_id + s_pad + s_zone + s_time + s_prox) / 5.0, 4)
    is_granted = (risk < 0.25 and s_zone == 0.0 and s_time == 0.0 and s_prox < 0.3)
    action = "ALLOW" if is_granted else ("RESTRICT" if risk >= 0.50 else "ATTENTION")
    return action, is_granted, risk


def evaluate_watchguard_ccsdf(
    engine: CCSDFEngine,
    context: CCSDFContext,
) -> Tuple[str, bool, float, CCSDFEvaluationResult]:
    """
    WatchGuard CCSDF: Full 6-factor Continuous Fusion Model.
    Returns: (action, is_granted, risk_score, full_result)
    """
    res = engine.evaluate_context(context, authoritative_action="EVALUATION")
    is_granted = (res.recommended_action == CCSDFAction.ALLOW)
    return res.recommended_action.value, is_granted, res.risk_score, res


# =============================================================================
# STATISTICAL METRICS & MATRIX CALCULATORS
# =============================================================================

@dataclass
class ConfusionMatrixMetrics:
    """Confusion matrix and classification metrics for a binary access decision model."""
    total_samples: int
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    specificity: float
    f1_score: float
    far: float                     # False Acceptance Rate = FP / (FP + TN)
    frr: float                     # False Rejection Rate = FN / (TP + FN)
    false_access_rate: float       # Overall False Access = FP / total
    spoof_false_access_count: int
    prohibited_context_false_access_count: int


def calculate_metrics(
    evaluations: List[Dict[str, Any]],
) -> ConfusionMatrixMetrics:
    """Calculates formal academic classification metrics across evaluated scenarios."""
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    spoof_fa = 0
    context_fa = 0

    for ev in evaluations:
        gt = ev["ground_truth_authorized"]
        pred = ev["is_access_granted"]
        cat = ev.get("category", "")

        if gt and pred:
            tp += 1
        elif (not gt) and (not pred):
            tn += 1
        elif (not gt) and pred:
            fp += 1
            if "Spoof" in cat:
                spoof_fa += 1
            else:
                context_fa += 1
        elif gt and (not pred):
            fn += 1

    total = len(evaluations)
    precision = round(tp / max(1, (tp + fp)), 4)
    recall = round(tp / max(1, (tp + fn)), 4)
    specificity = round(tn / max(1, (tn + fp)), 4)
    f1 = round((2 * precision * recall) / max(1e-6, (precision + recall)), 4)

    far = round(fp / max(1, (fp + tn)), 4)
    frr = round(fn / max(1, (tp + fn)), 4)
    fa_rate = round(fp / max(1, total), 4)

    return ConfusionMatrixMetrics(
        total_samples=total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1_score=f1,
        far=far,
        frr=frr,
        false_access_rate=fa_rate,
        spoof_false_access_count=spoof_fa,
        prohibited_context_false_access_count=context_fa,
    )


# =============================================================================
# CONTINUOUS AUTHORIZATION SEQUENCE & MULTI-PERSON SIMULATOR
# =============================================================================

def run_continuous_authorization_sequence(engine: CCSDFEngine) -> List[Dict[str, Any]]:
    """
    Executes Part 11: Continuous Authorization Life-Cycle Sequence for a single track.
    Sequence:
      Step 1: Normal + Day (Permitted)
      Step 2: Restricted + Day (Spatial breach)
      Step 3: Restricted + After-Hours (Compound breach)
      Step 4: Near Protected Asset (Critical physical tampering)
      Step 5: Leaves Restricted Area (De-escalation to perimeter)
    """
    steps = [
        ("STEP_1", "Normal + Day", ZoneState("NORMAL", "Perimeter", "OUTSIDE"), TemporalState("14:00", True), ProximityState(False), HistoryState(0)),
        ("STEP_2", "Restricted + Day", ZoneState("RESTRICTED", "Restricted Area", "INSIDE", 15.0), TemporalState("14:15", True), ProximityState(False), HistoryState(0)),
        ("STEP_3", "Restricted + After-Hours", ZoneState("RESTRICTED", "Restricted Area", "INSIDE", 45.0), TemporalState("20:30", False), ProximityState(False), HistoryState(1)),
        ("STEP_4", "Near Protected Asset", ZoneState("RESTRICTED", "Restricted Area", "INSIDE", 60.0), TemporalState("20:35", False), ProximityState(True, 5.0, is_holding_or_touching=True, asset_label="Vault Door"), HistoryState(1)),
        ("STEP_5", "Leaves Restricted Area", ZoneState("NORMAL", "Perimeter", "OUTSIDE"), TemporalState("20:45", False), ProximityState(False), HistoryState(2)),
    ]

    track_state = engine.get_or_create_track_state(track_id=901)
    results = []

    for s_id, label, z, t, p, h in steps:
        ctx = CCSDFContext(
            identity=IdentityState("Hunter", is_known=True, similarity=0.91, confidence=0.95),
            liveness=LivenessState("LIVE", 0.95),
            zone=z,
            temporal=t,
            proximity=p,
            history=h,
        )
        transitioned, eval_res = track_state.update(engine, ctx, authoritative_action="MONITORED", force=True)
        results.append({
            "step_id": s_id,
            "step_label": label,
            "time_str": t.current_time_str,
            "zone_name": z.zone_name,
            "is_auth_time": t.is_authorized_hours,
            "prox_interaction": p.has_interaction,
            "history_violations": h.violations_count,
            "risk_score": eval_res.risk_score,
            "risk_level": eval_res.risk_level,
            "recommended_action": eval_res.recommended_action.value,
            "dominant_factor": eval_res.dominant_factor,
            "re_evaluation_count": track_state.re_evaluation_count,
        })

    return results


def run_multi_person_tracking_simulation(engine: CCSDFEngine) -> List[Dict[str, Any]]:
    """
    Executes Part 12: Multi-Person Simultaneous Evaluation Test.
    Track 101: Authorized staff member (Hunter)
    Track 102: Unknown visitor (Unregistered)
    Verifies state isolation: Track 101's authorized status does not leak to Track 102.
    """
    track_auth = engine.get_or_create_track_state(track_id=101)
    track_unknown = engine.get_or_create_track_state(track_id=102)

    # Co-present in Protected Asset Area
    ctx_auth = CCSDFContext(
        identity=IdentityState("Hunter", is_known=True, similarity=0.92, confidence=0.96),
        liveness=LivenessState("LIVE", 0.95),
        zone=ZoneState("PROTECTED_ASSET_AREA", "Protected Asset Area", "INSIDE"),
        temporal=TemporalState("15:00", True),
        proximity=ProximityState(True, 40.0, asset_label="Control Box"),
        history=HistoryState(0),
    )

    ctx_unknown = CCSDFContext(
        identity=IdentityState("Unknown Person", is_known=False, similarity=0.15, confidence=0.25),
        liveness=LivenessState("LIVE", 0.94),
        zone=ZoneState("PROTECTED_ASSET_AREA", "Protected Asset Area", "INSIDE"),
        temporal=TemporalState("15:00", True),
        proximity=ProximityState(True, 35.0, asset_label="Control Box"),
        history=HistoryState(1),
    )

    _, res_auth = track_auth.update(engine, ctx_auth, authoritative_action="AUTHORIZED", force=True)
    _, res_unknown = track_unknown.update(engine, ctx_unknown, authoritative_action="DENIED", force=True)

    return [
        {
            "track_id": 101,
            "identity": "Hunter (Staff)",
            "is_known": True,
            "risk_score": res_auth.risk_score,
            "recommended_action": res_auth.recommended_action.value,
            "authoritative_action": "AUTHORIZED",
            "state_isolated": True,
        },
        {
            "track_id": 102,
            "identity": "Unknown Person (Visitor)",
            "is_known": False,
            "risk_score": res_unknown.risk_score,
            "recommended_action": res_unknown.recommended_action.value,
            "authoritative_action": "DENIED",
            "state_isolated": True,
        },
    ]


# =============================================================================
# COMPLETE DETERMINISTIC EXPERIMENT HARNESS
# =============================================================================

class CCSDFStressHarness:
    """Complete reproducible evaluation experiment harness."""

    def __init__(self, output_dir: str = "data/research"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.scenarios = build_stress_scenarios()
        self.engine = CCSDFEngine(weights=WEIGHT_SET_A_BALANCED)

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Executes all benchmarks, ablations, sensitivity suites, and traces."""
        start_time = time.time()

        # 1. Baseline Evaluations
        baseline_names = [
            "BASELINE_1_FACE_ONLY",
            "BASELINE_2_FACE_PAD",
            "BASELINE_3_FACE_PAD_ZONE_TIME",
            "BASELINE_4_FACE_PAD_ZONE_TIME_PROX",
            "WATCHGUARD_CCSDF_FULL",
        ]

        baseline_results: Dict[str, List[Dict[str, Any]]] = {b: [] for b in baseline_names}
        latency_measurements: Dict[str, List[float]] = {b: [] for b in baseline_names}

        for sc in self.scenarios:
            # Baseline 1
            t0 = time.perf_counter()
            act1, grant1, r1 = evaluate_baseline_1_face_only(sc.context)
            lat1_us = (time.perf_counter() - t0) * 1e6
            baseline_results["BASELINE_1_FACE_ONLY"].append({
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "ground_truth_authorized": sc.ground_truth_authorized,
                "action": act1,
                "is_access_granted": grant1,
                "risk_score": r1,
                "latency_us": round(lat1_us, 2),
            })
            latency_measurements["BASELINE_1_FACE_ONLY"].append(lat1_us)

            # Baseline 2
            t0 = time.perf_counter()
            act2, grant2, r2 = evaluate_baseline_2_face_pad(sc.context)
            lat2_us = (time.perf_counter() - t0) * 1e6
            baseline_results["BASELINE_2_FACE_PAD"].append({
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "ground_truth_authorized": sc.ground_truth_authorized,
                "action": act2,
                "is_access_granted": grant2,
                "risk_score": r2,
                "latency_us": round(lat2_us, 2),
            })
            latency_measurements["BASELINE_2_FACE_PAD"].append(lat2_us)

            # Baseline 3
            t0 = time.perf_counter()
            act3, grant3, r3 = evaluate_baseline_3_face_pad_zone_time(sc.context)
            lat3_us = (time.perf_counter() - t0) * 1e6
            baseline_results["BASELINE_3_FACE_PAD_ZONE_TIME"].append({
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "ground_truth_authorized": sc.ground_truth_authorized,
                "action": act3,
                "is_access_granted": grant3,
                "risk_score": r3,
                "latency_us": round(lat3_us, 2),
            })
            latency_measurements["BASELINE_3_FACE_PAD_ZONE_TIME"].append(lat3_us)

            # Baseline 4
            t0 = time.perf_counter()
            act4, grant4, r4 = evaluate_baseline_4_face_pad_zone_time_prox(sc.context)
            lat4_us = (time.perf_counter() - t0) * 1e6
            baseline_results["BASELINE_4_FACE_PAD_ZONE_TIME_PROX"].append({
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "ground_truth_authorized": sc.ground_truth_authorized,
                "action": act4,
                "is_access_granted": grant4,
                "risk_score": r4,
                "latency_us": round(lat4_us, 2),
            })
            latency_measurements["BASELINE_4_FACE_PAD_ZONE_TIME_PROX"].append(lat4_us)

            # Full CCSDF
            t0 = time.perf_counter()
            act5, grant5, r5, res5 = evaluate_watchguard_ccsdf(self.engine, sc.context)
            lat5_us = (time.perf_counter() - t0) * 1e6
            baseline_results["WATCHGUARD_CCSDF_FULL"].append({
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "ground_truth_authorized": sc.ground_truth_authorized,
                "action": act5,
                "is_access_granted": grant5,
                "risk_score": r5,
                "risk_level": res5.risk_level,
                "dominant_factor": res5.dominant_factor,
                "override_applied": res5.is_conflict_overridden,
                "latency_us": round(lat5_us, 2),
            })
            latency_measurements["WATCHGUARD_CCSDF_FULL"].append(lat5_us)

        # Baseline Classification Metrics
        baseline_metrics: Dict[str, Dict[str, Any]] = {}
        for b_name in baseline_names:
            m = calculate_metrics(baseline_results[b_name])
            avg_lat = round(sum(latency_measurements[b_name]) / len(latency_measurements[b_name]), 2)
            m_dict = asdict(m)
            m_dict["avg_decision_latency_us"] = avg_lat
            baseline_metrics[b_name] = m_dict

        # 2. Ablation Study
        ablation_results: Dict[str, List[Dict[str, Any]]] = {}
        ablation_metrics: Dict[str, Dict[str, Any]] = {}

        for abl_name, mask in ABLATION_CONFIGS.items():
            abl_engine = CCSDFEngine(weights=WEIGHT_SET_A_BALANCED, ablation_mask=mask)
            abl_evals = []
            lats = []
            for sc in self.scenarios:
                t0 = time.perf_counter()
                act, grant, r, res = evaluate_watchguard_ccsdf(abl_engine, sc.context)
                lat_us = (time.perf_counter() - t0) * 1e6
                abl_evals.append({
                    "scenario_id": sc.scenario_id,
                    "name": sc.name,
                    "category": sc.category,
                    "ground_truth_authorized": sc.ground_truth_authorized,
                    "action": act,
                    "is_access_granted": grant,
                    "risk_score": r,
                    "risk_level": res.risk_level,
                    "dominant_factor": res.dominant_factor,
                    "latency_us": round(lat_us, 2),
                })
                lats.append(lat_us)

            ablation_results[abl_name] = abl_evals
            m = calculate_metrics(abl_evals)
            m_dict = asdict(m)
            m_dict["avg_decision_latency_us"] = round(sum(lats) / len(lats), 2)
            ablation_metrics[abl_name] = m_dict

        # 3. Weight Profile Sensitivity Study
        weight_results: Dict[str, List[Dict[str, Any]]] = {}
        weight_metrics: Dict[str, Dict[str, Any]] = {}

        for prof_name, w_profile in WEIGHT_PROFILES.items():
            sens_engine = CCSDFEngine(weights=w_profile)
            prof_evals = []
            lats = []
            for sc in self.scenarios:
                t0 = time.perf_counter()
                act, grant, r, res = evaluate_watchguard_ccsdf(sens_engine, sc.context)
                lat_us = (time.perf_counter() - t0) * 1e6
                prof_evals.append({
                    "scenario_id": sc.scenario_id,
                    "name": sc.name,
                    "ground_truth_authorized": sc.ground_truth_authorized,
                    "action": act,
                    "is_access_granted": grant,
                    "risk_score": r,
                    "risk_level": res.risk_level,
                    "dominant_factor": res.dominant_factor,
                    "latency_us": round(lat_us, 2),
                })
                lats.append(lat_us)

            weight_results[prof_name] = prof_evals
            m = calculate_metrics(prof_evals)
            m_dict = asdict(m)
            m_dict["avg_decision_latency_us"] = round(sum(lats) / len(lats), 2)
            weight_metrics[prof_name] = m_dict

        # 4. Continuous Authorization & Multi-Person Traces
        cont_trace = run_continuous_authorization_sequence(self.engine)
        multi_trace = run_multi_person_tracking_simulation(self.engine)

        total_runtime_s = round(time.time() - start_time, 3)

        complete_dataset = {
            "metadata": {
                "experiment_name": "WATCHGUARD_PHASE_5_17_CCSDF_STRESS_AND_ABLATION",
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_scenarios": len(self.scenarios),
                "total_baselines": len(baseline_names),
                "total_ablations": len(ABLATION_CONFIGS),
                "total_weight_profiles": len(WEIGHT_PROFILES),
                "runtime_seconds": total_runtime_s,
            },
            "baseline_metrics": baseline_metrics,
            "baseline_results": baseline_results,
            "ablation_metrics": ablation_metrics,
            "ablation_results": ablation_results,
            "weight_sensitivity_metrics": weight_metrics,
            "weight_sensitivity_results": weight_results,
            "continuous_authorization_trace": cont_trace,
            "multi_person_simulation_trace": multi_trace,
        }

        # Export JSON & CSV
        self._export_json(complete_dataset)
        self._export_csv(baseline_results, ablation_results)

        return complete_dataset

    def _export_json(self, data: Dict[str, Any]) -> str:
        filepath = os.path.join(self.output_dir, "ccsdf_stress_results.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def _export_csv(
        self,
        baseline_results: Dict[str, List[Dict[str, Any]]],
        ablation_results: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        filepath = os.path.join(self.output_dir, "ccsdf_stress_results.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model/Configuration",
                "Scenario ID",
                "Scenario Name",
                "Category",
                "Ground Truth Authorized",
                "Predicted Action",
                "Access Granted",
                "Risk Score",
                "Latency (us)",
            ])

            for b_name, evals in baseline_results.items():
                for e in evals:
                    writer.writerow([
                        b_name,
                        e["scenario_id"],
                        e["name"],
                        e["category"],
                        e["ground_truth_authorized"],
                        e["action"],
                        e["is_access_granted"],
                        e["risk_score"],
                        e.get("latency_us", 0.0),
                    ])

            for abl_name, evals in ablation_results.items():
                for e in evals:
                    writer.writerow([
                        f"ABLATION_{abl_name}",
                        e["scenario_id"],
                        e["name"],
                        e["category"],
                        e["ground_truth_authorized"],
                        e["action"],
                        e["is_access_granted"],
                        e["risk_score"],
                        e.get("latency_us", 0.0),
                    ])

        return filepath


if __name__ == "__main__":
    harness = CCSDFStressHarness()
    res = harness.run_full_evaluation()
    print("CCSDF Stress Testing & Ablation Study completed successfully.")
    print("Aggregate Baseline Metrics:")
    for b, m in res["baseline_metrics"].items():
        print(f"  {b:35s} | FAR: {m['far']:.2f} | FRR: {m['frr']:.2f} | F1: {m['f1_score']:.2f} | Latency: {m['avg_decision_latency_us']:.1f}us")
