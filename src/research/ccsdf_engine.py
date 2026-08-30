"""
WatchGuard Vision - Continuous Contextual Security Decision Fusion (CCSDF) Engine
---------------------------------------------------------------------------------
Implements the research-layer continuous fusion model combining:
  1. Identity (S_id)
  2. Liveness / PAD (S_pad)
  3. Spatial Zone (S_zone)
  4. Temporal Policy (S_time)
  5. Asset Proximity (S_prox)
  6. Historical Violations (S_hist)

Enforces mathematical normalization into [0, 1], conflict/safety override guards,
track-persistent continuous authorization state machines, and human-auditable
explanation traces without destabilizing the authoritative deterministic engine.
"""

import time
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from src.research.ccsdf_config import (
    CCSDFWeights,
    CCSDFThresholds,
    CCSDFAction,
    WEIGHT_SET_A_BALANCED,
    ABLATION_CONFIGS,
)
from src.research.uncertainty_engine import DecisionConfidence, UncertaintyAggregator


# =============================================================================
# 1. STRUCTURED CONTEXTUAL STATES & NORMALIZATION (Parts 2 & 3)
# =============================================================================

@dataclass
class IdentityState:
    """Identity recognition signal state."""
    name: str = "Unknown Person"
    is_known: bool = False
    similarity: float = 0.0
    confidence: float = 0.0
    role: str = "Visitor"
    similarity_threshold: float = 0.70

    def get_normalized_risk(self) -> float:
        """
        Normalized Identity Risk S_id in [0.0, 1.0].
        0.0 = Verified known authorized identity meeting or exceeding match threshold.
        1.0 = Unregistered / completely unknown visitor or mismatched identity.
        """
        if not self.is_known:
            return 1.0
        if self.name in ("Unregistered", "None", ""):
            return 1.0
        if self.similarity >= self.similarity_threshold:
            return 0.0
        else:
            # Sub-threshold known face: elevated risk
            ratio = self.similarity / max(1e-4, self.similarity_threshold)
            return min(1.0, round(1.0 - (0.5 * ratio), 4))


@dataclass
class LivenessState:
    """Presentation attack detection (PAD) signal state."""
    liveness_state: str = "LIVE"  # "LIVE", "WARMUP", "SPOOF"
    confidence: float = 0.90
    attack_type: Optional[str] = None

    def get_normalized_risk(self) -> float:
        """
        Normalized PAD Risk S_pad in [0.0, 1.0].
        0.0 = Confirmed natural live subject.
        0.5 = Verification in progress (warmup / eye-blink accumulating).
        1.0 = Confirmed presentation attack (photo/video replay/spoof).
        """
        state_upper = (self.liveness_state or "LIVE").upper()
        if state_upper == "SPOOF":
            return 1.0
        elif state_upper in ("WARMUP", "VERIFYING", "PENDING"):
            return 0.5
        elif state_upper == "LIVE":
            # Small residual risk if confidence is low
            return max(0.0, round(1.0 - min(1.0, self.confidence), 4))
        return 0.5


@dataclass
class ZoneState:
    """Spatial surveillance zone occupancy state."""
    zone_id: str = "NORMAL"  # "NORMAL", "PROTECTED_ASSET_AREA", "RESTRICTED"
    zone_name: str = "Standard Perimeter"
    occupancy: str = "OUTSIDE"  # "INSIDE", "OUTSIDE"
    duration_seconds: float = 0.0

    def __post_init__(self):
        if "RESTRICTED" in self.zone_id.upper() and self.zone_name == "Standard Perimeter":
            self.zone_name = "Restricted Area"
        elif ("PROTECTED" in self.zone_id.upper() or "ASSET" in self.zone_id.upper()) and self.zone_name == "Standard Perimeter":
            self.zone_name = "Protected Asset Area"

    def get_normalized_risk(self) -> float:
        """
        Normalized Zone Risk S_zone in [0.0, 1.0].
        0.0 = Standard perimeter / public lobby / unrestricted zone.
        0.5 = Semi-restricted protected asset zone (e.g. equipment bench).
        1.0 = Strictly restricted security area (e.g. server enclosure / vault).
        """
        z_upper = (self.zone_id or "NORMAL").upper()
        if "RESTRICTED" in z_upper:
            return 1.0
        elif "PROTECTED" in z_upper or "ASSET" in z_upper:
            return 0.5
        return 0.0


@dataclass
class TemporalState:
    """Temporal operating policy state."""
    current_time_str: str = "--:--"
    is_authorized_hours: bool = True
    operating_window: str = "09:00–18:00"

    def get_normalized_risk(self) -> float:
        """
        Normalized Temporal Risk S_time in [0.0, 1.0].
        0.0 = Within authorized operational schedule window.
        1.0 = Outside authorized schedule window (after-hours / holiday).
        """
        return 0.0 if self.is_authorized_hours else 1.0


@dataclass
class ProximityState:
    """Protected asset spatial interaction state."""
    has_interaction: bool = False
    nearest_distance_pixels: float = 999.0
    proximity_threshold_pixels: float = 120.0
    asset_label: Optional[str] = None
    is_holding_or_touching: bool = False

    def get_normalized_risk(self) -> float:
        """
        Normalized Proximity Risk S_prox in [0.0, 1.0].
        0.0 = Safe perimeter separation (> proximity threshold) or no protected asset.
        (0.0, 1.0) = Distance-inversely scaled proximity risk.
        1.0 = Direct physical contact / holding / tampering.
        """
        if self.is_holding_or_touching:
            return 1.0
        if not self.has_interaction or self.nearest_distance_pixels >= self.proximity_threshold_pixels:
            return 0.0
        # Inversely proportional to pixel distance: 0px -> 1.0, 120px -> 0.0
        ratio = max(0.0, min(1.0, self.nearest_distance_pixels / max(1.0, self.proximity_threshold_pixels)))
        return round(1.0 - ratio, 4)


@dataclass
class HistoryState:
    """Historical security non-compliance and violation tracking state."""
    violations_count: int = 0
    repeated_entries_count: int = 0
    violation_types: List[str] = field(default_factory=list)

    def get_normalized_risk(self) -> float:
        """
        Normalized History Risk S_hist in [0.0, 1.0].
        0.0 = Pristine security record (0 prior violations).
        Saturates at 1.0 when cumulative violation count >= 3.
        """
        return min(1.0, round(self.violations_count / 3.0, 4))


@dataclass
class SmallAssetState:
    """Small asset security signal state (Phase 5.19D)."""
    has_detected_asset: bool = False
    asset_label: Optional[str] = None
    security_category: str = "context-relevant"
    confidence: float = 0.0
    is_confirmed: bool = True

    def get_normalized_risk(self) -> float:
        """
        Normalized Small Asset Risk S_asset in [0.0, 1.0].
        0.0 = Standard context or no high-risk asset detected.
        0.3 = Security-relevant asset (phone).
        0.5 = Protected asset (laptop).
        0.7 = Credential asset (access badge / keys).
        1.0 = High-risk asset (USB drive in sensitive zone).
        """
        if not self.has_detected_asset or not self.is_confirmed:
            return 0.0
        if self.security_category == "high-risk-asset":
            return 1.0
        elif self.security_category in ("credential", "physical-access"):
            return 0.70
        elif self.security_category == "protected":
            return 0.50
        elif self.security_category == "security-relevant":
            return 0.30
        return 0.0


@dataclass
class BehaviorState:
    """Behavioral pattern and spatial telemetry state (Phase 5.20)."""
    s_behavior: float = 0.0
    status: str = "NORMAL_ACTIVITY"  # "NORMAL_ACTIVITY", "UNUSUAL_PATTERN", "BEHAVIOR_ANOMALY"
    dwell_seconds: float = 0.0
    approach_count: int = 0
    restricted_retries: int = 0

    def get_normalized_risk(self) -> float:
        """Normalized Behavioral Risk S_behavior in [0.0, 1.0]."""
        return max(0.0, min(1.0, float(self.s_behavior)))


@dataclass
class TailgatingState:
    """Tailgating and multi-person piggybacking state (Phase 5.20)."""
    is_tailgating_suspected: bool = False
    s_tailgating: float = 0.0
    status: str = "NONE"  # "NONE", "TAILGATING_SUSPECTED"
    confidence: float = 0.0
    follower_identity: Optional[str] = None
    authorized_identity: Optional[str] = None
    spatial_distance: float = 999.0

    def get_normalized_risk(self) -> float:
        """Normalized Tailgating Risk S_tailgating in [0.0, 1.0]."""
        return max(0.0, min(1.0, float(self.s_tailgating)))


# =============================================================================
# 2. CONTEXT AGGREGATOR & EXPLANATION TRACE (Parts 2 & 9 & Phase 5.20)
# =============================================================================

@dataclass
class CCSDFContext:
    """Aggregated multi-dimensional contextual input vector."""
    identity: IdentityState = field(default_factory=IdentityState)
    liveness: LivenessState = field(default_factory=LivenessState)
    zone: ZoneState = field(default_factory=ZoneState)
    temporal: TemporalState = field(default_factory=TemporalState)
    proximity: ProximityState = field(default_factory=ProximityState)
    history: HistoryState = field(default_factory=HistoryState)
    small_asset: SmallAssetState = field(default_factory=SmallAssetState)
    behavior: BehaviorState = field(default_factory=BehaviorState)
    tailgating: TailgatingState = field(default_factory=TailgatingState)

    def compute_context_hash(self) -> str:
        """
        Computes a deterministic hash of meaningful context attributes.
        Used for event-driven continuous re-evaluation (avoiding per-frame re-compute).
        """
        raw = (
            f"{self.identity.name}_{self.identity.is_known}_{round(self.identity.similarity, 2)}|"
            f"{self.liveness.liveness_state}_{round(self.liveness.confidence, 2)}|"
            f"{self.zone.zone_id}_{self.zone.occupancy}|"
            f"{self.temporal.is_authorized_hours}|"
            f"{self.proximity.has_interaction}_{self.proximity.is_holding_or_touching}_{round(self.proximity.nearest_distance_pixels, -1)}|"
            f"{self.history.violations_count}|"
            f"{self.small_asset.has_detected_asset}_{self.small_asset.security_category}|"
            f"{round(self.behavior.s_behavior, 2)}_{self.behavior.status}|"
            f"{self.tailgating.is_tailgating_suspected}"
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_normalized_signals(self) -> Dict[str, float]:
        """Returns dict of all normalized risk signals in [0.0, 1.0]."""
        return {
            "identity": self.identity.get_normalized_risk(),
            "pad": self.liveness.get_normalized_risk(),
            "zone": self.zone.get_normalized_risk(),
            "time": self.temporal.get_normalized_risk(),
            "proximity": self.proximity.get_normalized_risk(),
            "history": self.history.get_normalized_risk(),
            "asset": self.small_asset.get_normalized_risk(),
            "behavior": self.behavior.get_normalized_risk(),
        }


@dataclass
class CCSDFExplanationTrace:
    """Human-auditable explanation trace for a CCSDF evaluation."""
    who: str
    identity_summary: str
    liveness_summary: str
    zone_summary: str
    time_summary: str
    proximity_summary: str
    history_summary: str
    risk_score: float
    risk_level: str
    recommended_action: str
    authoritative_action: str
    dominant_factor: str
    safety_override_applied: bool = False
    override_reason: Optional[str] = None
    behavior_summary: str = "Normal activity"
    tailgating_summary: str = "No tailgating detected"
    confidence_summary: str = "High (1.00)"

    def to_string(self) -> str:
        override_str = f" [SAFETY OVERRIDE: {self.override_reason}]" if self.safety_override_applied else ""
        return (
            f"WHO: {self.who}\n"
            f"IDENTITY: {self.identity_summary}\n"
            f"LIVENESS: {self.liveness_summary}\n"
            f"ZONE: {self.zone_summary}\n"
            f"TIME: {self.time_summary}\n"
            f"PROXIMITY: {self.proximity_summary}\n"
            f"HISTORY: {self.history_summary}\n"
            f"BEHAVIOR: {self.behavior_summary}\n"
            f"TAILGATING: {self.tailgating_summary}\n"
            f"CONFIDENCE: {self.confidence_summary}\n"
            f"CCSDF RISK: {self.risk_score:.2f} ({self.risk_level})\n"
            f"DOMINANT FACTOR: {self.dominant_factor}\n"
            f"RECOMMENDED: {self.recommended_action}{override_str}\n"
            f"AUTHORITATIVE: {self.authoritative_action}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "who": self.who,
            "identity_summary": self.identity_summary,
            "liveness_summary": self.liveness_summary,
            "zone_summary": self.zone_summary,
            "time_summary": self.time_summary,
            "proximity_summary": self.proximity_summary,
            "history_summary": self.history_summary,
            "behavior_summary": self.behavior_summary,
            "tailgating_summary": self.tailgating_summary,
            "confidence_summary": self.confidence_summary,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
            "authoritative_action": self.authoritative_action,
            "dominant_factor": self.dominant_factor,
            "safety_override_applied": self.safety_override_applied,
            "override_reason": self.override_reason,
        }


@dataclass
class CCSDFEvaluationResult:
    """Complete structured output of a CCSDF risk fusion computation."""
    risk_score: float
    risk_level: str
    recommended_action: CCSDFAction
    authoritative_action: str
    normalized_signals: Dict[str, float]
    weights_applied: Dict[str, float]
    trace: CCSDFExplanationTrace
    is_conflict_overridden: bool = False
    override_reason: Optional[str] = None
    dominant_factor: str = "None"
    confidence: Optional[Any] = None
    confidence_tier: str = "HIGH"
    behavior_status: str = "NORMAL_ACTIVITY"
    tailgating_status: str = "NONE"
    timestamp_iso: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action.value,
            "authoritative_action": self.authoritative_action,
            "normalized_signals": self.normalized_signals,
            "weights_applied": self.weights_applied,
            "trace": self.trace.to_dict(),
            "is_conflict_overridden": self.is_conflict_overridden,
            "override_reason": self.override_reason,
            "dominant_factor": self.dominant_factor,
            "timestamp_iso": self.timestamp_iso,
        }


# =============================================================================
# 3. CONTINUOUS TRACK-PERSISTENT AUTHORIZATION STATE (Part 6)
# =============================================================================

class TrackCCSDFState:
    """
    Persistent runtime contextual authorization state for a single tracked face.
    Maintains continuous authorization trust over time with change detection.
    """

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        self.last_context_hash: Optional[str] = None
        self.latest_result: Optional[CCSDFEvaluationResult] = None
        self.evaluation_history: List[CCSDFEvaluationResult] = []
        self.re_evaluation_count: int = 0
        self.state_since: float = time.time()
        self.last_updated: float = time.time()

    def update(
        self,
        engine: "CCSDFEngine",
        context: CCSDFContext,
        authoritative_action: str,
        force: bool = False,
    ) -> Tuple[bool, CCSDFEvaluationResult]:
        """
        Evaluates context only when a meaningful contextual transition occurs.
        Returns: (has_transitioned, evaluation_result)
        """
        current_hash = context.compute_context_hash()
        now = time.time()

        if not force and current_hash == self.last_context_hash and self.latest_result is not None:
            self.last_updated = now
            return False, self.latest_result

        # Meaningful context change detected: Execute CCSDF fusion
        result = engine.evaluate_context(context, authoritative_action=authoritative_action)

        self.last_context_hash = current_hash
        self.latest_result = result
        self.last_updated = now
        self.re_evaluation_count += 1

        # Keep rolling history of last 20 evaluations
        self.evaluation_history.append(result)
        if len(self.evaluation_history) > 20:
            self.evaluation_history.pop(0)

        return True, result


# =============================================================================
# 4. CORE CCSDF RISK FUSION ENGINE (Parts 4 & 5)
# =============================================================================

class CCSDFEngine:
    """
    Continuous Contextual Security Decision Fusion Engine.
    Executes parameterized weighted risk fusion, conflict override safety guards,
    and explanation trace generation without altering authoritative decisions.
    """

    def __init__(
        self,
        weights: Optional[CCSDFWeights] = None,
        thresholds: Optional[CCSDFThresholds] = None,
        ablation_mask: Optional[Dict[str, bool]] = None,
    ):
        self.weights: CCSDFWeights = weights or WEIGHT_SET_A_BALANCED
        self.thresholds: CCSDFThresholds = thresholds or CCSDFThresholds()
        self.ablation_mask: Dict[str, bool] = ablation_mask or ABLATION_CONFIGS["FULL_CCSDF"]
        self._tracks: Dict[int, TrackCCSDFState] = {}

    def set_weights(self, weights: CCSDFWeights) -> None:
        """Updates active weighting profile."""
        self.weights = weights

    def set_ablation_mask(self, mask: Dict[str, bool]) -> None:
        """Applies feature ablation mask (Part 15)."""
        self.ablation_mask = mask

    def get_or_create_track_state(self, track_id: int) -> TrackCCSDFState:
        """Retrieves or creates isolated track state."""
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackCCSDFState(track_id=track_id)
        return self._tracks[track_id]

    def evaluate_context(
        self,
        context: CCSDFContext,
        authoritative_action: str = "DENIED",
    ) -> CCSDFEvaluationResult:
        """
        Executes CCSDF weighted fusion formula:
          R(t) = sum(w_i * S_i)
        Applies strict conflict override safety guards and returns evaluation result.
        """
        signals = context.get_normalized_signals()
        w_dict = self.weights.to_dict()

        # Apply ablation mask (zero-out disabled signals and re-normalize weights)
        active_weights: Dict[str, float] = {}
        total_active_weight = 0.0
        for k in signals.keys():
            w_val = w_dict.get(k, 0.0)
            if self.ablation_mask.get(k, True):
                active_weights[k] = w_val
                total_active_weight += w_val
            else:
                active_weights[k] = 0.0

        # Normalize active weights to sum to 1.0
        if total_active_weight > 0:
            for k in active_weights:
                active_weights[k] /= total_active_weight

        # Compute weighted risk sum
        raw_risk = 0.0
        weighted_contributions: Dict[str, float] = {}
        for k, val in signals.items():
            contrib = active_weights.get(k, 0.0) * val
            weighted_contributions[k] = contrib
            raw_risk += contrib

        raw_risk = round(min(1.0, max(0.0, raw_risk)), 4)

        # Identify dominant risk factor
        dominant_factor = "None"
        max_contrib = -1.0
        for k, contrib in weighted_contributions.items():
            if contrib > max_contrib and contrib > 0.01:
                max_contrib = contrib
                dominant_factor = k

        # Initial recommendation based on continuous risk thresholds
        if raw_risk < self.thresholds.allow_threshold:
            recommended = CCSDFAction.ALLOW
            risk_level = "GREEN"
        elif raw_risk < self.thresholds.attention_threshold:
            recommended = CCSDFAction.ATTENTION
            risk_level = "YELLOW"
        elif raw_risk < self.thresholds.restrict_threshold:
            recommended = CCSDFAction.RESTRICT
            risk_level = "ORANGE"
        else:
            recommended = CCSDFAction.DENY
            risk_level = "RED"

        # ---------------------------------------------------------------------
        # 5. CONFLICT / SAFETY OVERRIDE RULES (Part 5 & Phase 5.20 Invariants)
        # ---------------------------------------------------------------------
        override_applied = False
        override_reason = None

        # Guard 1: Presentation Attack / Spoof Guard
        if signals["pad"] >= 0.8:
            recommended = CCSDFAction.DENY
            risk_level = "RED"
            override_applied = True
            override_reason = "CRITICAL: Presentation attack detected (Liveness failure override)"

        # Guard 2: Liveness Uncertainty Guard (WARMUP / Verifying)
        elif signals["pad"] == 0.5 and recommended == CCSDFAction.ALLOW:
            recommended = CCSDFAction.PENDING_VERIFICATION
            risk_level = "YELLOW"
            override_applied = True
            override_reason = "Liveness verification in progress (Warmup hold)"

        # Guard 3: Unregistered Person Guard
        elif signals["identity"] >= 1.0 and recommended == CCSDFAction.ALLOW:
            recommended = CCSDFAction.ATTENTION
            risk_level = "YELLOW"
            override_applied = True
            override_reason = "Unregistered visitor cannot be granted ALLOW (Identity guard override)"

        # Guard 4: Authoritative Baseline Ceiling Guard (Never override deterministic DENIED)
        auth_upper = (authoritative_action or "DENIED").upper()
        if "DENIED" in auth_upper or "DENY" in auth_upper:
            if recommended in (CCSDFAction.ALLOW, CCSDFAction.ATTENTION) and (signals["time"] >= 1.0 or signals["zone"] >= 1.0 or signals["history"] >= 1.0):
                recommended = CCSDFAction.DENY
                override_applied = True
                override_reason = "Deterministic security engine authority overrides CCSDF score (Policy breach ceiling)"
            elif recommended == CCSDFAction.ALLOW:
                recommended = CCSDFAction.DENY
                override_applied = True
                override_reason = "Deterministic security engine authority overrides CCSDF score (Authority ceiling)"

        # ---------------------------------------------------------------------
        # 6. UNCERTAINTY AGGREGATION & CONFIDENCE GATING (Phase 5.20)
        # ---------------------------------------------------------------------
        uncertainty_engine = UncertaintyAggregator()
        id_conf = context.identity.confidence if context.identity.confidence > 0 else (1.0 if context.identity.is_known else 0.40)
        pad_conf = context.liveness.confidence if context.liveness.confidence > 0 else 1.0
        obj_conf = context.small_asset.confidence if context.small_asset.confidence > 0 else 1.0
        beh_conf = max(0.0, 1.0 - context.behavior.s_behavior)

        conf = uncertainty_engine.evaluate_confidence(
            identity_confidence=id_conf,
            pad_confidence=pad_conf,
            object_confidence=obj_conf,
            zone_confidence=1.0,
            behavior_confidence=beh_conf,
        )

        guarded_act_str, conf_override, conf_reason = uncertainty_engine.apply_uncertainty_action_guards(
            recommended_action=recommended.value,
            confidence=conf,
            is_known_identity=context.identity.is_known,
            is_live_pad=(context.liveness.liveness_state.upper() != "SPOOF"),
            authoritative_action=authoritative_action,
        )
        if conf_override:
            recommended = CCSDFAction(guarded_act_str)
            override_applied = True
            override_reason = conf_reason

        # Build Explanation Trace
        id_state = context.identity
        trace = CCSDFExplanationTrace(
            who=id_state.name,
            identity_summary=f"{'Known' if id_state.is_known else 'Unregistered'} (similarity={id_state.similarity:.2f}, role={id_state.role})",
            liveness_summary=f"{context.liveness.liveness_state} (confidence={context.liveness.confidence:.2f})",
            zone_summary=f"{context.zone.zone_name} [{context.zone.zone_id}]",
            time_summary=f"{'Permitted Window' if context.temporal.is_authorized_hours else 'AFTER-HOURS'} ({context.temporal.current_time_str})",
            proximity_summary=f"{'In Proximity' if context.proximity.has_interaction else 'Clear'} ({context.proximity.asset_label or 'None'}, d={context.proximity.nearest_distance_pixels:.1f}px)",
            history_summary=f"{context.history.violations_count} previous violations recorded",
            behavior_summary=f"{context.behavior.status} (S_beh={context.behavior.s_behavior:.2f})",
            tailgating_summary=f"{context.tailgating.status} (follower={context.tailgating.follower_identity or 'None'})",
            confidence_summary=f"{conf.confidence_tier} ({conf.overall_confidence:.2f})",
            risk_score=raw_risk,
            risk_level=risk_level,
            recommended_action=recommended.value,
            authoritative_action=authoritative_action,
            dominant_factor=dominant_factor,
            safety_override_applied=override_applied,
            override_reason=override_reason,
        )

        return CCSDFEvaluationResult(
            risk_score=raw_risk,
            risk_level=risk_level,
            recommended_action=recommended,
            authoritative_action=authoritative_action,
            normalized_signals=signals,
            weights_applied=active_weights,
            trace=trace,
            is_conflict_overridden=override_applied,
            override_reason=override_reason,
            dominant_factor=dominant_factor,
            confidence=conf,
            confidence_tier=conf.confidence_tier,
            behavior_status=context.behavior.status,
            tailgating_status=context.tailgating.status,
        )

    def evaluate_decision_context(
        self,
        decision: Any,
        primary_person: Optional[Dict[str, Any]],
        primary_track_state: Optional[Any],
        context_summary: Dict[str, Any],
        protected_interactions: List[Dict[str, Any]],
        is_auth_time: bool,
        time_str: str,
    ) -> CCSDFEvaluationResult:
        """
        Helper method to bridge ContextEngine's evaluate_context into CCSDF context vector.
        Does not mutate the authoritative SecurityDecision.
        """
        # 1. Identity State
        p = primary_person or {}
        p_name = p.get("name", "Unknown Person")
        p_known = bool(p.get("is_known", False))
        p_sim = float(p.get("similarity", 0.0))
        p_role = p.get("role", "Staff" if p_known else "Visitor")

        id_state = IdentityState(
            name=p_name,
            is_known=p_known,
            similarity=p_sim,
            confidence=float(p.get("confidence", 0.0)),
            role=p_role,
        )

        # 2. Liveness State
        p_spoof = bool(p.get("is_spoof", False))
        p_live_state = "SPOOF" if p_spoof else p.get("liveness_state", "LIVE")
        live_state = LivenessState(
            liveness_state=p_live_state,
            confidence=float(p.get("liveness_confidence", 0.90 if not p_spoof else 0.10)),
            attack_type=p.get("attack_type", "PRESENTATION_ATTACK" if p_spoof else None),
        )

        # 3. Zone State
        z_id = context_summary.get("zone", "NORMAL")
        z_name = context_summary.get("zone_name", "Standard Perimeter")
        occupancy = "INSIDE" if primary_track_state and getattr(primary_track_state, "zone_occupancy", None) and getattr(primary_track_state.zone_occupancy, "value", "") == "INSIDE" else "OUTSIDE"
        dur = getattr(primary_track_state, "duration_inside_zone_seconds", 0.0) if primary_track_state else 0.0

        zone_state = ZoneState(
            zone_id=z_id,
            zone_name=z_name,
            occupancy=occupancy,
            duration_seconds=dur,
        )

        # 4. Temporal State
        temp_state = TemporalState(
            current_time_str=time_str,
            is_authorized_hours=is_auth_time,
        )

        # 5. Proximity State
        has_int = len(protected_interactions) > 0
        min_dist = 999.0
        nearest_label = None
        is_touching = False
        if has_int:
            first_int = protected_interactions[0]
            min_dist = float(first_int.get("distance_pixels", 50.0))
            nearest_label = first_int.get("object_label")
            is_touching = min_dist <= 5.0

        prox_state = ProximityState(
            has_interaction=has_int,
            nearest_distance_pixels=min_dist,
            asset_label=nearest_label,
            is_holding_or_touching=is_touching,
        )

        # 6. History State
        v_count = len(getattr(primary_track_state, "distinct_violations", [])) if primary_track_state else 0
        r_entries = len(getattr(primary_track_state, "distinct_entries", [])) if primary_track_state else 0
        hist_state = HistoryState(
            violations_count=v_count,
            repeated_entries_count=r_entries,
        )

        # 7. Small Asset State (Phase 5.19D)
        asset_state = SmallAssetState(
            has_detected_asset=has_int,
            asset_label=nearest_label,
            security_category=context_summary.get("security_category", "protected" if has_int else "context-relevant"),
            confidence=0.85 if has_int else 0.0,
            is_confirmed=True,
        )

        # 8. Behavior State (Phase 5.20)
        beh_state = BehaviorState(
            s_behavior=float(context_summary.get("s_behavior", 0.0)),
            status=context_summary.get("behavior_status", "NORMAL_ACTIVITY"),
            dwell_seconds=dur,
            approach_count=len(protected_interactions),
            restricted_retries=r_entries,
        )

        # 9. Tailgating State (Phase 5.20)
        tail_state = TailgatingState(
            is_tailgating_suspected=bool(context_summary.get("is_tailgating_suspected", False)),
            s_tailgating=float(context_summary.get("s_tailgating", 0.0)),
            status=context_summary.get("tailgating_status", "NONE"),
            confidence=float(context_summary.get("tailgating_confidence", 0.0)),
            follower_identity=context_summary.get("follower_identity"),
            authorized_identity=context_summary.get("authorized_identity"),
            spatial_distance=float(context_summary.get("tailgating_distance", 999.0)),
        )

        context = CCSDFContext(
            identity=id_state,
            liveness=live_state,
            zone=zone_state,
            temporal=temp_state,
            proximity=prox_state,
            history=hist_state,
            small_asset=asset_state,
            behavior=beh_state,
            tailgating=tail_state,
        )

        auth_act = getattr(decision, "access_state", "DENIED")
        tid = p.get("track_id", 1)
        track_ccsdf = self.get_or_create_track_state(tid)
        _, eval_res = track_ccsdf.update(self, context, authoritative_action=auth_act)
        return eval_res
