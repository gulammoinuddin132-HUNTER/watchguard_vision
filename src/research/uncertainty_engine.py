"""
WatchGuard Vision - Uncertainty-Aware Decision Layer (Phase 5.20)
----------------------------------------------------------------
Computes multi-source uncertainty metrics across visual perception, identity recognition,
presentation attack detection (PAD), and spatial telemetry.
Enforces critical safety invariants preventing low confidence from granting false access.
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging

from config import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
)

logger = logging.getLogger("WatchGuardVision.UncertaintyEngine")


@dataclass
class DecisionConfidence:
    """Multi-dimensional confidence vector and tier classification."""
    identity_confidence: float = 1.0
    pad_confidence: float = 1.0
    object_confidence: float = 1.0
    zone_confidence: float = 1.0
    behavior_confidence: float = 1.0
    overall_confidence: float = 1.0
    confidence_tier: str = "HIGH"  # "HIGH", "MEDIUM", "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_confidence": round(self.identity_confidence, 3),
            "pad_confidence": round(self.pad_confidence, 3),
            "object_confidence": round(self.object_confidence, 3),
            "zone_confidence": round(self.zone_confidence, 3),
            "behavior_confidence": round(self.behavior_confidence, 3),
            "overall_confidence": round(self.overall_confidence, 3),
            "confidence_tier": self.confidence_tier,
        }


class UncertaintyAggregator:
    """
    Computes rigorous uncertainty and confidence scores for security decisions.
    Employs minimum-critical gating for primary biometric modalities (Identity & PAD)
    combined with weighted aggregation for secondary telemetry.
    """

    def __init__(
        self,
        high_threshold: float = CONFIDENCE_HIGH_THRESHOLD,
        low_threshold: float = CONFIDENCE_LOW_THRESHOLD,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate_confidence(
        self,
        identity_confidence: float = 1.0,
        pad_confidence: float = 1.0,
        object_confidence: float = 1.0,
        zone_confidence: float = 1.0,
        behavior_confidence: float = 1.0,
    ) -> DecisionConfidence:
        """
        Aggregates multi-source confidence values into a unified DecisionConfidence.
        
        Aggregation Strategy:
          1. Biometric Gating: C_bio = sqrt(C_id * C_pad)
          2. Telemetry Weighted Average: C_tele = 0.40 * C_obj + 0.35 * C_zone + 0.25 * C_beh
          3. Overall Confidence: C_overall = min(C_bio, 0.70 * C_bio + 0.30 * C_tele)
        """
        c_id = max(0.0, min(1.0, float(identity_confidence)))
        c_pad = max(0.0, min(1.0, float(pad_confidence)))
        c_obj = max(0.0, min(1.0, float(object_confidence)))
        c_zone = max(0.0, min(1.0, float(zone_confidence)))
        c_beh = max(0.0, min(1.0, float(behavior_confidence)))

        # 1. Biometric Gating (Geometric mean ensures that a single collapsed modality constrains the upper bound)
        c_bio = math.sqrt(c_id * c_pad)

        # 2. Contextual Telemetry Weighted Combination
        c_tele = (0.40 * c_obj) + (0.35 * c_zone) + (0.25 * c_beh)

        # 3. Overall Gated Fusion
        overall = min(c_bio, (0.70 * c_bio) + (0.30 * c_tele))
        overall = round(max(0.0, min(1.0, overall)), 4)

        # Classify Tier
        if overall >= self.high_threshold:
            tier = "HIGH"
        elif overall >= self.low_threshold:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return DecisionConfidence(
            identity_confidence=c_id,
            pad_confidence=c_pad,
            object_confidence=c_obj,
            zone_confidence=c_zone,
            behavior_confidence=c_beh,
            overall_confidence=overall,
            confidence_tier=tier,
        )

    def apply_uncertainty_action_guards(
        self,
        recommended_action: str,
        confidence: DecisionConfidence,
        is_known_identity: bool,
        is_live_pad: bool,
        authoritative_action: str,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Enforces Critical Safety Rules:
          - PAD Failure -> NEVER authorize (forces DENY).
          - UNKNOWN identity -> NEVER become ALLOW solely due to low context risk.
          - Policy Denial -> Cannot become ALLOW.
          - Low Confidence -> PENDING_VERIFICATION (never silently ALLOW).
          - Medium Confidence -> ATTENTION (if previously ALLOW).
        
        Returns:
            (guarded_action, was_overridden, reason_string)
        """
        # Invariant 1: PAD Failure Guard
        if not is_live_pad:
            if recommended_action != "DENY":
                return "DENY", True, "PAD_LIVENESS_FAILURE_GUARD"

        # Invariant 2: Unknown Identity Guard
        if not is_known_identity:
            if recommended_action == "ALLOW":
                return "DENY", True, "UNREGISTERED_IDENTITY_GUARD"

        # Invariant 3: Authoritative Denial Guard
        if authoritative_action in ("DENIED", "DENY") and recommended_action == "ALLOW":
            return "DENY", True, "POLICY_AUTHORITY_DENIAL_GUARD"

        # Invariant 4: Low Confidence Uncertainty Guard
        if confidence.confidence_tier == "LOW":
            if recommended_action == "ALLOW":
                return "PENDING_VERIFICATION", True, f"LOW_CONFIDENCE_GUARD ({confidence.overall_confidence:.2f})"

        # Invariant 5: Medium Confidence Caution Guard
        if confidence.confidence_tier == "MEDIUM":
            if recommended_action == "ALLOW":
                return "ATTENTION", True, f"MEDIUM_CONFIDENCE_CAUTION ({confidence.overall_confidence:.2f})"

        return recommended_action, False, None
