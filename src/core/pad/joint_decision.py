"""
Joint Access Decision Evaluator (Identity ⊗ Liveness)
----------------------------------------------------
Combines facial identification (SFace) and Presentation-Attack Detection (PAD)
into a unified, deterministic security access evaluation.
Enforces the critical security rule: NO AUTHORIZATION BEFORE LIVENESS VERIFICATION.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, Optional

from src.core.pad.base_pad import AttackType, PADResult
from src.events.event_types import RiskLevel

logger = logging.getLogger("WatchGuardVision.JointDecision")


class JointStatus(str, Enum):
    """Combined Identity + Liveness authorization statuses."""
    AUTHORIZED = "AUTHORIZED"                                 # Known Identity + Bona-fide Live (ONLY STATE FOR ACCESS GRANTED)
    LIVENESS_VERIFYING = "LIVENESS_VERIFYING"                 # Identity Matched + Liveness Verifying (ACCESS PENDING)
    SPOOF_IMPERSONATION_REJECTED = "SPOOF_IMPERSONATION"     # Known Identity + Presentation Attack (CRITICAL RED)
    UNKNOWN_PERSON = "UNKNOWN_PERSON"                         # Unknown / Ambiguous + Bona-fide Live
    SPOOF_ATTACK_DETECTED = "SPOOF_ATTACK_DETECTED"           # Unknown / Ambiguous + Presentation Attack (HIGH RISK ORANGE)
    UNVERIFIED_AMBIGUOUS = "UNVERIFIED_AMBIGUOUS"             # Ambiguous match (Fail-secure YELLOW)
    UNVERIFIED_TIMEOUT = "UNVERIFIED_TIMEOUT"                 # PAD verification window (5s) timed out without live confirmation (Fail-secure YELLOW)

    @property
    def display_label(self) -> str:
        labels = {
            JointStatus.AUTHORIZED: "AUTHORIZED",
            JointStatus.LIVENESS_VERIFYING: "ACCESS PENDING",
            JointStatus.SPOOF_IMPERSONATION_REJECTED: "SPOOF REJECTED",
            JointStatus.UNKNOWN_PERSON: "UNKNOWN VISITOR",
            JointStatus.SPOOF_ATTACK_DETECTED: "SPOOF DETECTED",
            JointStatus.UNVERIFIED_AMBIGUOUS: "UNVERIFIED",
            JointStatus.UNVERIFIED_TIMEOUT: "ACCESS DENIED",
        }
        return labels.get(self, "STANDBY")

    @property
    def color_hex(self) -> str:
        colors = {
            JointStatus.AUTHORIZED: "#10B981",                   # Emerald Green
            JointStatus.LIVENESS_VERIFYING: "#F59E0B",           # Amber / Golden Yellow
            JointStatus.SPOOF_IMPERSONATION_REJECTED: "#EF4444", # Crimson Red
            JointStatus.UNKNOWN_PERSON: "#F59E0B",               # Amber
            JointStatus.SPOOF_ATTACK_DETECTED: "#F97316",        # Orange
            JointStatus.UNVERIFIED_AMBIGUOUS: "#6B7280",         # Gray
            JointStatus.UNVERIFIED_TIMEOUT: "#F59E0B",           # Amber
        }
        return colors.get(self, "#9CA3AF")


@dataclass
class JointDecision:
    """Output of the joint identity and liveness policy check."""
    status: JointStatus
    is_access_granted: bool
    is_spoof: bool
    risk_level: RiskLevel
    reason: str
    person_name: str
    role: str
    identity_similarity: float
    liveness_confidence: float
    attack_type: AttackType
    pad_model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class JointAccessEvaluator:
    """
    Deterministic rule evaluator taking identity classification and PAD output
    to produce a verified access state.
    """

    @staticmethod
    def evaluate(
        identity_data: Dict[str, Any],
        pad_result: PADResult,
    ) -> JointDecision:
        """
        Computes joint decision from identity match and presentation attack detection.

        Strict Security Invariant:
        is_access_granted is True IF AND ONLY IF (is_known == True and is_live == True and not is_warmup and not is_timeout).
        Zero frames of AUTHORIZED access are permitted during PAD warmup or unverified states.
        """
        is_known = bool(identity_data.get("is_known", False))
        is_ambiguous = bool(identity_data.get("is_ambiguous", False))
        name = str(identity_data.get("name", "Unknown Person"))
        role = str(identity_data.get("role", "Visitor"))
        sim = float(identity_data.get("similarity", 0.0))

        is_live = pad_result.is_live
        liveness_conf = pad_result.confidence
        attack_type = pad_result.attack_type
        model_name = pad_result.model_name

        stable_state = pad_result.metadata.get("stable_state", "")
        is_timeout = bool(pad_result.metadata.get("is_timeout", False) or stable_state == "UNVERIFIED_TIMEOUT")
        is_warmup = bool(
            not is_timeout
            and (
                pad_result.metadata.get("warmup_active", False)
                or pad_result.metadata.get("tag") == "WARMUP_UNVERIFIED"
                or stable_state in ("WARMUP", "UNVERIFIED_PENDING")
            )
        )

        hist_str = pad_result.metadata.get("history_symbols", "")
        hist_info = f" [Hist: {hist_str}]" if hist_str else ""
        elapsed_s = float(pad_result.metadata.get("elapsed_seconds", 0.0))
        timeout_s = float(pad_result.metadata.get("timeout_seconds", 5.0))

        # Case 0: PAD Verification Window Timed Out (5s elapsed without conclusive liveness) -> UNVERIFIED / ACCESS DENIED
        if is_timeout:
            status = JointStatus.UNVERIFIED_TIMEOUT
            is_access_granted = False
            risk_level = RiskLevel.YELLOW
            reason = (
                f"Liveness verification timed out for '{name}' after {timeout_s:.1f}s without conclusive evidence. "
                f"PAD score: {int(liveness_conf*100)}%. Access: DENIED."
            )

        # Case 1: Registered Identity Matched, but Liveness is still Verifying (Warmup/Pending Buffer) -> ACCESS PENDING
        elif is_known and is_warmup:
            status = JointStatus.LIVENESS_VERIFYING
            is_access_granted = False
            risk_level = RiskLevel.YELLOW
            reason = f"Identity Matched: '{name}' identified. Liveness verifying{hist_info} ({int(liveness_conf*100)}%). Access: PENDING."

        # Case 2: Unregistered Person, Liveness Warming Up / Pending -> ACCESS PENDING
        elif not is_known and is_warmup:
            status = JointStatus.LIVENESS_VERIFYING
            is_access_granted = False
            risk_level = RiskLevel.YELLOW
            reason = f"Unregistered Individual: Analyzing presentation buffer{hist_info}. Access: PENDING."

        # Case 3: Authorized Identity + Presentation Attack (CRITICAL IMPERSONATION ATTEMPT)
        elif is_known and not is_live:
            status = JointStatus.SPOOF_IMPERSONATION_REJECTED
            is_access_granted = False
            risk_level = RiskLevel.RED
            reason = (
                f"SPOOF ATTACK DETECTED: Presentation attack ({attack_type.label}) attempted using "
                f"credentials of registered user '{name}' (identity match: {int(sim*100)}%, liveness score: {int(liveness_conf*100)}%{hist_info}). Access Denied."
            )

        # Case 4: Authorized Identity + Bona-fide Live Person (ONLY VALID AUTHORIZATION PATH)
        elif is_known and is_live:
            status = JointStatus.AUTHORIZED
            is_access_granted = True
            risk_level = RiskLevel.GREEN
            reason = f"Access Authorized: Verified live presence of {name} ({role}) with {int(sim*100)}% match and {int(liveness_conf*100)}% liveness."

        # Case 5: Unknown / Ambiguous + Presentation Attack (SPOOF ATTACK DETECTED)
        elif not is_known and not is_live:
            status = JointStatus.SPOOF_ATTACK_DETECTED
            is_access_granted = False
            risk_level = RiskLevel.ORANGE
            reason = f"SPOOF ATTACK DETECTED: Unregistered presentation attack ({attack_type.label}) observed (liveness: {int(liveness_conf*100)}%{hist_info})."

        # Case 6: Ambiguous Match + Bona-fide Live Person (FAIL-SECURE UNVERIFIED)
        elif is_ambiguous and is_live:
            status = JointStatus.UNVERIFIED_AMBIGUOUS
            is_access_granted = False
            risk_level = RiskLevel.YELLOW
            reason = f"Unverified Identity: Live person detected with ambiguous facial similarity ({int(sim*100)}%). Identity verification required."

        # Case 7: Unregistered Visitor + Bona-fide Live Person (ROUTINE UNKNOWN PERSON)
        else:
            status = JointStatus.UNKNOWN_PERSON
            is_access_granted = False
            risk_level = RiskLevel.YELLOW
            reason = f"Unknown Visitor: Unregistered live individual detected in camera perimeter (liveness: {int(liveness_conf*100)}%)."

        is_spoof = status in (JointStatus.SPOOF_IMPERSONATION_REJECTED, JointStatus.SPOOF_ATTACK_DETECTED)

        return JointDecision(
            status=status,
            is_access_granted=is_access_granted,
            is_spoof=is_spoof,
            risk_level=risk_level,
            reason=reason,
            person_name=name,
            role=role,
            identity_similarity=sim,
            liveness_confidence=liveness_conf,
            attack_type=attack_type,
            pad_model=model_name,
            metadata={
                "is_warmup": is_warmup,
                "pad_tag": pad_result.metadata.get("tag", "UNKNOWN"),
                "raw_tag": pad_result.metadata.get("raw_tag", "UNKNOWN"),
                "stable_state": stable_state,
                "history_symbols": hist_str,
            },
        )
