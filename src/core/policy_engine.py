"""
WatchGuard Vision - Configurable Policy Engine (Phase 4.0)
---------------------------------------------------------
Evaluates multi-factor access policies based on:
- Subject Identity & Role
- Presentation Attack Detection (Liveness State)
- Spatial Surveillance Zone
- Local Time Window
- Nearby Objects & Protected Asset Interactions

Produces structured, deterministic PolicyEvaluationResults and
human-explainable 7-tuple decision packages (WHO, WHAT, WHERE, WHEN, POLICY, WHY, RESULT).
"""

from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import List, Dict, Any, Optional, Tuple
import logging

from src.events.event_types import RiskLevel, EventType
from config import (
    AUTHORIZED_HOURS_START,
    AUTHORIZED_HOURS_END,
    DEFAULT_ZONES,
)

logger = logging.getLogger("WatchGuardVision.PolicyEngine")


@dataclass
class ZonePolicy:
    """Configurable security policy for a surveillance zone."""
    zone_id: str
    zone_name: str
    allowed_roles: List[str]                  # e.g. ["Administrator", "Security Officer", "Security Engineer", "Staff"] or ["*"]
    authorized_hours_start: str = AUTHORIZED_HOURS_START
    authorized_hours_end: str = AUTHORIZED_HOURS_END
    requires_liveness: bool = True
    after_hours_action: str = "DENY"          # "DENY" or "ALLOW"
    after_hours_risk: RiskLevel = RiskLevel.ORANGE
    unknown_action: str = "DENY"              # "DENY" or "ALLOW"
    unknown_risk: RiskLevel = RiskLevel.RED
    spoof_action: str = "DENY"
    spoof_risk: RiskLevel = RiskLevel.RED

    def is_role_allowed(self, role: str) -> bool:
        """Checks if a user's role is in the permitted list."""
        if "*" in self.allowed_roles:
            return True
        return role in self.allowed_roles

    def is_within_hours(self, check_time: Optional[dtime] = None) -> bool:
        """Evaluates whether the given time falls within permitted operating window."""
        t = check_time or datetime.now().time()
        try:
            sh, sm = map(int, self.authorized_hours_start.split(":"))
            eh, em = map(int, self.authorized_hours_end.split(":"))
            start_t = dtime(sh, sm)
            end_t = dtime(eh, em)
            if start_t <= end_t:
                return start_t <= t <= end_t
            else:
                return t >= start_t or t <= end_t
        except Exception:
            return True


@dataclass
class ExplainableDecisionTuple:
    """7-tuple explainable breakdown of a security decision."""
    who: str
    what: str
    where: str
    when: str
    policy: str
    why: str
    result: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "who": self.who,
            "what": self.what,
            "where": self.where,
            "when": self.when,
            "policy": self.policy,
            "why": self.why,
            "result": self.result,
        }

    def format_explanation(self) -> str:
        return (
            f"WHO: {self.who}\n"
            f"WHAT: {self.what}\n"
            f"WHERE: {self.where}\n"
            f"WHEN: {self.when}\n"
            f"POLICY: {self.policy}\n"
            f"WHY: {self.why}\n"
            f"RESULT: {self.result}"
        )


@dataclass
class PolicyEvaluationResult:
    """Outcome of evaluating a subject against zone policy."""
    policy_code: str
    is_access_granted: bool
    risk_level: RiskLevel
    event_type: EventType
    reason: str
    recommended_action: str
    explainable_tuple: ExplainableDecisionTuple
    evidence_required: bool = False
    matched_rules: List[str] = field(default_factory=list)


class PolicyEngine:
    """
    Unified Policy Engine for WatchGuard Vision Phase 4.
    Evaluates role, zone, time, liveness, and spatial object interactions.
    """

    def __init__(self, custom_policies: Optional[Dict[str, ZonePolicy]] = None):
        self.policies: Dict[str, ZonePolicy] = {}
        self._initialize_default_policies()
        if custom_policies:
            for zid, pol in custom_policies.items():
                self.policies[zid] = pol

    def _initialize_default_policies(self) -> None:
        """Initializes default policies based on config.py zones."""
        # 1. Normal Perimeter (24/7 access for all registered staff)
        self.policies["NORMAL"] = ZonePolicy(
            zone_id="NORMAL",
            zone_name="Standard Perimeter",
            allowed_roles=["*"],
            after_hours_action="ALLOW",
            after_hours_risk=RiskLevel.GREEN,
            unknown_action="DENY",
            unknown_risk=RiskLevel.YELLOW,
            spoof_action="DENY",
            spoof_risk=RiskLevel.RED,
        )

        # 2. Protected Asset Area (24/7 access for staff; after-hours asset interaction flagged)
        self.policies["PROTECTED_ASSET_AREA"] = ZonePolicy(
            zone_id="PROTECTED_ASSET_AREA",
            zone_name="Protected Asset Area",
            allowed_roles=["*"],
            after_hours_action="ALLOW",
            after_hours_risk=RiskLevel.GREEN,
            unknown_action="DENY",
            unknown_risk=RiskLevel.RED,
            spoof_action="DENY",
            spoof_risk=RiskLevel.RED,
        )
        self.policies["PROTECTED"] = self.policies["PROTECTED_ASSET_AREA"]

        # 3. Restricted Area (Strict 09:00-18:00 window, role clearance required)
        self.policies["RESTRICTED"] = ZonePolicy(
            zone_id="RESTRICTED",
            zone_name="Restricted Area",
            allowed_roles=["Administrator", "Security Officer", "Security Engineer", "Staff", "Engineer"],
            authorized_hours_start=AUTHORIZED_HOURS_START,
            authorized_hours_end=AUTHORIZED_HOURS_END,
            requires_liveness=True,
            after_hours_action="DENY",
            after_hours_risk=RiskLevel.ORANGE,
            unknown_action="DENY",
            unknown_risk=RiskLevel.RED,
            spoof_action="DENY",
            spoof_risk=RiskLevel.RED,
        )

    def register_zone_policy(self, policy: ZonePolicy) -> None:
        """Adds or updates a zone policy."""
        self.policies[policy.zone_id] = policy

    def get_zone_policy(self, zone_id: str) -> ZonePolicy:
        """Retrieves policy for zone_id, falling back to NORMAL policy."""
        return self.policies.get(zone_id, self.policies.get("NORMAL"))

    def evaluate(
        self,
        identity: str,
        role: str,
        is_known: bool,
        liveness_state: str,
        liveness_conf: float,
        zone_id: str,
        time_str: str,
        is_auth_time: bool,
        nearby_objects: Optional[List[Dict[str, Any]]] = None,
        is_spoof: bool = False,
        attack_type: Optional[str] = None,
    ) -> PolicyEvaluationResult:
        """
        Executes the 8-step access decision pipeline for an individual face track.
        """
        policy = self.get_zone_policy(zone_id)
        objects = nearby_objects or []
        protected_interactions = [o for o in objects if o.get("is_protected", False) and o.get("is_in_proximity", False)]

        obj_summary = ", ".join([o.get("object_label", "object") for o in objects]) if objects else "None"
        what_desc = f"Live person + {obj_summary}" if (not is_spoof and liveness_state == "LIVE") else f"Presentation artifact ({attack_type or 'Photo/Replay'})"
        if not objects or obj_summary == "None":
            what_desc = "Live person" if (not is_spoof and liveness_state == "LIVE") else f"Face presentation ({liveness_state})"

        who_desc = f"{identity} ({role} - Authorized)" if is_known else f"{identity} (Unregistered Visitor)"
        when_desc = f"{time_str} ({'PERMITTED HOURS' if is_auth_time else 'AFTER-HOURS'}, Permitted: {policy.authorized_hours_start}–{policy.authorized_hours_end})"

        # --- RULE 1: Presentation Attack / Spoof ---
        if is_spoof or liveness_state in ("SPOOF", "STABLE_SPOOF"):
            atk = attack_type or "PHOTO_REPLAY_ATTACK"
            if is_known:
                code = "PRESENTATION_ATTACK_IMPERSONATION"
                reason = f"CRITICAL SECURITY INCIDENT: Spoof attack detected! Presentation attack ({atk}) attempted using credentials of registered user '{identity}'."
                risk = RiskLevel.RED
                rules = ["RULE_PAD_IMPERSONATION", "RULE_PAD_PRESENTATION_ATTACK"]
            else:
                code = "PRESENTATION_ATTACK_UNKNOWN"
                reason = f"Presentation attack detected: Unregistered individual presenting artificial face artifact ({atk})."
                risk = RiskLevel.RED if (zone_id == "RESTRICTED" or len(protected_interactions) > 0) else RiskLevel.ORANGE
                rules = ["RULE_PAD_UNKNOWN_SPOOF", "RULE_PAD_PRESENTATION_ATTACK"]

            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=f"Presentation attack artifact ({atk})",
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} Policy (Requires Bona-fide Live)",
                why=f"Presentation attack artifact detected with liveness score {int(liveness_conf * 100)}%",
                result=f"ACCESS DENIED | Risk: {risk.value} | Decision: {code}",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=False,
                risk_level=risk,
                event_type=EventType.PRESENTATION_ATTACK,
                reason=reason,
                recommended_action="DENY ACCESS IMMEDIATELY. Dispatch security response team.",
                explainable_tuple=exp,
                evidence_required=True,
                matched_rules=rules,
            )

        # --- RULE 2: Liveness Verification In Progress (WARMUP / UNVERIFIED) ---
        if liveness_state in ("WARMUP", "UNVERIFIED_PENDING", "LIVENESS_VERIFYING"):
            code = "LIVENESS_VERIFYING"
            reason = f"Identity matched for '{identity}'. Liveness verification in progress. Access: PENDING."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what="Face in view (Analyzing temporal PAD buffer)",
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} Policy (Multi-frame temporal liveness verification)",
                why="Temporal evidence accumulation in progress",
                result="ACCESS PENDING | Risk: YELLOW | Decision: LIVENESS_VERIFYING",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=False,
                risk_level=RiskLevel.YELLOW,
                event_type=EventType.PERSON_DETECTED,
                reason=reason,
                recommended_action="Hold access gate. Awaiting presentation-attack detection confirmation.",
                explainable_tuple=exp,
                evidence_required=False,
                matched_rules=["RULE_PAD_VERIFYING_PENDING"],
            )

        # --- RULE 3: Unregistered Visitor in Restricted Area ---
        if not is_known and zone_id == "RESTRICTED":
            code = "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS"
            reason = f"Unauthorized access: Unregistered individual detected inside restricted security zone ({policy.zone_name})."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=what_desc,
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} Policy (Requires Registered Credentials & Role Clearance)",
                why="Unregistered subject entered restricted perimeter without active credentials",
                result="ACCESS DENIED | Risk: RED | Decision: UNAUTHORIZED_RESTRICTED_ZONE_ACCESS",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=False,
                risk_level=RiskLevel.RED,
                event_type=EventType.UNAUTHORIZED_ACCESS,
                reason=reason,
                recommended_action="Dispatch security response team; sound local perimeter alert.",
                explainable_tuple=exp,
                evidence_required=True,
                matched_rules=["RULE_UNAUTHORIZED_RESTRICTED_ZONE"],
            )

        # --- RULE 4: Unregistered Visitor near Protected Asset ---
        if not is_known and len(protected_interactions) > 0:
            target_obj = protected_interactions[0]
            obj_lbl = target_obj.get("object_label", "asset")
            dist = target_obj.get("distance_pixels", 0)
            code = "POTENTIAL_UNAUTHORIZED_INTERACTION"
            reason = f"Potential unauthorized interaction: Unregistered person in visual proximity ({dist}px) to protected asset ({obj_lbl.upper()})."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=f"Unregistered person near {obj_lbl.upper()} ({dist}px)",
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} Asset Protection Policy",
                why=f"Unregistered individual in close proximity ({dist}px) to protected asset",
                result="ACCESS DENIED | Risk: RED | Decision: POTENTIAL_UNAUTHORIZED_INTERACTION",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=False,
                risk_level=RiskLevel.RED,
                event_type=EventType.SECURITY_ALERT,
                reason=reason,
                recommended_action="Verify subject credentials immediately; alert facility security officer.",
                explainable_tuple=exp,
                evidence_required=True,
                matched_rules=["RULE_POTENTIAL_UNAUTHORIZED_INTERACTION"],
            )

        # --- RULE 5: Unregistered Visitor in Standard / Protected Area ---
        if not is_known:
            code = "UNKNOWN_PERSON"
            reason = f"Unregistered individual detected in {policy.zone_name}."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=what_desc,
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} General Surveillance Policy",
                why="Individual is not registered in the facial database",
                result="ACCESS DENIED | Risk: YELLOW | Decision: UNKNOWN_PERSON",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=False,
                risk_level=RiskLevel.YELLOW,
                event_type=EventType.UNKNOWN_PERSON,
                reason=reason,
                recommended_action="Log visitor appearance; monitor subject on security feed.",
                explainable_tuple=exp,
                evidence_required=True,
                matched_rules=["RULE_UNKNOWN_PERSON"],
            )

        # --- RULE 6: Authorized Personnel in Restricted Area Outside Permitted Hours ---
        if is_known and zone_id == "RESTRICTED" and not is_auth_time:
            code = "AFTER_HOURS_RESTRICTED_ACCESS_DENIED"
            reason = f"Authorized personnel detected inside Restricted Area outside permitted operating hours ({time_str}). Permitted window is {policy.authorized_hours_start} to {policy.authorized_hours_end}."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=what_desc,
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} Time Policy (Permitted: {policy.authorized_hours_start}–{policy.authorized_hours_end})",
                why="Authorized user detected inside Restricted Area outside permitted operating hours",
                result="ACCESS DENIED | Risk: ORANGE | Decision: AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=False,
                risk_level=policy.after_hours_risk,
                event_type=EventType.POLICY_VIOLATION,
                reason=reason,
                recommended_action="DENY ACCESS. Advise authorized personnel of operating hours (09:00–18:00).",
                explainable_tuple=exp,
                evidence_required=True,
                matched_rules=["RULE_AFTER_HOURS_RESTRICTED_ZONE_DENIED"],
            )

        # --- RULE 7: Authorized Personnel in Restricted Area During Permitted Hours ---
        if is_known and zone_id == "RESTRICTED" and is_auth_time:
            code = "AUTHORIZED_RESTRICTED_ZONE_ACCESS"
            reason = f"Authorized personnel verified: {identity} inside Restricted Area during permitted operating hours ({time_str})."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=what_desc,
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} Policy (Permitted: {policy.authorized_hours_start}–{policy.authorized_hours_end})",
                why="Verified personnel with valid role credentials during operating hours",
                result="ACCESS GRANTED | Risk: GREEN | Decision: AUTHORIZED_RESTRICTED_ZONE_ACCESS",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=True,
                risk_level=RiskLevel.GREEN,
                event_type=EventType.AUTHORIZED_PERSON,
                reason=reason,
                recommended_action="Grant access to Restricted Area. Monitor routine operational activity.",
                explainable_tuple=exp,
                evidence_required=False,
                matched_rules=["RULE_AUTHORIZED_RESTRICTED_ZONE"],
            )

        # --- RULE 8: Authorized Personnel + Protected Asset After-Hours ---
        if is_known and len(protected_interactions) > 0 and not is_auth_time:
            target_obj = protected_interactions[0]
            obj_lbl = target_obj.get("object_label", "asset")
            code = "SUSPICIOUS_AUTHORIZED_USER_ACTIVITY"
            reason = f"After-hours asset activity: Authorized user ({identity}) interacting with {obj_lbl.upper()} outside authorized hours ({time_str})."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=f"Authorized user interacting with {obj_lbl.upper()}",
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} Asset Policy (After-hours logging)",
                why="Asset interaction outside standard operating schedule",
                result="ACCESS GRANTED (MONITORED) | Risk: ORANGE | Decision: SUSPICIOUS_AUTHORIZED_USER_ACTIVITY",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=True,
                risk_level=RiskLevel.ORANGE,
                event_type=EventType.POLICY_VIOLATION,
                reason=reason,
                recommended_action="Flag after-hours audit log; request supervisor authorization confirmation.",
                explainable_tuple=exp,
                evidence_required=True,
                matched_rules=["RULE_SUSPICIOUS_AUTHORIZED_USER_ACTIVITY"],
            )

        # --- RULE 9: Authorized Personnel in Normal / Protected Area (24/7 Access) ---
        if is_known:
            code = "NORMAL_ACTIVITY"
            desc = f"Authorized personnel verified: {identity} in {policy.zone_name}."
            if not is_auth_time:
                desc += " After-hours restriction does not apply outside the Restricted Area."
            exp = ExplainableDecisionTuple(
                who=who_desc,
                what=what_desc,
                where=policy.zone_name,
                when=when_desc,
                policy=f"{policy.zone_name} 24/7 Access Policy",
                why="Authorized personnel verified with bona-fide liveness credentials",
                result="ACCESS GRANTED | Risk: GREEN | Decision: NORMAL_ACTIVITY",
            )
            return PolicyEvaluationResult(
                policy_code=code,
                is_access_granted=True,
                risk_level=RiskLevel.GREEN,
                event_type=EventType.AUTHORIZED_PERSON,
                reason=desc,
                recommended_action="Continue standard operational monitoring.",
                explainable_tuple=exp,
                evidence_required=False,
                matched_rules=["RULE_NORMAL_ACTIVITY"],
            )

        # --- RULE 10: Clean State / Routine Monitoring ---
        code = "NORMAL_ACTIVITY"
        reason = "Surveillance perimeter clear. Routine monitoring active."
        exp = ExplainableDecisionTuple(
            who="None (Perimeter clear)",
            what="Standard surveillance feed",
            where=policy.zone_name,
            when=when_desc,
            policy=f"{policy.zone_name} Baseline Policy",
            why="No policy violations or unauthorized subjects detected",
            result="ACCESS NEUTRAL | Risk: GREEN | Decision: NORMAL_ACTIVITY",
        )
        return PolicyEvaluationResult(
            policy_code=code,
            is_access_granted=False,
            risk_level=RiskLevel.GREEN,
            event_type=EventType.PERSON_DETECTED,
            reason=reason,
            recommended_action="Continue standard operational monitoring.",
            explainable_tuple=exp,
            evidence_required=False,
            matched_rules=["RULE_PERIMETER_CLEAR"],
        )
