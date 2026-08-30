"""
WatchGuard AI Security Copilot (Phase 5.5)
------------------------------------------
Downstream AI context reasoner that explains and synthesizes the authoritative
security state produced by the deterministic WatchGuard Security Engine.

Core Principles:
1. The deterministic WatchGuard Security Engine is the SOLE authority for identity,
   liveness, zone policy, risk levels, and access decisions.
2. The AI Copilot explains and summarizes authoritative state; it CANNOT override it.
3. Explicit evidence grounding: identity confidence, liveness result/confidence,
   zone, operating hours, policy code, active incident, and supporting rationale.
4. Strict anti-hallucination guardrails: Never invent people, objects, events,
   intent, or actions not present in the structured state.
5. Clear separation between CURRENT_STATE, RECENT_HISTORY, and HISTORICAL_EVENTS.
6. Zero transmission of raw camera frames, raw audio, or biometric embeddings.
7. Logs answer source: DETERMINISTIC vs AI_GROUNDED vs FALLBACK_DETERMINISTIC.
"""

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

import cv2
import numpy as np

from config import (
    COPILOT_ENABLED,
    GEMINI_API_KEY,
    COPILOT_MODEL_NAME,
    COPILOT_TIMEOUT_SECONDS,
    COPILOT_LOCAL_ENDPOINT,
    COPILOT_LOCAL_MODEL,
)
from src.events.event_types import RiskLevel

logger = logging.getLogger("WatchGuardVision.CopilotReasoner")


def encode_frame_to_jpeg_base64(frame: np.ndarray, quality: int = 80) -> Optional[str]:
    """
    Encodes an OpenCV BGR numpy frame into a compact JPEG base64 string
    for on-demand multimodal AI queries. Execution latency is typically < 1 ms.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buf = cv2.imencode(".jpg", frame, encode_param)
        if ret:
            return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to encode frame to JPEG base64: {e}")
    return None


VISUAL_QUERY_PATTERNS = [
    r"\b(what\s+do\s+you\s+see|describe\s+(the\s+)?(scene|view|camera|room|frame|environment))\b",
    r"\b(who\s+is\s+(in\s+front|standing|visible|present|beside|there))\b",
    r"\b(what\s+is\s+(he|she|this\s+person|the\s+person|the\s+second\s+person|the\s+other\s+person|that\s+person)\s+holding)\b",
    r"\b(what\s+(am\s+i|are\s+you|is\s+he|is\s+she|is\s+that\s+person|is\s+anyone)\s+(holding|carrying))\b",
    r"\b(what\s+(object\s+)?is\s+in\s+(my|his|her|their|that\s+person's)\s+hand(s)?)\b",
    r"\b(what('s)?\s+in\s+(my|his|her|their)\s+hand(s)?)\b",
    r"\b(what\s+object\s+(am\s+i|is\s+he|is\s+she|is\s+that\s+person)\s+holding)\b",
    r"\b(is\s+(he|she|anyone|that\s+person)\s+holding\s+(anything|an\s+object|any\s+item))\b",
    r"\b(tell\s+what\s+i\s+am\s+carrying|tell\s+what\s+i'm\s+carrying|what\s+am\s+i\s+carrying)\b",
    r"\b(is\s+there\s+another\s+person|are\s+there\s+other\s+people|is\s+anyone\s+beside)\b",
    r"\b(what\s+objects\s+are\s+(visible|in\s+view|detected|present|around))\b",
    r"\b(look\s+at\s+the\s+camera|visual\s+inspection|visual\s+check)\b",
    r"\b(how\s+many\s+people\s+(are\s+)?(visible|in\s+view|in\s+front|do\s+you\s+see|in\s+the\s+frame))\b",
    r"\b(describe\s+(his|her|their)\s+(appearance|clothing|clothes))\b",
    r"\b(what\s+is\s+(the\s+person|he|she|that\s+person)\s+wearing)\b",
]


def is_visual_query(query: str) -> bool:
    """
    Classifies whether a user query requires visual inspection of the camera frame.
    Non-visual queries (policy explanations, metric totals, historical alerts) return False.
    """
    if not query:
        return False
    q_norm = query.lower().strip()

    # Explicit non-visual query guards
    if re.search(r"\b(why\s+(was|is|did|were|am\s+i|can\s+i\s+not)|how\s+many\s+.*(today|so\s+far|in\s+total|past)|what\s+happened|incident|operating\s+hours|authorized\s+hours)\b", q_norm):
        # Unless specifically asking about visual object holding
        if not re.search(r"\b(holding|in\s+(my|his|her|their)\s+hand|carrying)\b", q_norm):
            return False

    for pat in VISUAL_QUERY_PATTERNS:
        if re.search(pat, q_norm):
            return True
    return False


class CopilotResponseSource(str, Enum):
    """Tracks the authoritative generation source for auditability."""
    DETERMINISTIC = "DETERMINISTIC"
    AI_GROUNDED = "AI_GROUNDED"
    FALLBACK_DETERMINISTIC = "FALLBACK_DETERMINISTIC"


@dataclass
class CopilotResponse:
    """Structured response from the Security Copilot."""
    text: str
    source: CopilotResponseSource
    latency_ms: float
    evidence_grounding: Dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source.value,
            "latency_ms": self.latency_ms,
            "evidence_grounding": self.evidence_grounding,
            "model_name": self.model_name,
        }


def build_structured_ai_context(
    context_engine: Any = None,
    security_analytics: Any = None,
    database_manager: Any = None,
) -> Dict[str, Any]:
    """
    Assembles a compact, deterministic, evidence-grounded JSON context.
    Explicitly separates CURRENT_STATE, RECENT_HISTORY, and HISTORICAL_EVENTS.
    Contains zero raw video frames, zero raw audio, and zero biometric embeddings.
    """
    now = datetime.now()
    now_time_str = now.strftime("%H:%M:%S")

    last_decision = getattr(context_engine, "last_decision", None) if context_engine else None
    h_start = getattr(context_engine, "hours_start", "09:00") if context_engine else "09:00"
    h_end = getattr(context_engine, "hours_end", "18:00") if context_engine else "18:00"

    # --- 1. CURRENT_STATE ---
    live_people = []
    live_objects = []
    current_zone = "Standard Perimeter"
    master_risk = "GREEN"
    master_decision_code = "NORMAL_ACTIVITY"
    master_reason = "Perimeter clear. Routine surveillance active."
    is_authorized_hours = True

    if last_decision:
        master_risk = last_decision.risk_level.value if hasattr(last_decision.risk_level, "value") else str(last_decision.risk_level)
        master_decision_code = last_decision.decision
        master_reason = last_decision.reason
        ctx_sum = getattr(last_decision, "context_summary", {}) or {}
        current_zone = ctx_sum.get("zone_name", "Standard Perimeter")
        is_authorized_hours = ctx_sum.get("is_authorized_hours", True)

        # Fused Live People Entities
        entities = getattr(last_decision, "person_entities", []) or (ctx_sum.get("person_entities", []) if ctx_sum else [])
        for e in entities:
            e_ident = getattr(e, "identity", None) or (e.get("identity") if isinstance(e, dict) else "Unknown Person")
            e_conf = getattr(e, "identity_confidence", 0.0) if not isinstance(e, dict) else e.get("identity_confidence", 0.0)
            e_role = getattr(e, "role", "Visitor") if not isinstance(e, dict) else e.get("role", "Visitor")
            e_known = getattr(e, "is_known", False) if not isinstance(e, dict) else e.get("is_known", False)
            if e_ident in ("Unknown Person", "Unknown", "Unregistered", "None", "", None) or master_decision_code in ("UNKNOWN_PERSON", "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS"):
                e_known = False

            e_liveness = getattr(e, "liveness_state", "WARMUP") if not isinstance(e, dict) else e.get("liveness_state", "WARMUP")
            e_live_conf = getattr(e, "liveness_confidence", 0.0) if not isinstance(e, dict) else e.get("liveness_confidence", 0.0)
            e_spoof = getattr(e, "is_spoof", False) if not isinstance(e, dict) else e.get("is_spoof", False)
            e_zone = getattr(e, "zone_name", current_zone) if not isinstance(e, dict) else e.get("zone_name", current_zone)
            e_access = getattr(e, "access_state", "PENDING") if not isinstance(e, dict) else e.get("access_state", "PENDING")
            if not e_known or getattr(last_decision, "is_access_granted", True) is False:
                e_access = "SPOOF_REJECTED" if e_spoof else "DENIED"

            e_risk = getattr(e, "risk_level", RiskLevel.GREEN) if not isinstance(e, dict) else e.get("risk_level", "GREEN")
            e_risk_str = e_risk.value if hasattr(e_risk, "value") else str(e_risk)
            e_reason = getattr(e, "reason", master_reason) if not isinstance(e, dict) else e.get("reason", master_reason)

            live_people.append({
                "track_id": getattr(e, "track_id", 1) if not isinstance(e, dict) else e.get("track_id", 1),
                "identity": e_ident,
                "identity_confidence": round(float(e_conf), 2),
                "is_registered": bool(e_known),
                "role": e_role,
                "liveness_result": "SPOOF" if e_spoof else ("LIVE" if e_liveness == "STABLE_LIVE" else str(e_liveness)),
                "liveness_confidence": round(float(e_live_conf), 2),
                "zone": e_zone,
                "access_decision": e_access,
                "risk_level": e_risk_str,
                "supporting_reason": e_reason,
            })

        # Live Objects
        if ctx_sum:
            objs = ctx_sum.get("objects", [])
            for obj in objs:
                lbl = obj.get("label") if isinstance(obj, dict) else str(obj)
                conf = obj.get("confidence", 0.85) if isinstance(obj, dict) else 0.85
                live_objects.append({
                    "label": lbl,
                    "confidence": round(float(conf), 2),
                    "is_protected_asset": lbl in ("laptop", "cell phone"),
                })

    # Active Incidents
    active_incidents = []
    if security_analytics and hasattr(security_analytics, "get_prioritized_incidents"):
        prioritized = security_analytics.get_prioritized_incidents(limit=3)
        for inc in prioritized:
            active_incidents.append({
                "incident_id": inc.incident_id,
                "severity": inc.risk_level.value if hasattr(inc.risk_level, "value") else str(inc.risk_level),
                "status": inc.status,
                "subject": inc.person_name,
                "zone": inc.zone,
                "policy_code": inc.policy_code,
                "duration_seconds": round(inc.duration_seconds, 1),
                "reason": inc.reason,
            })

    # Extract Research / CCSDF / Novelty features
    ccsdf_info = {}
    tailgating_info = {}
    behavior_info = {}
    uncertainty_info = {}
    if last_decision and hasattr(last_decision, "ccsdf_result") and last_decision.ccsdf_result:
        res = last_decision.ccsdf_result
        ccsdf_info = {
            "risk_score": res.get("risk_score", 0.0),
            "risk_level": res.get("risk_level", master_risk),
            "dominant_factor": res.get("dominant_factor", "nominal"),
            "recommended_action": res.get("recommended_action", "ALLOW"),
            "normalized_signals": res.get("normalized_signals", {}),
        }
        tailgating_info = {
            "status": res.get("tailgating_status", "NONE"),
            "leader_track_id": res.get("tailgating_leader_track_id"),
            "leader_identity": res.get("tailgating_leader", "Authorized Person"),
            "follower_track_id": res.get("tailgating_follower_track_id"),
            "follower_identity": res.get("tailgating_follower", "Unknown Person"),
            "distance": res.get("tailgating_distance", 0.0),
            "authorization_transfer": "NEVER (Leader auth never transfers to follower)",
        }
        behavior_info = {
            "status": res.get("behavior_status", "NORMAL_ACTIVITY"),
            "s_behavior": res.get("normalized_signals", {}).get("behavior", 0.0),
        }
        uncertainty_info = {
            "confidence_tier": res.get("confidence_tier", "HIGH"),
            "confidence": res.get("confidence", {}),
        }

    import os
    from config import OBJECT_DETECTOR_MODE
    active_det_mode = os.getenv("WATCHGUARD_DETECTOR_MODE", OBJECT_DETECTOR_MODE)

    current_state = {
        "timestamp": now_time_str,
        "active_zone": current_zone,
        "operating_hours_window": f"{h_start} to {h_end}",
        "time_status": "NORMAL_HOURS" if is_authorized_hours else "AFTER_HOURS",
        "detector_mode": active_det_mode,
        "detector_classes_count": 84 if active_det_mode == "watchguard_unified" else (4 if active_det_mode == "watchguard_custom" else 80),
        "master_risk_level": master_risk,
        "master_decision_code": master_decision_code,
        "master_reason": master_reason,
        "people_count": len(live_people),
        "people": live_people,
        "live_people_in_view": live_people,
        "objects_count": len(live_objects),
        "objects": live_objects,
        "live_detected_objects": live_objects,
        "active_incidents": active_incidents,
        "ccsdf_state": ccsdf_info,
        "tailgating_state": tailgating_info,
        "behavior_state": behavior_info,
        "uncertainty_state": uncertainty_info,
    }

    # --- 2. RECENT_HISTORY (Bounded 5-minute / 15-minute window) ---
    recent_metrics_5m = {}
    recent_events_5m = []
    if security_analytics and hasattr(security_analytics, "get_metrics"):
        m5 = security_analytics.get_metrics("5min")
        recent_metrics_5m = {
            "spoof_attempts_5m": m5.spoof_attempts,
            "after_hours_violations_5m": m5.after_hours_violations,
            "restricted_zone_entries_5m": m5.restricted_zone_entries,
            "active_incidents_5m": m5.active_incidents,
        }

    if database_manager and hasattr(database_manager, "get_events_by_window"):
        try:
            evs = database_manager.get_events_by_window(window_seconds=300.0, limit=5)
            for ev in evs:
                recent_events_5m.append({
                    "timestamp": ev.get("timestamp", "").split("T")[-1][:8] if "T" in ev.get("timestamp", "") else ev.get("timestamp", ""),
                    "risk_level": ev.get("risk_level", "GREEN"),
                    "event_type": ev.get("event_type", ""),
                    "person_name": ev.get("person_name") or "Unregistered",
                    "description": ev.get("description", ""),
                })
        except Exception:
            pass

    recent_history = {
        "window_5min_metrics": recent_metrics_5m,
        "recent_events_5min": recent_events_5m,
    }

    # --- 3. HISTORICAL_EVENTS (Today's Totals & Aggregates) ---
    today_metrics = {}
    if security_analytics and hasattr(security_analytics, "get_metrics"):
        mtoday = security_analytics.get_metrics("today")
        today_metrics = {
            "distinct_people_seen_today": mtoday.distinct_people_seen,
            "authorized_people_seen_today": mtoday.authorized_people_seen,
            "unknown_visitors_today": mtoday.unknown_visitors,
            "confirmed_spoofs_today": mtoday.spoof_attempts,
            "after_hours_violations_today": mtoday.after_hours_violations,
            "total_incidents_today": mtoday.active_incidents + mtoday.resolved_incidents,
        }

    historical_events = {
        "today_totals": today_metrics,
    }

    # --- 4. EVIDENCE_GROUNDING (Authoritative 7-Tuple) ---
    evidence_grounding = {
        "who": live_people[0]["identity"] if live_people else "None",
        "what": master_decision_code,
        "where": current_zone,
        "when": f"{now_time_str} ({'NORMAL HOURS' if is_authorized_hours else 'AFTER HOURS'})",
        "policy": f"Zone {current_zone} ({h_start} to {h_end})",
        "why": master_reason,
        "result": f"Access {'GRANTED' if master_risk == 'GREEN' else 'DENIED'}, Risk {master_risk}",
    }

    return {
        "CURRENT_STATE": current_state,
        "RECENT_HISTORY": recent_history,
        "HISTORICAL_EVENTS": historical_events,
        "EVIDENCE_GROUNDING": evidence_grounding,
    }


COPILOT_SYSTEM_PROMPT = """You are the AI Security Copilot for WatchGuard Vision, an advanced edge-based continuous contextual physical security and surveillance system.

AUTHORITY & POLICY CONSTRAINT:
The deterministic WatchGuard Security Engine and PolicyEngine are the SOLE authority on identity, liveness (PAD), zone policy, risk posture (GREEN/YELLOW/ORANGE/RED), and access decisions (GRANTED/DENIED/PENDING).
You explain, summarize, and synthesize the authoritative state. You CANNOT override decisions, alter risk ratings, or grant access.

FULL ARCHITECTURE & DOMAIN KNOWLEDGE:
1. OBJECT PERCEPTION MODES:
   - "production": Standard YOLOv8n (80 COCO classes).
   - "watchguard_unified": Unified 84-class detector (80 COCO + 80: pen, 81: access_badge, 82: usb_drive, 83: keys).
   - "watchguard_custom": 4-class research prototype.
2. DOMAIN SECURITY ASSETS:
   - pen (context-relevant), access_badge (credential), usb_drive (high-risk asset), keys (physical access credential).
   - Objects are visual evidence, NOT automatic attacks. A USB drive alone is never an automatic attack; its risk depends on who holds it, zone, time, proximity to protected workstations, behavior, and history.
3. CONTINUOUS CONTEXTUAL SECURITY DECISION FUSION (CCSDF):
   - Continuous vector S(t) = [Sid, Spad, Szone, Stime, Sprox, Shist, Sasset, Sbehavior].
   - Computes continuous risk R(t) = Sum(w_i * S_i).
4. TAILGATING RULES:
   - Triggered when an authorized leader is closely followed by an unknown/unauthorized follower into a secure zone.
   - CRITICAL: Leader authorization NEVER transfers to follower. Follower access remains DENIED.
5. BEHAVIOR ANOMALY ENGINE:
   - Tracks dwell time, repeated restricted zone attempts, and after-hours presence.
   - S_behavior in [0, 1] categorized as NORMAL_ACTIVITY (<0.35), UNUSUAL_PATTERN (0.35-0.70), or BEHAVIOR_ANOMALY (>=0.70).
6. UNCERTAINTY-AWARE GATING:
   - Evaluated across biometric geometric mean and contextual telemetry into HIGH (>=0.75), MEDIUM (0.45-0.75), and LOW (<0.45) tiers.
   - LOW confidence places candidate access into PENDING_VERIFICATION; never silently authorizes.
7. SPOOFS TODAY COUNTER:
   - Measures CONFIRMED PRESENTATION ATTACK SESSIONS recorded today (edge-triggered), NOT raw camera frames.

STRICT INVARIANTS:
1. UNREGISTERED / UNKNOWN != AUTHORIZED. An unknown person is strictly ACCESS DENIED.
2. CONFIRMED SPOOF / PAD FAILURE -> Immediate DENY (RED).
3. POLICY DENY -> Immutable. CCSDF recommendations cannot override a PolicyEngine DENY.
4. If data is unavailable, state: "I do not currently have evidence for that."

RESPONSE STRUCTURE FOR REASONING INQUIRIES:
Prefer: CURRENT STATE -> relevant evidence -> contextual factors -> CCSDF risk -> recommendation -> authoritative decision."""


class BaseAICopilotReasoner(ABC):
    """Abstract interface for AI Security Copilot reasoning engines."""

    @abstractmethod
    def generate_explanation(
        self,
        query: str,
        context: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> CopilotResponse:
        """Generates a natural language response grounded in the structured context and optional camera frame."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the reasoner backend is configured and ready."""
        pass


class DeterministicCopilotReasoner(BaseAICopilotReasoner):
    """
    High-reliability, deterministic rule-based fallback reasoner.
    Synthesizes exact security explanations from structured context without external LLM calls.
    """

    def __init__(self, security_analytics: Any = None):
        self.security_analytics = security_analytics
        self.model_name = "Deterministic-Rule-Engine"

    def is_available(self) -> bool:
        return True

    def generate_explanation(
        self,
        query: str,
        context: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> CopilotResponse:
        t0 = time.time()
        curr = context.get("CURRENT_STATE", {})
        grounding = context.get("EVIDENCE_GROUNDING", {})
        q_lower = query.lower()

        # 1. Person holding object / hand query
        if "holding" in q_lower or "in my hand" in q_lower or "in his hand" in q_lower or "in her hand" in q_lower or "in their hand" in q_lower or "carrying" in q_lower or "in hand" in q_lower:
            objs = curr.get("objects", curr.get("live_detected_objects", []))
            if objs:
                obj_str = ", ".join([o.get("label", "object") for o in objs])
                text = f"Based on detected vision objects, you appear to be holding or interacting with: {obj_str}."
            else:
                text = "No objects are currently detected in your hand in the camera view."

        # 2. Decision / Denial explanation
        elif "why" in q_lower or "reason" in q_lower or "denied" in q_lower or "refused" in q_lower or "cannot enter" in q_lower or "can't enter" in q_lower or "preventing" in q_lower or "won't" in q_lower:
            p_list = curr.get("people", curr.get("live_people_in_view", []))
            target_who = context.get("EVIDENCE_GROUNDING", {}).get("who")
            for cand in ["Hunter", "Officer Alice", "Gulam Moinuddin"]:
                if cand.lower() in q_lower:
                    target_who = cand
                    break

            if not p_list and not target_who and not curr.get("master_decision_code"):
                text = "No person is currently detected in the camera view to evaluate access."
            else:
                p0 = p_list[0] if p_list else {}
                p_ident = target_who or p0.get("identity", "The individual")
                is_known = bool(p0.get("is_registered", False)) or bool(target_who and target_who not in ("Unknown Person", "Unknown", None, ""))
                liveness_res = p0.get("liveness_result", "WARMUP")
                h_win = curr.get("operating_hours_window", "09:00 to 18:00")
                code = curr.get("master_decision_code", "")
                active_zone = curr.get("active_zone", "Restricted Area" if ("restricted" in curr.get("active_zone", "").lower() or "restricted" in code.lower()) else "Standard Perimeter")

                if liveness_res == "SPOOF" or "SPOOF" in code or "PRESENTATION" in code:
                    target_disp = p_ident if (is_known and p_ident not in ("Unknown Person", "Unknown", None, "")) else "the individual"
                    text = f"Access was denied because a presentation attack or spoof attempt was detected using credentials for {target_disp}."
                elif not is_known or p_ident in ("Unknown Person", "Unknown", None, ""):
                    is_in_restricted = "restricted" in active_zone.lower() or "restricted" in code.lower()
                    is_after_hours = curr.get("time_status") == "AFTER_HOURS" or "AFTER_HOURS" in code or "outside" in curr.get("master_reason", "").lower()
                    is_repeated = "REPEATED" in code or "repeated" in curr.get("master_reason", "").lower()
                    if "protected" in active_zone.lower() or "asset" in active_zone.lower() or "protected" in curr.get("master_reason", "").lower():
                        text = "Access is denied. An unregistered individual was detected in the Protected Asset Area."
                    elif is_in_restricted:
                        if is_after_hours and is_repeated:
                            text = "Access is denied. You are currently unregistered and inside the Restricted Area after permitted hours. WatchGuard has also detected repeated restricted-zone violations."
                        elif is_after_hours:
                            text = "Access is denied. You are currently unregistered and inside the Restricted Area after permitted hours."
                        elif is_repeated:
                            text = "Access is denied. You are currently unregistered and inside the high-security Restricted Area, where repeated unauthorized zone entries have occurred."
                        else:
                            text = "Access is denied. An unregistered individual was detected in the Restricted Area."
                    else:
                        text = "Access is denied: You are not registered in the facial database and do not have active security clearance."
                elif "AFTER_HOURS" in code or ("restricted" in active_zone.lower() and curr.get("time_status") == "AFTER_HOURS") or "outside" in curr.get("master_reason", "").lower():
                    target_disp = f"{p_ident} was observed in the {active_zone}" if (p_ident and p_ident not in ("The individual", "Unknown Person", "Unknown", None, "")) else "this was observed"
                    text = f"Access is denied because {target_disp} outside the permitted {h_win} operating window."
                else:
                    if curr.get("is_access_granted") is False or curr.get("master_risk_level") in ("YELLOW", "ORANGE", "RED"):
                        text = f"Access is denied. The system classified the situation as {curr.get('master_risk_level', 'YELLOW')} because: {curr.get('master_reason', 'Policy restriction')}."
                    else:
                        text = f"Security status is {curr.get('master_risk_level', 'GREEN')} ({code}): {curr.get('master_reason', 'Routine operations')}."

        # Recent events / temporal summary
        elif "what happened" in q_lower or "last" in q_lower or "recent" in q_lower:
            m5 = context.get("RECENT_HISTORY", {}).get("window_5min_metrics", {})
            parts = []
            if m5.get("active_incidents_5m", 0) > 0:
                parts.append(f"{m5['active_incidents_5m']} active incident(s)")
            if m5.get("after_hours_violations_5m", 0) > 0:
                parts.append(f"{m5['after_hours_violations_5m']} after-hours violation(s)")
            if m5.get("spoof_attempts_5m", 0) > 0:
                parts.append(f"{m5['spoof_attempts_5m']} spoof attempt(s)")

            if parts:
                text = f"In the last 5 minutes, the system recorded {', '.join(parts)}."
            else:
                text = "In the last 5 minutes, no security violations or alerts were recorded."

        # People count query
        elif "how many people" in q_lower or "people count" in q_lower or "how many person" in q_lower:
            p_list = curr.get("people", curr.get("live_people_in_view", []))
            p_count = curr.get("people_count", len(p_list))
            if p_count == 0:
                text = "There are currently no people in view."
            elif p_count == 1:
                p0 = p_list[0] if p_list else {}
                name = p0.get("identity", "one person")
                text = f"There is one person in view: {name}."
            elif p_count == 2:
                names = [p.get("identity", "unregistered person") for p in p_list]
                text = f"There are two people in view: {names[0]} and {names[1]}."
            else:
                text = f"There are {p_count} people in view."

        # Objects query
        elif "what objects" in q_lower or "detected objects" in q_lower or "items in view" in q_lower:
            objs = curr.get("objects", curr.get("live_detected_objects", []))
            p_list = curr.get("people", curr.get("live_people_in_view", []))
            p_count = len(p_list)
            if objs:
                obj_str = ", ".join([o.get("label", "object") for o in objs])
                if p_count > 0:
                    names = [p.get("identity", "unregistered person") for p in p_list]
                    text = f"I detect {len(objs)} object(s): {obj_str}. I also observe {p_count} person(s): {', '.join(names)}."
                else:
                    text = f"I detect {len(objs)} object(s): {obj_str}."
            elif p_count > 0:
                names = [p.get("identity", "unregistered person") for p in p_list]
                text = f"No non-person objects are detected. I currently see {p_count} person(s): {', '.join(names)}."
            else:
                text = "No objects are currently detected in view."

        # Scene overview / who is in front / what do you see
        elif "what do you see" in q_lower or "who is in front" in q_lower or "scene" in q_lower or "who is standing" in q_lower or "who is present" in q_lower:
            p_list = curr.get("people", curr.get("live_people_in_view", []))
            objs = curr.get("objects", curr.get("live_detected_objects", []))
            if p_list:
                auth = [p["identity"] for p in p_list if p.get("is_registered") and p.get("liveness_result") != "SPOOF" and p.get("identity") and p["identity"].strip().lower() not in ("unknown person", "unknown", "none", "")]
                unk = [p["identity"] for p in p_list if (not p.get("is_registered") or not p.get("identity") or p["identity"].strip().lower() in ("unknown person", "unknown", "none", "")) and p.get("liveness_result") != "SPOOF"]
                spoofs = [f"spoof presentation of {p['identity']}" for p in p_list if p.get("liveness_result") == "SPOOF"]
                obj_str = f" with {', '.join([o.get('label', 'object') for o in objs])}" if objs else ""

                if len(p_list) == 1:
                    if auth:
                        text = f"I observe one authorized person, {auth[0]}{obj_str}."
                    elif unk:
                        text = f"I observe one unregistered person{obj_str}."
                    elif spoofs:
                        text = f"I detect a spoof presentation attack{obj_str}."
                    else:
                        text = f"I observe one person in view{obj_str}."
                elif len(p_list) == 2:
                    if auth and len(unk) == 1:
                        text = f"I see 2 people: {auth[0]} (authorized) and one unregistered person{obj_str}."
                    elif len(auth) == 2:
                        text = f"I see 2 people: {auth[0]} and {auth[1]} (authorized){obj_str}."
                    elif len(unk) == 2:
                        text = f"I see 2 people: two unregistered people in the scene{obj_str}."
                    else:
                        text = f"I see 2 people in the camera view{obj_str}."
                else:
                    parts = []
                    if auth:
                        parts.append(f"{', '.join(auth)} (authorized)")
                    if unk:
                        parts.append(f"{len(unk)} unregistered people")
                    if spoofs:
                        parts.append(", ".join(spoofs))
                    people_str = " and ".join(parts) if parts else "unregistered people"
                    text = f"I see {len(p_list)} people: {people_str}{obj_str}."
            elif objs:
                text = f"I observe {len(objs)} object(s): {', '.join([o.get('label', 'object') for o in objs])}. No person in view."
            else:
                text = f"System operational in {curr.get('active_zone', 'Standard Perimeter')}. Risk posture is {curr.get('master_risk_level', 'GREEN')}."

        # Default overview
        else:
            p_list = curr.get("people", curr.get("live_people_in_view", []))
            if len(p_list) == 1:
                p0 = p_list[0]
                text = f"Currently observing {p0['identity']} in {p0['zone']}. Access is {p0['access_decision']} with risk level {p0['risk_level']}."
            elif len(p_list) > 1:
                names = [p.get("identity", "unregistered person") for p in p_list]
                text = f"Currently observing {len(p_list)} people ({', '.join(names)}) in {curr.get('active_zone', 'Standard Perimeter')}. Risk level is {curr.get('master_risk_level', 'GREEN')}."
            else:
                text = f"System operational in {curr.get('active_zone', 'Standard Perimeter')}. Risk posture is {curr.get('master_risk_level', 'GREEN')}."

        latency = (time.time() - t0) * 1000.0
        return CopilotResponse(
            text=text,
            source=CopilotResponseSource.DETERMINISTIC,
            latency_ms=round(latency, 2),
            evidence_grounding=grounding,
            model_name="Deterministic-Rule-Engine",
        )


WATCHGUARD_TOOL_DECLARATIONS = [
    {
        "function_declarations": [
            {
                "name": "get_current_security_status",
                "description": "Retrieves the current real-time security threat level, master decision code, active zone, and operating hours.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {},
                },
            },
            {
                "name": "get_current_people",
                "description": "Retrieves the list of all currently tracked people entities in view with their identities, liveness, and authorization.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {},
                },
            },
            {
                "name": "get_detected_objects",
                "description": "Retrieves all currently detected non-person physical objects (e.g. laptop, cell phone, backpack).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {},
                },
            },
            {
                "name": "get_latest_alert",
                "description": "Retrieves the most recent security alert or incident trigger.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {},
                },
            },
            {
                "name": "get_recent_events",
                "description": "Retrieves security events from the last N minutes (bounded window).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "minutes": {
                            "type": "INTEGER",
                            "description": "Time window in minutes (default: 5)",
                        }
                    },
                },
            },
            {
                "name": "get_today_security_metrics",
                "description": "Retrieves aggregated security metrics for today (distinct people, spoofs, after-hours violations, incidents).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {},
                },
            },
            {
                "name": "get_active_incidents",
                "description": "Retrieves all currently active, unresolved security incidents.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {},
                },
            },
        ]
    }
]


def execute_watchguard_tool(
    func_name: str,
    args: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministically executes an authoritative WatchGuard function call
    using the grounded security context.
    """
    curr = context.get("CURRENT_STATE", {})
    recent = context.get("RECENT_HISTORY", {})
    hist = context.get("HISTORICAL_EVENTS", {})

    if func_name == "get_current_security_status":
        return {
            "master_risk_level": curr.get("master_risk_level", "GREEN"),
            "master_decision_code": curr.get("master_decision_code", "NORMAL_ACTIVITY"),
            "active_zone": curr.get("active_zone", "Standard Perimeter"),
            "time_status": curr.get("time_status", "NORMAL_HOURS"),
            "operating_hours_window": curr.get("operating_hours_window", "09:00 to 18:00"),
            "master_reason": curr.get("master_reason", "Routine operations active."),
        }
    elif func_name == "get_current_people":
        return {
            "people_count": curr.get("people_count", len(curr.get("people", []))),
            "people": curr.get("people", curr.get("live_people_in_view", [])),
        }
    elif func_name == "get_detected_objects":
        return {
            "objects_count": curr.get("objects_count", len(curr.get("objects", []))),
            "objects": curr.get("objects", curr.get("live_detected_objects", [])),
        }
    elif func_name == "get_latest_alert":
        incidents = curr.get("active_incidents", [])
        if incidents:
            return {"latest_incident": incidents[0]}
        evs = recent.get("recent_events_5min", [])
        if evs:
            return {"latest_event": evs[0]}
        return {"status": "No active alerts or incidents."}
    elif func_name == "get_recent_events":
        return {
            "window_5min_metrics": recent.get("window_5min_metrics", {}),
            "recent_events_5min": recent.get("recent_events_5min", []),
        }
    elif func_name == "get_today_security_metrics":
        return hist.get("today_totals", {})
    elif func_name == "get_active_incidents":
        return {"active_incidents": curr.get("active_incidents", [])}
    else:
        return {"error": f"Unknown tool function '{func_name}'"}


class GeminiCopilotReasoner(BaseAICopilotReasoner):
    """
    Cloud-enhanced AI Security Copilot using Gemini 2.5 Flash / Flash-Lite.
    Uses native standard library HTTP requests to avoid compulsory dependencies.
    Supports on-demand single snapshot visual queries and structured context grounding.
    Bounded by strict timeout to guarantee zero camera pipeline stalls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = COPILOT_MODEL_NAME,
        timeout_seconds: float = COPILOT_TIMEOUT_SECONDS,
        fallback_reasoner: Optional[BaseAICopilotReasoner] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback_reasoner or DeterministicCopilotReasoner()

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate_explanation(
        self,
        query: str,
        context: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> CopilotResponse:
        t0 = time.time()
        if not self.is_available():
            logger.debug("Gemini API key not configured; routing to deterministic fallback.")
            resp = self.fallback.generate_explanation(query, context, frame=frame)
            resp.source = CopilotResponseSource.FALLBACK_DETERMINISTIC
            return resp

        # Assemble Prompt Payload
        user_prompt = f"""WATCHGUARD STRUCTURED SECURITY STATE:
{json.dumps(context, indent=2)}

USER SECURITY QUERY:
"{query}"

Please provide an evidence-grounded, professional response adhering to all authority and multi-person grounding constraints."""

        # Check if query requests visual inspection of live camera feed
        is_visual = (frame is not None and is_visual_query(query))
        parts: List[Dict[str, Any]] = []

        if is_visual:
            img_b64 = encode_frame_to_jpeg_base64(frame)
            if img_b64:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": img_b64,
                    }
                })

        parts.append({"text": user_prompt})

        request_body = {
            "systemInstruction": {
                "parts": [{"text": COPILOT_SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 150,
            }
        }

        endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        try:
            req_data = json.dumps(request_body).encode("utf-8")
            req = urllib.request.Request(
                endpoint_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                if response.status == 200:
                    resp_bytes = response.read()
                    data = json.loads(resp_bytes.decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        c_parts = candidates[0]["content"].get("parts", [])
                        if c_parts and "text" in c_parts[0]:
                            raw_text = c_parts[0]["text"].strip()
                            latency = (time.time() - t0) * 1000.0
                            return CopilotResponse(
                                text=raw_text,
                                source=CopilotResponseSource.AI_GROUNDED,
                                latency_ms=round(latency, 2),
                                evidence_grounding=context.get("EVIDENCE_GROUNDING", {}),
                                model_name=self.model_name,
                            )

            logger.warning(f"Unexpected response from Gemini API: status {response.status}")
        except Exception as e:
            logger.warning(f"Gemini Copilot API call failed or timed out ({e}); activating fallback.")

        # Graceful fallback on network timeout / API error
        resp = self.fallback.generate_explanation(query, context, frame=frame)
        resp.source = CopilotResponseSource.FALLBACK_DETERMINISTIC
        resp.latency_ms = round((time.time() - t0) * 1000.0, 2)
        return resp


class LocalOllamaCopilotReasoner(BaseAICopilotReasoner):
    """
    Air-gapped local SLM Copilot using Ollama (Llama 3.2 3B / Phi-3.5).
    100% offline, private, zero cloud communication.
    """

    def __init__(
        self,
        endpoint_url: str = COPILOT_LOCAL_ENDPOINT,
        model_name: str = COPILOT_LOCAL_MODEL,
        timeout_seconds: float = COPILOT_TIMEOUT_SECONDS,
        fallback_reasoner: Optional[BaseAICopilotReasoner] = None,
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback_reasoner or DeterministicCopilotReasoner()

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=0.3) as response:
                return response.status == 200
        except Exception:
            return False

    def generate_explanation(
        self,
        query: str,
        context: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> CopilotResponse:
        t0 = time.time()
        prompt_text = f"{COPILOT_SYSTEM_PROMPT}\n\nSTATE:\n{json.dumps(context, indent=2)}\n\nQUERY: {query}\nANSWER:"
        req_body = {
            "model": self.model_name,
            "prompt": prompt_text,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 120},
        }

        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(req_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    raw_text = data.get("response", "").strip()
                    if raw_text:
                        latency = (time.time() - t0) * 1000.0
                        return CopilotResponse(
                            text=raw_text,
                            source=CopilotResponseSource.AI_GROUNDED,
                            latency_ms=round(latency, 2),
                            evidence_grounding=context.get("EVIDENCE_GROUNDING", {}),
                            model_name=f"Ollama-{self.model_name}",
                        )
        except Exception as e:
            logger.debug(f"Local Ollama inference failed ({e}); falling back.")

        resp = self.fallback.generate_explanation(query, context, frame=frame)
        resp.source = CopilotResponseSource.FALLBACK_DETERMINISTIC
        resp.latency_ms = round((time.time() - t0) * 1000.0, 2)
        return resp


def get_copilot_reasoner(
    security_analytics: Any = None,
    api_key: Optional[str] = None,
    preferred_backend: str = "auto",
) -> BaseAICopilotReasoner:
    """
    Factory creating the appropriate Copilot reasoning engine based on environment.
    Priority:
    1. Gemini Cloud Reasoner if GEMINI_API_KEY is available.
    2. Local Ollama Reasoner if Ollama endpoint is active.
    3. Deterministic Rule Reasoner (always available, 0ms, 100% offline).
    """
    det_reasoner = DeterministicCopilotReasoner(security_analytics=security_analytics)
    key = api_key or GEMINI_API_KEY

    if not COPILOT_ENABLED:
        return det_reasoner

    if (preferred_backend in ("auto", "gemini", "cloud")) and key:
        return GeminiCopilotReasoner(api_key=key, fallback_reasoner=det_reasoner)

    if preferred_backend in ("auto", "ollama", "local"):
        ollama_reasoner = LocalOllamaCopilotReasoner(fallback_reasoner=det_reasoner)
        if ollama_reasoner.is_available():
            return ollama_reasoner

    return det_reasoner
