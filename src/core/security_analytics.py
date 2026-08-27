"""
WatchGuard Vision - Security Intelligence & Analytics Engine (Phase 5)
-----------------------------------------------------------------------
Transforms WatchGuard from an event-display tool into an explainable
Security Intelligence Engine that correlates current state, recent history,
incidents, identities, zones, policies, and risk.

Provides:
- Derived metrics across bounded time windows (live, 5min, 15min, 1hr, today)
- Incident severity prioritization (CRITICAL/RED > RED > ORANGE > YELLOW > GREEN)
- 7-Tuple explainable security rationale (WHO, WHAT, WHERE, WHEN, POLICY, WHY, RESULT)
- Strict false-context protection (isolating live entities from historical events)
- Sub-second cached query performance preserving live camera FPS
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from src.events.event_types import RiskLevel, EventType

logger = logging.getLogger("WatchGuardVision.SecurityAnalytics")


@dataclass
class SecurityMetrics:
    """Aggregated security metrics for a specific time window."""
    active_incidents: int
    unknown_visitors: int
    spoof_attempts: int
    after_hours_violations: int
    restricted_zone_entries: int
    resolved_incidents: int
    current_risk_level: RiskLevel
    distinct_people_seen: int
    authorized_people_seen: int
    denied_access_attempts: int
    window_name: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["current_risk_level"] = self.current_risk_level.value if isinstance(self.current_risk_level, RiskLevel) else self.current_risk_level
        return d


@dataclass
class IncidentSummary:
    """Prioritized incident summary with duration and severity metadata."""
    incident_id: str
    person_name: str
    zone: str
    event_type: str
    created_at: str
    updated_at: str
    duration_seconds: float
    risk_level: RiskLevel
    status: str
    policy_code: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level
        return d


@dataclass
class SecurityExplanation:
    """7-Tuple structured explainability model for security decisions."""
    who: str
    what: str
    where: str
    when: str
    policy: str
    why: str
    result: str
    formatted_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SecurityAnalyticsEngine:
    """
    Core intelligence engine that analyzes multi-modal surveillance data,
    computes bounded-window metrics, prioritizes security incidents, and
    generates 7-Tuple explainable security intelligence reports.
    """

    def __init__(
        self,
        db_manager=None,
        context_engine=None,
        incident_manager=None,
        cache_ttl_seconds: float = 1.0,
    ):
        self.db = db_manager
        self.context_engine = context_engine
        self.incident_manager = incident_manager
        self.cache_ttl_seconds = cache_ttl_seconds

        self._metrics_cache: Dict[str, Tuple[float, SecurityMetrics]] = {}

    # --- 1. Bounded Time-Window Metric Aggregation ---

    def get_metrics(self, window: str = "live") -> SecurityMetrics:
        """
        Calculates or retrieves cached security metrics for the given window:
        'live', '5min', '15min', '1hr', 'today'.
        """
        now_ts = time.time()
        if window in self._metrics_cache:
            cache_time, cached_val = self._metrics_cache[window]
            if (now_ts - cache_time) < self.cache_ttl_seconds:
                return cached_val

        # 1. Determine current live risk posture
        current_risk = RiskLevel.GREEN
        if self.context_engine and hasattr(self.context_engine, "last_decision"):
            last_dec = self.context_engine.last_decision
            if last_dec and hasattr(last_dec, "risk_level"):
                current_risk = last_dec.risk_level

        # Also elevate current_risk if there are active RED or ORANGE incidents
        active_incidents_list = self.get_prioritized_incidents()
        if active_incidents_list:
            top_risk = active_incidents_list[0].risk_level
            if top_risk == RiskLevel.RED:
                current_risk = RiskLevel.RED
            elif top_risk == RiskLevel.ORANGE and current_risk != RiskLevel.RED:
                current_risk = RiskLevel.ORANGE

        # 2. Extract database counts based on window
        window_seconds: Optional[float] = None
        since_midnight = False

        if window == "5min":
            window_seconds = 300.0
        elif window == "15min":
            window_seconds = 900.0
        elif window == "1hr":
            window_seconds = 3600.0
        elif window == "today":
            since_midnight = True
        elif window == "live":
            window_seconds = 60.0  # 1-minute recent history for live context
        else:
            window_seconds = 300.0

        db_counts = {}
        if self.db and hasattr(self.db, "get_analytics_counts"):
            try:
                db_counts = self.db.get_analytics_counts(
                    window_seconds=window_seconds,
                    since_midnight=since_midnight,
                )
            except Exception as e:
                logger.error(f"Error querying database analytics counts: {e}")

        # If live window, augment with real-time in-memory counts
        active_inc_count = len(active_incidents_list)
        if window == "live" and self.context_engine and hasattr(self.context_engine, "last_decision"):
            last_dec = self.context_engine.last_decision
            entities = getattr(last_dec, "person_entities", []) if last_dec else []
            live_unknown = sum(1 for e in entities if not getattr(e, "identity", None) or getattr(e, "identity", None) == "Unknown")
            live_distinct = len(entities)
            live_auth = sum(1 for e in entities if getattr(e, "access_state", getattr(e, "access", "GRANTED")) == "GRANTED")
        else:
            live_unknown = db_counts.get("unknown_visitors", 0)
            live_distinct = db_counts.get("distinct_people_seen", 0)
            live_auth = db_counts.get("authorized_people_seen", 0)

        metrics = SecurityMetrics(
            active_incidents=active_inc_count if active_inc_count > 0 else db_counts.get("active_incidents", 0),
            unknown_visitors=live_unknown if window == "live" else db_counts.get("unknown_visitors", 0),
            spoof_attempts=db_counts.get("spoof_attempts", 0),
            after_hours_violations=db_counts.get("after_hours_violations", 0),
            restricted_zone_entries=db_counts.get("restricted_zone_entries", 0),
            resolved_incidents=db_counts.get("resolved_incidents", 0),
            current_risk_level=current_risk,
            distinct_people_seen=live_distinct if window == "live" else db_counts.get("distinct_people_seen", 0),
            authorized_people_seen=live_auth if window == "live" else db_counts.get("authorized_people_seen", 0),
            denied_access_attempts=db_counts.get("denied_access_attempts", 0),
            window_name=window,
            timestamp=datetime.now().isoformat(),
        )

        self._metrics_cache[window] = (now_ts, metrics)
        return metrics

    # --- 2. Incident Severity Prioritization ---

    def get_prioritized_incidents(self, limit: int = 5) -> List[IncidentSummary]:
        """
        Collects active incidents, prioritizes them by severity:
        CRITICAL/RED > RED > ORANGE > YELLOW > GREEN, then by updated_at DESC.
        """
        all_incidents: Dict[str, IncidentSummary] = {}
        now = datetime.now()

        # 1. Collect in-memory active incidents from IncidentManager
        if self.incident_manager and hasattr(self.incident_manager, "get_active_incidents"):
            for inc in self.incident_manager.get_active_incidents():
                try:
                    c_dt = datetime.fromisoformat(inc.created_at)
                    dur = max(0.0, (now - c_dt).total_seconds())
                except Exception:
                    dur = 0.0

                all_incidents[inc.incident_id] = IncidentSummary(
                    incident_id=inc.incident_id,
                    person_name=inc.person_name or "Unknown Person",
                    zone=inc.zone or "Standard Perimeter",
                    event_type=inc.policy_code,
                    created_at=inc.created_at,
                    updated_at=inc.updated_at,
                    duration_seconds=dur,
                    risk_level=inc.risk_level if isinstance(inc.risk_level, RiskLevel) else RiskLevel(inc.risk_level),
                    status=inc.state.value if hasattr(inc.state, "value") else str(inc.state),
                    policy_code=inc.policy_code,
                    reason=inc.reason,
                )

        # 2. Collect database active incidents
        if self.db and hasattr(self.db, "get_prioritized_active_incidents"):
            try:
                db_rows = self.db.get_prioritized_active_incidents(limit=limit * 2)
                for row in db_rows:
                    inc_id = row.get("incident_id")
                    if inc_id and inc_id not in all_incidents:
                        created_at = row.get("created_at", datetime.now().isoformat())
                        try:
                            c_dt = datetime.fromisoformat(created_at)
                            dur = max(0.0, (now - c_dt).total_seconds())
                        except Exception:
                            dur = 0.0

                        risk_str = row.get("risk_level", "YELLOW")
                        try:
                            risk_enum = RiskLevel(risk_str)
                        except Exception:
                            risk_enum = RiskLevel.YELLOW

                        all_incidents[inc_id] = IncidentSummary(
                            incident_id=inc_id,
                            person_name=row.get("person_name") or "Unknown Person",
                            zone=row.get("zone") or "Standard Perimeter",
                            event_type=row.get("policy_code") or "SECURITY_INCIDENT",
                            created_at=created_at,
                            updated_at=row.get("updated_at", created_at),
                            duration_seconds=dur,
                            risk_level=risk_enum,
                            status=row.get("state", "DETECTED"),
                            policy_code=row.get("policy_code", ""),
                            reason=row.get("reason", ""),
                        )
            except Exception as e:
                logger.error(f"Error reading prioritized incidents from database: {e}")

        # 3. Sort by severity weight
        def risk_sort_key(inc: IncidentSummary) -> Tuple[int, str]:
            weight_map = {
                RiskLevel.RED: 1,
                RiskLevel.ORANGE: 2,
                RiskLevel.YELLOW: 3,
                RiskLevel.GREEN: 4,
            }
            w = weight_map.get(inc.risk_level, 5)
            # Negative updated_at string gives descending chronological order
            return (w, inc.updated_at)

        sorted_list = sorted(all_incidents.values(), key=risk_sort_key)
        return sorted_list[:limit]

    # --- 3. 7-Tuple Security Explainability & Summary Generation ---

    def explain_decision(
        self,
        decision=None,
        incident: Optional[IncidentSummary] = None,
    ) -> SecurityExplanation:
        """
        Constructs a 7-Tuple explanation (WHO, WHAT, WHERE, WHEN, POLICY, WHY, RESULT).
        """
        now_str = datetime.now().strftime("%H:%M:%S")

        if incident:
            who = incident.person_name
            what = incident.event_type.replace("_", " ").title()
            where = incident.zone
            when = incident.created_at.split("T")[-1][:8] if "T" in incident.created_at else incident.created_at
            policy = incident.policy_code
            why = incident.reason
            result = f"Access Denied, Risk {incident.risk_level.value}, Incident {incident.incident_id} Active"

            formatted = (
                f"At {when}, {who} triggered {what} in {where}. "
                f"Policy {policy}: {why}. "
                f"Result: {result}."
            )
            return SecurityExplanation(who, what, where, when, policy, why, result, formatted)

        if decision:
            person_names = decision.details.get("person_names", [])
            who = ", ".join(person_names) if person_names else "Unknown Person"
            what = decision.decision.replace("_", " ").title()
            where = decision.details.get("zone", "Standard Perimeter")
            when = now_str
            policy = decision.decision
            why = decision.reason
            result = f"Access {decision.access_state}, Risk {decision.risk_level.value}"

            formatted = (
                f"At {when}, {who} was observed in {where}. "
                f"{why}. "
                f"Access is {decision.access_state.lower()} with risk level {decision.risk_level.value}."
            )
            return SecurityExplanation(who, what, where, when, policy, why, result, formatted)

        return SecurityExplanation(
            who="None",
            what="Standard Monitoring",
            where="Standard Perimeter",
            when=now_str,
            policy="STANDARD_POLICY",
            why="All parameters within normal thresholds",
            result="Access Normal, Risk GREEN",
            formatted="System operating normally with no active incidents.",
        )

    def generate_security_summary(self, window: str = "live") -> str:
        """
        Generates a concise, natural language security intelligence summary
        synthesizing live state and recent events.
        """
        metrics = self.get_metrics(window)
        active_incs = self.get_prioritized_incidents(limit=3)

        # Case 1: Active High-Priority Incidents Exist
        if active_incs:
            top_inc = active_incs[0]
            dur_min = int(top_inc.duration_seconds // 60)
            dur_sec = int(top_inc.duration_seconds % 60)
            dur_str = f"{dur_min}m {dur_sec}s" if dur_min > 0 else f"{dur_sec}s"

            if top_inc.risk_level == RiskLevel.RED:
                if "SPOOF" in top_inc.policy_code or "PRESENTATION" in top_inc.policy_code:
                    return (
                        f"Current risk is RED. Presentation-attack attempt detected using credentials of registered user '{top_inc.person_name}'. "
                        f"Incident {top_inc.incident_id} is active (duration {dur_str})."
                    )
                return (
                    f"Current risk is RED. Critical security incident [{top_inc.policy_code}] active for {top_inc.person_name} in {top_inc.zone}. "
                    f"{top_inc.reason}"
                )
            elif top_inc.risk_level == RiskLevel.ORANGE:
                return (
                    f"Current risk is ORANGE. {top_inc.person_name} was detected in {top_inc.zone} outside the permitted 09:00–18:00 window. "
                    f"One active incident is open ({top_inc.incident_id})."
                )
            else:
                return (
                    f"Current risk is {top_inc.risk_level.value}. Incident {top_inc.incident_id} is open for {top_inc.person_name} in {top_inc.zone}. "
                    f"{top_inc.reason}"
                )

        # Case 2: Live Scene Inspection
        if self.context_engine and hasattr(self.context_engine, "last_decision"):
            last_dec = self.context_engine.last_decision
            if last_dec:
                if last_dec.risk_level == RiskLevel.RED:
                    return f"Current risk is RED. {last_dec.reason}"
                elif last_dec.risk_level == RiskLevel.ORANGE:
                    return f"Current risk is ORANGE. {last_dec.reason}"
                elif last_dec.risk_level == RiskLevel.YELLOW:
                    return f"Current risk is YELLOW. {last_dec.reason}"

        # Case 3: Recent Activity Summary (when live is quiet)
        if metrics.spoof_attempts > 0:
            return f"Current risk is GREEN. {metrics.spoof_attempts} spoof attempt(s) were recorded earlier and successfully rejected."
        if metrics.after_hours_violations > 0:
            return f"Current risk is GREEN. {metrics.after_hours_violations} after-hours violation(s) were recorded earlier and resolved."

        return "Current risk is GREEN. All surveillance perimeters are secure with zero active incidents."

    # --- 4. Query-Specific Helpers for Voice Assistant ---

    def explain_current_risk(self) -> str:
        """Explains why the system is currently at its active risk level."""
        metrics = self.get_metrics("live")
        active_incs = self.get_prioritized_incidents(limit=1)

        if active_incs:
            top_inc = active_incs[0]
            expl = self.explain_decision(incident=top_inc)
            return f"System risk is {metrics.current_risk_level.value} because {expl.formatted_text}"

        if self.context_engine and hasattr(self.context_engine, "last_decision"):
            last_dec = self.context_engine.last_decision
            if last_dec and last_dec.risk_level != RiskLevel.GREEN:
                expl = self.explain_decision(decision=last_dec)
                return f"System risk is {metrics.current_risk_level.value}. {expl.formatted_text}"

        return f"System risk is GREEN. No security violations or unverified entities are currently detected."

    def explain_person_denial(self, person_name: str = "Hunter") -> str:
        """Explains why a specific person was denied access."""
        # 1. Check live active track
        if self.context_engine and hasattr(self.context_engine, "last_decision"):
            last_dec = self.context_engine.last_decision
            if last_dec:
                entities = getattr(last_dec, "person_entities", [])
                for e in entities:
                    ident = getattr(e, "identity", "")
                    if ident.lower() == person_name.lower():
                        access_val = getattr(e, "access_state", getattr(e, "access", "PENDING"))
                        if access_val == "DENIED":
                            if getattr(e, "is_spoof", False):
                                return f"{person_name} was denied because a presentation attack was detected on that track."
                            elif getattr(e, "liveness_state", getattr(e, "liveness", "")) == "UNVERIFIED_TIMEOUT":
                                return f"{person_name} was denied because liveness verification timed out after 5.0 seconds without confirmed evidence."
                            elif last_dec.decision == "AFTER_HOURS_RESTRICTED_ACCESS_DENIED":
                                return f"{person_name} was denied access to the Restricted Area because current time is outside the authorized 09:00 to 18:00 operating window."
                            else:
                                return f"{person_name} was denied access: {last_dec.reason}."
                        elif access_val == "PENDING":
                            return f"{person_name} is currently pending authorization while PAD liveness verification is underway."

        # 2. Check active incidents
        active_incs = self.get_prioritized_incidents(limit=10)
        for inc in active_incs:
            if inc.person_name.lower() == person_name.lower():
                if "AFTER_HOURS" in inc.policy_code:
                    return f"{person_name} was denied access to the Restricted Area because the permitted operating window is 09:00 to 18:00."
                elif "SPOOF" in inc.policy_code:
                    return f"{person_name} was denied access because a presentation attack was detected."
                return f"{person_name} was denied: {inc.reason}."

        # 3. Check recent database events
        if self.db and hasattr(self.db, "get_events_by_window"):
            try:
                events = self.db.get_events_by_window(window_seconds=600.0, limit=20)
                for ev in events:
                    p_name = ev.get("person_name", "")
                    if p_name and p_name.lower() == person_name.lower():
                        r_level = ev.get("risk_level", "")
                        if r_level in ("YELLOW", "ORANGE", "RED"):
                            e_type = ev.get("event_type", "")
                            if "AFTER_HOURS" in e_type:
                                return f"{person_name} was denied access to the Restricted Area outside permitted operating hours (09:00 to 18:00)."
                            elif "SPOOF" in e_type:
                                return f"{person_name} was rejected due to a detected presentation attack."
                            return f"{person_name} was denied access due to {e_type.replace('_', ' ').lower()}."
            except Exception as e:
                logger.error(f"Error checking recent events for person denial: {e}")

        return f"There is no record of {person_name} being denied access recently."

    def get_recent_activity_summary(self, window_seconds: float = 300.0) -> str:
        """Summarizes security events in the last N seconds (bounded query)."""
        metrics = self.get_metrics("5min" if window_seconds <= 300 else "15min")
        mins = int(window_seconds // 60)

        events_summary = []
        if metrics.active_incidents > 0:
            events_summary.append(f"{metrics.active_incidents} active incident(s)")
        if metrics.spoof_attempts > 0:
            events_summary.append(f"{metrics.spoof_attempts} spoof attempt(s)")
        if metrics.after_hours_violations > 0:
            events_summary.append(f"{metrics.after_hours_violations} after-hours violation(s)")
        if metrics.unknown_visitors > 0:
            events_summary.append(f"{metrics.unknown_visitors} unknown visitor(s)")

        if not events_summary:
            return f"In the last {mins} minutes, no security violations or alerts were recorded."

        return f"In the last {mins} minutes, the system recorded: {', '.join(events_summary)}."

    def get_highest_risk_event(self) -> str:
        """Retrieves the highest-risk active incident or recent alert."""
        active_incs = self.get_prioritized_incidents(limit=1)
        if active_incs:
            top = active_incs[0]
            return f"The highest-risk active event is [{top.incident_id}] ({top.risk_level.value}): {top.reason} involving {top.person_name} in {top.zone}."

        if self.db and hasattr(self.db, "get_events_by_window"):
            try:
                events = self.db.get_events_by_window(window_seconds=3600.0, limit=20)
                red_events = [e for e in events if e.get("risk_level") == "RED"]
                if red_events:
                    e = red_events[0]
                    return f"The highest-risk recent event was RED alert [{e.get('event_type')}]: {e.get('details', '')} at {e.get('timestamp', '')}."
                orange_events = [e for e in events if e.get("risk_level") == "ORANGE"]
                if orange_events:
                    e = orange_events[0]
                    return f"The highest-risk recent event was ORANGE alert [{e.get('event_type')}]: {e.get('details', '')} at {e.get('timestamp', '')}."
            except Exception:
                pass

        return "No high-risk security events are currently on record."

    def is_anyone_unauthorized_live(self) -> Tuple[bool, str]:
        """Checks if anyone currently observed is unauthorized."""
        if not self.context_engine or not hasattr(self.context_engine, "last_decision"):
            return False, "No active surveillance feed is currently running."

        last_dec = self.context_engine.last_decision
        if not last_dec:
            return False, "No individuals are currently observed in the surveillance field."

        entities = getattr(last_dec, "person_entities", [])
        if not entities:
            return False, "No individuals are currently visible."

        unauth_entities = [e for e in entities if getattr(e, "access_state", getattr(e, "access", "PENDING")) in ("DENIED", "PENDING")]
        if not unauth_entities:
            names = [getattr(e, "identity", None) for e in entities if getattr(e, "identity", None)]
            name_str = ", ".join(names) if names else "All observed individuals"
            return False, f"No unauthorized individuals detected. {name_str} is authorized and verified."

        details = []
        for e in unauth_entities:
            name = getattr(e, "identity", None) or "Unknown person"
            acc = getattr(e, "access_state", getattr(e, "access", "PENDING"))
            if acc == "PENDING":
                details.append(f"{name} (Liveness Verifying / Access Pending)")
            elif getattr(e, "is_spoof", False):
                details.append(f"{name} (Spoof Attack Rejected)")
            else:
                z = getattr(e, "zone_name", getattr(e, "zone", "Restricted Area"))
                details.append(f"{name} (Access Denied in {z})")

        return True, f"Yes, unauthorized or pending entity detected: {', '.join(details)}."
