"""
WatchGuard Vision - Incident Lifecycle Manager (Phase 4.1)
----------------------------------------------------------
Manages the complete lifecycle of ORANGE/RED security incidents:
DETECTED -> VERIFIED -> ALERTED -> ACKNOWLEDGED -> RESOLVED

Provides:
- Edge-triggered distinct incident creation and state progression
- Debounced continuous presence updates without duplicate records
- Track-associated active incident lifecycle & auto-closure on zone exit
- Structured evidence packaging
- Operator action handling (Acknowledge, Resolve)
- Seamless SQLite persistence via DatabaseManager
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import time
from typing import Dict, Any, List, Optional, Set
import uuid
import logging

from src.events.event_types import RiskLevel, EventType
from src.database.models import IncidentRecord

logger = logging.getLogger("WatchGuardVision.IncidentManager")


class IncidentState(str, Enum):
    DETECTED = "DETECTED"
    VERIFIED = "VERIFIED"
    ALERTED = "ALERTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class SecurityIncident:
    """Represents an active or historical security incident."""
    incident_id: str
    risk_level: RiskLevel
    state: IncidentState
    policy_code: str
    reason: str
    zone: str
    created_at: str
    updated_at: str
    event_id: Optional[int] = None
    track_id: Optional[int] = None
    person_name: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    operator_action: Optional[str] = None
    evidence_package: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level
        d["state"] = self.state.value if isinstance(self.state, IncidentState) else self.state
        return d


class IncidentManager:
    """
    Coordinates incident lifecycle tracking, evidence packaging,
    operator state transitions, and SQLite synchronization.
    """

    def __init__(self, db_manager=None, incident_cooldown_seconds: float = 30.0):
        self.db = db_manager
        self.cooldown_seconds = incident_cooldown_seconds
        self._active_incidents: Dict[str, SecurityIncident] = {}
        self._resolved_incident_ids: Set[str] = set()
        self._latest_incident: Optional[SecurityIncident] = None
        self._last_incident_time_per_sig: Dict[str, float] = {}
        self._incident_seq = 1

    def _generate_incident_id(self) -> str:
        """Generates a sequential human-readable incident identifier (e.g. INC-20260824-001)."""
        ts = datetime.now().strftime("%Y%m%d")
        inc_id = f"INC-{ts}-{self._incident_seq:03d}"
        self._incident_seq += 1
        return inc_id

    def process_incident(
        self,
        risk_level: RiskLevel,
        policy_code: str,
        reason: str,
        zone_name: str,
        person_name: Optional[str] = None,
        track_id: Optional[int] = None,
        event_id: Optional[int] = None,
        evidence_snapshot_path: Optional[str] = None,
        identity_confidence: float = 0.0,
        liveness_state: str = "LIVE",
        liveness_confidence: float = 0.0,
        time_status: str = "PERMITTED_HOURS",
        nearby_objects: Optional[List[str]] = None,
        duration_seconds: float = 0.0,
        active_incident_id: Optional[str] = None,
        force_new: bool = False,
    ) -> Optional[SecurityIncident]:
        """
        Creates or updates a security incident for ORANGE or RED risk levels.
        If an active incident already exists for this track/policy, updates its duration
        without creating duplicate incident IDs or duplicate database records.
        """
        if risk_level not in (RiskLevel.ORANGE, RiskLevel.RED):
            return None

        now_iso = datetime.now().isoformat()

        # Check if active incident already exists
        if active_incident_id and active_incident_id in self._active_incidents:
            active_inc = self._active_incidents[active_incident_id]
            if active_inc.state != IncidentState.RESOLVED and not force_new:
                active_inc.updated_at = now_iso
                active_inc.evidence_package["duration_seconds"] = round(duration_seconds, 1)
                active_inc.reason = reason
                return active_inc

        # Check by track_id & policy_code
        if not force_new and track_id is not None:
            for inc in self._active_incidents.values():
                if inc.track_id == track_id and inc.policy_code == policy_code and inc.state != IncidentState.RESOLVED:
                    inc.updated_at = now_iso
                    inc.evidence_package["duration_seconds"] = round(duration_seconds, 1)
                    inc.reason = reason
                    return inc

        # Deduplicate active (unresolved) incident matching signature across active pool
        if not force_new:
            for inc in self._active_incidents.values():
                if inc.state != IncidentState.RESOLVED and inc.policy_code == policy_code:
                    same_person = (
                        (inc.person_name == person_name)
                        or (not inc.person_name and not person_name)
                        or (person_name in ("Unknown", "Unregistered", None) and inc.person_name in ("Unknown", "Unregistered", None))
                    )
                    if same_person:
                        inc.updated_at = now_iso
                        inc.evidence_package["duration_seconds"] = round(duration_seconds, 1)
                        inc.reason = reason
                        return inc

        sig = f"{policy_code}_{person_name or 'Unregistered'}_{zone_name}"
        now_ts = time.time()
        self._last_incident_time_per_sig[sig] = now_ts

        # 1. Build Evidence Package
        evidence_pkg = {
            "primary_snapshot": evidence_snapshot_path,
            "timestamp": now_iso,
            "track_id": track_id,
            "identity_result": {
                "name": person_name or "Unregistered",
                "confidence": identity_confidence,
            },
            "liveness_result": {
                "state": liveness_state,
                "confidence": liveness_confidence,
            },
            "zone": zone_name,
            "time_status": time_status,
            "matched_policy": policy_code,
            "detected_objects": nearby_objects or [],
            "risk_level": risk_level.value,
            "reason": reason,
            "duration_seconds": round(duration_seconds, 1),
        }

        # 2. Instantiate New Incident in ALERTED state
        incident_id = self._generate_incident_id()
        incident = SecurityIncident(
            incident_id=incident_id,
            event_id=event_id,
            track_id=track_id,
            person_name=person_name if person_name != "None" else None,
            risk_level=risk_level,
            state=IncidentState.ALERTED,
            policy_code=policy_code,
            reason=reason,
            zone=zone_name,
            created_at=now_iso,
            updated_at=now_iso,
            evidence_package=evidence_pkg,
        )

        # 3. Store in Memory
        self._active_incidents[incident_id] = incident
        self._latest_incident = incident

        # 4. Persist to Database if available
        if self.db:
            try:
                rec = IncidentRecord(
                    incident_id=incident.incident_id,
                    event_id=incident.event_id,
                    track_id=incident.track_id,
                    person_name=incident.person_name,
                    risk_level=incident.risk_level.value,
                    state=incident.state.value,
                    policy_code=incident.policy_code,
                    reason=incident.reason,
                    zone=incident.zone,
                    created_at=incident.created_at,
                    updated_at=incident.updated_at,
                    evidence_package_json=json.dumps(evidence_pkg),
                )
                db_id = self.db.create_incident(rec)
                incident.id = db_id
            except Exception as e:
                logger.debug(f"Error persisting incident to database: {e}")

        logger.info(f"INCIDENT_DISPATCHED: [{incident.incident_id}] {risk_level.value} - {policy_code}: {reason}")
        return incident

    def close_incident(
        self,
        incident_id: str,
        reason: str = "Subject exited restricted perimeter",
    ) -> bool:
        """Closes/resolves an active incident when the subject exits the zone."""
        return self.resolve_incident(incident_id, operator=f"System ({reason})")

    def acknowledge_incident(
        self,
        incident_id: str,
        operator: str = "Operator",
    ) -> bool:
        """Transitions incident to ACKNOWLEDGED state."""
        incident = self._active_incidents.get(incident_id)
        if not incident and self.db:
            row = self.db.get_incident_by_id(incident_id)
            if row:
                incident = self._row_to_incident(row)
                self._active_incidents[incident_id] = incident

        if not incident:
            logger.warning(f"Acknowledge failed: Incident '{incident_id}' not found.")
            return False

        now_iso = datetime.now().isoformat()
        incident.state = IncidentState.ACKNOWLEDGED
        incident.acknowledged_at = now_iso
        incident.updated_at = now_iso
        incident.operator_action = f"Acknowledged by {operator}"

        if self.db:
            try:
                self.db.acknowledge_incident(incident_id, operator_action=incident.operator_action)
            except Exception as e:
                logger.debug(f"Database error acknowledging incident: {e}")

        logger.info(f"INCIDENT_ACKNOWLEDGED: [{incident_id}] by {operator}")
        return True

    def resolve_incident(
        self,
        incident_id: str,
        operator: str = "Operator",
    ) -> bool:
        """Transitions incident to RESOLVED state and removes from active tracking."""
        if incident_id in self._resolved_incident_ids:
            # Already resolved; strict no-op
            return True

        incident = self._active_incidents.get(incident_id)
        if not incident and self.db:
            row = self.db.get_incident_by_id(incident_id)
            if row:
                incident = self._row_to_incident(row)

        if not incident:
            logger.warning(f"Resolve failed: Incident '{incident_id}' not found.")
            return False

        if incident.state == IncidentState.RESOLVED:
            # Already resolved; record in tracking set and avoid duplicate logging/DB updates
            self._resolved_incident_ids.add(incident_id)
            return True

        now_iso = datetime.now().isoformat()
        incident.state = IncidentState.RESOLVED
        incident.resolved_at = now_iso
        incident.updated_at = now_iso
        incident.operator_action = f"Resolved by {operator}"

        if incident_id in self._active_incidents:
            del self._active_incidents[incident_id]
        self._resolved_incident_ids.add(incident_id)

        if self.db:
            try:
                self.db.resolve_incident(incident_id, operator_action=incident.operator_action)
            except Exception as e:
                logger.debug(f"Database error resolving incident: {e}")

        logger.info(f"INCIDENT_RESOLVED: [{incident_id}] by {operator}")
        return True

    def get_latest_incident(self) -> Optional[SecurityIncident]:
        """Returns the most recently recorded security incident."""
        if self._latest_incident:
            return self._latest_incident
        if self.db:
            row = self.db.get_latest_incident()
            if row:
                return self._row_to_incident(row)
        return None

    def get_active_incidents(self) -> List[SecurityIncident]:
        """Returns all unresolved incidents."""
        return list(self._active_incidents.values())

    def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        """Looks up incident by ID."""
        if incident_id in self._active_incidents:
            return self._active_incidents[incident_id]
        if self.db:
            row = self.db.get_incident_by_id(incident_id)
            if row:
                return self._row_to_incident(row)
        return None

    def _row_to_incident(self, row: Dict[str, Any]) -> SecurityIncident:
        """Converts database row dictionary to SecurityIncident instance."""
        pkg = {}
        if row.get("evidence_package_json"):
            try:
                pkg = json.loads(row["evidence_package_json"])
            except Exception:
                pkg = {}

        return SecurityIncident(
            incident_id=row["incident_id"],
            risk_level=RiskLevel(row["risk_level"]) if row.get("risk_level") in RiskLevel._value2member_map_ else RiskLevel.ORANGE,
            state=IncidentState(row["state"]) if row.get("state") in IncidentState._value2member_map_ else IncidentState.DETECTED,
            policy_code=row["policy_code"],
            reason=row["reason"],
            zone=row["zone"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            event_id=row.get("event_id"),
            track_id=row.get("track_id"),
            person_name=row.get("person_name"),
            acknowledged_at=row.get("acknowledged_at"),
            resolved_at=row.get("resolved_at"),
            operator_action=row.get("operator_action"),
            evidence_package=pkg,
            id=row.get("id"),
        )

    def clear_active_incidents(self) -> None:
        """Clears in-memory active incidents and resets incident sequence counter."""
        self._active_incidents.clear()
        self._resolved_incident_ids.clear()
        self._latest_incident = None
        self._last_incident_time_per_sig.clear()
        self._incident_seq = 1
        logger.info("[INCIDENT] In-memory active incidents cleared and sequence reset.")
