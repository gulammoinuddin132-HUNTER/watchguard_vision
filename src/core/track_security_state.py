"""
WatchGuard Vision - Per-Face Security State & Zone Entry/Exit Manager (Phase 4.1)
---------------------------------------------------------------------------------
Maintains persistent, completely isolated runtime security states for each tracked face.
Enforces:
- 4-State Zone Occupancy State Machine (OUTSIDE -> ENTERING -> INSIDE -> EXITING)
- Transition-based violation semantics (edge-triggered, not frame-by-frame)
- Incident debouncing during continuous presence
- Accurate entry timestamps and duration inside zones
- Distinct entry counting for repeated violation escalations
- Presentation attack session debouncing
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.events.event_types import RiskLevel

logger = logging.getLogger("WatchGuardVision.TrackSecurityState")


class ZoneOccupancyState(str, Enum):
    OUTSIDE = "OUTSIDE"
    ENTERING = "ENTERING"
    INSIDE = "INSIDE"
    EXITING = "EXITING"


@dataclass
class TrackStateSnapshot:
    """Historical observation snapshot for temporal analysis."""
    timestamp: float
    time_str: str
    identity: str
    liveness_state: str
    liveness_conf: float
    zone: str
    zone_occupancy: str
    access_state: str
    risk_level: RiskLevel
    decision_code: str
    is_violation: bool


class TrackSecurityState:
    """
    Persistent runtime security state for a single tracked face.
    Completely isolated from all other tracks.
    """

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        self.identity: str = "Unknown Person"
        self.identity_confidence: float = 0.0
        self.role: str = "Visitor"
        self.is_known: bool = False
        self.authorization_status: str = "UNVERIFIED"

        self.liveness_state: str = "WARMUP"
        self.liveness_confidence: float = 0.0
        self.is_spoof: bool = False
        self.attack_type: Optional[str] = None

        # Zone state and duration tracking (Phase 4.1)
        self.zone: str = "NORMAL"
        self.zone_name: str = "Standard Perimeter"
        self.zone_occupancy: ZoneOccupancyState = ZoneOccupancyState.OUTSIDE
        self.zone_entry_time: Optional[float] = None
        self.zone_entry_time_str: Optional[str] = None
        self.duration_inside_zone_seconds: float = 0.0
        self.active_zone_incident_id: Optional[str] = None

        self.current_time: str = "--:--"
        self.time_policy_status: str = "PERMITTED_HOURS"
        self.nearby_objects: List[Dict[str, Any]] = []

        self.matched_policy: str = "NORMAL_ACTIVITY"
        self.access_state: str = "PENDING"
        self.risk_level: RiskLevel = RiskLevel.GREEN
        self.decision_code: str = "NORMAL_ACTIVITY"
        self.reason: str = "Initializing track state."

        now = time.time()
        self.state_since: float = now
        self.last_update: float = now

        # In-memory rolling history (max 30 snapshots)
        self.history: deque[TrackStateSnapshot] = deque(maxlen=30)

        # Distinct entry and violation records (Phase 4.1)
        self.distinct_entries: List[Dict[str, Any]] = []
        self.distinct_violations: List[Dict[str, Any]] = []

        # Distinct spoof attack sessions (Phase 4.1)
        self.active_spoof_incident_id: Optional[str] = None
        self.active_spoof_start_time: Optional[float] = None
        self.distinct_spoof_sessions: List[Dict[str, Any]] = []

        # Temporal confirmation counters
        self.consecutive_live_count: int = 0
        self.consecutive_spoof_count: int = 0
        self.consecutive_restricted_frames: int = 0
        self.consecutive_outside_frames: int = 0
        self.confirmed_live: bool = False
        self.confirmed_spoof: bool = False

    def update(
        self,
        identity: str,
        identity_conf: float,
        role: str,
        is_known: bool,
        liveness_state: str,
        liveness_conf: float,
        is_spoof: bool,
        attack_type: Optional[str],
        zone_id: str,
        zone_name: str,
        time_str: str,
        is_auth_time: bool,
        nearby_objects: List[Dict[str, Any]],
        policy_code: str,
        is_access_granted: bool,
        risk_level: RiskLevel,
        reason: str,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Updates this track's state, executes zone occupancy state machine,
        calculates durations, and returns transition event flags.
        """
        now = timestamp or time.time()
        prev_access = self.access_state
        prev_risk = self.risk_level
        prev_zone = self.zone
        prev_occupancy = self.zone_occupancy
        prev_spoof = self.is_spoof

        self.identity = identity
        self.identity_confidence = identity_conf
        self.role = role
        self.is_known = is_known
        self.authorization_status = "AUTHORIZED" if (is_known and is_access_granted) else ("UNREGISTERED_VISITOR" if not is_known else "DENIED")

        self.liveness_state = liveness_state
        self.liveness_confidence = liveness_conf
        self.is_spoof = is_spoof
        self.attack_type = attack_type

        # Update temporal confirmation counters
        if is_spoof or liveness_state in ("SPOOF", "STABLE_SPOOF"):
            self.consecutive_spoof_count += 1
            self.consecutive_live_count = 0
            if self.consecutive_spoof_count >= 3:
                self.confirmed_spoof = True
                self.confirmed_live = False
        elif liveness_state in ("LIVE", "STABLE_LIVE") and not is_spoof:
            self.consecutive_live_count += 1
            self.consecutive_spoof_count = 0
            if self.consecutive_live_count >= 3:
                self.confirmed_live = True
                self.confirmed_spoof = False
        else:
            self.consecutive_live_count = max(0, self.consecutive_live_count - 1)

        # -------------------------------------------------------------
        # Zone Entry / Exit State Machine (Phase 4.1)
        # -------------------------------------------------------------
        is_new_entry = False
        is_exit = False
        is_new_violation = False
        closed_incident_id = None

        if zone_id == "RESTRICTED":
            self.consecutive_restricted_frames += 1
            if prev_occupancy == ZoneOccupancyState.OUTSIDE or self.zone_entry_time is None:
                # Fresh entry transition: OUTSIDE -> ENTERING -> INSIDE
                self.zone_occupancy = ZoneOccupancyState.INSIDE
                self.zone_entry_time = now
                self.zone_entry_time_str = time_str
                self.duration_inside_zone_seconds = 0.0
                is_new_entry = True

                entry_record = {
                    "entry_time": now,
                    "time_str": time_str,
                    "zone": zone_id,
                    "identity": identity,
                }
                self.distinct_entries.append(entry_record)

                # Check if this entry constitutes a security violation
                if risk_level in (RiskLevel.ORANGE, RiskLevel.RED) or not is_auth_time or not is_known:
                    is_new_violation = True
                    violation_record = {
                        "violation_time": now,
                        "time_str": time_str,
                        "zone": zone_id,
                        "policy": policy_code,
                        "risk": risk_level.value,
                        "identity": identity,
                    }
                    self.distinct_violations.append(violation_record)
            else:
                # Continuous presence inside Restricted Area (INSIDE -> INSIDE)
                self.zone_occupancy = ZoneOccupancyState.INSIDE
                self.duration_inside_zone_seconds = max(0.0, now - (self.zone_entry_time or now))
                is_new_entry = False
                is_new_violation = False
        else:
            # Subject is in NORMAL or outside restricted zone
            self.consecutive_restricted_frames = 0
            if prev_occupancy == ZoneOccupancyState.INSIDE:
                # Exit transition: INSIDE -> EXITING -> OUTSIDE
                self.zone_occupancy = ZoneOccupancyState.OUTSIDE
                is_exit = True
                closed_incident_id = self.active_zone_incident_id
                self.active_zone_incident_id = None
                self.zone_entry_time = None
                self.duration_inside_zone_seconds = 0.0
            else:
                self.zone_occupancy = ZoneOccupancyState.OUTSIDE

        # -------------------------------------------------------------
        # Spoof Session Debouncing (Phase 4.1)
        # -------------------------------------------------------------
        is_new_spoof_session = False
        is_spoof_cleared = False

        if is_spoof:
            if not prev_spoof or self.active_spoof_start_time is None:
                # Fresh spoof presentation session
                self.active_spoof_start_time = now
                is_new_spoof_session = True
                self.distinct_spoof_sessions.append({
                    "start_time": now,
                    "time_str": time_str,
                    "attack_type": attack_type or "PHOTO_REPLAY_ATTACK",
                    "identity": identity,
                })
        else:
            if prev_spoof and self.active_spoof_start_time is not None:
                is_spoof_cleared = True
                self.active_spoof_incident_id = None
                self.active_spoof_start_time = None

        self.zone = zone_id
        self.zone_name = zone_name
        self.current_time = time_str
        self.time_policy_status = "PERMITTED_HOURS" if is_auth_time else "AFTER_HOURS"
        self.nearby_objects = nearby_objects

        self.matched_policy = policy_code
        self.decision_code = policy_code
        self.reason = reason

        if is_spoof or self.confirmed_spoof:
            self.access_state = "SPOOF_REJECTED"
        elif is_access_granted and self.confirmed_live:
            self.access_state = "GRANTED"
        elif policy_code == "LIVENESS_VERIFYING" or not self.confirmed_live and is_known:
            self.access_state = "PENDING"
        else:
            self.access_state = "DENIED"

        self.risk_level = risk_level

        # Track state_since duration
        if self.access_state != prev_access or self.risk_level != prev_risk:
            self.state_since = now
        self.last_update = now

        # Append snapshot to rolling history
        is_violation = risk_level in (RiskLevel.ORANGE, RiskLevel.RED)
        snapshot = TrackStateSnapshot(
            timestamp=now,
            time_str=time_str,
            identity=identity,
            liveness_state=liveness_state,
            liveness_conf=liveness_conf,
            zone=zone_id,
            zone_occupancy=self.zone_occupancy.value,
            access_state=self.access_state,
            risk_level=self.risk_level,
            decision_code=policy_code,
            is_violation=is_violation,
        )
        self.history.append(snapshot)

        return {
            "is_new_entry": is_new_entry,
            "is_exit": is_exit,
            "is_new_violation": is_new_violation,
            "is_new_spoof_session": is_new_spoof_session,
            "is_spoof_cleared": is_spoof_cleared,
            "closed_incident_id": closed_incident_id,
            "duration_inside_zone": self.duration_inside_zone_seconds,
        }

    def handle_exit(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        """Explicitly handles track exiting/disappearing from camera view."""
        now = now_ts or time.time()
        closed_id = None
        was_inside = (self.zone_occupancy == ZoneOccupancyState.INSIDE)

        if was_inside:
            closed_id = self.active_zone_incident_id
            self.active_zone_incident_id = None
            self.zone_occupancy = ZoneOccupancyState.OUTSIDE
            self.zone_entry_time = None
            self.duration_inside_zone_seconds = 0.0

        if self.active_spoof_incident_id:
            self.active_spoof_incident_id = None
            self.active_spoof_start_time = None

        return {
            "was_inside": was_inside,
            "closed_incident_id": closed_id,
        }

    def get_recent_distinct_violations(self, time_window_seconds: float = 600.0) -> List[Dict[str, Any]]:
        """Returns distinct entry-based violations within the time window."""
        now = time.time()
        return [
            v for v in self.distinct_violations
            if (now - v["violation_time"] <= time_window_seconds)
        ]

    def get_recent_distinct_entries(self, time_window_seconds: float = 600.0) -> List[Dict[str, Any]]:
        """Returns distinct physical entry events within the time window."""
        now = time.time()
        return [
            e for e in self.distinct_entries
            if (now - e["entry_time"] <= time_window_seconds)
        ]

    def get_recent_distinct_spoof_sessions(self, time_window_seconds: float = 300.0) -> List[Dict[str, Any]]:
        """Returns distinct presentation attack sessions within the window."""
        now = time.time()
        return [
            s for s in self.distinct_spoof_sessions
            if (now - s["start_time"] <= time_window_seconds)
        ]

    def get_recent_violations(self, time_window_seconds: float = 600.0) -> List[TrackStateSnapshot]:
        """Returns violation snapshots within the specified time window."""
        now = time.time()
        return [s for s in self.history if s.is_violation and (now - s.timestamp <= time_window_seconds)]

    def check_repeated_violations_escalation(self) -> Optional[Tuple[RiskLevel, str, str]]:
        """
        Evaluates distinct events for repeated violation patterns:
        - 3 DISTINCT restricted-zone entries outside permitted hours within 10m -> RED
        - 3 DISTINCT spoof presentation sessions within 5m -> RED
        - 3 DISTINCT unauthorized unknown access entries -> RED
        Returns: (escalated_risk, escalation_code, escalation_reason) or None
        """
        # 1. Repeated Restricted Zone Entries (3 distinct physical entries in 10 mins)
        distinct_viols = self.get_recent_distinct_violations(time_window_seconds=600.0)
        if len(distinct_viols) >= 3:
            return (
                RiskLevel.RED,
                "REPEATED_RESTRICTED_ZONE_VIOLATIONS",
                f"Repeated security violations detected: {len(distinct_viols)} distinct restricted-zone violations recorded within the last 10 minutes.",
            )

        # 2. Repeated Presentation Attacks (3 distinct sessions in 5 mins)
        spoof_sessions = self.get_recent_distinct_spoof_sessions(time_window_seconds=300.0)
        if len(spoof_sessions) >= 3:
            return (
                RiskLevel.RED,
                "REPEATED_PRESENTATION_ATTACK_ATTEMPTS",
                f"Repeated presentation attack attempts: {len(spoof_sessions)} distinct spoof sessions detected from this subject within 5 minutes.",
            )

        # 3. Repeated Unknown Visitor entries (3 distinct physical entries in 10 mins)
        if not self.is_known:
            distinct_entries = self.get_recent_distinct_entries(time_window_seconds=600.0)
            if len(distinct_entries) >= 3:
                return (
                    RiskLevel.RED,
                    "REPEATED_UNAUTHORIZED_ACCESS_ATTEMPTS",
                    f"Repeated unauthorized access attempts: {len(distinct_entries)} distinct restricted entries recorded for unregistered subject within 10 minutes.",
                )

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary representation of track security state."""
        return {
            "track_id": self.track_id,
            "identity": self.identity,
            "identity_confidence": self.identity_confidence,
            "role": self.role,
            "authorization_status": self.authorization_status,
            "liveness_state": self.liveness_state,
            "liveness_confidence": self.liveness_confidence,
            "is_spoof": self.is_spoof,
            "zone": self.zone,
            "zone_name": self.zone_name,
            "zone_occupancy": self.zone_occupancy.value,
            "zone_entry_time_str": self.zone_entry_time_str,
            "duration_inside_zone_seconds": round(self.duration_inside_zone_seconds, 1),
            "distinct_violation_count": len(self.distinct_violations),
            "distinct_entry_count": len(self.distinct_entries),
            "active_zone_incident_id": self.active_zone_incident_id,
            "current_time": self.current_time,
            "time_policy_status": self.time_policy_status,
            "matched_policy": self.matched_policy,
            "access_state": self.access_state,
            "risk_level": self.risk_level.value,
            "decision_code": self.decision_code,
            "reason": self.reason,
            "state_since": self.state_since,
            "last_update": self.last_update,
            "history_count": len(self.history),
        }


class TrackSecurityStateManager:
    """
    Manages collection of TrackSecurityState instances for all active tracks.
    Guarantees strict isolation between tracks.
    """

    def __init__(self):
        self._tracks: Dict[int, TrackSecurityState] = {}

    def get_or_create_track(self, track_id: int) -> TrackSecurityState:
        """Retrieves or creates isolated security state for track_id."""
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackSecurityState(track_id)
        return self._tracks[track_id]

    def get_track(self, track_id: int) -> Optional[TrackSecurityState]:
        """Returns track state if it exists."""
        return self._tracks.get(track_id)

    def get_all_tracks(self) -> List[TrackSecurityState]:
        """Returns list of all active track security states."""
        return list(self._tracks.values())

    def prune_stale_tracks(self, max_age_seconds: float = 60.0) -> int:
        """Removes tracks that have not been updated recently, closing any active zone presences."""
        now = time.time()
        stale_ids = [
            tid for tid, tstate in self._tracks.items()
            if now - tstate.last_update > max_age_seconds
        ]
        for tid in stale_ids:
            self._tracks[tid].handle_exit(now)
            del self._tracks[tid]
        return len(stale_ids)

    def reset(self) -> None:
        """Clears all track security states."""
        self._tracks.clear()
