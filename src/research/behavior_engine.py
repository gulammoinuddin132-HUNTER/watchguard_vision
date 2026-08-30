"""
WatchGuard Vision - Behavioral & Temporal Anomaly Engine (Phase 5.20)
---------------------------------------------------------------------
Tracks persistent per-track spatial and temporal movement metrics:
  - Dwell time in sensitive / restricted surveillance zones
  - Approach and retreat frequency to protected assets
  - Repeated entry / retry counts
  - After-hours presence anomalies
Produces normalized S_behavior risk signal in [0.0, 1.0].
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import logging

from config import (
    BEHAVIOR_DWELL_THRESHOLD_SECONDS,
    BEHAVIOR_APPROACH_REPETITION_THRESHOLD,
    BEHAVIOR_RESTRICTED_RETRY_THRESHOLD,
)

logger = logging.getLogger("WatchGuardVision.BehaviorEngine")


@dataclass
class TrackBehaviorMetrics:
    """Per-track behavioral and spatial telemetry metrics."""
    track_id: int
    identity: str = "Unknown Person"
    is_known: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    current_zone: str = "NORMAL"
    zone_dwell_start: float = field(default_factory=time.time)
    restricted_dwell_seconds: float = 0.0
    asset_proximity_dwell_seconds: float = 0.0
    
    zone_entries: int = 0
    zone_exits: int = 0
    restricted_entries: int = 0
    protected_asset_approaches: int = 0
    approach_count: int = 0
    retreat_count: int = 0
    after_hours_presence: bool = False
    repeated_violation_count: int = 0

    last_asset_distance: float = 999.0
    is_currently_near_asset: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BehaviorAnomalyEngine:
    """
    Lightweight, track-persistent research engine for behavioral pattern analysis.
    Evaluates spatial telemetry transitions without per-frame heavy re-compute.
    """

    def __init__(
        self,
        dwell_threshold: float = BEHAVIOR_DWELL_THRESHOLD_SECONDS,
        approach_threshold: int = BEHAVIOR_APPROACH_REPETITION_THRESHOLD,
        restricted_retry_threshold: int = BEHAVIOR_RESTRICTED_RETRY_THRESHOLD,
    ):
        self.dwell_threshold = dwell_threshold
        self.approach_threshold = approach_threshold
        self.restricted_retry_threshold = restricted_retry_threshold
        self._track_metrics: Dict[int, TrackBehaviorMetrics] = {}

    def get_or_create_metrics(self, track_id: int, identity: str = "Unknown Person", is_known: bool = False) -> TrackBehaviorMetrics:
        if track_id not in self._track_metrics:
            now = time.time()
            self._track_metrics[track_id] = TrackBehaviorMetrics(
                track_id=track_id,
                identity=identity,
                is_known=is_known,
                first_seen=now,
                last_seen=now,
                zone_dwell_start=now,
            )
        return self._track_metrics[track_id]

    def update_track_observation(
        self,
        track_id: int,
        identity: str,
        is_known: bool,
        zone: str,
        is_after_hours: bool = False,
        nearest_asset_dist: float = 999.0,
        is_violation: bool = False,
        current_time: Optional[float] = None,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Updates telemetry for a specific person track and computes S_behavior in [0.0, 1.0].
        
        Returns:
            (S_behavior_risk, anomaly_status_string, metrics_dict)
        """
        now = current_time if current_time is not None else time.time()
        m = self.get_or_create_metrics(track_id, identity=identity, is_known=is_known)
        
        dt = max(0.0, now - m.last_seen)
        m.last_seen = now
        m.identity = identity
        m.is_known = is_known
        m.after_hours_presence = is_after_hours

        # 1. Zone transition & Dwell tracking
        zone_upper = (zone or "NORMAL").upper()
        if zone_upper != m.current_zone.upper():
            m.zone_entries += 1
            m.current_zone = zone_upper
            m.zone_dwell_start = now
            if "RESTRICTED" in zone_upper:
                m.restricted_entries += 1
        else:
            # Accumulate dwell in sensitive zones
            if "RESTRICTED" in zone_upper or "PROTECTED" in zone_upper:
                m.restricted_dwell_seconds += dt

        # 2. Asset approach / retreat tracking
        if nearest_asset_dist <= 150.0:
            if not m.is_currently_near_asset:
                m.protected_asset_approaches += 1
                m.approach_count += 1
                m.is_currently_near_asset = True
            m.asset_proximity_dwell_seconds += dt
        else:
            if m.is_currently_near_asset:
                m.retreat_count += 1
                m.is_currently_near_asset = False

        m.last_asset_distance = nearest_asset_dist

        # 3. Violation count
        if is_violation:
            m.repeated_violation_count += 1

        # 4. Calculate Normalized Behavioral Risk S_behavior
        # Factors:
        #  - Excessive restricted dwell (0.0 -> 0.40)
        #  - Repeated asset approaches (0.0 -> 0.35)
        #  - Repeated restricted zone retries (0.0 -> 0.40)
        #  - After-hours presence (0.0 -> 0.30)
        dwell_risk = min(0.40, round((m.restricted_dwell_seconds / max(1.0, self.dwell_threshold)) * 0.40, 3))
        approach_risk = min(0.35, round((m.protected_asset_approaches / max(1.0, self.approach_threshold)) * 0.35, 3))
        retry_risk = min(0.40, round((m.restricted_entries / max(1.0, self.restricted_retry_threshold)) * 0.40, 3))
        after_hours_risk = 0.30 if is_after_hours and ("RESTRICTED" in zone_upper or m.protected_asset_approaches > 0) else 0.0

        raw_s_behavior = min(1.0, dwell_risk + approach_risk + retry_risk + after_hours_risk)
        s_behavior = round(raw_s_behavior, 3)

        # Status categorization (neutral terminology)
        if s_behavior >= 0.70:
            status = "BEHAVIOR_ANOMALY"
        elif s_behavior >= 0.35:
            status = "UNUSUAL_PATTERN"
        else:
            status = "NORMAL_ACTIVITY"

        return s_behavior, status, m.to_dict()

    def cleanup_lost_tracks(self, active_track_ids: List[int], timeout_seconds: float = 30.0) -> None:
        """Removes tracks inactive for longer than timeout."""
        now = time.time()
        active_set = set(active_track_ids)
        stale_ids = [
            tid for tid, m in self._track_metrics.items()
            if tid not in active_set and (now - m.last_seen > timeout_seconds)
        ]
        for tid in stale_ids:
            self._track_metrics.pop(tid, None)
