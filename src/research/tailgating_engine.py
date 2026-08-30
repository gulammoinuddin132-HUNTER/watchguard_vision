"""
WatchGuard Vision - Tailgating & Multi-Person Access Anomaly Detector (Phase 5.20)
----------------------------------------------------------------------------------
Detects suspicious physical piggybacking and tailgating patterns when an unauthorized
or unregistered individual closely follows an authorized person into a secured zone.
Enforces multi-observation temporal confirmation and edge-triggered event debouncing.
"""

import time
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import logging

from config import (
    TAILGATING_TIME_WINDOW_SECONDS,
    TAILGATING_SPATIAL_THRESHOLD_PIXELS,
    TAILGATING_CONFIRMATION_FRAMES,
    TAILGATING_ALERT_COOLDOWN_SECONDS,
)

logger = logging.getLogger("WatchGuardVision.TailgatingEngine")


@dataclass
class TailgatingEvent:
    """Structured representation of a suspected tailgating incident."""
    authorized_track_id: int
    follower_track_id: int
    time_delta: float
    spatial_distance: float
    authorized_identity: str
    follower_identity: str
    confidence: float
    zone: str
    timestamp: float = field(default_factory=time.time)
    is_confirmed: bool = False
    status: str = "TAILGATING_SUSPECTED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_box_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Returns centroid (cx, cy) of bounding box (x, y, w, h)."""
    x, y, w, h = bbox
    return float(x + w * 0.5), float(y + h * 0.5)


def compute_centroid_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Computes Euclidean distance between box centroids."""
    cx1, cy1 = compute_box_center(bbox1)
    cx2, cy2 = compute_box_center(bbox2)
    return math.hypot(cx1 - cx2, cy1 - cy2)


class TailgatingDetector:
    """
    Stateful research engine for detecting multi-person access anomalies.
    Maintains pair-wise temporal observations across active PersonEntity tracks.
    """

    def __init__(
        self,
        time_window: float = TAILGATING_TIME_WINDOW_SECONDS,
        spatial_threshold: float = TAILGATING_SPATIAL_THRESHOLD_PIXELS,
        confirmation_frames: int = TAILGATING_CONFIRMATION_FRAMES,
        cooldown_seconds: float = TAILGATING_ALERT_COOLDOWN_SECONDS,
    ):
        self.time_window = time_window
        self.spatial_threshold = spatial_threshold
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds

        # Pair tracking state: key = (auth_id, follower_id), value = {"hits": int, "first_seen": float, "last_seen": float, "confirmed": bool}
        self._active_pairs: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._last_alert_time: Dict[Tuple[int, int], float] = {}
        self._active_events: List[TailgatingEvent] = []

    def evaluate_tracks(
        self,
        person_entities: List[Dict[str, Any]],
        current_time: Optional[float] = None,
    ) -> Tuple[List[TailgatingEvent], float]:
        """
        Evaluates a frame's active PersonEntities for tailgating patterns.
        
        Returns:
            (confirmed_events, S_tailgating_risk in [0.0, 1.0])
        """
        now = current_time if current_time is not None else time.time()
        if len(person_entities) < 2:
            # Clean up stale pairs if fewer than 2 people
            self._cleanup_stale_pairs(now)
            self._active_events = []
            return [], 0.0

        # Separate authorized leaders from unauthorized/unknown followers
        authorized_candidates = []
        follower_candidates = []

        for p in person_entities:
            track_id = p.get("track_id", 0)
            identity = p.get("identity", "Unknown Person")
            is_known = p.get("is_known", False)
            access_state = p.get("access_state", "PENDING")
            liveness = (p.get("liveness_state") or "LIVE").upper()
            bbox = p.get("bbox", (0, 0, 0, 0))
            zone = p.get("zone", "NORMAL")

            # Authorized person: known identity + live PAD + access GRANTED/PENDING (not spoof/denied)
            if is_known and liveness == "LIVE" and access_state != "DENIED":
                authorized_candidates.append(p)
            else:
                # Follower: unknown visitor or denied identity
                follower_candidates.append(p)

        observed_pair_keys = set()
        detected_events: List[TailgatingEvent] = []

        for auth in authorized_candidates:
            auth_id = auth.get("track_id", 0)
            auth_bbox = auth.get("bbox", (0, 0, 0, 0))
            auth_ident = auth.get("identity", "Authorized Person")
            auth_zone = auth.get("zone", "NORMAL")

            for foll in follower_candidates:
                foll_id = foll.get("track_id", 1)
                foll_bbox = foll.get("bbox", (0, 0, 0, 0))
                foll_ident = foll.get("identity", "Unknown Person")
                foll_zone = foll.get("zone", "NORMAL")

                # Measure spatial separation
                dist = compute_centroid_distance(auth_bbox, foll_bbox)

                if dist <= self.spatial_threshold:
                    pair_key = (auth_id, foll_id)
                    observed_pair_keys.add(pair_key)

                    if pair_key not in self._active_pairs:
                        self._active_pairs[pair_key] = {
                            "hits": 1,
                            "first_seen": now,
                            "last_seen": now,
                            "confirmed": (self.confirmation_frames <= 1),
                            "zone": auth_zone,
                            "min_distance": dist,
                        }
                    else:
                        pair_state = self._active_pairs[pair_key]
                        pair_state["hits"] += 1
                        pair_state["last_seen"] = now
                        pair_state["min_distance"] = min(pair_state["min_distance"], dist)
                        if pair_state["hits"] >= self.confirmation_frames:
                            pair_state["confirmed"] = True

                    pair_state = self._active_pairs[pair_key]
                    time_delta = round(now - pair_state["first_seen"], 2)

                    # Compute confidence based on spatial proximity and observation consistency
                    proximity_factor = max(0.0, min(1.0, 1.0 - (dist / max(1.0, self.spatial_threshold))))
                    hit_factor = min(1.0, pair_state["hits"] / max(1.0, self.confirmation_frames * 2))
                    confidence = round(0.5 * proximity_factor + 0.5 * hit_factor, 3)

                    event = TailgatingEvent(
                        authorized_track_id=auth_id,
                        follower_track_id=foll_id,
                        time_delta=time_delta,
                        spatial_distance=round(dist, 1),
                        authorized_identity=auth_ident,
                        follower_identity=foll_ident,
                        confidence=confidence,
                        zone=auth_zone,
                        timestamp=now,
                        is_confirmed=pair_state["confirmed"],
                    )
                    detected_events.append(event)

        # Cleanup unobserved pairs
        self._cleanup_stale_pairs(now, active_keys=observed_pair_keys)
        self._active_events = [e for e in detected_events if e.is_confirmed]

        # Calculate normalized tailgating risk
        if not self._active_events:
            return detected_events, 0.0

        max_conf = max(e.confidence for e in self._active_events)
        # In Restricted Area, tailgating is critical risk (1.0)
        in_restricted = any("RESTRICTED" in e.zone.upper() or "PROTECTED" in e.zone.upper() for e in self._active_events)
        s_tailgating = 1.0 if in_restricted else round(max(0.60, max_conf), 3)

        return self._active_events, s_tailgating

    def should_emit_alert_event(self, event: TailgatingEvent, current_time: Optional[float] = None) -> bool:
        """Edge-triggered rate limiter preventing database spam."""
        if not event.is_confirmed:
            return False

        now = current_time if current_time is not None else time.time()
        pair_key = (event.authorized_track_id, event.follower_track_id)
        last_time = self._last_alert_time.get(pair_key, 0.0)

        if now - last_time >= self.cooldown_seconds:
            self._last_alert_time[pair_key] = now
            return True
        return False

    def _cleanup_stale_pairs(self, now: float, active_keys: Optional[set] = None) -> None:
        """Removes tracking state for separated pairs after timeout."""
        keys_to_remove = []
        for pair_key, state in self._active_pairs.items():
            if active_keys is not None and pair_key in active_keys:
                continue
            if now - state["last_seen"] > self.time_window:
                keys_to_remove.append(pair_key)

        for k in keys_to_remove:
            self._active_pairs.pop(k, None)
            self._last_alert_time.pop(k, None)
