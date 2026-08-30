"""
WatchGuard Vision - CCSDF Configuration & Weight Profiles
---------------------------------------------------------
Defines configurable weights, risk action thresholds, and ablation masks
for the Continuous Contextual Security Decision Fusion (CCSDF) research layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


class CCSDFAction(str, Enum):
    """Recommended contextual authorization actions."""
    ALLOW = "ALLOW"
    ATTENTION = "ATTENTION"
    RESTRICT = "RESTRICT"
    DENY = "DENY"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"


@dataclass
class CCSDFWeights:
    """Configurable weight parameters for the multi-dimensional contextual risk factors."""
    identity: float = 0.20
    pad: float = 0.25
    zone: float = 0.20
    time: float = 0.15
    proximity: float = 0.10
    history: float = 0.10
    asset: float = 0.0
    behavior: float = 0.0

    def __post_init__(self):
        total = self.identity + self.pad + self.zone + self.time + self.proximity + self.history + self.asset + self.behavior
        if abs(total - 1.0) > 1e-4 and total > 0:
            # Auto-normalize to sum to 1.0
            self.identity /= total
            self.pad /= total
            self.zone /= total
            self.time /= total
            self.proximity /= total
            self.history /= total
            self.asset /= total
            self.behavior /= total

    def to_dict(self) -> Dict[str, float]:
        return {
            "identity": round(self.identity, 4),
            "pad": round(self.pad, 4),
            "zone": round(self.zone, 4),
            "time": round(self.time, 4),
            "proximity": round(self.proximity, 4),
            "history": round(self.history, 4),
            "asset": round(self.asset, 4),
            "behavior": round(self.behavior, 4),
        }


# Standard Parameterized Weight Sets for Sensitivity Analysis (Part 14 & Phase 5.20)
WEIGHT_SET_A_BALANCED = CCSDFWeights(
    identity=0.20,
    pad=0.25,
    zone=0.20,
    time=0.15,
    proximity=0.10,
    history=0.10,
    behavior=0.0,
)

WEIGHT_SET_B_PAD_HEAVY = CCSDFWeights(
    identity=0.15,
    pad=0.40,
    zone=0.15,
    time=0.10,
    proximity=0.10,
    history=0.10,
    behavior=0.0,
)

WEIGHT_SET_C_ZONE_TIME_HEAVY = CCSDFWeights(
    identity=0.15,
    pad=0.15,
    zone=0.35,
    time=0.25,
    proximity=0.05,
    history=0.05,
    behavior=0.0,
)

WEIGHT_SET_D_HISTORY_HEAVY = CCSDFWeights(
    identity=0.15,
    pad=0.15,
    zone=0.15,
    time=0.15,
    proximity=0.10,
    history=0.30,
    behavior=0.0,
)

WEIGHT_PROFILES: Dict[str, CCSDFWeights] = {
    "SET_A_BALANCED": WEIGHT_SET_A_BALANCED,
    "SET_B_PAD_HEAVY": WEIGHT_SET_B_PAD_HEAVY,
    "SET_C_ZONE_TIME_HEAVY": WEIGHT_SET_C_ZONE_TIME_HEAVY,
    "SET_D_HISTORY_HEAVY": WEIGHT_SET_D_HISTORY_HEAVY,
}


@dataclass
class CCSDFThresholds:
    """Action decision thresholds mapping continuous risk R(t) in [0, 1] to discrete actions."""
    allow_threshold: float = 0.25      # R < 0.25 -> ALLOW (if identity is known & verified)
    attention_threshold: float = 0.50  # 0.25 <= R < 0.50 -> ATTENTION
    restrict_threshold: float = 0.75   # 0.50 <= R < 0.75 -> RESTRICT
    # R >= 0.75 -> DENY


# Standard Ablation Masks (Part 15)
ABLATION_CONFIGS: Dict[str, Dict[str, bool]] = {
    "FULL_CCSDF": {
        "identity": True, "pad": True, "zone": True, "time": True, "proximity": True, "history": True
    },
    "MINUS_HISTORY": {
        "identity": True, "pad": True, "zone": True, "time": True, "proximity": True, "history": False
    },
    "MINUS_PROXIMITY": {
        "identity": True, "pad": True, "zone": True, "time": True, "proximity": False, "history": True
    },
    "MINUS_TIME": {
        "identity": True, "pad": True, "zone": True, "time": False, "proximity": True, "history": True
    },
    "MINUS_ZONE": {
        "identity": True, "pad": True, "zone": False, "time": True, "proximity": True, "history": True
    },
    "MINUS_PAD": {
        "identity": True, "pad": False, "zone": True, "time": True, "proximity": True, "history": True
    },
}
