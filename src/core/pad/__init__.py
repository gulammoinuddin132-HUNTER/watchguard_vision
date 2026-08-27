"""
WatchGuard Vision - Presentation Attack Detection (PAD) Module
--------------------------------------------------------------
Public package exports for Phase 3 Liveness & Presentation Attack Detection.
"""

from src.core.pad.base_pad import AttackType, BasePADEngine, PADResult
from src.core.pad.joint_decision import JointAccessEvaluator, JointDecision, JointStatus
from src.core.pad.pad_manager import PADManager

__all__ = [
    "AttackType",
    "BasePADEngine",
    "PADResult",
    "JointStatus",
    "JointDecision",
    "JointAccessEvaluator",
    "PADManager",
]
