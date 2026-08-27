"""
Presentation Attack Detection (PAD) Base Interface & Data Structures
---------------------------------------------------------------------
Defines standardized data classes and abstract base class for all pluggable
Liveness / Presentation-Attack Detection algorithms in WatchGuard Vision.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class AttackType(str, Enum):
    """Categorized presentation attack modalities."""
    BONAFIDE = "BONAFIDE"
    PRINT = "PRINT_ATTACK"
    REPLAY = "REPLAY_ATTACK"
    MASK = "3D_MASK"
    UNKNOWN = "UNKNOWN_ATTACK"

    @property
    def label(self) -> str:
        labels = {
            AttackType.BONAFIDE: "Bona-fide Live",
            AttackType.PRINT: "Printed Photo Spoof",
            AttackType.REPLAY: "Screen / Video Replay Spoof",
            AttackType.MASK: "3D Mask / Silicone Spoof",
            AttackType.UNKNOWN: "Presentation Attack",
        }
        return labels.get(self, "Unknown Presentation")


@dataclass
class PADResult:
    """Standardized output for an individual face presentation liveness evaluation."""
    is_live: bool                    # True = bona-fide live person, False = presentation attack
    confidence: float                # Calibrated liveness score [0.0, 1.0] (Provisional threshold: >= 0.80 live, < 0.50 spoof)
    attack_type: AttackType          # Categorized attack class
    inference_latency_ms: float      # Computation latency in milliseconds
    model_name: str                  # Name of active PAD algorithm
    raw_score: float = 0.0           # Raw model logit / distance / classifier output
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_spoof(self) -> bool:
        return not self.is_live

    @property
    def display_tag(self) -> str:
        """Formatted status tag for UI HUD annotations."""
        if self.is_live:
            return f"LIVE ({int(self.confidence * 100)}%)"
        else:
            return f"SPOOF ({int(self.confidence * 100)}%)"


class BasePADEngine(ABC):
    """Abstract Base Class for all interchangeable PAD algorithms."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the human-readable identifier of the PAD engine."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the engine is initialized and ready for inference."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Loads weights, initializes feature extractors, and prepares runtime."""
        pass

    @abstractmethod
    def evaluate(
        self,
        raw_face_crop: np.ndarray,
        aligned_face: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
        frame_history: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
        track_id: Optional[int] = None,
        face_bbox: Optional[Any] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> PADResult:
        """
        Evaluates a detected face presentation for liveness.

        Args:
            raw_face_crop: BGR cropped image of the detected face with boundary padding.
            aligned_face: 112x112 aligned face crop from SFace alignCrop if available.
            landmarks: 5 or 68 facial landmark coordinates (YuNet: [x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc]).
            frame_history: Recent sequence of consecutive frames for temporal/motion tracking.
            timestamps: Monotonic timestamps corresponding to frame_history.
            track_id: Unique spatial track ID associated with this face instance.
            face_bbox: (x, y, w, h) bounding box of the face in original frame.
            detected_objects: Active detected objects in frame for spatial context fusion (e.g. phone/screen overlap).

        Returns:
            PADResult containing liveness determination, confidence, attack type, and latency.
        """
        pass
        pass
