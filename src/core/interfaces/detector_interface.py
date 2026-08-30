"""
WatchGuard Vision - Standardized Object Detector Interface
----------------------------------------------------------
Phase 5.19D: Unified abstraction for object detectors in WatchGuard Vision.
Enables pluggable switching between:
  - YOLOv8nCOCODetector (Production baseline)
  - WatchGuardYOLODetector (Unified Custom Small-Asset model)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class DetectedObject:
    """Standardized representation of a detected visual entity in WatchGuard Vision."""
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in native 640x480 camera pixel space
    area_px: int = 0
    distance_tier: Optional[str] = None  # "CLOSE" (>10k px^2), "MEDIUM" (1.6k-10k px^2), "FAR" (<1.6k px^2)
    is_protected_asset: bool = False
    security_category: str = "context-relevant" # "protected", "security-relevant", "high-risk-asset", "credential", "entity"
    associated_person_track: Optional[int] = None
    association_confidence: float = 0.0
    pixel_distance_to_person: float = -1.0
    is_confirmed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.area_px == 0 and len(self.bbox) == 4:
            self.area_px = self.bbox[2] * self.bbox[3]
        if self.distance_tier is None:
            if self.area_px > 10000:
                self.distance_tier = "CLOSE"
            elif self.area_px >= 1600:
                self.distance_tier = "MEDIUM"
            else:
                self.distance_tier = "FAR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "area_px": self.area_px,
            "distance_tier": self.distance_tier,
            "is_protected_asset": self.is_protected_asset,
            "security_category": self.security_category,
            "associated_person_track": self.associated_person_track,
            "association_confidence": round(self.association_confidence, 3),
            "pixel_distance_to_person": round(self.pixel_distance_to_person, 1),
            "is_confirmed": self.is_confirmed,
            "metadata": self.metadata,
        }


class ObjectDetector(ABC):
    """Universal abstract base class for object detectors in WatchGuard Vision."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if model weights are loaded and ready for inference."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Returns the list of supported class names."""
        pass

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Returns the human-readable identifier of the detector mode."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Loads weights and allocates execution buffers."""
        pass

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> List[DetectedObject]:
        """Performs forward inference and returns a list of standardized DetectedObjects."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Frees model resources."""
        pass
