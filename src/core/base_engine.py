"""
Base Engine Module
------------------
Abstract base classes and common data structures for modular AI processing components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseVisionEngine(ABC):
    """Abstract base class for vision processing engines."""

    @abstractmethod
    def initialize(self) -> bool:
        """Loads models, weights, or initializes engine dependencies."""
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyzes an input BGR frame and returns structured detections."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases memory, hardware models, or thread handles."""
        pass
