"""
PAD Manager (Presentation Attack Detection Coordinator)
-------------------------------------------------------
Pluggable manager coordinating all modular PAD implementations.
Allows seamless runtime switching between LBP-SVM, MobileNetV3,
EfficientNet-B0, and Blink-EAR without destabilizing the vision pipeline.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from config import PAD_DEFAULT_MODEL
from src.core.pad.base_pad import AttackType, BasePADEngine, PADResult
from src.core.pad.models.blink_ear_pad import BlinkEARPADEngine
from src.core.pad.models.lbp_svm_pad import LBPSVMPADEngine
from src.core.pad.models.mobilenet_pad import MobileNetPADEngine
from src.core.pad.models.efficientnet_pad import EfficientNetPADEngine

logger = logging.getLogger("WatchGuardVision.PAD.Manager")


class PADManager:
    """
    Central coordinator for Presentation Attack Detection.
    Manages active PAD engine lifecycle and dispatches frame crops for evaluation.
    """

    AVAILABLE_MODELS = {
        "Blink-EAR": BlinkEARPADEngine,
        "LBP-SVM": LBPSVMPADEngine,
        "MobileNetV3-Small": MobileNetPADEngine,
        "EfficientNet-B0": EfficientNetPADEngine,
    }

    def __init__(self, default_model: str = PAD_DEFAULT_MODEL):
        self._engines: Dict[str, BasePADEngine] = {}
        self._active_model_name = default_model if default_model in self.AVAILABLE_MODELS else "Blink-EAR"
        self._is_ready = False

    @property
    def active_model_name(self) -> str:
        return self._active_model_name

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def active_engine(self) -> BasePADEngine:
        if self._active_model_name not in self._engines:
            self._engines[self._active_model_name] = self.AVAILABLE_MODELS[self._active_model_name]()
        return self._engines[self._active_model_name]

    def initialize(self) -> bool:
        """Initializes the default active PAD engine."""
        try:
            engine = self.active_engine
            if not engine.is_ready:
                self._is_ready = engine.initialize()
            else:
                self._is_ready = True
            logger.info(f"PAD Manager initialized with active model: '{self._active_model_name}'")
            return self._is_ready
        except Exception as e:
            logger.error(f"Failed to initialize PAD Manager: {e}")
            self._is_ready = False
            return False

    def set_active_model(self, model_name: str) -> bool:
        """Switches the active PAD engine at runtime."""
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown PAD model '{model_name}'. Available: {list(self.AVAILABLE_MODELS.keys())}")
            return False

        self._active_model_name = model_name
        engine = self.active_engine
        if not engine.is_ready:
            self._is_ready = engine.initialize()
        else:
            self._is_ready = True
        logger.info(f"PAD active engine switched to: '{model_name}' ({engine.model_name})")
        return True

    def evaluate_presentation(
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
        Dispatches face data to the active PAD engine.
        Guaranteed to return a valid PADResult without raising unhandled exceptions.
        """
        if not self._is_ready:
            self.initialize()

        try:
            return self.active_engine.evaluate(
                raw_face_crop=raw_face_crop,
                aligned_face=aligned_face,
                landmarks=landmarks,
                frame_history=frame_history,
                timestamps=timestamps,
                track_id=track_id,
                face_bbox=face_bbox,
                detected_objects=detected_objects,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error evaluating PAD presentation with {self._active_model_name}: {e}", exc_info=True)
            return PADResult(
                is_live=False,  # FAIL-SECURE: Never grant authorization on unhandled exceptions
                confidence=0.0,
                attack_type=AttackType.UNKNOWN,
                inference_latency_ms=0.0,
                model_name=self.active_engine.model_name,
                metadata={"error": str(e)},
            )
