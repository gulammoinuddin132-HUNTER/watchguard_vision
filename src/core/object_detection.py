"""
Object Detection Engine & Unified Detector Architecture (Phase 2.2 & 5.19D)
---------------------------------------------------------------------------
High-performance, lightweight local object detection engine for WatchGuard Vision.
Supports pluggable execution modes via the standardized ObjectDetector interface:
  1. "production": YOLOv8nCOCODetector (80 standard COCO classes, direct stretch)
  2. "watchguard_custom": WatchGuardYOLODetector (4 focused security classes, aspect letterbox)

Maintains temporal candidate confirmation, distance tier categorization,
and native OpenCV DNN CPU acceleration.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    YOLOV8N_MODEL_PATH,
    OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
    OBJECT_DETECTION_NMS_THRESHOLD,
    OBJECT_ALERT_COOLDOWN_SECONDS,
    OBJECT_DETECTOR_MODE,
    WATCHGUARD_CUSTOM_MODEL_PATH,
    WATCHGUARD_CUSTOM_CONF_THRESHOLD,
    WATCHGUARD_CUSTOM_NMS_THRESHOLD,
    WATCHGUARD_CUSTOM_CLASSES,
    WATCHGUARD_UNIFIED_MODEL_PATH,
    WATCHGUARD_UNIFIED_CONF_THRESHOLD,
    WATCHGUARD_UNIFIED_NMS_THRESHOLD,
    WATCHGUARD_UNIFIED_CLASSES,
    SECURITY_OBJECT_CATEGORIES,
    PROTECTED_OBJECT_CLASSES,
    COCO_CLASSES,
)
from src.core.base_engine import BaseVisionEngine
from src.core.interfaces.detector_interface import ObjectDetector, DetectedObject
from src.database.db_manager import DatabaseManager

logger = logging.getLogger("WatchGuardVision.ObjectDetection")


def compute_iou_boxes(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculates Intersection-over-Union between two boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


# =============================================================================
# 1. PRODUCTION DETECTOR: YOLOv8n COCO (80 Classes)
# =============================================================================

class YOLOv8nCOCODetector(ObjectDetector):
    """
    Production baseline detector using YOLOv8n ONNX trained on MS COCO 80.
    Uses standard direct-stretch preprocessing (640x480 -> 640x640).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
        nms_threshold: float = OBJECT_DETECTION_NMS_THRESHOLD,
    ):
        self.model_path = Path(model_path or YOLOV8N_MODEL_PATH)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._net: Optional[cv2.dnn.Net] = None
        self._is_ready = False
        self._classes = list(COCO_CLASSES)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def class_names(self) -> List[str]:
        return self._classes

    @property
    def mode_name(self) -> str:
        return "Production YOLOv8n (COCO 80)"

    def initialize(self) -> bool:
        if not self.model_path.exists():
            logger.error(f"Production YOLOv8 ONNX model not found at: {self.model_path}")
            return False
        try:
            logger.info(f"Loading Production YOLOv8n model from: {self.model_path}")
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._is_ready = True
            logger.info("Production YOLOv8n Detector initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Production YOLOv8n: {e}", exc_info=True)
            self._is_ready = False
            return False

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> List[DetectedObject]:
        if not self._is_ready or self._net is None or frame is None or frame.size == 0:
            return []

        h_orig, w_orig = frame.shape[:2]
        if h_orig <= 0 or w_orig <= 0:
            return []

        conf_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        nms_th = nms_threshold if nms_threshold is not None else self.nms_threshold

        try:
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 255.0,
                size=(640, 640),
                swapRB=True,
                crop=False,
            )
            self._net.setInput(blob)
            outputs = self._net.forward()

            predictions = np.transpose(outputs[0])  # (8400, 84)
            scale_x = w_orig / 640.0
            scale_y = h_orig / 640.0

            scores_matrix = predictions[:, 4:]
            max_scores = np.max(scores_matrix, axis=1)
            valid_mask = max_scores >= conf_th

            if not np.any(valid_mask):
                return []

            filt_preds = predictions[valid_mask]
            filt_scores = scores_matrix[valid_mask]
            class_ids_arr = np.argmax(filt_scores, axis=1)
            confidences_arr = max_scores[valid_mask]

            cx, cy, w, h = filt_preds[:, 0], filt_preds[:, 1], filt_preds[:, 2], filt_preds[:, 3]
            x = np.clip((cx - 0.5 * w) * scale_x, 0, w_orig - 1).astype(int)
            y = np.clip((cy - 0.5 * h) * scale_y, 0, h_orig - 1).astype(int)
            bw = np.clip(w * scale_x, 1, w_orig - x).astype(int)
            bh = np.clip(h * scale_y, 1, h_orig - y).astype(int)

            boxes = np.stack([x, y, bw, bh], axis=1).tolist()
            confidences = confidences_arr.astype(float).tolist()
            class_ids = class_ids_arr.astype(int).tolist()

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
            detected_objects: List[DetectedObject] = []

            if len(indices) > 0:
                for idx in indices:
                    i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                    c_id = class_ids[i]
                    lbl = self._classes[c_id] if 0 <= c_id < len(self._classes) else f"class_{c_id}"
                    conf = float(confidences[i])
                    bbox = tuple(boxes[i])

                    # Standardize label variations (e.g. 'cell phone' vs 'cell_phone')
                    norm_label = lbl.replace(" ", "_") if lbl == "cell phone" else lbl
                    is_protected = (lbl in PROTECTED_OBJECT_CLASSES or norm_label in PROTECTED_OBJECT_CLASSES)
                    sec_cat = SECURITY_OBJECT_CATEGORIES.get(norm_label, SECURITY_OBJECT_CATEGORIES.get(lbl, "context-relevant"))

                    detected_objects.append(DetectedObject(
                        class_id=c_id,
                        label=lbl,
                        confidence=conf,
                        bbox=bbox,
                        is_protected_asset=is_protected,
                        security_category=sec_cat,
                    ))

            return detected_objects

        except Exception as e:
            logger.error(f"Error during Production YOLOv8 forward pass: {e}", exc_info=True)
            return []

    def release(self) -> None:
        self._net = None
        self._is_ready = False


# =============================================================================
# 2. CUSTOM DETECTOR: WatchGuard Unified Small-Asset Detector (4 Classes)
# =============================================================================

class WatchGuardYOLODetector(ObjectDetector):
    """
    Unified WatchGuard-specific small-asset detector.
    Features:
      - 4 focused security classes: [person, laptop, cell_phone, mouse]
      - Aspect-ratio-preserving letterbox preprocessing (640x640)
      - Native OpenCV DNN CPU execution with calibrated small asset head
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = WATCHGUARD_CUSTOM_CONF_THRESHOLD,
        nms_threshold: float = WATCHGUARD_CUSTOM_NMS_THRESHOLD,
    ):
        self.model_path = Path(model_path or WATCHGUARD_CUSTOM_MODEL_PATH)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._net: Optional[cv2.dnn.Net] = None
        self._is_ready = False
        self._classes = list(WATCHGUARD_CUSTOM_CLASSES)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def class_names(self) -> List[str]:
        return self._classes

    @property
    def mode_name(self) -> str:
        return "WatchGuard Custom Small-Asset Detector (v3)"

    def initialize(self) -> bool:
        if not self.model_path.exists():
            logger.error(f"WatchGuard Custom ONNX model not found at: {self.model_path}")
            return False
        try:
            logger.info(f"Loading WatchGuard Custom Small-Asset model from: {self.model_path}")
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._is_ready = True
            logger.info("WatchGuard Custom Small-Asset Detector initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WatchGuard Custom Detector: {e}", exc_info=True)
            self._is_ready = False
            return False

    @staticmethod
    def preprocess_letterbox(img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, int, int]:
        """Aspect-preserving letterbox with 114 gray border padding."""
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        dx = (target_size - nw) // 2
        dy = (target_size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, scale, dx, dy

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> List[DetectedObject]:
        if not self._is_ready or self._net is None or frame is None or frame.size == 0:
            return []

        h_orig, w_orig = frame.shape[:2]
        if h_orig <= 0 or w_orig <= 0:
            return []

        conf_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        nms_th = nms_threshold if nms_threshold is not None else self.nms_threshold

        try:
            canvas, scale, dx, dy = self.preprocess_letterbox(frame, target_size=640)
            blob = cv2.dnn.blobFromImage(
                canvas,
                scalefactor=1.0 / 255.0,
                size=(640, 640),
                swapRB=True,
                crop=False,
            )
            self._net.setInput(blob)
            outputs = self._net.forward()

            predictions = np.transpose(outputs[0])  # (8400, 8)
            scores_matrix = predictions[:, 4:]      # 4 target classes
            max_scores = np.max(scores_matrix, axis=1)
            valid_mask = max_scores >= conf_th

            if not np.any(valid_mask):
                return []

            filt_preds = predictions[valid_mask]
            filt_scores = scores_matrix[valid_mask]
            class_ids_arr = np.argmax(filt_scores, axis=1)
            confidences_arr = max_scores[valid_mask]

            cx, cy, w, h = filt_preds[:, 0], filt_preds[:, 1], filt_preds[:, 2], filt_preds[:, 3]
            # Inverse letterbox transform back to native camera space
            x = np.clip((cx - 0.5 * w - dx) / scale, 0, w_orig - 1).astype(int)
            y = np.clip((cy - 0.5 * h - dy) / scale, 0, h_orig - 1).astype(int)
            bw = np.clip(w / scale, 1, w_orig - x).astype(int)
            bh = np.clip(h / scale, 1, h_orig - y).astype(int)

            boxes = np.stack([x, y, bw, bh], axis=1).tolist()
            confidences = confidences_arr.astype(float).tolist()
            class_ids = class_ids_arr.astype(int).tolist()

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
            detected_objects: List[DetectedObject] = []

            if len(indices) > 0:
                for idx in indices:
                    i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                    c_id = class_ids[i]
                    lbl = self._classes[c_id] if 0 <= c_id < len(self._classes) else f"class_{c_id}"
                    conf = float(confidences[i])
                    bbox = tuple(boxes[i])

                    norm_label = lbl.replace(" ", "_")
                    is_protected = (lbl in PROTECTED_OBJECT_CLASSES or norm_label in PROTECTED_OBJECT_CLASSES)
                    sec_cat = SECURITY_OBJECT_CATEGORIES.get(norm_label, SECURITY_OBJECT_CATEGORIES.get(lbl, "context-relevant"))

                    detected_objects.append(DetectedObject(
                        class_id=c_id,
                        label=lbl,
                        confidence=conf,
                        bbox=bbox,
                        is_protected_asset=is_protected,
                        security_category=sec_cat,
                    ))

            return detected_objects

        except Exception as e:
            logger.error(f"Error during WatchGuard Custom Detector forward pass: {e}", exc_info=True)
            return []

    def release(self) -> None:
        self._net = None
        self._is_ready = False


# =============================================================================
# 3. UNIFIED DETECTOR: WatchGuard Unified 84-Class Model (80 COCO + 4 Security)
# =============================================================================

class WatchGuardUnified84Detector(ObjectDetector):
    """
    Unified WatchGuard-specific 84-class detector (Phase 5.21).
    Preserves all 80 standard COCO classes (0 to 79) and adds 4 domain security classes:
      - 80: pen
      - 81: access_badge
      - 82: usb_drive
      - 83: keys
    Expected output tensor shape: (1, 88, 8400) where 88 = 4 box coordinates + 84 class confidences.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = WATCHGUARD_UNIFIED_CONF_THRESHOLD,
        nms_threshold: float = WATCHGUARD_UNIFIED_NMS_THRESHOLD,
    ):
        self.model_path = Path(model_path or WATCHGUARD_UNIFIED_MODEL_PATH)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._net: Optional[cv2.dnn.Net] = None
        self._is_ready = False
        self._classes = list(WATCHGUARD_UNIFIED_CLASSES)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def class_names(self) -> List[str]:
        return self._classes

    @property
    def mode_name(self) -> str:
        return "WatchGuard Unified 84-Class Detector"

    def initialize(self) -> bool:
        if not self.model_path.exists():
            logger.info(f"Unified 84-class ONNX model not found at: {self.model_path} (Foundation Mode)")
            return False
        try:
            logger.info(f"Loading WatchGuard Unified 84-class model from: {self.model_path}")
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._is_ready = True
            logger.info("WatchGuard Unified 84-class Detector initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WatchGuard Unified 84-class Detector: {e}", exc_info=True)
            self._is_ready = False
            return False

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> List[DetectedObject]:
        if not self._is_ready or self._net is None or frame is None or frame.size == 0:
            return []

        h_orig, w_orig = frame.shape[:2]
        if h_orig <= 0 or w_orig <= 0:
            return []

        conf_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        nms_th = nms_threshold if nms_threshold is not None else self.nms_threshold

        try:
            canvas, scale, dx, dy = WatchGuardYOLODetector.preprocess_letterbox(frame, target_size=640)
            blob = cv2.dnn.blobFromImage(
                canvas,
                scalefactor=1.0 / 255.0,
                size=(640, 640),
                swapRB=True,
                crop=False,
            )
            self._net.setInput(blob)
            outputs = self._net.forward()

            predictions = np.transpose(outputs[0])  # (8400, 88)
            scores_matrix = predictions[:, 4:]      # 84 classes
            max_scores = np.max(scores_matrix, axis=1)
            valid_mask = max_scores >= conf_th

            if not np.any(valid_mask):
                return []

            filt_preds = predictions[valid_mask]
            filt_scores = scores_matrix[valid_mask]
            class_ids_arr = np.argmax(filt_scores, axis=1)
            confidences_arr = max_scores[valid_mask]

            cx, cy, w, h = filt_preds[:, 0], filt_preds[:, 1], filt_preds[:, 2], filt_preds[:, 3]
            x = np.clip((cx - 0.5 * w - dx) / scale, 0, w_orig - 1).astype(int)
            y = np.clip((cy - 0.5 * h - dy) / scale, 0, h_orig - 1).astype(int)
            bw = np.clip(w / scale, 1, w_orig - x).astype(int)
            bh = np.clip(h / scale, 1, h_orig - y).astype(int)

            boxes = np.stack([x, y, bw, bh], axis=1).tolist()
            confidences = confidences_arr.astype(float).tolist()
            class_ids = class_ids_arr.astype(int).tolist()

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
            detected_objects: List[DetectedObject] = []

            if len(indices) > 0:
                for idx in indices:
                    i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                    c_id = class_ids[i]
                    lbl = self._classes[c_id] if 0 <= c_id < len(self._classes) else f"class_{c_id}"
                    conf = float(confidences[i])
                    bbox = tuple(boxes[i])

                    norm_label = lbl.replace(" ", "_")
                    is_protected = (lbl in PROTECTED_OBJECT_CLASSES or norm_label in PROTECTED_OBJECT_CLASSES)
                    sec_cat = SECURITY_OBJECT_CATEGORIES.get(norm_label, SECURITY_OBJECT_CATEGORIES.get(lbl, "context-relevant"))

                    detected_objects.append(DetectedObject(
                        class_id=c_id,
                        label=lbl,
                        confidence=conf,
                        bbox=bbox,
                        is_protected_asset=is_protected,
                        security_category=sec_cat,
                    ))

            return detected_objects

        except Exception as e:
            logger.error(f"Error during WatchGuard Unified Detector forward pass: {e}", exc_info=True)
            return []

    def release(self) -> None:
        self._net = None
        self._is_ready = False


# =============================================================================
# 4. TEMPORAL CANDIDATE CONFIRMATION TRACKER
# =============================================================================

class TemporalObjectTracker:
    """
    Maintains short-lived temporal candidate confirmation for detected objects.
    Prevents single-frame transient noise from causing database spam or risk jitter.
    """

    def __init__(self, confirmation_frames: int = 2, max_missed_frames: int = 3):
        self.confirmation_frames = confirmation_frames
        self.max_missed_frames = max_missed_frames
        # State: list of {"label": str, "bbox": tuple, "hits": int, "misses": int, "confirmed": bool}
        self._tracked_objects: List[Dict[str, Any]] = []

    def update(self, raw_objects: List[DetectedObject]) -> List[DetectedObject]:
        """
        Updates temporal confirmation state with raw frame detections.
        Returns the list of DetectedObjects with updated `is_confirmed` flags.
        """
        if not raw_objects:
            # Increment misses for existing tracks
            for t in self._tracked_objects:
                t["misses"] += 1
            self._tracked_objects = [t for t in self._tracked_objects if t["misses"] <= self.max_missed_frames]
            return []

        updated_detections: List[DetectedObject] = []
        matched_track_indices = set()

        for obj in raw_objects:
            best_match_idx = None
            best_iou = 0.0

            for idx, track in enumerate(self._tracked_objects):
                if idx in matched_track_indices:
                    continue
                if track["label"] == obj.label:
                    iou = compute_iou_boxes(track["bbox"], obj.bbox)
                    if iou > best_iou and iou >= 0.30:
                        best_iou = iou
                        best_match_idx = idx

            if best_match_idx is not None:
                track = self._tracked_objects[best_match_idx]
                track["bbox"] = obj.bbox
                track["hits"] += 1
                track["misses"] = 0
                if track["hits"] >= self.confirmation_frames:
                    track["confirmed"] = True
                obj.is_confirmed = track["confirmed"]
                matched_track_indices.add(best_match_idx)
            else:
                # New candidate
                is_conf = (self.confirmation_frames <= 1)
                self._tracked_objects.append({
                    "label": obj.label,
                    "bbox": obj.bbox,
                    "hits": 1,
                    "misses": 0,
                    "confirmed": is_conf,
                })
                obj.is_confirmed = is_conf

            updated_detections.append(obj)

        # Cleanup lost tracks
        for idx, track in enumerate(self._tracked_objects):
            if idx not in matched_track_indices:
                track["misses"] += 1

        self._tracked_objects = [t for t in self._tracked_objects if t["misses"] <= self.max_missed_frames]
        return updated_detections


# =============================================================================
# 4. PRIMARY ENGINE FACADE: ObjectDetectionEngine
# =============================================================================

class ObjectDetectionEngine(BaseVisionEngine):
    """
    Main Object Detection Engine for WatchGuard Vision.
    Dispatches to the active ObjectDetector implementation based on OBJECT_DETECTOR_MODE.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        detector_mode: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.mode = (detector_mode or OBJECT_DETECTOR_MODE).lower()
        self.db = db_manager
        self.temporal_tracker = TemporalObjectTracker(confirmation_frames=2, max_missed_frames=3)
        self._alert_cooldowns: Dict[str, float] = {}

        # Instantiate active backend detector
        if self.mode == "watchguard_unified":
            conf = conf_threshold if conf_threshold is not None else WATCHGUARD_UNIFIED_CONF_THRESHOLD
            nms = nms_threshold if nms_threshold is not None else WATCHGUARD_UNIFIED_NMS_THRESHOLD
            m_path = model_path or WATCHGUARD_UNIFIED_MODEL_PATH
            self._detector: ObjectDetector = WatchGuardUnified84Detector(
                model_path=m_path,
                conf_threshold=conf,
                nms_threshold=nms,
            )
        elif self.mode == "watchguard_custom":
            conf = conf_threshold if conf_threshold is not None else WATCHGUARD_CUSTOM_CONF_THRESHOLD
            nms = nms_threshold if nms_threshold is not None else WATCHGUARD_CUSTOM_NMS_THRESHOLD
            m_path = model_path or WATCHGUARD_CUSTOM_MODEL_PATH
            self._detector: ObjectDetector = WatchGuardYOLODetector(
                model_path=m_path,
                conf_threshold=conf,
                nms_threshold=nms,
            )
        else:
            # Default to production (80 COCO classes)
            conf = conf_threshold if conf_threshold is not None else OBJECT_DETECTION_CONFIDENCE_THRESHOLD
            nms = nms_threshold if nms_threshold is not None else OBJECT_DETECTION_NMS_THRESHOLD
            m_path = model_path or YOLOV8N_MODEL_PATH
            self._detector: ObjectDetector = YOLOv8nCOCODetector(
                model_path=m_path,
                conf_threshold=conf,
                nms_threshold=nms,
            )

    @property
    def is_ready(self) -> bool:
        return self._detector.is_ready

    @property
    def _is_ready(self) -> bool:
        return self._detector.is_ready

    @_is_ready.setter
    def _is_ready(self, val: bool) -> None:
        if hasattr(self._detector, "_is_ready"):
            self._detector._is_ready = val

    @property
    def _net(self) -> Optional[cv2.dnn.Net]:
        return getattr(self._detector, "_net", None)

    @_net.setter
    def _net(self, net: Optional[cv2.dnn.Net]) -> None:
        if hasattr(self._detector, "_net"):
            self._detector._net = net

    @property
    def class_names(self) -> List[str]:
        return self._detector.class_names

    @property
    def active_mode(self) -> str:
        return self._detector.mode_name

    def initialize(self) -> bool:
        """Initializes the active detector backend."""
        return self._detector.initialize()

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects objects in the input frame and returns standardized output.
        Backward-compatible with WatchGuard GUI and ContextEngine.
        """
        empty_result = {
            "detected": False,
            "count": 0,
            "objects": [],
            "summary": "No objects detected",
            "counts_by_class": {},
            "detector_mode": self.active_mode,
        }

        if not self._detector.is_ready or frame is None or frame.size == 0:
            return empty_result

        raw_objects = self._detector.detect(frame)
        if not raw_objects:
            self.temporal_tracker.update([])
            return empty_result

        confirmed_objects = self.temporal_tracker.update(raw_objects)
        objects_to_report = confirmed_objects if confirmed_objects else raw_objects

        counts_by_class: Dict[str, int] = {}
        objects_dicts: List[Dict[str, Any]] = []

        for obj in objects_to_report:
            objects_dicts.append(obj.to_dict())
            lbl = obj.label
            counts_by_class[lbl] = counts_by_class.get(lbl, 0) + 1

        summary_parts = [f"{lbl}" if cnt == 1 else f"{lbl} ({cnt})" for lbl, cnt in counts_by_class.items()]
        summary_str = ", ".join(summary_parts)

        return {
            "detected": len(objects_dicts) > 0,
            "count": len(objects_dicts),
            "objects": objects_dicts,
            "summary": summary_str,
            "counts_by_class": counts_by_class,
            "detector_mode": self.active_mode,
        }

    def draw_annotations(
        self,
        frame: np.ndarray,
        results: Dict[str, Any],
        suppress_person_boxes: bool = True,
    ) -> np.ndarray:
        """Renders HUD bounding boxes and badges on video frames."""
        if frame is None or not results.get("detected", False):
            return frame

        annotated = frame.copy()
        objects = results.get("objects", [])

        COLOR_DEFAULT = (255, 200, 0)    # Electric cyan/blue
        COLOR_PROTECTED = (0, 165, 255)  # Amber
        COLOR_SECURITY = (255, 140, 0)   # Deep cyan
        COLOR_HIGH_RISK = (0, 0, 255)    # Red

        for obj in objects:
            lbl = obj["label"]
            conf = obj["confidence"]
            x, y, w, h = obj["bbox"]
            sec_cat = obj.get("security_category", "context-relevant")

            # Suppress person box if face tracker is handling identity badges
            if suppress_person_boxes and lbl == "person":
                continue

            if sec_cat == "high-risk-asset":
                color = COLOR_HIGH_RISK
            elif sec_cat in ("protected", "credential"):
                color = COLOR_PROTECTED
            elif sec_cat == "security-relevant":
                color = COLOR_SECURITY
            else:
                color = COLOR_DEFAULT

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw badge header
            badge_text = f"{lbl.upper()} {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, max(0, y - th - 6)), (x + tw + 8, y), color, -1)
            cv2.putText(
                annotated,
                badge_text,
                (x + 4, max(th + 2, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def should_trigger_alert(
        self,
        label: str,
        cooldown_seconds: float = OBJECT_ALERT_COOLDOWN_SECONDS,
    ) -> bool:
        """Rate-limits notifications for a specific object class using a sliding cooldown."""
        now = time.time()
        last_alert_time = self._alert_cooldowns.get(label, 0.0)

        if now - last_alert_time >= cooldown_seconds:
            self._alert_cooldowns[label] = now
            return True
        return False

    def log_detection_record(
        self,
        session_id: str,
        label: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        camera_id: int = 0,
    ) -> Optional[int]:
        """Logs a confirmed detected object record into the SQLite database."""
        if not self.db:
            return None

        try:
            bbox_dict = (
                {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4
                else bbox
            )
            return self.db.log_detected_object(
                session_id=session_id,
                label=label,
                confidence=confidence,
                bbox_json=json.dumps(bbox_dict),
            )
        except Exception as e:
            logger.error(f"Failed to log detected object to database: {e}", exc_info=True)
            return None

    def release(self) -> None:
        """Releases underlying model resources."""
        self._detector.release()
