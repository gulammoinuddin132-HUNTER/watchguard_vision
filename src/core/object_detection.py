"""
Object Detection Engine (OpenCV DNN + YOLOv8n ONNX)
----------------------------------------------------
High-performance, lightweight local object detection engine for WatchGuard Vision.
Uses native OpenCV DNN with YOLOv8 Nano ONNX model, supporting 80 standard COCO classes
(laptop, cell phone, bottle, backpack, chair, book, etc.) with real-time inference on laptop CPU.
"""

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
    COCO_CLASSES,
)
from src.core.base_engine import BaseVisionEngine
from src.database.db_manager import DatabaseManager

logger = logging.getLogger("WatchGuardVision.ObjectDetection")


class ObjectDetectionEngine(BaseVisionEngine):
    """
    Object Detection Engine using OpenCV DNN with YOLOv8 Nano ONNX.
    Provides local, CPU-accelerated inference with Non-Maximum Suppression (NMS)
    and seamless HUD overlay rendering.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
        nms_threshold: float = OBJECT_DETECTION_NMS_THRESHOLD,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.model_path = Path(model_path or YOLOV8N_MODEL_PATH)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.db = db_manager

        self._net: Optional[cv2.dnn.Net] = None
        self._is_ready = False
        self._classes = list(COCO_CLASSES)
        self._alert_cooldowns: Dict[str, float] = {}

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def class_names(self) -> List[str]:
        return self._classes

    def initialize(self) -> bool:
        """Loads YOLOv8 Nano ONNX weights into OpenCV DNN."""
        if not self.model_path.exists():
            logger.error(f"YOLOv8 ONNX model not found at {self.model_path}")
            return False

        try:
            logger.info(f"Loading YOLOv8 ONNX model from: {self.model_path}")
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._is_ready = True
            logger.info(f"YOLOv8 Object Detection Engine initialized successfully ({len(self._classes)} classes supported).")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 Object Detection Engine: {e}", exc_info=True)
            self._is_ready = False
            return False

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects objects in the input BGR frame.

        Returns:
            {
                "detected": bool,
                "count": int,
                "objects": [
                    {
                        "label": "laptop",
                        "confidence": 0.89,
                        "bbox": (x, y, w, h),
                        "class_id": 63
                    }
                ],
                "summary": "laptop, cell phone",
                "counts_by_class": {"laptop": 1, "cell phone": 1}
            }
        """
        empty_result = {
            "detected": False,
            "count": 0,
            "objects": [],
            "summary": "No objects detected",
            "counts_by_class": {},
        }

        if not self._is_ready or self._net is None or frame is None or frame.size == 0:
            return empty_result

        h_orig, w_orig = frame.shape[:2]
        if h_orig == 0 or w_orig == 0:
            return empty_result

        try:
            # 1. Preprocess: 640x640 normalized RGB blob
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 255.0,
                size=(640, 640),
                swapRB=True,
                crop=False,
            )
            self._net.setInput(blob)

            # 2. Forward inference pass -> shape (1, 84, 8400)
            outputs = self._net.forward()

            # 3. Transpose output to (8400, 84)
            predictions = np.transpose(outputs[0])

            scale_x = w_orig / 640.0
            scale_y = h_orig / 640.0

            # 4. Filter predictions by confidence using vectorized numpy operations
            scores_matrix = predictions[:, 4:]
            max_scores = np.max(scores_matrix, axis=1)
            valid_mask = max_scores >= self.conf_threshold

            if not np.any(valid_mask):
                return empty_result

            filt_preds = predictions[valid_mask]
            filt_scores = scores_matrix[valid_mask]
            class_ids_arr = np.argmax(filt_scores, axis=1)
            confidences_arr = max_scores[valid_mask]

            cx = filt_preds[:, 0]
            cy = filt_preds[:, 1]
            w = filt_preds[:, 2]
            h = filt_preds[:, 3]

            x = np.clip((cx - 0.5 * w) * scale_x, 0, w_orig - 1).astype(int)
            y = np.clip((cy - 0.5 * h) * scale_y, 0, h_orig - 1).astype(int)
            bw = np.clip(w * scale_x, 1, w_orig - x).astype(int)
            bh = np.clip(h * scale_y, 1, h_orig - y).astype(int)

            boxes = np.stack([x, y, bw, bh], axis=1).tolist()
            confidences = confidences_arr.astype(float).tolist()
            class_ids = class_ids_arr.astype(int).tolist()

            # 5. Apply Non-Maximum Suppression (NMS)
            if not boxes:
                return empty_result

            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, self.conf_threshold, self.nms_threshold
            )

            detected_objects: List[Dict[str, Any]] = []
            counts_by_class: Dict[str, int] = {}

            if len(indices) > 0:
                for idx in indices:
                    i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                    c_id = class_ids[i]
                    lbl = self._classes[c_id] if 0 <= c_id < len(self._classes) else f"class_{c_id}"
                    conf = confidences[i]
                    bbox = tuple(boxes[i])

                    detected_objects.append({
                        "label": lbl,
                        "confidence": round(conf, 3),
                        "bbox": bbox,
                        "class_id": c_id,
                    })

                    counts_by_class[lbl] = counts_by_class.get(lbl, 0) + 1

            if not detected_objects:
                return empty_result

            # Construct readable summary string (e.g. "laptop (1), cell phone (2)")
            summary_parts = [f"{lbl}" if cnt == 1 else f"{lbl} ({cnt})" for lbl, cnt in counts_by_class.items()]
            summary_str = ", ".join(summary_parts)

            return {
                "detected": True,
                "count": len(detected_objects),
                "objects": detected_objects,
                "summary": summary_str,
                "counts_by_class": counts_by_class,
            }

        except Exception as e:
            logger.error(f"Error during object detection forward pass: {e}", exc_info=True)
            return empty_result

    def draw_annotations(
        self,
        frame: np.ndarray,
        results: Dict[str, Any],
        suppress_person_boxes: bool = True,
    ) -> np.ndarray:
        """
        Renders HUD bounding boxes and label badges for detected objects.
        
        Args:
            frame: Input BGR frame.
            results: Dictionary returned by process_frame().
            suppress_person_boxes: If True, skips generic object boxes for 'person'
                                   to avoid duplicate bounding boxes over face badges.
        """
        if frame is None or not results.get("detected", False):
            return frame

        annotated = frame.copy()
        objects = results.get("objects", [])

        # Color palette for object categories (in BGR)
        COLOR_DEFAULT = (255, 200, 0)   # High-tech Cyan/Electric Blue
        COLOR_BAG = (0, 165, 255)       # Amber for backpacks/handbags
        COLOR_TECH = (255, 140, 0)      # Deep Cyan for electronics (laptop, phone)

        for obj in objects:
            lbl = obj["label"]
            conf = obj["confidence"]
            x, y, w, h = obj["bbox"]

            # Suppress person box if face recognition is handling persons
            if suppress_person_boxes and lbl == "person":
                continue

            # Select color styling
            if lbl in ("backpack", "handbag", "suitcase"):
                color = COLOR_BAG
            elif lbl in ("laptop", "cell phone", "mouse", "keyboard", "tv"):
                color = COLOR_TECH
            else:
                color = COLOR_DEFAULT

            # 1. Bounding box rectangle with corner accentuation
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # 2. Tag Pill Header
            badge_text = f"{lbl.upper()} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (t_w, t_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
            tag_y1 = max(0, y - t_h - 8)
            tag_y2 = y

            # Background fill for tag
            cv2.rectangle(annotated, (x, tag_y1), (x + t_w + 10, tag_y2), (20, 25, 35), -1)
            cv2.rectangle(annotated, (x, tag_y1), (x + t_w + 10, tag_y2), color, 1)

            # Text string
            cv2.putText(
                annotated,
                badge_text,
                (x + 5, y - 4),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    def should_trigger_alert(
        self,
        object_label: str,
        cooldown_seconds: float = OBJECT_ALERT_COOLDOWN_SECONDS,
    ) -> bool:
        """
        Thread-safe check to determine if an object event should be logged.
        Throttles duplicate notifications for continuous detections.
        """
        now = time.time()
        last_time = self._alert_cooldowns.get(object_label, 0.0)
        if now - last_time >= cooldown_seconds:
            self._alert_cooldowns[object_label] = now
            return True
        return False

    def log_detection_record(
        self,
        session_id: str,
        label: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[int]:
        """Persists detected object metadata to SQLite."""
        if self.db is None:
            return None
        try:
            bbox_json = json.dumps({"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]})
            return self.db.log_detected_object(
                session_id=session_id,
                label=label,
                confidence=confidence,
                bbox_json=bbox_json,
            )
        except Exception as e:
            logger.error(f"Failed to record detected object to SQLite: {e}")
            return None

    def release(self) -> None:
        """Frees model resources."""
        self._net = None
        self._is_ready = False
        logger.info("Object Detection Engine released.")
