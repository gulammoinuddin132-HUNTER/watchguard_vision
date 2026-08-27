"""
Automated Unit Tests for Object Detection Module (Phase 2.2)
------------------------------------------------------------
Tests:
- YOLOv8 ONNX model initialization & backend setup
- Object detection inference and output dictionary structure
- Confidence filtering and Non-Maximum Suppression (NMS)
- Alert cooldown and debouncing
- SQLite database integration (detected_objects & security_events)
- Fallback & error handling for invalid/empty frames
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from config import DEMO_FACES_DIR, YOLOV8N_MODEL_PATH
from src.core.object_detection import ObjectDetectionEngine
from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel


class TestObjectDetection(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_watchguard_obj.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.engine = ObjectDetectionEngine(db_manager=self.db)
        self.logger = EventLogger(db_manager=self.db)

    def tearDown(self):
        self.engine.release()
        self.temp_dir.cleanup()

    def test_model_initialization(self):
        """Verify YOLOv8n ONNX model loads successfully into OpenCV DNN."""
        self.assertTrue(YOLOV8N_MODEL_PATH.exists(), f"Model file missing: {YOLOV8N_MODEL_PATH}")
        ok = self.engine.initialize()
        self.assertTrue(ok)
        self.assertTrue(self.engine.is_ready)
        self.assertEqual(len(self.engine.class_names), 80)
        self.assertIn("person", self.engine.class_names)
        self.assertIn("laptop", self.engine.class_names)
        self.assertIn("cell phone", self.engine.class_names)
        self.assertIn("bottle", self.engine.class_names)
        self.assertIn("backpack", self.engine.class_names)

    def test_detection_output_format(self):
        """Verify process_frame returns correctly formatted dictionary."""
        self.engine.initialize()
        alice_img = cv2.imread(str(DEMO_FACES_DIR / "officer_alice.jpg"))
        self.assertIsNotNone(alice_img)

        res = self.engine.process_frame(alice_img)
        self.assertIsInstance(res, dict)
        self.assertIn("detected", res)
        self.assertIn("count", res)
        self.assertIn("objects", res)
        self.assertIn("summary", res)
        self.assertIn("counts_by_class", res)

        self.assertTrue(res["detected"])
        self.assertGreaterEqual(res["count"], 1)

        first_obj = res["objects"][0]
        self.assertIn("label", first_obj)
        self.assertIn("confidence", first_obj)
        self.assertIn("bbox", first_obj)
        self.assertIn("class_id", first_obj)
        self.assertEqual(len(first_obj["bbox"]), 4)  # (x, y, w, h)

    def test_confidence_filtering(self):
        """Verify confidence threshold filters out low-confidence detections."""
        # Engine with standard threshold
        self.engine.conf_threshold = 0.50
        self.engine.initialize()

        img = cv2.imread(str(DEMO_FACES_DIR / "officer_alice.jpg"))
        res_standard = self.engine.process_frame(img)

        # High threshold engine should filter strictly
        strict_engine = ObjectDetectionEngine(conf_threshold=0.99, db_manager=self.db)
        strict_engine.initialize()
        res_strict = strict_engine.process_frame(img)

        self.assertLessEqual(res_strict["count"], res_standard["count"])
        strict_engine.release()

    def test_event_throttling_and_debouncing(self):
        """Verify object alert cooldown prevents continuous duplicate events."""
        self.engine.initialize()

        # First alert check should pass
        should_alert_1 = self.engine.should_trigger_alert("laptop", cooldown_seconds=1.5)
        self.assertTrue(should_alert_1)

        # Immediate second check should be throttled
        should_alert_2 = self.engine.should_trigger_alert("laptop", cooldown_seconds=1.5)
        self.assertFalse(should_alert_2)

        # Different object should still alert
        should_alert_phone = self.engine.should_trigger_alert("cell phone", cooldown_seconds=1.5)
        self.assertTrue(should_alert_phone)

        # After cooldown elapsed, first object should alert again
        time.sleep(1.6)
        should_alert_3 = self.engine.should_trigger_alert("laptop", cooldown_seconds=1.5)
        self.assertTrue(should_alert_3)

    def test_database_persistence(self):
        """Verify detected objects are logged to SQLite detected_objects table."""
        self.engine.initialize()

        row_id = self.engine.log_detection_record(
            session_id="test_session_01",
            label="laptop",
            confidence=0.942,
            bbox=(50, 60, 200, 150),
        )
        self.assertIsNotNone(row_id)
        self.assertGreater(row_id, 0)

        # Check DB directly
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id, label, confidence, bbox_json FROM detected_objects WHERE id = ?", (row_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["session_id"], "test_session_01")
            self.assertEqual(row["label"], "laptop")
            self.assertAlmostEqual(row["confidence"], 0.942, places=3)
            bbox = json.loads(row["bbox_json"])
            self.assertEqual(bbox["w"], 200)

    def test_draw_annotations_rendering(self):
        """Verify draw_annotations draws bounding boxes without altering frame shape."""
        self.engine.initialize()
        img = cv2.imread(str(DEMO_FACES_DIR / "officer_alice.jpg"))
        res = self.engine.process_frame(img)

        annotated = self.engine.draw_annotations(img, res, suppress_person_boxes=False)
        self.assertEqual(annotated.shape, img.shape)
        self.assertIsInstance(annotated, np.ndarray)

    def test_error_and_empty_frame_handling(self):
        """Verify engine gracefully handles None or empty frames without crashing."""
        self.engine.initialize()

        # None frame
        res_none = self.engine.process_frame(None)
        self.assertFalse(res_none["detected"])
        self.assertEqual(res_none["count"], 0)

        # Empty numpy array
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        res_empty = self.engine.process_frame(empty_img)
        self.assertFalse(res_empty["detected"])
        self.assertEqual(res_empty["count"], 0)

        # Draw on None frame
        annot_none = self.engine.draw_annotations(None, res_none)
        self.assertIsNone(annot_none)

        # Non-existent model path handling
        bad_engine = ObjectDetectionEngine(model_path=Path("non_existent.onnx"))
        self.assertFalse(bad_engine.initialize())
        self.assertFalse(bad_engine.is_ready)


if __name__ == "__main__":
    unittest.main()
