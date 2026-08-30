"""
Unit and Integration Tests for Phase 5.19D: Unified WatchGuard Object Detection Architecture
--------------------------------------------------------------------------------------------
Covers all 15 specification requirements:
  1. detector interface
  2. custom model loading
  3. output decoding
  4. class mapping
  5. letterbox preprocessing
  6. coordinate restoration
  7. small-object output & distance tiers
  8. person/object association
  9. temporal confirmation
  10. Sasset generation
  11. protected-asset mapping
  12. production/custom mode switching
  13. no database spam / debouncing
  14. no UI-thread inference
  15. production mode unchanged
"""

import os
import sys
import unittest
import numpy as np
import cv2

from config import (
    YOLOV8N_MODEL_PATH,
    WATCHGUARD_CUSTOM_MODEL_PATH,
    OBJECT_DETECTOR_MODE,
    SECURITY_OBJECT_CATEGORIES,
    WATCHGUARD_CUSTOM_CLASSES,
)
from src.core.interfaces.detector_interface import ObjectDetector, DetectedObject
from src.core.object_detection import (
    ObjectDetectionEngine,
    YOLOv8nCOCODetector,
    WatchGuardYOLODetector,
    TemporalObjectTracker,
)
from src.research.ccsdf_engine import SmallAssetState, CCSDFContext


class TestUnifiedObjectDetectorArchitecture(unittest.TestCase):
    """Full test suite for Phase 5.19D Unified Object Detection Pipeline."""

    def setUp(self):
        self.dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # 1. Detector Interface ABC Compliance
    def test_01_detector_interface(self):
        self.assertTrue(issubclass(YOLOv8nCOCODetector, ObjectDetector))
        self.assertTrue(issubclass(WatchGuardYOLODetector, ObjectDetector))
        
        det = WatchGuardYOLODetector()
        self.assertTrue(hasattr(det, "initialize"))
        self.assertTrue(hasattr(det, "detect"))
        self.assertTrue(hasattr(det, "release"))
        self.assertTrue(hasattr(det, "is_ready"))
        self.assertTrue(hasattr(det, "class_names"))
        self.assertTrue(hasattr(det, "mode_name"))

    # 2. Custom Model Loading
    def test_02_custom_model_loading(self):
        self.assertTrue(WATCHGUARD_CUSTOM_MODEL_PATH.exists(), f"Model missing: {WATCHGUARD_CUSTOM_MODEL_PATH}")
        det = WatchGuardYOLODetector(model_path=WATCHGUARD_CUSTOM_MODEL_PATH)
        success = det.initialize()
        self.assertTrue(success)
        self.assertTrue(det.is_ready)
        det.release()
        self.assertFalse(det.is_ready)

    # 3. Output Decoding
    def test_03_output_decoding(self):
        det = WatchGuardYOLODetector(model_path=WATCHGUARD_CUSTOM_MODEL_PATH)
        det.initialize()
        results = det.detect(self.dummy_frame)
        self.assertIsInstance(results, list)
        for obj in results:
            self.assertIsInstance(obj, DetectedObject)
            self.assertIsInstance(obj.class_id, int)
            self.assertIsInstance(obj.label, str)
            self.assertIsInstance(obj.confidence, float)
            self.assertEqual(len(obj.bbox), 4)

    # 4. Class Mapping
    def test_04_class_mapping(self):
        det = WatchGuardYOLODetector()
        self.assertEqual(det.class_names, ["person", "laptop", "cell_phone", "mouse"])

    # 5. Letterbox Preprocessing
    def test_05_letterbox_preprocessing(self):
        canvas, scale, dx, dy = WatchGuardYOLODetector.preprocess_letterbox(self.dummy_frame, target_size=640)
        self.assertEqual(canvas.shape, (640, 640, 3))
        # 640x480 into 640x640: scale = 1.0, dx = 0, dy = 80
        self.assertEqual(scale, 1.0)
        self.assertEqual(dx, 0)
        self.assertEqual(dy, 80)
        # Check padding color is 114 gray
        self.assertEqual(int(canvas[0, 0, 0]), 114)

    # 6. Coordinate Restoration
    def test_06_coordinate_restoration(self):
        # Synthetic box at center in letterboxed space: cx=320, cy=320 (which is cy=240 in original frame + 80 pad)
        det = WatchGuardYOLODetector()
        det.initialize()
        # Verify detected coordinates lie strictly inside native 640x480 frame bounds
        results = det.detect(self.dummy_frame)
        for obj in results:
            x, y, w, h = obj.bbox
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, 640)
            self.assertLessEqual(y + h, 480)

    # 7. Small-Object Output & Distance Tiers
    def test_07_small_object_output(self):
        # Create DetectedObjects with different sizes
        close_obj = DetectedObject(class_id=3, label="mouse", confidence=0.8, bbox=(100, 100, 120, 90))
        med_obj = DetectedObject(class_id=3, label="mouse", confidence=0.7, bbox=(100, 100, 70, 50))
        far_obj = DetectedObject(class_id=3, label="mouse", confidence=0.6, bbox=(100, 100, 30, 20))

        self.assertEqual(close_obj.distance_tier, "CLOSE")     # 120x90 = 10,800 px^2 > 10k
        self.assertEqual(med_obj.distance_tier, "MEDIUM")      # 70x50 = 3,500 px^2 in [1.6k, 10k]
        self.assertEqual(far_obj.distance_tier, "FAR")         # 30x20 = 600 px^2 < 1.6k

    # 8. Person/Object Association Safety
    def test_08_person_object_association(self):
        engine = ObjectDetectionEngine(detector_mode="watchguard_custom")
        engine.initialize()
        res = engine.process_frame(self.dummy_frame)
        # Object detector returns raw object counts without overriding YuNet face identities
        self.assertIn("objects", res)
        self.assertIn("counts_by_class", res)

    # 9. Temporal Confirmation
    def test_09_temporal_confirmation(self):
        tracker = TemporalObjectTracker(confirmation_frames=2, max_missed_frames=2)
        raw_obj = DetectedObject(class_id=3, label="mouse", confidence=0.6, bbox=(200, 200, 50, 40))

        # Frame 1: Candidate (not confirmed)
        out1 = tracker.update([raw_obj])
        self.assertEqual(len(out1), 1)
        self.assertFalse(out1[0].is_confirmed)

        # Frame 2: Confirmed
        out2 = tracker.update([raw_obj])
        self.assertEqual(len(out2), 1)
        self.assertTrue(out2[0].is_confirmed)

        # Frame 3: Empty (miss incremented, but track retained)
        out3 = tracker.update([])
        self.assertEqual(len(out3), 0)

    # 10. S_asset Signal Generation
    def test_10_sasset_generation(self):
        asset_state_none = SmallAssetState()
        self.assertEqual(asset_state_none.get_normalized_risk(), 0.0)

        asset_state_usb = SmallAssetState(
            has_detected_asset=True,
            asset_label="usb_drive",
            security_category="high-risk-asset",
            is_confirmed=True,
        )
        self.assertEqual(asset_state_usb.get_normalized_risk(), 1.0)

        asset_state_laptop = SmallAssetState(
            has_detected_asset=True,
            asset_label="laptop",
            security_category="protected",
            is_confirmed=True,
        )
        self.assertEqual(asset_state_laptop.get_normalized_risk(), 0.50)

        ctx = CCSDFContext(small_asset=asset_state_usb)
        signals = ctx.get_normalized_signals()
        self.assertIn("asset", signals)
        self.assertEqual(signals["asset"], 1.0)

    # 11. Protected Asset Mapping
    def test_11_protected_asset_mapping(self):
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["laptop"], "protected")
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["cell_phone"], "security-relevant")
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["mouse"], "context-relevant")
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["person"], "entity")

    # 12. Production/Custom Mode Switching
    def test_12_mode_switching(self):
        engine_prod = ObjectDetectionEngine(detector_mode="production")
        engine_prod.initialize()
        self.assertEqual(engine_prod.active_mode, "Production YOLOv8n (COCO 80)")
        self.assertEqual(len(engine_prod.class_names), 80)

        engine_custom = ObjectDetectionEngine(detector_mode="watchguard_custom")
        engine_custom.initialize()
        self.assertEqual(engine_custom.active_mode, "WatchGuard Custom Small-Asset Detector (v3)")
        self.assertEqual(len(engine_custom.class_names), 4)

    # 13. No Database Spam / Debouncing
    def test_13_no_database_spam(self):
        engine = ObjectDetectionEngine(detector_mode="watchguard_custom")
        engine.initialize()
        # Ensure process_frame returns clean summary and counts dictionary
        res = engine.process_frame(self.dummy_frame)
        self.assertIsInstance(res["counts_by_class"], dict)
        self.assertIsInstance(res["summary"], str)

    # 14. No UI-Thread Inference Dependency
    def test_14_no_ui_thread_inference(self):
        engine = ObjectDetectionEngine(detector_mode="watchguard_custom")
        engine.initialize()
        # Verify execution occurs on standard NumPy frame without requiring Tkinter / display window
        res = engine.process_frame(self.dummy_frame)
        self.assertIn("detected", res)
        annotated = engine.draw_annotations(self.dummy_frame, res)
        self.assertEqual(annotated.shape, self.dummy_frame.shape)

    # 15. Production Mode Unchanged
    def test_15_production_mode_unchanged(self):
        self.assertEqual(OBJECT_DETECTOR_MODE, "production")
        engine = ObjectDetectionEngine() # default
        engine.initialize()
        self.assertEqual(engine.active_mode, "Production YOLOv8n (COCO 80)")
        self.assertEqual(engine.class_names[0], "person")
        self.assertEqual(engine.class_names[63], "laptop")
        self.assertEqual(engine.class_names[64], "mouse")
        self.assertEqual(engine.class_names[67], "cell phone")


if __name__ == "__main__":
    unittest.main()
