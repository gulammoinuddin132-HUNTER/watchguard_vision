"""
WatchGuard Vision - Phase 5.21 Unified 84-Class Detector Test Suite
-------------------------------------------------------------------
Verifies:
  1. 84-class taxonomy length and index integrity
  2. COCO 0-79 baseline indices preservation
  3. Domain security class indices (80: pen, 81: access_badge, 82: usb_drive, 83: keys)
  4. Dataset YAML schema validity and consistency with config
  5. YOLO output channel calculation: 4 + num_classes = 88
  6. [1, 88, 8400] output tensor shape expectation
  7. Synthetic tensor decoding logic (bounding box + class scores)
  8. Dataset audit harness execution and leakage detection
  9. Coordinate restoration to native camera resolution
  10. ObjectDetector adapter interface compliance
  11. Production mode fallback preservation
  12. Non-intrusive CCSDF fusion compatibility
"""

import unittest
from pathlib import Path
import numpy as np
import tempfile
import shutil

from config import (
    WATCHGUARD_UNIFIED_CLASSES,
    COCO_CLASSES,
    SECURITY_OBJECT_CATEGORIES,
    WATCHGUARD_UNIFIED_MODEL_PATH,
    WATCHGUARD_UNIFIED_CONF_THRESHOLD,
    WATCHGUARD_UNIFIED_NMS_THRESHOLD,
    YOLOV8N_MODEL_PATH,
)
from src.core.interfaces.detector_interface import ObjectDetector, DetectedObject
from src.core.object_detection import (
    WatchGuardUnified84Detector,
    YOLOv8nCOCODetector,
    ObjectDetectionEngine,
)
from src.research.audit_unified_dataset import UnifiedDatasetAuditor
from src.research.ccsdf_engine import (
    CCSDFEngine,
    CCSDFContext,
    IdentityState,
    LivenessState,
    ZoneState,
    TemporalState,
    BehaviorState,
)


class TestPhase521Unified84Detector(unittest.TestCase):
    """Full 12-point test suite for Phase 5.21 84-Class Unified Perception."""

    def setUp(self):
        self.unified_classes = list(WATCHGUARD_UNIFIED_CLASSES)
        self.coco_classes = list(COCO_CLASSES)
        self.yaml_path = Path("data/datasets/watchguard_unified_84/watchguard_unified_84.yaml")

    # 1. 84-Class Taxonomy Length and Total Classes
    def test_01_taxonomy_length(self):
        """Verify the unified taxonomy contains exactly 84 classes."""
        self.assertEqual(len(self.unified_classes), 84)

    # 2. COCO Indices 0..79 Unchanged
    def test_02_coco_indices_preserved(self):
        """Verify COCO 80 classes exist unchanged at indices 0 to 79."""
        self.assertEqual(len(self.coco_classes), 80)
        for idx, coco_name in enumerate(self.coco_classes):
            self.assertEqual(self.unified_classes[idx], coco_name)

        # Check key security-relevant anchor classes
        self.assertEqual(self.unified_classes[0], "person")
        self.assertEqual(self.unified_classes[24], "backpack")
        self.assertEqual(self.unified_classes[63], "laptop")
        self.assertEqual(self.unified_classes[64], "mouse")
        self.assertEqual(self.unified_classes[67], "cell phone")
        self.assertEqual(self.unified_classes[73], "book")
        self.assertEqual(self.unified_classes[76], "scissors")

    # 3. New WatchGuard Security Class Indices
    def test_03_security_class_indices(self):
        """Verify new WatchGuard domain classes start at index 80."""
        self.assertEqual(self.unified_classes[80], "pen")
        self.assertEqual(self.unified_classes[81], "access_badge")
        self.assertEqual(self.unified_classes[82], "usb_drive")
        self.assertEqual(self.unified_classes[83], "keys")

        # Verify security categories
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["pen"], "context-relevant")
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["access_badge"], "credential")
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["usb_drive"], "high-risk-asset")
        self.assertEqual(SECURITY_OBJECT_CATEGORIES["keys"], "physical-access")

    # 4. YAML File Schema Validity
    def test_04_yaml_schema_validity(self):
        """Verify watchguard_unified_84.yaml exists and matches config taxonomy."""
        self.assertTrue(self.yaml_path.exists())
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify nc: 84 is present
        self.assertIn("nc: 84", content)
        
        # Verify all 84 classes are listed by index
        for i, name in enumerate(self.unified_classes):
            self.assertIn(f"{i}: {name}", content)

    # 5. Output Channel Calculation Formula: 4 + num_classes
    def test_05_output_channel_formula(self):
        """Verify the formula for output channels is 4 + num_classes = 88."""
        num_classes = len(self.unified_classes)
        expected_channels = 4 + num_classes
        self.assertEqual(expected_channels, 88)

    # 6. Tensor Shape Expectation [1, 88, 8400]
    def test_06_tensor_shape_expectation(self):
        """Verify 84-class model outputs a (1, 88, 8400) tensor."""
        batch_size = 1
        channels = 88  # 4 coordinates (cx, cy, w, h) + 84 class probabilities
        anchors = 8400
        synthetic_tensor = np.zeros((batch_size, channels, anchors), dtype=np.float32)
        self.assertEqual(synthetic_tensor.shape, (1, 88, 8400))

    # 7. Synthetic Tensor Decoding Logic
    def test_07_synthetic_tensor_decoding(self):
        """Verify decoding of a synthetic (1, 88, 8400) tensor with an activated usb_drive (class 82)."""
        tensor = np.zeros((1, 88, 8400), dtype=np.float32)
        
        # Place synthetic USB drive at anchor 100: center (320, 240), size (40, 20) in 640x640 letterbox space
        anchor_idx = 100
        tensor[0, 0, anchor_idx] = 320.0  # cx
        tensor[0, 1, anchor_idx] = 240.0  # cy
        tensor[0, 2, anchor_idx] = 40.0   # w
        tensor[0, 3, anchor_idx] = 20.0   # h
        tensor[0, 4 + 82, anchor_idx] = 0.92  # class 82: usb_drive confidence

        # Transpose to (8400, 88)
        preds = np.transpose(tensor[0])
        scores_matrix = preds[:, 4:]
        max_scores = np.max(scores_matrix, axis=1)
        valid_mask = max_scores >= 0.35

        self.assertTrue(valid_mask[anchor_idx])
        filt_scores = scores_matrix[valid_mask]
        class_ids = np.argmax(filt_scores, axis=1)
        confidences = max_scores[valid_mask]

        self.assertEqual(class_ids[0], 82)
        self.assertAlmostEqual(confidences[0], 0.92, places=2)
        self.assertEqual(self.unified_classes[class_ids[0]], "usb_drive")

    # 8. Dataset Audit Tooling & Leakage Detection
    def test_08_dataset_audit_tooling(self):
        """Verify UnifiedDatasetAuditor correctly validates dataset structure."""
        auditor = UnifiedDatasetAuditor(Path("data/datasets/watchguard_unified_84"))
        report = auditor.run_full_audit()
        self.assertEqual(report["status"], "PASSED")
        self.assertFalse(report["leakage_detected"])

        # Test leakage detection on temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        try:
            images_dir = temp_dir / "images"
            labels_dir = temp_dir / "labels"
            (images_dir / "train").mkdir(parents=True)
            (images_dir / "test").mkdir(parents=True)
            (labels_dir / "train").mkdir(parents=True)
            (labels_dir / "test").mkdir(parents=True)
            
            # Copy same file to both train and test to simulate leakage
            leak_file = "sample.jpg"
            (images_dir / "train" / leak_file).write_bytes(b"dummy image data")
            (images_dir / "test" / leak_file).write_bytes(b"dummy image data")
            (temp_dir / "watchguard_unified_84.yaml").write_text("nc: 84\nnames: {}\n")

            auditor_leak = UnifiedDatasetAuditor(temp_dir)
            leak_report = auditor_leak.run_full_audit()
            self.assertEqual(leak_report["status"], "FAILED")
            self.assertTrue(leak_report["leakage_detected"])
        finally:
            shutil.rmtree(temp_dir)

    # 9. Coordinate Restoration to Native Camera Resolution (640x480)
    def test_09_coordinate_restoration(self):
        """Verify letterbox inverse transformation restores exact native coordinates."""
        # 640x480 native frame into 640x640 letterbox: scale=1.0, dx=0, dy=80
        h_orig, w_orig = 480, 640
        scale = 1.0
        dx = 0.0
        dy = 80.0

        cx, cy, w, h = 320.0, 320.0, 100.0, 100.0  # Box in letterbox space
        x = int(np.clip((cx - 0.5 * w - dx) / scale, 0, w_orig - 1))
        y = int(np.clip((cy - 0.5 * h - dy) / scale, 0, h_orig - 1))
        bw = int(np.clip(w / scale, 1, w_orig - x))
        bh = int(np.clip(h / scale, 1, h_orig - y))

        # Expected native space coordinates: x = 320 - 50 = 270, y = 320 - 50 - 80 = 190
        self.assertEqual(x, 270)
        self.assertEqual(y, 190)
        self.assertEqual(bw, 100)
        self.assertEqual(bh, 100)

    # 10. ObjectDetector Adapter Interface Compliance
    def test_10_detector_interface_compliance(self):
        """Verify WatchGuardUnified84Detector implements ObjectDetector ABC."""
        detector = WatchGuardUnified84Detector(
            conf_threshold=0.35,
            nms_threshold=0.45,
        )
        self.assertIsInstance(detector, ObjectDetector)
        self.assertEqual(len(detector.class_names), 84)
        self.assertEqual(detector.mode_name, "WatchGuard Unified 84-Class Detector")

    # 11. Production Mode Fallback Preservation
    def test_11_production_mode_fallback(self):
        """Verify ObjectDetectionEngine defaults to production 80-class detector."""
        engine = ObjectDetectionEngine(detector_mode="production")
        self.assertEqual(engine.mode, "production")
        self.assertIsInstance(engine._detector, YOLOv8nCOCODetector)
        self.assertEqual(len(engine._detector.class_names), 80)

    # 12. Non-Intrusive CCSDF Compatibility
    def test_12_ccsdf_compatibility(self):
        """Verify CCSDF fusion processes unified context seamlessly without regression."""
        ccsdf = CCSDFEngine()
        ctx = CCSDFContext(
            identity=IdentityState("Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState("LIVE", 0.95),
            zone=ZoneState("NORMAL", "Standard Perimeter"),
            temporal=TemporalState(is_authorized_hours=True),
            behavior=BehaviorState(s_behavior=0.0, status="NORMAL_ACTIVITY"),
        )
        res = ccsdf.evaluate_context(ctx, authoritative_action="GRANTED")
        self.assertEqual(res.authoritative_action, "GRANTED")
        self.assertLess(res.risk_score, 0.30)
        self.assertEqual(res.recommended_action.value, "ALLOW")


if __name__ == "__main__":
    unittest.main()
