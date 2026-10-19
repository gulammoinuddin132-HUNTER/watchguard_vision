"""
Unit & Integration Tests for CameraManager
------------------------------------------
Validates:
1. Physical device discovery and probing (index 0).
2. Explicit startup diagnostics (REAL, SYNTHETIC, ERROR).
3. No silent fallback: Physical camera failure reports CameraStatus.ERROR when fallback is disallowed.
4. Explicit synthetic simulator modes ('none', 'authorized', 'unknown').
5. Continuous frame streaming and snapshot capture.
"""

from pathlib import Path
import sys
import time
import unittest
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT
from src.camera.camera_manager import CameraManager, CameraStatus, CameraMode


class TestCameraManager(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "camera") and self.camera.is_running:
            self.camera.stop()

    def test_physical_device_probe(self):
        """Verify CameraManager probe finds connected physical webcam devices."""
        self.camera = CameraManager(camera_index=0)
        devices = self.camera.probe_available_devices(max_indices=1)
        self.assertIsInstance(devices, list)
        if devices:
            self.assertIn("index", devices[0])
            self.assertIn("backend", devices[0])
            self.assertTrue(devices[0]["opened"])

    def test_physical_camera_stream_and_diagnostics(self):
        """Verify real physical camera opens on index 0 and streams valid frames."""
        self.camera = CameraManager(camera_index=0)
        started = self.camera.start(camera_index=0, mode=CameraMode.PHYSICAL, allow_synthetic_fallback=False)
        
        # If hardware camera exists, status MUST be REAL (never silently synthetic)
        if started:
            self.assertEqual(self.camera.status, CameraStatus.REAL)
            self.assertTrue(self.camera.is_physical)
            self.assertFalse(self.camera.is_fallback_mode)
            self.assertIn("REAL", self.camera.status_label)

            time.sleep(0.1)
            success, frame = self.camera.get_frame()
            self.assertTrue(success)
            self.assertIsNotNone(frame)
            self.assertEqual(len(frame.shape), 3)
            self.assertEqual(frame.shape[2], 3)

            # Test RGB conversion
            success_rgb, frame_rgb = self.camera.get_frame_rgb()
            self.assertTrue(success_rgb)
            self.assertIsNotNone(frame_rgb)

            # Test snapshot
            snap_path = self.camera.capture_snapshot(filename_prefix="test_cam_real")
            self.assertIsNotNone(snap_path)
            self.camera.stop()

    def test_explicit_error_when_physical_device_missing(self):
        """
        CRITICAL TEST:
        When an invalid physical camera index (99) is requested without synthetic fallback,
        CameraManager MUST report CameraStatus.ERROR rather than silently falling back to synthetic.
        """
        self.camera = CameraManager(camera_index=99)
        started = self.camera.start(camera_index=99, mode=CameraMode.PHYSICAL, allow_synthetic_fallback=False, auto_discover=False)

        self.assertFalse(started)
        self.assertEqual(self.camera.status, CameraStatus.ERROR)
        self.assertIsNotNone(self.camera.error_message)
        self.assertIn("ERROR", self.camera.status_label)

    def test_explicit_synthetic_simulation_mode(self):
        """Verify synthetic simulator mode functions properly when explicitly requested."""
        self.camera = CameraManager()
        self.camera.set_demo_mode("authorized")
        started = self.camera.start(mode=CameraMode.SYNTHETIC)

        self.assertTrue(started)
        self.assertEqual(self.camera.status, CameraStatus.SYNTHETIC)
        self.assertTrue(self.camera.is_fallback_mode)
        self.assertFalse(self.camera.is_physical)

        success, frame = self.camera.get_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)

        # Switch demo targets
        self.camera.set_demo_mode("unknown")
        self.assertEqual(self.camera.demo_mode, "unknown")

    def test_horizontal_flip_toggle(self):
        """Verify horizontal_flip parameter correctly flips frames on acquisition."""
        cam_unflipped = CameraManager(camera_index=0, horizontal_flip=False)
        self.assertFalse(cam_unflipped.horizontal_flip)

        cam_flipped = CameraManager(camera_index=0, horizontal_flip=True)
        self.assertTrue(cam_flipped.horizontal_flip)

    def test_horizontal_flip_get_frame_pipeline_validation(self):
        """
        Verify that CameraManager.get_frame() delivers horizontally flipped frames
        when horizontal_flip=True, and unflipped frames when horizontal_flip=False.
        """
        from unittest.mock import MagicMock

        class MockAsymmetricCapture:
            def __init__(self, frame):
                self._frame = frame
                self._opened = True
            def isOpened(self):
                return self._opened
            def read(self):
                return True, self._frame.copy()
            def release(self):
                self._opened = False
            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_WIDTH: return float(self._frame.shape[1])
                if prop == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._frame.shape[0])
                if prop == cv2.CAP_PROP_FPS: return 30.0
                return 0.0
            def set(self, prop, val):
                return True

        # Asymmetric test frame with feature on the left (x: 50..150)
        asym_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        asym_frame[100:200, 50:150] = 255

        # 1. Test horizontal_flip=True through get_frame()
        cam_flipped = CameraManager(camera_index=0, horizontal_flip=True)
        cam_flipped._test_and_open_device = MagicMock(
            return_value=(MockAsymmetricCapture(asym_frame), 640, 480, asym_frame.copy())
        )
        started = cam_flipped.start(mode=CameraMode.PHYSICAL, allow_synthetic_fallback=False)
        self.assertTrue(started)
        time.sleep(0.04)
        success, frame_flipped = cam_flipped.get_frame()
        cam_flipped.stop()

        self.assertTrue(success)
        self.assertIsNotNone(frame_flipped)
        # Left region must be 0; right region (490..590) must have the feature
        self.assertEqual(frame_flipped[100:200, 50:150].sum(), 0)
        self.assertGreater(frame_flipped[100:200, 490:590].sum(), 0)

        # 2. Test horizontal_flip=False through get_frame()
        cam_unflipped = CameraManager(camera_index=0, horizontal_flip=False)
        cam_unflipped._test_and_open_device = MagicMock(
            return_value=(MockAsymmetricCapture(asym_frame), 640, 480, asym_frame.copy())
        )
        started = cam_unflipped.start(mode=CameraMode.PHYSICAL, allow_synthetic_fallback=False)
        self.assertTrue(started)
        time.sleep(0.04)
        success, frame_unflipped = cam_unflipped.get_frame()
        cam_unflipped.stop()

        self.assertTrue(success)
        self.assertIsNotNone(frame_unflipped)
        # Left region must retain the feature; right region must be 0
        self.assertGreater(frame_unflipped[100:200, 50:150].sum(), 0)
        self.assertEqual(frame_unflipped[100:200, 490:590].sum(), 0)

    def test_synthetic_demo_frames_are_not_flipped(self):
        """
        Verify that demo/synthetic simulator mode is independent of horizontal_flip,
        does not pass frames through cv2.flip(), and preserves intended left-to-right orientation.
        """
        from unittest.mock import patch

        with patch("cv2.flip", wraps=cv2.flip) as mock_flip:
            cam_demo = CameraManager(horizontal_flip=True)
            cam_demo.set_demo_mode("authorized")
            started = cam_demo.start(mode=CameraMode.SYNTHETIC)
            self.assertTrue(started)

            time.sleep(0.04)
            success, demo_frame = cam_demo.get_frame()
            cam_demo.stop()

            self.assertTrue(success)
            self.assertIsNotNone(demo_frame)
            expected_shape = (CAMERA_FRAME_HEIGHT, CAMERA_FRAME_WIDTH, 3)
            self.assertEqual(demo_frame.shape, expected_shape)

            # 1. Confirm synthetic frames do NOT pass through physical-camera cv2.flip()
            self.assertEqual(mock_flip.call_count, 0)

            # 2. Confirm orientation: synthetic header text is placed on the left
            # and is un-mirrored (left header region energy is consistently higher than the mirrored right region)
            h, w = demo_frame.shape[:2]
            box_w = min(300, w // 4)
            box_h = min(110, int(h * 0.25))
            left_header_energy = demo_frame[20:box_h, 30:30 + box_w].sum()
            right_mirrored_energy = demo_frame[20:box_h, w - box_w - 30:w - 30].sum()
            self.assertGreater(left_header_energy, right_mirrored_energy * 2.5)

    def test_initial_preview_and_subsequent_physical_frames_flipped_independently(self):
        """
        Verify that the initial physical-camera preview frame from device discovery
        and subsequent frames from the capture loop are each flipped exactly once independently.
        """
        import threading
        from unittest.mock import MagicMock, patch

        class ControlledMockCap:
            def __init__(self, subsequent_frame):
                self._frame = subsequent_frame
                self._opened = True
            def isOpened(self):
                return self._opened
            def read(self):
                return True, self._frame.copy()
            def release(self):
                self._opened = False
            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_WIDTH: return 640.0
                if prop == cv2.CAP_PROP_FRAME_HEIGHT: return 480.0
                if prop == cv2.CAP_PROP_FPS: return 30.0
                return 0.0
            def set(self, prop, val):
                return True

        # Initial probe frame: asymmetric feature on top-left (y: 100..200, x: 50..150)
        initial_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        initial_raw[100:200, 50:150] = 255

        # Subsequent capture frame: asymmetric feature on lower-left (y: 250..350, x: 60..160)
        subsequent_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        subsequent_raw[250:350, 60:160] = 200

        mock_cap = ControlledMockCap(subsequent_raw)

        with patch("cv2.flip", wraps=cv2.flip) as mock_flip:
            cam = CameraManager(camera_index=0, horizontal_flip=True)
            cam._test_and_open_device = MagicMock(
                return_value=(mock_cap, 640, 480, initial_raw.copy())
            )

            # Step 1: Open device without starting capture thread to isolate initial preview frame
            opened = cam._open_physical_camera(auto_discover=False)
            self.assertTrue(opened)

            # Verify initial preview frame in _latest_frame: flipped exactly once
            self.assertEqual(mock_flip.call_count, 1)
            success_init, init_frame = cam.get_frame()
            self.assertTrue(success_init)
            self.assertIsNotNone(init_frame)
            # Feature was on left (50..150); now on right (490..590)
            self.assertEqual(init_frame[100:200, 50:150].sum(), 0)
            self.assertGreater(init_frame[100:200, 490:590].sum(), 0)

            # Step 2: Start worker thread to capture subsequent frame from capture loop
            cam._running = True
            worker_thread = threading.Thread(target=cam._capture_loop, daemon=True)
            worker_thread.start()
            time.sleep(0.05)

            # Retrieve subsequent frame before stopping
            success_sub, sub_frame = cam.get_frame()
            cam.stop()

            self.assertTrue(success_sub)
            self.assertIsNotNone(sub_frame)
            # Subsequent feature was on left (60..160); now on right (480..580)
            self.assertEqual(sub_frame[250:350, 60:160].sum(), 0)
            self.assertGreater(sub_frame[250:350, 480:580].sum(), 0)

            # Total flip calls: initial frame (1) + capture loop frames (>= 1)
            self.assertGreaterEqual(mock_flip.call_count, 2)


if __name__ == "__main__":
    unittest.main()


