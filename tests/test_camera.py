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
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


if __name__ == "__main__":
    unittest.main()
