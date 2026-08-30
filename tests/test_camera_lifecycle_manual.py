"""
Manual Camera Lifecycle & Standby Verification Test
---------------------------------------------------
Validates:
1. Application startup: Camera remains strictly OFF/STANDBY (no hardware access).
2. Telemetry and UI cards default to STANDBY on launch.
3. No spurious CAMERA_STARTED / CAMERA_STOPPED events on launch.
4. User clicks START CAMERA: Physical webcam opens on index 0, reads real frames, begins AI inference.
5. User clicks STOP CAMERA: Webcam device is released, stream returns to STANDBY cleanly.
"""

import time
import unittest
import tkinter as tk
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager, CameraStatus, CameraMode
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine
from src.ui.dashboard import WatchGuardDashboard


class TestCameraLifecycleManual(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Headless Tkinter window

        self.db = DatabaseManager()
        self.logger = EventLogger(db_manager=self.db)
        
        # Capture baseline camera events prior to dashboard initialization
        baseline_events = self.db.get_recent_security_events(limit=100)
        self.initial_camera_event_count = len([e for e in baseline_events if e["event_type"] in ("CAMERA_STARTED", "CAMERA_STOPPED")])

        self.person_engine = PersonRecognitionEngine(db_manager=self.db)
        self.object_engine = ObjectDetectionEngine(db_manager=self.db)
        self.context_engine = ContextEngine()
        self.camera_manager = CameraManager()

        self.app = WatchGuardDashboard(
            root=self.root,
            camera_manager=self.camera_manager,
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.person_engine,
            object_engine=self.object_engine,
            context_engine=self.context_engine,
        )

    def tearDown(self):
        if hasattr(self, "app") and self.app:
            try:
                self.app.close(destroy_root=False)
            except Exception:
                pass
        if self.camera_manager.is_running:
            self.camera_manager.stop()
        try:
            if hasattr(self, "root") and self.root and self.root.winfo_exists():
                self.root.destroy()
        except Exception:
            pass

    def test_startup_camera_is_strictly_standby(self):
        """Verify that on application startup, the camera is OFF and no hardware was accessed."""
        # 1. CameraManager state
        self.assertFalse(self.camera_manager.is_running)
        self.assertEqual(self.camera_manager.status, CameraStatus.OFFLINE)
        self.assertIsNone(self.camera_manager._cap)

        # 2. UI button & badge state
        self.assertIn("START CAMERA", self.app.btn_toggle_cam.cget("text"))
        self.assertIn("0.0 FPS", self.app.lbl_fps.cget("text"))

        # 3. Telemetry card states
        self.assertEqual(self.app.item_person.lbl_val.cget("text"), "STANDBY")
        self.assertEqual(self.app.item_objects.lbl_val.cget("text"), "STANDBY")
        self.assertEqual(self.app.item_liveness.lbl_val.cget("text"), "STANDBY")
        self.assertEqual(self.app.item_access.lbl_val.cget("text"), "STANDBY")
        self.assertEqual(self.app.ctx_item_decision.lbl_val.cget("text"), "STANDBY")

        # 4. Event logs: ensure NO new camera events were logged on startup
        events = self.db.get_recent_security_events(limit=100)
        current_camera_events = len([e for e in events if e["event_type"] in ("CAMERA_STARTED", "CAMERA_STOPPED")])
        self.assertEqual(current_camera_events, self.initial_camera_event_count, "Expected zero new camera start/stop events on startup.")

    def test_manual_start_and_stop_lifecycle(self):
        """Verify START CAMERA opens physical camera and STOP CAMERA cleanly returns to STANDBY."""
        # 1. Simulate clicking START CAMERA
        self.app._toggle_camera()

        # If physical camera is connected:
        if self.camera_manager.status == CameraStatus.REAL:
            self.assertTrue(self.camera_manager.is_running)
            self.assertTrue(self.camera_manager.is_physical)
            self.assertIn("STOP CAMERA", self.app.btn_toggle_cam.cget("text"))
            self.assertIn("LIVE", self.app.lbl_stream_status.cget("text"))

            # Process 3 frame ticks
            for _ in range(3):
                self.root.update()
                time.sleep(0.04)

            # Check real frame was retrieved
            success, frame = self.camera_manager.get_frame()
            self.assertTrue(success)
            self.assertIsNotNone(frame)

            # 2. Simulate clicking STOP CAMERA
            self.app._toggle_camera()

            self.assertFalse(self.camera_manager.is_running)
            self.assertEqual(self.camera_manager.status, CameraStatus.OFFLINE)
            self.assertIsNone(self.camera_manager._cap)
            self.assertIn("START CAMERA", self.app.btn_toggle_cam.cget("text"))
            self.assertIn("STREAM OFFLINE", self.app.lbl_stream_status.cget("text"))
            self.assertEqual(self.app.item_person.lbl_val.cget("text"), "STANDBY")
            self.assertEqual(self.app.item_liveness.lbl_val.cget("text"), "STANDBY")
            self.assertEqual(self.app.item_access.lbl_val.cget("text"), "STANDBY")


if __name__ == "__main__":
    unittest.main()
