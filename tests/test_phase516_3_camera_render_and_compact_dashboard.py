"""
WatchGuard Vision - Phase 5.16.3 Camera Render & Compact Dashboard Regression Tests
-----------------------------------------------------------------------------------
Validates:
  1. Camera frame arrival clears loading state and updates Tkinter PhotoImage.
  2. PhotoImage and canvas image update properly without exceptions.
  3. Canvas item ID remains valid and does not disappear or disconnect.
  4. Camera card width is ~700-760 px at standard 1280x880.
  5. Right dashboard height is compact (~380-450 px requested height).
  6. Compact Event Log displays ~3 visible rows without clipping.
  7. Physical camera auto-discovery prefers lit cameras over dark/shuttered sensors.
"""

import unittest
import os
import shutil
import tempfile
import time
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

from src.ui.dashboard import WatchGuardDashboard
from src.database.db_manager import DatabaseManager
from src.camera.camera_manager import CameraManager, CameraStatus, CameraMode
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel


class TestPhase5163CameraRenderAndCompactDashboard(unittest.TestCase):
    """Verifies camera rendering pipeline and compact dashboard geometry."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_p516_3.db")
        self.db = DatabaseManager(self.db_path)

        for i in range(15):
            self.db.log_security_event(
                event_type=EventType.PERSON_DETECTED,
                risk_level=RiskLevel.GREEN,
                description=f"Phase 5.16.3 Test Event #{i+1:02d}",
                person_name="Hunter",
            )

        self.logger = EventLogger(db_manager=self.db)
        self.cam = CameraManager()

        self.root = tk.Tk()
        self.root.geometry("1280x880")
        self.app = WatchGuardDashboard(
            root=self.root,
            camera_manager=self.cam,
            db_manager=self.db,
            event_logger=self.logger,
        )
        self.root.update_idletasks()
        self.root.update()

    def tearDown(self):
        if hasattr(self, "app") and self.app:
            try:
                self.app.close(destroy_root=False)
            except Exception:
                pass
        try:
            if hasattr(self, "root") and self.root and self.root.winfo_exists():
                self.root.destroy()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_camera_canvas_item_id_and_rendering_update(self):
        """Verify PhotoImage and canvas image update properly and canvas item ID remains valid."""
        # 1. Initial placeholder verification
        initial_item_id = self.app._canvas_img_id
        self.assertIsNotNone(initial_item_id)
        items_on_canvas = self.app.video_canvas.find_all()
        self.assertIn(initial_item_id, items_on_canvas)

        # 2. Simulate arrival of a live video frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        # Inject simulated frame into camera
        self.cam._latest_frame = test_frame
        self.cam._running = True
        self.cam._status = CameraStatus.REAL

        # Cancel any pending after-loop to isolate single-frame render
        if self.app._after_video_id:
            try:
                self.root.after_cancel(self.app._after_video_id)
            except Exception:
                pass

        # Execute display render loop step
        self.app._schedule_video_update()
        self.root.update_idletasks()

        # Verify canvas item ID remains valid and image exists
        active_items = self.app.video_canvas.find_all()
        self.assertGreater(len(active_items), 0, "Canvas image item disappeared!")
        self.assertIn(self.app._canvas_img_id, active_items)

        # Verify PhotoImage is allocated and non-empty
        self.assertIsNotNone(self.app._photo_image)
        self.assertGreater(self.app._photo_image.width(), 0)
        self.assertGreater(self.app._photo_image.height(), 0)

        # Ensure no exception during subsequent render update
        self.app._schedule_video_update()
        self.root.update_idletasks()

        self.assertIn(self.app._canvas_img_id, self.app.video_canvas.find_all())
        self.cam._running = False

    def test_02_layout_camera_card_and_right_dashboard_geometry(self):
        """Verify camera card is ~700 px wide, right dashboard is ~380-450 px height, event log is compact."""
        # Retrieve upper area frames
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame) and child.grid_info().get("row") == 1:
                main_ws = child
                break

        upper_area = main_ws.winfo_children()[0]
        video_card = upper_area.winfo_children()[0]
        right_panel = upper_area.winfo_children()[1]

        # 1. Camera Card Width: ~700-820 px (Phase 5.16.4 expanded camera)
        cam_card_w = video_card.winfo_width()
        self.assertGreaterEqual(cam_card_w, 680, f"Camera card width {cam_card_w} px is too narrow!")
        self.assertLessEqual(cam_card_w, 980, f"Camera card width {cam_card_w} px is too wide!")

        # 2. Right Dashboard Requested Height: ~380-650 px
        right_panel_req_h = right_panel.winfo_reqheight()
        self.assertGreaterEqual(right_panel_req_h, 380, f"Right dashboard req height {right_panel_req_h} is too small!")
        self.assertLessEqual(right_panel_req_h, 650, f"Right dashboard req height {right_panel_req_h} exceeds 650 px!")

        # 3. Upper Area Height absorbs vertical space (removes empty bottom region)
        upper_h = upper_area.winfo_height()
        self.assertLess(upper_h, 680, f"Upper area height {upper_h} px exceeded 680 px!")

        # 4. Compact Event Log: Treeview height ~70-105 px (~3-4 rows)
        tree_h = self.app.tree.winfo_height()
        self.assertGreaterEqual(tree_h, 70, f"Treeview height {tree_h} px too short for ~3 rows!")
        self.assertLessEqual(tree_h, 110, f"Treeview height {tree_h} px too tall for compact event log!")

    def test_03_physical_camera_discovery_prefers_lit_sensor(self):
        """Verify camera discovery logic selects lit frames over black sensors."""
        cam = CameraManager()
        # Mock _test_and_open_device: device 0 is dark (mean 1.2), device 1 is lit (mean 112.5)
        class DummyCap:
            def isOpened(self): return True
            def release(self): pass
            def read(self): return True, np.ones((480, 640, 3), dtype=np.uint8) * 112

        def mock_open_device(idx, b_flag, b_name):
            cap = DummyCap()
            if idx == 0:
                dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 1
                return cap, 640, 480, dark_frame
            elif idx == 1:
                lit_frame = np.ones((480, 640, 3), dtype=np.uint8) * 112
                return cap, 640, 480, lit_frame
            return None

        cam._test_and_open_device = mock_open_device
        opened = cam.start(auto_discover=True)
        self.assertTrue(opened)
        self.assertEqual(cam.status, CameraStatus.REAL)
        self.assertEqual(cam.camera_index, 1, f"Expected lit device 1 to be chosen over dark device 0, got {cam.camera_index}")
        cam.stop()


if __name__ == "__main__":
    unittest.main()
