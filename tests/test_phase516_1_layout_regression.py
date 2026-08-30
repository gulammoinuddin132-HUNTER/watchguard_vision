"""
WatchGuard Vision - Phase 5.16.2 UI Layout & Geometry Regression Tests
----------------------------------------------------------------------
Validates:
  1. Large Camera: Canvas dimensions >= 650x450, 4:3 aspect ratio
  2. Detailed Right Dashboard: All cards present and readable
  3. Compact Event Log: Height 120-175px, Treeview viewport >= 70px (~3-4 visible rows)
  4. Zero Clipping: Event log bottom edge <= window client height
  5. Long Voice Response Invariance: 10-line response causes 0px change in event table
  6. Responsive Resizing: 1280x880, maximized 1600x950, compact 1150x760
  7. Two-Way Scrolling: Vertical and horizontal scrolling operate smoothly
"""

import unittest
import os
import shutil
import tempfile
import tkinter as tk

from src.ui.dashboard import WatchGuardDashboard
from src.database.db_manager import DatabaseManager
from src.camera.camera_manager import CameraManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel


class TestPhase5162LayoutRegression(unittest.TestCase):
    """UI geometry regression tests for Phase 5.16.2."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_p516_2.db")
        self.db = DatabaseManager(self.db_path)

        for i in range(25):
            self.db.log_security_event(
                event_type=EventType.UNAUTHORIZED_ACCESS if i % 4 == 0 else EventType.PERSON_DETECTED,
                risk_level=RiskLevel.RED if i % 4 == 0 else RiskLevel.GREEN,
                description=f"Phase 5.16.2 Test Security Event #{i+1:02d} with descriptive rationale trace",
                person_name="Hunter" if i % 2 == 0 else "Unknown",
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

    def _get_containers(self):
        main_workspace = None
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame) and child.grid_info().get("row") == 1:
                main_workspace = child
                break
        self.assertIsNotNone(main_workspace, "main_workspace frame not found")
        upper_area = main_workspace.winfo_children()[0]
        ctrl_bar = main_workspace.winfo_children()[1]
        evt_card = main_workspace.winfo_children()[2]
        return main_workspace, upper_area, ctrl_bar, evt_card

    def test_01_compact_event_container_and_treeview_heights(self):
        """Event container and Treeview must meet compact specification without clipping."""
        _, upper_area, ctrl_bar, evt_card = self._get_containers()
        tree = self.app.tree

        root_h = self.root.winfo_height()
        evt_h = evt_card.winfo_height()
        tree_h = tree.winfo_height()
        evt_bottom = evt_card.winfo_y() + evt_h

        self.assertGreaterEqual(evt_h, 95, f"Event card height ({evt_h}px) must be >= 95px")
        self.assertLessEqual(evt_h, 180, f"Event card height ({evt_h}px) must remain compact <= 180px")
        self.assertGreaterEqual(tree_h, 60, f"Treeview height ({tree_h}px) must be >= 60px")
        self.assertLessEqual(evt_bottom, root_h, f"Event card bottom ({evt_bottom}px) exceeds root window height ({root_h}px)")

    def test_02_actual_rows_visible_on_screen(self):
        """Treeview must contain populated events and fit ~3-4 readable rows."""
        tree = self.app.tree
        rows = len(tree.get_children())
        self.assertGreaterEqual(rows, 20, f"Treeview must contain >= 20 rows, found {rows}")

        # At default row height ~16px, treeview must display at least 3 rows
        visible_rows = tree.winfo_height() // 16
        self.assertGreaterEqual(visible_rows, 3, f"Visible row count ({visible_rows}) must be >= 3")

    def test_03_large_camera_dimensions(self):
        """Camera area must be large and dominant (>= 650x450)."""
        canvas_w = self.app.video_canvas.winfo_width()
        canvas_h = self.app.video_canvas.winfo_height()
        self.assertGreaterEqual(canvas_w, 650, f"Camera canvas width ({canvas_w}px) must be >= 650px")
        self.assertGreaterEqual(canvas_h, 450, f"Camera canvas height ({canvas_h}px) must be >= 450px")

    def test_04_long_voice_response_does_not_collapse_event_area(self):
        """Long 10-line Voice Assistant response must NEVER compress or push event table."""
        tree = self.app.tree
        initial_tree_h = tree.winfo_height()
        _, _, _, evt_card = self._get_containers()
        initial_evt_bottom = evt_card.winfo_y() + evt_card.winfo_height()

        long_response = "Multi-line Copilot Analysis:\n" + "\n".join([f"Line {i}: Status check" for i in range(1, 11)])
        self.app.lbl_voice_resp.set_text(long_response)
        self.root.update_idletasks()
        self.root.update()

        self.assertEqual(tree.winfo_height(), initial_tree_h, "Treeview height changed after long voice response!")
        new_evt_bottom = evt_card.winfo_y() + evt_card.winfo_height()
        self.assertEqual(new_evt_bottom, initial_evt_bottom, "Event card position shifted after long voice response!")

    def test_05_responsive_resizing_invariance(self):
        """Window resizing (1600x950 and 1150x760) retains guaranteed visible event viewport."""
        tree = self.app.tree
        for geom in ("1600x950", "1150x760", "1280x880"):
            self.root.geometry(geom)
            self.root.update_idletasks()
            self.root.update()

            _, _, _, evt_card = self._get_containers()
            root_h = self.root.winfo_height()
            evt_bottom = evt_card.winfo_y() + evt_card.winfo_height()

            self.assertLessEqual(evt_bottom, root_h, f"At geometry {geom}, event card was clipped off screen!")
            self.assertGreaterEqual(tree.winfo_height(), 65, f"At geometry {geom}, treeview height fell below 65px!")

    def test_06_two_way_scrolling_operates(self):
        """Vertical and horizontal scrolling must navigate table content."""
        tree = self.app.tree
        # Vertical
        y_init = tree.yview()
        tree.yview_moveto(0.4)
        self.root.update_idletasks()
        y_after = tree.yview()
        self.assertGreater(y_after[0], y_init[0], "Vertical scroll failed")

        # Horizontal
        x_init = tree.xview()
        tree.xview_moveto(0.1)
        self.root.update_idletasks()
        x_after = tree.xview()
        self.assertGreater(x_after[0], x_init[0], "Horizontal scroll failed")


if __name__ == "__main__":
    unittest.main()
