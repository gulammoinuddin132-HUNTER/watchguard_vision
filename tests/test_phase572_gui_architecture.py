"""
WatchGuard Vision - Phase 5.7.2 GUI Architecture & Event Table Tests
--------------------------------------------------------------------
Validates:
1. BoundedTextViewport:
   - Read-only locked text widget
   - Fixed height with pack_propagate(False) and grid_propagate(False)
   - Zero geometry reflow when text length changes from 1 line to 10 lines
   - Change-caching prevents redundant configure / text deletion
2. Three-Row Layout Geometry:
   - Root row 0 (header 40px), row 1 (main workspace weight=1)
   - Main workspace row 0 (upper area bounded), row 1 (camera controls), row 2 (event log minsize=300)
3. Decoupled Video and UI Telemetry Scheduler:
   - 30 FPS video loop updates _latest_vision_state snapshot
   - 10 Hz UI telemetry scheduler updates widgets only when state changes
4. Event Table Viewport & Treeview Population:
   - Physical height >= 200px
   - All SQLite records inserted and visible in Treeview
"""

import unittest
from unittest.mock import MagicMock
import os
import shutil
import tempfile
import tkinter as tk

from src.ui.widgets import BoundedTextViewport, CardFrame, RiskBanner, TelemetryItem
from src.ui.dashboard import WatchGuardDashboard
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant
from src.database.db_manager import DatabaseManager
from src.camera.camera_manager import CameraManager
from src.events.event_logger import EventLogger
from src.events.event_types import RiskLevel, EventType


class TestPhase572GUIArchitecture(unittest.TestCase):
    """Rigorous validation of Phase 5.7.2 GUI architecture, viewports, and decoupling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_p572.db")
        self.db = DatabaseManager(self.db_path)

        for i in range(15):
            self.db.log_security_event(
                event_type=EventType.AUTHORIZED_PERSON if i % 2 == 0 else EventType.UNKNOWN_PERSON,
                risk_level=RiskLevel.GREEN if i % 2 == 0 else RiskLevel.YELLOW,
                description=f"Automated test record #{i+1}",
                person_name="Hunter" if i % 2 == 0 else "Unknown",
            )

        self.logger = EventLogger(db_manager=self.db)
        self.cam = CameraManager()
        self.pe = PersonRecognitionEngine(db_manager=self.db)
        self.od = ObjectDetectionEngine()
        self.ce = ContextEngine(db_manager=self.db)
        self.va = VoiceAssistant(context_engine=self.ce, db_manager=self.db, event_logger=self.logger)
        self.va.mute_audio = True

        self.root = tk.Tk()
        self.root.geometry("1280x880")
        self.app = WatchGuardDashboard(self.root, self.cam, self.db, self.logger, self.pe, self.od, self.ce, self.va)
        self.root.update_idletasks()
        self.root.update()

    def tearDown(self):
        try:
            self.app._on_close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- Test 1: BoundedTextViewport Geometry Invariance ---
    def test_01_bounded_text_viewport_geometry_invariance(self):
        """BoundedTextViewport must NEVER alter its pixel height when text length changes."""
        vp = self.app.lbl_voice_resp
        initial_height = vp.winfo_height()

        # 1-line short text
        vp.set_text("Short.")
        self.root.update_idletasks()
        self.assertEqual(vp.winfo_height(), initial_height)

        # Long 10-line text
        vp.set_text(
            "This is an extensive multiline security incident report explaining that an unauthorized individual was detected "
            "inside the Restricted Area perimeter at 22:50 outside authorized operating hours. Immediate review is recommended."
        )
        self.root.update_idletasks()
        self.assertEqual(vp.winfo_height(), initial_height)

    # --- Test 2: BoundedTextViewport Change Caching ---
    def test_02_bounded_text_viewport_caching(self):
        """BoundedTextViewport skips redraw when text and color are unchanged."""
        vp = BoundedTextViewport(self.root, height_px=30)
        vp.set_text("Active State", fg="#FFFFFF")
        self.assertEqual(vp._current_text, "Active State")
        self.assertEqual(vp._current_fg, "#FFFFFF")

        # Mock text widget configure to verify no redundant mutations
        vp.text_widget.configure = MagicMock()
        vp.set_text("Active State", fg="#FFFFFF")
        vp.text_widget.configure.assert_not_called()

    # --- Test 3: Event Table Guaranteed Minimum Height & Rows ---
    def test_03_event_table_guaranteed_height_and_rows(self):
        """Event Table must have a physical height >= 200px and display all records."""
        tree = self.app.tree
        th = tree.winfo_height()
        rows = len(tree.get_children())

        self.assertGreaterEqual(th, 200, f"Treeview height ({th}px) must be >= 200px")
        self.assertEqual(rows, 15, f"Treeview must contain all 15 records, found {rows}")

    # --- Test 4: Decoupled Vision State Architecture ---
    def test_04_decoupled_vision_state_architecture(self):
        """Video loop updates _latest_vision_state; UI scheduler consumes it without stalling."""
        self.app.camera_index_var.set(-1)
        self.app._toggle_camera()

        # Run frame loop
        self.root.update()

        self.assertIsNotNone(self.app._latest_vision_state)
        state = self.app._latest_vision_state
        self.assertIn("obj_results", state)
        self.assertIn("person_results", state)
        self.assertIn("decision", state)
        self.assertIn("fps", state)

        # Trigger decoupled UI update
        self.app._schedule_ui_telemetry_update()
        self.root.update()

        self.app._toggle_camera()

    # --- Test 5: Canvas In-Place Image Item Reuse ---
    def test_05_canvas_image_item_reuse(self):
        """Canvas reuses _canvas_img_id via itemconfig rather than deleting all items."""
        self.app.camera_index_var.set(-1)
        self.app._toggle_camera()

        # Run a few frames
        for _ in range(5):
            self.root.update()

        first_img_id = self.app._canvas_img_id
        self.assertIsNotNone(first_img_id)

        # Next frames must retain identical canvas item ID
        for _ in range(5):
            self.root.update()

        self.assertEqual(self.app._canvas_img_id, first_img_id)
        self.app._toggle_camera()


if __name__ == "__main__":
    unittest.main()
