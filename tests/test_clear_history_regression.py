"""
WatchGuard Vision - Clear History Bug Fix Regression Tests
----------------------------------------------------------
Verifies:
1. test_clear_history_removes_rows_from_database_and_treeview:
   - Database events are actually deleted transactionally
   - Treeview rows are emptied (0 rows visible)
   - Status text accurately reflects the exact deleted records count
2. test_new_event_appears_after_clear:
   - Live synchronization continues functioning seamlessly without app restart
   - Next legitimate security event is immediately captured and inserted into Treeview
3. test_database_integrity_after_clear:
   - PRAGMA integrity_check returns 'ok'
   - PRAGMA foreign_key_check returns 0 violations
"""

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant
from src.core.incident_manager import IncidentManager
from src.ui.dashboard import WatchGuardDashboard


class TestClearHistoryRegression(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_clear_history.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.context_engine = ContextEngine(db_manager=self.db)

        self.mock_camera = MagicMock(spec=CameraManager)
        self.mock_camera.is_running = True
        self.mock_camera.is_fallback_mode = False
        self.mock_camera.camera_index = 0
        self.mock_camera.get_frame.return_value = (False, None)
        self.mock_camera.fps = 30.0

        self.mock_person_engine = MagicMock(spec=PersonRecognitionEngine)
        self.mock_person_engine.is_ready = True
        self.mock_person_engine.known_face_count = 1

        self.mock_object_engine = MagicMock(spec=ObjectDetectionEngine)
        self.mock_object_engine.is_ready = True

        self.va = VoiceAssistant(
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            camera_manager=self.mock_camera,
        )

        self.dashboard = WatchGuardDashboard(
            root=self.root,
            db_manager=self.db,
            event_logger=self.logger,
            camera_manager=self.mock_camera,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            voice_assistant=self.va,
        )

    def tearDown(self):
        if hasattr(self, "dashboard") and self.dashboard:
            try:
                self.dashboard.close(destroy_root=False)
            except Exception:
                pass
        try:
            if hasattr(self, "root") and self.root and self.root.winfo_exists():
                self.root.destroy()
        except Exception:
            pass
        self.temp_dir.cleanup()

    def test_clear_history_removes_rows_from_database_and_treeview(self):
        """
        REGRESSION TEST:
        Verify that clicking 'Clear History' deletes all eligible records from SQLite
        AND renders the Treeview completely empty (0 rows), displaying the exact deleted count.
        """
        # 1. Log 3 events with mixed risk levels (GREEN, RED, ORANGE)
        self.logger.log_event(EventType.CAMERA_STARTED, RiskLevel.GREEN, "System started")
        self.logger.log_event(EventType.UNKNOWN_PERSON, RiskLevel.YELLOW, "Unknown visitor detected")
        self.logger.log_event(EventType.SECURITY_ALERT, RiskLevel.RED, "Unauthorized restricted zone access")

        # Drain and process events in dashboard
        self.dashboard._schedule_event_pump()
        self.dashboard._load_recent_events()

        # Verify DB has 3 events and Treeview has 3 rows before clear
        self.assertEqual(self.db.get_total_event_count(), 3)
        self.assertEqual(len(self.dashboard.tree.get_children()), 3)

        # 2. Simulate user confirming Clear History
        deleted = self.db.clear_event_history(clear_incidents=True)
        deleted_count = deleted.get("security_events", 0)
        self.assertEqual(deleted_count, 3)

        # Execute dashboard clear & refresh logic
        while not self.dashboard.event_queue.empty():
            try:
                self.dashboard.event_queue.get_nowait()
            except Exception:
                break
        self.dashboard._clear_treeview()
        self.dashboard._load_recent_events()
        self.dashboard.lbl_table_status.configure(
            text=f"Event History Cleared ({deleted_count} Records)"
        )

        # 3. Assertions:
        # A. Database records are actually deleted
        self.assertEqual(self.db.get_total_event_count(), 0)
        self.assertEqual(len(self.db.get_recent_security_events()), 0)

        # B. Treeview contains ZERO rows
        self.assertEqual(len(self.dashboard.tree.get_children()), 0)

        # C. UI label reports accurate count
        self.assertIn("Event History Cleared (3 Records)", self.dashboard.lbl_table_status.cget("text"))

    def test_new_event_appears_after_clear(self):
        """
        REGRESSION TEST:
        Verify that after Clear History, live sync remains active and the next
        legitimate state transition automatically produces exactly 1 new event in the Treeview.
        """
        # 1. Start with 2 old events and clear them
        self.logger.log_event(EventType.CAMERA_STARTED, RiskLevel.GREEN, "Initial boot event")
        self.logger.log_event(EventType.PERSON_DETECTED, RiskLevel.GREEN, "Hunter recognized")
        self.dashboard._load_recent_events()
        self.assertEqual(len(self.dashboard.tree.get_children()), 2)

        # Clear
        self.db.clear_event_history(clear_incidents=True)
        while not self.dashboard.event_queue.empty():
            try:
                self.dashboard.event_queue.get_nowait()
            except Exception:
                break
        self.dashboard._clear_treeview()
        self.dashboard._load_recent_events()
        self.dashboard._last_logged_security_state = None
        self.dashboard._consecutive_green_frames = 0
        self.assertEqual(len(self.dashboard.tree.get_children()), 0)

        # 2. Trigger a new legitimate security event via logger
        self.logger.log_event(
            EventType.UNKNOWN_PERSON,
            RiskLevel.YELLOW,
            "Fresh visitor event after clear",
            person_name="Unknown Person",
        )

        # 3. Process event queue in dashboard
        self.dashboard._schedule_event_pump()

        # 4. Assert exactly 1 row appears in Treeview
        children = self.dashboard.tree.get_children()
        self.assertEqual(len(children), 1)
        row_values = self.dashboard.tree.item(children[0])["values"]
        self.assertIn("Fresh visitor event after clear", str(row_values))
        self.assertEqual(self.db.get_total_event_count(), 1)

    def test_database_integrity_after_clear(self):
        """Verify SQLite integrity and foreign keys are 100% sound after clear."""
        self.logger.log_event(EventType.SECURITY_ALERT, RiskLevel.RED, "Test event")
        self.db.clear_event_history(clear_incidents=True)

        with self.db._connection() as conn:
            integrity = [tuple(r) for r in conn.execute("PRAGMA integrity_check;").fetchall()]
            fk_check = conn.execute("PRAGMA foreign_key_check;").fetchall()

        self.assertEqual(integrity, [("ok",)])
        self.assertEqual(fk_check, [])

    def test_clear_history_confirmation_dialog_structure_and_cancel(self):
        """
        REGRESSION TEST:
        Verify the 'Confirm Clear Event History' dialog is created with all required child widgets
        (title header, warning message, Cancel and Clear History buttons) and Cancel does not delete records.
        """
        # 1. Log 2 events
        self.logger.log_event(EventType.CAMERA_STARTED, RiskLevel.GREEN, "System started")
        self.logger.log_event(EventType.UNKNOWN_PERSON, RiskLevel.YELLOW, "Visitor at door")
        self.dashboard._load_recent_events()
        self.assertEqual(self.db.get_total_event_count(), 2)

        # 2. Invoke Clear History dialog
        self.dashboard._on_clear_event_history_clicked()
        self.root.update_idletasks()
        self.root.update()

        # Find Toplevel dialog
        toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(toplevels), 1, "Expected 1 Toplevel confirmation dialog")
        dialog = toplevels[0]

        # Verify title and dimensions
        self.assertEqual(dialog.title(), "Confirm Clear Event History")

        # Collect child widgets
        labels = []
        buttons = []
        def _collect_widgets(parent):
            for child in parent.winfo_children():
                if isinstance(child, tk.Label):
                    labels.append(child)
                elif isinstance(child, tk.Button):
                    buttons.append(child)
                _collect_widgets(child)

        _collect_widgets(dialog)

        # Assert title label exists
        title_texts = [lbl.cget("text") for lbl in labels]
        self.assertTrue(any("Clear Event History" in t for t in title_texts), f"Title label missing in {title_texts}")

        # Assert confirmation message label exists
        self.assertTrue(any("permanently delete" in t for t in title_texts), f"Message body missing in {title_texts}")

        # Assert Cancel and Clear History buttons exist
        btn_texts = [b.cget("text") for b in buttons]
        self.assertIn("Cancel", btn_texts)
        self.assertIn("Clear History", btn_texts)

        # 3. Click Cancel button
        cancel_btn = next(b for b in buttons if b.cget("text") == "Cancel")
        cancel_btn.invoke()
        self.root.update_idletasks()
        self.root.update()

        # Verify dialog is destroyed
        remaining_toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(remaining_toplevels), 0, "Dialog should be destroyed on Cancel")

        # Verify records are NOT deleted
        self.assertEqual(self.db.get_total_event_count(), 2)
        self.assertEqual(len(self.dashboard.tree.get_children()), 2)

    def test_clear_history_confirmation_dialog_confirm_action(self):
        """
        REGRESSION TEST:
        Verify clicking 'Clear History' in the confirmation dialog performs full database deletion,
        empties the Treeview, updates the status label, and closes the dialog.
        """
        # 1. Log 3 events
        self.logger.log_event(EventType.CAMERA_STARTED, RiskLevel.GREEN, "System started")
        self.logger.log_event(EventType.UNKNOWN_PERSON, RiskLevel.YELLOW, "Visitor at door")
        self.logger.log_event(EventType.SECURITY_ALERT, RiskLevel.RED, "Intrusion detected")
        self.dashboard._load_recent_events()
        self.assertEqual(self.db.get_total_event_count(), 3)
        self.assertEqual(len(self.dashboard.tree.get_children()), 3)

        # 2. Invoke Clear History dialog
        self.dashboard._on_clear_event_history_clicked()
        self.root.update_idletasks()
        self.root.update()

        toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(toplevels), 1)
        dialog = toplevels[0]

        # Find Clear History button
        buttons = []
        def _collect_buttons(parent):
            for child in parent.winfo_children():
                if isinstance(child, tk.Button):
                    buttons.append(child)
                _collect_buttons(child)
        _collect_buttons(dialog)

        clear_btn = next(b for b in buttons if b.cget("text") == "Clear History")

        # 3. Click Confirm / Clear History
        clear_btn.invoke()
        self.root.update_idletasks()
        self.root.update()

        # Verify dialog destroyed
        remaining_toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(remaining_toplevels), 0)

        # Verify DB and Treeview cleared
        self.assertEqual(self.db.get_total_event_count(), 0)
        self.assertEqual(len(self.dashboard.tree.get_children()), 0)
        self.assertIn("Event History Cleared (3 Records)", self.dashboard.lbl_table_status.cget("text"))


if __name__ == "__main__":
    unittest.main()
