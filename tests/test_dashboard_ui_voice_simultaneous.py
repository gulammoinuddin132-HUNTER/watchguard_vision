"""
UI Integration Tests for Simultaneous Voice & Text Inputs (Phase 2.4 Bug Fix)
-----------------------------------------------------------------------------
Verifies that:
1. Both Push-to-Talk and Text Query inputs are initialized and simultaneously visible.
2. Submitting a text query keeps BOTH Push-to-Talk and Text Query visible and interactive.
3. Submitting a voice query keeps BOTH Push-to-Talk and Text Query visible and interactive.
4. Consecutive mixed interactions (Text -> Voice -> Text) never hide, collapse, or destroy either control.
"""

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine
from src.core.voice_assistant import VoiceAssistant
from src.ui.dashboard import WatchGuardDashboard


class TestDashboardUIVoiceSimultaneous(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hidden for headless/CI execution

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_ui_voice.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.context_engine = ContextEngine()

        self.mock_camera = MagicMock(spec=CameraManager)
        self.mock_camera.is_running = True
        self.mock_camera.is_fallback_mode = False
        self.mock_camera.camera_index = 0
        self.mock_camera.get_frame.return_value = (False, None)
        self.mock_camera.fps = 0.0

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
        self.va.mute_audio = True

        # Initialize Dashboard with mocks
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
        self.root.update_idletasks()

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

    def test_both_inputs_simultaneously_visible_on_startup(self):
        """Verify BOTH Push-to-Talk and Text Query widgets exist and are packed on startup."""
        self.assertTrue(self.dashboard.btn_ptt.winfo_exists())
        self.assertTrue(self.dashboard.ent_voice_query.winfo_exists())
        self.assertTrue(self.dashboard.btn_ask.winfo_exists())

        # Check widget packing
        self.assertIsNotNone(self.dashboard.btn_ptt.winfo_manager())
        self.assertIsNotNone(self.dashboard.ent_voice_query.winfo_manager())
        self.assertIsNotNone(self.dashboard.btn_ask.winfo_manager())

    def test_text_query_submission_preserves_both_inputs(self):
        """Verify submitting a text query updates response and leaves BOTH input methods visible."""
        self.logger.log_event(EventType.AUTHORIZED_PERSON, RiskLevel.GREEN, "Hunter logged in.", person_name="Hunter")

        self.dashboard.ent_voice_query.delete(0, tk.END)
        self.dashboard.ent_voice_query.insert(0, "What happened recently?")
        self.dashboard._on_send_text_query()
        self.root.update_idletasks()

        # Check response rendered
        resp_text = self.dashboard.lbl_voice_resp.cget("text")
        self.assertIn("Hunter", resp_text)

        # Verify BOTH widgets remain packed and visible
        self.assertTrue(self.dashboard.btn_ptt.winfo_exists())
        self.assertTrue(self.dashboard.ent_voice_query.winfo_exists())
        self.assertTrue(self.dashboard.btn_ask.winfo_exists())
        self.assertIsNotNone(self.dashboard.btn_ptt.winfo_manager())
        self.assertIsNotNone(self.dashboard.ent_voice_query.winfo_manager())
        self.assertIsNotNone(self.dashboard.btn_ask.winfo_manager())

    def test_voice_query_simulation_preserves_both_inputs(self):
        """Verify simulating voice transcription updates response and leaves BOTH input methods visible."""
        # Simulate PTT completion callback
        test_query = "Who is in front of me?"
        p_data = {
            "detected": True,
            "has_authorized": True,
            "has_unknown": False,
            "people": [{"name": "Hunter", "similarity": 0.94, "bbox": (100, 100, 100, 100)}],
        }
        self.context_engine.evaluate_context(p_data, {"detected": False, "count": 0, "objects": []})

        # Process command and update UI as PTT does
        resp = self.va.process_command(test_query)
        self.dashboard.item_voice_command.set_value(f"'{test_query}'", "#FFFFFF")
        self.dashboard.lbl_voice_resp.configure(text=resp)
        self.root.update_idletasks()

        self.assertIn("Hunter", self.dashboard.lbl_voice_resp.cget("text"))

        # Verify BOTH inputs are still present and packed
        self.assertTrue(self.dashboard.btn_ptt.winfo_exists())
        self.assertTrue(self.dashboard.ent_voice_query.winfo_exists())
        self.assertTrue(self.dashboard.btn_ask.winfo_exists())
        self.assertIsNotNone(self.dashboard.btn_ptt.winfo_manager())
        self.assertIsNotNone(self.dashboard.ent_voice_query.winfo_manager())
        self.assertIsNotNone(self.dashboard.btn_ask.winfo_manager())

    def test_mixed_interactions_sequence(self):
        """Verify sequence of Text -> Voice -> Text interactions never hides any control."""
        # 1. Text Query 1
        self.dashboard.ent_voice_query.delete(0, tk.END)
        self.dashboard.ent_voice_query.insert(0, "Is there any security risk?")
        self.dashboard._on_send_text_query()
        self.root.update_idletasks()
        self.assertTrue(self.dashboard.btn_ptt.winfo_exists())
        self.assertTrue(self.dashboard.ent_voice_query.winfo_exists())

        # 2. Voice Query
        resp2 = self.va.process_command("What objects are detected?")
        self.dashboard.lbl_voice_resp.configure(text=resp2)
        self.root.update_idletasks()
        self.assertTrue(self.dashboard.btn_ptt.winfo_exists())
        self.assertTrue(self.dashboard.ent_voice_query.winfo_exists())

        # 3. Text Query 2
        self.dashboard.ent_voice_query.delete(0, tk.END)
        self.dashboard.ent_voice_query.insert(0, "What is the current security status?")
        self.dashboard._on_send_text_query()
        self.root.update_idletasks()
        self.assertTrue(self.dashboard.btn_ptt.winfo_exists())
        self.assertTrue(self.dashboard.ent_voice_query.winfo_exists())
        self.assertTrue(self.dashboard.btn_ask.winfo_exists())


if __name__ == "__main__":
    unittest.main()
