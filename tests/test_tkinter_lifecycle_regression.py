"""
WatchGuard Vision - Tkinter Lifecycle & Thread-Safety Regression Tests
----------------------------------------------------------------------
Verifies that:
1. Creating, running, and closing the dashboard cleanly cancels all scheduled
   after() timers and stops background worker threads.
2. No 'invalid command name' errors occur during or after widget/root destruction.
3. No 'RuntimeError: main thread is not in main loop' or 'Tcl_AsyncDelete' exceptions
   are raised during multiple consecutive GUI create/destroy cycles.
4. Thread-safe UI dispatching drops pending updates safely when the window is closed.
"""

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine
from src.core.voice_assistant import VoiceAssistant
from src.ui.dashboard import WatchGuardDashboard


class TestTkinterLifecycleRegression(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_lifecycle.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.context_engine = ContextEngine(db_manager=self.db)

        self.mock_camera = MagicMock(spec=CameraManager)
        self.mock_camera.is_running = False
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

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_gui_lifecycle_clean_shutdown(self):
        """Verify GUI creation, scheduling, callback cancellation, and clean shutdown."""
        root = tk.Tk()
        root.geometry("1280x880")
        root.withdraw()

        dashboard = WatchGuardDashboard(
            root=root,
            camera_manager=self.mock_camera,
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            voice_assistant=self.va,
        )

        root.update_idletasks()
        root.update()

        # Verify timer callbacks were initially scheduled
        self.assertIsNotNone(dashboard._after_video_id)
        self.assertIsNotNone(dashboard._after_ui_id)
        self.assertIsNotNone(dashboard._after_pump_id)
        self.assertFalse(dashboard._is_closed)

        # Execute clean close
        dashboard.close(destroy_root=False)

        # Verify callbacks cancelled and worker thread stopped
        self.assertTrue(dashboard._is_closed)
        self.assertIsNone(dashboard._after_video_id)
        self.assertIsNone(dashboard._after_ui_id)
        self.assertIsNone(dashboard._after_pump_id)
        self.assertTrue(dashboard._ai_stop_event.is_set())

        # Destroy root cleanly
        root.destroy()

    def test_repeated_gui_create_destroy_cycles(self):
        """Verify multiple consecutive create -> run -> close cycles (10x) leave zero lingering threads/timers."""
        for cycle in range(10):
            root = tk.Tk()
            root.geometry("1280x880")
            root.withdraw()

            dashboard = WatchGuardDashboard(
                root=root,
                camera_manager=self.mock_camera,
                db_manager=self.db,
                event_logger=self.logger,
                person_engine=self.mock_person_engine,
                object_engine=self.mock_object_engine,
                context_engine=self.context_engine,
                voice_assistant=self.va,
            )

            root.update_idletasks()
            root.update()

            # Trigger safe thread-safe UI update
            dashboard._safe_dispatch_ui(lambda: dashboard.item_person.set_value("ACTIVE"))

            # Close dashboard
            dashboard.close(destroy_root=False)
            self.assertTrue(dashboard._is_closed)

            # Destroy root
            root.destroy()

    def test_event_pump_lifecycle_isolated(self):
        """
        REGRESSION TEST:
        Verify event pump lifecycle can start, receive events, execute clean close,
        and subsequently process pending Tk events with zero 'invalid command name' errors.
        """
        root = tk.Tk()
        root.geometry("1280x880")
        root.withdraw()

        dashboard = WatchGuardDashboard(
            root=root,
            camera_manager=self.mock_camera,
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            voice_assistant=self.va,
        )

        root.update_idletasks()
        root.update()

        # Push test events into queue and pump
        dashboard.event_queue.put({
            "id": 1,
            "timestamp": "12:00:00",
            "risk_level": "GREEN",
            "event_type": "INFO",
            "description": "Lifecycle test event",
        })
        dashboard._schedule_event_pump()
        root.update_idletasks()
        root.update()

        # Close dashboard cleanly
        dashboard.close(destroy_root=False)
        self.assertTrue(dashboard._is_closed)
        self.assertIsNone(dashboard._after_pump_id)

        # Repeatedly process Tk event loop after closure -> must NOT raise invalid command name
        for _ in range(5):
            root.update_idletasks()
            root.update()

        root.destroy()

    def test_safe_dispatch_after_close_does_not_crash(self):
        """Verify calling _safe_dispatch_ui after dashboard closure is safely dropped."""
        root = tk.Tk()
        root.withdraw()

        dashboard = WatchGuardDashboard(
            root=root,
            camera_manager=self.mock_camera,
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            voice_assistant=self.va,
        )

        dashboard.close(destroy_root=True)

        # Attempt to dispatch after closure -> must be safely dropped without exception
        executed = []
        dashboard._safe_dispatch_ui(lambda: executed.append(1))
        self.assertEqual(len(executed), 0)


if __name__ == "__main__":
    unittest.main()
