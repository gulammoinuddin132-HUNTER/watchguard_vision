"""
WatchGuard Vision - Main Application Entry Point
------------------------------------------------
Initializes SQLite database, Event Logging subsystem, Person Recognition Engine,
Object Detection Engine, Context Intelligence Engine, Camera Manager, and launches
the WatchGuard Vision Desktop Dashboard.
"""

import os
import sys
import tkinter as tk
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine
from src.core.voice_assistant import VoiceAssistant
from src.ui.dashboard import WatchGuardDashboard


def main() -> None:
    """Bootstraps all services and launches UI."""
    print("==================================================")
    print("  WATCHGUARD VISION - AI VISUAL SECURITY ASSISTANT")
    print("==================================================")

    # 1. Initialize SQLite Database
    print("[1/7] Initializing SQLite database...")
    db_manager = DatabaseManager()

    # 2. Initialize Event Logger Framework
    print("[2/7] Initializing Event Logging framework...")
    event_logger = EventLogger(db_manager=db_manager)
    event_logger.log_event(
        event_type=EventType.SYSTEM_STARTUP,
        risk_level=RiskLevel.GREEN,
        description="WatchGuard Vision core services started.",
    )

    # 3. Initialize Person Recognition Engine (Phase 2.1)
    print("[3/7] Initializing OpenCV Person Recognition Engine...")
    person_engine = PersonRecognitionEngine(db_manager=db_manager)
    person_engine.initialize()
    print(f"      -> {person_engine.known_face_count} face profile(s) loaded.")

    # 4. Initialize Object Detection Engine (Phase 2.2)
    print("[4/7] Initializing YOLOv8 Object Detection Engine...")
    object_engine = ObjectDetectionEngine(db_manager=db_manager)
    object_engine.initialize()
    print(f"      -> {len(object_engine.class_names)} COCO object classes supported.")

    # 5. Initialize Context & Security Intelligence Engine (Phase 2.3)
    print("[5/7] Initializing Context & Security Intelligence Engine...")
    context_engine = ContextEngine()
    print(f"      -> {len(context_engine.protected_objects)} protected asset classes & {len(context_engine.zones_config)} zones configured.")

    # 6. Initialize Voice Assistant Engine (Phase 2.4)
    print("[6/8] Initializing Voice Assistant Engine...")
    voice_assistant = VoiceAssistant(
        db_manager=db_manager,
        event_logger=event_logger,
        person_engine=person_engine,
        object_engine=object_engine,
        context_engine=context_engine,
        camera_manager=None,  # Will be attached after camera init
    )

    # 7. Initialize Camera Manager
    print("[7/8] Initializing Camera Manager...")
    camera_manager = CameraManager()
    voice_assistant.camera = camera_manager

    # 8. Launch Desktop Dashboard UI
    print("[8/8] Launching WatchGuard Vision Desktop Dashboard...")
    root = tk.Tk()
    app = WatchGuardDashboard(
        root=root,
        camera_manager=camera_manager,
        db_manager=db_manager,
        event_logger=event_logger,
        person_engine=person_engine,
        object_engine=object_engine,
        context_engine=context_engine,
        voice_assistant=voice_assistant,
    )

    print("\nWatchGuard Vision is running. Close window to exit.\n")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        if camera_manager.is_running:
            camera_manager.stop()
        event_logger.log_event(
            event_type=EventType.SYSTEM_SHUTDOWN,
            risk_level=RiskLevel.GREEN,
            description="WatchGuard Vision stopped gracefully.",
        )
        print("WatchGuard Vision shutdown complete.")


if __name__ == "__main__":
    main()
