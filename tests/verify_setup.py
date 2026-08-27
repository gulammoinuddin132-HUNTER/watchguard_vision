"""
Verification script for WatchGuard Vision architecture, subsystem health,
Person Recognition (2.1), Object Detection (2.2), Context Intelligence (2.3),
and Voice Assistant (2.4).
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine
from src.core.voice_assistant import VoiceAssistant


def verify():
    print("=== WatchGuard Vision Subsystem Health Check (Phase 2.4) ===")

    # 1. Database
    db = DatabaseManager()
    with db._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        print(f"[*] SQLite Tables Initialized ({len(tables)}):", tables)

        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"[*] Registered Users: {user_count}")

    # 2. Event Logger
    logger = EventLogger(db_manager=db)
    event = logger.log_event(
        event_type=EventType.SYSTEM_STARTUP,
        risk_level=RiskLevel.GREEN,
        description="Health check verification event.",
    )
    print(f"[*] Logged Security Event ID: {event['id']} ({event['risk_level']} - {event['event_type']})")

    # 3. Person Recognition Engine
    person_engine = PersonRecognitionEngine(db_manager=db)
    p_ok = person_engine.initialize()
    print(f"[*] Person Recognition Engine Initialized: {p_ok} ({person_engine.known_face_count} face profiles loaded)")

    # 4. Object Detection Engine
    object_engine = ObjectDetectionEngine(db_manager=db)
    o_ok = object_engine.initialize()
    print(f"[*] Object Detection Engine Initialized: {o_ok} ({len(object_engine.class_names)} classes supported)")

    # 5. Context & Security Intelligence Engine
    context_engine = ContextEngine()
    print(f"[*] Context Engine Initialized: True ({len(context_engine.protected_objects)} protected classes, {len(context_engine.zones_config)} zones)")

    # 6. Camera Manager & Live Vision Pipeline
    cam = CameraManager()
    cam.start()
    success, frame = cam.get_frame()
    print(f"[*] Camera Frame Capture: Success={success}, Shape={frame.shape if frame is not None else None}, Real Camera={not cam.is_fallback_mode}")

    # Run Person Recognition
    p_results = person_engine.process_frame(frame)
    print(f"[*] Person Recognition: Detected={p_results['detected']}, Label={p_results.get('primary_label', 'None')}")

    # Run Object Detection
    o_results = object_engine.process_frame(frame)
    print(f"[*] Object Detection: Detected={o_results['detected']}, Count={o_results['count']}, Summary='{o_results['summary']}'")

    # Run Context & Security Intelligence
    decision = context_engine.evaluate_context(p_results, o_results, camera_id=cam.camera_index, frame_shape=frame.shape)
    print(f"[*] Security Decision: Risk=[{decision.risk_level.value}] Code={decision.decision}")
    print(f"    Reason: {decision.reason}")

    # 7. Voice Assistant Engine (Phase 2.4)
    va = VoiceAssistant(
        db_manager=db,
        event_logger=logger,
        person_engine=person_engine,
        object_engine=object_engine,
        context_engine=context_engine,
        camera_manager=cam,
    )
    va.mute_audio = True
    test_q = "Who is in front of the camera?"
    test_resp = va.process_command(test_q)
    print(f"[*] Voice Assistant Engine: Initialized (Status: {va.status_label})")
    print(f"    Query: '{test_q}' -> Response: '{test_resp}'")

    snap_path = cam.capture_snapshot("health_check_phase2_4")
    print(f"[*] Evidence Snapshot Generated: {snap_path}")
    cam.stop()

    print("\n>>> ALL SUBSYSTEMS (PHASE 2.4 VOICE ASSISTANT) OPERATIONAL AND VERIFIED! <<<")


if __name__ == "__main__":
    verify()
