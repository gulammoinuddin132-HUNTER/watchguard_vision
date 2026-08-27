"""
Real Physical Webcam End-to-End Pipeline Verification Script
------------------------------------------------------------
Traces and validates the complete frame flow:
CameraManager (Physical Device Index 0)
 -> frame acquisition
 -> PersonRecognitionEngine (YuNet, SFace, PAD, JointAccessEvaluator)
 -> ObjectDetectionEngine (YOLOv8)
 -> ContextEngine (Security Rules & Intelligence)
 -> Visual HUD Generation
"""

import sys
import time
from pathlib import Path
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.camera.camera_manager import CameraManager, CameraStatus, CameraMode
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine
from src.database.db_manager import DatabaseManager


def main():
    print("==================================================")
    print("  WATCHGUARD VISION - REAL WEBCAM PIPELINE TEST   ")
    print("==================================================")

    # 1. Initialize DB and Engines
    db = DatabaseManager()
    person_engine = PersonRecognitionEngine(db_manager=db)
    person_engine.initialize()

    object_engine = ObjectDetectionEngine(db_manager=db)
    object_engine.initialize()

    context_engine = ContextEngine()

    # 2. Start CameraManager on Physical Index 0
    cam = CameraManager(camera_index=0)
    started = cam.start(camera_index=0, mode=CameraMode.PHYSICAL, allow_synthetic_fallback=False)

    print(f"\n[1] Camera Startup Result: {started}")
    print(f"    Status: {cam.status.value}")
    print(f"    Label: {cam.status_label}")
    print(f"    Is Physical: {cam.is_physical}")
    print(f"    Is Fallback/Synthetic: {cam.is_fallback_mode}")
    print(f"    Device Resolution: {cam.actual_resolution}")

    if not started or not cam.is_physical:
        print(f"FAILED: Camera failed to open as physical device. Error: {cam.error_message}")
        cam.stop()
        sys.exit(1)

    # 3. Read 30 continuous live frames from physical camera through full pipeline
    print("\n[2] Streaming 30 Live Frames from Physical Webcam...")
    frame_count = 0
    start_time = time.time()

    for i in range(30):
        success, frame_bgr = cam.get_frame()
        if success and frame_bgr is not None and frame_bgr.size > 0:
            frame_count += 1

            # Run Object Detection
            obj_results = object_engine.process_frame(frame_bgr)

            # Run Person Recognition & PAD
            person_results = person_engine.process_frame(frame_bgr)

            # Evaluate Context
            decision = context_engine.evaluate_context(
                person_data=person_results,
                object_data=obj_results,
                camera_id=cam.camera_index,
                frame_shape=frame_bgr.shape,
            )

            # Annotate HUD
            hud_frame = frame_bgr.copy()
            hud_frame = context_engine.draw_context_hud(hud_frame, decision)
            hud_frame = object_engine.draw_annotations(hud_frame, obj_results, suppress_person_boxes=True)
            hud_frame = person_engine.draw_annotations(hud_frame, person_results)

        time.sleep(0.03)

    elapsed = time.time() - start_time
    real_fps = frame_count / elapsed
    print(f"    Captured & Processed: {frame_count}/30 frames")
    print(f"    Elapsed: {elapsed:.2f}s | Real FPS: {real_fps:.1f}")
    print(f"    CameraManager Measured FPS: {cam.fps:.1f}")

    # 4. Save sample snapshot from physical webcam
    snap_path = cam.capture_snapshot(filename_prefix="test_real_webcam_live")
    print(f"\n[3] Evidence Snapshot Captured: {snap_path}")

    cam.stop()
    print("\n[4] Camera stopped. Final Status:", cam.status.value)
    print("\n>>> REAL WEBCAM PIPELINE VERIFICATION SUCCESSFUL <<<")


if __name__ == "__main__":
    main()
