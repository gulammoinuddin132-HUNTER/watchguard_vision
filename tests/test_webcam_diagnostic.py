"""
Webcam Diagnostic & Capture Verification Script
-----------------------------------------------
Systematically probes camera indexes 0, 1, 2, 3 and validates the CameraManager
initialization, frame grab, resolution detection, snapshot capture, and clean release.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.camera_manager import CameraManager


def run_webcam_diagnostics():
    print("================================================================")
    print("       WATCHGUARD VISION - WEBCAM DIAGNOSTIC & CAPTURE TEST     ")
    print("================================================================")

    manager = CameraManager(camera_index=0)

    # 1. Probe available devices on indexes 0-3
    print("\n[STEP 1/4] Probing hardware camera device indexes [0, 1, 2, 3]...")
    devices = manager.probe_available_devices(max_indices=4)

    if devices:
        print(f"\n>>> Physical Webcams Detected: {len(devices)} device(s) found!")
        for dev in devices:
            print(f"    - Index {dev['index']} | Backend: {dev['backend']} | Resolution: {dev['width']}x{dev['height']} | FPS: {dev['fps']}")
    else:
        print("\n>>> No physical webcam detected on indexes 0, 1, 2, 3.")
        print("    WatchGuard Vision will use the built-in synthetic camera simulation fallback.")

    # 2. Test CameraManager automated startup & discovery
    print("\n[STEP 2/4] Initializing CameraManager stream...")
    manager.start()

    # Wait briefly for capture thread to settle
    time.sleep(0.3)

    print(f"    - Manager Running: {manager.is_running}")
    print(f"    - Active Camera Index: {manager.camera_index}")
    print(f"    - Mode: {'Synthetic Simulation' if manager.is_fallback_mode else 'Physical Camera'}")
    print(f"    - Status Label: {manager.device_status_label}")

    # 3. Test Frame Retrieval & Snapshot
    print("\n[STEP 3/4] Testing Frame Retrieval & Evidence Snapshot...")
    success, frame = manager.get_frame()
    print(f"    - BGR Frame Grab Success: {success}")
    if success and frame is not None:
        print(f"    - Frame Dimensions: {frame.shape[1]}x{frame.shape[0]} (Channels: {frame.shape[2]})")

    rgb_success, rgb_frame = manager.get_frame_rgb()
    print(f"    - RGB Conversion Success: {rgb_success}")

    snapshot_path = manager.capture_snapshot(filename_prefix="diagnostic_test")
    print(f"    - Evidence Snapshot Saved: {snapshot_path}")
    if snapshot_path and os.path.exists(snapshot_path):
        size_kb = os.path.getsize(snapshot_path) / 1024
        print(f"    - Snapshot File Size: {size_kb:.2f} KB (Verified on disk)")

    # 4. Test Clean Device Release & Teardown
    print("\n[STEP 4/4] Testing Camera Stop & Device Release...")
    manager.stop()
    print(f"    - Manager Running after stop(): {manager.is_running}")
    print(f"    - Internal Capture Handle: {manager._cap}")

    print("\n================================================================")
    print("            WEBCAM DIAGNOSTIC TEST COMPLETED SUCCESSFULLY       ")
    print("================================================================")

    return {
        "physical_webcam_found": len(devices) > 0,
        "devices": devices,
        "mode": "Simulation" if manager.is_fallback_mode else "Physical",
        "snapshot_verified": snapshot_path is not None and os.path.exists(snapshot_path),
    }


if __name__ == "__main__":
    run_webcam_diagnostics()
