"""
Threaded Camera Manager
-----------------------
Provides robust, thread-safe webcam frame capture with OpenCV, automated
device index probing, startup hardware diagnostics, dynamic FPS calculation,
graceful disconnection handling, snapshot capture, and an explicit synthetic
test pattern simulator for testing environments.
"""

from enum import Enum
import logging
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from config import (
    CAMERA_FRAME_WIDTH,
    CAMERA_FRAME_HEIGHT,
    DEFAULT_CAMERA_INDEX,
    TARGET_FPS,
    EVIDENCE_DIR,
    DEMO_FACES_DIR,
)

logger = logging.getLogger("WatchGuardVision.Camera")


class CameraStatus(Enum):
    """Current operating status of the Camera Manager."""
    OFFLINE = "OFFLINE"
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    ERROR = "ERROR"


class CameraMode(Enum):
    """Requested operating mode."""
    PHYSICAL = "PHYSICAL"
    SYNTHETIC = "SYNTHETIC"


class CameraManager:
    """Threaded OpenCV Camera Manager for WatchGuard Vision."""

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.actual_resolution: Tuple[int, int] = (CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        self._status = CameraStatus.OFFLINE
        self._mode = CameraMode.PHYSICAL
        self._allow_synthetic_fallback = False
        self._error_message: Optional[str] = None
        self._consecutive_drops = 0

        self._demo_mode = "none"  # "none", "authorized", "unknown"
        self._demo_alice_cache: Optional[np.ndarray] = None
        self._demo_unknown_cache: Optional[np.ndarray] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = time.time()
        self._listeners: List[Callable[[np.ndarray], None]] = []
        self._on_device_selected: Optional[Callable[[int], None]] = None

        # Standby frame
        self._latest_frame: Optional[np.ndarray] = None

    # --- Properties ---

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def status_label(self) -> str:
        if self._status == CameraStatus.REAL:
            return f"REAL (Camera #{self.camera_index})"
        elif self._status == CameraStatus.SYNTHETIC:
            return f"SYNTHETIC ({self._demo_mode.upper()})"
        elif self._status == CameraStatus.ERROR:
            return f"ERROR: {self._error_message or 'Hardware Failure'}"
        return "OFFLINE"

    @property
    def is_physical(self) -> bool:
        return self._status == CameraStatus.REAL

    @property
    def is_fallback_mode(self) -> bool:
        """Compatibility property: True only when operating in synthetic simulation mode."""
        return self._status == CameraStatus.SYNTHETIC

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @property
    def demo_mode(self) -> str:
        return self._demo_mode

    @property
    def device_status_label(self) -> str:
        if not self._running:
            return "Offline"
        if self._status == CameraStatus.SYNTHETIC:
            return "Synthetic Simulation"
        if self._status == CameraStatus.ERROR:
            return f"Error: {self._error_message or 'Camera Failed'}"
        return f"Physical Camera #{self.camera_index} ({self.actual_resolution[0]}x{self.actual_resolution[1]})"

    def set_demo_mode(self, mode: str) -> None:
        """Sets the simulation target: 'none' (clear room), 'authorized' (Hunter/Alice), 'unknown' (Visitor)."""
        self._demo_mode = mode
        if mode in ("authorized", "unknown"):
            self._mode = CameraMode.SYNTHETIC
            if self._running:
                self._status = CameraStatus.SYNTHETIC
                with self._lock:
                    self._latest_frame = self._generate_synthetic_frame(time.time())
            logger.info(f"[CAMERA] Demo target simulation mode set to: '{mode}' (Status: SYNTHETIC)")
        else:
            self._demo_mode = "none"
            self._mode = CameraMode.PHYSICAL
            logger.info("[CAMERA] Demo mode cleared; default mode is PHYSICAL.")

    def set_on_device_selected_callback(self, callback: Callable[[int], None]) -> None:
        """Register a callback when a camera device is selected or auto-detected."""
        self._on_device_selected = callback

    def add_frame_listener(self, listener: Callable[[np.ndarray], None]) -> None:
        """Registers a callback that gets invoked whenever a new frame arrives."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_frame_listener(self, listener: Callable[[np.ndarray], None]) -> None:
        """Unregisters a frame listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    # --- Hardware Probing & Opening ---

    def probe_available_devices(self, max_indices: int = 4) -> List[Dict[str, Any]]:
        """
        Scans indexes 0 through max_indices-1 and returns diagnostics for all found devices.
        """
        found_devices = []
        backends = [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("CAP_MSMF", cv2.CAP_MSMF),
        ]

        logger.debug(f"[CAMERA DIAGNOSTIC] Probing camera device indexes 0 to {max_indices - 1}...")
        for idx in range(max_indices):
            for b_name, b_flag in backends:
                try:
                    cap = cv2.VideoCapture(idx, b_flag)
                    is_opened = cap.isOpened()
                    if is_opened:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = float(cap.get(cv2.CAP_PROP_FPS))
                            dev_info = {
                                "index": idx,
                                "backend": b_name,
                                "opened": True,
                                "frame_read": True,
                                "width": w,
                                "height": h,
                                "fps": fps,
                            }
                            found_devices.append(dev_info)
                            logger.debug(
                                f"[CAMERA DIAGNOSTIC] -> Index {idx} [{b_name}]: OPENED=True, Read=True, Resolution={w}x{h}, FPS={fps}"
                            )
                            cap.release()
                            break
                        cap.release()
                except Exception as e:
                    logger.debug(f"[CAMERA DIAGNOSTIC] Probe index {idx} ({b_name}) error: {e}")

        logger.debug(f"[CAMERA DIAGNOSTIC] Probe complete. Found {len(found_devices)} active physical camera device(s).")
        return found_devices

    def _test_and_open_device(self, idx: int, backend_flag: int, backend_name: str) -> Optional[Tuple[cv2.VideoCapture, int, int, np.ndarray]]:
        """Attempts to open a specific camera index with a specific backend and reads a verification frame."""
        logger.debug(f"[CAMERA DIAGNOSTIC] Attempting camera index {idx} with backend {backend_name}...")
        try:
            cap = cv2.VideoCapture(idx, backend_flag)
            is_opened = cap.isOpened()
            logger.debug(f"[CAMERA DIAGNOSTIC] Index {idx} ({backend_name}) VideoCapture.isOpened(): {is_opened}")

            if not is_opened:
                cap.release()
                return None

            # Configure hardware codec, resolution, and target FPS before first read
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Test frame read to confirm active data stream
            ret, frame = cap.read()
            frame_valid = ret and frame is not None and frame.size > 0
            logger.debug(f"[CAMERA DIAGNOSTIC] Index {idx} ({backend_name}) frame read status: ret={ret}, valid={frame_valid}")

            if not frame_valid:
                logger.warning(f"[CAMERA DIAGNOSTIC] Index {idx} ({backend_name}) opened but returned invalid frame.")
                cap.release()
                return None

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAMERA_FRAME_WIDTH
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAMERA_FRAME_HEIGHT
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            logger.debug(f"[CAMERA DIAGNOSTIC] Index {idx} ({backend_name}) SUCCESS! Detected Resolution: {w}x{h}, FPS: {fps}")
            return cap, w, h, frame
        except Exception as e:
            logger.error(f"[CAMERA DIAGNOSTIC] Exception while opening index {idx} with {backend_name}: {e}")
            return None

    def _open_physical_camera(self, auto_discover: bool = True) -> bool:
        """
        Attempts to open the physical camera device.
        Prioritizes the selected index, and optionally auto-probes [0, 1, 2, 3].
        """
        if auto_discover and self.camera_index >= 0:
            candidates = [self.camera_index]
            for i in [0, 1, 2, 3]:
                if i not in candidates:
                    candidates.append(i)
        else:
            candidates = [self.camera_index]

        backends = [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
        ]

        logger.debug(f"[CAMERA DIAGNOSTIC] Starting physical webcam discovery. Candidates: {candidates}")

        dark_candidate = None
        for idx in candidates:
            for b_name, b_flag in backends:
                res = self._test_and_open_device(idx, b_flag, b_name)
                if res is not None:
                    cap, w, h, frame = res
                    frame_mean = float(frame.mean())
                    if not auto_discover or frame_mean > 5.0:
                        # Direct operator selection or active lit webcam confirmed
                        self._cap = cap
                        self.camera_index = idx
                        self.actual_resolution = (w, h)
                        self._status = CameraStatus.REAL
                        self._error_message = None
                        self._consecutive_drops = 0
                        with self._lock:
                            self._latest_frame = frame.copy()

                        print("==================================================")
                        print("  CAMERA INITIALIZATION DIAGNOSTIC")
                        print(f"  STATUS: REAL")
                        print(f"  DEVICE INDEX: {idx}")
                        print(f"  BACKEND: {b_name}")
                        print(f"  RESOLUTION: {w}x{h}")
                        print(f"  FRAME MEAN BRIGHTNESS: {frame_mean:.1f}")
                        print(f"  TARGET FPS: {TARGET_FPS}")
                        print("==================================================")
                        logger.info(f"[CAMERA] Physical webcam confirmed and activated on index {idx} ({w}x{h}, brightness={frame_mean:.1f}).")

                        if self._on_device_selected is not None:
                            try:
                                self._on_device_selected(idx)
                            except Exception as e:
                                logger.error(f"Error in on_device_selected callback: {e}")

                        return True
                    elif dark_candidate is None:
                        logger.debug(f"[CAMERA DIAGNOSTIC] Candidate index {idx} opened but is dark (mean={frame_mean:.1f}). Checking other candidates...")
                        dark_candidate = (cap, w, h, frame, idx, b_name, frame_mean)
                    else:
                        cap.release()

        # If only dark candidate was available and no lit camera was found
        if dark_candidate is not None:
            cap, w, h, frame, idx, b_name, frame_mean = dark_candidate
            self._cap = cap
            self.camera_index = idx
            self.actual_resolution = (w, h)
            self._status = CameraStatus.REAL
            self._error_message = None
            self._consecutive_drops = 0
            with self._lock:
                self._latest_frame = frame.copy()

            print("==================================================")
            print("  CAMERA INITIALIZATION DIAGNOSTIC")
            print(f"  STATUS: REAL (Dark / Shuttered Sensor)")
            print(f"  DEVICE INDEX: {idx}")
            print(f"  BACKEND: {b_name}")
            print(f"  RESOLUTION: {w}x{h}")
            print(f"  FRAME MEAN BRIGHTNESS: {frame_mean:.1f}")
            print("==================================================")
            logger.info(f"[CAMERA] Physical webcam activated on index {idx} (Dark sensor fallback, {w}x{h}).")

            if self._on_device_selected is not None:
                try:
                    self._on_device_selected(idx)
                except Exception as e:
                    logger.error(f"Error in on_device_selected callback: {e}")

            return True

        # Failed to open any physical camera candidate
        if self._allow_synthetic_fallback:
            self._status = CameraStatus.SYNTHETIC
            self._latest_frame = self._generate_synthetic_frame(time.time())
            logger.warning(f"[CAMERA DIAGNOSTIC] No physical camera found. Falling back to synthetic simulator.")
            print("==================================================")
            print("  CAMERA INITIALIZATION DIAGNOSTIC")
            print("  STATUS: SYNTHETIC (Explicit fallback)")
            print(f"  DEVICE INDEX: {self.camera_index}")
            print("==================================================")
            return True
        else:
            self._status = CameraStatus.ERROR
            self._error_message = f"No physical camera device detected on candidate indexes {candidates}."
            logger.error(f"[CAMERA DIAGNOSTIC] {self._error_message}")
            print("==================================================")
            print("  CAMERA INITIALIZATION DIAGNOSTIC")
            print("  STATUS: ERROR")
            print(f"  ERROR: {self._error_message}")
            print("==================================================")
            return False

    # --- Lifecycle Controls ---

    def start(
        self,
        camera_index: Optional[int] = None,
        mode: CameraMode = CameraMode.PHYSICAL,
        allow_synthetic_fallback: bool = False,
        auto_discover: bool = True,
    ) -> bool:
        """
        Starts the threaded camera capture loop.
        
        Args:
            camera_index: Physical camera index (default: 0).
            mode: Operating mode (PHYSICAL or SYNTHETIC).
            allow_synthetic_fallback: If True, falls back to synthetic mode when physical device is missing.
                                      Default is False (reports explicit ERROR).
            auto_discover: If True, probes fallback indexes 0..3 if the requested index fails.
        """
        with self._lock:
            if self._running and self._status in (CameraStatus.REAL, CameraStatus.SYNTHETIC):
                return True

            if camera_index is not None:
                self.camera_index = camera_index

            self._mode = mode
            self._allow_synthetic_fallback = allow_synthetic_fallback
            self._running = True
            self._consecutive_drops = 0

            if mode == CameraMode.SYNTHETIC or self._demo_mode in ("authorized", "unknown"):
                self._mode = CameraMode.SYNTHETIC
                self._status = CameraStatus.SYNTHETIC
                self._latest_frame = self._generate_synthetic_frame(time.time())
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                return True

            # Open physical camera
            opened = self._open_physical_camera(auto_discover=auto_discover)
            if not opened and not allow_synthetic_fallback:
                self._running = False
                self._latest_frame = self._generate_error_frame(self._error_message or "Physical Camera Unavailable")
                return False

            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        """Stops capture loop and safely releases the webcam device."""
        with self._lock:
            if not self._running and self._cap is None:
                self._status = CameraStatus.OFFLINE
                return
            self._running = False

        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            self._status = CameraStatus.OFFLINE
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.debug(f"[CAMERA DIAGNOSTIC] Error releasing VideoCapture in stop: {e}")
                finally:
                    self._cap = None
            self._latest_frame = None
            self._fps = 0.0

    def restart(self, camera_index: Optional[int] = None) -> bool:
        """Restarts capture with a specified or existing camera index."""
        self.stop()
        return self.start(camera_index=camera_index, mode=self._mode, allow_synthetic_fallback=self._allow_synthetic_fallback)

    # --- Frame Retrieval ---

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns latest captured frame (BGR format) and success flag."""
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def get_frame_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns latest captured frame converted to RGB for GUI rendering."""
        success, frame = self.get_frame()
        if success and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, rgb_frame
        return False, None

    def capture_snapshot(self, filename_prefix: str = "evidence") -> Optional[str]:
        """Saves current frame to evidence directory and returns filepath."""
        success, frame = self.get_frame()
        if not success or frame is None:
            return None

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{filename_prefix}_{timestamp_str}.jpg"
        filepath = str(EVIDENCE_DIR / filename)

        try:
            cv2.imwrite(filepath, frame)
            logger.debug(f"[CAMERA] Evidence snapshot captured: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"[CAMERA] Failed to capture snapshot: {e}")
            return None

    # --- Capture Loop ---

    def _capture_loop(self) -> None:
        """Background thread capture loop."""
        frame_counter = 0
        fps_timer = time.time()

        while self._running:
            start_loop = time.time()

            if self._status == CameraStatus.REAL and self._cap is not None and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    self._consecutive_drops = 0
                    captured_frame = frame
                else:
                    self._consecutive_drops += 1
                    if self._consecutive_drops < 15:
                        # Transient drop; keep last valid frame briefly
                        captured_frame = self._latest_frame if self._latest_frame is not None else self._generate_error_frame("Camera Frame Dropped")
                    else:
                        # Hardware disconnected
                        logger.error("[CAMERA] Physical camera stream lost (15 consecutive drop frames).")
                        self._status = CameraStatus.ERROR
                        self._error_message = "Webcam hardware stream disconnected."
                        captured_frame = self._generate_error_frame("Hardware Stream Disconnected")
            elif self._status == CameraStatus.SYNTHETIC:
                captured_frame = self._generate_synthetic_frame(time.time())
            elif self._status == CameraStatus.ERROR:
                captured_frame = self._generate_error_frame(self._error_message or "Camera Hardware Unavailable")
            else:
                captured_frame = self._generate_synthetic_frame(time.time())

            # Update latest frame
            with self._lock:
                self._latest_frame = captured_frame

            # Calculate FPS
            frame_counter += 1
            if time.time() - fps_timer >= 1.0:
                self._fps = frame_counter / (time.time() - fps_timer)
                frame_counter = 0
                fps_timer = time.time()

            # Notify frame listeners
            for listener in self._listeners:
                try:
                    listener(captured_frame)
                except Exception:
                    pass

            # Frame rate pacing (~30 FPS)
            elapsed = time.time() - start_loop
            sleep_time = max(0.001, (1.0 / TARGET_FPS) - elapsed)
            time.sleep(sleep_time)

        # Cleanup on thread exit
        with self._lock:
            if self._cap is not None:
                try:
                    logger.debug(f"[CAMERA DIAGNOSTIC] Loop terminating. Releasing VideoCapture index {self.camera_index}...")
                    self._cap.release()
                except Exception as e:
                    logger.error(f"[CAMERA DIAGNOSTIC] Error releasing VideoCapture in loop: {e}")
                finally:
                    self._cap = None

    # --- Frame Generators (Synthetic Simulator & Error Screens) ---

    def _generate_error_frame(self, message: str) -> np.ndarray:
        """Generates a high-contrast hardware error frame."""
        w, h = CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Red warning border
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 0, 220), 2)
        cv2.putText(
            frame,
            "WATCHGUARD VISION // CAMERA HARDWARE ERROR",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"STATUS: ERROR (Device #{self.camera_index})",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 165, 255),
            1,
        )
        cv2.putText(
            frame,
            f"DETAILS: {message[:55]}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "Please verify USB camera connection or Windows camera permissions.",
            (30, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (140, 140, 140),
            1,
        )
        return frame

    def _generate_synthetic_frame(self, t: float) -> np.ndarray:
        """Generates synthetic test pattern for explicit demo/test simulations."""
        w, h = CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw grid background
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (25, 30, 38), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (25, 30, 38), 1)

        # Center reticle
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 120, (40, 60, 80), 2)
        cv2.circle(frame, (cx, cy), 60, (0, 180, 240), 1)
        cv2.line(frame, (cx - 140, cy), (cx + 140, cy), (0, 180, 240), 1)
        cv2.line(frame, (cx, cy - 140), (cx, cy + 140), (0, 180, 240), 1)

        # Simulation demo targets
        target_label = "PERIMETER CLEAR"
        if self._demo_mode == "authorized":
            target_label = "SIMULATING: AUTHORIZED (HUNTER)"
            if self._demo_alice_cache is None:
                alice_path = DEMO_FACES_DIR / "officer_alice.jpg"
                if alice_path.exists():
                    self._demo_alice_cache = cv2.imread(str(alice_path))
            if self._demo_alice_cache is not None:
                face_crop = cv2.resize(self._demo_alice_cache, (240, 240))
                ox = int(math.sin(t * 1.2) * 15)
                oy = int(math.cos(t * 1.0) * 10)
                fx1 = max(0, cx - 120 + ox)
                fy1 = max(0, cy - 120 + oy)
                frame[fy1:fy1 + 240, fx1:fx1 + 240] = face_crop
                cv2.rectangle(frame, (fx1 - 2, fy1 - 2), (fx1 + 242, fy1 + 242), (0, 200, 255), 1)

        elif self._demo_mode == "unknown":
            target_label = "SIMULATING: UNKNOWN VISITOR"
            if self._demo_unknown_cache is None:
                unk_path = DEMO_FACES_DIR / "unknown_visitor.jpg"
                if unk_path.exists():
                    self._demo_unknown_cache = cv2.imread(str(unk_path))
            if self._demo_unknown_cache is not None:
                face_crop = cv2.resize(self._demo_unknown_cache, (200, 200))
                ox = int(math.sin(t * 1.5) * 15)
                oy = int(math.cos(t * 1.2) * 10)
                fx1 = max(0, cx - 100 + ox)
                fy1 = max(0, cy - 100 + oy)
                frame[fy1:fy1 + 200, fx1:fx1 + 200] = face_crop
                cv2.rectangle(frame, (fx1 - 2, fy1 - 2), (fx1 + 202, fy1 + 202), (0, 200, 255), 1)

        # Pulse scan line
        scan_y = int((math.sin(t * 2.0) + 1.0) * 0.5 * (h - 60)) + 30
        cv2.line(frame, (20, scan_y), (w - 20, scan_y), (0, 255, 180), 2)

        # Header overlays
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(
            frame,
            "WATCHGUARD VISION // LIVE FEED [SYNTHETIC SIMULATOR]",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            2,
        )
        cv2.putText(
            frame,
            f"TIME: {now_str}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 200, 220),
            1,
        )
        cv2.putText(
            frame,
            f"STATUS: DEMO TARGET | {target_label}",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 220, 100) if self._demo_mode != "unknown" else (50, 100, 240),
            1,
        )
        cv2.putText(
            frame,
            f"[Synthetic Mode Active - Target: {self._demo_mode.upper()}]",
            (30, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
        )
        return frame
