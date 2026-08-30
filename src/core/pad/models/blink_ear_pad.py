"""
Multi-Cue Behavioral & Texture Presentation Attack Detection (PAD) Engine
-------------------------------------------------------------------------
Engineered baseline fusing four complementary, per-track liveness signals:
1. Spatial Screen/Display Object Containment (Cell Phone / Monitor Enclosure)
2. Eye-Region Temporal Photometric Transition (Blink Cycle Tracking per Track)
3. Normalized 3D Facial Non-Rigidity (Affine-Invariant Geometric Ratios & 3D Parallax)
4. Screen Moiré & Frequency Texture Analysis (Laplacian Variance & Color Dispersion)
5. Per-Track Temporal Decision Consistency & Hysteresis State Stabilizer

Guarantees strict isolation across simultaneous face tracks, prevents single-frame
authorization glitches, and stabilizes access decisions across frame sequences.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    PAD_PROVISIONAL_LIVENESS_THRESHOLD,
    PAD_PROVISIONAL_SPOOF_THRESHOLD,
    PAD_VERIFICATION_TIMEOUT_SECONDS,
)
from src.core.pad.base_pad import AttackType, BasePADEngine, PADResult

logger = logging.getLogger("WatchGuardVision.PAD.MultiCue")

# Screen / display COCO classes to check for face enclosure
SCREEN_OBJECT_CLASSES = {"cell phone", "laptop", "tv", "monitor", "tablet", "book"}


class PADStabilizedState(str, Enum):
    """Temporally filtered per-track PAD decision state."""
    WARMUP = "WARMUP"
    UNVERIFIED_PENDING = "UNVERIFIED_PENDING"
    STABLE_LIVE = "STABLE_LIVE"
    STABLE_SPOOF = "STABLE_SPOOF"
    UNVERIFIED_TIMEOUT = "UNVERIFIED_TIMEOUT"


@dataclass
class DiagnosticSignals:
    """Underlying signals tracked per face track for explainable PAD decisions."""
    track_id: int
    buffer_len: int
    eye_gradient_energy: float
    eye_contrast: float
    blink_count: int
    blink_state: str
    norm_motion_std: float
    ratio_motion_std: float
    is_rigid_2d: bool
    is_inside_screen: bool
    screen_label: str
    texture_sharpness: float
    color_dispersion: float
    warmup_active: bool
    stable_state: str
    history_symbols: str


class TrackedPADStabilizer:
    """
    Per-track temporal evidence accumulator and hysteresis state machine.
    Prevents single-frame decision flipping, filters detector noise, and
    requires sustained consecutive evidence before granting or rejecting access.
    Enforces a strict configurable maximum verification window (default 5.0s).
    """

    def __init__(
        self,
        window_size: int = 10,
        live_confirm_frames: int = 5,
        spoof_confirm_frames: int = 4,
        warmup_frames: int = 6,
        timeout_seconds: float = PAD_VERIFICATION_TIMEOUT_SECONDS,
    ):
        self.window_size = window_size
        self.live_confirm_frames = live_confirm_frames
        self.spoof_confirm_frames = spoof_confirm_frames
        self.warmup_frames = warmup_frames
        self.timeout_seconds = timeout_seconds

        self.history: deque = deque(maxlen=window_size)
        self.state: PADStabilizedState = PADStabilizedState.WARMUP
        self.consecutive_live: int = 0
        self.consecutive_spoof: int = 0
        self.smoothed_confidence: float = 0.50
        self.stable_attack_type: AttackType = AttackType.UNKNOWN
        self.stable_tag: str = "WARMUP_UNVERIFIED"

    def update(
        self,
        raw_is_live: bool,
        raw_confidence: float,
        raw_attack_type: AttackType,
        raw_tag: str,
        is_screen: bool,
        elapsed_seconds: float = 0.0,
    ) -> Tuple[PADStabilizedState, float, AttackType, str]:
        # Record raw frame observation
        entry = {
            "is_live": raw_is_live,
            "confidence": raw_confidence,
            "attack_type": raw_attack_type,
            "tag": raw_tag,
            "is_screen": is_screen,
        }
        self.history.append(entry)

        # Update consecutive streaks
        if is_screen or not raw_is_live:
            self.consecutive_spoof += 1
            self.consecutive_live = 0
        else:
            self.consecutive_live += 1
            self.consecutive_spoof = 0

        # Calculate window statistics
        n_total = len(self.history)
        n_live = sum(1 for e in self.history if e["is_live"])
        n_spoof = n_total - n_live

        # Compute smoothed confidence with recency weighting
        weights = np.linspace(0.6, 1.0, n_total)
        weights /= np.sum(weights)
        conf_arr = np.array([e["confidence"] for e in self.history])
        self.smoothed_confidence = float(np.sum(conf_arr * weights))

        # 1. IMMEDIATE OVERRIDE: Physical Screen Containment (Phone / Screen detected)
        if is_screen:
            self.state = PADStabilizedState.STABLE_SPOOF
            self.stable_attack_type = AttackType.REPLAY
            self.stable_tag = raw_tag
            self.smoothed_confidence = min(self.smoothed_confidence, 0.10)
            return self.state, self.smoothed_confidence, self.stable_attack_type, self.stable_tag

        # 2. TIMEOUT OVERRIDE: Verification window exceeded (>=5.0s) without live confirmation -> ACCESS DENIED
        if elapsed_seconds >= self.timeout_seconds and self.state != PADStabilizedState.STABLE_LIVE:
            self.state = PADStabilizedState.UNVERIFIED_TIMEOUT
            self.stable_attack_type = AttackType.UNKNOWN
            self.stable_tag = "PAD_TIMEOUT_UNVERIFIED"
            self.smoothed_confidence = min(0.40, self.smoothed_confidence)
            return self.state, self.smoothed_confidence, self.stable_attack_type, self.stable_tag

        # 3. WARMUP STAGE (before warmup frames reached and not previously stabilized)
        if self.state == PADStabilizedState.WARMUP and n_total < self.warmup_frames:
            self.state = PADStabilizedState.WARMUP
            self.stable_attack_type = AttackType.UNKNOWN
            self.stable_tag = "WARMUP_UNVERIFIED"
            return self.state, self.smoothed_confidence, self.stable_attack_type, self.stable_tag

        # 4. HYSTERESIS STATE MACHINE TRANSITIONS
        if self.state in (PADStabilizedState.WARMUP, PADStabilizedState.UNVERIFIED_PENDING, PADStabilizedState.UNVERIFIED_TIMEOUT):
            # To become STABLE_LIVE: must have sustained consecutive live evidence (>= 5 frames)
            if self.consecutive_live >= self.live_confirm_frames:
                self.state = PADStabilizedState.STABLE_LIVE
                self.stable_attack_type = AttackType.BONAFIDE
                self.stable_tag = raw_tag if raw_is_live else "BONAFIDE_STABLE"
            # To become STABLE_SPOOF: must have sustained consecutive spoof evidence (>= 4 frames)
            elif self.consecutive_spoof >= self.spoof_confirm_frames:
                self.state = PADStabilizedState.STABLE_SPOOF
                self.stable_attack_type = raw_attack_type if not raw_is_live else AttackType.REPLAY
                self.stable_tag = raw_tag if not raw_is_live else "STABLE_SPOOF_REJECTED"
            else:
                self.state = PADStabilizedState.UNVERIFIED_PENDING
                self.stable_attack_type = AttackType.UNKNOWN
                self.stable_tag = "OSCILLATING_EVIDENCE_PENDING"

        elif self.state == PADStabilizedState.STABLE_LIVE:
            # Hysteresis: require sustained spoof evidence to demote or flip
            if self.consecutive_spoof >= self.spoof_confirm_frames:
                self.state = PADStabilizedState.STABLE_SPOOF
                self.stable_attack_type = raw_attack_type if not raw_is_live else AttackType.REPLAY
                self.stable_tag = raw_tag if not raw_is_live else "STABLE_SPOOF_REJECTED"
            elif self.consecutive_spoof >= 2:
                # Temporary glitch / ambiguous drop: drop to PENDING rather than immediate spoof
                self.state = PADStabilizedState.UNVERIFIED_PENDING
                self.stable_attack_type = AttackType.UNKNOWN
                self.stable_tag = "AMBIGUOUS_EVIDENCE_PENDING"
            else:
                # Maintain STABLE_LIVE (absorb isolated single-frame detector glitch)
                self.stable_attack_type = AttackType.BONAFIDE
                if raw_is_live:
                    self.stable_tag = raw_tag

        elif self.state == PADStabilizedState.STABLE_SPOOF:
            # Hysteresis: require strong sustained live evidence (>= 5 frames) to become live
            if self.consecutive_live >= self.live_confirm_frames:
                self.state = PADStabilizedState.STABLE_LIVE
                self.stable_attack_type = AttackType.BONAFIDE
                self.stable_tag = raw_tag if raw_is_live else "BONAFIDE_STABLE"
            elif self.consecutive_live >= 2:
                # Evidence is mixed: drop to PENDING (never directly grant authorization)
                self.state = PADStabilizedState.UNVERIFIED_PENDING
                self.stable_attack_type = AttackType.UNKNOWN
                self.stable_tag = "MIXED_EVIDENCE_PENDING"
            else:
                # Maintain STABLE_SPOOF (absorb isolated single-frame photo jitter)
                if not raw_is_live:
                    self.stable_attack_type = raw_attack_type
                    self.stable_tag = raw_tag

        # Calibrate smoothed confidence range to matching stable state
        if self.state == PADStabilizedState.STABLE_LIVE:
            self.smoothed_confidence = max(0.85, max(self.smoothed_confidence, raw_confidence))
        elif self.state == PADStabilizedState.STABLE_SPOOF:
            self.smoothed_confidence = min(0.25, min(self.smoothed_confidence, raw_confidence))

        return self.state, self.smoothed_confidence, self.stable_attack_type, self.stable_tag

    @property
    def history_symbols(self) -> str:
        """Returns compact string representation of last observations e.g. 'LLSLS'."""
        return "".join("L" if e["is_live"] else "S" for e in self.history)


class TrackedFacePADState:
    """Independent temporal state buffer for an individual face track."""

    def __init__(
        self,
        track_id: int,
        initial_bbox: Optional[Tuple[int, int, int, int]] = None,
        buffer_size: int = 30,
        warmup_frames: int = 6,
    ):
        self.track_id = track_id
        self.bbox: Tuple[int, int, int, int] = tuple(map(int, initial_bbox)) if initial_bbox is not None else (0, 0, 0, 0)
        self.first_seen_time: float = time.time()
        self.last_seen_time: float = time.time()
        self.consecutive_frames: int = 0
        self.missing_frames: int = 0

        # Temporal rolling history
        self.norm_landmarks_history: deque = deque(maxlen=buffer_size)
        self.distance_ratios_history: deque = deque(maxlen=buffer_size)
        self.eye_gradient_history: deque = deque(maxlen=buffer_size)
        self.timestamps: deque = deque(maxlen=buffer_size)

        # Eye Blink State Machine per track
        self.blink_count: int = 0
        self.blink_state: str = "OPEN"  # "OPEN", "CLOSING", "CLOSED", "REOPENING"
        self.closed_frame_count: int = 0
        self.baseline_eye_gradient: float = 15.0

        # Screen detection flag
        self.is_inside_screen: bool = False
        self.screen_object_label: str = ""

        # Per-track temporal stabilizer & hysteresis state machine
        self.stabilizer: TrackedPADStabilizer = TrackedPADStabilizer(
            window_size=10,
            live_confirm_frames=5,
            spoof_confirm_frames=4,
            warmup_frames=warmup_frames,
        )

        # Last diagnostic snapshot
        self.last_diagnostics: Optional[DiagnosticSignals] = None


class BlinkEARPADEngine(BasePADEngine):
    """
    Multi-Cue Behavioral & Photometric Presentation Attack Detector.
    Maintains isolated per-track states to evaluate individual face presentations.
    """

    def __init__(
        self,
        buffer_size: int = 30,
        warmup_frames: int = 6,
        rigid_motion_threshold: float = 0.010,
        ratio_motion_threshold: float = 0.008,
        eye_closure_drop_ratio: float = 0.65,
        eye_reopen_recovery_ratio: float = 0.80,
    ):
        self.buffer_size = buffer_size
        self.warmup_frames = warmup_frames
        self.rigid_motion_threshold = rigid_motion_threshold
        self.ratio_motion_threshold = ratio_motion_threshold
        self.eye_closure_drop_ratio = eye_closure_drop_ratio
        self.eye_reopen_recovery_ratio = eye_reopen_recovery_ratio

        # Multi-face track registry
        self._tracks: Dict[int, TrackedFacePADState] = {}
        self._next_track_id: int = 1
        self._is_ready = False

    @property
    def model_name(self) -> str:
        return "Blink-EAR (Multi-Cue Baseline)"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def initialize(self) -> bool:
        self._tracks.clear()
        self._next_track_id = 1
        self._is_ready = True
        logger.info("Multi-Cue Behavioral & Texture PAD Engine initialized.")
        return True

    def reset_tracks(self) -> None:
        """Clears all active face tracks."""
        self._tracks.clear()
        self._next_track_id = 1

    def _check_screen_containment(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
        objects: Optional[List[Dict[str, Any]]],
    ) -> Tuple[bool, str]:
        """
        Determines if a face bounding box is physically enclosed by or substantially
        overlapping with a detected smartphone/screen/display object.
        """
        if not objects or bbox is None:
            return False, ""

        fx, fy, fw, fh = bbox
        if fw <= 0 or fh <= 0:
            return False, ""

        fcx = fx + fw * 0.5
        fcy = fy + fh * 0.5
        face_area = float(fw * fh)

        for obj in objects:
            lbl = obj.get("label", "").lower()
            if lbl in SCREEN_OBJECT_CLASSES:
                ox, oy, ow, oh = obj.get("bbox", (0, 0, 0, 0))
                if ow <= 0 or oh <= 0:
                    continue

                # 1. Centroid containment (center of face is inside the phone/screen)
                if ox <= fcx <= (ox + ow) and oy <= fcy <= (oy + oh):
                    return True, lbl

                # 2. Significant intersection area relative to face
                ix1 = max(fx, ox)
                iy1 = max(fy, oy)
                ix2 = min(fx + fw, ox + ow)
                iy2 = min(fy + fh, oy + oh)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = float(iw * ih)

                if (intersection / face_area) >= 0.28:
                    return True, lbl

        return False, ""

    def _get_or_create_track(
        self,
        track_id: Optional[int],
        face_bbox: Optional[Tuple[int, int, int, int]],
        detected_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> TrackedFacePADState:
        """
        Matches a face instance to an existing track using track_id, IoU, or spatial proximity.
        Creates a new isolated track if no match is found.
        """
        now = time.time()

        # Clean stale tracks (> 1.5s unseen)
        for tid in list(self._tracks.keys()):
            if (now - self._tracks[tid].last_seen_time) > 1.5:
                del self._tracks[tid]

        # 1. Direct track_id lookup
        if track_id is not None and track_id in self._tracks:
            track = self._tracks[track_id]
            if face_bbox:
                track.bbox = face_bbox
            track.last_seen_time = now
            track.consecutive_frames += 1
            track.missing_frames = 0
            is_scr, scr_lbl = self._check_screen_containment(track.bbox, detected_objects)
            track.is_inside_screen = is_scr
            track.screen_object_label = scr_lbl
            return track

        # 2. Spatial IoU / Centroid matching
        best_match_tid: Optional[int] = None
        best_score = -1.0

        if face_bbox is not None and len(face_bbox) == 4:
            fx, fy, fw, fh = face_bbox
            fcx = fx + fw * 0.5
            fcy = fy + fh * 0.5

            for tid, track in self._tracks.items():
                tx, ty, tw, th = track.bbox
                tcx = tx + tw * 0.5
                tcy = ty + th * 0.5

                # Compute IoU
                ix1 = max(fx, tx)
                iy1 = max(fy, ty)
                ix2 = min(fx + fw, tx + tw)
                iy2 = min(fy + fh, ty + th)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih
                union = fw * fh + tw * th - intersection
                iou = (intersection / union) if union > 0 else 0.0

                # Compute centroid distance normalized by face size
                dist_norm = math.hypot(fcx - tcx, fcy - tcy) / max(1.0, float(max(fw, tw)))

                # Match criteria: IoU >= 0.25 OR close centroid (dist_norm <= 0.45)
                if iou >= 0.25 or dist_norm <= 0.45:
                    score = iou + max(0.0, 1.0 - dist_norm)
                    if score > best_score:
                        best_score = score
                        best_match_tid = tid

        if best_match_tid is not None:
            track = self._tracks[best_match_tid]
            if face_bbox:
                track.bbox = face_bbox
            track.last_seen_time = now
            track.consecutive_frames += 1
            track.missing_frames = 0
            is_scr, scr_lbl = self._check_screen_containment(track.bbox, detected_objects)
            track.is_inside_screen = is_scr
            track.screen_object_label = scr_lbl
            return track

        # Fallback: single active track continuity ONLY when face_bbox was not explicitly provided
        if face_bbox is None and track_id is None and len(self._tracks) == 1:
            track = next(iter(self._tracks.values()))
            track.last_seen_time = now
            track.consecutive_frames += 1
            track.missing_frames = 0
            is_scr, scr_lbl = self._check_screen_containment(track.bbox, detected_objects)
            track.is_inside_screen = is_scr
            track.screen_object_label = scr_lbl
            return track

        # 3. Create new isolated track
        new_tid = track_id if track_id is not None else self._next_track_id
        if track_id is None:
            self._next_track_id += 1

        new_track = TrackedFacePADState(
            track_id=new_tid,
            initial_bbox=face_bbox,
            buffer_size=self.buffer_size,
            warmup_frames=self.warmup_frames,
        )
        new_track.consecutive_frames = 1
        is_scr, scr_lbl = self._check_screen_containment(face_bbox, detected_objects)
        new_track.is_inside_screen = is_scr
        new_track.screen_object_label = scr_lbl
        self._tracks[new_tid] = new_track
        return new_track

    def _extract_eye_photometric_gradient(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Extracts eye patches around left and right eye centroids and computes
        vertical Sobel edge energy and contrast.
        """
        if frame_bgr is None or frame_bgr.size == 0 or landmarks is None or len(landmarks) < 10:
            return 0.0, 0.0

        h, w = frame_bgr.shape[:2]
        re_x, re_y = landmarks[0], landmarks[1]
        le_x, le_y = landmarks[2], landmarks[3]

        inter_ocular = math.hypot(le_x - re_x, le_y - re_y)
        if inter_ocular < 8.0:
            return 0.0, 0.0

        # Eye patch half-dimensions
        box_half_w = max(4, int(inter_ocular * 0.22))
        box_half_h = max(3, int(inter_ocular * 0.16))

        gradients = []
        contrasts = []

        for eye_x, eye_y in [(re_x, re_y), (le_x, le_y)]:
            x1 = max(0, int(eye_x - box_half_w))
            x2 = min(w, int(eye_x + box_half_w))
            y1 = max(0, int(eye_y - box_half_h))
            y2 = min(h, int(eye_y + box_half_h))

            if x2 > x1 and y2 > y1:
                patch = frame_bgr[y1:y2, x1:x2]
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                # Compute vertical edge energy (dark pupil + sclera produce high vertical edge energy)
                sobel_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
                grad_val = float(np.mean(np.abs(sobel_y)))
                contrast_val = float(np.std(gray_patch))
                gradients.append(grad_val)
                contrasts.append(contrast_val)

        mean_grad = float(np.mean(gradients)) if gradients else 0.0
        mean_contrast = float(np.mean(contrasts)) if contrasts else 0.0
        return mean_grad, mean_contrast

    def _compute_normalized_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalizes 5-point landmarks relative to nose-tip center and scaled by inter-ocular distance.
        """
        re_x, re_y = landmarks[0], landmarks[1]
        le_x, le_y = landmarks[2], landmarks[3]
        nt_x, nt_y = landmarks[4], landmarks[5]

        scale = math.hypot(le_x - re_x, le_y - re_y) + 1e-5
        norm_pts = np.zeros(10, dtype=np.float32)

        for i in range(5):
            px = landmarks[2 * i]
            py = landmarks[2 * i + 1]
            norm_pts[2 * i] = (px - nt_x) / scale
            norm_pts[2 * i + 1] = (py - nt_y) / scale

        return norm_pts

    def _compute_distance_ratios(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Computes 3 geometric landmark distance ratios:
        R1 = eye-to-eye / nose-to-mouth
        R2 = left_eye-to-nose / right_eye-to-nose
        R3 = mouth_width / eye-to-eye
        For a static 2D planar photo (held or moved), these ratios stay rigid (near-zero variance).
        """
        re_x, re_y = landmarks[0], landmarks[1]
        le_x, le_y = landmarks[2], landmarks[3]
        nt_x, nt_y = landmarks[4], landmarks[5]
        rc_x, rc_y = landmarks[6], landmarks[7]
        lc_x, lc_y = landmarks[8], landmarks[9]

        mc_x = (rc_x + lc_x) * 0.5
        mc_y = (rc_y + lc_y) * 0.5

        d_eyes = math.hypot(le_x - re_x, le_y - re_y)
        d_nose_mouth = math.hypot(mc_x - nt_x, mc_y - nt_y)
        d_re_nose = math.hypot(nt_x - re_x, nt_y - re_y)
        d_le_nose = math.hypot(nt_x - le_x, nt_y - le_y)
        d_mouth = math.hypot(lc_x - rc_x, lc_y - rc_y)

        r1 = d_eyes / (d_nose_mouth + 1e-5)
        r2 = d_re_nose / (d_le_nose + 1e-5)
        r3 = d_mouth / (d_eyes + 1e-5)

        return np.array([r1, r2, r3], dtype=np.float32)

    def _update_track_blink_machine(self, track: TrackedFacePADState, current_gradient: float) -> None:
        """
        Tracks the temporal trajectory of eye-patch photometric gradient for a specific track.
        """
        if current_gradient <= 0.0:
            return

        # Adaptive baseline tracking during steady open state
        if track.blink_state == "OPEN":
            track.baseline_eye_gradient = 0.95 * track.baseline_eye_gradient + 0.05 * current_gradient

            # Detect start of eyelid closure (gradient drops significantly)
            if current_gradient < (track.baseline_eye_gradient * self.eye_closure_drop_ratio):
                track.blink_state = "CLOSING"
                track.closed_frame_count = 1

        elif track.blink_state in ("CLOSING", "CLOSED"):
            if current_gradient < (track.baseline_eye_gradient * self.eye_closure_drop_ratio):
                track.closed_frame_count += 1
                if track.closed_frame_count >= 2:
                    track.blink_state = "CLOSED"
                # If eyes remain closed for > 1.5s (> 45 frames), reset (not a natural blink)
                if track.closed_frame_count > 45:
                    track.blink_state = "OPEN"
                    track.closed_frame_count = 0
            else:
                # Gradient begins recovering
                if track.closed_frame_count >= 1:
                    track.blink_state = "REOPENING"
                else:
                    track.blink_state = "OPEN"

        elif track.blink_state == "REOPENING":
            # Gradient recovered back toward baseline
            if current_gradient >= (track.baseline_eye_gradient * self.eye_reopen_recovery_ratio):
                track.blink_count += 1
                track.blink_state = "OPEN"
                track.closed_frame_count = 0
                logger.debug(f"[TRACK #{track.track_id}] Natural blink confirmed. Total verified blinks: {track.blink_count}")
            else:
                track.closed_frame_count += 1
                if track.closed_frame_count > 15:
                    track.blink_state = "OPEN"
                    track.closed_frame_count = 0

    def evaluate(
        self,
        raw_face_crop: np.ndarray,
        aligned_face: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
        frame_history: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
        track_id: Optional[int] = None,
        face_bbox: Optional[Any] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> PADResult:
        start_time = time.perf_counter()
        now = time.time()

        # Derive face_bbox if not provided
        if face_bbox is None:
            if landmarks is not None and len(landmarks) >= 10:
                xs = landmarks[0::2]
                ys = landmarks[1::2]
                min_x, max_x = float(np.min(xs)), float(np.max(xs))
                min_y, max_y = float(np.min(ys)), float(np.max(ys))
                pad_w = max(10.0, (max_x - min_x) * 0.3)
                pad_h = max(10.0, (max_y - min_y) * 0.3)
                face_bbox = (int(min_x - pad_w), int(min_y - pad_h), int(max_x - min_x + 2 * pad_w), int(max_y - min_y + 2 * pad_h))
            elif raw_face_crop is not None and raw_face_crop.size > 0:
                ch, cw = raw_face_crop.shape[:2]
                face_bbox = (0, 0, cw, ch)

        # 1. Retrieve or allocate isolated Track State
        track = self._get_or_create_track(track_id, face_bbox, detected_objects)
        track.timestamps.append(now)

        # 2. Texture & Screen Artifact Extraction
        img_for_texture = aligned_face if aligned_face is not None else raw_face_crop
        texture_sharpness = 0.0
        color_dispersion = 0.0
        if img_for_texture is not None and img_for_texture.size > 0:
            gray = cv2.cvtColor(cv2.resize(img_for_texture, (112, 112)), cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            texture_sharpness = lap_var

            ycrcb = cv2.cvtColor(cv2.resize(img_for_texture, (112, 112)), cv2.COLOR_BGR2YCrCb)
            color_dispersion = float(np.std(ycrcb[:, :, 1:]) + np.std(img_for_texture))

        # 3. Eye-Region Photometric Extraction per Track
        eye_grad, eye_contrast = 0.0, 0.0
        if raw_face_crop is not None and landmarks is not None and len(landmarks) >= 10:
            eye_grad, eye_contrast = self._extract_eye_photometric_gradient(raw_face_crop, landmarks)
            track.eye_gradient_history.append(eye_grad)
            self._update_track_blink_machine(track, eye_grad)

        # 4. Normalized Non-Rigid Motion & Geometric Ratios per Track
        norm_motion_std = 0.0
        ratio_motion_std = 0.0
        is_rigid_2d = True

        if landmarks is not None and len(landmarks) >= 10:
            norm_lm = self._compute_normalized_landmarks(landmarks)
            track.norm_landmarks_history.append(norm_lm)

            geom_ratios = self._compute_distance_ratios(landmarks)
            track.distance_ratios_history.append(geom_ratios)

            if len(track.norm_landmarks_history) >= 6:
                hist_arr = np.array(track.norm_landmarks_history)
                norm_motion_std = float(np.mean(np.std(hist_arr, axis=0)))

                ratio_arr = np.array(track.distance_ratios_history)
                ratio_motion_std = float(np.mean(np.std(ratio_arr, axis=0)))

                # 2D rigid motion (phone photo / paper print) has near-zero ratio motion
                is_rigid_2d = ratio_motion_std < self.ratio_motion_threshold

        buffer_len = len(track.norm_landmarks_history)
        warmup_active = buffer_len < self.warmup_frames

        # 5. Raw Frame-Level Multi-Cue Evaluation (FAIL-SECURE)

        # Case A: Physically Enclosed by Screen Object (Cell Phone / Monitor Replay)
        if track.is_inside_screen:
            raw_is_live = False
            raw_confidence = 0.08
            raw_attack_type = AttackType.REPLAY
            raw_decision_tag = f"SCREEN_REPLAY_{track.screen_object_label.upper()}"

        # Case B: Buffer Warming Up -> FAIL-SECURE UNVERIFIED
        elif warmup_active:
            raw_is_live = False
            raw_confidence = 0.40
            raw_attack_type = AttackType.UNKNOWN
            raw_decision_tag = "WARMUP_UNVERIFIED"

        # Case C: Confirmed Natural Eye Blink -> BONA-FIDE LIVE
        elif track.blink_count >= 1:
            raw_is_live = True
            raw_confidence = min(0.96, max(0.85, 0.82 + min(track.blink_count * 0.05, 0.10) + ratio_motion_std * 2.0))
            raw_attack_type = AttackType.BONAFIDE
            raw_decision_tag = "BONAFIDE_BLINK_VERIFIED"

        # Case D: Active 3D Non-Rigid Facial Dynamics & Micro-motion observed
        elif track.consecutive_frames >= self.warmup_frames and (ratio_motion_std >= self.ratio_motion_threshold or norm_motion_std >= self.rigid_motion_threshold):
            raw_is_live = True
            raw_confidence = min(0.92, 0.80 + max(ratio_motion_std, norm_motion_std) * 2.0)
            raw_attack_type = AttackType.BONAFIDE
            raw_decision_tag = "BONAFIDE_3D_MOTION_VERIFIED"

        # Case E: Static 2D Planar Presentation (Zero Blinks, Rigid Geometry) -> PRESENTATION ATTACK DETECTED
        else:
            raw_is_live = False
            if texture_sharpness > 650.0 or color_dispersion < 25.0:
                raw_attack_type = AttackType.REPLAY
                raw_confidence = 0.15
                raw_decision_tag = "RIGID_SCREEN_REPLAY_SPOOF"
            else:
                raw_attack_type = AttackType.PRINT
                raw_confidence = 0.20
                raw_decision_tag = "RIGID_PRINT_PHOTO_SPOOF"

        elapsed_seconds = time.time() - track.first_seen_time

        # 6. Temporal Decision Consistency & Hysteresis Filtering per Track
        stable_state, stable_conf, stable_atk, stable_tag = track.stabilizer.update(
            raw_is_live=raw_is_live,
            raw_confidence=raw_confidence,
            raw_attack_type=raw_attack_type,
            raw_tag=raw_decision_tag,
            is_screen=track.is_inside_screen,
            elapsed_seconds=elapsed_seconds,
        )

        is_live_final = (stable_state == PADStabilizedState.STABLE_LIVE)
        is_timeout_final = (stable_state == PADStabilizedState.UNVERIFIED_TIMEOUT)
        is_warmup_final = (not is_timeout_final and stable_state in (PADStabilizedState.WARMUP, PADStabilizedState.UNVERIFIED_PENDING))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Record diagnostic snapshot on track
        track.last_diagnostics = DiagnosticSignals(
            track_id=track.track_id,
            buffer_len=buffer_len,
            eye_gradient_energy=eye_grad,
            eye_contrast=eye_contrast,
            blink_count=track.blink_count,
            blink_state=track.blink_state,
            norm_motion_std=norm_motion_std,
            ratio_motion_std=ratio_motion_std,
            is_rigid_2d=is_rigid_2d,
            is_inside_screen=track.is_inside_screen,
            screen_label=track.screen_object_label,
            texture_sharpness=texture_sharpness,
            color_dispersion=color_dispersion,
            warmup_active=is_warmup_final,
            stable_state=stable_state.value,
            history_symbols=track.stabilizer.history_symbols,
        )

        return PADResult(
            is_live=is_live_final,
            confidence=float(stable_conf),
            attack_type=stable_atk,
            inference_latency_ms=elapsed_ms,
            model_name=self.model_name,
            raw_score=norm_motion_std,
            metadata={
                "track_id": track.track_id,
                "tag": stable_tag,
                "raw_tag": raw_decision_tag,
                "stable_state": stable_state.value,
                "history_symbols": track.stabilizer.history_symbols,
                "consecutive_live": track.stabilizer.consecutive_live,
                "consecutive_spoof": track.stabilizer.consecutive_spoof,
                "warmup_active": is_warmup_final,
                "is_timeout": is_timeout_final,
                "elapsed_seconds": elapsed_seconds,
                "timeout_seconds": track.stabilizer.timeout_seconds,
                "norm_motion_std": norm_motion_std,
                "ratio_motion_std": ratio_motion_std,
                "blink_count": track.blink_count,
                "is_rigid_2d": is_rigid_2d,
                "is_inside_screen": track.is_inside_screen,
                "screen_label": track.screen_object_label,
                "texture_sharpness": texture_sharpness,
                "buffer_len": buffer_len,
            },
        )
