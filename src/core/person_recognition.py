"""
Person Recognition Module (OpenCV YuNet + SFace)
------------------------------------------------
High-accuracy, lightweight face detection (YuNet) and deep-learning facial
embedding recognition (SFace) operating natively with OpenCV.
Supports local face registration, database persistence, real-time live video
annotation, unknown person identification, and debounced security alert integration.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    FACE_DETECTION_SCORE_THRESHOLD,
    FACE_COSINE_SIMILARITY_THRESHOLD,
    FACE_AMBIGUOUS_SIMILARITY_THRESHOLD,
    FACE_TEMPORAL_CONFIRM_FRAMES,
    PERSON_ALERT_COOLDOWN_SECONDS,
    FACES_DIR,
    DEMO_FACES_DIR,
)
from src.core.base_engine import BaseVisionEngine
from src.core.pad.joint_decision import JointAccessEvaluator, JointDecision, JointStatus
from src.core.pad.pad_manager import PADManager
from src.database.db_manager import DatabaseManager

logger = logging.getLogger("WatchGuardVision.PersonRecognition")


class PersonRecognitionEngine(BaseVisionEngine):
    """
    OpenCV-based Person Recognition Engine using YuNet Face Detector,
    SFace Feature Recognizer, and pluggable Liveness / PAD verification.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        pad_manager: Optional[PADManager] = None,
    ):
        self.db = db_manager or DatabaseManager()
        self.pad_manager = pad_manager or PADManager()
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._is_ready = False
        self._last_input_size: Tuple[int, int] = (640, 480)
        self._known_faces: List[Dict[str, Any]] = []
        self._last_alert_timestamps: Dict[str, float] = {}
        self._previous_person_states: Dict[str, str] = {}

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def known_face_count(self) -> int:
        return len(self._known_faces)

    def initialize(self, auto_seed_demo: bool = False) -> bool:
        """
        Initializes YuNet detector and SFace recognizer, loads registered faces
        from the database. Demo identities are seeded only if auto_seed_demo=True.
        """
        try:
            if not YUNET_MODEL_PATH.exists() or not SFACE_MODEL_PATH.exists():
                logger.error(f"Face model files missing: {YUNET_MODEL_PATH} or {SFACE_MODEL_PATH}")
                self._is_ready = False
                return False

            self._detector = cv2.FaceDetectorYN.create(
                model=str(YUNET_MODEL_PATH),
                config="",
                input_size=self._last_input_size,
                score_threshold=FACE_DETECTION_SCORE_THRESHOLD,
                nms_threshold=0.3,
                top_k=5000,
            )

            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(SFACE_MODEL_PATH),
                config="",
            )

            # Initialize PAD Manager
            self.pad_manager.initialize()

            self._is_ready = True
            logger.info("YuNet Face Detector, SFace Recognizer & PAD Subsystem initialized successfully.")

            # Load registered faces from database
            self.reload_known_faces()

            # Seed default demo identity only if explicitly requested
            if auto_seed_demo and len(self._known_faces) == 0:
                self.seed_demo_identities()

            return True
        except Exception as e:
            logger.error(f"Failed to initialize PersonRecognitionEngine: {e}")
            self._is_ready = False
            return False

    def reload_known_faces(self) -> None:
        """Loads and deserializes active face registration embeddings from SQLite."""
        self._known_faces.clear()
        try:
            registrations = self.db.get_active_face_registrations()
            for row in registrations:
                try:
                    encoding_bytes = row["face_encoding"]
                    if encoding_bytes:
                        feat = np.frombuffer(encoding_bytes, dtype=np.float32).reshape(1, -1)
                        self._known_faces.append(
                            {
                                "registration_id": row["id"],
                                "user_id": row["user_id"],
                                "username": row["username"],
                                "name": row["full_name"],
                                "role": row["role"],
                                "access_level": row["access_level"],
                                "image_path": row.get("image_path", ""),
                                "feature": feat,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error loading face registration record {row.get('id')}: {e}")

            logger.info(f"Loaded {len(self._known_faces)} registered face profile(s) from database.")
        except Exception as e:
            logger.error(f"Failed to reload registered faces: {e}")

    def seed_demo_identities(self) -> bool:
        """Registers Hunter from demo portrait if available (called explicitly on demand)."""
        portrait_path = DEMO_FACES_DIR / "officer_alice.jpg"
        if portrait_path.exists():
            portrait_img = cv2.imread(str(portrait_path))
            if portrait_img is not None:
                # Ensure user exists
                user = self.db.get_user_by_username("hunter")
                if not user:
                    user_id = self.db.create_user(
                        username="hunter",
                        full_name="Hunter",
                        role="Security Lead",
                        access_level=3,
                    )
                else:
                    user_id = user["id"]

                success, msg = self.register_face_from_image(
                    user_id=user_id,
                    image_bgr=portrait_img,
                    source_image_path=str(portrait_path),
                )
                if success:
                    logger.info("Successfully seeded authorized face profile: Hunter.")
                    return True
        return False

    def register_face_from_image(
        self,
        user_id: int,
        image_bgr: np.ndarray,
        source_image_path: str = "",
    ) -> Tuple[bool, str]:
        """
        Extracts facial embedding from provided image, saves face crop,
        and registers in the database.
        """
        if not self._is_ready or self._detector is None or self._recognizer is None:
            return False, "Person recognition engine is not initialized."

        try:
            h, w = image_bgr.shape[:2]
            self._detector.setInputSize((w, h))
            self._last_input_size = (w, h)
            _, faces = self._detector.detect(image_bgr)

            if faces is None or len(faces) == 0:
                return False, "No human face detected in the provided image."

            # Select the largest face by area
            primary_face = max(faces, key=lambda f: f[2] * f[3])
            aligned_face = self._recognizer.alignCrop(image_bgr, primary_face)
            feature = self._recognizer.feature(aligned_face)  # (1, 128) float32

            # Save face crop image
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            crop_filename = f"user_{user_id}_{timestamp_str}.jpg"
            crop_path = str(FACES_DIR / crop_filename)
            cv2.imwrite(crop_path, aligned_face)

            # Persist to database
            encoding_bytes = feature.astype(np.float32).tobytes()
            self.db.register_face(
                user_id=user_id,
                face_encoding=encoding_bytes,
                image_path=crop_path or source_image_path,
            )

            # Refresh in-memory list
            self.reload_known_faces()
            return True, f"Face registered successfully for user ID {user_id}."
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False, f"Registration failed: {e}"

    def process_frame(
        self,
        frame: np.ndarray,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Performs face detection, identification, per-track presentation attack detection (PAD),
        and spatial object fusion against known profiles.
        """
        if not self._is_ready or self._detector is None or self._recognizer is None or frame is None:
            return {
                "detected": False,
                "count": 0,
                "people": [],
                "has_unknown": False,
                "has_authorized": False,
                "primary_label": "Standby",
            }

        h, w = frame.shape[:2]
        if (w, h) != self._last_input_size:
            self._detector.setInputSize((w, h))
            self._last_input_size = (w, h)

        try:
            _, faces = self._detector.detect(frame)
        except Exception as e:
            logger.error(f"Face detection inference error: {e}")
            faces = None

        if faces is None or len(faces) == 0:
            return {
                "detected": False,
                "count": 0,
                "people": [],
                "has_unknown": False,
                "has_authorized": False,
                "primary_label": "No Person",
            }

        people = []
        has_unknown = False
        has_authorized = False
        has_spoof = False

        for idx, face in enumerate(faces):
            box = face[0:4].astype(int)
            x, y, bw, bh = max(0, box[0]), max(0, box[1]), box[2], box[3]
            det_score = float(face[-1])  # Face Detection Confidence (YuNet)
            landmarks = face[4:14]       # 5-point landmarks (YuNet)

            # Extract raw face crop
            raw_crop = frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]

            # Extract 128-d embedding (SFace) and match against known faces
            aligned = None
            if self._known_faces:
                try:
                    aligned = self._recognizer.alignCrop(frame, face)
                    feat = self._recognizer.feature(aligned)
                except Exception:
                    feat = None

                best_sim = -1.0
                best_match: Optional[Dict[str, Any]] = None

                if feat is not None:
                    for known in self._known_faces:
                        sim = float(self._recognizer.match(feat, known["feature"], cv2.FaceRecognizerSF_FR_COSINE))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = known

                # 3-Tier Security Identity Classification
                if best_match is not None and best_sim >= FACE_COSINE_SIMILARITY_THRESHOLD:
                    # TIER 1: Confirmed Authorized Identity (sim >= 0.55)
                    is_known = True
                    is_ambiguous = False
                    name = best_match["name"]
                    user_id = best_match["user_id"]
                    role = best_match["role"]
                    status_label = "AUTHORIZED"
                    # Calibrated Match %: maps [0.55, 1.0] -> [75%, 100%]
                    match_pct = int(min(100, max(75, 75 + (best_sim - 0.55) / 0.45 * 25)))
                elif best_match is not None and best_sim >= FACE_AMBIGUOUS_SIMILARITY_THRESHOLD:
                    # TIER 2: Ambiguous / Low-Confidence Match (0.38 <= sim < 0.55) -> FAIL-SECURE TO UNKNOWN
                    is_known = False
                    is_ambiguous = True
                    name = "UNKNOWN PERSON"
                    user_id = None
                    role = "Low-Confidence Match (Unverified)"
                    has_unknown = True
                    status_label = "UNVERIFIED"
                    match_pct = int(max(0, best_sim * 100))
                else:
                    # TIER 3: Unregistered Visitor (sim < 0.38)
                    is_known = False
                    is_ambiguous = False
                    name = "UNKNOWN PERSON"
                    user_id = None
                    role = "Unregistered Visitor"
                    has_unknown = True
                    status_label = "UNKNOWN"
                    match_pct = int(max(0, best_sim * 100)) if best_sim > 0 else 0
            else:
                # Fast path when no face profiles exist in database
                is_known = False
                is_ambiguous = False
                name = "UNKNOWN PERSON"
                user_id = None
                role = "Unregistered Visitor"
                has_unknown = True
                status_label = "UNKNOWN"
                best_sim = 0.0
                match_pct = 0

            # --- Presentation Attack Detection (PAD / Liveness) with Per-Track Isolation ---
            pad_result = self.pad_manager.evaluate_presentation(
                raw_face_crop=raw_crop,
                aligned_face=aligned,
                landmarks=landmarks,
                face_bbox=(x, y, bw, bh),
                detected_objects=detected_objects,
            )

            track_id = int(pad_result.metadata.get("track_id", idx + 1))

            identity_dict = {
                "name": name,
                "is_known": is_known,
                "is_ambiguous": is_ambiguous,
                "similarity": max(0.0, best_sim),
                "role": role,
                "match_pct": match_pct,
            }

            # --- Joint Identity ⊗ Liveness Access Evaluation ---
            joint_decision = JointAccessEvaluator.evaluate(identity_dict, pad_result)

            # Track State Transition for Traceability (Logged only when state changes)
            person_track_key = f"{name}#T{track_id}"
            prev_state = self._previous_person_states.get(person_track_key, "NONE")
            curr_state = joint_decision.status.value
            if prev_state != curr_state:
                transition_str = f"{prev_state} -> {curr_state}" if prev_state != "NONE" else f"INITIAL_OBSERVATION -> {curr_state}"
                logger.debug(
                    f"[STATE_TRANSITION Track #{track_id}] Identity='{name}' (Sim={match_pct}%): {transition_str} | "
                    f"Access={joint_decision.status.display_label} (Risk={joint_decision.risk_level.value})"
                )
                self._previous_person_states[person_track_key] = curr_state

            people.append(
                {
                    "track_id": track_id,
                    "name": name,
                    "is_known": is_known,
                    "is_ambiguous": is_ambiguous,
                    "status_label": status_label,
                    "confidence": det_score,             # Face detection confidence (YuNet)
                    "similarity": max(0.0, best_sim),    # Cosine similarity (SFace)
                    "match_pct": match_pct,              # Calibrated match percentage
                    "bbox": (x, y, bw, bh),
                    "user_id": user_id,
                    "role": role,
                    "pad_result": pad_result,            # Standardized PADResult
                    "joint_decision": joint_decision,    # Standardized JointDecision
                    "is_live": pad_result.is_live,
                    "is_spoof": joint_decision.is_spoof,
                    "liveness_confidence": pad_result.confidence,
                    "attack_type": pad_result.attack_type.value,
                    "joint_status": joint_decision.status.value,
                }
            )

        # STRICT ACCESS INVARIANT: has_authorized is ONLY true if JointStatus is strictly AUTHORIZED
        has_authorized = any(p["joint_decision"].status == JointStatus.AUTHORIZED for p in people)
        has_verifying = any(p["joint_decision"].status == JointStatus.LIVENESS_VERIFYING for p in people)
        has_spoof = any(p["joint_decision"].is_spoof for p in people)
        has_unknown = any(not p["is_known"] and not p["is_spoof"] for p in people)

        # Compute summary label
        if len(people) == 1:
            p = people[0]
            j_status = p["joint_decision"].status
            if p["is_spoof"]:
                primary_label = f"Spoof Attack Rejected: {p['name']}" if p["is_known"] else "Spoof Attack Detected"
            elif j_status == JointStatus.LIVENESS_VERIFYING:
                primary_label = f"Identity Matched: {p['name']} — Liveness Verifying"
            elif j_status == JointStatus.AUTHORIZED:
                primary_label = f"Authorized: {p['name']} ({p['match_pct']}%)"
            elif p.get("is_ambiguous", False):
                primary_label = f"Unverified Face ({p['match_pct']}% sim)"
            else:
                primary_label = "Unknown Person"
        elif len(people) > 1:
            spoof_cnt = sum(1 for p in people if p["is_spoof"])
            verifying_cnt = sum(1 for p in people if p["joint_decision"].status == JointStatus.LIVENESS_VERIFYING)
            unknown_cnt = sum(1 for p in people if not p["is_known"] and not p["is_spoof"])
            auth_cnt = sum(1 for p in people if p["joint_decision"].status == JointStatus.AUTHORIZED)
            primary_label = f"{len(people)} Persons ({auth_cnt} Auth, {verifying_cnt} Verifying, {unknown_cnt} Unknown, {spoof_cnt} Spoof)"
        else:
            primary_label = "No Person"

        return {
            "detected": len(people) > 0,
            "count": len(people),
            "people": people,
            "has_unknown": has_unknown,
            "has_authorized": has_authorized,
            "has_verifying": has_verifying,
            "has_spoof": has_spoof,
            "primary_label": primary_label,
            "pad_model": self.pad_manager.active_model_name,
        }

    def should_trigger_alert(
        self,
        person_key: str,
        cooldown_seconds: float = PERSON_ALERT_COOLDOWN_SECONDS,
    ) -> bool:
        """
        Debounce helper: Returns True if enough time has elapsed since the
        last logged event for this person/anomaly type.
        """
        now = time.time()
        last_time = self._last_alert_timestamps.get(person_key, 0.0)
        if (now - last_time) >= cooldown_seconds:
            self._last_alert_timestamps[person_key] = now
            return True
        return False

    def draw_annotations(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        context_decision: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Renders HUD overlays, high-tech bracketed bounding boxes, track badges,
        PAD liveness metrics, and zone/time-aware access states.
        """
        if image is None or image.size == 0 or not results.get("detected", False):
            return image

        annotated = image.copy()
        h_frame, w_frame = annotated.shape[:2]

        for p in results.get("people", []):
            x, y, w, h = p["bbox"]
            track_id = p.get("track_id", 1)
            is_known = p["is_known"]
            is_ambiguous = p.get("is_ambiguous", False)
            name = p["name"]
            sim = p["similarity"]
            match_pct = p.get("match_pct", int(sim * 100))
            det_pct = int(p.get("confidence", 0.0) * 100)
            role = p.get("role", "")
            joint_dec: Optional[JointDecision] = p.get("joint_decision")
            pad_res: Optional[PADResult] = p.get("pad_result")

            is_spoof = p.get("is_spoof", False)
            is_timeout = pad_res.metadata.get("is_timeout", False) if pad_res else False
            is_warmup = (pad_res.metadata.get("warmup_active", False) if pad_res else False) and not is_timeout
            live_pct = int(p.get("liveness_confidence", 0.90) * 100)
            elapsed_s = float(pad_res.metadata.get("elapsed_seconds", 0.0)) if pad_res else 0.0

            # --- 1. Presentation Attack Rendering (RED / ORANGE) ---
            if is_spoof:
                if is_known:
                    # Critical Spoof Impersonation of Registered Profile (RED)
                    box_color = (40, 50, 240)       # BGR: Crimson Red
                    bg_badge = (20, 20, 180)
                    badge_text = f"[T#{track_id}] SPOOF REJECTED: {name.upper()}"
                    score_text = f"IMPERSONATION ATTACK | ACCESS: DENIED | LIVE: {live_pct}%"
                else:
                    # Unregistered Presentation Attack (ORANGE)
                    box_color = (0, 115, 249)       # BGR: Bright Orange
                    bg_badge = (0, 70, 180)
                    badge_text = f"[T#{track_id}] SPOOF DETECTED [ALERT]"
                    score_text = f"UNREGISTERED SPOOF | ACCESS: DENIED | LIVE: {live_pct}%"

            # --- 2. PAD Verification Window Timeout (AMBER / ACCESS: DENIED) ---
            elif is_timeout or (joint_dec and joint_dec.status == JointStatus.UNVERIFIED_TIMEOUT):
                box_color = (0, 190, 255)           # BGR: Amber / Golden Yellow
                bg_badge = (0, 100, 180)
                if is_known:
                    badge_text = f"[T#{track_id}] {name.upper()} (PAD TIMEOUT)"
                    score_text = f"ACCESS: DENIED | UNVERIFIED (>5.0s) | MATCH: {match_pct}%"
                else:
                    badge_text = f"[T#{track_id}] UNVERIFIED (PAD TIMEOUT)"
                    score_text = f"ACCESS: DENIED | UNVERIFIED (>5.0s) | DET: {det_pct}%"

            # --- 3. Liveness Verifying / Warmup Phase (AMBER / ACCESS: PENDING) ---
            elif is_warmup or (joint_dec and joint_dec.status == JointStatus.LIVENESS_VERIFYING):
                box_color = (0, 190, 255)           # BGR: Amber / Golden Yellow
                bg_badge = (0, 100, 180)
                timer_tag = f" ({elapsed_s:.1f}s/5.0s)" if elapsed_s > 0 else ""
                if is_known:
                    badge_text = f"[T#{track_id}] {name.upper()} (VERIFYING PAD)"
                    score_text = f"ACCESS: PENDING | MATCH: {match_pct}% | VERIFYING{timer_tag}"
                else:
                    badge_text = f"[T#{track_id}] VERIFYING LIVENESS"
                    score_text = f"ACCESS: PENDING | ANALYZING BUFFER{timer_tag} | DET: {det_pct}%"

            # --- 4. Bona-fide Registered Presentation (GREEN or ORANGE for After-Hours Restricted) ---
            elif is_known and joint_dec and joint_dec.status == JointStatus.AUTHORIZED:
                is_after_hours_denied = False
                if context_decision:
                    dec_name = getattr(context_decision, "decision", "")
                    if dec_name == "AFTER_HOURS_RESTRICTED_ACCESS_DENIED":
                        is_after_hours_denied = True

                if is_after_hours_denied:
                    box_color = (0, 115, 249)       # BGR: Bright Orange
                    bg_badge = (0, 70, 180)
                    badge_text = f"[T#{track_id}] IDENTIFIED: {name.upper()}"
                    score_text = f"ACCESS: DENIED - AFTER HOURS | LIVE: {live_pct}% | MATCH: {match_pct}%"
                else:
                    box_color = (100, 220, 16)          # BGR: Emerald Green
                    bg_badge = (30, 140, 10)
                    badge_text = f"[T#{track_id}] AUTHORIZED: {name.upper()}"
                    score_text = f"ACCESS: GRANTED | LIVE: {live_pct}% | MATCH: {match_pct}%"

            # --- 5. Ambiguous Match (AMBER) ---
            elif is_ambiguous:
                box_color = (0, 190, 255)           # BGR: Amber / Golden Yellow
                bg_badge = (0, 100, 180)
                badge_text = f"[T#{track_id}] UNVERIFIED [SIM: {match_pct}%]"
                score_text = f"ACCESS: DENIED | NOT AUTH | DET: {det_pct}%"

            # --- 6. Unregistered Live Visitor (RED/AMBER) ---
            else:
                box_color = (40, 50, 240)           # BGR: Crimson Red
                bg_badge = (20, 20, 180)
                badge_text = f"[T#{track_id}] UNKNOWN PERSON [ALERT]"
                score_text = f"ACCESS: DENIED | LIVE: {live_pct}% | UNREGISTERED VISITOR"

            # 1. High-Tech Corner Brackets
            corner_len = min(20, w // 4, h // 4)
            th = 2
            # Top-Left
            cv2.line(annotated, (x, y), (x + corner_len, y), box_color, th + 1)
            cv2.line(annotated, (x, y), (x, y + corner_len), box_color, th + 1)
            # Top-Right
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), box_color, th + 1)
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), box_color, th + 1)
            # Bottom-Left
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), box_color, th + 1)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), box_color, th + 1)
            # Bottom-Right
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), box_color, th + 1)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), box_color, th + 1)

            # Subtle bounding box outline
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 1)

            # 2. Header Tag Badge
            badge_y1 = max(0, y - 32)
            badge_y2 = y
            badge_w = max(185, len(score_text) * 7 + 10)
            cv2.rectangle(annotated, (x, badge_y1), (x + badge_w, badge_y2), bg_badge, -1)
            cv2.rectangle(annotated, (x, badge_y1), (x + badge_w, badge_y2), box_color, 1)

            cv2.putText(
                annotated,
                badge_text,
                (x + 6, y - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                score_text,
                (x + 6, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (220, 240, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def release(self) -> None:
        """Releases model instances."""
        self._detector = None
        self._recognizer = None
        self._is_ready = False
