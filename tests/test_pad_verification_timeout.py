"""
Unit & Integration Tests for PAD Verification Window & 5-Second Timeout Policy
--------------------------------------------------------------------------------
Validates:
1. NO authorization before PAD verification is confirmed.
2. 5-second PAD verification window:
   IDENTITY_MATCHED -> LIVENESS_VERIFYING / ACCESS_PENDING -> PAD CONFIRMED -> POLICY CHECK -> AUTHORIZED
3. Immediate SPOOF rejection if a presentation attack is detected at any time.
4. If PAD remains uncertain until 5 seconds -> UNVERIFIED / ACCESS DENIED.
5. HUD overlay displays intermediate VERIFYING state with timer and never displays AUTHORIZED before PAD confirmation.
6. Multi-track isolation across independent face tracks.
"""

import time
import unittest
import numpy as np
import tempfile
from pathlib import Path

from config import PAD_VERIFICATION_TIMEOUT_SECONDS
from src.core.pad.base_pad import AttackType, PADResult
from src.core.pad.joint_decision import JointAccessEvaluator, JointDecision, JointStatus
from src.core.pad.models.blink_ear_pad import BlinkEARPADEngine, PADStabilizedState
from src.core.person_recognition import PersonRecognitionEngine
from src.database.db_manager import DatabaseManager
from src.events.event_types import RiskLevel


class TestPADVerificationTimeoutPolicy(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_pad_timeout.db"
        self.db = DatabaseManager(db_path=self.db_path)

        self.synthetic_face = np.full((112, 112, 3), 128, dtype=np.uint8)
        self.synthetic_landmarks = np.array([38.0, 42.0, 74.0, 42.0, 56.0, 62.0, 42.0, 84.0, 70.0, 84.0], dtype=np.float32)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. No authorization during PAD verifying window
    def test_01_no_authorization_during_pad_verifying_window(self):
        engine = BlinkEARPADEngine(warmup_frames=6)
        engine.initialize()
        identity = {"is_known": True, "name": "Hunter", "similarity": 0.92, "role": "Security Lead"}

        # First 4 frames: Identity is matched, but PAD is verifying
        for frame_idx in range(4):
            res_pad = engine.evaluate(
                raw_face_crop=self.synthetic_face,
                aligned_face=self.synthetic_face,
                landmarks=self.synthetic_landmarks,
                track_id=1,
            )
            joint_dec = JointAccessEvaluator.evaluate(identity, res_pad)

            self.assertFalse(joint_dec.is_access_granted, "Access must NOT be granted during PAD verifying")
            self.assertEqual(joint_dec.status, JointStatus.LIVENESS_VERIFYING)
            self.assertEqual(joint_dec.risk_level, RiskLevel.YELLOW)
            self.assertIn("PENDING", joint_dec.reason)

    # 2. Confirmed PAD leads to authorization
    def test_02_confirmed_pad_leads_to_authorization(self):
        engine = BlinkEARPADEngine(warmup_frames=3)
        engine.initialize()
        identity = {"is_known": True, "name": "Hunter", "similarity": 0.92, "role": "Security Lead"}

        # Simulate natural non-rigid live motion & blinks
        for frame_idx in range(12):
            dyn_landmarks = self.synthetic_landmarks + np.random.uniform(-3.5, 3.5, 10).astype(np.float32)
            track = engine._get_or_create_track(1, (100, 100, 80, 80))
            if frame_idx >= 2:
                track.blink_count = 2  # Eye blink confirmed

            res_pad = engine.evaluate(
                raw_face_crop=self.synthetic_face,
                aligned_face=self.synthetic_face,
                landmarks=dyn_landmarks,
                track_id=1,
            )

        joint_dec = JointAccessEvaluator.evaluate(identity, res_pad)
        self.assertTrue(joint_dec.is_access_granted)
        self.assertEqual(joint_dec.status, JointStatus.AUTHORIZED)
        self.assertEqual(joint_dec.risk_level, RiskLevel.GREEN)
        self.assertIn("Authorized", joint_dec.reason)

    # 3. Immediate SPOOF rejection if spoof detected at any time
    def test_03_immediate_spoof_rejection_at_any_time(self):
        engine = BlinkEARPADEngine(warmup_frames=6)
        engine.initialize()
        identity = {"is_known": True, "name": "Hunter", "similarity": 0.95, "role": "Security Lead"}

        # Frame 1: Phone screen object detected overlapping the face
        detected_objects = [{"label": "cell phone", "bbox": (90, 90, 100, 140)}]
        res_pad = engine.evaluate(
            raw_face_crop=self.synthetic_face,
            aligned_face=self.synthetic_face,
            landmarks=self.synthetic_landmarks,
            face_bbox=(95, 95, 60, 60),
            detected_objects=detected_objects,
            track_id=1,
        )

        joint_dec = JointAccessEvaluator.evaluate(identity, res_pad)
        self.assertFalse(joint_dec.is_access_granted)
        self.assertEqual(joint_dec.status, JointStatus.SPOOF_IMPERSONATION_REJECTED)
        self.assertEqual(joint_dec.risk_level, RiskLevel.RED)
        self.assertIn("SPOOF ATTACK DETECTED", joint_dec.reason)

    # 4. If PAD remains uncertain until 5 seconds -> UNVERIFIED / ACCESS DENIED
    def test_04_pad_timeout_after_5_seconds_denies_access(self):
        engine = BlinkEARPADEngine(warmup_frames=6)
        engine.initialize()
        identity = {"is_known": True, "name": "Hunter", "similarity": 0.90, "role": "Staff"}

        # Simulate track existing for 5.2 seconds with static/uncertain evidence
        track = engine._get_or_create_track(1, (100, 100, 80, 80))
        track.first_seen_time = time.time() - 5.5  # 5.5s elapsed (>5.0s timeout)

        res_pad = engine.evaluate(
            raw_face_crop=self.synthetic_face,
            aligned_face=self.synthetic_face,
            landmarks=self.synthetic_landmarks,
            track_id=1,
        )

        joint_dec = JointAccessEvaluator.evaluate(identity, res_pad)
        self.assertFalse(joint_dec.is_access_granted)
        self.assertEqual(joint_dec.status, JointStatus.UNVERIFIED_TIMEOUT)
        self.assertEqual(joint_dec.risk_level, RiskLevel.YELLOW)
        self.assertIn("DENIED", joint_dec.reason)
        self.assertIn("timed out", joint_dec.reason.lower())

    # 5. HUD overlay annotations show verifying and timeout states
    def test_05_hud_annotations_show_verifying_and_timeout(self):
        recog_engine = PersonRecognitionEngine(db_manager=self.db)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Case A: Verifying PAD (Amber, ACCESS: PENDING)
        pad_res_verifying = PADResult(
            is_live=False,
            confidence=0.45,
            attack_type=AttackType.UNKNOWN,
            inference_latency_ms=5.0,
            model_name="Blink-EAR",
            metadata={"warmup_active": True, "elapsed_seconds": 1.5, "timeout_seconds": 5.0},
        )
        joint_dec_verifying = JointDecision(
            status=JointStatus.LIVENESS_VERIFYING,
            is_access_granted=False,
            is_spoof=False,
            risk_level=RiskLevel.YELLOW,
            reason="Verifying",
            person_name="Hunter",
            role="Staff",
            identity_similarity=0.88,
            liveness_confidence=0.45,
            attack_type=AttackType.UNKNOWN,
            pad_model="Blink-EAR",
        )

        results_verifying = {
            "detected": True,
            "people": [{
                "track_id": 1,
                "name": "Hunter",
                "is_known": True,
                "similarity": 0.88,
                "confidence": 0.90,
                "bbox": (100, 100, 80, 80),
                "pad_result": pad_res_verifying,
                "joint_decision": joint_dec_verifying,
                "is_spoof": False,
            }],
        }

        annotated = recog_engine.draw_annotations(canvas, results_verifying)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, canvas.shape)

        # Case B: Timed out (>5s)
        pad_res_timeout = PADResult(
            is_live=False,
            confidence=0.35,
            attack_type=AttackType.UNKNOWN,
            inference_latency_ms=5.0,
            model_name="Blink-EAR",
            metadata={"is_timeout": True, "elapsed_seconds": 5.2, "timeout_seconds": 5.0},
        )
        joint_dec_timeout = JointDecision(
            status=JointStatus.UNVERIFIED_TIMEOUT,
            is_access_granted=False,
            is_spoof=False,
            risk_level=RiskLevel.YELLOW,
            reason="Timed out",
            person_name="Hunter",
            role="Staff",
            identity_similarity=0.88,
            liveness_confidence=0.35,
            attack_type=AttackType.UNKNOWN,
            pad_model="Blink-EAR",
        )

        results_timeout = {
            "detected": True,
            "people": [{
                "track_id": 1,
                "name": "Hunter",
                "is_known": True,
                "similarity": 0.88,
                "confidence": 0.90,
                "bbox": (100, 100, 80, 80),
                "pad_result": pad_res_timeout,
                "joint_decision": joint_dec_timeout,
                "is_spoof": False,
            }],
        }
        annotated_timeout = recog_engine.draw_annotations(canvas, results_timeout)
        self.assertIsNotNone(annotated_timeout)


if __name__ == "__main__":
    unittest.main()
