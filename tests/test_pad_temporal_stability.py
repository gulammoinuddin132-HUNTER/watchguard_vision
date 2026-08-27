"""
PAD Temporal Decision Stability & Hysteresis Test Suite
-------------------------------------------------------
Verifies:
1. Never authorize on a single PAD-positive frame.
2. Consecutive evidence requirement before granting LIVE or confirming SPOOF.
3. Oscillating evidence (L, S, L, S) remains strictly in LIVENESS_VERIFYING / ACCESS PENDING.
4. Hysteresis prevents single-frame noise from flipping AUTHORIZED <-> SPOOF.
5. Immediate screen enclosure override to STABLE_SPOOF.
6. Simultaneous multi-face independent temporal stabilizers (Track 1: Real vs Track 2: Phone).
7. Structured telemetry logging fields.
"""

import unittest
import numpy as np

from src.core.pad.base_pad import AttackType
from src.core.pad.joint_decision import JointAccessEvaluator, JointStatus
from src.core.pad.models.blink_ear_pad import BlinkEARPADEngine, PADStabilizedState, TrackedPADStabilizer
from src.core.person_recognition import PersonRecognitionEngine
from src.core.context_engine import ContextEngine
from src.database.db_manager import DatabaseManager


class TestPADTemporalStability(unittest.TestCase):

    def setUp(self):
        self.db = DatabaseManager()
        self.engine = PersonRecognitionEngine(db_manager=self.db)
        self.engine.initialize(auto_seed_demo=True)
        self.context_engine = ContextEngine()

    def _create_mock_landmarks(self, center: tuple = (150, 150), scale: float = 40.0) -> np.ndarray:
        cx, cy = center
        re = (cx - scale * 0.5, cy - scale * 0.3)
        le = (cx + scale * 0.5, cy - scale * 0.3)
        nt = (cx, cy)
        rc = (cx - scale * 0.35, cy + scale * 0.4)
        lc = (cx + scale * 0.35, cy + scale * 0.4)
        return np.array([re[0], re[1], le[0], le[1], nt[0], nt[1], rc[0], rc[1], lc[0], lc[1]], dtype=np.float32)

    def test_single_live_frame_never_authorizes_from_warmup(self):
        """A single live frame during warmup must NOT grant AUTHORIZED access."""
        pad_engine = BlinkEARPADEngine(warmup_frames=6)
        pad_engine.initialize()

        crop = np.full((120, 120, 3), 160, dtype=np.uint8)
        lm = self._create_mock_landmarks(center=(200, 200))
        identity = {"name": "Hunter", "is_known": True, "similarity": 0.95, "role": "Lead", "match_pct": 95}

        # Frame 1: artificially set blink_count = 1 on track (single raw live frame)
        res = pad_engine.evaluate(crop, landmarks=lm, face_bbox=(140, 140, 120, 120))
        track = list(pad_engine._tracks.values())[0]
        track.blink_count = 1

        # Frame 2: evaluate
        res = pad_engine.evaluate(crop, landmarks=lm, face_bbox=(140, 140, 120, 120))
        joint = JointAccessEvaluator.evaluate(identity, res)

        # Still warming up (2 total frames < 6 warmup frames)
        self.assertEqual(res.metadata.get("stable_state"), PADStabilizedState.WARMUP.value)
        self.assertFalse(joint.is_access_granted)
        self.assertEqual(joint.status, JointStatus.LIVENESS_VERIFYING)
        self.assertIn("PENDING", joint.reason)

    def test_single_live_frame_does_not_authorize_spoof_stream(self):
        """A phone photo receiving 1 anomalous live frame must NOT flip to AUTHORIZED."""
        pad_engine = BlinkEARPADEngine(warmup_frames=6)
        pad_engine.initialize()

        crop = np.full((120, 120, 3), 200, dtype=np.uint8)
        lm = self._create_mock_landmarks(center=(200, 200))
        identity = {"name": "Hunter", "is_known": True, "similarity": 0.90, "role": "Lead", "match_pct": 90}

        # Feed 8 frames of static spoof photo
        for _ in range(8):
            res = pad_engine.evaluate(crop, landmarks=lm, face_bbox=(140, 140, 120, 120))

        joint = JointAccessEvaluator.evaluate(identity, res)
        self.assertEqual(res.metadata.get("stable_state"), PADStabilizedState.STABLE_SPOOF.value)
        self.assertEqual(joint.status, JointStatus.SPOOF_IMPERSONATION_REJECTED)

        # Inject 1 single anomalous live frame (e.g. temporary landmark jitter or reflection)
        track = list(pad_engine._tracks.values())[0]
        track.blink_count = 1  # Raw frame becomes LIVE

        res = pad_engine.evaluate(crop, landmarks=lm, face_bbox=(140, 140, 120, 120))
        joint = JointAccessEvaluator.evaluate(identity, res)

        # Hysteresis requirement: 1 single live frame MUST NOT flip to STABLE_LIVE / AUTHORIZED
        self.assertNotEqual(joint.status, JointStatus.AUTHORIZED)
        self.assertFalse(joint.is_access_granted)
        self.assertEqual(res.metadata.get("stable_state"), PADStabilizedState.STABLE_SPOOF.value)
        self.assertEqual(joint.status, JointStatus.SPOOF_IMPERSONATION_REJECTED)

    def test_oscillating_evidence_stays_pending(self):
        """Alternating LIVE/SPOOF evidence (L, S, L, S, L, S) must stay PENDING, never AUTHORIZED."""
        stabilizer = TrackedPADStabilizer(window_size=10, live_confirm_frames=5, spoof_confirm_frames=4, warmup_frames=4)
        identity = {"name": "Hunter", "is_known": True, "similarity": 0.92, "role": "Lead", "match_pct": 92}

        # Feed alternating sequence: L, S, L, S, L, S, L, S
        for i in range(8):
            is_live_raw = (i % 2 == 0)
            state, conf, atk, tag = stabilizer.update(
                raw_is_live=is_live_raw,
                raw_confidence=0.85 if is_live_raw else 0.20,
                raw_attack_type=AttackType.BONAFIDE if is_live_raw else AttackType.PRINT,
                raw_tag="BONAFIDE" if is_live_raw else "SPOOF",
                is_screen=False,
            )

        # Confirm state is locked to UNVERIFIED_PENDING
        self.assertEqual(state, PADStabilizedState.UNVERIFIED_PENDING)
        self.assertEqual(tag, "OSCILLATING_EVIDENCE_PENDING")

        # Evaluate through JointAccessEvaluator
        res = pad_engine_res = type('obj', (object,), {
            'is_live': False,
            'confidence': conf,
            'attack_type': atk,
            'model_name': 'Blink-EAR',
            'metadata': {
                'stable_state': state.value,
                'tag': tag,
                'history_symbols': stabilizer.history_symbols,
                'warmup_active': True,
            }
        })()

        joint = JointAccessEvaluator.evaluate(identity, pad_engine_res)
        self.assertFalse(joint.is_access_granted)
        self.assertEqual(joint.status, JointStatus.LIVENESS_VERIFYING)
        self.assertEqual(joint.risk_level.value, "YELLOW")

    def test_continuous_live_evidence_authorizes_cleanly(self):
        """Continuous live evidence transitions cleanly from WARMUP -> STABLE_LIVE (AUTHORIZED)."""
        stabilizer = TrackedPADStabilizer(window_size=10, live_confirm_frames=5, spoof_confirm_frames=4, warmup_frames=4)
        identity = {"name": "Hunter", "is_known": True, "similarity": 0.95, "role": "Lead", "match_pct": 95}

        # Feed 6 consecutive live frames
        for i in range(6):
            state, conf, atk, tag = stabilizer.update(
                raw_is_live=True,
                raw_confidence=0.92,
                raw_attack_type=AttackType.BONAFIDE,
                raw_tag="BONAFIDE_BLINK_VERIFIED",
                is_screen=False,
            )

        self.assertEqual(state, PADStabilizedState.STABLE_LIVE)
        self.assertEqual(atk, AttackType.BONAFIDE)
        self.assertGreaterEqual(conf, 0.85)

        pad_engine_res = type('obj', (object,), {
            'is_live': True,
            'confidence': conf,
            'attack_type': atk,
            'model_name': 'Blink-EAR',
            'metadata': {
                'stable_state': state.value,
                'tag': tag,
                'history_symbols': stabilizer.history_symbols,
                'warmup_active': False,
            }
        })()

        joint = JointAccessEvaluator.evaluate(identity, pad_engine_res)
        self.assertTrue(joint.is_access_granted)
        self.assertEqual(joint.status, JointStatus.AUTHORIZED)
        self.assertEqual(joint.risk_level.value, "GREEN")

    def test_hysteresis_preserves_authorized_against_single_glitch(self):
        """Once STABLE_LIVE, a single transient frame glitch does NOT drop access."""
        stabilizer = TrackedPADStabilizer(window_size=10, live_confirm_frames=5, spoof_confirm_frames=4, warmup_frames=4)

        # Stabilize to LIVE (6 frames)
        for _ in range(6):
            stabilizer.update(True, 0.94, AttackType.BONAFIDE, "BONAFIDE", False)
        self.assertEqual(stabilizer.state, PADStabilizedState.STABLE_LIVE)

        # 1 glitch frame (SPOOF)
        state, conf, atk, tag = stabilizer.update(False, 0.20, AttackType.PRINT, "PRINT_SPOOF", False)

        # Hysteresis keeps STABLE_LIVE
        self.assertEqual(state, PADStabilizedState.STABLE_LIVE)
        self.assertEqual(atk, AttackType.BONAFIDE)

        # If 2 consecutive glitches occur -> drops to UNVERIFIED_PENDING (never immediate SPOOF)
        state, conf, atk, tag = stabilizer.update(False, 0.20, AttackType.PRINT, "PRINT_SPOOF", False)
        self.assertEqual(state, PADStabilizedState.UNVERIFIED_PENDING)

        # Only if 4 consecutive spoof frames occur -> flips to STABLE_SPOOF
        stabilizer.update(False, 0.20, AttackType.PRINT, "PRINT_SPOOF", False)
        state, conf, atk, tag = stabilizer.update(False, 0.20, AttackType.PRINT, "PRINT_SPOOF", False)
        self.assertEqual(state, PADStabilizedState.STABLE_SPOOF)

    def test_screen_enclosure_immediate_stable_spoof(self):
        """When face is physically enclosed by a detected cell phone, it immediately becomes STABLE_SPOOF."""
        stabilizer = TrackedPADStabilizer(window_size=10, live_confirm_frames=5, spoof_confirm_frames=4, warmup_frames=6)

        # Frame 1 with is_screen=True
        state, conf, atk, tag = stabilizer.update(
            raw_is_live=False,
            raw_confidence=0.08,
            raw_attack_type=AttackType.REPLAY,
            raw_tag="SCREEN_REPLAY_CELL PHONE",
            is_screen=True,
        )

        self.assertEqual(state, PADStabilizedState.STABLE_SPOOF)
        self.assertEqual(atk, AttackType.REPLAY)
        self.assertIn("SCREEN_REPLAY", tag)
        self.assertLessEqual(conf, 0.10)

    def test_simultaneous_multi_face_independent_temporal_stabilizers(self):
        """Track 1 (Real person) and Track 2 (Phone photo) maintain completely independent temporal stability buffers."""
        pad_engine = BlinkEARPADEngine(warmup_frames=6)
        pad_engine.initialize()

        # Track 1 (Real face on left)
        crop1 = np.full((120, 100, 3), 150, dtype=np.uint8)
        lm1 = self._create_mock_landmarks(center=(100, 150))
        bbox1 = (50, 90, 100, 120)

        # Track 2 (Phone photo on right inside cell phone)
        crop2 = np.full((120, 100, 3), 220, dtype=np.uint8)
        lm2 = self._create_mock_landmarks(center=(400, 150))
        bbox2 = (350, 90, 100, 120)
        objects = [{"label": "cell phone", "confidence": 0.94, "bbox": (340, 80, 120, 180)}]

        # Feed 8 frames of simultaneous presence
        for _ in range(8):
            res1 = pad_engine.evaluate(crop1, landmarks=lm1, face_bbox=bbox1)
            res2 = pad_engine.evaluate(crop2, landmarks=lm2, face_bbox=bbox2, detected_objects=objects)

        # Trigger blinks on Track 1
        t1 = pad_engine._tracks[1]
        t2 = pad_engine._tracks[2]
        t1.blink_count = 1

        # Feed 6 more frames with Track 1 live and Track 2 in phone screen
        for _ in range(6):
            res1 = pad_engine.evaluate(crop1, landmarks=lm1, face_bbox=bbox1)
            res2 = pad_engine.evaluate(crop2, landmarks=lm2, face_bbox=bbox2, detected_objects=objects)

        # Verify Track 1 is STABLE_LIVE
        self.assertTrue(res1.is_live)
        self.assertEqual(res1.metadata.get("stable_state"), PADStabilizedState.STABLE_LIVE.value)
        self.assertEqual(res1.attack_type, AttackType.BONAFIDE)

        # Verify Track 2 is STABLE_SPOOF (REPLAY)
        self.assertFalse(res2.is_live)
        self.assertEqual(res2.metadata.get("stable_state"), PADStabilizedState.STABLE_SPOOF.value)
        self.assertEqual(res2.attack_type, AttackType.REPLAY)
        self.assertIn("S", res2.metadata.get("history_symbols", ""))

        # Verify zero state leakage
        self.assertEqual(t2.blink_count, 0)
        self.assertNotEqual(t1.stabilizer.history_symbols, t2.stabilizer.history_symbols)

        # Joint Access Evaluation
        id_hunter = {"name": "Hunter", "is_known": True, "similarity": 0.95, "role": "Lead", "match_pct": 95}
        joint1 = JointAccessEvaluator.evaluate(id_hunter, res1)
        joint2 = JointAccessEvaluator.evaluate(id_hunter, res2)

        self.assertEqual(joint1.status, JointStatus.AUTHORIZED)
        self.assertTrue(joint1.is_access_granted)

        self.assertEqual(joint2.status, JointStatus.SPOOF_IMPERSONATION_REJECTED)
        self.assertFalse(joint2.is_access_granted)


if __name__ == "__main__":
    unittest.main()
