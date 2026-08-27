"""
Multi-Face Presentation Attack Detection & Simultaneous Stream Isolation Test Suite
-----------------------------------------------------------------------------------
Tests:
1. Real authorized person only -> Access: AUTHORIZED (Green).
2. Phone photo only -> Access: SPOOF_IMPERSONATION_REJECTED (Red).
3. Simultaneous real person + phone photo of same person -> Real: AUTHORIZED, Phone: SPOOF_REJECTED.
4. Simultaneous real person + printed paper photo -> Real: AUTHORIZED, Print: SPOOF_REJECTED.
5. Unknown person + Phone photo -> Unknown: UNKNOWN VISITOR, Phone: SPOOF_DETECTED.
6. Moving phone photo (2D hand motion) -> Preserves rigid ratios -> Access: SPOOF_REJECTED.
7. Zero-leakage isolation: Track #1 (Real) and Track #2 (Phone) never share PAD buffers or blink counts.
"""

import math
import os
import unittest
import numpy as np
import cv2

from config import DEMO_FACES_DIR
from src.core.pad.base_pad import AttackType, PADResult
from src.core.pad.joint_decision import JointAccessEvaluator, JointDecision, JointStatus
from src.core.pad.models.blink_ear_pad import BlinkEARPADEngine
from src.core.pad.pad_manager import PADManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.context_engine import ContextEngine
from src.database.db_manager import DatabaseManager


class TestMultiFacePAD(unittest.TestCase):

    def setUp(self):
        self.db = DatabaseManager()
        self.pad_manager = PADManager(default_model="Blink-EAR")
        self.engine = PersonRecognitionEngine(db_manager=self.db, pad_manager=self.pad_manager)
        self.engine.initialize(auto_seed_demo=True)
        self.context_engine = ContextEngine()

    def _create_mock_face(self, center: tuple = (100, 100), scale: float = 40.0) -> np.ndarray:
        """Generates synthetic 5-point YuNet landmarks."""
        cx, cy = center
        re = (cx - scale * 0.5, cy - scale * 0.3)
        le = (cx + scale * 0.5, cy - scale * 0.3)
        nt = (cx, cy)
        rc = (cx - scale * 0.35, cy + scale * 0.4)
        lc = (cx + scale * 0.35, cy + scale * 0.4)
        return np.array([re[0], re[1], le[0], le[1], nt[0], nt[1], rc[0], rc[1], lc[0], lc[1]], dtype=np.float32)

    def test_single_real_face_becomes_authorized(self):
        """Scenario A: Live face with natural blink transitions to AUTHORIZED."""
        engine = BlinkEARPADEngine(warmup_frames=5)
        engine.initialize()

        landmarks = self._create_mock_face(center=(200, 200))
        crop = np.full((120, 120, 3), 160, dtype=np.uint8)
        # Draw eyes and mouth
        cv2.circle(crop, (40, 40), 8, (20, 20, 20), -1)
        cv2.circle(crop, (80, 40), 8, (20, 20, 20), -1)

        # 1. Warmup frames
        for _ in range(4):
            res = engine.evaluate(raw_face_crop=crop, landmarks=landmarks, face_bbox=(140, 140, 120, 120))
            self.assertFalse(res.is_live)
            self.assertEqual(res.metadata.get("tag"), "WARMUP_UNVERIFIED")

        # 2. Simulate natural blink and feed 6 live frames
        track = list(engine._tracks.values())[0]
        track.blink_count = 1  # Natural blink confirmed

        for _ in range(6):
            res = engine.evaluate(raw_face_crop=crop, landmarks=landmarks, face_bbox=(140, 140, 120, 120))

        self.assertTrue(res.is_live)
        self.assertEqual(res.attack_type, AttackType.BONAFIDE)
        self.assertGreaterEqual(res.confidence, 0.85)

        # 3. Joint Decision
        identity = {"name": "Hunter", "is_known": True, "similarity": 0.92, "role": "Lead", "match_pct": 92}
        joint = JointAccessEvaluator.evaluate(identity, res)
        self.assertEqual(joint.status, JointStatus.AUTHORIZED)
        self.assertTrue(joint.is_access_granted)

    def test_phone_photo_in_cell_phone_box_is_rejected(self):
        """Scenario B: Face inside a detected cell phone is immediately flagged as SCREEN REPLAY."""
        engine = BlinkEARPADEngine(warmup_frames=5)
        engine.initialize()

        face_bbox = (300, 150, 100, 120)
        landmarks = self._create_mock_face(center=(350, 210))
        crop = np.full((120, 100, 3), 200, dtype=np.uint8)

        # Detected cell phone enclosing the face
        detected_objects = [
            {"label": "cell phone", "confidence": 0.95, "bbox": (280, 120, 140, 220)}
        ]

        res = engine.evaluate(
            raw_face_crop=crop,
            landmarks=landmarks,
            face_bbox=face_bbox,
            detected_objects=detected_objects,
        )

        self.assertFalse(res.is_live)
        self.assertEqual(res.attack_type, AttackType.REPLAY)
        self.assertTrue(res.metadata.get("is_inside_screen"))
        self.assertIn("SCREEN_REPLAY", res.metadata.get("tag", ""))

        # Joint decision must reject
        identity = {"name": "Hunter", "is_known": True, "similarity": 0.88, "role": "Lead", "match_pct": 88}
        joint = JointAccessEvaluator.evaluate(identity, res)
        self.assertEqual(joint.status, JointStatus.SPOOF_IMPERSONATION_REJECTED)
        self.assertFalse(joint.is_access_granted)

    def test_simultaneous_real_person_and_phone_photo_independent_states(self):
        """Scenario C: Real face (left) and phone photo (right) are evaluated independently without leakage."""
        engine = BlinkEARPADEngine(warmup_frames=5)
        engine.initialize()

        # Face 1: Real face (Left: x=50, y=100)
        face1_bbox = (50, 100, 100, 120)
        lm1 = self._create_mock_face(center=(100, 160))
        crop1 = np.full((120, 100, 3), 160, dtype=np.uint8)

        # Face 2: Phone photo (Right: x=400, y=100) inside cell phone
        face2_bbox = (400, 100, 90, 110)
        lm2 = self._create_mock_face(center=(445, 155))
        crop2 = np.full((110, 90, 3), 220, dtype=np.uint8)

        detected_objects = [
            {"label": "cell phone", "confidence": 0.92, "bbox": (380, 80, 130, 200)}
        ]

        # Feed 10 frames of simultaneous presence
        for _ in range(10):
            res1 = engine.evaluate(
                raw_face_crop=crop1,
                landmarks=lm1,
                face_bbox=face1_bbox,
                detected_objects=detected_objects,
            )
            res2 = engine.evaluate(
                raw_face_crop=crop2,
                landmarks=lm2,
                face_bbox=face2_bbox,
                detected_objects=detected_objects,
            )

        # Confirm 2 distinct tracks were created
        self.assertEqual(len(engine._tracks), 2)
        tracks = list(engine._tracks.values())
        t1, t2 = tracks[0], tracks[1]

        # Track 1 is real: trigger a blink and feed 6 live frames
        t1.blink_count = 1
        for _ in range(6):
            res1 = engine.evaluate(raw_face_crop=crop1, landmarks=lm1, face_bbox=face1_bbox, detected_objects=detected_objects)
            res2 = engine.evaluate(raw_face_crop=crop2, landmarks=lm2, face_bbox=face2_bbox, detected_objects=detected_objects)

        # Verify Real Face (Track 1) is LIVE
        self.assertTrue(res1.is_live)
        self.assertEqual(res1.attack_type, AttackType.BONAFIDE)

        # Verify Phone Face (Track 2) is SPOOF REPLAY
        self.assertFalse(res2.is_live)
        self.assertEqual(res2.attack_type, AttackType.REPLAY)
        self.assertEqual(t2.blink_count, 0, "Track 2 must NEVER inherit Track 1's blink count!")

        # Joint Access Decisions
        id_hunter = {"name": "Hunter", "is_known": True, "similarity": 0.94, "role": "Lead", "match_pct": 94}
        joint1 = JointAccessEvaluator.evaluate(id_hunter, res1)
        joint2 = JointAccessEvaluator.evaluate(id_hunter, res2)

        self.assertEqual(joint1.status, JointStatus.AUTHORIZED)
        self.assertEqual(joint2.status, JointStatus.SPOOF_IMPERSONATION_REJECTED)

        # Verify ContextEngine security prioritization (Scene Risk is RED because of impersonation spoof)
        people_data = [
            {"name": "Hunter", "is_known": True, "is_spoof": False, "bbox": face1_bbox, "joint_decision": joint1},
            {"name": "Hunter", "is_known": True, "is_spoof": True, "bbox": face2_bbox, "attack_type": "REPLAY_ATTACK", "joint_decision": joint2},
        ]
        decision = self.context_engine.evaluate_context(
            person_data={"detected": True, "has_authorized": True, "has_spoof": True, "people": people_data},
            object_data={"objects": detected_objects, "summary": "cell phone"},
        )
        self.assertEqual(decision.decision, "PRESENTATION_ATTACK_IMPERSONATION")
        self.assertEqual(decision.risk_level.value, "RED")

    def test_moving_phone_photo_rigid_2d_variance_rejected(self):
        """Scenario E: Moving a phone photo around changes 2D position but preserves rigid distance ratios -> SPOOF."""
        engine = BlinkEARPADEngine(warmup_frames=5)
        engine.initialize()

        base_lm = self._create_mock_face(center=(200, 200))
        crop = np.full((120, 120, 3), 180, dtype=np.uint8)

        # Simulate moving phone (applying 2D translation and scaling)
        for i in range(15):
            dx = math.sin(i * 0.5) * 20.0
            dy = math.cos(i * 0.5) * 15.0
            scale_mod = 1.0 + math.sin(i * 0.3) * 0.05

            moved_lm = base_lm.copy()
            moved_lm[0::2] = (base_lm[0::2] - 200) * scale_mod + 200 + dx
            moved_lm[1::2] = (base_lm[1::2] - 200) * scale_mod + 200 + dy

            res = engine.evaluate(
                raw_face_crop=crop,
                landmarks=moved_lm,
                face_bbox=(int(140 + dx), int(140 + dy), 120, 120),
            )

        # After warmup without genuine blinks or non-rigid deformations
        self.assertFalse(res.is_live)
        self.assertIn("RIGID", res.metadata.get("tag", ""))
        self.assertIn(res.attack_type, (AttackType.REPLAY, AttackType.PRINT))

    def test_unknown_person_and_phone_photo_simultaneous(self):
        """Scenario F: Unknown visitor (Track 1) and phone photo (Track 2)."""
        engine = BlinkEARPADEngine(warmup_frames=5)
        engine.initialize()

        # Track 1: Unknown live person
        lm1 = self._create_mock_face(center=(150, 150))
        crop1 = np.full((120, 100, 3), 150, dtype=np.uint8)
        # Track 2: Phone photo
        lm2 = self._create_mock_face(center=(400, 150))
        crop2 = np.full((120, 100, 3), 200, dtype=np.uint8)

        objects = [{"label": "cell phone", "confidence": 0.90, "bbox": (360, 100, 120, 180)}]

        for _ in range(8):
            res1 = engine.evaluate(crop1, landmarks=lm1, face_bbox=(100, 100, 100, 120), detected_objects=objects)
            res2 = engine.evaluate(crop2, landmarks=lm2, face_bbox=(360, 100, 100, 120), detected_objects=objects)

        self.assertFalse(res2.is_live)
        self.assertEqual(res2.attack_type, AttackType.REPLAY)

        # Joint Decisions
        id_unknown = {"name": "UNKNOWN PERSON", "is_known": False, "similarity": 0.20, "role": "Visitor", "match_pct": 20}
        joint1 = JointAccessEvaluator.evaluate(id_unknown, res1)
        joint2 = JointAccessEvaluator.evaluate(id_unknown, res2)

        self.assertEqual(joint2.status, JointStatus.SPOOF_ATTACK_DETECTED)
        self.assertFalse(joint2.is_access_granted)

    def test_real_person_and_printed_paper_photo(self):
        """Scenario G: Real person + printed paper photo."""
        engine = BlinkEARPADEngine(warmup_frames=5)
        engine.initialize()

        lm1 = self._create_mock_face(center=(150, 150))
        crop1 = np.full((120, 100, 3), 150, dtype=np.uint8)
        lm2 = self._create_mock_face(center=(400, 150))
        crop2 = np.full((120, 100, 3), 200, dtype=np.uint8)

        # 10 frames without screen object
        for _ in range(10):
            res1 = engine.evaluate(crop1, landmarks=lm1, face_bbox=(100, 100, 100, 120))
            res2 = engine.evaluate(crop2, landmarks=lm2, face_bbox=(350, 100, 100, 120))

        # Real person blinks and feeds 6 live frames
        list(engine._tracks.values())[0].blink_count = 1

        for _ in range(6):
            res1 = engine.evaluate(crop1, landmarks=lm1, face_bbox=(100, 100, 100, 120))
            res2 = engine.evaluate(crop2, landmarks=lm2, face_bbox=(350, 100, 100, 120))

        self.assertTrue(res1.is_live)
        self.assertFalse(res2.is_live)
        self.assertIn(res2.attack_type, (AttackType.PRINT, AttackType.REPLAY))


if __name__ == "__main__":
    unittest.main()
