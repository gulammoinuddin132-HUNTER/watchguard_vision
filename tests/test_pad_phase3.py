"""
Phase 3: Presentation Attack Detection (PAD / Liveness) Unit & Integration Tests
--------------------------------------------------------------------------------
Validates:
1. Strict Security Invariant: NO AUTHORIZATION BEFORE LIVENESS CONFIRMATION.
   - Zero frames of AUTHORIZED access granted during PAD warmup or ambiguous states.
2. State Transition Tracking:
   - Real person: IDENTITY_MATCHED -> LIVENESS_VERIFYING -> AUTHORIZED
   - Photo attack: IDENTITY_MATCHED -> LIVENESS_VERIFYING -> SPOOF_REJECTED
3. Multi-cue behavioral, photometric, and texture PAD engine.
4. Pluggable PADManager runtime switching.
5. JointAccessEvaluator policy matrix (Identity ⊗ Liveness):
   - Real registered person (Live) -> AUTHORIZED (Risk GREEN)
   - Registered person's phone photo -> SPOOF_IMPERSONATION_REJECTED (Risk RED)
   - Unknown real person (Live) -> UNKNOWN_PERSON (Risk YELLOW)
   - Unknown person's phone photo -> SPOOF_ATTACK_DETECTED (Risk ORANGE)
   - PAD warmup -> LIVENESS_VERIFYING (Access PENDING, Risk YELLOW)
6. ContextEngine presentation attack & verification rules (RULE_PAD_IMPERSONATION, RULE_PAD_UNKNOWN_SPOOF, RULE_PAD_VERIFYING_PENDING).
7. Research Evaluation Harness & ISO/IEC 30107-3 metrics (APCER, BPCER, ACER, HTER).
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.pad.base_pad import AttackType, PADResult
from src.core.pad.joint_decision import JointAccessEvaluator, JointDecision, JointStatus
from src.core.pad.models.blink_ear_pad import BlinkEARPADEngine
from src.core.pad.models.efficientnet_pad import EfficientNetPADEngine
from src.core.pad.models.lbp_svm_pad import LBPSVMPADEngine
from src.core.pad.models.mobilenet_pad import MobileNetPADEngine
from src.core.pad.pad_manager import PADManager
from src.database.db_manager import DatabaseManager
from src.evaluation.metrics import calculate_pad_metrics
from src.evaluation.pad_harness import DatasetSample, PADBenchmarkHarness
from src.events.event_types import EventType, RiskLevel


class TestPADPhase3(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_pad.db"
        self.db = DatabaseManager(db_path=self.db_path)

        # Synthetic test crops (112x112 RGB)
        self.synthetic_face = np.full((112, 112, 3), 128, dtype=np.uint8)
        noise = np.random.randint(0, 50, (112, 112, 3), dtype=np.uint8)
        self.synthetic_face = np.clip(self.synthetic_face + noise, 0, 255).astype(np.uint8)

        # Synthetic landmarks: [x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc]
        self.synthetic_landmarks = np.array([38.0, 42.0, 74.0, 42.0, 56.0, 62.0, 42.0, 84.0, 70.0, 84.0], dtype=np.float32)

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- 1. Strict Authorization & Warmup Invariants ---

    def test_zero_frames_authorized_before_pad_warmup(self):
        """
        CRITICAL SECURITY INVARIANT:
        Verify that during the initial PAD warmup frames, a matched registered user
        is NEVER granted AUTHORIZED access. Access must remain strictly PENDING.
        """
        engine = BlinkEARPADEngine(warmup_frames=6)
        engine.initialize()
        identity = {"is_known": True, "name": "Hunter", "similarity": 0.94, "role": "Security Lead"}

        # Simulate first 5 warmup frames
        for frame_idx in range(5):
            res_pad = engine.evaluate(
                raw_face_crop=self.synthetic_face,
                aligned_face=self.synthetic_face,
                landmarks=self.synthetic_landmarks,
            )
            joint_dec = JointAccessEvaluator.evaluate(identity, res_pad)

            # Assert strictly no authorization granted
            self.assertFalse(joint_dec.is_access_granted, f"Security Breach! Access granted on warmup frame {frame_idx+1}")
            self.assertEqual(joint_dec.status, JointStatus.LIVENESS_VERIFYING)
            self.assertNotEqual(joint_dec.risk_level, RiskLevel.GREEN)
            self.assertEqual(joint_dec.risk_level, RiskLevel.YELLOW)
            self.assertIn("PENDING", joint_dec.reason)

    def test_photo_attack_trajectory_never_authorized(self):
        """
        CRITICAL REAL-WORLD TEST:
        When a photo of registered user 'Hunter' is held in front of the camera:
        Frame 1..7 (Warmup) -> LIVENESS_VERIFYING (Access: PENDING, Risk: YELLOW)
        Frame 8..20 (Rigid detection) -> SPOOF_IMPERSONATION_REJECTED (Access: DENIED, Risk: RED)
        Under NO circumstances is status ever AUTHORIZED.
        """
        engine = BlinkEARPADEngine(warmup_frames=8)
        engine.initialize()
        identity = {"is_known": True, "name": "Hunter", "similarity": 0.95, "role": "Security Lead"}

        for frame_idx in range(16):
            # Rigid 2D jitter simulating hand holding phone with photo
            rigid_jitter = np.random.normal(0, 1.5, (1, 2)).astype(np.float32)
            rigid_lm = (self.synthetic_landmarks.reshape(5, 2) + rigid_jitter).ravel()

            res_pad = engine.evaluate(
                raw_face_crop=self.synthetic_face,
                aligned_face=self.synthetic_face,
                landmarks=rigid_lm,
            )
            joint_dec = JointAccessEvaluator.evaluate(identity, res_pad)

            # Invariant: Access MUST NEVER be granted on any frame
            self.assertFalse(joint_dec.is_access_granted, f"Access incorrectly granted on frame {frame_idx+1}")
            self.assertNotEqual(joint_dec.status, JointStatus.AUTHORIZED)

            if frame_idx < 7:
                self.assertEqual(joint_dec.status, JointStatus.LIVENESS_VERIFYING)
            else:
                self.assertEqual(joint_dec.status, JointStatus.SPOOF_IMPERSONATION_REJECTED)
                self.assertEqual(joint_dec.risk_level, RiskLevel.RED)
                self.assertTrue(joint_dec.is_spoof)

    def test_real_person_trajectory_reaches_authorized(self):
        """
        Verify real person trajectory:
        Warmup -> LIVENESS_VERIFYING (Access: PENDING)
        Sufficient 3D Motion / Blink -> AUTHORIZED (Access: GRANTED, Risk: GREEN)
        """
        engine = BlinkEARPADEngine(warmup_frames=8)
        engine.initialize()
        identity = {"is_known": True, "name": "Hunter", "similarity": 0.92, "role": "Security Lead"}

        # Feed 16 dynamic non-rigid frames
        for frame_idx in range(16):
            dynamic_lm = self.synthetic_landmarks.copy()
            dynamic_lm[6:] += np.array([frame_idx * 1.5, -frame_idx * 1.0, -frame_idx * 1.5, frame_idx * 1.2], dtype=np.float32)

            res_pad = engine.evaluate(
                raw_face_crop=self.synthetic_face,
                aligned_face=self.synthetic_face,
                landmarks=dynamic_lm,
            )
            joint_dec = JointAccessEvaluator.evaluate(identity, res_pad)

            if frame_idx < 11:
                self.assertEqual(joint_dec.status, JointStatus.LIVENESS_VERIFYING)
                self.assertFalse(joint_dec.is_access_granted)
            else:
                self.assertEqual(joint_dec.status, JointStatus.AUTHORIZED)
                self.assertTrue(joint_dec.is_access_granted)
                self.assertEqual(joint_dec.risk_level, RiskLevel.GREEN)

    # --- 2. Four Modular PAD Engines ---

    def test_lbp_svm_pad_engine(self):
        """Test LBP-SVM texture baseline engine."""
        engine = LBPSVMPADEngine()
        self.assertTrue(engine.initialize())
        self.assertIn("LBP-SVM", engine.model_name)

        res = engine.evaluate(raw_face_crop=self.synthetic_face, aligned_face=self.synthetic_face)
        self.assertIsInstance(res, PADResult)
        self.assertIsInstance(res.attack_type, AttackType)

    def test_mobilenet_pad_engine(self):
        """Test MobileNetV3-Small CNN PAD engine."""
        engine = MobileNetPADEngine()
        self.assertTrue(engine.initialize())
        self.assertIn("MobileNetV3", engine.model_name)

        res = engine.evaluate(raw_face_crop=self.synthetic_face, aligned_face=self.synthetic_face)
        self.assertIsInstance(res, PADResult)

    def test_efficientnet_pad_engine(self):
        """Test EfficientNet-B0 high-capacity CNN PAD engine."""
        engine = EfficientNetPADEngine()
        self.assertTrue(engine.initialize())
        self.assertIn("EfficientNet", engine.model_name)

        res = engine.evaluate(raw_face_crop=self.synthetic_face, aligned_face=self.synthetic_face)
        self.assertIsInstance(res, PADResult)

    # --- 3. PAD Manager Lifecycle & Model Switching ---

    def test_pad_manager_switching(self):
        """Verify seamless runtime switching between all 4 PAD engines."""
        manager = PADManager()
        self.assertTrue(manager.initialize())

        for m_name in ["Blink-EAR", "LBP-SVM", "MobileNetV3-Small", "EfficientNet-B0"]:
            self.assertTrue(manager.set_active_model(m_name))
            self.assertEqual(manager.active_model_name, m_name)
            res = manager.evaluate_presentation(raw_face_crop=self.synthetic_face, aligned_face=self.synthetic_face)
            self.assertIsInstance(res, PADResult)

        self.assertFalse(manager.set_active_model("NonExistentPADModel"))

    # --- 4. Joint Access Evaluator Matrix ---

    def test_joint_decision_unknown_real_person(self):
        """Unknown real person: Unknown + Live -> UNKNOWN_PERSON (Risk YELLOW)."""
        identity = {"is_known": False, "name": "UNKNOWN PERSON", "similarity": 0.20, "role": "Visitor"}
        pad_res = PADResult(
            is_live=True,
            confidence=0.88,
            attack_type=AttackType.BONAFIDE,
            inference_latency_ms=3.0,
            model_name="Blink-EAR",
        )
        dec = JointAccessEvaluator.evaluate(identity, pad_res)

        self.assertEqual(dec.status, JointStatus.UNKNOWN_PERSON)
        self.assertFalse(dec.is_access_granted)
        self.assertEqual(dec.risk_level, RiskLevel.YELLOW)

    def test_joint_decision_unknown_person_phone_photo(self):
        """Unknown person's phone photo: Unknown + Spoof -> SPOOF_ATTACK_DETECTED (Risk ORANGE)."""
        identity = {"is_known": False, "name": "UNKNOWN PERSON", "similarity": 0.15, "role": "Visitor"}
        pad_res = PADResult(
            is_live=False,
            confidence=0.16,
            attack_type=AttackType.PRINT,
            inference_latency_ms=2.8,
            model_name="Blink-EAR",
        )
        dec = JointAccessEvaluator.evaluate(identity, pad_res)

        self.assertEqual(dec.status, JointStatus.SPOOF_ATTACK_DETECTED)
        self.assertFalse(dec.is_access_granted)
        self.assertTrue(dec.is_spoof)
        self.assertEqual(dec.risk_level, RiskLevel.ORANGE)

    # --- 5. ContextEngine Presentation Attack & Verification Rules ---

    def test_context_engine_liveness_verifying_rule(self):
        """Verify ContextEngine evaluates RULE_PAD_VERIFYING_PENDING during warmup (Risk YELLOW)."""
        engine = ContextEngine()
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_verifying": True,
            "has_unknown": False,
            "has_spoof": False,
            "people": [
                {
                    "name": "Hunter",
                    "is_known": True,
                    "is_spoof": False,
                    "attack_type": "UNKNOWN_ATTACK",
                    "similarity": 0.94,
                    "liveness_confidence": 0.45,
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.YELLOW)
        self.assertEqual(decision.decision, "LIVENESS_VERIFYING")
        self.assertIn("RULE_PAD_VERIFYING_PENDING", decision.matched_rules)
        self.assertFalse(decision.evidence_required)
        self.assertIn("PENDING", decision.reason)

    def test_context_engine_impersonation_spoof_rule(self):
        """Verify ContextEngine triggers Risk RED for spoofed registered profile."""
        engine = ContextEngine()
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_verifying": False,
            "has_unknown": False,
            "has_spoof": True,
            "people": [
                {
                    "name": "Hunter",
                    "is_known": True,
                    "is_spoof": True,
                    "attack_type": "REPLAY_ATTACK",
                    "similarity": 0.94,
                    "liveness_confidence": 0.15,
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.RED)
        self.assertEqual(decision.decision, "PRESENTATION_ATTACK_IMPERSONATION")
        self.assertTrue(decision.evidence_required)
        self.assertIn("RULE_PAD_IMPERSONATION", decision.matched_rules)
        self.assertEqual(decision.event_type, EventType.PRESENTATION_ATTACK)

    def test_context_engine_unknown_spoof_rule(self):
        """Verify ContextEngine triggers Risk ORANGE for unregistered spoof attack."""
        engine = ContextEngine()
        person_data = {
            "detected": True,
            "has_authorized": False,
            "has_verifying": False,
            "has_unknown": True,
            "has_spoof": True,
            "people": [
                {
                    "name": "UNKNOWN PERSON",
                    "is_known": False,
                    "is_spoof": True,
                    "attack_type": "PRINT_ATTACK",
                    "similarity": 0.10,
                    "liveness_confidence": 0.20,
                }
            ],
        }
        object_data = {"objects": [], "summary": "None"}

        decision = engine.evaluate_context(person_data, object_data)
        self.assertEqual(decision.risk_level, RiskLevel.ORANGE)
        self.assertEqual(decision.decision, "PRESENTATION_ATTACK_UNKNOWN")
        self.assertTrue(decision.evidence_required)
        self.assertIn("RULE_PAD_UNKNOWN_SPOOF", decision.matched_rules)

    # --- 6. Research Evaluation Metrics & Harness ---

    def test_calculate_pad_metrics(self):
        """Verify APCER, BPCER, ACER, and per-attack breakdown calculations."""
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        y_scores = [0.95, 0.90, 0.88, 0.85, 0.40, 0.85, 0.30, 0.25, 0.20, 0.10]
        attack_types = ["BONAFIDE", "BONAFIDE", "BONAFIDE", "BONAFIDE", "BONAFIDE", "PRINT", "PRINT", "REPLAY", "REPLAY", "MASK"]

        metrics = calculate_pad_metrics(
            y_true=y_true,
            y_scores=y_scores,
            attack_types=attack_types,
            threshold=0.80,
            latencies_ms=[10.0, 12.0, 8.0, 11.0, 9.0, 14.0, 10.0, 13.0, 11.0, 15.0],
            model_name="Test-Model",
            dataset_name="Test-Dataset",
        )

        self.assertAlmostEqual(metrics.bpcer, 1 / 5, places=3)
        self.assertAlmostEqual(metrics.apcer, 1 / 5, places=3)
        self.assertAlmostEqual(metrics.acer, 0.20, places=3)
        self.assertAlmostEqual(metrics.apcer_print, 1 / 2, places=3)
        self.assertAlmostEqual(metrics.apcer_replay, 0.0, places=3)
        self.assertGreater(metrics.mean_latency_ms, 0.0)

    def test_benchmark_harness_execution(self):
        """Verify PADBenchmarkHarness evaluates dataset samples and outputs valid report."""
        engine = LBPSVMPADEngine()
        harness = PADBenchmarkHarness(pad_engine=engine)

        samples = [
            DatasetSample(image_path="", is_live=True, attack_type=AttackType.BONAFIDE),
            DatasetSample(image_path="", is_live=False, attack_type=AttackType.PRINT),
        ]
        metrics = harness.evaluate_dataset(samples=samples, dataset_name="Synthetic-Validation")

        self.assertIsInstance(metrics.acer, float)
        self.assertEqual(metrics.total_samples, 2)
        markdown_table = metrics.to_markdown_table()
        self.assertIn("LBP-SVM", markdown_table)
        self.assertIn("APCER", markdown_table)


if __name__ == "__main__":
    unittest.main()
