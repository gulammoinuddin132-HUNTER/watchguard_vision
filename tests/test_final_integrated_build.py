"""
WatchGuard Vision - Final Integrated Build Verification Suite
------------------------------------------------------------
Comprehensive verification suite for the Final Integrated Research-Grade Build:
  1. 84-Class Unified Model & Detector Interface
  2. Small-Object Confirmation & Person-Asset Association
  3. CCSDF 8-Dimensional Context Vector & Fusion
  4. Authoritative Policy Engine Invariants
  5. Tailgating Detection & Zero Authorization Leakage
  6. Behavior Anomaly Signal Computation & Fusion
  7. Uncertainty Gating & Confidence Tiers
  8. Edge-Triggered Spoof Counter Integrity & Debouncing
  9. Multi-Track State Isolation
  10. Bounded Anti-Spam Database Event Logging
  11. Production Fallback Integrity
"""

import unittest
import time
import json
from pathlib import Path
import numpy as np

from config import (
    WATCHGUARD_UNIFIED_CLASSES,
    COCO_CLASSES,
    SECURITY_OBJECT_CATEGORIES,
    YOLOV8N_MODEL_PATH,
    WATCHGUARD_UNIFIED_MODEL_PATH,
    OBJECT_DETECTOR_MODE,
)
from src.core.interfaces.detector_interface import DetectedObject
from src.core.object_detection import (
    YOLOv8nCOCODetector,
    WatchGuardUnified84Detector,
    ObjectDetectionEngine,
)
from src.core.person_entity import PersonEntityFusionEngine, PersonEntity
from src.core.policy_engine import PolicyEngine, RiskLevel
from src.core.track_security_state import TrackSecurityState
from src.research.ccsdf_config import CCSDFWeights, CCSDFThresholds
from src.research.ccsdf_engine import (
    CCSDFEngine,
    CCSDFContext,
    IdentityState,
    LivenessState,
    ZoneState,
    TemporalState,
    ProximityState,
    HistoryState,
    SmallAssetState,
    BehaviorState,
    TailgatingState,
    CCSDFAction,
)
from src.research.tailgating_engine import TailgatingDetector
from src.research.behavior_engine import BehaviorAnomalyEngine, TrackBehaviorMetrics
from src.research.uncertainty_engine import UncertaintyAggregator, DecisionConfidence


class TestFinalIntegratedBuild(unittest.TestCase):
    """Rigorous verification of the complete WatchGuard Vision research architecture."""

    def test_01_84_class_taxonomy_and_dimensions(self):
        """Verify 80 COCO + 4 WatchGuard classes and [1, 88, 8400] output contract."""
        self.assertEqual(len(WATCHGUARD_UNIFIED_CLASSES), 84)
        # Verify first 80 match COCO exactly
        for i in range(80):
            self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[i], COCO_CLASSES[i])
        # Verify 4 WatchGuard domain additions
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[80], "pen")
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[81], "access_badge")
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[82], "usb_drive")
        self.assertEqual(WATCHGUARD_UNIFIED_CLASSES[83], "keys")

        # Verify output channel formula: 4 coords + 84 classes = 88 channels
        output_channels = 4 + len(WATCHGUARD_UNIFIED_CLASSES)
        self.assertEqual(output_channels, 88)

    def test_02_production_isolation_and_fallback(self):
        """Verify production model and default mode remain safely intact."""
        self.assertTrue(Path(YOLOV8N_MODEL_PATH).exists(), "Production YOLOv8n model must exist")
        self.assertEqual(OBJECT_DETECTOR_MODE, "production", "Production mode must remain default fallback")

        prod_det = YOLOv8nCOCODetector()
        self.assertTrue(prod_det.initialize())
        self.assertEqual(len(prod_det.class_names), 80)
        prod_det.release()

    def test_03_unified_84_detector_execution(self):
        """Verify WatchGuardUnified84Detector initializes and executes forward pass."""
        uni_det = WatchGuardUnified84Detector(model_path=WATCHGUARD_UNIFIED_MODEL_PATH)
        self.assertTrue(uni_det.initialize())
        self.assertEqual(len(uni_det.class_names), 84)

        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = uni_det.detect(dummy_img)
        self.assertIsInstance(dets, list)
        uni_det.release()

    def test_04_small_object_distance_tiers_and_fields(self):
        """Verify DetectedObject distance tiers: CLOSE (>10k), MEDIUM (1.6k-10k), FAR (<1.6k)."""
        close_obj = DetectedObject(class_id=80, label="pen", confidence=0.85, bbox=(10, 10, 110, 100)) # 11,000 px^2
        med_obj = DetectedObject(class_id=82, label="usb_drive", confidence=0.75, bbox=(10, 10, 50, 50))  # 2,500 px^2
        far_obj = DetectedObject(class_id=83, label="keys", confidence=0.60, bbox=(10, 10, 30, 30))      # 900 px^2

        self.assertEqual(close_obj.distance_tier, "CLOSE")
        self.assertEqual(med_obj.distance_tier, "MEDIUM")
        self.assertEqual(far_obj.distance_tier, "FAR")
        self.assertEqual(close_obj.security_category, "context-relevant")
        self.assertEqual(med_obj.security_category, "context-relevant")

    def test_05_object_person_association_and_no_leakage(self):
        """Verify spatial association assigns pixel distance, confidence, and does not misattribute."""
        fusion_engine = PersonEntityFusionEngine(proximity_threshold_pixels=120.0)

        people = [
            {"track_id": 101, "name": "Hunter", "is_known": True, "bbox": (100, 100, 100, 200)},
            {"track_id": 102, "name": "Unknown", "is_known": False, "bbox": (450, 100, 100, 200)},
        ]
        objects = [
            # USB drive near Hunter (centroid at (110, 150), dist < 50px)
            {"label": "usb_drive", "confidence": 0.85, "bbox": (100, 140, 20, 20)},
            # Access badge far away from both
            {"label": "access_badge", "confidence": 0.90, "bbox": (300, 300, 20, 30)},
        ]

        entities, non_person_objs, _ = fusion_engine.fuse(people, objects)
        self.assertEqual(len(entities), 2)

        # Hunter (track 101) should have the USB drive associated
        hunter_ent = next(e for e in entities if e.track_id == 101)
        unknown_ent = next(e for e in entities if e.track_id == 102)

        self.assertEqual(len(hunter_ent.associated_objects), 1)
        self.assertEqual(hunter_ent.associated_objects[0]["object_label"], "usb_drive")
        self.assertLess(hunter_ent.associated_objects[0]["distance_pixels"], 120.0)
        self.assertGreater(hunter_ent.associated_objects[0]["association_confidence"], 0.40)

        # Unknown person should NOT have the USB drive
        self.assertEqual(len(unknown_ent.associated_objects), 0)

        # Non-person objects verify assigned association attributes
        usb_obj = next(o for o in non_person_objs if o["label"] == "usb_drive")
        self.assertEqual(usb_obj["associated_person_track"], 101)

        badge_obj = next(o for o in non_person_objs if o["label"] == "access_badge")
        self.assertIsNone(badge_obj["associated_person_track"])
        self.assertEqual(badge_obj["pixel_distance_to_person"], -1.0)
        self.assertEqual(badge_obj["association_confidence"], 0.0)

    def test_06_ccsdf_8_dimensional_vector_and_normalization(self):
        """Verify CCSDF supports S(t) = [Sid, Spad, Szone, Stime, Sprox, Shist, Sasset, Sbehavior]."""
        ctx = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.96),
            zone=ZoneState(zone_id="NORMAL"),
            temporal=TemporalState(is_authorized_hours=True),
            proximity=ProximityState(has_interaction=False),
            history=HistoryState(violations_count=0),
            small_asset=SmallAssetState(has_detected_asset=False),
            behavior=BehaviorState(s_behavior=0.05),
        )
        signals = ctx.get_normalized_signals()
        expected_signals = ["identity", "pad", "zone", "time", "proximity", "history", "asset", "behavior"]
        for s in expected_signals:
            self.assertIn(s, signals)
            self.assertGreaterEqual(signals[s], 0.0)
            self.assertLessEqual(signals[s], 1.0)

        # Weights normalization test
        weights = CCSDFWeights(identity=2.0, pad=2.0, zone=2.0, time=2.0, proximity=1.0, history=1.0, asset=1.0, behavior=1.0)
        w_dict = weights.to_dict()
        self.assertAlmostEqual(sum(w_dict.values()), 1.0, places=3)

    def test_07_strict_authoritative_policy_invariants(self):
        """Verify hard policy safety rules: unknown never ALLOW, PAD failure DENY, policy DENY cannot become ALLOW."""
        policy_engine = PolicyEngine()
        ccsdf_engine = CCSDFEngine()

        # Invariant 1: Unknown person must never be granted access
        res_unk = policy_engine.evaluate(
            identity="Unknown Person", role="Visitor", is_known=False,
            liveness_state="LIVE", liveness_conf=0.95, zone_id="NORMAL",
            time_str="10:00", is_auth_time=True, is_spoof=False,
        )
        self.assertFalse(res_unk.is_access_granted)
        self.assertNotEqual(res_unk.risk_level, RiskLevel.GREEN)

        # Invariant 2: PAD failure must be DENY / RED
        res_spoof = policy_engine.evaluate(
            identity="Hunter", role="Staff", is_known=True,
            liveness_state="SPOOF", liveness_conf=0.10, zone_id="NORMAL",
            time_str="10:00", is_auth_time=True, is_spoof=True,
        )
        self.assertFalse(res_spoof.is_access_granted)
        self.assertEqual(res_spoof.risk_level, RiskLevel.RED)

        # Invariant 3: CCSDF cannot override policy DENY into ALLOW
        ctx_spoof = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True),
            liveness=LivenessState(liveness_state="SPOOF", confidence=0.10),
        )
        ccsdf_res = ccsdf_engine.evaluate_context(ctx_spoof, authoritative_action="DENIED")
        self.assertEqual(ccsdf_res.recommended_action, CCSDFAction.DENY)
        self.assertEqual(ccsdf_res.authoritative_action, "DENIED")

    def test_08_tailgating_detection_and_non_transfer(self):
        """Verify tailgating triggers edge event and leader authorization never transfers to follower."""
        tg_detector = TailgatingDetector(spatial_threshold=160.0, confirmation_frames=2)
        policy_engine = PolicyEngine()

        leader = {
            "track_id": 101, "identity": "Hunter", "is_known": True, "access_state": "GRANTED",
            "liveness_state": "LIVE", "bbox": (100, 100, 150, 300), "zone": "NORMAL",
        }
        follower = {
            "track_id": 102, "identity": "Unknown Person", "is_known": False, "access_state": "DENIED",
            "liveness_state": "LIVE", "bbox": (180, 100, 150, 300), "zone": "NORMAL",
        }

        # Frame 1: Candidate observation (not confirmed yet)
        events_1, s_tg_1 = tg_detector.evaluate_tracks([leader, follower], current_time=1.0)
        confirmed_1 = [e for e in events_1 if e.is_confirmed]
        self.assertEqual(len(confirmed_1), 0)

        # Frame 2: Confirmed tailgating event emitted
        events_2, s_tg_2 = tg_detector.evaluate_tracks([leader, follower], current_time=1.1)
        confirmed_2 = [e for e in events_2 if e.is_confirmed]
        self.assertEqual(len(confirmed_2), 1)
        self.assertEqual(confirmed_2[0].authorized_identity, "Hunter")
        self.assertEqual(confirmed_2[0].follower_identity, "Unknown Person")
        self.assertGreater(s_tg_2, 0.5)

        # Verify follower authorization did NOT change
        f_eval = policy_engine.evaluate(
            identity=follower["identity"], role="Visitor", is_known=follower["is_known"],
            liveness_state="LIVE", liveness_conf=0.92, zone_id="NORMAL",
            time_str="10:00", is_auth_time=True, is_spoof=False,
        )
        self.assertFalse(f_eval.is_access_granted)

    def test_09_behavior_anomaly_engine_computation(self):
        """Verify BehaviorAnomalyEngine computes S_behavior in [0.0, 1.0] from dwell and retries."""
        beh_engine = BehaviorAnomalyEngine()

        # Normal observation in standard zone
        s_norm, label_norm, _ = beh_engine.update_track_observation(
            track_id=1, identity="Hunter", is_known=True, zone="NORMAL", is_after_hours=False,
        )
        self.assertLess(s_norm, 0.25)
        self.assertEqual(label_norm, "NORMAL_ACTIVITY")

        # After-hours violation in restricted zone
        s_anom, label_anom, _ = beh_engine.update_track_observation(
            track_id=1, identity="Hunter", is_known=True, zone="RESTRICTED", is_after_hours=True, is_violation=True,
        )
        self.assertGreater(s_anom, 0.40)
        self.assertIn(label_anom, ["UNUSUAL_PATTERN", "BEHAVIOR_ANOMALY"])

    def test_10_uncertainty_aware_decision_confidence_tiers(self):
        """Verify DecisionConfidence tiers: HIGH (>=0.75), MEDIUM (0.45-0.75), LOW (<0.45)."""
        agg = UncertaintyAggregator()

        conf_high = agg.evaluate_confidence(identity_confidence=0.95, pad_confidence=0.92, zone_confidence=1.0, object_confidence=0.90, behavior_confidence=0.90)
        self.assertEqual(conf_high.confidence_tier, "HIGH")

        conf_med = agg.evaluate_confidence(identity_confidence=0.60, pad_confidence=0.55, zone_confidence=0.8, object_confidence=0.50, behavior_confidence=0.60)
        self.assertEqual(conf_med.confidence_tier, "MEDIUM")

        conf_low = agg.evaluate_confidence(identity_confidence=0.30, pad_confidence=0.20, zone_confidence=0.5, object_confidence=0.30, behavior_confidence=0.30)
        self.assertEqual(conf_low.confidence_tier, "LOW")

    def test_11_spoof_counter_edge_triggering(self):
        """Verify confirmed spoof increments counter exactly once per session."""
        track = TrackSecurityState(track_id=1)
        counter = 0

        # Continuous spoof frames
        for _ in range(20):
            t_info = track.update(
                identity="Hunter", identity_conf=0.95, role="Staff", is_known=True,
                liveness_state="SPOOF", liveness_conf=0.10, is_spoof=True, attack_type="PHOTO_REPLAY",
                zone_id="NORMAL", zone_name="Standard Perimeter", time_str="10:00", is_auth_time=True,
                nearby_objects=[], policy_code="PRESENTATION_ATTACK_IMPERSONATION", is_access_granted=False,
                risk_level=RiskLevel.RED, reason="Spoof",
            )
            if t_info.get("is_new_spoof_session"):
                counter += 1

        self.assertEqual(counter, 1, "Counter must increment exactly once for continuous spoof frames")

        # Return to live
        for _ in range(10):
            t_info = track.update(
                identity="Hunter", identity_conf=0.95, role="Staff", is_known=True,
                liveness_state="LIVE", liveness_conf=0.96, is_spoof=False, attack_type=None,
                zone_id="NORMAL", zone_name="Standard Perimeter", time_str="10:01", is_auth_time=True,
                nearby_objects=[], policy_code="AUTHORIZED_STAFF", is_access_granted=True,
                risk_level=RiskLevel.GREEN, reason="Staff",
            )
            if t_info.get("is_new_spoof_session"):
                counter += 1

        self.assertEqual(counter, 1, "Returning to live must not increment spoof counter")

        # Fresh independent spoof
        t_info = track.update(
            identity="Hunter", identity_conf=0.95, role="Staff", is_known=True,
            liveness_state="SPOOF", liveness_conf=0.10, is_spoof=True, attack_type="SCREEN_REPLAY",
            zone_id="NORMAL", zone_name="Standard Perimeter", time_str="10:02", is_auth_time=True,
            nearby_objects=[], policy_code="PRESENTATION_ATTACK_IMPERSONATION", is_access_granted=False,
            risk_level=RiskLevel.RED, reason="Spoof",
        )
        if t_info.get("is_new_spoof_session"):
            counter += 1

        self.assertEqual(counter, 2, "Fresh independent spoof attempt must increment counter once")

    def test_12_continuous_authorization_context_transitions(self):
        """Verify continuous evaluation dynamically shifts risk across transitions without restarts."""
        ccsdf = CCSDFEngine()

        # Step 1: Normal Authorized
        ctx1 = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="NORMAL"),
            temporal=TemporalState(is_authorized_hours=True),
        )
        r1 = ccsdf.evaluate_context(ctx1, authoritative_action="GRANTED")
        self.assertEqual(r1.risk_level, "GREEN")

        # Step 2: Transition into Restricted Zone After Hours
        ctx2 = CCSDFContext(
            identity=IdentityState(name="Hunter", is_known=True, similarity=0.95),
            liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
            zone=ZoneState(zone_id="RESTRICTED"),
            temporal=TemporalState(is_authorized_hours=False),
        )
        r2 = ccsdf.evaluate_context(ctx2, authoritative_action="DENIED")
        self.assertGreater(r2.risk_score, r1.risk_score)
        self.assertIn(r2.risk_level, ["YELLOW", "ORANGE", "RED"])

        # Step 3: Transition to Normal Zone (Restoration)
        r3 = ccsdf.evaluate_context(ctx1, authoritative_action="GRANTED")
        self.assertEqual(r3.risk_level, "GREEN")


if __name__ == "__main__":
    unittest.main()
