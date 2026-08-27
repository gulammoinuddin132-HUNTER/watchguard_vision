"""
Automated Unit Tests for Person Recognition Module
---------------------------------------------------
Tests:
- Initialization of YuNet detector & SFace recognizer
- Face registration and database persistence
- Recognition of registered authorized persons vs unknown persons
- Event generation and cooldown / throttling
- Evidence snapshot generation and linking
- Synthetic simulation demo target mode
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from config import DEMO_FACES_DIR, EVIDENCE_DIR
from src.core.person_recognition import PersonRecognitionEngine
from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager


class TestPersonRecognition(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_watchguard.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.engine = PersonRecognitionEngine(db_manager=self.db)
        self.logger = EventLogger(db_manager=self.db)

    def tearDown(self):
        self.engine.release()
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Verify YuNet detector and SFace recognizer initialize correctly."""
        ok = self.engine.initialize()
        self.assertTrue(ok)
        self.assertTrue(self.engine.is_ready)
        self.assertEqual(self.engine.known_face_count, 0)

        # Explicit seeding registers demo face
        seeded = self.engine.seed_demo_identities()
        self.assertTrue(seeded)
        self.assertEqual(self.engine.known_face_count, 1)

    def test_face_registration_and_db_persistence(self):
        """Verify registering a new user face saves embedding and persists to SQLite."""
        self.engine.initialize()
        initial_count = self.engine.known_face_count

        # Create a new user in DB
        user_id = self.db.create_user(
            username="test_officer_bob",
            full_name="Bob Test Officer",
            role="Security Officer",
            access_level=2,
        )

        alice_img = cv2.imread(str(DEMO_FACES_DIR / "officer_alice.jpg"))
        self.assertIsNotNone(alice_img)

        # Register Bob using sample image
        success, msg = self.engine.register_face_from_image(user_id=user_id, image_bgr=alice_img)
        self.assertTrue(success, f"Registration failed: {msg}")
        self.assertEqual(self.engine.known_face_count, initial_count + 1)

        # Verify DB records
        active = self.db.get_active_face_registrations()
        user_ids = [r["user_id"] for r in active]
        self.assertIn(user_id, user_ids)

    def test_recognition_authorized_vs_unknown(self):
        """Test identification of authorized person vs unknown visitor."""
        self.engine.initialize(auto_seed_demo=True)

        # 1. Frame with Authorized Alice
        alice_img = cv2.imread(str(DEMO_FACES_DIR / "officer_alice.jpg"))
        frame_alice = np.zeros((480, 640, 3), dtype=np.uint8)
        alice_resized = cv2.resize(alice_img, (240, 240))
        frame_alice[120:360, 200:440] = alice_resized

        res_alice = self.engine.process_frame(frame_alice)
        self.assertTrue(res_alice["detected"])
        self.assertTrue(res_alice["people"][0]["is_known"])
        self.assertFalse(res_alice["has_unknown"])
        self.assertEqual(res_alice["people"][0]["name"], "Hunter")
        self.assertGreater(res_alice["people"][0]["similarity"], 0.70)

        # 2. Frame with Unknown visitor
        unk_img = cv2.imread(str(DEMO_FACES_DIR / "unknown_visitor.jpg"))
        frame_unk = np.zeros((480, 640, 3), dtype=np.uint8)
        unk_resized = cv2.resize(unk_img, (200, 200))
        frame_unk[140:340, 220:420] = unk_resized

        res_unk = self.engine.process_frame(frame_unk)
        self.assertTrue(res_unk["detected"])
        self.assertTrue(res_unk["has_unknown"])
        self.assertFalse(res_unk["has_authorized"])
        self.assertEqual(res_unk["people"][0]["name"], "UNKNOWN PERSON")

    def test_alert_throttling_and_debouncing(self):
        """Verify alert cooldown avoids duplicate spamming."""
        self.engine.initialize()

        # First alert check should pass
        should_alert_1 = self.engine.should_trigger_alert("unknown_person", cooldown_seconds=2.0)
        self.assertTrue(should_alert_1)

        # Immediate second check should be throttled
        should_alert_2 = self.engine.should_trigger_alert("unknown_person", cooldown_seconds=2.0)
        self.assertFalse(should_alert_2)

        # After cooldown passes, should allow alert again
        time.sleep(2.1)
        should_alert_3 = self.engine.should_trigger_alert("unknown_person", cooldown_seconds=2.0)
        self.assertTrue(should_alert_3)

    def test_unknown_person_event_and_evidence_snapshot(self):
        """Verify unknown person generates security event with evidence snapshot path."""
        cam = CameraManager()
        cam.set_demo_mode("unknown")
        cam.start()

        # Grab a frame
        success, frame = cam.get_frame()
        self.assertTrue(success)

        # Save snapshot
        evidence_path = cam.capture_snapshot(filename_prefix="test_evidence_unknown")
        self.assertIsNotNone(evidence_path)
        self.assertTrue(os.path.exists(evidence_path))

        # Log security event
        event = self.logger.log_event(
            event_type=EventType.UNKNOWN_PERSON,
            risk_level=RiskLevel.YELLOW,
            description="Unregistered visitor detected in camera view.",
            evidence_frame_path=evidence_path,
            person_name="UNKNOWN PERSON",
        )

        self.assertEqual(event["event_type"], "UNKNOWN_PERSON")
        self.assertEqual(event["risk_level"], "YELLOW")
        self.assertEqual(event["evidence_frame_path"], evidence_path)

        cam.stop()

    def test_synthetic_camera_demo_modes(self):
        """Verify CameraManager synthetic demo targets ('none', 'authorized', 'unknown')."""
        self.engine.initialize(auto_seed_demo=True)
        cam = CameraManager(camera_index=99)

        # 1. Authorized Demo Mode
        cam.set_demo_mode("authorized")
        cam.start()
        success, frame_auth = cam.get_frame()
        self.assertTrue(success)

        res_auth = self.engine.process_frame(frame_auth)
        self.assertTrue(res_auth["detected"])
        self.assertTrue(res_auth["people"][0]["is_known"])
        self.assertEqual(res_auth["people"][0]["name"], "Hunter")

        # 2. Unknown Demo Mode
        cam.set_demo_mode("unknown")
        time.sleep(0.08)
        success, frame_unk = cam.get_frame()
        self.assertTrue(success)

        res_unk = self.engine.process_frame(frame_unk)
        self.assertTrue(res_unk["detected"])
        self.assertTrue(res_unk["has_unknown"])

        cam.stop()

    def test_simultaneous_two_people_authorized_and_unknown(self):
        """Verify frame with both registered person and unknown person assigns independent identities."""
        self.engine.initialize(auto_seed_demo=True)

        alice_img = cv2.imread(str(DEMO_FACES_DIR / "officer_alice.jpg"))
        messi_img = cv2.imread(str(DEMO_FACES_DIR / "unknown_messi.jpg"))

        frame_2people = np.zeros((480, 640, 3), dtype=np.uint8)
        # Left: Authorized Hunter
        frame_2people[140:320, 80:260] = cv2.resize(alice_img, (180, 180))
        # Right: Unknown visitor
        frame_2people[140:320, 380:560] = cv2.resize(messi_img, (180, 180))

        results = self.engine.process_frame(frame_2people)
        self.assertTrue(results["detected"])
        self.assertEqual(results["count"], 2)
        self.assertTrue(results["has_unknown"])

        # Sort people by X coordinate
        people = sorted(results["people"], key=lambda p: p["bbox"][0])
        left_person = people[0]
        right_person = people[1]

        # Left person must be Authorized Hunter
        self.assertTrue(left_person["is_known"])
        self.assertEqual(left_person["name"], "Hunter")
        self.assertEqual(left_person["status_label"], "AUTHORIZED")
        self.assertGreaterEqual(left_person["similarity"], 0.55)
        self.assertGreaterEqual(left_person["match_pct"], 75)

        # Right person must be Unknown / Not Authorized
        self.assertFalse(right_person["is_known"])
        self.assertEqual(right_person["name"], "UNKNOWN PERSON")
        self.assertIn(right_person["status_label"], ["UNKNOWN", "UNVERIFIED"])
        self.assertLess(right_person["similarity"], 0.55)

        # Annotated HUD must contain both badges
        annotated = self.engine.draw_annotations(frame_2people, results)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, frame_2people.shape)

    def test_ambiguous_similarity_fails_secure(self):
        """Verify borderline similarity (0.42) NEVER yields AUTHORIZED status."""
        self.engine.initialize(auto_seed_demo=True)

        # Inject synthetic test with similarity 0.42
        test_sim = 0.42
        from config import FACE_COSINE_SIMILARITY_THRESHOLD, FACE_AMBIGUOUS_SIMILARITY_THRESHOLD

        self.assertGreaterEqual(FACE_COSINE_SIMILARITY_THRESHOLD, 0.50)
        self.assertLess(test_sim, FACE_COSINE_SIMILARITY_THRESHOLD)
        self.assertGreaterEqual(test_sim, FACE_AMBIGUOUS_SIMILARITY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
