"""
Unit and Integration Tests for WatchGuard 2.0 Phase 4.2
--------------------------------------------------------
Multi-Modal Person Entity Association Test Suite:
1. Face + overlapping person object = one entity
2. Face + non-overlapping person object = two entities
3. Two faces + two person objects = two entities
4. Hunter + cell phone = one person + one object
5. Unknown person + person object = one entity
6. Real Hunter + phone photo of Hunter = two distinct entities
7. Voice "What do you observe?" uses deduplicated entities
8. Voice "What objects are detected?" uses deduplicated entities
9. Association remains stable across frames
10. Existing Phase 4.1 tests remain intact
"""

import unittest
from datetime import datetime, time as dtime
import tempfile
from pathlib import Path

from src.events.event_types import RiskLevel
from src.database.db_manager import DatabaseManager
from src.core.person_entity import PersonEntity, PersonEntityFusionEngine, compute_containment_score, compute_iof
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant


class TestPersonEntityPhase42(unittest.TestCase):
    """Test suite verifying all Phase 4.2 specifications."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_watchguard_phase42.db"
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.context_engine = ContextEngine(db_manager=self.db_manager)
        self.fusion_engine = PersonEntityFusionEngine(proximity_threshold_pixels=120.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Face + overlapping person object = one entity
    def test_01_face_plus_overlapping_person_object_fused(self):
        people = [{
            "track_id": 1,
            "name": "Hunter",
            "is_known": True,
            "role": "Security Lead",
            "similarity": 0.89,
            "is_spoof": False,
            "liveness_confidence": 0.95,
            "bbox": (100, 100, 60, 60),  # Face box
        }]
        objects = [{
            "label": "person",
            "confidence": 0.88,
            "bbox": (80, 90, 120, 260),  # Overlapping person body box
        }]

        entities, non_person_objs, summary = self.fusion_engine.fuse(people, objects)

        # Must produce exactly 1 PersonEntity
        self.assertEqual(len(entities), 1)
        e = entities[0]
        self.assertEqual(e.identity, "Hunter")
        self.assertTrue(e.person_object_detected)
        self.assertEqual(e.face_bbox, (100, 100, 60, 60))
        self.assertEqual(e.body_bbox, (80, 90, 120, 260))
        # Non-person objects should be empty
        self.assertEqual(len(non_person_objs), 0)
        self.assertEqual(summary, "None")

    # 2. Face + non-overlapping person object = two entities
    def test_02_face_plus_separate_person_object_two_entities(self):
        people = [{
            "track_id": 1,
            "name": "Hunter",
            "is_known": True,
            "role": "Staff",
            "similarity": 0.88,
            "is_spoof": False,
            "liveness_confidence": 0.95,
            "bbox": (50, 50, 60, 60),  # Face on left
        }]
        objects = [{
            "label": "person",
            "confidence": 0.82,
            "bbox": (400, 100, 120, 250),  # Person body on far right
        }]

        entities, non_person_objs, summary = self.fusion_engine.fuse(people, objects)

        # Must produce 2 distinct PersonEntities
        self.assertEqual(len(entities), 2)
        identities = [e.identity for e in entities]
        self.assertIn("Hunter", identities)
        self.assertIn("Unknown Person", identities)

    # 3. Two faces + two person objects = two entities
    def test_03_two_faces_two_person_objects_two_entities(self):
        people = [
            {"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.90, "is_spoof": False, "bbox": (100, 100, 60, 60)},
            {"track_id": 2, "name": "Alice", "is_known": True, "role": "Staff", "similarity": 0.85, "is_spoof": False, "bbox": (350, 100, 60, 60)},
        ]
        objects = [
            {"label": "person", "confidence": 0.88, "bbox": (80, 90, 120, 260)},   # Overlaps Hunter
            {"label": "person", "confidence": 0.86, "bbox": (330, 90, 120, 260)},  # Overlaps Alice
        ]

        entities, non_person_objs, summary = self.fusion_engine.fuse(people, objects)

        # Must produce exactly 2 entities
        self.assertEqual(len(entities), 2)
        names = {e.identity for e in entities}
        self.assertEqual(names, {"Hunter", "Alice"})
        self.assertTrue(all(e.person_object_detected for e in entities))

    # 4. Hunter + cell phone = one person + one object
    def test_04_hunter_plus_cell_phone(self):
        people = [{
            "track_id": 1,
            "name": "Hunter",
            "is_known": True,
            "role": "Staff",
            "similarity": 0.88,
            "is_spoof": False,
            "liveness_confidence": 0.95,
            "bbox": (100, 100, 60, 60),
        }]
        objects = [
            {"label": "person", "confidence": 0.90, "bbox": (80, 90, 120, 260)},
            {"label": "cell phone", "confidence": 0.85, "bbox": (150, 180, 30, 50)},
        ]

        entities, non_person_objs, summary = self.fusion_engine.fuse(people, objects)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].identity, "Hunter")
        self.assertEqual(len(non_person_objs), 1)
        self.assertEqual(non_person_objs[0]["label"], "cell phone")
        self.assertEqual(summary, "cell phone")
        self.assertEqual(len(entities[0].associated_objects), 1)
        self.assertEqual(entities[0].associated_objects[0]["object_label"], "cell phone")

    # 5. Unknown person + person object = one entity
    def test_05_unknown_person_plus_person_object(self):
        people = [{
            "track_id": 5,
            "name": "Unknown Person",
            "is_known": False,
            "role": "Visitor",
            "confidence": 0.91,
            "is_spoof": False,
            "liveness_confidence": 0.90,
            "bbox": (100, 100, 60, 60),
        }]
        objects = [{
            "label": "person",
            "confidence": 0.84,
            "bbox": (80, 90, 120, 260),
        }]

        entities, non_person_objs, summary = self.fusion_engine.fuse(people, objects)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].identity, "Unknown Person")
        self.assertTrue(entities[0].person_object_detected)

    # 6. Real Hunter + phone photo of Hunter = two distinct entities
    def test_06_real_hunter_plus_spoof_phone_photo(self):
        people = [
            # Real person face
            {"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.92, "is_spoof": False, "liveness_confidence": 0.96, "bbox": (80, 100, 60, 60)},
            # Phone photo face
            {"track_id": 2, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": True, "attack_type": "PHOTO_REPLAY_ATTACK", "liveness_confidence": 0.12, "bbox": (350, 150, 50, 50)},
        ]
        objects = [
            # Real person body
            {"label": "person", "confidence": 0.90, "bbox": (60, 90, 120, 260)},
            # Phone object holding spoof face
            {"label": "cell phone", "confidence": 0.87, "bbox": (340, 140, 70, 90)},
        ]

        entities, non_person_objs, summary = self.fusion_engine.fuse(people, objects)

        # Must produce exactly 2 entities (Real person + Spoof presentation entity)
        self.assertEqual(len(entities), 2)

        live_entities = [e for e in entities if not e.is_spoof]
        spoof_entities = [e for e in entities if e.is_spoof]

        self.assertEqual(len(live_entities), 1)
        self.assertEqual(live_entities[0].identity, "Hunter")
        self.assertTrue(live_entities[0].person_object_detected)

        self.assertEqual(len(spoof_entities), 1)
        self.assertEqual(spoof_entities[0].identity, "Hunter")
        self.assertTrue(spoof_entities[0].is_spoof)
        self.assertFalse(spoof_entities[0].person_object_detected)

    # 7. Voice "What do you observe?" uses deduplicated entities
    def test_07_voice_what_do_you_observe(self):
        assistant = VoiceAssistant(
            db_manager=self.db_manager,
            context_engine=self.context_engine,
        )

        # Case A: Hunter + person object (Should NOT say "Hunter along with person")
        person_data_a = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (100, 100, 60, 60)}],
        }
        object_data_a = {"objects": [{"label": "person", "confidence": 0.85, "bbox": (80, 90, 120, 260)}]}
        self.context_engine.evaluate_context(person_data_a, object_data_a)

        resp_a = assistant._query_scene_overview()
        self.assertIn("Hunter", resp_a)
        self.assertNotIn("along with person", resp_a)
        self.assertNotIn("along with 1 person", resp_a)

        # Case B: Hunter + cell phone
        object_data_b = {
            "objects": [
                {"label": "person", "confidence": 0.85, "bbox": (80, 90, 120, 260)},
                {"label": "cell phone", "confidence": 0.90, "bbox": (150, 180, 30, 50)},
            ]
        }
        self.context_engine.evaluate_context(person_data_a, object_data_b)
        resp_b = assistant._query_scene_overview()
        self.assertIn("Hunter", resp_b)
        self.assertIn("cell phone", resp_b)
        self.assertNotIn("along with person", resp_b)

        # Case C: Unknown Person alone
        person_data_c = {
            "detected": True, "has_authorized": False, "has_unknown": True, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 3, "name": "Unknown Person", "is_known": False, "role": "Visitor", "confidence": 0.89, "is_spoof": False, "liveness_confidence": 0.90, "bbox": (100, 100, 60, 60)}],
        }
        object_data_c = {"objects": [{"label": "person", "confidence": 0.85, "bbox": (80, 90, 120, 260)}]}
        self.context_engine.evaluate_context(person_data_c, object_data_c)
        resp_c = assistant._query_scene_overview()
        self.assertIn("unregistered person", resp_c.lower())
        self.assertNotIn("along with person", resp_c)

    # 8. Voice "What objects are detected?" uses deduplicated entities
    def test_08_voice_detected_objects_deduplicated(self):
        assistant = VoiceAssistant(
            db_manager=self.db_manager,
            context_engine=self.context_engine,
        )

        person_data = {
            "detected": True, "has_authorized": True, "has_unknown": False, "has_spoof": False, "has_verifying": False,
            "people": [{"track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (100, 100, 60, 60)}],
        }
        object_data = {
            "objects": [
                {"label": "person", "confidence": 0.85, "bbox": (80, 90, 120, 260)},
                {"label": "cell phone", "confidence": 0.90, "bbox": (150, 180, 30, 50)},
            ]
        }
        self.context_engine.evaluate_context(person_data, object_data)

        resp = assistant._query_detected_objects()
        self.assertIn("Hunter", resp)
        self.assertIn("cell phone", resp)
        # Should not duplicate "Hunter, one person, and one cell phone"
        self.assertNotIn("Hunter, one person", resp)

    # 9. Association remains stable across frames
    def test_09_association_temporal_stability(self):
        people = [{
            "track_id": 1, "name": "Hunter", "is_known": True, "role": "Staff", "similarity": 0.88, "is_spoof": False, "liveness_confidence": 0.95, "bbox": (100, 100, 60, 60),
        }]
        objects = [{"label": "person", "confidence": 0.85, "bbox": (80, 90, 120, 260)}]

        for _ in range(10):
            entities, non_person_objs, _ = self.fusion_engine.fuse(people, objects)
            self.assertEqual(len(entities), 1)
            self.assertEqual(entities[0].identity, "Hunter")
            self.assertTrue(entities[0].person_object_detected)


if __name__ == "__main__":
    unittest.main()
