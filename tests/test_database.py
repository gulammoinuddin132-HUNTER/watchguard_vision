"""
Unit tests for DatabaseManager and SQLite schema
"""

import tempfile
import unittest
from pathlib import Path

from src.database.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_watchguard.db"
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_initialization(self):
        """Verify all 6 tables exist after initialization."""
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            expected = {
                "users",
                "face_registrations",
                "detected_objects",
                "security_events",
                "ocr_documents",
                "system_logs",
            }
            self.assertTrue(expected.issubset(tables), f"Missing tables: {expected - tables}")

    def test_user_creation_and_retrieval(self):
        """Test creating and retrieving a user."""
        user_id = self.db.create_user(
            username="john_doe",
            full_name="John Doe",
            role="Security Analyst",
            access_level=2,
        )
        self.assertIsInstance(user_id, int)
        self.assertGreater(user_id, 0)

        users = self.db.get_users()
        usernames = [u["username"] for u in users]
        self.assertIn("john_doe", usernames)

    def test_security_event_logging(self):
        """Test logging and retrieving security events."""
        event_id = self.db.log_security_event(
            event_type="PERSON_DETECTED",
            risk_level="GREEN",
            description="Authorized personnel detected at entry.",
            person_name="John Doe",
        )
        self.assertIsInstance(event_id, int)

        recent = self.db.get_recent_security_events(limit=5)
        self.assertGreater(len(recent), 0)
        self.assertEqual(recent[0]["event_type"], "PERSON_DETECTED")
        self.assertEqual(recent[0]["risk_level"], "GREEN")
        self.assertEqual(recent[0]["person_name"], "John Doe")

    def test_reset_development_data_requires_confirmation(self):
        """Verify reset_development_data raises ValueError without confirm=True."""
        with self.assertRaises(ValueError):
            self.db.reset_development_data(confirm=False)

    def test_reset_development_data_execution_and_backup(self):
        """Verify reset_development_data clears records, creates backup, and preserves schema."""
        # Insert sample data
        u_id = self.db.create_user("temp_user", "Temp User", "Staff", 1)
        self.db.register_face(u_id, b"\x00" * 512, "sample_path.jpg")
        self.db.log_security_event("TEST_EVENT", "YELLOW", "Sample alert")
        self.db.log_detected_object("sess_1", "laptop", 0.9, "{}")
        self.db.log_ocr_document("text", "SECRET", "RESTRICTED")

        self.assertGreater(self.db.get_total_event_count(), 0)
        self.assertGreater(len(self.db.get_active_face_registrations()), 0)

        # Run reset with backup and explicit confirmation
        summary = self.db.reset_development_data(backup=True, confirm=True)

        self.assertEqual(summary["status"], "CLEAN_RESET_COMPLETE")
        self.assertEqual(self.db.get_total_event_count(), 0)
        self.assertEqual(len(self.db.get_active_face_registrations()), 0)
        self.assertEqual(summary["remaining_face_profiles"], 0)
        self.assertEqual(summary["remaining_security_events"], 0)

        # Verify backup was created
        if summary["backup_path"]:
            self.assertTrue(Path(summary["backup_path"]).exists())


if __name__ == "__main__":
    unittest.main()
