"""
WatchGuard Vision - Development Data Reset Tool
------------------------------------------------
Safely clears test records, face profiles, and temporary evidence files
while preserving models, schema, and system configuration.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.database.db_manager import DatabaseManager


def main():
    print("==================================================")
    print("  WATCHGUARD VISION - CLEAN DEVELOPMENT RESET")
    print("==================================================")
    print("\nThis operation will:")
    print("  1. Create a timestamped SQLite database backup in data/backups/")
    print("  2. Clear all registered face profiles from the database")
    print("  3. Clear all security events, object detections, and system logs")
    print("  4. Delete temporary evidence snapshots and face crop files")
    print("  5. PRESERVE database schema, YOLOv8/YuNet/SFace models, code, and config.")
    print("  6. Leave exactly 0 registered face profiles and 0 security events.")
    print("")

    if len(sys.argv) > 1 and sys.argv[1] in ["--force", "-y", "--confirm"]:
        confirm = True
    else:
        ans = input("Are you sure you want to proceed with this reset? (yes/no): ").strip().lower()
        confirm = ans in ["yes", "y"]

    if not confirm:
        print("\n[ABORTED] Reset cancelled by user. No data was modified.")
        sys.exit(0)

    print("\nExecuting clean development reset...")
    db = DatabaseManager()
    summary = db.reset_development_data(backup=True, confirm=True)

    print("\n>>> CLEAN DEVELOPMENT RESET SUCCESSFUL <<<")
    print(f"Backup created at: {summary['backup_path']}")
    print(f"Deleted face profiles: {summary['deleted_records'].get('face_registrations', 0)}")
    print(f"Deleted security events: {summary['deleted_records'].get('security_events', 0)}")
    print(f"Deleted detected object records: {summary['deleted_records'].get('detected_objects', 0)}")
    print(f"Deleted temporary evidence files: {summary['deleted_evidence_images']}")
    print(f"Deleted temporary face crop files: {summary['deleted_face_images']}")
    print(f"Preserved AI models: {summary['preserved_models']}")
    print(f"Preserved demo portraits: {summary['preserved_demo_faces']}")
    print(f"Remaining registered faces: {summary['remaining_face_profiles']}")
    print(f"Remaining security events: {summary['remaining_security_events']}")


if __name__ == "__main__":
    main()
