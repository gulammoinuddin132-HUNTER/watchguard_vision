"""
SQLite Database Manager
-----------------------
Thread-safe connection handling, table initialization, and structured CRUD operations
for WatchGuard Vision.
"""

import logging
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator

logger = logging.getLogger("WatchGuardVision.DatabaseManager")

from config import (
    DATABASE_PATH,
    DATA_DIR,
    EVIDENCE_DIR,
    FACES_DIR,
    MODELS_DIR,
    DEMO_FACES_DIR,
    MAX_DATABASE_NORMAL_EVENTS,
    EVENT_RETENTION_CHECK_INTERVAL,
)
from src.database.models import (
    User,
    FaceRegistration,
    DetectedObjectRecord,
    SecurityEventRecord,
    OCRDocumentRecord,
    SystemLogRecord,
    IncidentRecord,
)


class DatabaseManager:
    """Thread-safe SQLite Database Manager for WatchGuard Vision."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DATABASE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._event_insert_counter: int = 0
        self._initialize_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a connection and guarantees it is closed."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        """Creates tables and indexes if they do not already exist."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()

                # 1. Users Table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        full_name TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'User',
                        access_level INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL
                    )
                    """
                )

                # 2. Face Registrations Table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_registrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        face_encoding BLOB NOT NULL,
                        image_path TEXT,
                        registered_at TEXT NOT NULL,
                        is_active INTEGER NOT NULL DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                    )
                    """
                )

                # 3. Detected Objects Table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detected_objects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        bbox_json TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                    """
                )

                # 4. Security Events Table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS security_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        description TEXT NOT NULL,
                        evidence_frame_path TEXT,
                        person_name TEXT,
                        object_summary TEXT,
                        metadata_json TEXT,
                        timestamp TEXT NOT NULL
                    )
                    """
                )

                # 5. OCR Documents Table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ocr_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        raw_text TEXT NOT NULL,
                        sensitive_keywords_found TEXT,
                        document_status TEXT NOT NULL,
                        snapshot_path TEXT,
                        timestamp TEXT NOT NULL
                    )
                    """
                )

                # 6. System Logs Table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        level TEXT NOT NULL,
                        module TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                    """
                )

                # 7. Security Incidents Table (Phase 4)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS security_incidents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        incident_id TEXT UNIQUE NOT NULL,
                        event_id INTEGER,
                        track_id INTEGER,
                        person_name TEXT,
                        risk_level TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT 'DETECTED',
                        policy_code TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        zone TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        acknowledged_at TEXT,
                        resolved_at TEXT,
                        operator_action TEXT,
                        evidence_package_json TEXT,
                        FOREIGN KEY (event_id) REFERENCES security_events (id) ON DELETE SET NULL
                    )
                    """
                )

                # Create helpful indexes for performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON security_events (timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_risk ON security_events (risk_level)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_objects_timestamp ON detected_objects (timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs (timestamp DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_incidents_state ON security_incidents (state)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_incidents_created ON security_incidents (created_at DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_incidents_id ON security_incidents (incident_id)"
                )

                conn.commit()

    # --- Security Events Operations ---

    def log_security_event(
        self,
        event_type: str,
        risk_level: str,
        description: str,
        evidence_frame_path: Optional[str] = None,
        person_name: Optional[str] = None,
        object_summary: Optional[str] = None,
        metadata_json: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Inserts a security event into the database and returns the new event ID."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO security_events (
                        event_type, risk_level, description, evidence_frame_path,
                        person_name, object_summary, metadata_json, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_type,
                        risk_level,
                        description,
                        evidence_frame_path,
                        person_name,
                        object_summary,
                        metadata_json,
                        ts,
                    ),
                )
                conn.commit()
                event_id = cursor.lastrowid

            # Periodic non-critical event retention check (throttled by insert interval, never per-frame)
            self._event_insert_counter += 1
            if self._event_insert_counter % EVENT_RETENTION_CHECK_INTERVAL == 0:
                try:
                    self.enforce_event_retention(max_normal_events=MAX_DATABASE_NORMAL_EVENTS)
                except Exception:
                    pass

            return event_id

    def enforce_event_retention(self, max_normal_events: Optional[int] = None) -> int:
        """
        Enforces database event retention policy:
        - Keeps at most max_normal_events (default 500) normal/non-critical events (risk_level = 'GREEN').
        - Automatically deletes the oldest eligible non-critical events beyond the limit.
        - Strictly protects critical events (ORANGE, RED).
        - Strictly protects events associated with active (unresolved) security incidents.
        - Returns the number of pruned event records.
        """
        limit = max_normal_events if max_normal_events is not None else MAX_DATABASE_NORMAL_EVENTS
        if limit <= 0:
            return 0

        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                # Find and delete oldest GREEN events beyond the limit,
                # explicitly protecting any event associated with an active/unresolved incident
                delete_sql = """
                    DELETE FROM security_events
                    WHERE id IN (
                        SELECT id FROM security_events
                        WHERE risk_level = 'GREEN'
                          AND id NOT IN (
                              SELECT event_id FROM security_incidents
                              WHERE event_id IS NOT NULL AND state != 'RESOLVED'
                          )
                        ORDER BY id DESC
                        LIMIT -1 OFFSET ?
                    )
                """
                cursor.execute(delete_sql, (limit,))
                deleted_count = cursor.rowcount
                conn.commit()
                return max(0, deleted_count)

    def get_recent_security_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches the most recent security events."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, event_type, risk_level, description, evidence_frame_path,
                           person_name, object_summary, metadata_json, timestamp
                    FROM security_events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_total_event_count(self) -> int:
        """Returns the total number of security events stored in SQLite."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM security_events")
                return int(cursor.fetchone()[0])

    def get_events_in_last_minutes(self, minutes: int = 5, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetches events that occurred in the last N minutes, excluding internal voice query events."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, timestamp, risk_level, event_type, description, person_name, object_summary
                    FROM security_events
                    WHERE event_type != 'VOICE_COMMAND_RECEIVED'
                      AND timestamp >= datetime('now', 'localtime', ? || ' minutes')
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (f"-{minutes}", limit),
                )
                rows = cursor.fetchall()
                if not rows:
                    cursor.execute(
                        """
                        SELECT id, timestamp, risk_level, event_type, description, person_name, object_summary
                        FROM security_events
                        WHERE event_type != 'VOICE_COMMAND_RECEIVED'
                          AND timestamp >= datetime('now', ? || ' minutes')
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (f"-{minutes}", limit),
                    )
                    rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_alert_count_today(self) -> int:
        """Returns the total number of non-GREEN security alerts recorded today."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM security_events
                    WHERE risk_level != 'GREEN'
                      AND event_type != 'VOICE_COMMAND_RECEIVED'
                      AND (date(timestamp) = date('now', 'localtime') OR date(timestamp) = date('now'))
                    """
                )
                return int(cursor.fetchone()[0])

    def get_latest_alert_by_risk(self, risk_level: str) -> Optional[Dict[str, Any]]:
        """Returns the latest security alert matching the specified risk level (e.g. RED, ORANGE, YELLOW)."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, timestamp, risk_level, event_type, description, person_name, object_summary
                    FROM security_events
                    WHERE risk_level = ? AND event_type != 'VOICE_COMMAND_RECEIVED'
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (risk_level.upper(),),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def get_repeated_unauthorized_count(self, minutes: int = 30) -> int:
        """Counts recent unauthorized visitor detections and threat events."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM security_events
                    WHERE (event_type = 'UNKNOWN_PERSON' OR risk_level IN ('YELLOW', 'ORANGE', 'RED'))
                      AND event_type != 'VOICE_COMMAND_RECEIVED'
                      AND (timestamp >= datetime('now', 'localtime', ? || ' minutes')
                           OR timestamp >= datetime('now', ? || ' minutes'))
                    """,
                    (f"-{minutes}", f"-{minutes}"),
                )
                return int(cursor.fetchone()[0])

    def get_most_serious_event(self) -> Optional[Dict[str, Any]]:
        """Returns the highest-risk security alert in the database."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, timestamp, risk_level, event_type, description, person_name, object_summary
                    FROM security_events
                    WHERE event_type != 'VOICE_COMMAND_RECEIVED'
                    ORDER BY CASE risk_level
                        WHEN 'RED' THEN 1
                        WHEN 'ORANGE' THEN 2
                        WHEN 'YELLOW' THEN 3
                        ELSE 4 END, id DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    # --- User Management Operations ---

    def create_user(self, username: str, full_name: str, role: str = "User", access_level: int = 1) -> int:
        """Creates a new registered user."""
        now = datetime.now().isoformat()
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO users (username, full_name, role, access_level, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (username, full_name, role, access_level, now),
                )
                conn.commit()
                return cursor.lastrowid

    def get_users(self) -> List[Dict[str, Any]]:
        """Retrieves all registered users."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, username, full_name, role, access_level, created_at FROM users")
                return [dict(row) for row in cursor.fetchall()]

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user by username."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, username, full_name, role, access_level, created_at FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                return dict(row) if row else None

    # --- Face Registration Operations ---

    def register_face(self, user_id: int, face_encoding: bytes, image_path: str = "") -> int:
        """Saves a facial feature vector encoding linked to a user."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO face_registrations (user_id, face_encoding, image_path, registered_at, is_active)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (user_id, face_encoding, image_path, now),
                )
                conn.commit()
                return cursor.lastrowid

    def get_active_face_registrations(self) -> List[Dict[str, Any]]:
        """Retrieves all active face registration records with user identity details."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT fr.id, fr.user_id, u.username, u.full_name, u.role, u.access_level,
                           fr.face_encoding, fr.image_path, fr.registered_at
                    FROM face_registrations fr
                    JOIN users u ON fr.user_id = u.id
                    WHERE fr.is_active = 1
                    """
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    # --- System Log Operations ---

    def log_system_message(self, level: str, module: str, message: str) -> int:
        """Records a system-level audit/debug log."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO system_logs (level, module, message, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (level, module, message, now),
                )
                conn.commit()
                return cursor.lastrowid

    # --- Object & Document Log Operations ---

    def log_detected_object(self, session_id: str, label: str, confidence: float, bbox_json: str) -> int:
        """Records a detected object frame record."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO detected_objects (session_id, label, confidence, bbox_json, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, label, confidence, bbox_json, now),
                )
                conn.commit()
                return cursor.lastrowid

    def log_ocr_document(
        self,
        raw_text: str,
        sensitive_keywords_found: str,
        document_status: str,
        snapshot_path: Optional[str] = None,
    ) -> int:
        """Records an OCR analyzed document record."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO ocr_documents (raw_text, sensitive_keywords_found, document_status, snapshot_path, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (raw_text, sensitive_keywords_found, document_status, snapshot_path, now),
                )
                conn.commit()
                return cursor.lastrowid

    # --- Phase 4 Security Incidents Management ---

    def create_incident(self, incident: IncidentRecord) -> int:
        """Inserts a new security incident record and returns its primary key id."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO security_incidents (
                        incident_id, event_id, track_id, person_name, risk_level,
                        state, policy_code, reason, zone, created_at, updated_at,
                        acknowledged_at, resolved_at, operator_action, evidence_package_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        incident.incident_id,
                        incident.event_id,
                        incident.track_id,
                        incident.person_name,
                        incident.risk_level,
                        incident.state,
                        incident.policy_code,
                        incident.reason,
                        incident.zone,
                        incident.created_at,
                        incident.updated_at,
                        incident.acknowledged_at,
                        incident.resolved_at,
                        incident.operator_action,
                        incident.evidence_package_json,
                    ),
                )
                conn.commit()
                return cursor.lastrowid

    def get_incident_by_id(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single incident by its unique incident_id."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM security_incidents WHERE incident_id = ?",
                    (incident_id,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def get_active_incidents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns unresolved incidents (DETECTED, VERIFIED, ALERTED, ACKNOWLEDGED) ordered newest first."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM security_incidents
                    WHERE state != 'RESOLVED'
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_latest_incident(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent security incident regardless of state."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM security_incidents ORDER BY created_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    def acknowledge_incident(
        self,
        incident_id: str,
        operator_action: str = "Operator acknowledged via Dashboard",
    ) -> bool:
        """Transitions an incident to ACKNOWLEDGED state and records timestamp."""
        now = datetime.now().isoformat()
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE security_incidents
                    SET state = 'ACKNOWLEDGED',
                        acknowledged_at = ?,
                        updated_at = ?,
                        operator_action = ?
                    WHERE incident_id = ?
                    """,
                    (now, now, operator_action, incident_id),
                )
                conn.commit()
                return cursor.rowcount > 0

    def resolve_incident(
        self,
        incident_id: str,
        operator_action: str = "Operator marked incident resolved",
    ) -> bool:
        """Transitions an incident to RESOLVED state and records timestamp."""
        now = datetime.now().isoformat()
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE security_incidents
                    SET state = 'RESOLVED',
                        resolved_at = ?,
                        updated_at = ?,
                        operator_action = ?
                    WHERE incident_id = ?
                    """,
                    (now, now, operator_action, incident_id),
                )
                conn.commit()
                return cursor.rowcount > 0

    def update_incident_state(
        self,
        incident_id: str,
        new_state: str,
        operator_action: Optional[str] = None,
    ) -> bool:
        """Updates the state of an incident record."""
        now = datetime.now().isoformat()
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                if operator_action:
                    cursor.execute(
                        """
                        UPDATE security_incidents
                        SET state = ?, updated_at = ?, operator_action = ?
                        WHERE incident_id = ?
                        """,
                        (new_state, now, operator_action, incident_id),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE security_incidents
                        SET state = ?, updated_at = ?
                        WHERE incident_id = ?
                        """,
                        (new_state, now, incident_id),
                    )
                conn.commit()
                return cursor.rowcount > 0

    def get_incident_stats(self) -> Dict[str, int]:
        """Returns counts of incidents grouped by state."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state, COUNT(*) as cnt FROM security_incidents GROUP BY state"
                )
                stats = {row["state"]: row["cnt"] for row in cursor.fetchall()}
                cursor.execute("SELECT COUNT(*) FROM security_incidents")
                stats["TOTAL"] = cursor.fetchone()[0]
                return stats

    def get_analytics_counts(
        self,
        window_seconds: Optional[float] = None,
        since_midnight: bool = False,
    ) -> Dict[str, int]:
        """
        Derives security metrics over a bounded time window (or since midnight).
        Uses efficient indexed SQL queries without scanning the entire table.
        """
        now = datetime.now()
        if since_midnight:
            cutoff = datetime(now.year, now.month, now.day, 0, 0, 0)
            cutoff_iso = cutoff.isoformat()
        elif window_seconds is not None:
            cutoff = now - timedelta(seconds=window_seconds)
            cutoff_iso = cutoff.isoformat()
        else:
            cutoff_iso = None

        time_clause_events = "WHERE timestamp >= ?" if cutoff_iso else ""
        params_events = (cutoff_iso,) if cutoff_iso else ()

        time_clause_incidents = "WHERE created_at >= ?" if cutoff_iso else ""
        params_incidents = (cutoff_iso,) if cutoff_iso else ()

        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()

                # Active incidents (state != 'RESOLVED')
                cursor.execute(
                    "SELECT COUNT(*) FROM security_incidents WHERE state != 'RESOLVED'"
                )
                active_incidents = cursor.fetchone()[0]

                # Resolved incidents in window
                if cutoff_iso:
                    cursor.execute(
                        "SELECT COUNT(*) FROM security_incidents WHERE state = 'RESOLVED' AND updated_at >= ?",
                        (cutoff_iso,),
                    )
                else:
                    cursor.execute(
                        "SELECT COUNT(*) FROM security_incidents WHERE state = 'RESOLVED'"
                    )
                resolved_incidents = cursor.fetchone()[0]

                # Spoof attempts
                spoof_sql = f"""
                    SELECT COUNT(*) FROM security_events
                    {time_clause_events}
                    {'AND' if time_clause_events else 'WHERE'}
                    (event_type LIKE '%SPOOF%' OR event_type LIKE '%PRESENTATION_ATTACK%' OR description LIKE '%spoof%' OR description LIKE '%presentation attack%')
                """
                cursor.execute(spoof_sql, params_events)
                spoof_attempts = cursor.fetchone()[0]

                # After-hours violations
                after_hours_sql = f"""
                    SELECT COUNT(*) FROM security_events
                    {time_clause_events}
                    {'AND' if time_clause_events else 'WHERE'}
                    (event_type LIKE '%AFTER_HOURS%' OR description LIKE '%after-hours%' OR description LIKE '%operating hours%')
                """
                cursor.execute(after_hours_sql, params_events)
                after_hours_violations = cursor.fetchone()[0]

                # Restricted zone entries
                zone_entries_sql = f"""
                    SELECT COUNT(*) FROM security_events
                    {time_clause_events}
                    {'AND' if time_clause_events else 'WHERE'}
                    (event_type LIKE '%RESTRICTED%' OR description LIKE '%Restricted Area%' OR metadata_json LIKE '%Restricted Area%')
                """
                cursor.execute(zone_entries_sql, params_events)
                restricted_zone_entries = cursor.fetchone()[0]

                # Unknown visitors
                unknown_sql = f"""
                    SELECT COUNT(DISTINCT id)
                    FROM security_events
                    {time_clause_events}
                    {'AND' if time_clause_events else 'WHERE'}
                    (person_name IS NULL OR person_name = '' OR person_name = 'Unknown' OR event_type LIKE '%UNKNOWN%' OR description LIKE '%unregistered%' OR description LIKE '%unknown%')
                """
                cursor.execute(unknown_sql, params_events)
                unknown_visitors = cursor.fetchone()[0]

                # Distinct people seen
                distinct_sql = f"""
                    SELECT COUNT(DISTINCT CASE 
                        WHEN person_name IS NOT NULL AND person_name != '' AND person_name != 'Unknown' THEN person_name 
                        ELSE 'EVENT_' || id 
                    END)
                    FROM security_events
                    {time_clause_events}
                    {'AND' if time_clause_events else 'WHERE'}
                    (event_type NOT LIKE '%CAMERA%' AND event_type NOT LIKE '%VOICE%')
                """
                cursor.execute(distinct_sql, params_events)
                distinct_people_seen = cursor.fetchone()[0]

                # Authorized people seen
                auth_sql = f"""
                    SELECT COUNT(DISTINCT person_name)
                    FROM security_events
                    {time_clause_events}
                    {'AND' if time_clause_events else 'WHERE'}
                    (risk_level = 'GREEN' AND person_name IS NOT NULL AND person_name != '' AND person_name != 'Unknown')
                """
                cursor.execute(auth_sql, params_events)
                authorized_people_seen = cursor.fetchone()[0]

                # Denied access attempts
                denied_sql = f"""
                    SELECT COUNT(*) FROM security_events
                    {time_clause_events}
                    {'AND' if time_clause_events else 'WHERE'}
                    (risk_level IN ('YELLOW', 'ORANGE', 'RED') AND event_type NOT LIKE '%CAMERA%' AND event_type NOT LIKE '%VOICE%')
                """
                cursor.execute(denied_sql, params_events)
                denied_access_attempts = cursor.fetchone()[0]

                return {
                    "active_incidents": active_incidents,
                    "resolved_incidents": resolved_incidents,
                    "spoof_attempts": spoof_attempts,
                    "after_hours_violations": after_hours_violations,
                    "restricted_zone_entries": restricted_zone_entries,
                    "unknown_visitors": unknown_visitors,
                    "distinct_people_seen": distinct_people_seen,
                    "authorized_people_seen": authorized_people_seen,
                    "denied_access_attempts": denied_access_attempts,
                }

    def get_prioritized_active_incidents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves active security incidents prioritized by severity:
        CRITICAL/RED > RED > ORANGE > YELLOW > GREEN, then by updated_at DESC.
        """
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM security_incidents
                    WHERE state != 'RESOLVED'
                    ORDER BY 
                        CASE risk_level 
                            WHEN 'CRITICAL' THEN 1 
                            WHEN 'RED' THEN 2 
                            WHEN 'ORANGE' THEN 3 
                            WHEN 'YELLOW' THEN 4 
                            ELSE 5 
                        END ASC,
                        updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_events_by_window(
        self,
        window_seconds: Optional[float] = None,
        since_midnight: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieves security events bounded by time window ordered by recency."""
        now = datetime.now()
        if since_midnight:
            cutoff = datetime(now.year, now.month, now.day, 0, 0, 0)
            cutoff_iso = cutoff.isoformat()
        elif window_seconds is not None:
            cutoff = now - timedelta(seconds=window_seconds)
            cutoff_iso = cutoff.isoformat()
        else:
            cutoff_iso = None

        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                if cutoff_iso:
                    cursor.execute(
                        """
                        SELECT * FROM security_events
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (cutoff_iso, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM security_events
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    # --- Database Maintenance & Clean Reset (Phase 2.7 & 5.9) ---

    def vacuum_database(self) -> bool:
        """
        Explicit maintenance operation: reclaims unused disk space.
        Should only be run on operator request or maintenance scripts, never continuously.
        """
        with self._lock:
            with self._connection() as conn:
                conn.execute("VACUUM")
                return True

    def optimize_database(self) -> Dict[str, Any]:
        """
        Explicit maintenance operation: optimizes SQLite query planner and analyzes tables.
        """
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA optimize")
                cursor.execute("PRAGMA integrity_check(1)")
                integrity = cursor.fetchone()[0]

                table_counts: Dict[str, int] = {}
                for tbl in [
                    "users",
                    "face_registrations",
                    "detected_objects",
                    "security_events",
                    "ocr_documents",
                    "system_logs",
                    "security_incidents",
                ]:
                    cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                    table_counts[tbl] = cursor.fetchone()[0]

                return {
                    "status": "OPTIMIZE_COMPLETE",
                    "integrity": integrity,
                    "table_counts": table_counts,
                }

    def clear_event_history(self, clear_incidents: bool = True) -> Dict[str, int]:
        """
        Clears stored security events and related runtime history records from SQLite.
        Strictly preserves users, face registrations, database schema, and indexes.
        
        Args:
            clear_incidents: If True, clears security_incidents before security_events
                             to avoid foreign key constraint violations or orphaned incident records.
        Returns:
            Dictionary mapping table name to number of deleted records.
        """
        tables_to_clear = ["security_events", "detected_objects", "system_logs"]
        if clear_incidents:
            # Delete incidents first due to foreign key referencing security_events
            tables_to_clear.insert(0, "security_incidents")

        deleted_counts: Dict[str, int] = {}
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("BEGIN TRANSACTION;")
                    for tbl in tables_to_clear:
                        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                        cnt = cursor.fetchone()[0]
                        deleted_counts[tbl] = cnt
                        cursor.execute(f"DELETE FROM {tbl}")

                    # Reset SQLite auto-increment sequence for cleared history tables
                    placeholders = ",".join("?" for _ in tables_to_clear)
                    cursor.execute(
                        f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})",
                        tables_to_clear,
                    )

                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"[DATABASE] Transaction failed during clear_event_history: {e}")
                    raise e

                # Reclaim fragmented disk space
                cursor.execute("VACUUM")

                self._event_insert_counter = 0

        logger.info(f"[DATABASE] Event history cleared successfully: {deleted_counts}")
        return deleted_counts

    def reset_development_data(self, backup: bool = True, confirm: bool = False) -> Dict[str, Any]:
        """
        Performs a safe, clean development data reset:
        1. Requires explicit confirm=True before executing.
        2. Creates a timestamped SQLite database backup in data/backups/ if backup=True.
        3. Clears face_registrations, security_events, detected_objects, ocr_documents, system_logs.
        4. Resets SQLite auto-increment sequences (sqlite_sequence).
        5. Deletes temporary testing evidence snapshots and registered face crops.
        6. Preserves database schema, table structure, admin user, AI models, and demo portraits.
        """
        if not confirm:
            raise ValueError(
                "Reset aborted: Explicit confirmation required. Call reset_development_data(confirm=True) to proceed."
            )

        backup_file_path: Optional[Path] = None

        # 1. Create backup if requested
        if backup and self.db_path.exists():
            backup_dir = DATA_DIR / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_path = backup_dir / f"watchguard_backup_{ts}.db"
            shutil.copy2(self.db_path, backup_file_path)

        deleted_counts: Dict[str, int] = {}

        # 2. Delete database records inside a lock & transaction
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()

                for table in [
                    "face_registrations",
                    "users",
                    "security_events",
                    "security_incidents",
                    "detected_objects",
                    "ocr_documents",
                    "system_logs",
                ]:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    deleted_counts[table] = count
                    cursor.execute(f"DELETE FROM {table}")

                # Reset SQLite auto-increment counters
                cursor.execute(
                    """
                    DELETE FROM sqlite_sequence 
                    WHERE name IN ('face_registrations', 'users', 'security_events', 'security_incidents', 'detected_objects', 'ocr_documents', 'system_logs')
                    """
                )
                conn.commit()

                # Reclaim fragmented space
                cursor.execute("VACUUM")
                conn.commit()

        # 3. Clean generated evidence images from data/evidence/
        deleted_evidence_count = 0
        if EVIDENCE_DIR.exists():
            for f in EVIDENCE_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    try:
                        f.unlink()
                        deleted_evidence_count += 1
                    except Exception:
                        pass

        # 4. Clean generated face registration crops from data/faces/
        deleted_face_images_count = 0
        if FACES_DIR.exists():
            for f in FACES_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    try:
                        f.unlink()
                        deleted_face_images_count += 1
                    except Exception:
                        pass

        # 5. Inventory preserved assets
        preserved_models = [f.name for f in MODELS_DIR.iterdir() if f.is_file()] if MODELS_DIR.exists() else []
        preserved_demo_faces = [f.name for f in DEMO_FACES_DIR.iterdir() if f.is_file()] if DEMO_FACES_DIR.exists() else []

        return {
            "status": "CLEAN_RESET_COMPLETE",
            "backup_path": str(backup_file_path) if backup_file_path else None,
            "deleted_records": deleted_counts,
            "deleted_evidence_images": deleted_evidence_count,
            "deleted_face_images": deleted_face_images_count,
            "preserved_models": preserved_models,
            "preserved_demo_faces": preserved_demo_faces,
            "remaining_face_profiles": 0,
            "remaining_security_events": 0,
        }
