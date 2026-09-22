"""
Azure SQL synchronization for WatchGuard Vision.

Keeps the existing local SQLite database as the primary database
and copies important security events to Azure SQL.
"""

import os
import hashlib

import pyodbc
from dotenv import load_dotenv


class AzureSQLSync:
    """Synchronize WatchGuard security events with Azure SQL."""

    def __init__(self):
        # Load variables from the project-root .env file
        load_dotenv()

        self.server = os.getenv("AZURE_SQL_SERVER")
        self.database = os.getenv("AZURE_SQL_DATABASE")
        self.username = os.getenv("AZURE_SQL_USERNAME")
        self.password = os.getenv("AZURE_SQL_PASSWORD")

        # Azure sync is enabled only when all required variables exist
        self.enabled = all(
            [
                self.server,
                self.database,
                self.username,
                self.password,
            ]
        )

        self.connection_string = (
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER=tcp:{self.server},1433;"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
        )

        self._table_ready = False

    def _connect(self):
        """Create and return an Azure SQL connection."""

        if not self.enabled:
            raise RuntimeError(
                "Azure SQL environment variables are not configured."
            )

        return pyodbc.connect(self.connection_string)

    def _ensure_table(self, conn):
        """Create the Azure security-events table if it doesn't exist."""

        if self._table_ready:
            return

        cursor = conn.cursor()

        cursor.execute(
            """
            IF OBJECT_ID('dbo.watchguard_security_events', 'U') IS NULL
            BEGIN
                CREATE TABLE dbo.watchguard_security_events (
                    sync_id VARCHAR(64) NOT NULL PRIMARY KEY,
                    local_event_id INT NULL,
                    event_type NVARCHAR(100) NOT NULL,
                    risk_level NVARCHAR(20) NOT NULL,
                    description NVARCHAR(MAX) NOT NULL,
                    evidence_frame_path NVARCHAR(1000) NULL,
                    person_name NVARCHAR(255) NULL,
                    object_summary NVARCHAR(MAX) NULL,
                    metadata_json NVARCHAR(MAX) NULL,
                    event_timestamp DATETIME2 NOT NULL,
                    synced_at DATETIME2 NOT NULL
                );

                CREATE INDEX IX_watchguard_events_timestamp
                ON dbo.watchguard_security_events (event_timestamp DESC);

                CREATE INDEX IX_watchguard_events_risk
                ON dbo.watchguard_security_events (risk_level);
            END
            """
        )

        conn.commit()

        self._table_ready = True

    def sync_security_event(
        self,
        local_event_id,
        event_type,
        risk_level,
        description,
        evidence_frame_path=None,
        person_name=None,
        object_summary=None,
        metadata_json=None,
        timestamp=None,
    ):
        """
        Copy one security event from the local SQLite database
        to Azure SQL.

        Returns:
            True  -> event successfully synchronized
            False -> Azure sync is disabled
        """

        if not self.enabled:
            return False

        # Use the event information to create a deterministic ID.
        # This prevents duplicate cloud records if the same event
        # is accidentally synchronized more than once.
        raw_key = "|".join(
            [
                str(local_event_id),
                str(event_type),
                str(risk_level),
                str(description),
                str(timestamp),
            ]
        )

        sync_id = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        with self._connect() as conn:

            # Make sure the Azure table exists
            self._ensure_table(conn)

            cursor = conn.cursor()

            # Insert only when this event has not already been synced
            cursor.execute(
                """
                IF NOT EXISTS (
                    SELECT 1
                    FROM dbo.watchguard_security_events
                    WHERE sync_id = ?
                )
                BEGIN
                    INSERT INTO dbo.watchguard_security_events (
                        sync_id,
                        local_event_id,
                        event_type,
                        risk_level,
                        description,
                        evidence_frame_path,
                        person_name,
                        object_summary,
                        metadata_json,
                        event_timestamp,
                        synced_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        SYSUTCDATETIME()
                    )
                END
                """,
                (
                    sync_id,
                    sync_id,
                    local_event_id,
                    event_type,
                    risk_level,
                    description,
                    evidence_frame_path,
                    person_name,
                    object_summary,
                    metadata_json,
                    timestamp,
                ),
            )

            conn.commit()

        return True