"""
Database Models & Dataclasses
-----------------------------
Structured entities representing records in the WatchGuard Vision database.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class User:
    id: Optional[int]
    username: str
    full_name: str
    role: str
    access_level: int  # 1 = Basic, 2 = Staff, 3 = Admin/Security
    created_at: str = ""


@dataclass
class FaceRegistration:
    id: Optional[int]
    user_id: int
    face_encoding: bytes  # Binary or serialized vector
    image_path: str
    registered_at: str = ""
    is_active: bool = True


@dataclass
class DetectedObjectRecord:
    id: Optional[int]
    session_id: str
    label: str
    confidence: float
    bbox_json: str
    timestamp: str = ""


@dataclass
class SecurityEventRecord:
    id: Optional[int]
    event_type: str
    risk_level: str  # GREEN, YELLOW, ORANGE, RED
    description: str
    evidence_frame_path: Optional[str] = None
    person_name: Optional[str] = None
    object_summary: Optional[str] = None
    metadata_json: Optional[str] = None
    timestamp: str = ""


@dataclass
class OCRDocumentRecord:
    id: Optional[int]
    raw_text: str
    sensitive_keywords_found: str
    document_status: str  # e.g., 'CONFIDENTIAL', 'PUBLIC', 'RESTRICTED'
    snapshot_path: Optional[str] = None
    timestamp: str = ""


@dataclass
class SystemLogRecord:
    id: Optional[int]
    level: str  # INFO, WARNING, ERROR, CRITICAL
    module: str
    message: str
    timestamp: str = ""


@dataclass
class IncidentRecord:
    incident_id: str
    risk_level: str
    state: str  # DETECTED, VERIFIED, ALERTED, ACKNOWLEDGED, RESOLVED
    policy_code: str
    reason: str
    zone: str
    created_at: str
    updated_at: str
    event_id: Optional[int] = None
    track_id: Optional[int] = None
    person_name: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    operator_action: Optional[str] = None
    evidence_package_json: Optional[str] = None
    id: Optional[int] = None
