"""
WatchGuard Vision - Event Types & Risk Levels
---------------------------------------------
Defines security risk classifications and standardized event types.
"""

from enum import Enum


class RiskLevel(str, Enum):
    GREEN = "GREEN"      # Normal operation, authorized person / no threat
    YELLOW = "YELLOW"    # Attention: Unknown person detected / new object
    ORANGE = "ORANGE"    # High Risk: Restricted area access / unauthorized object
    RED = "RED"          # Security Incident: Unknown person + Confidential doc / direct breach

    @property
    def label(self) -> str:
        labels = {
            RiskLevel.GREEN: "NORMAL",
            RiskLevel.YELLOW: "ATTENTION",
            RiskLevel.ORANGE: "HIGH RISK",
            RiskLevel.RED: "SECURITY INCIDENT",
        }
        return labels.get(self, "UNKNOWN")

    @property
    def color_hex(self) -> str:
        colors = {
            RiskLevel.GREEN: "#10B981",    # Emerald Green
            RiskLevel.YELLOW: "#F59E0B",   # Amber
            RiskLevel.ORANGE: "#F97316",   # Orange
            RiskLevel.RED: "#EF4444",      # Crimson Red
        }
        return colors.get(self, "#9CA3AF")


class EventType(str, Enum):
    # System & Camera
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    CAMERA_STARTED = "CAMERA_STARTED"
    CAMERA_STOPPED = "CAMERA_STOPPED"
    CAMERA_ERROR = "CAMERA_ERROR"

    # Vision Detections
    PERSON_DETECTED = "PERSON_DETECTED"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    AUTHORIZED_PERSON = "AUTHORIZED_PERSON"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    DOCUMENT_SCANNED = "DOCUMENT_SCANNED"

    # Context & Security Alerts
    SECURITY_ALERT = "SECURITY_ALERT"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    AFTER_HOURS_ACCESS_DENIED = "AFTER_HOURS_ACCESS_DENIED"
    PRESENTATION_ATTACK = "PRESENTATION_ATTACK"
    ZONE_ENTRY = "ZONE_ENTRY"
    ZONE_EXIT = "ZONE_EXIT"

    # Phase 4 Incident & Temporal Policy Events
    INCIDENT_DETECTED = "INCIDENT_DETECTED"
    INCIDENT_VERIFIED = "INCIDENT_VERIFIED"
    INCIDENT_ALERTED = "INCIDENT_ALERTED"
    INCIDENT_ACKNOWLEDGED = "INCIDENT_ACKNOWLEDGED"
    INCIDENT_RESOLVED = "INCIDENT_RESOLVED"
    INCIDENT_ESCALATED = "INCIDENT_ESCALATED"
    REPEATED_POLICY_VIOLATION = "REPEATED_POLICY_VIOLATION"

    # Voice Assistant
    VOICE_COMMAND_RECEIVED = "VOICE_COMMAND_RECEIVED"
    VOICE_RESPONSE_SPOKEN = "VOICE_RESPONSE_SPOKEN"
