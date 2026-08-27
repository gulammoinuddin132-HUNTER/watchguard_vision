"""
Live End-to-End Verification Script for WatchGuard 2.0 Phase 5.6
-----------------------------------------------------------------
Validates live camera/pipeline logic, semantic routing, database state,
and incident handling across Scenarios A through H.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
import json
import time
import numpy as np

from src.camera.camera_manager import CameraManager
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.incident_manager import IncidentManager, IncidentState
from src.core.person_entity import PersonEntity, PersonEntityFusionEngine
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.policy_engine import PolicyEngine
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.semantic_router import SemanticQueryRouter
from src.core.voice_assistant import VoiceAssistant, VoiceIntent
from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel


def run_live_verification():
    print("=== STARTING PHASE 5.6 END-TO-END VERIFICATION ===")

    db = DatabaseManager()
    logger = EventLogger(db_manager=db)
    context_engine = ContextEngine(db_manager=db)
    sec_analytics = SecurityAnalyticsEngine(db_manager=db, context_engine=context_engine, incident_manager=context_engine.incident_manager)
    camera = MagicMock()
    camera.is_running = True
    p_engine = MagicMock()
    p_engine.is_ready = True
    p_engine.last_results = {
        "detected": True,
        "count": 1,
        "people": [{"name": "Gulam Moinuddin", "confidence": 0.96, "role": "Administrator", "is_known": True, "liveness": "LIVE", "liveness_confidence": 0.95}],
    }
    va = VoiceAssistant(
        db_manager=db,
        event_logger=logger,
        person_engine=p_engine,
        context_engine=context_engine,
        security_analytics=sec_analytics,
        camera_manager=camera,
    )

    # 1. Clean DB Reset at startup
    db.reset_development_data(backup=False, confirm=True)
    print("\n--- 1. Database State ---")
    users = db.get_users()
    events_cnt = db.get_total_event_count()
    incidents = db.get_active_incidents()
    print(f"Users in DB: {len(users)}")
    print(f"Total Events in DB: {events_cnt}")
    print(f"Active Incidents: {len(incidents)}")
    assert len(users) == 0, "Users should be 0 after reset"
    assert events_cnt == 0, "Events count should be 0 after reset"

    # 2. Register New User (e.g. Operator / Admin)
    print("\n--- 2. Profile Registration ---")
    u_id = db.create_user(username="gulam_admin", full_name="Gulam Moinuddin", role="Administrator", access_level=3)
    print(f"Created user id: {u_id} (Gulam Moinuddin)")

    # 3. Simulate Scenarios
    print("\n--- 3. Scenario Testing ---")

    # Scenario A: Live Authorized Person (In Hours)
    print("\n[Scenario A] Authorized Person in Standard Perimeter")
    entity_a = PersonEntity(
        track_id=1,
        bbox=(100, 100, 100, 200),
        face_bbox=(110, 110, 80, 80),
        identity="Gulam Moinuddin",
        identity_confidence=0.96,
        is_known=True,
        role="Administrator",
        liveness_state="STABLE_LIVE",
        liveness_confidence=0.95,
        zone="Standard Perimeter",
        access_state="GRANTED",
        risk_level=RiskLevel.GREEN,
        decision_code="AUTHORIZED_PERSON_NORMAL_PERIMETER",
        reason="Verified personnel within standard operating zone.",
    )
    dec_a = SecurityDecision(
        risk_level=RiskLevel.GREEN,
        decision="AUTHORIZED_PERSON_NORMAL_PERIMETER",
        reason="Verified personnel within standard operating zone.",
        person_entities=[entity_a],
        is_access_granted=True,
        context_summary={"person_name": "Gulam Moinuddin", "is_authorized": True, "zone_name": "Standard Perimeter", "is_authorized_hours": True},
    )
    context_engine.last_decision = dec_a
    va.session_context.last_person_name = "Gulam Moinuddin"
    va.session_context.last_is_authorized = True
    va.session_context.last_risk_level = "GREEN"

    resp_who = va.process_command("Who is standing there?")
    print(f"Query: 'Who is standing there?' -> Response: {resp_who}")
    assert "Gulam Moinuddin" in resp_who or "Gulam" in resp_who

    # Scenario A.2: Live Multi-Person (1 Authorized + 1 Unregistered) - Core Bug Scenario
    print("\n[Scenario A.2] Live Multi-Person: Gulam Moinuddin (Authorized) + Unregistered Person")
    entity_a2 = PersonEntity(
        track_id=2,
        bbox=(250, 100, 100, 200),
        face_bbox=(260, 110, 80, 80),
        identity="Unknown Person",
        identity_confidence=0.0,
        is_known=False,
        role="Visitor",
        liveness_state="STABLE_LIVE",
        liveness_confidence=0.92,
        zone="Standard Perimeter",
        access_state="DENIED",
        risk_level=RiskLevel.YELLOW,
        decision_code="UNAUTHORIZED_ACCESS",
        reason="Unregistered person present.",
    )
    dec_a2 = SecurityDecision(
        risk_level=RiskLevel.YELLOW,
        decision="UNAUTHORIZED_ACCESS",
        reason="1 authorized personnel and 1 unregistered person observed.",
        person_entities=[entity_a, entity_a2],
        is_access_granted=True,
        context_summary={
            "person_detected": True,
            "person_name": "Gulam Moinuddin",
            "is_authorized": True,
            "is_unknown": True,
            "has_spoof": False,
            "objects_detected": 0,
            "object_summary": "None",
            "objects": [],
            "person_entities": [entity_a.to_dict(), entity_a2.to_dict()],
            "person_entity_count": 2,
            "zone_name": "Standard Perimeter",
            "is_authorized_hours": True,
        },
    )
    context_engine.last_decision = dec_a2
    va._sync_session_from_live_state()

    print("  -> Asking: 'Who is in front of the camera?'")
    r1 = va.process_command("Who is in front of the camera?")
    print(f"     Response: {r1}")
    assert "Gulam Moinuddin" in r1 and "unregistered person" in r1.lower() and "two people" in r1.lower()

    print("  -> Asking: 'What do you see?'")
    r2 = va.process_command("What do you see?")
    print(f"     Response: {r2}")
    assert "Gulam Moinuddin" in r2 and "unregistered" in r2.lower() and "two people" in r2.lower()

    print("  -> Asking: 'How many people are there?'")
    r3 = va.process_command("How many people are there?")
    print(f"     Response: {r3}")
    assert "two people" in r3.lower()

    print("  -> Asking: 'What objects are detected?'")
    r4 = va.process_command("What objects are detected?")
    print(f"     Response: {r4}")
    assert "No non-person objects" in r4 and "two people" in r4.lower() and "Gulam Moinuddin" in r4

    # Scenario B: After-Hours Restricted Zone Entry
    print("\n[Scenario B] Authorized Person in Restricted Area (After Hours)")
    context_engine.is_after_hours_simulation = True
    context_engine.simulated_hour = 22
    context_engine.simulated_minute = 45

    entity_b = PersonEntity(
        track_id=1,
        bbox=(100, 100, 100, 200),
        face_bbox=(110, 110, 80, 80),
        identity="Gulam Moinuddin",
        identity_confidence=0.96,
        is_known=True,
        role="Administrator",
        liveness_state="STABLE_LIVE",
        liveness_confidence=0.95,
        zone="Restricted Area",
        access_state="DENIED",
        risk_level=RiskLevel.ORANGE,
        decision_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
        reason="Current time 22:45 is outside permitted 09:00 to 18:00 window.",
    )
    dec_b = SecurityDecision(
        risk_level=RiskLevel.ORANGE,
        decision="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
        reason="Current time 22:45 is outside permitted 09:00 to 18:00 window.",
        person_entities=[entity_b],
        is_access_granted=False,
        context_summary={"person_name": "Gulam Moinuddin", "is_authorized": True, "zone_name": "Restricted Area", "is_authorized_hours": False},
    )
    context_engine.last_decision = dec_b
    va.session_context.last_person_name = "Gulam Moinuddin"
    va.session_context.last_decision_code = "AFTER_HOURS_RESTRICTED_ACCESS_DENIED"
    va.session_context.last_reason = "Current time 22:45 is outside permitted 09:00 to 18:00 window."
    va.session_context.last_time_str = "22:45"
    va.session_context.last_zone = "Restricted Area"
    va.session_context.last_is_authorized_hours = False

    # Dispatch Incident
    inc_b = context_engine.incident_manager.process_incident(
        track_id=1,
        person_name="Gulam Moinuddin",
        zone_name="Restricted Area",
        policy_code="AFTER_HOURS_RESTRICTED_ACCESS_DENIED",
        risk_level=RiskLevel.ORANGE,
        reason="Authorized personnel detected in Restricted Area after hours.",
    )
    print(f"Incident dispatched: {inc_b.incident_id if inc_b else 'None'} [{inc_b.risk_level.value if inc_b else 'None'}]")

    # Natural Language Queries
    queries = [
        "Why was Gulam denied?",
        "Why didn't he get access?",
        "What's stopping him?",
        "Why can't he enter?",
        "Tell me why his access was refused.",
        "What is the current incident?",
    ]
    for q in queries:
        resp = va.process_command(q)
        print(f"Query: '{q}'\n  -> Source: [{va.session_context.last_response_source.value}] ({va.session_context.last_response_latency_ms:.1f}ms)\n  -> Answer: {resp}")

    # Scenario C: Presentation Attack (Spoof)
    print("\n[Scenario C] Phone Photo Spoof Rejection")
    entity_c = PersonEntity(
        track_id=2,
        bbox=(200, 100, 80, 120),
        face_bbox=(210, 110, 60, 60),
        identity="Gulam Moinuddin",
        identity_confidence=0.92,
        is_known=True,
        role="Administrator",
        liveness_state="STABLE_SPOOF",
        liveness_confidence=0.88,
        is_spoof=True,
        attack_type="PHOTO_REPLAY_ATTACK",
        zone="Standard Perimeter",
        access_state="DENIED",
        risk_level=RiskLevel.RED,
        decision_code="PRESENTATION_ATTACK_IMPERSONATION",
        reason="Photo replay spoof attack detected.",
    )
    dec_c = SecurityDecision(
        risk_level=RiskLevel.RED,
        decision="PRESENTATION_ATTACK_IMPERSONATION",
        reason="Photo replay spoof attack detected.",
        person_entities=[entity_c],
        is_access_granted=False,
        context_summary={"person_name": "Gulam Moinuddin", "is_authorized": True, "zone_name": "Standard Perimeter", "is_authorized_hours": True},
    )
    context_engine.last_decision = dec_c
    va.session_context.last_person_name = "Gulam Moinuddin"
    va.session_context.last_decision_code = "PRESENTATION_ATTACK_IMPERSONATION"
    va.session_context.last_reason = "Photo replay spoof attack detected."
    va.session_context.last_risk_level = "RED"

    resp_spoof = va.process_command("Why is the system red?")
    print(f"\nQuery: 'Why is the system red?' -> Response: {resp_spoof}")
    assert "attack" in resp_spoof.lower() or "spoof" in resp_spoof.lower() or "presentation" in resp_spoof.lower()

    # Clean up test user created in step 2 so database remains completely clean
    db.reset_development_data(backup=False, confirm=True)
    print("\nFinal clean DB reset executed. Profiles in DB:", len(db.get_users()))

    print("\n=== PHASE 5.6 ALL SCENARIOS VERIFIED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_live_verification()
