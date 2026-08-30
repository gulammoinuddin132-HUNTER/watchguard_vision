"""
WatchGuard Vision - Reproducible Security Evaluation Scenarios (Part 12)
-----------------------------------------------------------------------
Defines structured, reproducible security evaluation fixtures for Scenarios A through J.
Allows repeatable experimentation across Baselines 1–4 and CCSDF without requiring
a live physical webcam.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.research.ccsdf_engine import (
    CCSDFContext,
    IdentityState,
    LivenessState,
    ZoneState,
    TemporalState,
    ProximityState,
    HistoryState,
)


@dataclass
class SecurityScenario:
    """Represents an experimental evaluation test scenario."""
    scenario_id: str
    name: str
    description: str
    context: CCSDFContext
    expected_authoritative_decision: str  # "GRANTED" or "DENIED"
    expected_ccsdf_action: str             # "ALLOW", "ATTENTION", "RESTRICT", "DENY", "PENDING_VERIFICATION"
    is_spoof_scenario: bool = False
    is_prohibited_context: bool = False
    is_unauthorized_person: bool = False
    frames_count: int = 30                 # Simulated duration in video frames


def get_all_scenarios() -> List[SecurityScenario]:
    """Returns the full suite of 10 benchmark scenarios (A through J)."""
    return [
        # Scenario A: Authorized person in standard perimeter during business hours
        SecurityScenario(
            scenario_id="SCENARIO_A",
            name="Authorized Person / Normal Zone / Permitted Time",
            description="Enrolled staff member (Hunter) in standard perimeter during working hours.",
            context=CCSDFContext(
                identity=IdentityState(name="Hunter", is_known=True, similarity=0.92, role="Staff"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.95),
                zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="10:30", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="GRANTED",
            expected_ccsdf_action="ALLOW",
        ),

        # Scenario B: Unknown / unregistered person in standard perimeter
        SecurityScenario(
            scenario_id="SCENARIO_B",
            name="Unknown Person / Normal Zone",
            description="Unregistered visitor in standard perimeter during daytime.",
            context=CCSDFContext(
                identity=IdentityState(name="Unknown Person", is_known=False, similarity=0.0, role="Visitor"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.92),
                zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="11:15", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="DENIED",
            expected_ccsdf_action="ATTENTION",
            is_unauthorized_person=True,
        ),

        # Scenario C: Authorized person in restricted zone during permitted operating hours
        SecurityScenario(
            scenario_id="SCENARIO_C",
            name="Authorized Person / Restricted Zone / Permitted Time",
            description="Authorized staff member inside Restricted Area during working hours.",
            context=CCSDFContext(
                identity=IdentityState(name="Hunter", is_known=True, similarity=0.94, role="Staff"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.94),
                zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="14:00", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="GRANTED",
            expected_ccsdf_action="ATTENTION",  # Elevated zone risk raises score from ALLOW to ATTENTION
        ),

        # Scenario D: Authorized person in restricted zone after hours (Prohibited Context)
        SecurityScenario(
            scenario_id="SCENARIO_D",
            name="Authorized Person / Restricted Zone / After-Hours",
            description="Authorized staff member inside Restricted Area at 22:45 outside operating hours.",
            context=CCSDFContext(
                identity=IdentityState(name="Hunter", is_known=True, similarity=0.93, role="Staff"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.93),
                zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="22:45", is_authorized_hours=False),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="DENIED",
            expected_ccsdf_action="DENY",
            is_prohibited_context=True,
        ),

        # Scenario E: Unknown person in restricted zone (Critical Intrusion)
        SecurityScenario(
            scenario_id="SCENARIO_E",
            name="Unknown Person / Restricted Zone",
            description="Unregistered individual trespassing inside Restricted Area.",
            context=CCSDFContext(
                identity=IdentityState(name="Unknown Person", is_known=False, similarity=0.0, role="Visitor"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.90),
                zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="15:30", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="DENIED",
            expected_ccsdf_action="DENY",
            is_unauthorized_person=True,
        ),

        # Scenario F: Photo presentation attack with authorized identity (Spoof Attack)
        SecurityScenario(
            scenario_id="SCENARIO_F",
            name="Photo Spoof / Authorized Identity",
            description="Photo replay attack presenting printed photo of registered user Hunter.",
            context=CCSDFContext(
                identity=IdentityState(name="Hunter", is_known=True, similarity=0.95, role="Staff"),
                liveness=LivenessState(liveness_state="SPOOF", confidence=0.15, attack_type="PHOTO_REPLAY"),
                zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="10:45", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="DENIED",
            expected_ccsdf_action="DENY",
            is_spoof_scenario=True,
        ),

        # Scenario G: Authorized person + protected asset proximity (Sensitive Interaction)
        SecurityScenario(
            scenario_id="SCENARIO_G",
            name="Authorized Person + Protected Asset Proximity",
            description="Hunter standing directly adjacent to Protected Asset Area (laptop workstation).",
            context=CCSDFContext(
                identity=IdentityState(name="Hunter", is_known=True, similarity=0.91, role="Staff"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.92),
                zone=ZoneState(zone_id="PROTECTED_ASSET_AREA", zone_name="Protected Asset Area", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="11:00", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=True, nearest_distance_pixels=25.0, asset_label="laptop"),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="GRANTED",
            expected_ccsdf_action="ATTENTION",
        ),

        # Scenario H: Repeated restricted-zone entry (Cumulative Violations)
        SecurityScenario(
            scenario_id="SCENARIO_H",
            name="Repeated Restricted-Zone Entry",
            description="Subject with 3 prior unauthorized entries entering restricted area.",
            context=CCSDFContext(
                identity=IdentityState(name="Hunter", is_known=True, similarity=0.90, role="Staff"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.90),
                zone=ZoneState(zone_id="RESTRICTED", zone_name="Restricted Area", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="16:00", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=3, repeated_entries_count=3),
            ),
            expected_authoritative_decision="DENIED",
            expected_ccsdf_action="DENY",
            is_prohibited_context=True,
        ),

        # Scenario I: Tailgating / Co-Presence (Simultaneous authorized and unregistered)
        SecurityScenario(
            scenario_id="SCENARIO_I",
            name="Tailgating / Simultaneous Co-Presence",
            description="Unregistered visitor shadowing authorized staff member into perimeter.",
            context=CCSDFContext(
                identity=IdentityState(name="Unknown Person", is_known=False, similarity=0.0, role="Visitor"),
                liveness=LivenessState(liveness_state="LIVE", confidence=0.91),
                zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="09:15", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=True, nearest_distance_pixels=15.0, asset_label="person"),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="DENIED",
            expected_ccsdf_action="ATTENTION",
            is_unauthorized_person=True,
        ),

        # Scenario J: PAD Uncertainty / Warmup Verification
        SecurityScenario(
            scenario_id="SCENARIO_J",
            name="PAD Uncertainty (Warmup Hold)",
            description="Enrolled subject approaching camera while blink detector is still accumulating EAR frames.",
            context=CCSDFContext(
                identity=IdentityState(name="Hunter", is_known=True, similarity=0.92, role="Staff"),
                liveness=LivenessState(liveness_state="WARMUP", confidence=0.50),
                zone=ZoneState(zone_id="NORMAL", zone_name="Standard Perimeter", occupancy="INSIDE"),
                temporal=TemporalState(current_time_str="10:00", is_authorized_hours=True),
                proximity=ProximityState(has_interaction=False),
                history=HistoryState(violations_count=0),
            ),
            expected_authoritative_decision="DENIED",
            expected_ccsdf_action="PENDING_VERIFICATION",
        ),
    ]
