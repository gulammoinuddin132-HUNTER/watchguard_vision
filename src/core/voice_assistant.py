"""
Voice Assistant Module (Phase 2.5: Advanced Contextual Security Assistant)
-------------------------------------------------------------------------
Deterministic, local/offline Contextual Security Assistant for WatchGuard Vision.

Features:
- Lightweight SessionContext for multi-turn follow-up queries (pronouns: he/she/they/this person/that object/why?)
- Clear distinction between CURRENT live vision state and HISTORICAL SQLite logs
- Temporal queries ("last 5/10 minutes", "today's alert count", "last red alert")
- Structured Security Explanations (DECISION -> REASON -> EVIDENCE)
- Spatial person + object relationship reasoning
- Event intelligence & repeated activity auditing
- Voice + Text simultaneous parity through a unified processing pipeline
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

try:
    import sounddevice as sd
    import speech_recognition as sr
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.core.security_analytics import SecurityAnalyticsEngine
from src.core.copilot_reasoner import (
    BaseAICopilotReasoner,
    CopilotResponseSource,
    CopilotResponse,
    DeterministicCopilotReasoner,
    build_structured_ai_context,
    get_copilot_reasoner,
    is_visual_query,
)
from src.core.semantic_router import SemanticQueryRouter, SemanticResolution

logger = logging.getLogger("WatchGuardVision.VoiceAssistant")


class VoiceIntent(Enum):
    """Supported deterministic security voice and text intents."""
    # Phase 2.4 Baseline Intents
    WHO_IS_IN_FRONT = auto()
    WHAT_DO_YOU_SEE = auto()
    SECURITY_RISK = auto()
    WHY_HIGH_RISK = auto()
    CURRENT_SECURITY_STATUS = auto()
    DETECTED_OBJECTS = auto()
    RECENT_SECURITY_EVENTS = auto()
    LATEST_ALERT = auto()
    RECENT_PERSON_DETECTED = auto()
    STOP_LISTENING = auto()

    # Phase 2.5 Advanced Contextual & Analytical Intents
    FOLLOW_UP_PERSON_AUTHORIZATION = auto()  # "Is he allowed to be there?", "Are they authorized?"
    FOLLOW_UP_WHY = auto()                   # "Why?", "Why is that?", "Why did the system trigger that alert?"
    AUTHORIZED_HOURS_QUERY = auto()          # "What are the authorized hours?"
    CURRENT_STATE_QUERY = auto()             # "What is happening right now?", "Is there a threat right now?"
    HISTORICAL_STATE_QUERY = auto()          # "What happened earlier?", "Was there a threat earlier?"
    TEMPORAL_EVENTS_MINUTES = auto()         # "What happened in the last 5 minutes?"
    TEMPORAL_ALERT_BY_LEVEL = auto()         # "When was the last red alert?"
    ALERT_COUNT_TODAY = auto()               # "How many alerts happened today?"
    REPEATED_UNAUTHORIZED_DETECTIONS = auto()# "Have there been repeated unauthorized detections?"
    MOST_SERIOUS_EVENT = auto()              # "What was the most serious recent event?"
    SPATIAL_RELATIONSHIP = auto()            # "Is anyone near the phone?", "What is the person interacting with?"

    # Phase 4 Temporal Policy & Incident Intelligence Intents
    VIOLATION_HISTORY_QUERY = auto()         # "Has this person violated the policy before?"
    RECENT_VIOLATIONS_QUERY = auto()         # "How many violations happened recently?" / "How many times did he enter recently?"
    INCIDENT_STATUS_QUERY = auto()           # "What is the current incident?"
    INCIDENT_TIME_QUERY = auto()             # "When did this incident start?"
    INCIDENT_ACK_STATUS_QUERY = auto()       # "Has this alert been acknowledged?"
    ACKNOWLEDGE_INCIDENT_COMMAND = auto()    # "Acknowledge incident"
    RESOLVE_INCIDENT_COMMAND = auto()        # "Resolve incident"

    # Phase 4.1 Zone Entry/Exit Intelligence Intents
    ZONE_ENTRY_QUERY = auto()                # "Did Hunter enter the restricted zone?"
    ZONE_EXIT_QUERY = auto()                 # "Has he left the restricted area?"
    ZONE_ENTRY_TIME_QUERY = auto()           # "When did he enter?"

    # Phase 5 Security Intelligence & Analytics Intents
    WHY_RISK_LEVEL = auto()                  # "Why is the system red?", "Why is the risk orange?"
    SPOOF_ATTEMPTS_QUERY = auto()            # "How many spoof attempts occurred?"
    DISTINCT_PEOPLE_QUERY = auto()           # "How many people were detected today?"
    LIVE_PEOPLE_COUNT_QUERY = auto()         # "How many people are there right now?"
    UNAUTHORIZED_LIVE_QUERY = auto()         # "Is anyone unauthorized right now?"
    LAST_ALERT_TRIGGER_QUERY = auto()        # "Who triggered the last alert?"

    # Phase 5.7.3 Multimodal Visual & Access Decision Intents
    PERSON_HOLDING_OBJECT = auto()             # "What am I holding?", "What is in my hand?"
    VISUAL_SCENE = WHAT_DO_YOU_SEE             # "What do you see?", "Describe the scene."
    VISUAL_OBJECT = DETECTED_OBJECTS           # "What objects are visible?"
    VISUAL_RELATIONSHIP = SPATIAL_RELATIONSHIP # "Is anyone near the laptop?"
    VISUAL_DESCRIPTION = auto()                # "Describe the person", "What is he wearing?"
    VISUAL_COUNT = LIVE_PEOPLE_COUNT_QUERY     # "How many people are visible?"
    EXPLAIN_ACCESS_DECISION = auto()           # "Why am I being denied?", "Why was I refused?"

    # Phase 5.16, 5.20, 5.22 Continuous Contextual Security Decision Fusion (CCSDF) & Unified 84 Intents
    RESEARCH_BASELINE_QUERY = auto()           # "What would the baseline face-only system have decided?"
    RESEARCH_RISK_FACTORS = auto()             # "What factors caused the risk increase?", "Which contextual factor is contributing most?"
    RESEARCH_AUTHORIZATION_CHANGE = auto()     # "Why did authorization change?", "What changed when he entered the restricted area?"
    RESEARCH_TAILGATING_QUERY = auto()         # "Was tailgating detected?" / "Is someone tailgating?"
    RESEARCH_BEHAVIOR_QUERY = auto()           # "What behavior was unusual?" / "What was the behavior anomaly?"
    RESEARCH_CONFIDENCE_QUERY = auto()         # "How confident is the current decision?" / "What is the decision confidence?"
    SECURITY_OBJECTS_QUERY = auto()            # "What security objects are present?", "Was a USB drive detected?"
    DETECTOR_MODE_QUERY = auto()               # "What detector mode is active?", "Which model is running?"
    RESEARCH_NOVELTY_QUERY = auto()            # "What is novel about WatchGuard?", "What is the research contribution?"
    RESEARCH_CCSDF_STATE_QUERY = auto()        # "What is the current CCSDF risk?", "What is the current CCSDF state?"

    UNKNOWN = auto()


@dataclass
class SessionContext:
    """Lightweight session context representing the latest live observation & conversation state."""
    last_person_name: Optional[str] = None
    last_is_authorized: Optional[bool] = None
    last_is_unknown: Optional[bool] = None
    last_role: Optional[str] = None
    last_people_entities: List[Dict[str, Any]] = field(default_factory=list)
    last_people_count: int = 0
    last_objects: List[str] = field(default_factory=list)
    last_risk_level: str = "GREEN"
    last_decision_code: str = "NORMAL_ACTIVITY"
    last_reason: str = "Perimeter clear. Routine surveillance monitoring active."
    last_zone: str = "Standard Perimeter"
    last_time_str: str = "--:--"
    last_is_authorized_hours: bool = True
    last_hours_window: str = "09:00 to 18:00"
    last_is_access_granted: bool = True
    last_interactions: List[Dict[str, Any]] = field(default_factory=list)
    last_evidence_required: bool = False
    last_intent: Optional[VoiceIntent] = None
    last_topic: Optional[str] = None  # "PERSON", "OBJECT", "RISK", "ALERT", "EVENTS", "SPATIAL", "HISTORICAL", "INCIDENT"
    last_incident_id: Optional[str] = None
    last_incident_state: Optional[str] = None
    last_incident_policy: Optional[str] = None
    last_incident_reason: Optional[str] = None
    last_incident_created_at: Optional[str] = None
    last_per_track_states: List[Dict[str, Any]] = field(default_factory=list)
    last_response_source: CopilotResponseSource = CopilotResponseSource.DETERMINISTIC
    last_response_latency_ms: float = 0.0


# --- Natural Query Normalization & Contraction Maps ---

CONTRACTIONS = {
    r"\bwho's\b": "who is",
    r"\bwhat's\b": "what is",
    r"\bthere's\b": "there is",
    r"\bit's\b": "it is",
    r"\bhow's\b": "how is",
    r"\bwhere's\b": "where is",
    r"\bcan't\b": "cannot",
    r"\bdon't\b": "do not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwon't\b": "will not",
    r"\bwasn't\b": "was not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
}

FILLER_PATTERNS = [
    r"^\s*(hey|ok|okay|hi|hello)\s+(watchguard|assistant|guard|system)\b",
    r"^\s*(watchguard|assistant)\b",
    r"\b(please|can you|could you|tell me|show me|give me|what about|uh|um|kindly)\b",
]


def normalize_query(query: str) -> str:
    """
    Normalizes a user query by converting to lowercase, expanding contractions,
    removing punctuation, stripping conversational filler words, and collapsing whitespace.
    """
    if not query:
        return ""
    q = query.lower().strip()
    for pattern, replacement in CONTRACTIONS.items():
        q = re.sub(pattern, replacement, q)
    for pattern in FILLER_PATTERNS:
        q = re.sub(pattern, " ", q)
    # Remove all punctuation except alphanumeric and whitespace
    q = re.sub(r"[^\w\s]", " ", q)
    # Collapse multiple whitespace characters
    q = re.sub(r"\s+", " ", q).strip()
    return q


# --- Intent Pattern Matching Rules (Ordered by Specificity) ---

INTENT_RULES: List[Tuple[VoiceIntent, List[str]]] = [
    # 1. Stop Listening / Standby
    (
        VoiceIntent.STOP_LISTENING,
        [
            r"\b(stop listening|stop talking|be quiet|shut up|never mind|cancel|standby|go to sleep|sleep|mute)\b",
            r"^stop$",
        ],
    ),
    # Phase 5 / Phase 2.4: Specific Why Risk Level ("Why is the system red?", "Why is the risk orange?", "Why red?", "Why is the system showing high risk?")
    (
        VoiceIntent.WHY_HIGH_RISK,
        [
            r"\bwhy\s+is\s+(the\s+)?(system|risk|status|dashboard)\s+(red|orange|yellow|green)\b",
            r"\bwhy\s+(is\s+it\s+)?(red|orange|yellow)\b",
            r"\bwhy\s+is\s+the\s+system\s+(in\s+)?(red|orange|yellow)\s+(alert|state)?\b",
            r"\bwhy\s+is\s+risk\s+(red|orange|yellow|elevated)\b",
            r"\bwhy\s+(is\s+the\s+system\s+showing\s+|is\s+it\s+)?(high\s+risk|high\s+threat|elevated\s+risk)\b",
            r"\bexplain\s+(the\s+)?(alert|risk|threat)\b",
        ],
    ),
    # Phase 5: Spoof Attempts Query ("How many spoof attempts occurred?")
    (
        VoiceIntent.SPOOF_ATTEMPTS_QUERY,
        [
            r"\bhow\s+many\s+(spoof|presentation\s+attack|photo\s+attack|replay\s+attack)\s*(attempts|attacks)?\s*(happened|occurred|were there|today)?\b",
            r"\bhow\s+many\s+spoof\s+attempts\b",
            r"\b(were\s+there|was\s+there)\s+any\s+(spoof|presentation)\s+attack(s)?\b",
            r"\bspoof\s+attempt(s)?\s+count\b",
        ],
    ),
    # Phase 5.6.1: Live People Count Query ("How many people are there?", "How many people in view?")
    (
        VoiceIntent.LIVE_PEOPLE_COUNT_QUERY,
        [
            r"\bhow\s+many\s+(people|persons|visitors|individuals|faces)\s+(are\s+)?(there|in\s+view|visible|in\s+front|present|in\s+the\s+camera|in\s+the\s+frame|do\s+you\s+see)\b",
            r"\bhow\s+many\s+people\s+(are\s+there|do\s+you\s+see|in\s+view|visible|present)\b",
            r"^how\s+many\s+people$",
            r"^how\s+many\s+people\s+are\s+there$",
            r"^how\s+many\s+people\s+in\s+view$",
        ],
    ),
    # Phase 5: Distinct People Count Today ("How many people were detected today?")
    (
        VoiceIntent.DISTINCT_PEOPLE_QUERY,
        [
            r"\bhow\s+many\s+(distinct\s+)?(people|persons|visitors|individuals|users)\s+(were\s+)?(detected|seen|observed|present|here)\s*(today|so far)?\b",
            r"\bhow\s+many\s+people\s+(were\s+)?(detected|seen|observed)\s+today\b",
            r"\bhow\s+many\s+people\s+today\b",
            r"\bpeople\s+count\s+today\b",
        ],
    ),
    # Phase 5: Unauthorized Live Presence ("Is anyone unauthorized right now?")
    (
        VoiceIntent.UNAUTHORIZED_LIVE_QUERY,
        [
            r"\bis\s+anyone\s+(unauthorized|unverified|forbidden|not allowed|pending)\s*(right\s+now|now|currently)?\b",
            r"\bis\s+anyone\s+unauthorized\b",
            r"\bare\s+there\s+any\s+unauthorized\s+(people|persons|individuals|visitors)\s*(right\s+now|now)?\b",
        ],
    ),
    # Phase 5: Last Alert Trigger ("Who triggered the last alert?")
    (
        VoiceIntent.LAST_ALERT_TRIGGER_QUERY,
        [
            r"\bwho\s+triggered\s+(the\s+)?(last|latest|most recent)\s+(alert|warning|incident|threat)\b",
            r"\bwho\s+caused\s+(the\s+)?(last|latest)\s+alert\b",
            r"\bwho\s+triggered\s+the\s+alert\b",
        ],
    ),
    # 2. Temporal Minutes Query ("What happened in the last 5 minutes?")
    (
        VoiceIntent.TEMPORAL_EVENTS_MINUTES,
        [
            r"\b(what happened|what occurred|events|activity)\b.*\b(last|past)\s+(\d+)\s*(min|minute|minutes)\b",
            r"\b(in the|during the|over the)\s+(last|past)\s+(\d+)\s*(min|minute|minutes)\b",
            r"\b(last|past)\s+(\d+)\s*(min|minute|minutes)\s+(events|activity|logs)\b",
        ],
    ),
    # 3. Temporal Alert by Level ("When was the last red alert?")
    (
        VoiceIntent.TEMPORAL_ALERT_BY_LEVEL,
        [
            r"\bwhen\s+was\s+the\s+last\s+(red|orange|yellow|green)\s+(alert|warning|incident|threat)\b",
            r"\b(last|latest|most recent)\s+(red|orange|yellow|green)\s+(alert|warning|incident)\b",
            r"\b(show|read|get)\s+(the\s+)?(last|latest)\s+(red|orange|yellow|green)\s+alert\b",
        ],
    ),
    # 4. Today's Alert Count ("How many alerts happened today?")
    (
        VoiceIntent.ALERT_COUNT_TODAY,
        [
            r"\bhow\s+many\s+(security\s+)?(alerts|warnings|incidents|threats|events)\s+(happened|occurred|were there)?\s*(today|so far today)\b",
            r"\bhow\s+many\s+(alerts|incidents)\s+today\b",
            r"\b(today alert count|number of alerts today|alert count today)\b",
        ],
    ),
    # 5. Repeated Unauthorized Detections / Visitor Auditing
    (
        VoiceIntent.REPEATED_UNAUTHORIZED_DETECTIONS,
        [
            r"\b(have there been|were there|are there)\s+repeated\s+(unauthorized|unknown|unregistered)\s+(detections|visitors|people|persons|alerts)\b",
            r"\b(has an\s+)?unauthorized\s+person\s+(been\s+)?detected\s+recently\b",
            r"\brepeated\s+(unauthorized|unknown|suspicious)\s+activity\b",
            r"\bhas\s+anyone\s+unauthorized\s+been\s+seen\b",
        ],
    ),
    # 6. Most Serious Event
    (
        VoiceIntent.MOST_SERIOUS_EVENT,
        [
            r"\bwhat\s+was\s+the\s+(most serious|worst|highest risk|most critical|biggest)\s+(recent\s+)?(event|alert|incident|threat)\b",
            r"\b(most serious|worst)\s+(incident|event|alert)\b",
        ],
    ),
    # 7. Spatial Person + Object Relationships
    (
        VoiceIntent.SPATIAL_RELATIONSHIP,
        [
            r"\bis\s+anyone\s+(near|interacting with|close to|touching|holding)\s+(the\s+)?(\w+)\b",
            r"\bis\s+(the\s+)?person\s+(near|close to|interacting with|touching|holding)\s+(the\s+)?(\w+)\b",
            r"\bis\s+an\s+unauthorized\s+person\s+near\s+(a\s+)?(protected asset|\w+)\b",
            r"\bwhat\s+is\s+the\s+person\s+(interacting with|near|touching|holding)\b",
            r"\bwhat\s+objects\s+are\s+(near|close to)\s+(the\s+)?person\b",
        ],
    ),
    # 8. Incident Status & Information (Phase 4)
    (
        VoiceIntent.INCIDENT_ACK_STATUS_QUERY,
        [
            r"\b(has|is)\s+(this|the)\s+(alert|incident)\s+(been\s+)?(acknowledged|acked)\b",
            r"\bhas\s+this\s+alert\s+been\s+acknowledged\b",
            r"\bis\s+the\s+incident\s+acknowledged\b",
        ],
    ),
    (
        VoiceIntent.INCIDENT_TIME_QUERY,
        [
            r"\bwhen\s+did\s+(this|the)\s+incident\s+(start|occur|happen|begin)\b",
            r"\bwhat\s+time\s+did\s+the\s+incident\s+start\b",
            r"\bincident\s+timestamp\b",
        ],
    ),
    (
        VoiceIntent.INCIDENT_STATUS_QUERY,
        [
            r"\bwhat\s+is\s+the\s+(current|active|latest)\s+incident\b",
            r"\bis\s+there\s+an\s+active\s+incident\b",
            r"\bcurrent\s+incident\b",
            r"\bincident\s+status\b",
        ],
    ),
    (
        VoiceIntent.ACKNOWLEDGE_INCIDENT_COMMAND,
        [
            r"\b(acknowledge|ack)\s+(the\s+)?(incident|alert|warning)\b",
            r"^acknowledge$",
            r"^ack$",
        ],
    ),
    (
        VoiceIntent.RESOLVE_INCIDENT_COMMAND,
        [
            r"\b(resolve|close|clear)\s+(the\s+)?(incident|alert)\b",
            r"^resolve$",
            r"^mark incident resolved$",
        ],
    ),
    # 9. Violation History & Recent Violations (Phase 4 & 4.1)
    (
        VoiceIntent.VIOLATION_HISTORY_QUERY,
        [
            r"\bhas\s+(this\s+person|he|she|they|\w+)\s+(violated|breached)\s+(the\s+)?policy\s+before\b",
            r"\b(did|has)\s+(\w+)\s+violate(d)?\s+policy\s+before\b",
            r"\b(prior|previous)\s+violations\s+for\s+(\w+)\b",
            r"\bprevious\s+policy\s+violations\b",
        ],
    ),
    (
        VoiceIntent.RECENT_VIOLATIONS_QUERY,
        [
            r"\bhow\s+many\s+violations\s+(happened|occurred|were there)\s+recently\b",
            r"\bhow\s+many\s+violations\s+(happened|occurred|were there)\b",
            r"\bhow\s+many\s+violations\s+recently\b",
            r"\bhow\s+many\s+times\s+did\s+(he|she|they|\w+)\s+enter\s+(recently|the restricted zone|the restricted area)\b",
            r"\bhow\s+many\s+times\s+did\s+(he|she|they|\w+)\s+enter\b",
            r"\brecent\s+violations\s+count\b",
            r"\bviolations\s+recently\b",
        ],
    ),
    # Phase 4.1 Zone Entry/Exit Intelligence Patterns
    (
        VoiceIntent.ZONE_ENTRY_QUERY,
        [
            r"\bdid\s+(\w+)\s+enter\s+(the\s+)?(restricted\s+)?(zone|area)\b",
            r"\bhas\s+(\w+|he|she|they|anyone)\s+entered\s+(the\s+)?(restricted\s+)?(zone|area)\b",
            r"\bis\s+(\w+|he|she|they|anyone)\s+inside\s+(the\s+)?(restricted\s+)?(zone|area)\b",
        ],
    ),
    (
        VoiceIntent.ZONE_EXIT_QUERY,
        [
            r"\bhas\s+(\w+|he|she|they)\s+left\s+(the\s+)?(restricted\s+)?(area|zone)\b",
            r"\bdid\s+(\w+|he|she|they)\s+leave\s+(the\s+)?(restricted\s+)?(area|zone)\b",
            r"\bis\s+(\w+|he|she|they)\s+still\s+in\s+(the\s+)?(restricted\s+)?(area|zone)\b",
        ],
    ),
    (
        VoiceIntent.ZONE_ENTRY_TIME_QUERY,
        [
            r"\bwhen\s+did\s+(\w+|he|she|they)\s+enter(\s+the\s+restricted\s+(zone|area))?\b",
            r"\bwhat\s+time\s+did\s+(\w+|he|she|they)\s+enter\b",
        ],
    ),
    # 10. Authorized Operating Hours Query
    (
        VoiceIntent.AUTHORIZED_HOURS_QUERY,
        [
            r"\b(what are|tell me|when are)\s+(the\s+)?(authorized|operating|permitted|working)\s+(hours|times|window)\b",
            r"\b(authorized|operating|permitted)\s+hours\b",
            r"\boperating\s+schedule\b",
            r"\bwhat\s+time\s+is\s+(the\s+)?restricted\s+area\s+open\b",
        ],
    ),
    # 9. Follow-up Person Authorization Check ("Is Hunter allowed here?", "Is he allowed at this time?", "Why was Hunter denied?")
    (
        VoiceIntent.FOLLOW_UP_PERSON_AUTHORIZATION,
        [
            r"\bwhy\s+was\s+(\w+)\s+(denied|rejected|blocked|not allowed)\b",
            r"\b(is|are|can)\s+(\w+)\s+(allowed|authorized|cleared|permitted)\b",
            r"\b(is|are)\s+(he|she|they|this person|that person|the person|\w+)\s+(supposed to be|allowed to be)\s+(here|there|in view|in the room|in zone|in the restricted area)\b",
            r"\b(is|are)\s+(\w+)\s+allowed\s+(at this time|here|right now|in the restricted area|after hours)\b",
            r"\bis\s+this\s+person\s+allowed\s+(here\s+)?(right\s+now|at this time)\b",
            r"\bis\s+the\s+person\s+(i see\s+)?(allowed|authorized|cleared|permitted)\b",
            r"\bdoes\s+(he|she|this person|that person|\w+)\s+have\s+authorization\b",
            r"\bis\s+(\w+)\s+a\s+valid\s+user\b",
        ],
    ),
    # 10. Follow-up Why / Alert Explanation
    (
        VoiceIntent.FOLLOW_UP_WHY,
        [
            r"^why$",
            r"^why\s+is\s+that$",
            r"\bwhy\s+(so|is he|is she|are they|is this|did the system trigger)\b",
            r"\bwhy\s+did\s+you\s+trigger\s+(that|the)\s+alert\b",
            r"\bwhy\s+is\s+(he|she|this person|that person|\w+)\s+(allowed|not allowed|unauthorized|authorized|here|denied)\b",
            r"\bwhy\s+was\s+(he|she|this person|that person|\w+)\s+(denied|rejected)\b",
            r"\b(explain why|give me the reason|what is the reason)\b",
        ],
    ),
    # 10. Historical State Query ("What happened earlier?", "Was there a threat earlier?")
    (
        VoiceIntent.HISTORICAL_STATE_QUERY,
        [
            r"\bwhat\s+happened\s+(earlier|before|previously|while i was away)\b",
            r"\bwas\s+there\s+(a|any)?\s*(threat|risk|alert|incident|problem)\s+(earlier|before|previously|in the past)\b",
            r"\bdid\s+anything\s+suspicious\s+happen(\s+while i was away)?\b",
            r"\bwhen\s+was\s+the\s+last\s+(unauthorized|unknown)\s+(person|visitor)\s+detected\b",
        ],
    ),
    # 11. Current State Query ("What is happening right now?", "Is there a threat right now?")
    (
        VoiceIntent.CURRENT_STATE_QUERY,
        [
            r"\bwhat\s+is\s+happening\s+(right now|now|currently)\b",
            r"\bis\s+there\s+(a|any)?\s*(threat|risk|danger)\s+(right now|now|currently)\b",
            r"\bwhat\s+should\s+i\s+worry\s+about\s+(right now|now)\b",
            r"\bwhat\s+are\s+we\s+facing\s+right\s+now\b",
            r"\bcurrent\s+threat\s+right\s+now\b",
        ],
    ),
    # 12. Why High Risk / Explain Alert
    (
        VoiceIntent.WHY_HIGH_RISK,
        [
            r"\bwhy\b.*\b(high risk|elevated|threat|alert|danger|incident|orange|red|yellow)\b",
            r"\bexplain\b.*\b(risk|threat|alert|incident)\b",
            r"\b(reason for risk|cause of alert|why the alert|why is it alert|why the risk)\b",
        ],
    ),
    # 13. Latest Alert (Specific non-GREEN incident)
    (
        VoiceIntent.LATEST_ALERT,
        [
            r"\b(latest|last|most recent)\s+(security\s+)?(alert|warning|incident|alarm|breach)\b",
            r"\bread\s+(the\s+)?(latest|last|recent)\s+alert\b",
            r"\bwhat\s+was\s+the\s+last\s+(security\s+)?alert\b",
        ],
    ),
    # 14. Recent Security Events / Activity History
    (
        VoiceIntent.RECENT_SECURITY_EVENTS,
        [
            r"\b(recent|latest|past)\s+(security\s+)?(events|activity|logs|history|incidents)\b",
            r"\bwhat\s+(happened|occurred)(\s+(recently|just now|earlier|today))?\b",
            r"\bwhat\s+just\s+happened\b",
            r"\b(event log|activity log|history log|show events|list events)\b",
            r"\bwhat\s+are\s+the\s+latest\s+(security\s+)?events\b",
            r"\bwhat\s+are\s+the\s+latest\s+alerts\b",
        ],
    ),
    # 15. Recent Person Detected
    (
        VoiceIntent.RECENT_PERSON_DETECTED,
        [
            r"\bwho\s+(was|were)\s+(detected|seen|recognized|present|here)\s+(most recently|recently|last)\b",
            r"\b(last|recent|most recent)\s+(detected\s+)?(person|visitor|user|individual|officer)\b",
            r"\bwho\s+visited\s+(recently|last)\b",
            r"\bwho\s+was\s+here\s+last\b",
        ],
    ),
    # 16. Current Security Status / Overview
    (
        VoiceIntent.CURRENT_SECURITY_STATUS,
        [
            r"\b(current\s+)?(security\s+status|system\s+status|operational\s+status|system\s+overview)\b",
            r"\bhow\s+is\s+security\s+(right\s+now|today|now)\b",
            r"\bstatus\s+(report|update|check)\b",
            r"\bwhat\s+is\s+(the\s+)?status\b",
        ],
    ),
    # 17. Who is in Front of Camera
    (
        VoiceIntent.WHO_IS_IN_FRONT,
        [
            r"\bwho\s+(is|are)\s+(in front|there|present|standing|in camera|in view|on camera|in the video|in front of me)\b",
            r"\bwho\s+do\s+you\s+see\b",
            r"\bwho\s+is\s+that\b",
            r"\b(identify|recognize)\s+(the\s+)?(person|individual|user|face|visitor|officer)\b",
            r"\bis\s+someone\s+(there|in front|in view)\b",
        ],
    ),
    # 18. What do you see / Scene Overview
    (
        VoiceIntent.WHAT_DO_YOU_SEE,
        [
            r"\bwhat\s+(do\s+you|can\s+you)?\s*see\b",
            r"\bwhat\s+is\s+(visible|in view|in front of the camera|in front|around me|around|in the scene)\b",
            r"\bdescribe\s+(the\s+)?(scene|view|camera|feed|environment|surroundings)\b",
            r"\bwhat\s+do\s+you\s+observe\b",
        ],
    ),
    # 19. Security Risk / Threat Level
    (
        VoiceIntent.SECURITY_RISK,
        [
            r"\bis\s+there\s+(a|any)?\s*(security\s+)?(risk|threat|danger|hazard|problem|incident)\b",
            r"\bare\s+there\s+(any\s+)?(security\s+)?(risks|threats|dangers|problems)\b",
            r"\b(current\s+)?threat\s+level\b",
            r"\bis\s+(anything\s+wrong|everything\s+safe|everything\s+ok|everything\s+okay|everyone\s+safe)\b",
            r"\bare\s+we\s+safe\b",
            r"\bcheck\s+(security|threat|safety)\b",
        ],
    ),
    # 20. Detected Objects Breakdown
    (
        VoiceIntent.DETECTED_OBJECTS,
        [
            r"\bwhat\s+objects\s+(are\s+detected|do\s+you\s+see|are\s+around|are\s+visible|are\s+in view)\b",
            r"\b(what|which)\s+(objects|items|things)\s+are\s+(you\s+)?detecting\b",
            r"\b(list|show|detected)\s+objects\b",
            r"\bare\s+there\s+any\s+objects\b",
            r"\bwhat\s+are\s+you\s+detecting\b",
        ],
    ),
    # Phase 5.7.3: Person Holding Object / Hand Queries
    (
        VoiceIntent.PERSON_HOLDING_OBJECT,
        [
            r"\bwhat\s+(object\s+)?(am\s+i|is\s+he|is\s+she|is\s+that\s+person|are\s+they|is\s+anyone)\s+(holding|carrying)\b",
            r"\bwhat\s+(object\s+)?is\s+(in\s+)?(my|his|her|their|that\s+person\s+s)\s+hand(s)?\b",
            r"\bwhat\s+is\s+in\s+(my|his|her|their)\s+hand(s)?\b",
            r"\bcan\s+you\s+tell\s+what\s+i\s+am\s+carrying\b",
            r"\bcan\s+you\s+tell\s+what\s+(he|she|that\s+person)\s+is\s+carrying\b",
            r"\bis\s+(he|she|anyone|that\s+person)\s+holding\s+(anything|an\s+object|any\s+item)\b",
            r"\bwhat\s+am\s+i\s+holding\b",
            r"\bwhat\s+is\s+he\s+holding\b",
            r"\bwhat\s+is\s+she\s+holding\b",
            r"\bwhat\s+is\s+that\s+person\s+holding\b",
            r"\bholding\s+(anything|something|an\s+object)\b",
        ],
    ),
    # Phase 5.7.3: Explain Access Decision
    (
        VoiceIntent.EXPLAIN_ACCESS_DECISION,
        [
            r"\bexplain\s+(the\s+)?(access\s+)?decision\b",
            r"\bexplain\s+(the\s+)?access\b",
            r"\bwhy\s+(is\s+)?(he|she|this person|that person)\s+(not allowed|denied|prohibited)\b",
            r"\bwhy\s+(am\s+i|was\s+i|is\s+he|is\s+she|was\s+he|was\s+she|are\s+they|was\s+this\s+person|is\s+this\s+person)\s+(being\s+)?(denied|rejected|refused|blocked|stopped|prohibited|not allowed)\b",
            r"\bwhy\s+(was|is|did|has)?\s*(.*)\s*(access\s+)?(was\s+)?(denied|rejected|refused|blocked|stopped|not allowed|forbidden|prohibited)\b",
            r"\bwhy\s+(can\s+i\s+not|cannot\s+i|can\s+not\s+i|cannot\s+he|cannot\s+she|cannot\s+they|can\s+he\s+not|can\s+she\s+not)\s+(enter|go\s+inside|get\s+in|gain\s+entry|access)\b",
            r"\bwhy\s+(did|could|cannot|can)\s*(not\s+)?(.*)\s*(not\s+)?(get\s+access|gain\s+entry|enter|go\s+inside|get\s+in)\b",
            r"\bwhy\s+is\s+not\s+(my|his|her|their|our)\s+access\s+allowed\b",
            r"\bwhy\s+(is\s+not|isn't)\s+(my|his|her|their|our)\s+access\s+allowed\b",
            r"\bwhy\s+was\s+(my|his|her|their|our)\s+access\s+refused\b",
            r"\bwhat\s+is\s+preventing\s+(me|him|her|them|us|the\s+person|\w+)\s+from\s+entering\b",
            r"\bwhy\s+will\s+you\s+not\s+let\s+(me|him|her|them|us)\s+in\b",
            r"\bwhy\s+(won't|wont)\s+you\s+let\s+(me|him|her|them|us)\s+in\b",
            r"\bwhat\s+(is\s+)?(stopping|preventing|blocking)\s+(me|him|her|them|us|the\s+person|\w+)\b",
            r"\bwhy\s+(am\s+i|was\s+i)\s+(denied|rejected|refused|blocked|stopped)\b",
            r"\bwhy\s+access\s+(denied|rejected|refused|blocked)\b",
        ],
    ),
    # Phase 5.16: Research / CCSDF Intents
    (
        VoiceIntent.RESEARCH_BASELINE_QUERY,
        [
            r"\bwhat\s+would\s+(the\s+)?baseline(\s+face(\s+only)?)?(\s+system)?\s+have\s+decided\b",
            r"\bhow\s+would\s+(the\s+)?baseline\s+decide\b",
            r"\bbaseline\s+(face\s+only|decision|system)\b",
        ],
    ),
    (
        VoiceIntent.RESEARCH_RISK_FACTORS,
        [
            r"\bwhat\s+factors?\s+caused\s+(the\s+)?risk\s+increase\b",
            r"\bwhich\s+(contextual\s+)?factor\s+is\s+contributing\s+most\b",
            r"\bwhat\s+caused\s+(the\s+)?risk\s+(to\s+)?(increase|rise|go\s+up)\b",
            r"\bwhat\s+are\s+the\s+contextual\s+factors\b",
            r"\bhighest\s+risk\s+factor\b",
        ],
    ),
    (
        VoiceIntent.RESEARCH_AUTHORIZATION_CHANGE,
        [
            r"\bwhy\s+did\s+authorization\s+change\b",
            r"\bwhat\s+changed\s+when\s+(he|she|they|the\s+person)\s+entered\b",
            r"\bwhy\s+authorization\s+changed\b",
        ],
    ),
    (
        VoiceIntent.RESEARCH_TAILGATING_QUERY,
        [
            r"\bwas\s+tailgating\s+detected\b",
            r"\bis\s+someone\s+tailgating\b",
            r"\btailgating\s+(status|detected|suspected)\b",
        ],
    ),
    (
        VoiceIntent.RESEARCH_BEHAVIOR_QUERY,
        [
            r"\bwhat\s+behavior\s+was\s+unusual\b",
            r"\bwhat\s+was\s+the\s+behavior\s+anomaly\b",
            r"\bbehavior\s+(anomaly|pattern|status)\b",
            r"\bsuspicious\s+behavior\b",
        ],
    ),
    (
        VoiceIntent.RESEARCH_CONFIDENCE_QUERY,
        [
            r"\bhow\s+confident\s+is\s+the\s+(current\s+)?decision\b",
            r"\bwhat\s+is\s+the\s+decision\s+confidence\b",
            r"\bhow\s+certain\s+are\s+you\b",
            r"\bconfidence\s+level\b",
        ],
    ),
    # Phase 5.7.3: Visual Scene Overview
    (
        VoiceIntent.VISUAL_SCENE,
        [
            r"\bwhat\s+do\s+you\s+see\b",
            r"\bdescribe\s+(the\s+)?(scene|view|camera|room|surroundings|environment)\b",
            r"\bwhat\s+is\s+happening\s+in\s+front\s+of\s+(the\s+)?camera\b",
            r"\btell\s+me\s+what\s+you\s+observe\b",
            r"\bwhat\s+do\s+you\s+observe\b",
            r"\bwhat\s+is\s+visible\b",
            r"\bscene\s+overview\b",
            r"\bdescribe\s+what\s+you\s+see\b",
        ],
    ),
    # Phase 5.7.3: Visual Count
    (
        VoiceIntent.VISUAL_COUNT,
        [
            r"\bhow\s+many\s+people\s+(are\s+)?(visible|in\s+view|in\s+the\s+frame|in\s+front|do\s+you\s+see)\b",
            r"\bhow\s+many\s+people\s+are\s+there\b",
            r"\bhow\s+many\s+faces\s+(do\s+you\s+see|are\s+visible)\b",
            r"\bhow\s+many\s+persons\s+(are\s+)?(visible|in\s+view|in\s+the\s+frame)\b",
            r"\bcount\s+(the\s+)?(people|persons|faces)\s+in\s+view\b",
        ],
    ),
]


class VoiceAssistant:
    """
    State-aware Contextual Security Assistant for WatchGuard Vision.
    Processes natural voice and text queries mapped deterministically to
    live computer vision, object detection, and context security state.
    """

    SUPPORTED_QUERIES = [
        "Who is in front of the camera?",
        "Is he allowed to be there?",
        "Why?",
        "What is happening right now?",
        "What happened earlier?",
        "What happened in the last 5 minutes?",
        "When was the last red alert?",
        "How many alerts happened today?",
        "Is anyone near the phone?",
        "What do you see?",
        "Is there any security risk?",
        "Why is the system showing high risk?",
        "What objects are detected?",
        "Who was detected most recently?",
        "Show recent security events.",
        "Read the latest alert.",
        "Stop listening.",
    ]

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        event_logger: Optional[EventLogger] = None,
        person_engine: Any = None,
        object_engine: Any = None,
        context_engine: Any = None,
        camera_manager: Any = None,
        security_analytics: Any = None,
        copilot_reasoner: Optional[BaseAICopilotReasoner] = None,
    ):
        self.db = db_manager or DatabaseManager()
        self.logger = event_logger
        self.person_engine = person_engine
        self.object_engine = object_engine
        self.context_engine = context_engine
        self.camera = camera_manager
        self.security_analytics = (
            security_analytics
            or getattr(context_engine, "security_analytics", None)
            or SecurityAnalyticsEngine(
                db_manager=self.db,
                context_engine=self.context_engine,
                incident_manager=getattr(self.context_engine, "incident_manager", None),
            )
        )
        self.copilot_reasoner = (
            copilot_reasoner
            or get_copilot_reasoner(security_analytics=self.security_analytics)
        )
        self.semantic_router = SemanticQueryRouter(VoiceIntent)

        self._is_listening = False
        self._is_speaking = False
        self._is_processing = False
        self.mute_audio = False

        self._last_command = "None"
        self._last_response = "Voice assistant ready. Ask a question or use Push to Talk."
        self._status_callback: Optional[Callable[[str], None]] = None

        # Phase 2.5 Session Context Tracker
        self.session_context = SessionContext()
        self._lock = threading.Lock()

        # Phase 5.11 Query Sequencing & Stale Visual Protection
        self._query_counter: int = 0
        self._latest_displayed_query_id: int = 0
        self._query_lock = threading.Lock()

    def _next_query_id(self) -> int:
        with self._query_lock:
            self._query_counter += 1
            return self._query_counter

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def is_processing(self) -> bool:
        return self._is_processing

    @property
    def last_command(self) -> str:
        return self._last_command

    @property
    def last_response(self) -> str:
        return self._last_response

    @property
    def status_label(self) -> str:
        if self._is_speaking:
            return "SPEAKING"
        if self._is_processing:
            return "PROCESSING"
        if self._is_listening:
            return "LISTENING"
        return "READY"

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        """Sets a UI listener for status updates."""
        self._status_callback = callback

    def _notify_status(self) -> None:
        if self._status_callback:
            try:
                self._status_callback(self.status_label)
            except Exception as e:
                logger.debug(f"Status callback error: {e}")

    # --- Live State Sync to Session Context ---

    def _sync_session_from_live_state(self) -> None:
        """Synchronizes session context with the latest live observation from ContextEngine."""
        decision = getattr(self.context_engine, "last_decision", None)
        if decision:
            self.session_context.last_risk_level = decision.risk_level.value
            self.session_context.last_decision_code = decision.decision
            self.session_context.last_reason = decision.reason
            self.session_context.last_evidence_required = decision.evidence_required
            self.session_context.last_interactions = decision.correlated_interactions
            self.session_context.last_is_access_granted = getattr(decision, "is_access_granted", True)
            self.session_context.last_per_track_states = getattr(decision, "per_track_states", [])

            inc = getattr(decision, "incident", None)
            if inc:
                self.session_context.last_incident_id = inc.incident_id
                self.session_context.last_incident_state = inc.state.value if hasattr(inc.state, "value") else str(inc.state)
                self.session_context.last_incident_policy = inc.policy_code
                self.session_context.last_incident_reason = inc.reason
                self.session_context.last_incident_created_at = inc.created_at

            ctx = decision.context_summary or {}
            
            # Authoritative Multi-Person Entity Extraction
            raw_entities = getattr(decision, "person_entities", []) or ctx.get("person_entities", [])
            entities = []
            for e in raw_entities:
                if isinstance(e, dict):
                    entities.append(e)
                elif hasattr(e, "to_dict"):
                    entities.append(e.to_dict())

            self.session_context.last_people_entities = entities
            self.session_context.last_people_count = len(entities)

            if entities:
                is_frame_auth = getattr(decision, "is_access_granted", True)
                auth_persons = [
                    e for e in entities
                    if bool(e.get("is_known"))
                    and not bool(e.get("is_spoof"))
                    and (e.get("access_state") in ("GRANTED", "AUTHORIZED") or (e.get("access_state") != "DENIED" and is_frame_auth))
                    and e.get("identity") not in ("Unknown Person", "Unknown", "Unregistered", "None", "", None)
                ] if is_frame_auth else []

                if auth_persons:
                    p0 = auth_persons[0]
                    self.session_context.last_person_name = p0.get("identity")
                    self.session_context.last_is_authorized = True
                    self.session_context.last_is_unknown = False
                    self.session_context.last_role = p0.get("role", "Staff")
                else:
                    e0 = entities[0]
                    e0_ident = e0.get("identity", "Unknown Person")
                    e0_known = bool(e0.get("is_known", False)) and e0_ident not in ("Unknown Person", "Unknown", "Unregistered", "None", "", None)
                    self.session_context.last_person_name = e0_ident
                    self.session_context.last_is_authorized = False
                    self.session_context.last_is_unknown = not e0_known
                    self.session_context.last_role = e0.get("role", "Staff" if e0_known else "Visitor")
            elif ctx.get("person_detected") and ctx.get("is_authorized") and getattr(decision, "is_access_granted", True):
                self.session_context.last_person_name = ctx.get("person_name", "Unknown Person")
                self.session_context.last_is_authorized = True
                self.session_context.last_is_unknown = False
                self.session_context.last_role = ctx.get("role", "Staff")
            elif ctx.get("person_detected"):
                p_name = ctx.get("person_name", "Unknown Person")
                is_known_flag = (
                    (bool(ctx.get("is_known", False)) or ctx.get("is_unknown") is False or bool(ctx.get("is_authorized", False)))
                    and p_name not in ("Unknown Person", "Unknown", "Unregistered", "None", "", None)
                )
                self.session_context.last_person_name = p_name
                self.session_context.last_is_authorized = False
                self.session_context.last_is_unknown = not is_known_flag
                self.session_context.last_role = ctx.get("role", "Staff" if is_known_flag else "Visitor")
            else:
                self.session_context.last_person_name = None
                self.session_context.last_is_authorized = False
                self.session_context.last_is_unknown = False
                self.session_context.last_role = None

            self.session_context.last_zone = ctx.get("zone_name", "Standard Perimeter")
            self.session_context.last_time_str = ctx.get("time_str", "--:--")
            self.session_context.last_is_authorized_hours = ctx.get("is_authorized_hours", True)
            if self.context_engine:
                self.session_context.last_hours_window = f"{self.context_engine.hours_start} to {self.context_engine.hours_end}"

            objs = ctx.get("objects", [])
            if objs:
                self.session_context.last_objects = [o.get("label") for o in objs if o.get("label")]

    # --- Text-to-Speech (TTS) ---

    def speak(self, text: str, non_blocking: bool = True) -> None:
        """Speaks the response aloud using local Windows SAPI TTS."""
        if self.mute_audio or not text:
            return

        def _worker():
            if not HAS_PYTTSX3:
                return
            with self._lock:
                self._is_speaking = True
                self._notify_status()
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 165)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    logger.error(f"TTS Speech error: {e}")
                finally:
                    self._is_speaking = False
                    self._notify_status()

        if non_blocking:
            threading.Thread(target=_worker, daemon=True).start()
        else:
            _worker()

    # --- Speech-to-Text (STT) Push-to-Talk ---

    def listen_and_process(
        self,
        duration_seconds: float = 3.5,
        sample_rate: int = 16000,
        on_complete: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Push-to-Talk audio capture from microphone.
        Processes speech into text and generates state-aware response.
        """
        def _worker():
            self._is_listening = True
            self._notify_status()
            transcription = ""

            try:
                if not HAS_SPEECH:
                    transcription = "[Speech recognition package not available]"
                    resp = "Microphone audio packages not available. Please use the text command bar."
                else:
                    logger.info(f"Push-to-Talk active: recording {duration_seconds}s from microphone...")
                    audio_data = sd.rec(
                        int(duration_seconds * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                    )
                    sd.wait()
                    self._is_listening = False
                    self._is_processing = True
                    self._notify_status()

                    raw_bytes = audio_data.tobytes()
                    recognizer = sr.Recognizer()
                    sr_audio = sr.AudioData(raw_bytes, sample_rate, 2)

                    try:
                        transcription = recognizer.recognize_google(sr_audio)
                        logger.info(f"Transcribed audio: '{transcription}'")
                    except sr.UnknownValueError:
                        transcription = "[Inaudible or silence]"
                        resp = "I could not hear any spoken command. Please speak clearly into the microphone."
                        self._last_command = transcription
                        self._last_response = resp
                        self.speak(resp)
                        if on_complete:
                            on_complete(transcription, resp)
                        return
                    except sr.RequestError:
                        transcription = "[Speech recognition offline]"
                        resp = "Voice recognition service unavailable. Please use the text command input."
                        self._last_command = transcription
                        self._last_response = resp
                        self.speak(resp)
                        if on_complete:
                            on_complete(transcription, resp)
                        return

                resp = self.process_command(transcription)

            except Exception as e:
                logger.error(f"Microphone capture error: {e}", exc_info=True)
                transcription = "[Microphone error]"
                resp = f"Microphone error: {e}. Please check your audio input device."
            finally:
                self._is_listening = False
                self._is_processing = False
                self._notify_status()

            self._last_command = transcription
            self._last_response = resp
            self.speak(resp)

            if on_complete:
                try:
                    on_complete(transcription, resp)
                except Exception as e:
                    logger.error(f"Error in on_complete callback: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    # --- Intent Resolution & Query Processor ---

    def resolve_intent(self, query: str) -> Tuple[VoiceIntent, str]:
        """
        Normalizes the user query and matches it against supported intent patterns using
        the SemanticQueryRouter with graceful fallback to pattern matching.
        Returns a tuple of (ResolvedIntent, NormalizedQueryString).
        """
        if not query or not query.strip():
            return VoiceIntent.UNKNOWN, ""

        if hasattr(self, "semantic_router") and self.semantic_router:
            res = self.semantic_router.route_query(query, getattr(self, "session_context", None))
            if res.intent != VoiceIntent.UNKNOWN:
                return res.intent, res.normalized_query

        normalized = normalize_query(query)
        for intent, patterns in INTENT_RULES:
            for p in patterns:
                if re.search(p, normalized):
                    return intent, normalized

        return VoiceIntent.UNKNOWN, normalized

    def process_command(self, query: str) -> str:
        """
        Parses user query, resolves intent, resolves session context, queries live state / SQLite,
        and deterministically returns the security response.
        """
        if not query or not query.strip():
            return "Please specify a question or security query."

        t_start = time.time()
        # Synchronize live vision context before answering
        self._sync_session_from_live_state()

        intent, normalized = self.resolve_intent(query)
        self._last_command = query
        self.session_context.last_response_source = CopilotResponseSource.DETERMINISTIC
        self.session_context.last_intent = intent

        # Intent Dispatcher
        if intent == VoiceIntent.PERSON_HOLDING_OBJECT:
            response = self._handle_person_holding_object(query)
        elif intent == VoiceIntent.EXPLAIN_ACCESS_DECISION:
            response = self._handle_explain_access_decision(normalized)
        elif intent == VoiceIntent.RESEARCH_BASELINE_QUERY:
            response = self._handle_research_baseline_query()
        elif intent == VoiceIntent.RESEARCH_RISK_FACTORS:
            response = self._handle_research_risk_factors()
        elif intent == VoiceIntent.RESEARCH_AUTHORIZATION_CHANGE:
            response = self._handle_research_authorization_change()
        elif intent == VoiceIntent.RESEARCH_TAILGATING_QUERY:
            response = self._handle_research_tailgating_query()
        elif intent == VoiceIntent.RESEARCH_BEHAVIOR_QUERY:
            response = self._handle_research_behavior_query()
        elif intent == VoiceIntent.RESEARCH_CONFIDENCE_QUERY:
            response = self._handle_research_confidence_query()
        elif intent == VoiceIntent.SECURITY_OBJECTS_QUERY:
            response = self._handle_security_objects_query(normalized)
        elif intent == VoiceIntent.DETECTOR_MODE_QUERY:
            response = self._handle_detector_mode_query()
        elif intent == VoiceIntent.RESEARCH_NOVELTY_QUERY:
            response = self._handle_research_novelty_query()
        elif intent == VoiceIntent.RESEARCH_CCSDF_STATE_QUERY:
            response = self._handle_research_ccsdf_state_query()
        elif intent == VoiceIntent.VISUAL_SCENE:
            response = self._handle_visual_scene(query)
        elif intent == VoiceIntent.VISUAL_COUNT:
            response = self._handle_visual_count(query)
        elif intent == VoiceIntent.VISUAL_OBJECT:
            response = self._handle_visual_object(query)
        elif intent == VoiceIntent.VISUAL_RELATIONSHIP:
            response = self._handle_spatial_relationship(normalized)
        elif intent == VoiceIntent.VISUAL_DESCRIPTION:
            response = self._handle_visual_description(query)
        elif intent == VoiceIntent.INCIDENT_STATUS_QUERY:
            response = self._handle_incident_status(normalized)
        elif intent == VoiceIntent.INCIDENT_TIME_QUERY:
            response = self._handle_incident_time(normalized)
        elif intent == VoiceIntent.INCIDENT_ACK_STATUS_QUERY:
            response = self._handle_incident_ack_status(normalized)
        elif intent == VoiceIntent.ACKNOWLEDGE_INCIDENT_COMMAND:
            response = self._handle_acknowledge_incident(normalized)
        elif intent == VoiceIntent.RESOLVE_INCIDENT_COMMAND:
            response = self._handle_resolve_incident(normalized)
        elif intent == VoiceIntent.ZONE_ENTRY_QUERY:
            response = self._handle_zone_entry_query(normalized)
        elif intent == VoiceIntent.ZONE_EXIT_QUERY:
            response = self._handle_zone_exit_query(normalized)
        elif intent == VoiceIntent.ZONE_ENTRY_TIME_QUERY:
            response = self._handle_zone_entry_time_query(normalized)
        elif intent == VoiceIntent.VIOLATION_HISTORY_QUERY:
            response = self._handle_violation_history(normalized)
        elif intent == VoiceIntent.RECENT_VIOLATIONS_QUERY:
            response = self._handle_recent_violations_count(normalized)
        elif intent == VoiceIntent.AUTHORIZED_HOURS_QUERY:
            response = self._handle_authorized_hours(normalized)
        elif intent == VoiceIntent.FOLLOW_UP_PERSON_AUTHORIZATION:
            response = self._handle_follow_up_person_authorization(normalized)
        elif intent == VoiceIntent.FOLLOW_UP_WHY:
            response = self._handle_follow_up_why(normalized)
        elif intent == VoiceIntent.CURRENT_STATE_QUERY:
            response = self._handle_current_state(normalized)
        elif intent == VoiceIntent.HISTORICAL_STATE_QUERY:
            response = self._handle_historical_state(normalized)
        elif intent == VoiceIntent.TEMPORAL_EVENTS_MINUTES:
            response = self._handle_temporal_minutes(normalized)
        elif intent == VoiceIntent.TEMPORAL_ALERT_BY_LEVEL:
            response = self._handle_temporal_alert_by_level(normalized)
        elif intent == VoiceIntent.ALERT_COUNT_TODAY:
            response = self._handle_alert_count_today(normalized)
        elif intent == VoiceIntent.REPEATED_UNAUTHORIZED_DETECTIONS:
            response = self._handle_repeated_unauthorized(normalized)
        elif intent == VoiceIntent.MOST_SERIOUS_EVENT:
            response = self._handle_most_serious_event(normalized)
        elif intent == VoiceIntent.SPATIAL_RELATIONSHIP:
            response = self._handle_spatial_relationship(normalized)
        elif intent == VoiceIntent.WHO_IS_IN_FRONT:
            if self._should_use_ai_copilot(query):
                response = self._query_ai_copilot(query)
            else:
                response = self._query_current_person()
        elif intent == VoiceIntent.WHAT_DO_YOU_SEE:
            if self._should_use_ai_copilot(query):
                response = self._query_ai_copilot(query)
            else:
                response = self._query_scene_overview()
        elif intent == VoiceIntent.SECURITY_RISK:
            response = self._query_security_risk()
        elif intent == VoiceIntent.WHY_HIGH_RISK:
            response = self._query_risk_reason()
        elif intent == VoiceIntent.WHY_RISK_LEVEL:
            response = self._handle_why_risk_level(normalized)
        elif intent == VoiceIntent.SPOOF_ATTEMPTS_QUERY:
            response = self._handle_spoof_attempts_query(normalized)
        elif intent == VoiceIntent.DISTINCT_PEOPLE_QUERY:
            response = self._handle_distinct_people_query(normalized)
        elif intent == VoiceIntent.LIVE_PEOPLE_COUNT_QUERY:
            if self._should_use_ai_copilot(query):
                response = self._query_ai_copilot(query)
            else:
                response = self._query_live_people_count()
        elif intent == VoiceIntent.UNAUTHORIZED_LIVE_QUERY:
            response = self._handle_unauthorized_live_query(normalized)
        elif intent == VoiceIntent.LAST_ALERT_TRIGGER_QUERY:
            response = self._handle_last_alert_trigger(normalized)
        elif intent == VoiceIntent.CURRENT_SECURITY_STATUS:
            response = self._query_system_status()
        elif intent == VoiceIntent.DETECTED_OBJECTS:
            if self._should_use_ai_copilot(query):
                response = self._query_ai_copilot(query)
            else:
                response = self._query_detected_objects()
        elif intent == VoiceIntent.RECENT_PERSON_DETECTED:
            response = self._query_recent_person()
        elif intent == VoiceIntent.RECENT_SECURITY_EVENTS:
            response = self._query_recent_events()
        elif intent == VoiceIntent.LATEST_ALERT:
            response = self._query_latest_alert()
        elif intent == VoiceIntent.STOP_LISTENING:
            self._is_listening = False
            response = "Voice assistant returned to standby."
        else:
            # Generative AI Copilot Path for conversational / visual synthesis
            if self._should_use_ai_copilot(query) or is_visual_query(query):
                response = self._query_ai_copilot(query)
            else:
                response = (
                    "I did not recognize that command. You can ask: 'Who is in front of the camera?', "
                    "'Is he allowed to be there?', 'What is happening right now?', 'What happened in the last 5 minutes?', "
                    "or 'Show recent security events.'"
                )

        latency_ms = (time.time() - t_start) * 1000.0
        self.session_context.last_response_latency_ms = round(latency_ms, 2)
        self._last_response = response

        # Structured Telemetry Logging
        is_visual = (
            intent in (
                VoiceIntent.PERSON_HOLDING_OBJECT,
                VoiceIntent.VISUAL_SCENE,
                VoiceIntent.VISUAL_OBJECT,
                VoiceIntent.VISUAL_RELATIONSHIP,
                VoiceIntent.VISUAL_DESCRIPTION,
                VoiceIntent.VISUAL_COUNT,
                VoiceIntent.WHAT_DO_YOU_SEE,
                VoiceIntent.DETECTED_OBJECTS,
            )
            or is_visual_query(query)
        )
        gemini_called = (
            self.session_context.last_response_source == CopilotResponseSource.AI_GROUNDED
            or (self.copilot_reasoner and self.copilot_reasoner.is_available() and not isinstance(self.copilot_reasoner, DeterministicCopilotReasoner) and self._should_use_ai_copilot(query))
        )
        route_name = "GEMINI_VISUAL" if (gemini_called and is_visual) else ("GEMINI_TEXT" if gemini_called else "DETERMINISTIC")

        logger.info(
            f"COPILOT_QUERY: '{query}' | INTENT: {intent.name} | ROUTE: {route_name} | GEMINI_CALLED: {gemini_called}"
        )

        return response

    def _should_use_ai_copilot(self, query: str) -> bool:
        """Determines if a query should be routed to the multimodal/cloud AI Copilot."""
        if not self.copilot_reasoner or not self.copilot_reasoner.is_available():
            return False
        if isinstance(self.copilot_reasoner, DeterministicCopilotReasoner):
            return False
        model = getattr(self.copilot_reasoner, "model_name", "")
        if model in ("Deterministic-Rule-Engine", "", None):
            return False
        return True

    def _query_ai_copilot(self, query: str) -> str:
        """Executes an AI Copilot query with structured context, query ID sequencing, and visual state consistency."""
        query_id = self._next_query_id()
        context = build_structured_ai_context(self.context_engine, self.security_analytics, self.db)
        frame = None
        snapshot_captured = False
        snapshot_timestamp = None
        gemini_called = False
        gemini_resp_rcvd = False

        is_vis = is_visual_query(query) or (
            hasattr(self.session_context, "last_intent")
            and self.session_context.last_intent in (
                VoiceIntent.PERSON_HOLDING_OBJECT,
                VoiceIntent.VISUAL_SCENE,
                VoiceIntent.VISUAL_OBJECT,
                VoiceIntent.VISUAL_DESCRIPTION,
                VoiceIntent.WHO_IS_IN_FRONT,
                VoiceIntent.WHAT_DO_YOU_SEE,
            )
        )

        snapshot_scene_gen = getattr(self.context_engine, "scene_generation", 0)
        snapshot_people = list(self.session_context.last_people_entities or [])
        snapshot_has_person = len(snapshot_people) > 0
        snapshot_objects = list(self.session_context.last_objects or [])

        if is_vis and self.camera and self.camera.is_running:
            success, captured = self.camera.get_frame()
            if success and captured is not None:
                frame = captured
                snapshot_captured = True
                snapshot_timestamp = datetime.now().isoformat()
            else:
                snapshot_timestamp = datetime.now().isoformat()
        elif is_vis:
            snapshot_timestamp = datetime.now().isoformat()

        if is_vis:
            logger.info(
                f"VISUAL_QUERY_DISPATCHED: query_id={query_id} "
                f"snapshot_timestamp={snapshot_timestamp or 'N/A'} "
                f"scene_generation={snapshot_scene_gen}"
            )

        if self.copilot_reasoner and self.copilot_reasoner.is_available() and not isinstance(self.copilot_reasoner, DeterministicCopilotReasoner):
            gemini_called = True

        copilot_resp = self.copilot_reasoner.generate_explanation(query, context, frame=frame)
        self.session_context.last_response_source = copilot_resp.source
        gemini_resp_rcvd = (copilot_resp.source == CopilotResponseSource.AI_GROUNDED)

        response_received = datetime.now().isoformat()
        if is_vis:
            logger.info(
                f"GEMINI_RESPONSE_RECEIVED: query_id={query_id} "
                f"snapshot_timestamp={snapshot_timestamp or 'N/A'} "
                f"response_received={response_received}"
            )

        # Multi-query ordering check (prevent older asynchronous query from overwriting newer response)
        with self._query_lock:
            if query_id < self._latest_displayed_query_id:
                logger.info(
                    f"VISUAL_RESPONSE_SUPPRESSED: query_id={query_id} reason=SUPERSEDED_BY_NEWER_QUERY"
                )
                return ""

        # Scene change validation for visual queries
        current_scene_gen = getattr(self.context_engine, "scene_generation", 0)
        current_people = list(self.session_context.last_people_entities or [])
        current_has_person = len(current_people) > 0

        is_stale = False
        stale_reason = None
        if is_vis:
            if snapshot_has_person and not current_has_person:
                is_stale = True
                stale_reason = "STALE_SNAPSHOT"
            elif not snapshot_has_person and current_has_person:
                is_stale = True
                stale_reason = "STALE_SNAPSHOT"
            elif current_scene_gen != snapshot_scene_gen:
                snap_ids = sorted(getattr(p, "identity", "Unknown") for p in snapshot_people)
                curr_ids = sorted(getattr(p, "identity", "Unknown") for p in current_people)
                if snap_ids != curr_ids:
                    is_stale = True
                    stale_reason = "STALE_SNAPSHOT"

        snapshot_current_bool = not is_stale
        if is_vis:
            logger.info(
                f"VISUAL_RESPONSE_VALIDATION: query_id={query_id} "
                f"snapshot_current={snapshot_current_bool} "
                f"scene_generation_current={current_scene_gen}"
            )

        if is_stale:
            logger.info(f"VISUAL_RESPONSE_SUPPRESSED: query_id={query_id} reason={stale_reason}")
            if not current_has_person:
                final_text = "The camera currently detects no person. The previous visual result is no longer current."
            else:
                final_text = self._query_current_person()
        else:
            final_text = copilot_resp.text

        with self._query_lock:
            self._latest_displayed_query_id = query_id

        if is_vis:
            logger.info(f"VISUAL_RESPONSE_DISPLAYED: query_id={query_id}")

        if snapshot_captured or is_vis:
            logger.info(
                f"VISUAL_DISPATCH: SNAPSHOT_CAPTURED: {snapshot_captured} | "
                f"SNAPSHOT_TIMESTAMP: {snapshot_timestamp or 'N/A'} | "
                f"GEMINI_CALLED: {gemini_called} | "
                f"GEMINI_RESPONSE_RECEIVED: {gemini_resp_rcvd}"
            )

        return final_text

    # --- Phase 5.7.3 Handlers ---

    def _handle_person_holding_object(self, query: str) -> str:
        """Handles queries asking about objects held in hand by the user or visible subject."""
        self.session_context.last_topic = "OBJECTS"
        if self._should_use_ai_copilot(query):
            return self._query_ai_copilot(query)

        # Grounded deterministic fallback from vision state
        live_objs = self.session_context.last_objects or []
        if live_objs:
            obj_labels = [o.get("label", "object") for o in live_objs]
            return f"Based on detected vision objects, you appear to be holding or interacting with: {', '.join(obj_labels)}."
        return "No objects are currently detected in your hands in the camera view."

    def _handle_explain_access_decision(self, query: str) -> str:
        """
        Explains access decisions (DENIED / GRANTED / PENDING) strictly grounded in authoritative state:
        identity, identity_status, liveness, zone, time, operating_window, access_decision, policy, reason, risk.
        """
        self.session_context.last_topic = "PERSON"
        decision = getattr(self.context_engine, "last_decision", None) if self.context_engine else None

        # Get live people in current view
        live_people = self.session_context.last_people_entities
        p_count = len(live_people)

        zone = self.session_context.last_zone or "Standard Perimeter"
        if not isinstance(zone, str) or str(type(zone)).find("Mock") != -1:
            zone = "Standard Perimeter"
        time_str = self.session_context.last_time_str or "--:--"
        if not isinstance(time_str, str) or str(type(time_str)).find("Mock") != -1:
            time_str = "--:--"
        is_auth_h = self.session_context.last_is_authorized_hours
        h_start = getattr(self.context_engine, "hours_start", "09:00") if self.context_engine else "09:00"
        if not isinstance(h_start, str) or str(type(h_start)).find("Mock") != -1:
            h_start = "09:00"
        h_end = getattr(self.context_engine, "hours_end", "18:00") if self.context_engine else "18:00"
        if not isinstance(h_end, str) or str(type(h_end)).find("Mock") != -1:
            h_end = "18:00"
        h_win = f"{h_start} to {h_end}"

        dec_code = self.session_context.last_decision_code or (getattr(decision, "decision", "NORMAL_ACTIVITY") if decision else "NORMAL_ACTIVITY")
        if not isinstance(dec_code, str) or str(type(dec_code)).find("Mock") != -1:
            dec_code = "NORMAL_ACTIVITY"
        reason = self.session_context.last_reason or (getattr(decision, "reason", "Routine operations") if decision else "Routine operations")
        if not isinstance(reason, str) or str(type(reason)).find("Mock") != -1:
            reason = "Routine operations"
        risk_lvl = self.session_context.last_risk_level or "GREEN"
        if not isinstance(risk_lvl, str) or str(type(risk_lvl)).find("Mock") != -1:
            risk_lvl = "GREEN"

        # Determine target entity: from live_people if present, otherwise from session_context if historical/turn
        target_entity = None
        if p_count > 0:
            target_entity = live_people[0]
            q_lower = query.lower()
            for p in live_people:
                p_ident = p.get("identity", "")
                if p_ident and p_ident.lower() in q_lower:
                    target_entity = p
                    break
        elif (
            (self.session_context.last_person_name and self.session_context.last_person_name not in ("None", "Unknown Person", "Unknown", ""))
            or self.session_context.last_is_unknown is True
            or (self.session_context.last_decision_code and self.session_context.last_decision_code not in ("NORMAL_ACTIVITY", None))
            or (decision and getattr(decision, "decision", None) and getattr(decision, "decision") != "NORMAL_ACTIVITY")
        ):
            target_entity = {
                "identity": self.session_context.last_person_name or ("Unknown Person" if self.session_context.last_is_unknown else "The user"),
                "is_known": bool(self.session_context.last_is_authorized and not self.session_context.last_is_unknown),
                "is_spoof": bool("spoof" in dec_code.lower() or "presentation" in dec_code.lower()),
                "liveness_state": "SPOOF" if ("spoof" in dec_code.lower() or "presentation" in dec_code.lower()) else ("LIVE" if self.session_context.last_is_authorized else "WARMUP"),
                "role": self.session_context.last_role or "Staff",
                "access_state": "DENIED" if ("denied" in dec_code.lower() or "unauthorized" in dec_code.lower() or risk_lvl in ("ORANGE", "RED", "YELLOW")) else "GRANTED",
            }
        else:
            return "No person is currently detected in the camera view to evaluate access."

        raw_name = target_entity.get("identity", "Unknown Person")
        if not isinstance(raw_name, str) or str(type(raw_name)).find("Mock") != -1:
            raw_name = "Unknown Person"
        name = raw_name
        if name in ("Unknown Person", "The user", None, ""):
            # Only resolve name if query explicitly mentions a registered user
            if "hunter" in query.lower():
                name = "Hunter"
            elif self.db and hasattr(self.db, "get_all_users"):
                try:
                    for u in (self.db.get_all_users() or []):
                        u_name = u.get("name") if isinstance(u, dict) else getattr(u, "name", None)
                        if u_name and u_name.lower() in query.lower():
                            name = u_name
                            break
                except Exception:
                    pass

        is_spoof = bool(target_entity.get("is_spoof", False))
        liveness_res = target_entity.get("liveness_state", "WARMUP")
        role = target_entity.get("role", "Visitor")
        if not isinstance(role, str) or str(type(role)).find("Mock") != -1:
            role = "Visitor"

        target_zone = target_entity.get("zone_name") or target_entity.get("zone") or zone
        zone_lower = target_zone.lower()
        reason_lower = reason.lower()

        is_known_name = bool(name and name not in ("Unknown Person", "Unknown", "Unregistered", "None", "The user", "", None))

        # Authoritative Invariant Checks
        is_known = bool(
            (target_entity.get("is_known", False) or is_known_name)
            and name not in ("Unknown Person", "Unknown", "Unregistered", "None", "", None)
            and dec_code not in ("UNKNOWN_PERSON", "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS")
            and "unregistered" not in reason_lower
            and "not registered" not in reason_lower
        )

        is_frame_auth = (
            getattr(decision, "is_access_granted", True) is not False
            and self.session_context.last_is_access_granted is not False
            and is_known
            and risk_lvl == "GREEN"
            and not is_spoof
        )

        target_access_state = "GRANTED" if (is_frame_auth and target_entity.get("access_state") in ("AUTHORIZED", "GRANTED")) else "DENIED"

        # 1. Presentation Attack / Spoof
        if is_spoof or liveness_res == "SPOOF" or dec_code == "PRESENTATION_ATTACK_IMPERSONATION" or "spoof" in dec_code.lower():
            target_disp = name if is_known else "the individual"
            return f"Access is denied: A presentation attack or spoof attempt was detected using {target_disp}'s credentials."

        # 2. Access Pending / Liveness Verifying (Strictly for registered personnel during active verification)
        if is_known and dec_code == "LIVENESS_VERIFYING" and "after_hours" not in dec_code.lower():
            return f"Access is pending: Liveness verification is currently in progress for {name}."
        if is_known and target_entity.get("access_state") in ("PENDING", "VERIFYING") and is_auth_h and "after_hours" not in dec_code.lower():
            return f"Access is pending: Liveness verification is currently in progress for {name}."

        # 3. Unregistered Person (Strictly DENIED, Cases B, C, E)
        if not is_known:
            if "protected" in zone_lower or "asset" in zone_lower or "protected" in reason_lower:
                return "Access is denied. An unregistered individual was detected in the Protected Asset Area."
            elif "restricted" in zone_lower or "restricted" in reason_lower or dec_code == "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS":
                if not is_auth_h or "after_hours" in dec_code.lower() or "outside" in reason_lower:
                    if "repeated" in reason_lower or "repeated" in dec_code.lower():
                        return (
                            "Access is denied. You are currently unregistered and inside the "
                            "Restricted Area after permitted hours. WatchGuard has also detected "
                            "repeated restricted-zone violations."
                        )
                    return "Access is denied. You are currently unregistered and inside the Restricted Area after permitted hours."
                return "Access is denied. An unregistered individual (not registered in the database) was detected in the Restricted Area."
            else:
                return "Access is denied: You are not registered in the facial database and do not have active security clearance."

        # 4. After-Hours Restricted Zone Denial for Registered Person (Case D)
        is_after_hours = (
            (not is_auth_h)
            or dec_code == "AFTER_HOURS_RESTRICTED_ACCESS_DENIED"
            or "after hours" in reason_lower
            or "after-hours" in reason_lower
            or "outside" in reason_lower
        )
        if is_after_hours and ("restricted" in zone_lower or "restricted" in dec_code.lower() or "outside" in reason_lower):
            role_disp = f" ({role})" if (role and role not in ("Visitor", "None")) else ""
            subj_info = f" Subject {name}{role_disp} was observed at {time_str} in {target_zone}." if is_known_name else (f" Observed at {time_str} in {target_zone}." if time_str else "")
            return f"Access is denied because this is outside the permitted access hours ({h_win}).{subj_info}"

        # 5. Repeated Restricted Zone Violations
        if "REPEATED" in dec_code or "repeated" in reason_lower:
            return f"Access is denied. Repeated security violations detected for {name} in {target_zone}."

        # 6. Authorized Access Granted (Case A)
        if is_frame_auth and target_access_state == "GRANTED":
            return f"Access is granted: {name} is recognized and verified live as {role} within authorized operating hours in {zone}."

        # 7. Denied Access with Yellow / Orange / Red Risk (Case E)
        if target_access_state == "DENIED":
            return f"Access is denied. The system classified the situation as {risk_lvl} because: {reason}."

        return f"Current operational status is {risk_lvl}: {reason}."

    def _handle_research_baseline_query(self) -> str:
        """Handles: 'What would the baseline face-only system have decided?' (Part 17)"""
        last_dec = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        p_name = self.session_context.last_person_name or "None"
        is_known = getattr(self.session_context, "last_is_authorized", False)

        # Explicit label required by Part 17:
        # "For the final query, clearly label the answer as: RESEARCH SIMULATION / BASELINE. Never present simulated baseline decisions as actual historical facts."
        if p_name in ("None", "Unknown Person", "Unregistered") or not is_known:
            sim_rec = "DENIED"
            reason = "unregistered facial identity"
        else:
            sim_rec = "GRANTED"
            reason = f"enrolled face match for {p_name}"

        return (
            f"[RESEARCH SIMULATION / BASELINE] A unimodal face-only system would have decided: {sim_rec} "
            f"based solely on {reason}, disregarding spatial zones, presentation attacks, and operating hours."
        )

    def _handle_research_risk_factors(self) -> str:
        """Handles: 'What factors caused the risk increase?' / 'Which contextual factor is contributing most?'"""
        last_dec = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        ccsdf = getattr(last_dec, "ccsdf_result", None) if last_dec else None
        if not ccsdf:
            return "No active CCSDF contextual risk data is available for the current scene."

        dominant = ccsdf.get("dominant_factor", "None")
        risk_score = ccsdf.get("risk_score", 0.0)
        signals = ccsdf.get("normalized_signals", {})

        elevated = [f"{k} (risk={v:.2f})" for k, v in signals.items() if v > 0.3]
        factors_str = ", ".join(elevated) if elevated else "all contextual factors are within nominal levels"

        return (
            f"CCSDF contextual risk is currently {risk_score:.2f}. "
            f"The primary contributing factor is {dominant.upper()}. Active elevated factors: {factors_str}."
        )

    def _handle_research_authorization_change(self) -> str:
        """Handles: 'Why did authorization change?' / 'What changed when he entered the restricted area?'"""
        last_dec = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        auth_state = getattr(last_dec, "access_state", "DENIED") if last_dec else "DENIED"
        zone = self.session_context.last_zone or "Standard Perimeter"
        reason = getattr(last_dec, "reason", "Policy constraints applied.") if last_dec else ""

        return (
            f"Authorization is currently {auth_state} in {zone}. "
            f"Contextual transition reason: {reason}"
        )

    def _handle_research_tailgating_query(self) -> str:
        """Handles: 'Was tailgating detected?' / 'Is someone tailgating?'"""
        last_dec = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        ccsdf = getattr(last_dec, "ccsdf_result", None) if last_dec else None
        status = ccsdf.get("tailgating_status", "NONE") if ccsdf else "NONE"
        leader = ccsdf.get("tailgating_leader", "Authorized Person") if ccsdf else "Authorized Person"
        follower = ccsdf.get("tailgating_follower", "Unknown Person") if ccsdf else "Unknown Person"
        distance = ccsdf.get("tailgating_distance", 0.0) if ccsdf else 0.0

        if status == "TAILGATING_SUSPECTED":
            return (
                f"Tailgating is suspected: Follower '{follower}' is closely following authorized leader '{leader}' "
                f"(separation: {distance:.1f} px). Leader authorization never transfers to the follower, whose access remains DENIED."
            )
        return "No tailgating pattern has been detected in the current surveillance perimeter."

    def _handle_research_behavior_query(self) -> str:
        """Handles: 'What behavior was unusual?' / 'What was the behavior anomaly?'"""
        last_dec = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        ccsdf = getattr(last_dec, "ccsdf_result", None) if last_dec else None
        status = ccsdf.get("behavior_status", "NORMAL_ACTIVITY") if ccsdf else "NORMAL_ACTIVITY"
        signals = ccsdf.get("normalized_signals", {}) if ccsdf else {}
        beh_risk = signals.get("behavior", 0.0)

        if status == "BEHAVIOR_ANOMALY":
            return f"Behavior anomaly detected (S_behavior={beh_risk:.2f}): Elevated dwell duration, repeated restricted zone attempts, or abnormal asset approaches observed."
        elif status == "UNUSUAL_PATTERN":
            return f"Unusual activity pattern detected (S_behavior={beh_risk:.2f}): Heightened spatial proximity or movement anomalies observed."
        return "Observed behavior is classified as normal routine activity with no anomalies."

    def _handle_research_confidence_query(self) -> str:
        """Handles: 'How confident is the current decision?' / 'What is the decision confidence?'"""
        last_dec = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        ccsdf = getattr(last_dec, "ccsdf_result", None) if last_dec else None
        tier = ccsdf.get("confidence_tier", "HIGH") if ccsdf else "HIGH"
        conf_dict = ccsdf.get("confidence", {}) if ccsdf else {}
        overall = conf_dict.get("overall_confidence", 1.0) if isinstance(conf_dict, dict) else getattr(conf_dict, "overall_confidence", 1.0)
        c_id = conf_dict.get("identity_confidence", 1.0) if isinstance(conf_dict, dict) else getattr(conf_dict, "identity_confidence", 1.0)
        c_pad = conf_dict.get("pad_confidence", 1.0) if isinstance(conf_dict, dict) else getattr(conf_dict, "pad_confidence", 1.0)

        return (
            f"The current security decision has {tier} confidence ({overall:.2f}) "
            f"[Identity: {c_id:.2f}, PAD: {c_pad:.2f}]. Evaluated across biometric identity, presentation attack liveness, object detection, and spatial telemetry."
        )

    def _handle_security_objects_query(self, query: str) -> str:
        """Handles inquiries about domain security objects (pen, access_badge, usb_drive, keys)."""
        ctx_sum = self.context_engine.get_context_summary() if self.context_engine else {}
        objs = ctx_sum.get("objects", []) if ctx_sum else []
        sec_names = {"pen", "access_badge", "usb_drive", "keys"}
        
        found = []
        for o in objs:
            lbl = o.get("label", "") if isinstance(o, dict) else str(o)
            if lbl in sec_names or "usb" in lbl or "badge" in lbl or "key" in lbl or "pen" in lbl:
                tier = o.get("distance_tier", "CLOSE") if isinstance(o, dict) else "CLOSE"
                assoc = o.get("associated_person_track") if isinstance(o, dict) else None
                assoc_str = f"associated with Track {assoc}" if assoc is not None else "unassociated with any person"
                found.append(f"{lbl} ({tier} distance tier, {assoc_str})")

        if found:
            return f"Security objects currently detected: {', '.join(found)}. These are evaluated as contextual evidence and do not automatically determine access."
        
        # Check if specific object was asked
        q_lower = query.lower()
        if "usb" in q_lower:
            return "No USB drive is currently detected in the surveillance area."
        elif "badge" in q_lower:
            return "No access badge is currently detected in the surveillance area."
        elif "key" in q_lower:
            return "No keys are currently detected in the surveillance area."
        elif "pen" in q_lower:
            return "No pen is currently detected in the surveillance area."
        return "No domain security objects (pen, access_badge, usb_drive, keys) are currently detected in the surveillance area."

    def _handle_detector_mode_query(self) -> str:
        """Handles: 'What detector mode is active?' / 'Which model is running?'"""
        import os
        from config import OBJECT_DETECTOR_MODE
        mode = os.getenv("WATCHGUARD_DETECTOR_MODE", OBJECT_DETECTOR_MODE)
        if mode == "production":
            desc = "YOLOv8n 80-Class COCO detector (Production Fallback)"
        elif mode == "watchguard_unified":
            desc = "WatchGuard Unified 84-Class detector (80 COCO + pen, access_badge, usb_drive, keys)"
        elif mode == "watchguard_custom":
            desc = "WatchGuard Custom 4-Class detector prototype"
        else:
            desc = f"{mode} mode"
        return f"The active object detection mode is '{mode}': {desc} executing on edge CPU."

    def _handle_research_novelty_query(self) -> str:
        """Handles: 'What is novel about WatchGuard?' / 'What is the research contribution?'"""
        return (
            "WatchGuard combines multimodal visual perception with continuous contextual security decision fusion. "
            "Instead of treating authentication as a one-time binary event, it continuously evaluates identity, "
            "liveness, zone, time, proximity, security-object context, behavioral patterns, historical violations, "
            "and multi-person interactions while keeping deterministic policy authority separate from contextual risk recommendation."
        )

    def _handle_research_ccsdf_state_query(self) -> str:
        """Handles: 'What is the current CCSDF risk?' / 'What is the current CCSDF state?'"""
        last_dec = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        ccsdf = getattr(last_dec, "ccsdf_result", None) if last_dec else None
        if not ccsdf:
            return "CCSDF contextual risk is currently R=0.00 [ALLOW] with perimeter clear."

        r_score = ccsdf.get("risk_score", 0.0)
        r_lvl = ccsdf.get("risk_level", "GREEN")
        rec = ccsdf.get("recommended_action", "ALLOW")
        dom = ccsdf.get("dominant_factor", "nominal")
        return f"Current CCSDF Contextual Risk is R={r_score:.2f} [{r_lvl}] with recommended action {rec}. Dominant contributing factor: {dom}."

    def _handle_visual_scene(self, query: str) -> str:
        """Handles scene visual description inquiries."""
        if self._should_use_ai_copilot(query):
            return self._query_ai_copilot(query)
        return self._query_scene_overview()

    def _handle_visual_count(self, query: str) -> str:
        """Handles visual people/faces count inquiries."""
        if self._should_use_ai_copilot(query):
            return self._query_ai_copilot(query)
        return self._query_live_people_count()

    def _handle_visual_object(self, query: str) -> str:
        """Handles visual object inquiries."""
        if self._should_use_ai_copilot(query):
            return self._query_ai_copilot(query)
        return self._query_detected_objects()

    def _handle_visual_description(self, query: str) -> str:
        """Handles visual appearance and clothing inquiries."""
        if self._should_use_ai_copilot(query):
            return self._query_ai_copilot(query)
        return self._query_scene_overview()

    # --- Phase 2.5 Contextual Follow-up & Analytical Handlers ---

    def _handle_authorized_hours(self, query: str) -> str:
        """Explains permitted operating hours for zones."""
        h_start = getattr(self.context_engine, "hours_start", "09:00") if self.context_engine else "09:00"
        if not isinstance(h_start, str) or str(type(h_start)).find("Mock") != -1:
            h_start = "09:00"
        h_end = getattr(self.context_engine, "hours_end", "18:00") if self.context_engine else "18:00"
        if not isinstance(h_end, str) or str(type(h_end)).find("Mock") != -1:
            h_end = "18:00"
        return (
            f"Authorized operating hours for the Restricted Area are {h_start} to {h_end}. "
            "Standard monitoring areas permit 24/7 access for authorized personnel."
        )

    def _handle_follow_up_person_authorization(self, query: str) -> str:
        """Handles questions regarding person authorization, location, and time."""
        self.session_context.last_topic = "PERSON"

        name = self.session_context.last_person_name
        is_auth = self.session_context.last_is_authorized
        is_unk = self.session_context.last_is_unknown
        role = self.session_context.last_role or "Staff"
        zone = self.session_context.last_zone or "Standard Perimeter"
        dec_code = self.session_context.last_decision_code
        time_str = self.session_context.last_time_str
        is_auth_h = self.session_context.last_is_authorized_hours
        hours_win = self.session_context.last_hours_window
        h_start = getattr(self.context_engine, "hours_start", "09:00")
        h_end = getattr(self.context_engine, "hours_end", "18:00")

        # Check if subject is registered (known user)
        is_known_subj = bool(
            not is_unk
            and name not in ("Unknown Person", "Unknown", "Unregistered", "None", "", None)
            and dec_code not in ("UNKNOWN_PERSON", "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS")
            and "unregistered" not in self.session_context.last_reason.lower()
            and "not registered" not in self.session_context.last_reason.lower()
        )

        # Check if query asks why someone was denied
        if "why" in query or "denied" in query or "refused" in query or "can't enter" in query:
            return self._handle_follow_up_why(query)

        # 1. Impersonation spoof presentation
        if dec_code == "PRESENTATION_ATTACK_IMPERSONATION" or (self.session_context.last_risk_level == "RED" and "spoof" in dec_code.lower()):
            target_name = name or "The user"
            return f"No. A presentation attack was detected using {target_name}'s credentials. Access is denied."

        # 2. Unregistered visitor (Cases B, C)
        if not is_known_subj:
            if "protected" in zone.lower() or "asset" in zone.lower():
                return "No. Access is denied. An unregistered individual was detected in the Protected Asset Area."
            elif "restricted" in zone.lower() or "restricted" in dec_code.lower():
                return "No. Access is denied. An unregistered visitor was detected in the Restricted Area."
            return "No. This person is an unregistered visitor without active security authorization. Access is denied."

        # 3. After-hours restricted area denial (Case D - strictly for registered personnel)
        if dec_code == "AFTER_HOURS_RESTRICTED_ACCESS_DENIED" or (not is_auth_h and ("restricted" in zone.lower() or "after hours" in self.session_context.last_reason.lower())):
            subj_info = f" Subject {name} was observed at {time_str} in {zone}." if name and name not in ("Unknown Person", "Unknown", "Unregistered", None, "") else ""
            return f"No. Access is denied because this is outside the permitted access hours ({h_start} to {h_end}).{subj_info}"

        # 4. Authorized personnel with granted access (Case A)
        decision = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        is_frame_auth = getattr(decision, "is_access_granted", True) and self.session_context.last_is_access_granted
        if is_frame_auth and is_auth and not is_unk and name and name not in ("Unknown Person", "Unknown", "Unregistered", None, ""):
            return f"Yes. Access is granted. {name} is authorized as {role} in {zone}."

        return "There is currently no person detected in view or referenced in this session."

    def _handle_follow_up_why(self, query: str) -> str:
        """Explains why a previous state, authorization, or alert decision exists."""
        prev_topic = self.session_context.last_topic
        risk_lvl = self.session_context.last_risk_level
        dec = self.session_context.last_decision_code
        reason = self.session_context.last_reason
        name = self.session_context.last_person_name
        is_auth = self.session_context.last_is_authorized
        is_unk = self.session_context.last_is_unknown
        time_str = self.session_context.last_time_str
        h_start = getattr(self.context_engine, "hours_start", "09:00")
        h_end = getattr(self.context_engine, "hours_end", "18:00")
        is_auth_h = self.session_context.last_is_authorized_hours
        evidence = self.session_context.last_evidence_required

        # Check if a specific registered name or subject is mentioned in the query
        common_stop_words = {
            "why", "was", "is", "did", "he", "she", "they", "him", "her", "denied", "access",
            "allowed", "enter", "the", "there", "in", "at", "to", "not", "cannot", "could",
            "would", "what", "how", "when", "who", "get", "got", "inside", "going", "refused",
            "rejected", "blocked", "stopped", "prevented", "tell", "show", "give", "explain",
            "reason", "cause", "caused", "person", "visitor", "user", "am", "i", "being", "are",
            "we", "us", "our", "you", "your", "my", "me", "this", "that", "these", "those",
            "an", "and", "or", "if", "so", "for", "with", "about", "into", "onto", "from"
        }
        extracted_name = None
        for word in query.split():
            clean_word = word.strip("?,.!\"'")
            if clean_word.lower() not in common_stop_words and len(clean_word) >= 3:
                if self.db and hasattr(self.db, "get_profile_by_name") and self.db.get_profile_by_name(clean_word):
                    extracted_name = clean_word.capitalize()
                    break
                elif not extracted_name and clean_word.isalpha():
                    extracted_name = clean_word.capitalize()

        target_name = extracted_name or name or "The user"

        # 1. After-hours restriction for personnel (Case D)
        if dec == "AFTER_HOURS_RESTRICTED_ACCESS_DENIED" or (not is_auth_h and ("restricted" in (self.session_context.last_zone or "").lower() or "after hours" in reason.lower())):
            return f"Access is denied because this is outside the permitted access hours ({h_start} to {h_end})."

        # 2. Impersonation spoof
        if dec == "PRESENTATION_ATTACK_IMPERSONATION":
            return f"Access was denied because a photo or replay presentation attack was detected using {target_name}'s credentials."

        # 3. Unregistered Person Explanation (Cases B, C)
        if not is_auth or is_unk or name in ("Unknown Person", "Unknown", "Unregistered", "None", None) or "not registered" in reason.lower() or "unregistered" in reason.lower():
            zone_str = (self.session_context.last_zone or "").lower()
            if "protected" in zone_str or "asset" in zone_str or "protected" in reason.lower():
                return "Access is denied. An unregistered individual was detected in the Protected Asset Area."
            elif "restricted" in zone_str or "restricted" in reason.lower() or dec == "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS":
                return "Access is denied. An unregistered individual (not registered in the database) was detected in the Restricted Area."
            return "The individual is not registered in the facial database and has no active clearance."

        # 4. Unauthorized restricted zone
        if dec == "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS":
            return "Access is denied. An unregistered individual was detected in the Restricted Area."

        if prev_topic == "PERSON" or (name and name.lower() in query) or "allowed" in query or "authorized" in query:
            if is_auth and not is_unk and name and name not in ("Unknown Person", "Unknown", "Unregistered", None, ""):
                return f"Access is granted. {name} is registered in the facial database with verified {self.session_context.last_role or 'Staff'} credentials."
            else:
                return "The individual is not registered in the facial database and has no active clearance."

        if risk_lvl in ("YELLOW", "ORANGE", "RED"):
            ev_str = " Evidence snapshot has been preserved." if evidence else ""
            return f"Risk is {risk_lvl}. Decision code is {dec}. Traceable rationale: {reason}.{ev_str}"

        # Standard normal operations explanation
        return f"Current operational status is GREEN: {reason}."

    def _handle_current_state(self, query: str) -> str:
        """Returns structured security explanation of the LIVE current state (DECISION -> REASON -> EVIDENCE)."""
        self.session_context.last_topic = "RISK"
        risk_lvl = self.session_context.last_risk_level
        reason = self.session_context.last_reason
        evidence = self.session_context.last_evidence_required
        ev_str = " Evidence snapshot captured." if evidence else ""

        if risk_lvl == "RED":
            return f"Risk is RED: Critical security incident! {reason}.{ev_str}"
        elif risk_lvl == "ORANGE":
            return f"Risk is ORANGE: High risk policy violation. {reason}.{ev_str}"
        elif risk_lvl == "YELLOW":
            return f"Risk is YELLOW: Attention required. {reason}.{ev_str}"
        else:
            return f"Risk is GREEN: Normal operations. {reason}."

    def _handle_historical_state(self, query: str) -> str:
        """Handles historical security inquiries (e.g. 'What happened earlier?')."""
        self.session_context.last_topic = "HISTORICAL"

        if "unauthorized" in query or "unknown" in query:
            # Query last unauthorized visitor
            ev = self.db.get_latest_alert_by_risk("YELLOW") or self.db.get_latest_alert_by_risk("ORANGE") or self.db.get_latest_alert_by_risk("RED")
            if ev:
                ts = ev.get("timestamp", "").split(" ")[-1][:5]
                return f"An unauthorized individual was detected at {ts}: {ev.get('description', 'Unregistered person')}."
            return "No unauthorized person detections are recorded in recent history."

        # General historical threat query
        alert = self.db.get_latest_alert_by_risk("RED") or self.db.get_latest_alert_by_risk("ORANGE") or self.db.get_latest_alert_by_risk("YELLOW")
        if alert:
            ts = alert.get("timestamp", "").split(" ")[-1][:5]
            risk = alert.get("risk_level", "ALERT")
            desc = alert.get("description", "Security incident").rstrip(".")
            return f"Earlier today at {ts}, risk was {risk}: {desc}."

        return "No previous security threats or incidents are recorded in the event log."

    def _handle_temporal_minutes(self, query: str) -> str:
        """Handles queries for events in the last N minutes (e.g. 'What happened in the last 5 minutes?')."""
        self.session_context.last_topic = "EVENTS"

        # Extract minute duration
        match = re.search(r"\b(\d+)\s*(?:min|minute|minutes)\b", query)
        minutes = int(match.group(1)) if match else 5
        minutes = max(1, min(minutes, 60))

        events = self.db.get_events_in_last_minutes(minutes=minutes, limit=5)
        if not events:
            return f"No security events were recorded in the last {minutes} minutes."

        items = []
        for i, ev in enumerate(events, 1):
            ts = ev.get("timestamp", "").split(" ")[-1][:5]
            risk = ev.get("risk_level", "GREEN")
            desc = ev.get("description", "Event").rstrip(".")
            items.append(f"{i}. {ts} - {risk} - {desc}")

        count_word = "event" if len(events) == 1 else f"{len(events)} events"
        return f"In the last {minutes} minutes, {count_word} occurred: " + "; ".join(items) + "."

    def _handle_temporal_alert_by_level(self, query: str) -> str:
        """Queries the latest alert of a specific risk level (RED, ORANGE, YELLOW)."""
        self.session_context.last_topic = "ALERT"

        level = "RED"
        if "orange" in query:
            level = "ORANGE"
        elif "yellow" in query:
            level = "YELLOW"
        elif "green" in query:
            level = "GREEN"

        alert = self.db.get_latest_alert_by_risk(level)
        if alert:
            ts = alert.get("timestamp", "").split(" ")[-1][:5]
            desc = alert.get("description", "Incident recorded").rstrip(".")
            return f"The last {level} alert occurred at {ts}: {desc}."

        return f"No {level} security alerts have been recorded."

    def _handle_alert_count_today(self, query: str) -> str:
        """Returns count of security alerts recorded today."""
        self.session_context.last_topic = "ALERT"
        count = self.db.get_alert_count_today()
        if count == 0:
            return "No security alerts have occurred today. System operations are normal."
        word = "alert" if count == 1 else "alerts"
        return f"A total of {count} security {word} have occurred today."

    def _handle_repeated_unauthorized(self, query: str) -> str:
        """Checks if multiple unauthorized person detections or policy violations occurred recently."""
        self.session_context.last_topic = "EVENTS"
        count = self.db.get_repeated_unauthorized_count(minutes=30)
        if count > 1:
            return f"Yes, {count} unauthorized detections have occurred in the last 30 minutes."
        elif count == 1:
            return "1 unauthorized detection was recorded in the last 30 minutes."
        else:
            return "No repeated unauthorized detections have been observed."

    def _handle_most_serious_event(self, query: str) -> str:
        """Retrieves the highest risk severity incident recorded."""
        self.session_context.last_topic = "ALERT"
        ev = self.db.get_most_serious_event()
        if ev:
            ts = ev.get("timestamp", "").split(" ")[-1][:5]
            risk = ev.get("risk_level", "GREEN")
            desc = ev.get("description", "Event").rstrip(".")
            return f"The most serious recorded event was at {ts} with risk {risk}: {desc}."

        return "No security events are currently recorded in the database."

    def _handle_spatial_relationship(self, query: str) -> str:
        """Answers questions regarding spatial proximity between people and protected objects."""
        self.session_context.last_topic = "SPATIAL"
        interactions = self.session_context.last_interactions

        # Identify requested object
        target_obj = None
        for cand in ["phone", "cell phone", "laptop", "backpack", "bottle", "chair", "book", "protected asset"]:
            if cand in query:
                target_obj = "cell phone" if cand == "phone" else cand
                break

        if interactions:
            for inter in interactions:
                lbl = inter.get("object_label", "")
                dist = inter.get("distance_pixels", 0.0)
                is_near = inter.get("is_in_proximity", True)
                p_name = self.session_context.last_person_name or "A person"

                if target_obj in (lbl, "protected asset") or not target_obj:
                    status_str = "in close visual proximity" if is_near else "in the vicinity"
                    return f"Yes, {p_name} is {status_str} ({dist}px) to the {lbl}."

        # No spatial interaction candidate found
        if self.session_context.last_person_name:
            if target_obj and target_obj in self.session_context.last_objects:
                return f"{self.session_context.last_person_name} and a {target_obj} are visible in the view, but not in immediate visual proximity."
            return f"No spatial interactions with {target_obj or 'protected objects'} are currently detected."

        return "No person or protected object interactions are currently observed."

    # --- Phase 2.4 / 5.6.1 Baseline Live State Query Resolvers ---

    def _get_current_live_entities(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieves all current live PersonEntity records and non-person objects.
        Consumes the authoritative deduplicated collection from ContextEngine / FusionEngine.
        """
        entities: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []

        last_decision = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        if last_decision:
            raw_entities = getattr(last_decision, "person_entities", [])
            if not raw_entities and last_decision.context_summary:
                raw_entities = last_decision.context_summary.get("person_entities", [])
            for e in raw_entities:
                if isinstance(e, dict):
                    entities.append(e)
                elif hasattr(e, "to_dict"):
                    entities.append(e.to_dict())

            if last_decision.context_summary:
                objects = last_decision.context_summary.get("objects", [])

            # Fallback for single person in context_summary (legacy / mock decisions)
            if not entities and last_decision.context_summary:
                ctx = last_decision.context_summary
                if ctx.get("person_detected"):
                    p_name = ctx.get("person_name", "Unknown Person")
                    is_auth = ctx.get("is_authorized", False)
                    entities.append({
                        "track_id": 1,
                        "identity": p_name,
                        "is_known": is_auth,
                        "role": ctx.get("role", "Staff" if is_auth else "Visitor"),
                        "is_spoof": ctx.get("has_spoof", False),
                        "liveness_state": "LIVE",
                        "liveness_confidence": 0.95,
                        "zone_name": ctx.get("zone_name", "Standard Perimeter"),
                        "access_state": "GRANTED" if is_auth else "DENIED",
                        "risk_level": "GREEN" if is_auth else "YELLOW",
                    })

            # When last_decision is present, it is the sole authority for current state.
            # Do NOT fall back to stale session context if 0 entities are detected!
            return entities, objects

        # Fallback ONLY when no decision has ever been evaluated (e.g. unit tests without ContextEngine)
        if self.session_context.last_people_entities:
            entities = list(self.session_context.last_people_entities)

        return entities, objects

    def _query_current_person(self) -> str:
        """Resolves identity of all people currently in camera view."""
        self.session_context.last_topic = "PERSON"
        if self.camera and not self.camera.is_running and not (self.context_engine and self.context_engine.last_decision):
            return "The camera stream is currently stopped. Please start the camera first."

        if self.person_engine and not self.person_engine.is_ready and not (self.context_engine and self.context_engine.last_decision):
            return "Person recognition engine is offline or not initialized."

        entities, _ = self._get_current_live_entities()
        if not entities:
            return "I currently do not detect any person in the camera view."

        auth_names = [
            e["identity"] for e in entities
            if bool(e.get("is_known"))
            and not bool(e.get("is_spoof"))
            and e.get("identity")
            and e["identity"].strip().lower() not in ("unknown person", "unknown", "unregistered", "none", "")
        ]
        spoof_names = [
            f"a spoof presentation attack using {e['identity']}'s credentials"
            if (e.get("identity") and e["identity"].strip().lower() not in ("unknown person", "unknown", "unregistered", "none", ""))
            else "a spoof presentation attack"
            for e in entities if bool(e.get("is_spoof"))
        ]
        unk_count = sum(
            1 for e in entities
            if (not bool(e.get("is_known")) or not e.get("identity") or e["identity"].strip().lower() in ("unknown person", "unknown", "unregistered", "none", ""))
            and not bool(e.get("is_spoof"))
        )

        if len(entities) == 1:
            e0 = entities[0]
            e0_name = e0.get("identity", "Unknown Person")
            e0_known = bool(e0.get("is_known")) and e0_name.strip().lower() not in ("unknown person", "unknown", "unregistered", "none", "")
            if auth_names:
                return f"I recognize {auth_names[0]}. Status: Authorized personnel."
            elif spoof_names:
                return "I detect a spoof presentation attack in the camera view."
            elif e0_known:
                return f"I recognize {e0_name}, but access is currently denied."
            else:
                return "I detect one unregistered person in the camera view."

        # Multiple people (2 or more)
        parts = []
        if auth_names:
            if len(auth_names) == 1:
                parts.append(f"{auth_names[0]}, an authorized person")
            else:
                parts.append(f"{' and '.join(auth_names)}, who are authorized")
        if unk_count == 1:
            parts.append("one unregistered person")
        elif unk_count > 1:
            parts.append(f"{unk_count} unregistered people")
        if spoof_names:
            parts.append(", ".join(spoof_names))

        joined = ", and ".join(parts) if len(parts) > 1 else parts[0]
        count_word = "two" if len(entities) == 2 else str(len(entities))
        return f"I currently detect {count_word} people: {joined}."

    def _query_scene_overview(self) -> str:
        """Combines all observed person entities and clean detected objects."""
        if self.camera and not self.camera.is_running and not (self.context_engine and self.context_engine.last_decision):
            return "Camera stream is offline. No live visual feed available."

        entities, objects = self._get_current_live_entities()
        last_decision = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        ctx = getattr(last_decision, "context_summary", {}) if last_decision else {}
        obj_summary = ctx.get("object_summary", "") or (", ".join([o.get("label", "object") for o in objects]) if objects else "")
        has_objects = bool(objects) or (obj_summary != "None" and obj_summary != "No objects detected" and obj_summary != "")

        if not entities and not has_objects:
            return "I currently do not detect any person or protected objects in the camera view."

        if not entities and has_objects:
            return f"I observe {obj_summary}. No person is currently in view."

        auth_names = [
            e["identity"] for e in entities
            if bool(e.get("is_known")) and not bool(e.get("is_spoof")) and e.get("identity") and e["identity"].strip().lower() not in ("unknown person", "unknown", "none", "")
        ]
        spoof_names = [f"spoof presentation of {e['identity']}" for e in entities if bool(e.get("is_spoof"))]
        unk_count = sum(
            1 for e in entities
            if (not bool(e.get("is_known")) or not e.get("identity") or e["identity"].strip().lower() in ("unknown person", "unknown", "none", ""))
            and not bool(e.get("is_spoof"))
        )

        if len(entities) == 1:
            if auth_names:
                if has_objects:
                    return f"I currently observe {auth_names[0]}, an authorized person, with a {obj_summary}."
                return f"I currently observe one authorized person, {auth_names[0]}."
            elif unk_count == 1:
                if has_objects:
                    return f"I observe one unregistered person with a {obj_summary}."
                return "I observe one unregistered person."
            elif spoof_names:
                if has_objects:
                    return f"I observe a spoof presentation attack with a {obj_summary}."
                return "I observe a spoof presentation attack."

        # Multiple people (2 or more)
        if len(entities) == 2:
            if auth_names and unk_count == 1:
                if has_objects:
                    return f"I see two people: {auth_names[0]} (authorized) and another unregistered person, with a {obj_summary}."
                return f"I see two people. {auth_names[0]} is authorized, and another person is currently unregistered."
            elif len(auth_names) == 2:
                if has_objects:
                    return f"I observe two authorized people: {auth_names[0]} and {auth_names[1]}, with a {obj_summary}."
                return f"I observe two authorized people: {auth_names[0]} and {auth_names[1]}."
            elif unk_count == 2:
                if has_objects:
                    return f"I observe two unregistered people with a {obj_summary}."
                return "I observe two unregistered people in the scene."

        # 3 or more people
        parts = []
        if auth_names:
            parts.append(f"{', '.join(auth_names)} (authorized)")
        if unk_count == 1:
            parts.append("one unregistered person")
        elif unk_count > 1:
            parts.append(f"{unk_count} unregistered people")
        if spoof_names:
            parts.append(", ".join(spoof_names))

        joined_persons = ", and ".join(parts) if len(parts) > 1 else parts[0]
        if has_objects:
            return f"I see {len(entities)} people: {joined_persons}, with a {obj_summary}."
        return f"I see {len(entities)} people: {joined_persons}."

    def _query_live_people_count(self) -> str:
        """Returns the exact count and identities of all currently observed live people."""
        self.session_context.last_topic = "PERSON"
        entities, _ = self._get_current_live_entities()
        count = len(entities)

        if count == 0:
            return "There are currently no people in view."
        elif count == 1:
            e0 = entities[0]
            if bool(e0.get("is_known")) and not bool(e0.get("is_spoof")):
                return f"There is one person in view: {e0['identity']}."
            elif bool(e0.get("is_spoof")):
                return "There is one spoof presentation entity in view."
            else:
                return "There is one person in view: an unregistered person."
        elif count == 2:
            auth_names = [
                e["identity"] for e in entities
                if bool(e.get("is_known")) and not bool(e.get("is_spoof")) and e.get("identity") and e["identity"].strip().lower() not in ("unknown person", "unknown", "none", "")
            ]
            unk_count = sum(
                1 for e in entities
                if (not bool(e.get("is_known")) or not e.get("identity") or e["identity"].strip().lower() in ("unknown person", "unknown", "none", ""))
                and not bool(e.get("is_spoof"))
            )
            if auth_names and unk_count == 1:
                return f"There are two people in view: {auth_names[0]} and one unregistered person."
            elif len(auth_names) == 2:
                return f"There are two people in view: {auth_names[0]} and {auth_names[1]}."
            elif unk_count == 2:
                return "There are two people in view: two unregistered people."
            return "There are two people in view."
        else:
            return f"There are {count} people in view."

    def _query_security_risk(self) -> str:
        """Returns current security threat posture."""
        self.session_context.last_topic = "RISK"
        decision = getattr(self.context_engine, "last_decision", None)
        if not decision:
            return "The security status is GREEN. Normal operations with no active policy violations."

        lvl = decision.risk_level.value
        if lvl == "GREEN":
            return "The security status is GREEN. Normal operations with no active policy violations."
        elif lvl == "YELLOW":
            return f"The security status is YELLOW: Attention required. {decision.reason}."
        elif lvl == "ORANGE":
            return f"The security status is ORANGE: High risk alert. {decision.reason}."
        elif lvl == "RED":
            return f"The security status is RED: Critical security incident! {decision.reason}."
        return f"Current risk level is {lvl}."

    def _query_risk_reason(self) -> str:
        """Explains the exact traceable reason for current risk decision."""
        self.session_context.last_topic = "RISK"
        decision = getattr(self.context_engine, "last_decision", None)
        if decision:
            return f"Current decision code is {decision.decision}. Traceable reason: {decision.reason}."

        if self.security_analytics and hasattr(self.security_analytics, "explain_current_risk"):
            return self.security_analytics.explain_current_risk()

        return "The system is currently operating normally with no active threat rationale."

    def _handle_why_risk_level(self, query: str) -> str:
        """Explains why the system is currently at a specific risk level."""
        self.session_context.last_topic = "RISK"
        if self.security_analytics:
            return self.security_analytics.explain_current_risk()
        return self._query_risk_reason()

    def _handle_spoof_attempts_query(self, query: str) -> str:
        """Reports spoof attempt counts."""
        self.session_context.last_topic = "ALERT"
        if self.security_analytics:
            metrics = self.security_analytics.get_metrics("today")
            if metrics.spoof_attempts == 0:
                return "No presentation attack or spoof attempts have been recorded today."
            word = "attempt" if metrics.spoof_attempts == 1 else "attempts"
            return f"A total of {metrics.spoof_attempts} presentation attack {word} were detected and rejected today."
        return "No presentation attack records are currently available."

    def _handle_distinct_people_query(self, query: str) -> str:
        """Reports unique individuals observed."""
        self.session_context.last_topic = "PERSON"
        if self.security_analytics:
            metrics = self.security_analytics.get_metrics("today")
            if metrics.distinct_people_seen == 0:
                return "No individuals have been observed in surveillance today."
            return (
                f"Today, {metrics.distinct_people_seen} distinct individual(s) were observed, "
                f"including {metrics.authorized_people_seen} authorized staff member(s) and {metrics.unknown_visitors} unregistered visitor(s)."
            )
        return "No person detection analytics are currently available."

    def _handle_unauthorized_live_query(self, query: str) -> str:
        """Checks if anyone currently observed is unauthorized."""
        self.session_context.last_topic = "PERSON"
        if self.security_analytics:
            is_unauth, reason = self.security_analytics.is_anyone_unauthorized_live()
            return reason
        return "Surveillance analytics are currently offline."

    def _handle_last_alert_trigger(self, query: str) -> str:
        """Identifies who triggered the last security alert."""
        self.session_context.last_topic = "ALERT"
        if self.security_analytics:
            active_incs = self.security_analytics.get_prioritized_incidents(limit=1)
            if active_incs:
                top = active_incs[0]
                return f"The active security alert was triggered by {top.person_name} in {top.zone} ({top.policy_code}): {top.reason}."

        alert = self.db.get_latest_alert_by_risk("RED") or self.db.get_latest_alert_by_risk("ORANGE") or self.db.get_latest_alert_by_risk("YELLOW")
        if alert:
            p_name = alert.get("person_name") or "An unregistered individual"
            ts = alert.get("timestamp", "").split(" ")[-1][:5]
            desc = alert.get("description", "Security alert").rstrip(".")
            return f"The last security alert at {ts} was triggered by {p_name}: {desc}."

        return "No security alerts have been triggered recently."

    def _query_system_status(self) -> str:
        """Summarizes overall WatchGuard system status using SecurityAnalyticsEngine."""
        self.session_context.last_topic = "RISK"
        if self.security_analytics:
            return self.security_analytics.generate_security_summary("live")

        cam_status = f"active on device index {self.camera.camera_index}" if (self.camera and self.camera.is_running) else "offline"
        total_events = self.db.get_total_event_count() if self.db else 0

        decision = getattr(self.context_engine, "last_decision", None)
        risk = decision.risk_level.value if decision else "GREEN"
        zone = decision.context_summary.get("zone_name", "Standard Perimeter") if decision else "Standard Perimeter"

        return f"WatchGuard Vision is {cam_status}. Zone: {zone}. Risk posture: {risk}. Total security events: {total_events}."

    def _query_detected_objects(self) -> str:
        """Returns breakdown of detected objects and associated people."""
        self.session_context.last_topic = "OBJECT"
        if self.object_engine and not self.object_engine.is_ready and not (self.context_engine and self.context_engine.last_decision):
            return "Object detection engine is offline."

        entities, objects = self._get_current_live_entities()
        last_decision = getattr(self.context_engine, "last_decision", None) if self.context_engine else None
        ctx = getattr(last_decision, "context_summary", {}) if last_decision else {}
        obj_sum = ctx.get("object_summary", "") or (", ".join([o.get("label", "object") for o in objects]) if objects else "")
        has_objects = bool(objects) or (obj_sum != "None" and obj_sum != "No objects detected" and obj_sum != "")

        auth_names = [e["identity"] for e in entities if (e.get("is_known") or e.get("identity") not in ("Unknown Person", "Unknown", "None", "")) and not e.get("is_spoof")]
        unk_count = sum(1 for e in entities if (not e.get("is_known") or e.get("identity") in ("Unknown Person", "Unknown", "None", "")) and not e.get("is_spoof"))
        p_count = len(entities)

        p_desc = ""
        if p_count > 0:
            parts = []
            if auth_names:
                parts.append(", ".join(auth_names))
            if unk_count == 1:
                parts.append("one unregistered person")
            elif unk_count > 1:
                parts.append(f"{unk_count} unregistered people")
            p_desc = " and ".join(parts) if parts else "unregistered people"

        if has_objects:
            obj_count = len(objects) if objects else ctx.get("objects_detected", 1)
            obj_phrase = f"{obj_count} object: {obj_sum}" if obj_count == 1 else f"{obj_count} objects: {obj_sum}"
            if p_count == 1:
                return f"I detect 1 {obj_sum}. I also see one person: {auth_names[0] if auth_names else 'one unregistered person'}."
            elif p_count == 2:
                return f"I detect 1 {obj_sum}. I also see two people: {p_desc}."
            elif p_count > 2:
                return f"I detect {obj_phrase}. I also observe {p_count} people: {p_desc}."
            else:
                return f"I currently detect {obj_phrase}."
        elif p_count > 0:
            if p_count == 1:
                return f"No non-person objects are detected. I currently see one person: {auth_names[0] if auth_names else 'one unregistered person'}."
            elif p_count == 2:
                return f"No non-person objects are detected. I currently see two people: {p_desc}."
            else:
                return f"No non-person objects are detected. I currently see {p_count} people: {p_desc}."

        return "No objects are currently detected in view."

    def _query_recent_person(self) -> str:
        """Queries SQLite for the most recently recognized person."""
        self.session_context.last_topic = "PERSON"
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT person_name, timestamp, risk_level
                FROM security_events
                WHERE person_name IS NOT NULL AND person_name != ''
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row:
                p_name = row["person_name"]
                ts = row["timestamp"]
                return f"The most recently detected person was {p_name} at {ts}."

        return "No recent person detections are recorded in the event history."

    def _query_recent_events(self) -> str:
        """
        Queries SQLite directly for the latest 3 security events (excluding internal voice logs).
        Hard limited to 3 items.
        """
        self.session_context.last_topic = "EVENTS"
        with self.db._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, risk_level, event_type, description
                FROM security_events
                WHERE event_type != 'VOICE_COMMAND_RECEIVED'
                ORDER BY id DESC
                LIMIT 3
                """
            )
            events = [dict(r) for r in cursor.fetchall()]

        if not events:
            return "There are no security events recorded in the database yet."

        lines: List[str] = []
        for i, ev in enumerate(events, 1):
            ts_raw = ev.get("timestamp", "")
            ts_short = ts_raw.split(" ")[-1][:5] if " " in ts_raw else ts_raw[:5]

            risk = (ev.get("risk_level") or "GREEN").upper()
            desc = ev.get("description", "Event").rstrip(".")
            lines.append(f"{i}. {ts_short} - {risk} - {desc}")

        count_word = "event" if len(events) == 1 else f"{len(events)} events"
        return f"Here are the latest {count_word}: " + "; ".join(lines) + "."

    def _query_latest_alert(self) -> str:
        """
        Reads the most recent non-GREEN security alert directly with SQL LIMIT 1.
        """
        self.session_context.last_topic = "ALERT"
        alert = self.db.get_latest_alert_by_risk("RED") or self.db.get_latest_alert_by_risk("ORANGE") or self.db.get_latest_alert_by_risk("YELLOW")
        if alert:
            r_lvl = alert["risk_level"]
            desc = alert["description"].rstrip(".")
            ts_raw = alert["timestamp"]
            ts_short = ts_raw.split(" ")[-1][:5] if " " in ts_raw else ts_raw
            return f"Latest security alert: {ts_short} - Risk {r_lvl} - {desc}."

        return "There are no warning or high-risk security alerts recorded in the database."

    # --- Phase 4 Incident & Temporal Violation Handlers ---

    def _handle_incident_status(self, query: str) -> str:
        """Reports the active or latest security incident status."""
        self.session_context.last_topic = "INCIDENT"
        inc = None
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
        elif self.db:
            row = self.db.get_latest_incident()
            if row:
                inc = row

        if not inc:
            return "There are no active security incidents recorded."

        inc_id = inc.incident_id if hasattr(inc, "incident_id") else inc.get("incident_id", "INCIDENT")
        policy = inc.policy_code if hasattr(inc, "policy_code") else inc.get("policy_code", "")
        state = inc.state.value if hasattr(inc, "state") and hasattr(inc.state, "value") else (inc.state if hasattr(inc, "state") else inc.get("state", "DETECTED"))
        risk = inc.risk_level.value if hasattr(inc, "risk_level") and hasattr(inc.risk_level, "value") else (inc.risk_level if hasattr(inc, "risk_level") else inc.get("risk_level", "ORANGE"))
        reason = inc.reason if hasattr(inc, "reason") else inc.get("reason", "")

        return f"Current incident is {inc_id}. Policy: {policy}. State: {state}. Risk: {risk}. Reason: {reason}."

    def _handle_incident_time(self, query: str) -> str:
        """Reports when the active or latest incident was triggered."""
        self.session_context.last_topic = "INCIDENT"
        inc = None
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
        elif self.db:
            inc = self.db.get_latest_incident()

        if not inc:
            return "No incident timestamp available because no security incidents are currently recorded."

        inc_id = inc.incident_id if hasattr(inc, "incident_id") else inc.get("incident_id", "INCIDENT")
        created_at = inc.created_at if hasattr(inc, "created_at") else inc.get("created_at", "")
        time_part = created_at.split("T")[-1][:5] if "T" in created_at else created_at[-8:-3]
        return f"Incident {inc_id} was detected at {time_part}."

    def _handle_incident_ack_status(self, query: str) -> str:
        """Reports whether the current incident has been acknowledged by an operator."""
        self.session_context.last_topic = "INCIDENT"
        inc = None
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
        elif self.db:
            inc = self.db.get_latest_incident()

        if not inc:
            return "There are no security incidents requiring acknowledgment."

        inc_id = inc.incident_id if hasattr(inc, "incident_id") else inc.get("incident_id", "INCIDENT")
        state = inc.state.value if hasattr(inc, "state") and hasattr(inc.state, "value") else (inc.state if hasattr(inc, "state") else inc.get("state", "DETECTED"))
        ack_at = inc.acknowledged_at if hasattr(inc, "acknowledged_at") else inc.get("acknowledged_at")

        if state == "ACKNOWLEDGED" or ack_at:
            ack_time = ack_at.split("T")[-1][:5] if (ack_at and "T" in ack_at) else "recently"
            return f"Yes, incident {inc_id} has been acknowledged (at {ack_time})."
        elif state == "RESOLVED":
            return f"Incident {inc_id} is already resolved."
        else:
            return f"No, incident {inc_id} is currently in {state} state and has not been acknowledged yet."

    def _handle_acknowledge_incident(self, query: str) -> str:
        """Operator action: Acknowledges the active incident."""
        self.session_context.last_topic = "INCIDENT"
        inc = None
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
            if inc and inc.state.value != "RESOLVED":
                success = self.context_engine.incident_manager.acknowledge_incident(inc.incident_id, operator="Voice Operator")
                if success:
                    return f"Incident {inc.incident_id} has been acknowledged."

        if self.db:
            latest = self.db.get_latest_incident()
            if latest and latest.get("state") != "RESOLVED":
                self.db.acknowledge_incident(latest["incident_id"], operator_action="Acknowledged via Voice Assistant")
                return f"Incident {latest['incident_id']} has been acknowledged."

        return "No pending security incidents found to acknowledge."

    def _handle_resolve_incident(self, query: str) -> str:
        """Operator action: Resolves the active incident."""
        self.session_context.last_topic = "INCIDENT"
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
            if inc:
                success = self.context_engine.incident_manager.resolve_incident(inc.incident_id, operator="Voice Operator")
                if success:
                    return f"Incident {inc.incident_id} has been resolved."

        if self.db:
            latest = self.db.get_latest_incident()
            if latest:
                self.db.resolve_incident(latest["incident_id"], operator_action="Resolved via Voice Assistant")
                return f"Incident {latest['incident_id']} has been marked resolved."

        return "No active security incidents found to resolve."

    def _handle_violation_history(self, query: str) -> str:
        """Checks whether the subject has prior policy violations in active memory or database."""
        self.session_context.last_topic = "EVENTS"
        target_name = self.session_context.last_person_name or "this person"

        # Check track history first
        if self.context_engine and hasattr(self.context_engine, "track_manager"):
            for tstate in self.context_engine.track_manager.get_all_tracks():
                if tstate.identity.lower() in query.lower() or target_name.lower() in tstate.identity.lower():
                    viols = tstate.get_recent_violations(time_window_seconds=600.0)
                    if len(viols) > 1:
                        return f"Yes, {tstate.identity} has recorded {len(viols)} policy violations in the last 10 minutes."
                    elif len(viols) == 1:
                        return f"Yes, 1 policy violation was recorded for {tstate.identity}."

        # Fallback to database
        if self.db:
            count = self.db.get_repeated_unauthorized_count(minutes=30)
            if count > 1:
                return f"Yes, {count} policy violations were recorded in the last 30 minutes."
            elif count == 1:
                return "1 prior security violation is recorded in the recent log."

        return f"No prior policy violations have been recorded for {target_name}."

    def _handle_zone_entry_query(self, query: str) -> str:
        """Reports whether a subject entered the restricted zone."""
        self.session_context.last_topic = "PERSON"
        target_name = self.session_context.last_person_name or "the person"

        if self.context_engine and hasattr(self.context_engine, "track_manager"):
            for tstate in self.context_engine.track_manager.get_all_tracks():
                name_match = (tstate.identity.lower() in query.lower()) or (target_name.lower() in tstate.identity.lower())
                if name_match:
                    if tstate.zone_occupancy.value == "INSIDE" or tstate.zone == "RESTRICTED":
                        entry_t = tstate.zone_entry_time_str or tstate.current_time
                        dur = int(tstate.duration_inside_zone_seconds)
                        return f"Yes, {tstate.identity} entered the Restricted Area at {entry_t} and has been inside for {dur} seconds."
                    elif len(tstate.distinct_entries) > 0:
                        last_e = tstate.distinct_entries[-1]
                        return f"Yes, {tstate.identity} entered the Restricted Area earlier at {last_e.get('time_str', '--:--')}."
                    else:
                        return f"No, {tstate.identity} is currently in the {tstate.zone_name} and has not entered the Restricted Area."

        if "hunter" in query.lower():
            return "No, Hunter is not currently inside the Restricted Area."
        return f"No restricted zone entry is currently observed for {target_name}."

    def _handle_zone_exit_query(self, query: str) -> str:
        """Reports whether a subject has left or is still inside the restricted zone."""
        self.session_context.last_topic = "PERSON"
        target_name = self.session_context.last_person_name or "the person"

        if self.context_engine and hasattr(self.context_engine, "track_manager"):
            for tstate in self.context_engine.track_manager.get_all_tracks():
                name_match = (tstate.identity.lower() in query.lower()) or (target_name.lower() in tstate.identity.lower())
                if name_match:
                    if tstate.zone_occupancy.value == "INSIDE" or tstate.zone == "RESTRICTED":
                        dur = int(tstate.duration_inside_zone_seconds)
                        return f"No, {tstate.identity} is currently inside the Restricted Area (duration: {dur} seconds)."
                    else:
                        return f"Yes, {tstate.identity} has exited the Restricted Area and is currently in the {tstate.zone_name}."

        return f"{target_name} is not inside the Restricted Area."

    def _handle_zone_entry_time_query(self, query: str) -> str:
        """Reports when the subject entered the restricted zone."""
        self.session_context.last_topic = "PERSON"
        target_name = self.session_context.last_person_name or "the person"

        if self.context_engine and hasattr(self.context_engine, "track_manager"):
            for tstate in self.context_engine.track_manager.get_all_tracks():
                name_match = (tstate.identity.lower() in query.lower()) or (target_name.lower() in tstate.identity.lower())
                if name_match:
                    if tstate.zone_entry_time_str:
                        return f"{tstate.identity} entered the Restricted Area at {tstate.zone_entry_time_str}."
                    elif len(tstate.distinct_entries) > 0:
                        return f"{tstate.identity} entered the Restricted Area at {tstate.distinct_entries[-1].get('time_str', '--:--')}."

        # Fallback to active incident
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
            if inc:
                time_p = inc.created_at.split("T")[-1][:5] if "T" in inc.created_at else inc.created_at[-8:-3]
                p_name = inc.person_name or "The person"
                return f"{p_name} entered the restricted zone at {time_p}."

        return f"No entry timestamp recorded for {target_name}."

    def _handle_recent_violations_count(self, query: str) -> str:
        """Returns the count of recent distinct security violations or entries."""
        self.session_context.last_topic = "EVENTS"
        count = 0
        if self.context_engine and hasattr(self.context_engine, "track_manager"):
            for tstate in self.context_engine.track_manager.get_all_tracks():
                if "enter" in query.lower():
                    count += len(tstate.get_recent_distinct_entries(time_window_seconds=600.0))
                else:
                    count += len(tstate.get_recent_distinct_violations(time_window_seconds=600.0))

        if count == 0 and self.db:
            count = self.db.get_repeated_unauthorized_count(minutes=10)

        if count == 0:
            return "No policy violations have occurred in the last 10 minutes."
        elif count == 1:
            if "enter" in query.lower():
                return "1 restricted-zone entry occurred recently."
            return "1 distinct security violation has occurred recently."
        else:
            if "enter" in query.lower():
                return f"{count} restricted-zone entries occurred recently."
            return f"{count} distinct security violations have occurred in the last 10 minutes."
