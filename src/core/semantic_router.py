"""
WatchGuard Vision - Semantic Query Router (Phase 5.6)
------------------------------------------------------
Robust semantic query parsing, intent routing, and contextual reference resolution
for the WatchGuard AI Security Copilot.

Core Capabilities:
1. Same Meaning -> Same WatchGuard Action across natural language variations.
2. Speech transcription tolerance (disfluencies, contractions, colloquialisms, minor grammar errors).
3. Entity extraction & contextual pronoun resolution ("he", "she", "they", "him", "this person").
4. Fast-path deterministic vs AI-grounded Copilot path dispatching.
5. Strict Voice + Text parity.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("WatchGuardVision.SemanticRouter")


# --- Contraction and Spoken Disfluency Normalization Maps ---

CONTRACTIONS = {
    r"\bwho's\b": "who is",
    r"\bwhat's\b": "what is",
    r"\bwhere's\b": "where is",
    r"\bhow's\b": "how is",
    r"\bthere's\b": "there is",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bcan't\b": "cannot",
    r"\bcouldn't\b": "could not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bwanna\b": "want to",
    r"\bgonna\b": "going to",
    r"\bgotta\b": "got to",
    r"\blemme\b": "let me",
}

SPEECH_DISFLUENCIES = [
    r"^\s*(hey|ok|okay|hi|hello)\s+(watchguard|assistant|guard|copilot|system)\b",
    r"^\s*(watchguard|assistant|copilot)\b",
    r"\b(please|can you|could you|tell me|explain to me|let me know|what about|uh|um|er|ah|kindly)\b",
]


@dataclass
class SemanticResolution:
    """Result of semantic query classification and context resolution."""
    raw_query: str
    normalized_query: str
    intent: Any  # VoiceIntent enum
    confidence: float
    target_person: Optional[str] = None
    target_zone: Optional[str] = None
    target_time_minutes: Optional[int] = None
    target_risk: Optional[str] = None
    is_fast_path: bool = True
    resolved_pronoun: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticQueryRouter:
    """
    Translates diverse spoken or typed natural language queries into
    deterministic WatchGuard intents and parameters.
    """

    def __init__(self, voice_intent_enum: Any):
        self.VoiceIntent = voice_intent_enum

    def normalize(self, query: str) -> str:
        """Cleans and standardizes raw speech / typed text."""
        if not query:
            return ""
        q = query.lower().strip()
        for pattern, replacement in CONTRACTIONS.items():
            q = re.sub(pattern, replacement, q)
        for pattern in SPEECH_DISFLUENCIES:
            q = re.sub(pattern, " ", q)
        # Normalize punctuation to spaces
        q = re.sub(r"[^\w\s]", " ", q)
        # Collapse whitespace
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def resolve_pronouns(self, normalized: str, session_context: Any) -> Tuple[str, Optional[str], bool]:
        """
        Replaces contextual pronouns with the active or last referenced subject name.
        """
        last_person = getattr(session_context, "last_person_name", None) if session_context else None
        if not last_person or last_person in ("None", "Unknown Person", "Unknown"):
            return normalized, None, False

        resolved_query = normalized
        was_resolved = False

        for pron in [r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bshe\b", r"\bher\b", r"\bthey\b", r"\bthem\b", r"\bthis person\b", r"\bthat person\b", r"\bthe person\b"]:
            if re.search(pron, resolved_query, re.IGNORECASE):
                resolved_query = re.sub(pron, last_person.lower(), resolved_query, flags=re.IGNORECASE)
                was_resolved = True

        return resolved_query, last_person, was_resolved

    def extract_time_minutes(self, text: str) -> Optional[int]:
        """Extracts integer minutes from time-bounded queries."""
        m = re.search(r"\b(last|past)\s+(\d+)\s*(min|minute|minutes)\b", text)
        if m:
            try:
                return int(m.group(2))
            except ValueError:
                pass
        text_num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
        for word, val in text_num_map.items():
            if f"last {word} minute" in text or f"past {word} minute" in text or f"{word} minutes" in text:
                return val
        return None

    def extract_risk_level(self, text: str) -> Optional[str]:
        """Extracts target risk color string from text."""
        for color in ["red", "orange", "yellow", "green"]:
            if re.search(rf"\b{color}\b", text):
                return color.upper()
        if "elevated" in text or "high risk" in text:
            return "ORANGE"
        if "incident" in text or "critical" in text:
            return "RED"
        return None

    def route_query(self, raw_query: str, session_context: Any = None) -> SemanticResolution:
        """
        Main semantic routing pipeline:
        Raw Query -> Normalization -> Pronoun/Context Resolution -> Semantic Intent Classifier -> Fast/AI Path Dispatch
        """
        normalized = self.normalize(raw_query)
        resolved_text, target_person, resolved_pronoun = self.resolve_pronouns(normalized, session_context)
        mins = self.extract_time_minutes(resolved_text)
        risk_color = self.extract_risk_level(resolved_text)

        # 1. Standby / Stop Listening
        if re.search(r"\b(stop listening|stop talking|be quiet|shut up|never mind|cancel|standby|go to sleep|sleep|mute)\b", resolved_text) or resolved_text == "stop":
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.STOP_LISTENING,
                confidence=1.0,
                is_fast_path=True,
            )

        # 2. Person Holding Object / Hand Queries (Phase 5.7.3)
        holding_patterns = [
            r"\bwhat\s+(object\s+)?(am\s+i|is\s+he|is\s+she|is\s+that\s+person|are\s+they|is\s+anyone)\s+(holding|carrying)\b",
            r"\bwhat\s+(object\s+)?is\s+(in\s+)?(my|his|her|their|that\s+person\s+s)\s+hand(s)?\b",
            r"\bwhat\s+is\s+in\s+(my|his|her|their)\s+hand(s)?\b",
            r"\bwhat\s+is\s+held\s+in\s+(my|his|her|their)\s+hand(s)?\b",
            r"\b(tell\s+)?what\s+(i\s+am|i\s+m|he\s+is|she\s+is|they\s+are|that\s+person\s+is|am\s+i|is\s+he|is\s+she)\s+carrying\b",
            r"\b(can\s+you\s+)?(tell\s+)?what\s+(i\s+am|he\s+is|she\s+is|they\s+are|that\s+person\s+is)\s+carrying\b",
            r"\bcan\s+you\s+tell\s+what\s+(i|he|she|that\s+person)\s+(am|is)\s+carrying\b",
            r"\bis\s+(he|she|anyone|that\s+person)\s+holding\s+(anything|an\s+object|any\s+item)\b",
            r"\bwhat\s+am\s+i\s+holding\b",
            r"\bwhat\s+is\s+he\s+holding\b",
            r"\bwhat\s+is\s+she\s+holding\b",
            r"\bwhat\s+is\s+that\s+person\s+holding\b",
            r"\bholding\s+(anything|something|an\s+object)\b",
            r"\bwhat\s+is\s+in\s+hand\b",
            r"\bwhat\s+am\s+i\s+carrying\b",
            r"\bwhat\s+carrying\b",
            r"\btell\s+what\s+i\s+am\s+carrying\b",
        ]
        for pat in holding_patterns:
            if re.search(pat, resolved_text):
                return SemanticResolution(
                    raw_query=raw_query,
                    normalized_query=resolved_text,
                    intent=self.VoiceIntent.PERSON_HOLDING_OBJECT,
                    confidence=0.98,
                    target_person=target_person,
                    is_fast_path=False,
                )

        # 3. Incident Operator Commands (Acknowledge / Resolve)
        if re.search(r"\b(acknowledge|ack|confirm)\s+(the\s+)?(incident|alert|warning)\b", resolved_text) or resolved_text in ("ack incident", "acknowledge incident", "ack alert"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.ACKNOWLEDGE_INCIDENT_COMMAND,
                confidence=0.98,
                is_fast_path=True,
            )

        if re.search(r"\b(resolve|close|clear|dismiss)\s+(the\s+)?(incident|alert|warning)\b", resolved_text) or resolved_text in ("resolve incident", "clear incident", "close alert"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.RESOLVE_INCIDENT_COMMAND,
                confidence=0.98,
                is_fast_path=True,
            )

        # 4. Zone Entry / Exit / Time Queries
        if re.search(r"\b(walked\s+into|walk\s+into|stepped\s+into|entered|enter|gone\s+into|go\s+into|go\s+inside)\b.*\b(restricted|security\s+zone|perimeter|area)\b", resolved_text) or re.search(r"\b(enter|entered)\s+(the\s+)?(restricted|security|zone|area)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.ZONE_ENTRY_QUERY,
                confidence=0.95,
                target_person=target_person,
                is_fast_path=True,
            )

        if re.search(r"\b(left|exit|exited|departed|leave|stepped\s+out\s+of)\s+(the\s+)?(restricted|security|zone|area)\b", resolved_text) or re.search(r"\b(left|exited)\s+(the\s+)?restricted\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.ZONE_EXIT_QUERY,
                confidence=0.95,
                target_person=target_person,
                is_fast_path=True,
            )

        if re.search(r"\bwhen\s+did\s+(.*)\s+(enter|go inside|step inside|walk in)\b", resolved_text) or re.search(r"\b(entry\s+time|when\s+did\s+he\s+enter|when\s+did\s+she\s+enter)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.ZONE_ENTRY_TIME_QUERY,
                confidence=0.95,
                target_person=target_person,
                is_fast_path=True,
            )

        # 5. Violation Count / History Queries
        if re.search(r"\bhas\s+(.*)\s+(violated|breached|broken)\s+(the\s+)?(policy|rule|rules|before)\b", resolved_text) or re.search(r"\b(done\s+this\s+before|prior\s+violations|violation\s+history)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.VIOLATION_HISTORY_QUERY,
                confidence=0.95,
                target_person=target_person,
                is_fast_path=True,
            )

        if re.search(r"\bhow\s+many\s+(violations|breaches|times\s+did\s+(.*)\s+enter|entries)\b", resolved_text) or re.search(r"\bhow\s+many\s+times\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.RECENT_VIOLATIONS_QUERY,
                confidence=0.92,
                target_person=target_person,
                is_fast_path=True,
            )

        # 6. Explain Access Decision / Why Denied / Why Blocked (Phase 5.7.3)
        explain_access_patterns = [
            r"\bwhy\s+(am\s+i|was\s+i|is\s+he|is\s+she|was\s+he|was\s+she|are\s+they|was\s+this\s+person|is\s+this\s+person)\s+(being\s+)?(denied|rejected|refused|blocked|stopped|prohibited|not allowed)\b",
            r"\bwhy\s+(was|is|did|has)?\s*(.*)\s*(access\s+)?(was\s+)?(denied|rejected|refused|blocked|stopped|not allowed|forbidden|prohibited)\b",
            r"\bwhy\s+(can\s+i\s+not|cannot\s+i|can\s+not\s+i|cannot\s+he|cannot\s+she|cannot\s+they|can\s+he\s+not|can\s+she\s+not)\s+(enter|go\s+inside|get\s+in|gain\s+entry|access)\b",
            r"\bwhy\s+(did|could|cannot|can)\s*(not\s+)?(.*)\s*(not\s+)?(get\s+access|gain\s+entry|enter|go\s+inside|get\s+in)\b",
            r"\bwhy\s+is\s+not\s+(my|his|her|their|our)\s+access\s+allowed\b",
            r"\bwhy\s+(is\s+not|isn't)\s+(my|his|her|their|our)\s+access\s+allowed\b",
            r"\bwhy\s+was\s+(my|his|her|their|our)\s+access\s+refused\b",
            r"\bwhat\s+is\s+preventing\s+(me|him|her|them|us|the\s+person|\w+)\s+from\s+entering\b",
            r"\bwhy\s+will\s+(not\s+you|you\s+not|you)\s+let\s+(me|him|her|them|us)\s+in\b",
            r"\bwhy\s+(won't|wont|will\s+not)\s+(you\s+)?let\s+(me|him|her|them|us)\s+in\b",
            r"\bwhy\s+(.*)\s+let\s+(me|him|her|them|us)\s+in\b",
            r"\bwhat\s+(is\s+)?(stopping|preventing|blocking)\s+(me|him|her|them|us|the\s+person|\w+)\b",
            r"\bwhy\s+(am\s+i|was\s+i)\s+(denied|rejected|refused|blocked|stopped)\b",
            r"\bwhy\s+access\s+(denied|rejected|refused|blocked)\b",
            r"\bwhy\s+(.*)\s+(cannot\s+go\s+inside|cannot\s+enter|is\s+not\s+allowed|is\s+denied|is\s+rejected|cannot\s+get\s+in)\b",
            r"\bwhat\s+is\s+the\s+reason\s+(.*)\s+(got\s+blocked|was\s+denied|was\s+refused|cannot\s+enter)\b",
            r"\bwhat\s+(stopped|prevented|blocked|caused\s+the\s+rejection\s+of)\s+(.*)\b",
            r"\bwhat\s+caused\s+(the\s+)?rejection\b",
            r"\bexplain\s+(why\s+)?(.*)\s+(was\s+)?(rejected|denied|blocked|refused)\b",
        ]
        for pat in explain_access_patterns:
            if re.search(pat, resolved_text):
                return SemanticResolution(
                    raw_query=raw_query,
                    normalized_query=resolved_text,
                    intent=self.VoiceIntent.EXPLAIN_ACCESS_DECISION,
                    confidence=0.98,
                    target_person=target_person,
                    resolved_pronoun=resolved_pronoun,
                    is_fast_path=True,
                )

        # 6.5 Phase 5.16: Research & CCSDF Queries
        if re.search(r"\b(what\s+would\s+(the\s+)?baseline|baseline\s+(face|system|decision)|how\s+would\s+(the\s+)?baseline)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_BASELINE_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        if re.search(r"\b(what\s+factors?\s+caused\s+(the\s+)?risk|which\s+(contextual\s+)?factor\s+is\s+contributing\s+most|factors?\s+caused\s+(the\s+)?risk\s+increase|contextual\s+factors?|which\s+factors?\s+caused\s+the\s+restriction)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_RISK_FACTORS", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        if re.search(r"\b(why\s+did\s+authorization\s+change|what\s+changed\s+when\s+(he|she|they|the\s+person)\s+entered|why\s+authorization\s+changed)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_AUTHORIZATION_CHANGE", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        # Phase 5.20 Novelty Feature Research Queries
        if re.search(r"\b(was\s+tailgating\s+detected|is\s+someone\s+tailgating|tailgating\s+(status|detected|suspected))\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_TAILGATING_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        if re.search(r"\b(what\s+behavior\s+was\s+unusual|what\s+was\s+the\s+behavior\s+anomaly|behavior\s+(anomaly|pattern|status)|suspicious\s+behavior)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_BEHAVIOR_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        if re.search(r"\b(how\s+confident\s+is\s+the\s+(current\s+)?decision|what\s+is\s+the\s+decision\s+confidence|how\s+certain\s+are\s+you|confidence\s+level)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_CONFIDENCE_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        # Security Objects (pen, access_badge, usb_drive, keys)
        if re.search(r"\b(what\s+security\s+objects|security\s+objects|(was|were)\s+(a\s+)?(usb|usb\s+drive|pen|badge|access\s+badge|key|keys)|who\s+is\s+associated\s+with\s+(the\s+)?(usb|badge|key|pen)|who\s+has\s+the\s+(usb|badge|key|pen))\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "SECURITY_OBJECTS_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        # Active Detector Mode Query
        if re.search(r"\b(what\s+(object\s+)?detector(\s+mode|\s+is\s+active)?|which\s+detector|active\s+detector|which\s+model\s+is\s+running|what\s+model\s+is\s+active|is\s+unified\s+detector\s+active|detector\s+mode)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "DETECTOR_MODE_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        # Research Novelty / Concept Query
        if re.search(r"\b(what\s+is\s+novel|novelty\s+of\s+watchguard|research\s+contribution|what\s+makes\s+watchguard\s+(different|unique|novel)|explain\s+the\s+research\s+architecture)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_NOVELTY_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        # CCSDF State / Risk Query
        if re.search(r"\b(what\s+is\s+(the\s+)?(current\s+)?ccsdf\s+(risk|state|score)|current\s+ccsdf\s+risk|what\s+is\s+the\s+dominant\s+risk\s+factor)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=getattr(self.VoiceIntent, "RESEARCH_CCSDF_STATE_QUERY", self.VoiceIntent.UNKNOWN),
                confidence=0.98,
                is_fast_path=True,
            )

        # 7. Follow-Up Person Authorization Check ("Is Hunter allowed to be here?", "Are they authorized?")
        if not re.search(r"\b(hours|window|times|schedule)\b", resolved_text):
            auth_check_patterns = [
                r"\b(is|are|can)\s+(.*)\s+(allowed|permitted|authorized|cleared)\s*(to\s+be\s+there|here|inside|to\s+enter|at\s+this\s+time)?\b",
                r"\bare\s+they\s+authorized\b",
                r"\bdoes\s+(.*)\s+have\s+authorization\b",
                r"\bis\s+(\w+)\s+a\s+valid\s+user\b",
            ]
            for pat in auth_check_patterns:
                if re.search(pat, resolved_text):
                    return SemanticResolution(
                        raw_query=raw_query,
                        normalized_query=resolved_text,
                        intent=self.VoiceIntent.FOLLOW_UP_PERSON_AUTHORIZATION,
                        confidence=0.96,
                        target_person=target_person,
                        resolved_pronoun=resolved_pronoun,
                        is_fast_path=True,
                    )

        # 8. Specific "Why Risk Level" ("Why is the system red?", "Why is the risk orange?", "Why red?")
        if re.search(r"\bwhy\s+(is\s+(the\s+)?(system|risk|status|dashboard|threat\s+level)\s+)?(red|orange|yellow|green|elevated|high\s+risk)\b", resolved_text) or re.search(r"\bwhy\s+(is\s+there\s+)?(elevated\s+risk|high\s+risk|red|orange|yellow)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.WHY_HIGH_RISK,
                confidence=0.96,
                target_risk=risk_color,
                is_fast_path=True,
            )

        # 9. General "Why?" Follow-Up
        if resolved_text in ("why", "why is that", "why did that happen", "what is the reason", "explain why", "why though"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.FOLLOW_UP_WHY,
                confidence=0.95,
                target_person=target_person,
                resolved_pronoun=resolved_pronoun,
                is_fast_path=True,
            )

        # 10. Spoof / Presentation Attack Queries
        if re.search(r"\bhow\s+many\s+(spoof|presentation|photo|replay)\s*(attempts|attacks)?\b", resolved_text) or re.search(r"\b(were\s+there|was\s+there)\s+any\s+(spoof|presentation)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.SPOOF_ATTEMPTS_QUERY,
                confidence=0.95,
                is_fast_path=True,
            )

        # 11. Visual Count / Live People Count vs Distinct People Today Queries
        visual_count_patterns = [
            r"\bhow\s+many\s+people\s+(are\s+)?(visible|in\s+view|in\s+the\s+frame|in\s+front|do\s+you\s+see)\b",
            r"\bhow\s+many\s+people\s+are\s+there\b",
            r"\bhow\s+many\s+faces\s+(do\s+you\s+see|are\s+visible)\b",
            r"\bhow\s+many\s+persons\s+(are\s+)?(visible|in\s+view|in\s+the\s+frame)\b",
            r"\bcount\s+(the\s+)?(people|persons|faces)\s+in\s+view\b",
        ]
        for pat in visual_count_patterns:
            if re.search(pat, resolved_text):
                return SemanticResolution(
                    raw_query=raw_query,
                    normalized_query=resolved_text,
                    intent=self.VoiceIntent.VISUAL_COUNT,
                    confidence=0.96,
                    is_fast_path=True,
                )

        if re.search(r"\bhow\s+many\s+(distinct\s+)?(people|persons|visitors|individuals|users)\s+.*(today|so\s+far|in\s+total|historically|past)\b", resolved_text) or resolved_text in ("people count today", "how many people today", "distinct people today"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.DISTINCT_PEOPLE_QUERY,
                confidence=0.95,
                is_fast_path=True,
            )

        if re.search(r"\bhow\s+many\s+(people|persons|visitors|individuals|faces)\b", resolved_text) or resolved_text in ("how many people", "people in view"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.VISUAL_COUNT,
                confidence=0.95,
                is_fast_path=True,
            )

        # 12. Unauthorized Live Presence
        if re.search(r"\bis\s+anyone\s+(unauthorized|unverified|forbidden|not allowed)\b", resolved_text) or re.search(r"\bare\s+there\s+any\s+unauthorized\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.UNAUTHORIZED_LIVE_QUERY,
                confidence=0.95,
                is_fast_path=True,
            )

        # 13. Last Alert Trigger
        if re.search(r"\bwho\s+(triggered|caused)\s+(the\s+)?(last|latest|most recent)\s+(alert|warning|incident|threat)\b", resolved_text) or resolved_text in ("who triggered the alert", "who caused the alert"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.LAST_ALERT_TRIGGER_QUERY,
                confidence=0.95,
                is_fast_path=True,
            )

        # 14. Temporal Minutes Query ("What happened in the last 5 minutes?")
        if mins is not None or re.search(r"\b(what\s+happened|what\s+occurred|what\s+happen|activity|events)\b.*\b(last|past)\s+(\d+|five|ten|fifteen|twenty)\s*(min|minute|minutes)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.TEMPORAL_EVENTS_MINUTES,
                confidence=0.95,
                target_time_minutes=mins or 5,
                is_fast_path=True,
            )

        # 15. Temporal Alert by Color ("When was the last red alert?")
        if re.search(r"\bwhen\s+was\s+(the\s+)?(last|latest)\s+(red|orange|yellow|green)\s+(alert|warning|incident)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.TEMPORAL_ALERT_BY_LEVEL,
                confidence=0.95,
                target_risk=risk_color,
                is_fast_path=True,
            )

        # 16. Alert Count Today
        if re.search(r"\bhow\s+many\s+(alerts|warnings|threats|incidents)\s+(happened|occurred|today|total)\b", resolved_text) or resolved_text in ("alert count today", "total alerts today"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.ALERT_COUNT_TODAY,
                confidence=0.95,
                is_fast_path=True,
            )

        # 17. Authorized Operating Hours
        if re.search(r"\bwhat\s+are\s+(the\s+)?(authorized|operating|permitted|working)\s+(hours|window|times|time)\b", resolved_text) or re.search(r"\boperating\s+hours\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.AUTHORIZED_HOURS_QUERY,
                confidence=0.96,
                is_fast_path=True,
            )

        # 18. Current Live Person / Who is in front
        if re.search(r"\bwho\s+(is\s+in\s+front|is\s+standing\s+there|is\s+present|is\s+here|is\s+in\s+the\s+camera|is\s+visible|do\s+you\s+see|is\s+that\s+person|is\s+that)\b", resolved_text) or re.search(r"\b(identify\s+(the\s+)?person|who\s+is\s+in\s+the\s+frame)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.WHO_IS_IN_FRONT,
                confidence=0.96,
                is_fast_path=True,
            )

        # 19. Visual Scene / What do you see / Describe the scene (Phase 5.7.3)
        visual_scene_patterns = [
            r"\bwhat\s+do\s+you\s+see\b",
            r"\bdescribe\s+(the\s+)?(scene|view|camera|room|surroundings|environment)\b",
            r"\bwhat\s+is\s+happening\s+in\s+front\s+of\s+(the\s+)?camera\b",
            r"\btell\s+me\s+what\s+you\s+observe\b",
            r"\bwhat\s+(do\s+)?you\s+observe\b",
            r"\bwhat\s+you\s+observe\b",
            r"\bwhat\s+is\s+visible\b",
            r"\bscene\s+overview\b",
            r"\bdescribe\s+what\s+you\s+see\b",
            r"\bwhat\s+do\s+you\s+observe\b",
            r"\boverview\s+of\s+(the\s+)?(view|scene|camera)\b",
            r"\boverview\s+of\s+view\b",
        ]
        for pat in visual_scene_patterns:
            if re.search(pat, resolved_text):
                return SemanticResolution(
                    raw_query=raw_query,
                    normalized_query=resolved_text,
                    intent=self.VoiceIntent.VISUAL_SCENE,
                    confidence=0.96,
                    is_fast_path=False,
                )

        # 20. Visual Object / Detected Objects (Phase 5.7.3)
        visual_object_patterns = [
            r"\bwhat\s+objects\s+(are\s+)?(visible|in\s+view|detected|present|around)\b",
            r"\bwhat\s+items\s+(are\s+)?(visible|in\s+view|detected|present)\b",
            r"\bshow\s+(detected\s+)?objects\b",
            r"\bwhat\s+are\s+you\s+detecting\b",
            r"\bwhat\s+objects\s+do\s+you\s+see\b",
            r"\bdetected\s+objects\b",
            r"\bobjects\s+detected\b",
            r"\blist\s+objects\b",
        ]
        for pat in visual_object_patterns:
            if re.search(pat, resolved_text):
                return SemanticResolution(
                    raw_query=raw_query,
                    normalized_query=resolved_text,
                    intent=self.VoiceIntent.VISUAL_OBJECT,
                    confidence=0.96,
                    is_fast_path=False,
                )

        # 21. Visual Relationship / Spatial Interaction (Phase 5.7.3)
        visual_rel_patterns = [
            r"\bis\s+anyone\s+(near|close\s+to|touching|interacting\s+with)\s+(the\s+)?(laptop|phone|cell\s+phone|backpack|asset|object)\b",
            r"\bwho\s+is\s+(near|closest\s+to|touching)\s+(the\s+)?(laptop|phone|asset)\b",
            r"\bwhat\s+is\s+(he|she|the\s+person)\s+interacting\s+with\b",
            r"\bspatial\s+relationship\b",
        ]
        for pat in visual_rel_patterns:
            if re.search(pat, resolved_text):
                return SemanticResolution(
                    raw_query=raw_query,
                    normalized_query=resolved_text,
                    intent=self.VoiceIntent.VISUAL_RELATIONSHIP,
                    confidence=0.95,
                    is_fast_path=True,
                )

        # 22. Visual Description / Appearance (Phase 5.7.3)
        visual_desc_patterns = [
            r"\bdescribe\s+(the\s+)?(person|individual|subject|visitor)\b",
            r"\bwhat\s+is\s+(the\s+person|he|she|that\s+person)\s+wearing\b",
            r"\bdescribe\s+(his|her|their)\s+(appearance|clothing|clothes)\b",
            r"\bwhat\s+color\s+is\s+(his|her|their)\s+(shirt|jacket|clothing)\b",
        ]
        for pat in visual_desc_patterns:
            if re.search(pat, resolved_text):
                return SemanticResolution(
                    raw_query=raw_query,
                    normalized_query=resolved_text,
                    intent=self.VoiceIntent.VISUAL_DESCRIPTION,
                    confidence=0.95,
                    is_fast_path=False,
                )

        # 19. Live State Query vs Historical State Query
        if re.search(r"\b(what\s+is\s+happening\s+right\s+now|is\s+there\s+a\s+threat\s+right\s+now|is\s+there\s+any\s+threat\s+right\s+now|live\s+state|what\s+is\s+happening\s+now)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.CURRENT_STATE_QUERY,
                confidence=0.96,
                is_fast_path=True,
            )

        if re.search(r"\b(what\s+happened\s+earlier|was\s+there\s+a\s+threat\s+earlier|was\s+there\s+any\s+threat\s+earlier|past\s+state|earlier\s+events)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.HISTORICAL_STATE_QUERY,
                confidence=0.96,
                is_fast_path=True,
            )

        # 20. Current Security Threat / Risk vs System Status
        if re.search(r"\b(security\s+risk|threat\s+level|risk\s+level|security\s+threat)\b", resolved_text) or re.search(r"\bis\s+there\s+a\s+(security\s+risk|threat)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.SECURITY_RISK,
                confidence=0.95,
                is_fast_path=True,
            )

        if re.search(r"\bwhat\s+is\s+(the\s+)?(current\s+)?(security\s+status|system\s+status|status)\b", resolved_text) or resolved_text in ("system status", "security status"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.CURRENT_SECURITY_STATUS,
                confidence=0.95,
                is_fast_path=True,
            )

        # 21. Recent Events
        if re.search(r"\b(show|what\s+are)\s+(the\s+)?recent\s+(security\s+)?(events|logs|activity)\b", resolved_text) or resolved_text in ("recent events", "show recent events", "show recent security events"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.RECENT_SECURITY_EVENTS,
                confidence=0.95,
                is_fast_path=True,
            )

        # 22. Latest Alert
        if re.search(r"\b(what\s+is|show)\s+(the\s+)?(latest|last|most\s+recent)\s+(alert|warning|threat)\b", resolved_text) or resolved_text in ("latest alert", "show latest alert"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.LATEST_ALERT,
                confidence=0.95,
                is_fast_path=True,
            )

        # 23. Active Incident Status
        if re.search(r"\bwhat\s+is\s+(the\s+)?(current|active|open)\s+incident\b", resolved_text) or resolved_text in ("incident status", "current incident"):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.INCIDENT_STATUS_QUERY,
                confidence=0.95,
                is_fast_path=True,
            )

        # 24. Most Serious Event
        if re.search(r"\b(most\s+serious|worst|highest\s+risk)\s+(recent\s+)?(event|incident|alert)\b", resolved_text):
            return SemanticResolution(
                raw_query=raw_query,
                normalized_query=resolved_text,
                intent=self.VoiceIntent.MOST_SERIOUS_EVENT,
                confidence=0.95,
                is_fast_path=True,
            )

        # Fallback to UNKNOWN / AI Copilot conversational path
        return SemanticResolution(
            raw_query=raw_query,
            normalized_query=resolved_text,
            intent=self.VoiceIntent.UNKNOWN,
            confidence=0.0,
            is_fast_path=False,
        )
