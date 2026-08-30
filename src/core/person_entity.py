"""
WatchGuard Vision - Multi-Modal Person Entity Association (Phase 4.2)
---------------------------------------------------------------------
Fuses face-recognition tracks (YuNet/SFace) with generic object detector bounding boxes
(YOLOv8 'person') into unified PersonEntity models.

Key Capabilities:
- Spatial containment (face center within upper body bounds)
- Intersection-over-Face (IoF) and IoU matching
- Deduplication: Generic YOLO 'person' boxes corresponding to a tracked face do NOT create duplicate persons
- Accurate multi-person counts (1 face + 1 overlapping person = 1 entity; 1 face + 1 separate person = 2 entities)
- Presentation attack isolation (real person + phone photo of person = 2 distinct entities)
- Clean non-person object association (cell phone, laptop, etc.)
"""

from dataclasses import dataclass, field, asdict
import math
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.events.event_types import RiskLevel

logger = logging.getLogger("WatchGuardVision.PersonEntity")


@dataclass
class PersonEntity:
    """Unified multi-modal representation of a person in view."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    body_bbox: Optional[Tuple[int, int, int, int]] = None
    identity: str = "Unknown Person"
    identity_confidence: float = 0.0
    role: str = "Visitor"
    is_known: bool = False
    liveness_state: str = "WARMUP"
    liveness_confidence: float = 0.0
    is_spoof: bool = False
    attack_type: Optional[str] = None
    person_object_detected: bool = False
    associated_objects: List[Dict[str, Any]] = field(default_factory=list)
    zone: str = "NORMAL"
    zone_name: str = "Standard Perimeter"
    access_state: str = "PENDING"
    risk_level: RiskLevel = RiskLevel.GREEN
    decision_code: str = "NORMAL_ACTIVITY"
    reason: str = "Initializing person entity."

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level
        return d


def compute_box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Returns center point (cx, cy) of a bounding box (x, y, w, h)."""
    x, y, w, h = box
    return float(x + w * 0.5), float(y + h * 0.5)


def compute_intersection_area(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Computes pixel intersection area between two boxes (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def compute_iof(face_box: Tuple[int, int, int, int], body_box: Tuple[int, int, int, int]) -> float:
    """
    Computes Intersection over Face Area (IoF).
    High IoF (>0.50) indicates the face is almost entirely contained within the body box.
    """
    _, _, fw, fh = face_box
    face_area = float(fw * fh)
    if face_area <= 0:
        return 0.0
    inter_area = compute_intersection_area(face_box, body_box)
    return inter_area / face_area


def compute_containment_score(face_box: Tuple[int, int, int, int], body_box: Tuple[int, int, int, int]) -> float:
    """
    Evaluates whether face center is in the upper portion of the body box.
    Returns a normalized spatial confidence score between 0.0 and 1.0.
    """
    fcx, fcy = compute_box_center(face_box)
    bx, by, bw, bh = body_box

    # Face center horizontally inside body
    if not (bx <= fcx <= bx + bw):
        return 0.0

    # Face center vertically inside upper 65% of body box
    upper_limit = by + bh * 0.65
    if not (by <= fcy <= upper_limit):
        return 0.0

    iof = compute_iof(face_box, body_box)
    return max(0.5, iof)


class PersonEntityFusionEngine:
    """
    Fuses face-recognition tracks and generic YOLO person detections
    into unified PersonEntity instances.
    """

    def __init__(self, proximity_threshold_pixels: float = 120.0):
        self.proximity_threshold = proximity_threshold_pixels

    def fuse(
        self,
        people: List[Dict[str, Any]],
        detected_objects: List[Dict[str, Any]],
        zones_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[PersonEntity], List[Dict[str, Any]], str]:
        """
        Fuses people (face tracks) with object detections (YOLOv8).

        Returns:
            (person_entities, non_person_objects, object_summary_string)
        """
        # 1. Separate person objects from non-person objects
        person_objects: List[Dict[str, Any]] = []
        non_person_objects: List[Dict[str, Any]] = []

        for obj in detected_objects:
            lbl = obj.get("label", "").lower()
            if lbl == "person":
                person_objects.append(obj)
            else:
                non_person_objects.append(obj)

        matched_person_indices = set()
        entities: List[PersonEntity] = []

        # 2. Match Face Tracks to Candidate Body Boxes
        for p in people:
            tid = p.get("track_id", 1)
            p_name = p.get("name", "Unknown Person")
            p_sim = float(p.get("similarity", 0.0))
            p_conf = float(p.get("confidence", 0.0))
            p_known = bool(p.get("is_known", False) or (p_name not in ("Unknown Person", "Unknown", "None", "") and p.get("name") is not None))
            p_role = p.get("role", "Staff" if p_known else "Visitor")
            p_spoof = bool(p.get("is_spoof", False))
            p_atk = p.get("attack_type", "PHOTO_REPLAY_ATTACK" if p_spoof else None)
            p_live_conf = float(p.get("liveness_confidence", 0.90 if not p_spoof else 0.10))
            p_live_state = "SPOOF" if p_spoof else ("WARMUP" if p.get("is_verifying") else "LIVE")

            f_bbox = p.get("bbox", (0, 0, 0, 0))

            best_body_idx = None
            best_score = 0.0

            # Only match bona-fide faces or non-phone-contained faces to YOLO body boxes
            if not p_spoof:
                for idx, p_obj in enumerate(person_objects):
                    if idx in matched_person_indices:
                        continue
                    b_bbox = p_obj.get("bbox", (0, 0, 0, 0))
                    score = compute_containment_score(f_bbox, b_bbox)
                    if score > 0.40 and score > best_score:
                        best_score = score
                        best_body_idx = idx

            if best_body_idx is not None:
                matched_person_indices.add(best_body_idx)
                b_bbox = person_objects[best_body_idx].get("bbox")
                fused_bbox = b_bbox  # Full body as primary bbox
                has_person_obj = True
            else:
                fused_bbox = f_bbox
                b_bbox = None
                has_person_obj = False

            entity = PersonEntity(
                track_id=tid,
                bbox=fused_bbox,
                face_bbox=f_bbox,
                body_bbox=b_bbox,
                identity=p_name,
                identity_confidence=p_sim if p_known else p_conf,
                role=p_role,
                is_known=p_known,
                liveness_state=p_live_state,
                liveness_confidence=p_live_conf,
                is_spoof=p_spoof,
                attack_type=p_atk,
                person_object_detected=has_person_obj,
                zone=p.get("zone", "NORMAL"),
                zone_name=p.get("zone_name", "Standard Perimeter"),
                access_state="GRANTED" if (p_known and not p_spoof) else ("SPOOF_REJECTED" if p_spoof else "DENIED"),
                risk_level=RiskLevel.RED if p_spoof else (RiskLevel.GREEN if p_known else RiskLevel.YELLOW),
            )
            entities.append(entity)

        # 3. Process Unmatched Generic Person Object Detections (e.g. person with back turned)
        unmatched_seq = 100
        for idx, p_obj in enumerate(person_objects):
            if idx in matched_person_indices:
                continue

            b_bbox = p_obj.get("bbox", (0, 0, 0, 0))
            unmatched_entity = PersonEntity(
                track_id=unmatched_seq,
                bbox=b_bbox,
                face_bbox=None,
                body_bbox=b_bbox,
                identity="Unknown Person",
                identity_confidence=float(p_obj.get("confidence", 0.0)),
                role="Visitor",
                is_known=False,
                liveness_state="UNVERIFIED",
                liveness_confidence=0.0,
                is_spoof=False,
                person_object_detected=True,
                zone="NORMAL",
                zone_name="Standard Perimeter",
                access_state="DENIED",
                risk_level=RiskLevel.YELLOW,
                decision_code="UNREGISTERED_PERSON",
                reason="Person detected without verified facial credentials.",
            )
            entities.append(unmatched_entity)
            unmatched_seq += 1

        # 4. Associate Non-Person Objects (cell phone, laptop, usb, etc.) with Entities
        for obj in non_person_objects:
            obj_bbox = obj.get("bbox", (0, 0, 0, 0))
            best_dist = float("inf")
            best_entity: Optional[PersonEntity] = None

            for entity in entities:
                dist = self._compute_box_distance(entity.bbox, obj_bbox)
                if dist <= self.proximity_threshold and dist < best_dist:
                    best_dist = dist
                    best_entity = entity

            if best_entity is not None:
                assoc_conf = max(0.0, 1.0 - (best_dist / max(self.proximity_threshold, 1.0)))
                obj["associated_person_track"] = best_entity.track_id
                obj["pixel_distance_to_person"] = round(best_dist, 1)
                obj["association_confidence"] = round(assoc_conf, 3)

                best_entity.associated_objects.append({
                    "object_label": obj.get("label", "object"),
                    "confidence": obj.get("confidence", 0.0),
                    "distance_pixels": round(best_dist, 1),
                    "association_confidence": round(assoc_conf, 3),
                    "bbox": obj_bbox,
                    "is_confirmed": obj.get("is_confirmed", True),
                    "security_category": obj.get("security_category", "context-relevant"),
                })
            else:
                obj["associated_person_track"] = None
                obj["pixel_distance_to_person"] = -1.0
                obj["association_confidence"] = 0.0

        # 5. Build Clean Semantic Object Summary (excluding absorbed person objects)
        if non_person_objects:
            obj_counts: Dict[str, int] = {}
            for o in non_person_objects:
                lbl = o.get("label", "object")
                obj_counts[lbl] = obj_counts.get(lbl, 0) + 1
            summary_parts = [f"{count} {lbl}" if count > 1 else lbl for lbl, count in obj_counts.items()]
            object_summary_str = ", ".join(summary_parts)
        else:
            object_summary_str = "None"

        return entities, non_person_objects, object_summary_str

    def _compute_box_distance(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Computes minimum Euclidean distance between centers or edges of two boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        acx, acy = ax + aw * 0.5, ay + ah * 0.5
        bcx, bcy = bx + bw * 0.5, by + bh * 0.5

        return math.hypot(acx - bcx, acy - bcy)
