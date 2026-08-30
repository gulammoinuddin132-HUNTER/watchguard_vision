"""
WatchGuard Vision - Global Configuration
----------------------------------------
Defines system paths, camera settings, database locations, UI parameters,
face recognition, object detection, and context security rules.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
FACES_DIR = DATA_DIR / "faces"
DEMO_FACES_DIR = DATA_DIR / "demo_faces"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)
DEMO_FACES_DIR.mkdir(parents=True, exist_ok=True)

# Face Recognition Models & Thresholds (Phase 2.1)
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
FACE_DETECTION_SCORE_THRESHOLD = 0.50            # YuNet face detector confidence threshold
FACE_COSINE_SIMILARITY_THRESHOLD = 0.55          # SFace cosine similarity required for AUTHORIZED status
FACE_AMBIGUOUS_SIMILARITY_THRESHOLD = 0.38       # SFace cosine similarity boundary for AMBIGUOUS/UNVERIFIED status
FACE_TEMPORAL_CONFIRM_FRAMES = 2                 # Consecutive frames required to confirm AUTHORIZED status
PERSON_ALERT_COOLDOWN_SECONDS = 10.0

# Presentation Attack Detection (PAD / Liveness) Configuration (Phase 3.0)
PAD_MODELS_DIR = MODELS_DIR / "pad"
PAD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
PAD_MOBILENET_MODEL_PATH = PAD_MODELS_DIR / "mobilenetv3_pad.onnx"
PAD_EFFICIENTNET_MODEL_PATH = PAD_MODELS_DIR / "efficientnet_pad.onnx"
PAD_LBP_SVM_MODEL_PATH = PAD_MODELS_DIR / "lbp_svm_pad.joblib"
PAD_PROVISIONAL_LIVENESS_THRESHOLD = 0.80        # Provisional liveness score threshold; calibrated via evaluation
PAD_PROVISIONAL_SPOOF_THRESHOLD = 0.50           # Below this score is classified as presentation attack
PAD_BLINK_EAR_THRESHOLD = 0.22                  # Eye Aspect Ratio threshold for eye closure detection
PAD_VERIFICATION_TIMEOUT_SECONDS = 5.0          # Max verification window before timing out to UNVERIFIED / ACCESS DENIED
PAD_DEFAULT_MODEL = "Blink-EAR"                 # Default active PAD model ("Blink-EAR", "LBP-SVM", "MobileNetV3-Small", "EfficientNet-B0")

# Object Detection Models & Thresholds (Phase 2.2 & 5.19D)
YOLOV8N_MODEL_PATH = MODELS_DIR / "yolov8n.onnx"
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.45
OBJECT_DETECTION_NMS_THRESHOLD = 0.45
OBJECT_ALERT_COOLDOWN_SECONDS = 15.0

# Phase 5.19D & 5.21 Unified WatchGuard Object Detection Configuration
OBJECT_DETECTOR_MODE = os.getenv("OBJECT_DETECTOR_MODE", "production")  # "production" | "watchguard_custom" | "watchguard_unified"
WATCHGUARD_CUSTOM_MODEL_PATH = MODELS_DIR / "yolo_watchguard_assets_v3.onnx"
WATCHGUARD_CUSTOM_CONF_THRESHOLD = float(os.getenv("WATCHGUARD_CUSTOM_CONF_THRESHOLD", "0.35"))
WATCHGUARD_CUSTOM_NMS_THRESHOLD = float(os.getenv("WATCHGUARD_CUSTOM_NMS_THRESHOLD", "0.45"))
WATCHGUARD_CUSTOM_CLASSES = ["person", "laptop", "cell_phone", "mouse"]

# Phase 5.21 84-Class Unified Model Settings
WATCHGUARD_UNIFIED_MODEL_PATH = MODELS_DIR / "yolo_watchguard_unified_84.onnx"
WATCHGUARD_UNIFIED_CONF_THRESHOLD = float(os.getenv("WATCHGUARD_UNIFIED_CONF_THRESHOLD", "0.35"))
WATCHGUARD_UNIFIED_NMS_THRESHOLD = float(os.getenv("WATCHGUARD_UNIFIED_NMS_THRESHOLD", "0.45"))

# COCO 80 Class Labels for YOLOv8
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Phase 5.21 Authoritative 84-Class Unified Taxonomy (80 COCO + 4 WatchGuard Security Assets)
WATCHGUARD_UNIFIED_CLASSES = COCO_CLASSES + [
    "pen",           # 80
    "access_badge",  # 81
    "usb_drive",     # 82
    "keys",          # 83
]

# Security Asset Role & Classification Mapping (Phase 5.19D & 5.21)
SECURITY_OBJECT_CATEGORIES: Dict[str, str] = {
    "laptop": "protected",
    "cell_phone": "security-relevant",
    "cell phone": "security-relevant",
    "mouse": "context-relevant",
    "person": "entity",
    "pen": "context-relevant",
    "access_badge": "credential",
    "usb_drive": "high-risk-asset",
    "keys": "physical-access",
}

# Context & Security Intelligence Configuration (Phase 2.3)
PROTECTED_OBJECT_CLASSES: List[str] = ["laptop", "cell phone", "cell_phone", "backpack", "book"]

# Virtual Surveillance Zones (Normalized bounding boxes [x, y, w, h] from 0.0 to 1.0)
DEFAULT_ZONES: Dict[str, Dict[str, Any]] = {
    "NORMAL": {
        "name": "Standard Perimeter",
        "rect_norm": (0.0, 0.0, 1.0, 1.0),
        "zone_type": "NORMAL",
    },
    "RESTRICTED": {
        "name": "Restricted Area",
        "rect_norm": (0.60, 0.0, 0.40, 1.0),  # Right 40% of camera view
        "zone_type": "RESTRICTED",
    },
    "PROTECTED_ASSET_AREA": {
        "name": "Protected Asset Area",
        "rect_norm": (0.15, 0.15, 0.70, 0.70),  # Center region
        "zone_type": "PROTECTED_ASSET_AREA",
    },
}

# Operational Time Windows (24-hour format)
AUTHORIZED_HOURS_START = "09:00"
AUTHORIZED_HOURS_END = "18:00"

# Spatial Proximity Thresholds (in pixels)
PROXIMITY_PIXEL_THRESHOLD = 150
CONTEXT_ALERT_COOLDOWN_SECONDS = 15.0

# Phase 5.20 Novelty Feature Research Configuration
# 1. Tailgating & Multi-Person Access Anomaly
TAILGATING_TIME_WINDOW_SECONDS = float(os.getenv("TAILGATING_TIME_WINDOW_SECONDS", "3.0"))
TAILGATING_SPATIAL_THRESHOLD_PIXELS = float(os.getenv("TAILGATING_SPATIAL_THRESHOLD_PIXELS", "160.0"))
TAILGATING_CONFIRMATION_FRAMES = int(os.getenv("TAILGATING_CONFIRMATION_FRAMES", "2"))
TAILGATING_ALERT_COOLDOWN_SECONDS = float(os.getenv("TAILGATING_ALERT_COOLDOWN_SECONDS", "10.0"))

# 2. Behavioral & Temporal Anomaly Engine
BEHAVIOR_DWELL_THRESHOLD_SECONDS = float(os.getenv("BEHAVIOR_DWELL_THRESHOLD_SECONDS", "15.0"))
BEHAVIOR_APPROACH_REPETITION_THRESHOLD = int(os.getenv("BEHAVIOR_APPROACH_REPETITION_THRESHOLD", "3"))
BEHAVIOR_RESTRICTED_RETRY_THRESHOLD = int(os.getenv("BEHAVIOR_RESTRICTED_RETRY_THRESHOLD", "2"))

# 3. Uncertainty-Aware Decision Layer
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.75"))
CONFIDENCE_LOW_THRESHOLD = float(os.getenv("CONFIDENCE_LOW_THRESHOLD", "0.45"))
CCSDF_WEIGHT_BEHAVIOR = float(os.getenv("CCSDF_WEIGHT_BEHAVIOR", "0.10"))

# Database Configuration
DATABASE_PATH = DATA_DIR / "watchguard.db"

# Camera Settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
TARGET_FPS = 30
CAMERA_WARMUP_FRAMES = 5

# Event & Logging
LOG_FILE_PATH = LOGS_DIR / "watchguard.log"
CONSOLE_LOG_LEVEL = os.getenv("CONSOLE_LOG_LEVEL", "INFO")
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "INFO")
MAX_RECENT_EVENTS_DISPLAY = 100
MAX_DATABASE_NORMAL_EVENTS = int(os.getenv("MAX_DATABASE_NORMAL_EVENTS", "500"))
EVENT_RETENTION_CHECK_INTERVAL = int(os.getenv("EVENT_RETENTION_CHECK_INTERVAL", "25"))
LOG_ROTATION_MAX_BYTES = int(os.getenv("LOG_ROTATION_MAX_BYTES", str(5 * 1024 * 1024)))
LOG_ROTATION_BACKUP_COUNT = int(os.getenv("LOG_ROTATION_BACKUP_COUNT", "3"))

# Risk Thresholds
RISK_LEVELS = {
    "GREEN": "Normal",
    "YELLOW": "Attention",
    "ORANGE": "High Risk",
    "RED": "Security Incident"
}

# UI Styling
APP_TITLE = "WatchGuard Vision - AI Visual Security Assistant"
APP_VERSION = "1.0.0-prototype"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 880
VIDEO_DISPLAY_WIDTH = 640
VIDEO_DISPLAY_HEIGHT = 480

# Phase 5.5 / 5.7 AI Security Copilot Configuration
COPILOT_ENABLED = os.getenv("COPILOT_ENABLED", "true").lower() in ("true", "1", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
COPILOT_MODEL_NAME = os.getenv("COPILOT_MODEL_NAME", "gemini-2.5-flash")
COPILOT_TIMEOUT_SECONDS = float(os.getenv("COPILOT_TIMEOUT_SECONDS", "3.0"))
COPILOT_LOCAL_ENDPOINT = os.getenv("COPILOT_LOCAL_ENDPOINT", "http://localhost:11434/api/generate")
COPILOT_LOCAL_MODEL = os.getenv("COPILOT_LOCAL_MODEL", "llama3.2:3b")
