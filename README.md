# WatchGuard Vision 👁️
**Context-Aware AI Visual Security Assistant**
Gulam Moinuddin
B.Tech CSE (Business Systems)

AI • Computer Vision • Software Engineering

Featured Projects
├── WatchGuard Vision
├── CareSync
├── Migration Pressure Index
└── Hidden Genius

Tech
Python • C++ • Java • SQL • OpenCV • YOLO • ONNX

WatchGuard Vision is an intelligent security operations platform and desktop visual assistant. It evolves standard computer vision security into a multi-modal context engine capable of analyzing person identity, object presence, location/surveillance zones, operational time windows, and configurable security policy rules in real-time.

---

## 🌟 System Architecture

```
watchguard_vision/
├── data/                       # Local SQLite database, models, logs, and evidence snapshots
│   ├── watchguard.db           # SQLite database (7 core tables + indexes)
│   ├── models/                 # Deep learning ONNX models
│   │   ├── face_detection_yunet_2023mar.onnx    # YuNet Face Detector
│   │   ├── face_recognition_sface_2021dec.onnx   # SFace Face Recognizer
│   │   └── yolov8n.onnx                         # YOLOv8 Nano Object Detector (80 classes)
│   ├── faces/                  # Registered face crop profiles
│   ├── demo_faces/             # Sample demo portraits (Officer Alice, Unknown Visitor)
│   ├── evidence/               # Captured security incident snapshots
│   └── logs/                   # System runtime logs (watchguard.log)
├── src/
│   ├── camera/                 # Video stream management
│   │   └── camera_manager.py   # Multi-index physical discovery, 16:9 widescreen (1280x720), mirror correction & simulation
│   ├── database/               # Data persistence layer
│   │   ├── db_manager.py       # SQLite connection manager, face & object CRUD
│   │   └── models.py           # Structured dataclasses for records
│   ├── events/                 # Event dispatch & security logging
│   │   ├── event_types.py      # EventType and RiskLevel (GREEN, YELLOW, ORANGE, RED)
│   │   └── event_logger.py     # Dual-target logger (SQLite + File) with UI callbacks
│   ├── core/                   # Modular AI component engines
│   │   ├── base_engine.py      # Abstract base vision engine contract
│   │   ├── person_recognition.py # OpenCV YuNet Face Detection + SFace Feature Recognition (Phase 2.1)
│   │   ├── object_detection.py # OpenCV DNN + YOLOv8 Nano Object Detection (Phase 2.2)
│   │   ├── context_engine.py   # Explainable Context & Security Intelligence Engine (Phase 2.3)
│   │   ├── document_intelligence.py # Document OCR interface (Phase 2.4)
│   │   ├── response_engine.py  # Security alerts, notifications & evidence capture
│   │   └── voice_assistant.py  # Voice command query interface
│   └── ui/                     # Desktop GUI
│       ├── theme.py            # High-tech SOC dark theme palette and typography
│       ├── widgets.py          # Custom Card frames, RiskBanner, and Telemetry rows
│       └── dashboard.py        # Desktop Dashboard with live video canvas, context panel & event feed
├── tests/                      # Automated unit test suite
│   ├── test_database.py        # Database schema and CRUD tests
│   ├── test_event_logger.py    # Event logger and subscriber tests
│   ├── test_camera.py          # Camera lifecycle, horizontal flip, resolution scaling & fallback tests
│   ├── test_person_recognition.py # Person recognition, registration, throttling & demo tests
│   ├── test_object_detection.py # YOLOv8n object detection, NMS, throttling & DB persistence tests
│   ├── test_context_engine.py  # Security rules (A-F), spatial proximity, time & debouncing tests
│   ├── test_webcam_diagnostic.py # Hardware webcam diagnostics
│   └── verify_setup.py         # Subsystem integration health check
├── config.py                   # Centralized configuration constants
├── main.py                     # Application entry point
└── requirements.txt            # Python dependencies
```

---

## 🧠 Context & Security Intelligence Pipeline (Phase 2.3)

WatchGuard Vision correlates visual observations within the surveillance frame against deterministic, explainable security rules:

$$\text{Person Observation} + \text{Object Observation} + \text{Zone Context} + \text{Time Context} \xrightarrow{\text{Rule Engine}} \text{SecurityDecision}$$

### Security Rules Matrix

| Rule ID | Conditions | Threat Level | Decision Code | Traceable Reason |
| :--- | :--- | :--- | :--- | :--- |
| **RULE A** | Known/Authorized Person + Normal Activity + Standard Zone | **`GREEN`** | `NORMAL_ACTIVITY` | Authorized personnel verified. Surveillance perimeter normal. |
| **RULE B** | Unknown Person detected in standard area | **`YELLOW`** | `UNKNOWN_PERSON` | Unregistered individual detected in standard perimeter. |
| **RULE C** | Unknown Person + Protected Object in visual proximity | **`RED`** | `POTENTIAL_UNAUTHORIZED_INTERACTION` | Potential unauthorized interaction: Unknown person near protected asset (e.g. LAPTOP). |
| **RULE D** | Authorized Person + Protected Object outside authorized hours | **`ORANGE`** | `SUSPICIOUS_AUTHORIZED_USER_ACTIVITY` | After-hours asset activity: Authorized user interacting with asset outside operational hours. |
| **RULE E** | Person detected inside configured Restricted Zone | **`ORANGE`** | `RESTRICTED_ZONE_ACTIVITY` | Restricted zone activity: Personnel detected within restricted perimeter. |
| **RULE F** | Unknown Person inside Restricted Zone | **`RED`** | `UNAUTHORIZED_RESTRICTED_ZONE_ACCESS` | Unauthorized access: Unregistered individual detected inside restricted security zone. |

### Configurable Parameters (`config.py`)
- **Protected Objects**: `["laptop", "cell phone", "backpack", "book"]`
- **Surveillance Zones**: `NORMAL` (full view), `RESTRICTED` (right 40% of frame), `PROTECTED_ASSET_AREA` (center 70%)
- **Authorized Hours**: `09:00` to `18:00` (configurable 24h format)
- **Spatial Proximity Threshold**: `150` pixels (edge-to-edge distance between bounding boxes)
- **Alert Debounce Cooldown**: `15.0` seconds
- **Camera Configuration & Environment Variables**:
  - `CAMERA_FRAME_WIDTH` (default: `1280` — requested horizontal resolution)
  - `CAMERA_FRAME_HEIGHT` (default: `720` — requested vertical resolution)
  - `CAMERA_HORIZONTAL_FLIP` (default: `true` — flips physical webcam frames horizontally for true physical orientation)
  - *Note:* These environment variables are evaluated when the Python configuration module loads at startup. Changes require setting the environment variables before launching or restarting the application; changing them does not automatically reconfigure an already running application.

---

## 🚦 Risk Classification Scale

| Level | Status | Visual Color | Description |
| :--- | :--- | :--- | :--- |
| **`GREEN`** | **NORMAL** | Emerald Green | Authorized personnel verified, standard objects, no policy breaches. |
| **`YELLOW`** | **ATTENTION** | Amber | Unknown/unregistered individual detected in surveillance area. |
| **`ORANGE`** | **HIGH RISK** | Bright Orange | After-hours asset activity or authorized entry into restricted zone. |
| **`RED`** | **SECURITY INCIDENT** | Crimson Red | Critical security violation: Unknown person near protected asset or in restricted zone. |

---

## 🚀 Getting Started

### 1. Launch Application
```bash
# Navigate to project directory
cd watchguard_vision

# Run WatchGuard Vision
.venv\Scripts\python.exe main.py
```

### 2. Run Automated Tests
```bash
# Run camera and flip-validation tests
.venv\Scripts\python.exe -m unittest tests/test_camera.py

# Run all automated tests
.venv\Scripts\python.exe -m unittest discover tests
```

---

## 🎯 Testing Context Scenarios with the Real Webcam

1. Run `.venv\Scripts\python.exe main.py`.
2. Click **`▶ START CAMERA`** on the bottom control bar (connects to Device `#0` in 1280×720 widescreen with physical mirror correction).
3. **Test Scenario 1 (Normal Operations - Rule A)**:
   - When the camera sees your registered face, the system reports `Risk: GREEN`, `Decision: NORMAL_ACTIVITY`, and shows Emerald Green badges.
4. **Test Scenario 2 (Unknown Person Near Protected Asset - Rule C)**:
   - Hold up a **laptop** or **cell phone** while unauthenticated (or choose synthetic "Unknown Visitor"):
   - The system detects the spatial proximity vector between the person and protected asset.
   - Escalates immediately to **`RED [SECURITY INCIDENT]`**, logs `POTENTIAL_UNAUTHORIZED_INTERACTION`, and preserves an evidence snapshot in `data/evidence/`.
5. **Test Scenario 3 (After-Hours Authorized User - Rule D)**:
   - In the **Scenario:** dropdown at the bottom, select **`Simulate After-Hours (22:30)`**.
   - When an authorized person interacts with a protected asset (laptop/phone), the system flags **`ORANGE [HIGH RISK]`** with reason `"After-hours asset activity: Authorized user interacting with LAPTOP outside authorized hours (22:30)"`.
6. **Test Scenario 4 (Restricted Zone - Rule E / F)**:
   - Select **`Force Restricted Zone`** from the Scenario dropdown, or walk physically toward the right side of the camera's field of view:
   - The live feed matches real-world physical orientation without horizontal inversion, correctly aligning your physical position with the restricted perimeter.
   - Instantly shows the red tint on the restricted perimeter and triggers **`ORANGE`** or **`RED`** depending on authorization.
