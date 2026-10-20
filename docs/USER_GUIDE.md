# WatchGuard Vision - User Guide

## Overview

WatchGuard Vision is an AI-powered desktop security assistant that uses computer vision to monitor people and objects in real-time, classify security risk levels, and log events to a SQLite database. It combines face recognition (YuNet + SFace), object detection (YOLOv8n), presentation attack detection (liveness/PAD), and a deterministic context security rule engine.

This guide assumes you are running the application from the project root using the bundled virtual environment.

---

## 1. Running the Project

### Prerequisites
- Python 3 with the dependencies listed in `requirements.txt` installed in `.venv`.
- The ONNX model files under `data/models/` (YuNet face detector, SFace face recognizer, YOLOv8n object detector).
- A working webcam, OR the ability to run in synthetic simulation mode when no physical camera is available.

### Launch command
From the project directory:

```bash
.venv\Scripts\python.exe main.py
```

On startup, the application:
1. Initializes the SQLite database at `data/watchguard.db`.
2. Starts the Event Logger and logs a `SYSTEM_STARTUP` event.
3. Initializes the Person Recognition Engine (YuNet + SFace + PAD) and loads any registered faces from the database.
4. Initializes the Object Detection Engine (YOLOv8n).
5. Initializes the Context & Security Intelligence Engine with the configured protected objects, zones, and authorized hours.
6. Initializes the Voice Assistant.
7. Initializes the Camera Manager and launches the Tkinter desktop dashboard.

The dashboard opens with the camera **offline**. The stream indicator in the header shows `ΓùÅ STREAM OFFLINE` until you start the camera.

---

## 2. Starting and Stopping the Camera

### Start the camera
1. In the bottom camera controls bar, click **Γû╢ START CAMERA**.
2. The app attempts to open the physical webcam selected in the **Dev:** dropdown (default index `0`).
3. If a physical camera is available, the stream indicator changes to show a live connection and the video panel begins showing the live feed with the vision HUD overlay.
4. If no physical camera is found and the app is configured to allow fallback, it may operate in **SYNTHETIC** simulation mode instead of reporting an error.

When the camera is running, the controls bar also shows the live FPS value and the selected device, demo target, and scenario.

### Stop the camera
1. While the camera is running, the button label changes to **Γûá STOP CAMERA**.
2. Click it to stop capture and release the webcam.
3. The UI returns to the idle/standby state: stream indicator shows `ΓùÅ STREAM OFFLINE`, telemetry and context panels reset to `STANDBY` / `None`, and the video canvas shows the "Camera Inactive" placeholder.

### Selecting a different camera
Use the **Dev:** combobox to choose a camera index (`0`ΓÇô`3`) before or while the camera is stopped. The selected index is used the next time the camera starts.

---

## 3. Face Registration

### Why register faces
Registered faces let the system distinguish **authorized** personnel from **unknown/unregistered** visitors. Recognition uses SFace cosine similarity against stored face embeddings in the database.

### How to register a face
1. Keep the camera running and position your face in the frame, or prepare a portrait image that contains a detectable face.
2. Click **≡ƒæñ REGISTER FACE** in the camera controls bar.
3. The registration dialog asks for the identity details used to label the profile in the database:
   - Username
   - Full name
   - Role
   - Access level
4. Provide a source image containing the face to enroll. The engine:
   - Runs YuNet face detection on the image.
   - Selects the largest detected face.
   - Aligns/crops the face with SFace.
   - Extracts the 128-d feature embedding.
   - Saves a face crop image into `data/faces/`.
   - Persists the embedding and metadata in the SQLite database.
5. Once registration succeeds, the engine reloads known faces. The **Profiles:** count in the Vision Telemetry card updates to reflect the new registered profile.

### Recognized identity levels
When the camera sees a person, the system classifies them into one of these tiers:
- **Authorized**: cosine similarity meets or exceeds the configured similarity threshold. The person is matched to a registered profile.
- **Unverified / Ambiguous**: similarity is in the ambiguous range. The system treats this as an unknown person in secure mode.
- **Unknown Person**: no sufficient match to any registered profile.

If no face profiles exist in the database at all, every detected person is treated as an unknown visitor.

> Note: face recognition is paired with presentation attack detection (liveness/PAD). A recognized face still goes through liveness verification before access is granted.

---

## 4. Security Dashboard Indicators

The right-side **Security Threat Level** panel shows the current risk level with a colored banner. The four levels correspond directly to the security rule engine output.

### GREEN ΓÇö NORMAL
- **Meaning**: Authorized personnel verified, normal objects, no policy breaches detected.
- **When you see it**: A registered/authorized person is present with no suspicious combination of objects, zone, or time-context violations.
- **Visual cue**: Emerald green banner.

### YELLOW ΓÇö ATTENTION
- **Meaning**: An unknown/unregistered individual has been detected in the surveillance area.
- **When you see it**: The system detects a person who does not match any registered face profile.
- **Visual cue**: Amber banner.

### ORANGE ΓÇö HIGH RISK
- **Meaning**: Elevated-risk context detected, such as:
  - An authorized person interacting with a protected asset outside authorized hours.
  - A person detected inside a configured restricted zone.
- **When you see it**: The context engine matches one of the elevated-risk rules based on person identity, object presence, zone, and time.
- **Visual cue**: Bright orange banner.

### RED ΓÇö SECURITY INCIDENT
- **Meaning**: Critical security violation.
- **When you see it**: For example, an unknown person is detected near a protected asset, or an unregistered individual is detected inside a restricted zone.
- **Visual cue**: Crimson red banner.

The same colors are used in the event log rows and in the manual simulate buttons in the threat level panel, so the dashboard stays visually consistent across the banner, telemetry, and event feed.

The context panel below the banner shows the contributing factors the engine is currently using: detected person, objects, zone/occupancy, time, and the current decision label.

---

## 5. Event History and Clear History

### Real-Time Security Event Log
The bottom panel is the live event feed. It displays recent security events from the SQLite database in a table with columns:
- **#** ΓÇö event ID
- **Timestamp** ΓÇö when the event was logged
- **Risk Level** ΓÇö GREEN / YELLOW / ORANGE / RED
- **Event Type** ΓÇö for example `CAMERA_STARTED`, `UNKNOWN_PERSON`, `SECURITY_ALERT`, `POLICY_VIOLATION`, etc.
- **Description / Traceable Rationale** ΓÇö a human-readable reason for the event

The feed is ordered with the newest events first and is styled with risk-colored row tags so incidents stand out.

### Refreshing the event list
Click **≡ƒöä Refresh Events** to reload the latest events from the database into the table. The header also shows a "Live Sync Active" indicator.

### Clearing history
Click **≡ƒº╣ Clear History** to clear the displayed event history. This is the UI operation for clearing event history from the log view.

Use the event feed to review:
- system start/stop
- camera start/stop/error events
- person and object detections
- security alerts and policy violations
- incident lifecycle events (detected, verified, alerted, acknowledged, resolved, escalated)

---

## 6. Common Camera Errors and Basic Troubleshooting

### "No physical camera device detected"
**Symptom**: Starting the camera results in an error status rather than a live feed.
**Basic checks**:
- Confirm a webcam is connected and recognized by the OS.
- Try a different camera index in the **Dev:** dropdown (e.g. `0`, `1`, `2`, `3`).
- Restart the application after connecting the camera.

### Camera stream becomes unstable or freezes
**Symptom**: The video feed stalls, drops repeatedly, or the status changes to an error state after working initially.
**Basic checks**:
- The capture loop treats many consecutive dropped frames as a hardware stream disconnect. If this happens, stop and restart the camera.
- If possible, try a different USB port or another camera device.
- If no physical camera is available, the system can fall back to synthetic simulation mode for testing.

### Camera works but no faces are detected
**Symptom**: The stream is live, but the person telemetry shows no person or keeps reporting "No Person".
**Basic checks**:
- Make sure the face is visible and reasonably lit in the camera view.
- Hold the face still long enough for detection and recognition to process.
- Confirm the face models exist under `data/models/`.
- Remember that face recognition requires a registered profile in the database; without one, detected people are reported as unknown.

### Recognized person still shows as unknown or unverified
**Symptom**: A registered user is detected but the system reports unverified/unknown or denies access.
**Basic checks**:
- Face registration may need to be repeated with a clearer, better-lit portrait.
- Liveness/PAD verification may still be in progress or may have timed out; wait for the verification state to resolve.
- If similarity is borderline, the system is designed to fail secure and treat the person as unknown.

### Synthetic / demo mode is active when you expected a real camera
**Symptom**: The feed shows the synthetic simulator overlay instead of a live webcam.
**Basic checks**:
- Check the device status text and the session info card; synthetic mode is used when no physical camera is available or when a demo target is selected.
- Set the **Target:** dropdown to **Clear Room** and ensure the correct camera index is selected, then start the camera again.
- If you intend to test scenarios without hardware, the simulator supports "Clear Room", "Hunter (Auth)", and "Unknown Visitor" targets.

### Scenario dropdowns and testing
The **Scen:** dropdown lets you simulate context conditions without waiting for real time/zone conditions:
- **Standard Monitoring** ΓÇö normal operation.
- **Simulate After-Hours (22:30)** ΓÇö triggers after-hours context for testing Rule D behavior.
- **Force Restricted Zone** ΓÇö forces restricted-zone context for testing Rule E / Rule F behavior.
- **Simulate Daytime (14:00)** ΓÇö daytime context for testing normal authorized hours behavior.

---

## File Layout (for orientation)

```
watchguard_vision/
Γö£ΓöÇΓöÇ main.py                  # Application entry point
Γö£ΓöÇΓöÇ config.py                # Paths, models, zones, authorized hours, thresholds
Γö£ΓöÇΓöÇ requirements.txt         # Python dependencies
Γö£ΓöÇΓöÇ data/
Γöé   Γö£ΓöÇΓöÇ watchguard.db        # SQLite database
Γöé   Γö£ΓöÇΓöÇ models/              # ONNX models
Γöé   Γö£ΓöÇΓöÇ faces/               # Registered face crops
Γöé   Γö£ΓöÇΓöÇ demo_faces/          # Sample demo portraits
Γöé   Γö£ΓöÇΓöÇ evidence/            # Captured incident snapshots
Γöé   ΓööΓöÇΓöÇ logs/                # Runtime log file
ΓööΓöÇΓöÇ src/
    Γö£ΓöÇΓöÇ camera/              # Camera manager
    Γö£ΓöÇΓöÇ database/            # SQLite persistence
    Γö£ΓöÇΓöÇ events/              # Event types and logging
    Γö£ΓöÇΓöÇ core/                # Face recognition, object detection, context engine, voice assistant, PAD
    ΓööΓöÇΓöÇ ui/                  # Desktop dashboard
```

---


