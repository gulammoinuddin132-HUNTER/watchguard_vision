"""
WatchGuard Vision - Main Desktop Dashboard
-------------------------------------------
Comprehensive Tkinter + Pillow Desktop Dashboard featuring real-time camera feed,
compact telemetry cards, risk state management, live security events table, camera hardware
controls, Face Recognition & Registration (Phase 2.1), Object Detection (Phase 2.2),
explainable Context Intelligence (Phase 2.3), and Voice Assistant (Phase 2.4).
"""

from datetime import datetime, time as dtime
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import (
    APP_TITLE,
    APP_VERSION,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    VIDEO_DISPLAY_WIDTH,
    VIDEO_DISPLAY_HEIGHT,
    DEFAULT_CAMERA_INDEX,
    DEMO_FACES_DIR,
    MAX_RECENT_EVENTS_DISPLAY,
)
from src.camera.camera_manager import CameraManager, CameraStatus, CameraMode
from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant
from src.core.pad.joint_decision import JointDecision, JointStatus, PADResult
from src.ui.theme import (
    BG_DARK,
    CARD_BG,
    CARD_BORDER,
    CARD_HEADER_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    ACCENT_BLUE,
    ACCENT_PURPLE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_RED,
    TABLE_BG,
    TABLE_BG_ALT,
    TABLE_TEXT,
    TABLE_HEADER_BG,
    TABLE_HEADER_FG,
    RISK_GREEN_FG,
    RISK_GREEN_BG,
    RISK_YELLOW_FG,
    RISK_YELLOW_BG,
    RISK_ORANGE_FG,
    RISK_ORANGE_BG,
    RISK_RED_FG,
    RISK_RED_BG,
    BTN_START_BG,
    BTN_STOP_BG,
    BTN_NEUTRAL_BG,
    BTN_NEUTRAL_BORDER,
    FONT_TITLE,
    FONT_SUBTITLE,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_SMALL,
    FONT_MONO,
)
from src.ui.widgets import BoundedTextViewport, CardFrame, RiskBanner, TelemetryItem


class WatchGuardDashboard:
    """Main desktop user interface for WatchGuard Vision."""

    def __init__(
        self,
        root: tk.Tk,
        camera_manager: CameraManager,
        db_manager: DatabaseManager,
        event_logger: EventLogger,
        person_engine: Optional[PersonRecognitionEngine] = None,
        object_engine: Optional[ObjectDetectionEngine] = None,
        context_engine: Optional[ContextEngine] = None,
        voice_assistant: Optional[VoiceAssistant] = None,
    ):
        self.root = root
        self.camera = camera_manager
        self.db = db_manager
        self.logger = event_logger
        self.person_engine = person_engine or PersonRecognitionEngine(db_manager=self.db)
        self.object_engine = object_engine or ObjectDetectionEngine(db_manager=self.db)
        self.context_engine = context_engine or ContextEngine(db_manager=self.db)
        self.voice_assistant = voice_assistant or VoiceAssistant(
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.person_engine,
            object_engine=self.object_engine,
            context_engine=self.context_engine,
            camera_manager=self.camera,
        )

        # Initialize vision engines
        if not self.person_engine.is_ready:
            self.person_engine.initialize()
        if not self.object_engine.is_ready:
            self.object_engine.initialize()

        # Connect Voice Assistant status listener
        self.voice_assistant.set_status_callback(self._on_voice_status_changed)

        # State Variables & Caches
        self.camera_index_var = tk.IntVar(value=DEFAULT_CAMERA_INDEX)
        self.demo_target_var = tk.StringVar(value="Clear Room")
        self.scenario_var = tk.StringVar(value="Standard Monitoring")
        self.current_risk_level = RiskLevel.GREEN
        self.event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._canvas_img_id: Optional[int] = None

        # Multi-Threaded Decoupled AI Worker & Display State (Phase 5.8)
        self._is_closed: bool = False
        self._after_video_id: Optional[str] = None
        self._after_ui_id: Optional[str] = None
        self._after_pump_id: Optional[str] = None
        self._latest_vision_state: Optional[Dict[str, Any]] = {
            "obj_results": {"detected": False, "count": 0, "objects": []},
            "person_results": {"detected": False, "people": []},
            "decision": None,
            "fps": 0.0,
            "camera_status": CameraStatus.OFFLINE,
            "camera_index": DEFAULT_CAMERA_INDEX,
            "demo_mode": "none",
            "ai_fps": 0.0,
            "ai_latency_ms": 0.0,
        }
        self._latest_ai_state: Optional[Dict[str, Any]] = None
        self._state_lock = threading.Lock()
        self._ai_worker_thread: Optional[threading.Thread] = None
        self._ai_stop_event = threading.Event()
        self._ai_frame_lock = threading.Lock()
        self._ai_input_frame: Optional[np.ndarray] = None
        self._ai_frame_event = threading.Event()
        self._ai_fps: float = 0.0
        self._ai_latency_ms: float = 0.0
        self._display_fps: float = 0.0
        self._display_frame_count: int = 0
        self._display_fps_timer: float = time.perf_counter()
        self._last_fps_str: str = ""
        self._last_stream_status_text: str = ""
        self._last_stream_status_fg: str = ""

        # Setup GUI Window
        self._configure_window()
        self._setup_styles()
        self._build_layout()

        # Connect event listeners & device discovery callbacks
        self.logger.subscribe(self._on_event_logged)
        self.camera.set_on_device_selected_callback(self._on_device_auto_selected)

        # Start dedicated background AI inference worker thread
        self._start_ai_worker()

        # Start Video loop, decoupled UI telemetry refresh loop & event pump
        self._schedule_video_update()
        self._schedule_ui_telemetry_update()
        self._schedule_event_pump()

        # Initial database population for event table
        self._load_recent_events()

    def _configure_window(self) -> None:
        """Sets up window geometry, title, and dark background."""
        self.root.title(f"{APP_TITLE} (v{APP_VERSION})")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(1150, 780)
        self.root.configure(bg=BG_DARK)

        # Handle window close cleanly
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Destroy>", self._on_destroy_event, add="+")

    def _on_destroy_event(self, event=None) -> None:
        """Ensures background worker threads and timers stop when widget is destroyed."""
        if event is None or event.widget == self.root:
            self._is_closed = True
            self._ai_stop_event.set()
            self._ai_frame_event.set()
            if hasattr(self, "camera") and self.camera and self.camera.is_running:
                try:
                    self.camera.stop()
                except Exception:
                    pass

    def _setup_styles(self) -> None:
        """Configures ttk styles for dark modern high-contrast look."""
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        # Configure Treeview for high-contrast dark theme
        self.style.configure(
            "Events.Treeview",
            background=TABLE_BG,
            foreground=TABLE_TEXT,
            fieldbackground=TABLE_BG,
            borderwidth=0,
            font=FONT_SMALL,
            rowheight=24,
        )
        self.style.configure(
            "Events.Treeview.Heading",
            background=TABLE_HEADER_BG,
            foreground=TABLE_HEADER_FG,
            font=FONT_BODY_BOLD,
            borderwidth=0,
            relief="flat",
        )
        self.style.map(
            "Events.Treeview",
            background=[("selected", "#2563EB")],
            foreground=[("selected", "#FFFFFF")],
        )
        self.style.map(
            "Events.Treeview.Heading",
            background=[("active", "#334155")],
            foreground=[("active", "#FFFFFF")],
        )

        # Configure Combobox
        self.style.configure(
            "TCombobox",
            fieldbackground=CARD_BG,
            background=CARD_HEADER_BG,
            foreground=TEXT_PRIMARY,
            arrowcolor=TEXT_PRIMARY,
        )

    def _build_layout(self) -> None:
        """Constructs the stable 3-row layout structure with bounded upper area and guaranteed event log height."""
        # Configure root grid weights
        self.root.grid_rowconfigure(0, weight=0)  # Top Header Bar (fixed height 40px)
        self.root.grid_rowconfigure(1, weight=1)  # Main Workspace Container (expands vertically)
        self.root.grid_columnconfigure(0, weight=1)

        # Top Header Bar
        self._build_header_bar()

        # Main Workspace Container
        main_workspace = tk.Frame(self.root, bg=BG_DARK)
        main_workspace.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 6))
        main_workspace.grid_rowconfigure(0, weight=0)  # Row 0: UPPER_AREA (bounded height ~395px)
        main_workspace.grid_rowconfigure(1, weight=0)  # Row 1: CAMERA_CONTROLS_BAR (compact height ~36px)
        main_workspace.grid_rowconfigure(2, weight=1, minsize=300)  # Row 2: EVENT_LOG_AREA (Guaranteed >= 300px)
        main_workspace.grid_columnconfigure(0, weight=1)

        # Row 0: Upper Section (Video Feed Left + Security Intelligence Right)
        upper_area = tk.Frame(main_workspace, bg=BG_DARK)
        upper_area.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        upper_area.grid_rowconfigure(0, weight=1)
        upper_area.grid_columnconfigure(0, weight=1)  # Left video column expands horizontally
        upper_area.grid_columnconfigure(1, weight=0)  # Right panel fixed width (470px)

        # Left Column: Video Card
        self._build_video_card(upper_area)

        # Right Column: Fixed 470px Container for 4 Compact Cards
        right_panel = tk.Frame(upper_area, bg=BG_DARK, width=470)
        right_panel.grid(row=0, column=1, sticky="nsew")

        self._build_risk_panel(right_panel)
        self._build_context_intelligence_card(right_panel)
        self._build_telemetry_card(right_panel)
        self._build_voice_assistant_card(right_panel)

        # Row 1: Horizontal Camera & Scenario Controls Bar
        self._build_camera_controls(main_workspace)

        # Row 2: Real-time Event Feed (consumes all remaining height, minsize=240)
        self._build_event_feed(main_workspace)

    def _build_header_bar(self) -> None:
        """Builds top brand / status header."""
        header = tk.Frame(self.root, bg=CARD_HEADER_BG, height=40, padx=14)
        header.grid(row=0, column=0, sticky="ew")
        header.pack_propagate(False)

        # Brand Title
        brand_frame = tk.Frame(header, bg=CARD_HEADER_BG)
        brand_frame.pack(side="left", fill="y")

        lbl_logo = tk.Label(
            brand_frame,
            text="👁️",
            font=("Segoe UI Emoji", 14),
            bg=CARD_HEADER_BG,
            fg=ACCENT_BLUE,
        )
        lbl_logo.pack(side="left", padx=(0, 6))

        lbl_brand = tk.Label(
            brand_frame,
            text="WATCHGUARD VISION",
            font=FONT_TITLE,
            bg=CARD_HEADER_BG,
            fg=TEXT_PRIMARY,
        )
        lbl_brand.pack(side="left")

        lbl_tag = tk.Label(
            brand_frame,
            text="AI VISUAL SECURITY ASSISTANT",
            font=FONT_SMALL,
            bg=CARD_HEADER_BG,
            fg=ACCENT_BLUE,
        )
        lbl_tag.pack(side="left", padx=(8, 0), pady=(3, 0))

        # Right status indicator
        sys_status_frame = tk.Frame(header, bg=CARD_HEADER_BG)
        sys_status_frame.pack(side="right", fill="y")

        self.lbl_stream_status = tk.Label(
            sys_status_frame,
            text="● STREAM OFFLINE",
            font=FONT_BODY_BOLD,
            bg=CARD_HEADER_BG,
            fg=TEXT_MUTED,
        )
        self.lbl_stream_status.pack(side="right", pady=9)

    def _build_video_card(self, parent: tk.Frame) -> None:
        """Builds the main video frame viewport inside the upper area."""
        self.video_card = CardFrame(parent, title="Live Video Stream & Vision HUD")
        self.video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Canvas for rendering video frames (bounded 350px height)
        self.video_canvas = tk.Canvas(
            self.video_card,
            bg="#000000",
            width=VIDEO_DISPLAY_WIDTH,
            height=350,
            highlightthickness=0,
        )
        self.video_canvas.pack(fill="both", expand=True, padx=4, pady=3)

        # Initial placeholder on canvas
        self._render_placeholder_frame("Camera Inactive - Click [Start Camera] below")

    def _build_camera_controls(self, parent: tk.Frame) -> None:
        """Builds compact horizontal controls bar across row 1."""
        ctrl_card = CardFrame(parent, title="Camera & Scenario Controls")
        ctrl_card.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        bar = tk.Frame(ctrl_card, bg=CARD_BG, padx=6, pady=2)
        bar.pack(fill="x")

        # Start / Stop Toggle Button
        self.btn_toggle_cam = tk.Button(
            bar,
            text="▶ START CAMERA",
            font=FONT_BODY_BOLD,
            bg=BTN_START_BG,
            fg="#FFFFFF",
            activebackground=COLOR_GREEN,
            activeforeground="#FFFFFF",
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._toggle_camera,
        )
        self.btn_toggle_cam.pack(side="left", padx=(0, 5))

        # Register Face Button
        self.btn_reg_face = tk.Button(
            bar,
            text="👤 REGISTER FACE",
            font=FONT_BODY_BOLD,
            bg="#3B82F6",
            fg="#FFFFFF",
            activebackground="#2563EB",
            relief="flat",
            padx=7,
            pady=2,
            cursor="hand2",
            command=self._open_face_registration_dialog,
        )
        self.btn_reg_face.pack(side="left", padx=(0, 5))

        # Snapshot Button
        self.btn_snapshot = tk.Button(
            bar,
            text="📷 SNAPSHOT",
            font=FONT_SMALL,
            bg=BTN_NEUTRAL_BG,
            fg=TEXT_PRIMARY,
            relief="groove",
            padx=5,
            pady=2,
            cursor="hand2",
            command=self._take_snapshot,
        )
        self.btn_snapshot.pack(side="left", padx=(0, 6))

        # Camera Index Selector
        lbl_idx = tk.Label(
            bar,
            text="Dev:",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
        )
        lbl_idx.pack(side="left", padx=(0, 2))

        self.combo_cam_idx = ttk.Combobox(
            bar,
            textvariable=self.camera_index_var,
            values=[0, 1, 2, 3],
            width=3,
            state="readonly",
        )
        self.combo_cam_idx.pack(side="left", padx=(0, 5))
        self.combo_cam_idx.bind("<<ComboboxSelected>>", self._on_camera_index_changed)

        # Simulation Target Dropdown
        lbl_demo = tk.Label(
            bar,
            text="Target:",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=ACCENT_BLUE,
        )
        lbl_demo.pack(side="left", padx=(0, 2))

        self.combo_demo = ttk.Combobox(
            bar,
            textvariable=self.demo_target_var,
            values=["Clear Room", "Hunter (Auth)", "Unknown Visitor"],
            width=13,
            state="readonly",
        )
        self.combo_demo.pack(side="left", padx=(0, 5))
        self.combo_demo.bind("<<ComboboxSelected>>", self._on_demo_target_changed)

        # Scenario Override Dropdown (Phase 2.3)
        lbl_scen = tk.Label(
            bar,
            text="Scen:",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=COLOR_ORANGE,
        )
        lbl_scen.pack(side="left", padx=(0, 2))

        self.combo_scenario = ttk.Combobox(
            bar,
            textvariable=self.scenario_var,
            values=[
                "Standard Monitoring",
                "Simulate After-Hours (22:30)",
                "Force Restricted Zone",
                "Simulate Daytime (14:00)",
            ],
            width=18,
            state="readonly",
        )
        self.combo_scenario.pack(side="left", padx=(0, 5))
        self.combo_scenario.bind("<<ComboboxSelected>>", self._on_scenario_changed)

        # FPS Display
        self.lbl_fps = tk.Label(
            bar,
            text="FPS: 0.0",
            font=FONT_MONO,
            bg=CARD_BG,
            fg=ACCENT_BLUE,
        )
        self.lbl_fps.pack(side="right")

    def _build_risk_panel(self, parent: tk.Frame) -> None:
        """Builds the Security Status & Risk Meter."""
        risk_card = CardFrame(parent, title="Security Threat Level")
        risk_card.pack(fill="x", pady=(0, 2))

        inner = tk.Frame(risk_card, bg=CARD_BG, padx=6, pady=2)
        inner.pack(fill="x")

        self.risk_banner = RiskBanner(inner)
        self.risk_banner.pack(fill="x")

        # Quick Manual Risk simulation buttons (for test & demo verification)
        sim_frame = tk.Frame(inner, bg=CARD_BG)
        sim_frame.pack(fill="x", pady=(2, 0))

        lbl_test = tk.Label(
            sim_frame,
            text="Simulate:",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=TEXT_MUTED,
        )
        lbl_test.pack(side="left", padx=(0, 4))

        for level, color in [
            (RiskLevel.GREEN, COLOR_GREEN),
            (RiskLevel.YELLOW, COLOR_YELLOW),
            (RiskLevel.ORANGE, COLOR_ORANGE),
            (RiskLevel.RED, COLOR_RED),
        ]:
            btn = tk.Button(
                sim_frame,
                text=level.value[0],
                font=FONT_SMALL,
                bg=color,
                fg="#FFFFFF",
                relief="flat",
                width=2,
                pady=0,
                cursor="hand2",
                command=lambda lvl=level: self._set_risk_level(lvl),
            )
            btn.pack(side="left", padx=1)

    def _build_context_intelligence_card(self, parent: tk.Frame) -> None:
        """Builds the Context & Security Intelligence card (Phase 2.3 Active)."""
        card = CardFrame(parent, title="Context & Security Intelligence (Phase 2.3 Active)")
        card.pack(fill="x", pady=(0, 2))

        content = tk.Frame(card, bg=CARD_BG, padx=6, pady=2)
        content.pack(fill="x")

        # Row 1: Person (left) & Zone (right)
        row1 = tk.Frame(content, bg=CARD_BG)
        row1.pack(fill="x")
        self.ctx_item_person = TelemetryItem(row1, "Person:", "None")
        self.ctx_item_person.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.ctx_item_zone = TelemetryItem(row1, "Zone:", "Perimeter")
        self.ctx_item_zone.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Row 2: Objects (left) & Time (right)
        row2 = tk.Frame(content, bg=CARD_BG)
        row2.pack(fill="x")
        self.ctx_item_object = TelemetryItem(row2, "Objects:", "None")
        self.ctx_item_object.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.ctx_item_time = TelemetryItem(row2, "Time:", "--:--")
        self.ctx_item_time.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Row 3 (Phase 4.1): Zone State/Duration (left) & Distinct Violations (right)
        row3 = tk.Frame(content, bg=CARD_BG)
        row3.pack(fill="x")
        self.ctx_item_zone_state = TelemetryItem(row3, "Occupancy:", "OUTSIDE")
        self.ctx_item_zone_state.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.ctx_item_violations = TelemetryItem(row3, "Violations:", "0 Distinct")
        self.ctx_item_violations.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Row 4 (Phase 5): Spoofs (left) & After-Hours Violations (right)
        row4 = tk.Frame(content, bg=CARD_BG)
        row4.pack(fill="x")
        self.ctx_item_analytics_spoofs = TelemetryItem(row4, "Spoofs Today:", "0")
        self.ctx_item_analytics_spoofs.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.ctx_item_analytics_after_hours = TelemetryItem(row4, "After-Hours:", "0")
        self.ctx_item_analytics_after_hours.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Decision row
        self.ctx_item_decision = TelemetryItem(content, "Decision:", "STANDBY")
        self.ctx_item_decision.pack(fill="x")

        # Incident Lifecycle & Action Row (Phase 4)
        inc_frame = tk.Frame(content, bg=CARD_BG)
        inc_frame.pack(fill="x", pady=(1, 0))

        self.ctx_item_incident = TelemetryItem(inc_frame, "Incident:", "None")
        self.ctx_item_incident.pack(side="left", fill="x", expand=True)

        self.btn_ack_inc = tk.Button(
            inc_frame,
            text="ACK",
            font=FONT_SMALL,
            bg="#D97706",
            fg="#FFFFFF",
            relief="flat",
            padx=4,
            pady=0,
            cursor="hand2",
            command=self._on_ack_incident_clicked,
        )
        self.btn_ack_inc.pack(side="right", padx=(2, 0))

        self.btn_resolve_inc = tk.Button(
            inc_frame,
            text="RESOLVE",
            font=FONT_SMALL,
            bg="#10B981",
            fg="#FFFFFF",
            relief="flat",
            padx=4,
            pady=0,
            cursor="hand2",
            command=self._on_resolve_incident_clicked,
        )
        self.btn_resolve_inc.pack(side="right", padx=(2, 0))

        # Reason Display row with BoundedTextViewport (Zero parent geometry expansion)
        reason_frame = tk.Frame(content, bg=CARD_BG, height=24)
        reason_frame.pack(fill="x", pady=(1, 0))
        reason_frame.pack_propagate(False)

        tk.Label(
            reason_frame,
            text="Reason:",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            anchor="nw",
        ).pack(side="left", padx=(0, 4), anchor="nw")

        self.lbl_ctx_reason = BoundedTextViewport(
            reason_frame,
            height_px=22,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=FONT_SMALL,
        )
        self.lbl_ctx_reason.pack(side="left", fill="both", expand=True)
        self.lbl_ctx_reason.set_text("Camera inactive - Stream stopped. Click [START CAMERA] to begin.")

    def _build_telemetry_card(self, parent: tk.Frame) -> None:
        """Builds active detection telemetry stats with Phase 3 PAD Liveness rows."""
        card = CardFrame(parent, title="Vision Telemetry (Phases 2.1, 2.2 & 3.0 Active)")
        card.pack(fill="x", pady=(0, 2))

        content = tk.Frame(card, bg=CARD_BG, padx=6, pady=2)
        content.pack(fill="x")

        # Compact 2-column grid for telemetry items
        row1 = tk.Frame(content, bg=CARD_BG)
        row1.pack(fill="x")
        self.item_person = TelemetryItem(row1, "Face AI:", "STANDBY")
        self.item_person.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.item_registered_faces = TelemetryItem(row1, "Profiles:", f"{self.person_engine.known_face_count}")
        self.item_registered_faces.pack(side="right", fill="x", expand=True, padx=(4, 0))

        row2 = tk.Frame(content, bg=CARD_BG)
        row2.pack(fill="x")
        self.item_objects = TelemetryItem(row2, "Object AI:", "STANDBY")
        self.item_objects.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.item_db_records = TelemetryItem(row2, "DB Events:", "0")
        self.item_db_records.pack(side="right", fill="x", expand=True, padx=(4, 0))

        row3 = tk.Frame(content, bg=CARD_BG)
        row3.pack(fill="x")
        self.item_liveness = TelemetryItem(row3, "Liveness:", "STANDBY")
        self.item_liveness.pack(side="left", fill="x", expand=True, padx=(0, 4))
        pad_name = getattr(getattr(self.person_engine, "pad_manager", None), "active_model_name", "Blink-EAR")
        self.item_pad_model = TelemetryItem(row3, "PAD Model:", f"{pad_name}")
        self.item_pad_model.pack(side="right", fill="x", expand=True, padx=(4, 0))

        row4 = tk.Frame(content, bg=CARD_BG)
        row4.pack(fill="x")
        self.item_access = TelemetryItem(row4, "Access:", "STANDBY")
        self.item_access.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.item_pad_latency = TelemetryItem(row4, "PAD Latency:", "-- ms")
        self.item_pad_latency.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _build_voice_assistant_card(self, parent: tk.Frame) -> None:
        """Builds the Voice Assistant Status & Interaction Card (Phase 2.4 Active)."""
        card = CardFrame(parent, title="Voice Assistant (Phase 2.4 Active)")
        card.pack(fill="x", pady=(0, 0))

        content = tk.Frame(card, bg=CARD_BG, padx=6, pady=2)
        content.pack(fill="x")

        # Row 1: Status & Mute Checkbox
        status_row = tk.Frame(content, bg=CARD_BG)
        status_row.pack(fill="x")

        self.item_voice_status = TelemetryItem(status_row, "Status:", "● READY / STANDBY")
        self.item_voice_status.pack(side="left", fill="x", expand=True)

        self.mute_var = tk.BooleanVar(value=False)
        chk_mute = tk.Checkbutton(
            status_row,
            text="Mute",
            variable=self.mute_var,
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            selectcolor="#0F172A",
            activebackground=CARD_BG,
            activeforeground=TEXT_PRIMARY,
            command=self._on_mute_toggled,
        )
        chk_mute.pack(side="right", padx=(4, 0))

        # Row 2: Last Command
        self.item_voice_command = TelemetryItem(content, "Last Query:", "None")
        self.item_voice_command.pack(fill="x")

        # Row 3: Response display with BoundedTextViewport (Zero parent geometry expansion)
        resp_frame = tk.Frame(content, bg=CARD_BG, height=26)
        resp_frame.pack(fill="x", pady=(1, 1))
        resp_frame.pack_propagate(False)

        tk.Label(
            resp_frame,
            text="Response:",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            anchor="nw",
        ).pack(side="left", padx=(0, 4), anchor="nw")

        self.lbl_voice_resp = BoundedTextViewport(
            resp_frame,
            height_px=24,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            font=FONT_SMALL,
        )
        self.lbl_voice_resp.pack(side="left", fill="both", expand=True)
        self.lbl_voice_resp.set_text("Voice assistant ready. Ask a question or use Push to Talk.")

        # Row 4: Combined PTT & Text Fallback Action Row
        action_row = tk.Frame(content, bg=CARD_BG)
        action_row.pack(fill="x", pady=(1, 0))

        self.btn_ptt = tk.Button(
            action_row,
            text="🎙️ PTT (3.5s)",
            font=FONT_BODY_BOLD,
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=6,
            pady=1,
            cursor="hand2",
            command=self._on_ptt_clicked,
        )
        self.btn_ptt.pack(side="left", padx=(0, 4))

        self.ent_voice_query = tk.Entry(
            action_row,
            font=FONT_SMALL,
            bg="#0F172A",
            fg=TEXT_PRIMARY,
            insertbackground="#FFFFFF",
        )
        self.ent_voice_query.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.ent_voice_query.insert(0, "What happened recently?")
        self.ent_voice_query.bind("<Return>", lambda e: self._on_send_text_query())

        self.btn_ask = tk.Button(
            action_row,
            text="ASK",
            font=FONT_BODY_BOLD,
            bg=BTN_NEUTRAL_BG,
            fg=TEXT_PRIMARY,
            relief="groove",
            padx=6,
            pady=1,
            cursor="hand2",
            command=self._on_send_text_query,
        )
        self.btn_ask.pack(side="right")

    def _build_event_feed(self, parent: tk.Frame) -> None:
        """Builds the bottom tabular live event feed with high-contrast text and two-way scrolling."""
        card = CardFrame(parent, title="Real-Time Security Event Log")
        card.grid(row=2, column=0, sticky="nsew")

        # Action Toolbar above table
        toolbar = tk.Frame(card, bg=CARD_BG, padx=6, pady=3)
        toolbar.pack(fill="x", padx=6, pady=(0, 2))

        btn_refresh = tk.Button(
            toolbar,
            text="🔄 Refresh Events",
            font=FONT_BODY_BOLD,
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._load_recent_events,
        )
        btn_refresh.pack(side="left", padx=(0, 6))

        btn_clear_ui = tk.Button(
            toolbar,
            text="🧹 Clear Table View",
            font=FONT_SMALL,
            bg=BTN_NEUTRAL_BG,
            fg=TEXT_PRIMARY,
            relief="groove",
            padx=6,
            pady=2,
            cursor="hand2",
            command=self._clear_treeview,
        )
        btn_clear_ui.pack(side="left", padx=(0, 8))

        self.lbl_table_status = tk.Label(
            toolbar,
            text="● Live Sync Active (Newest First)",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=COLOR_GREEN,
        )
        self.lbl_table_status.pack(side="left")

        # Treeview table container
        tree_frame = tk.Frame(card, bg=CARD_BG)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        columns = ("id", "timestamp", "risk", "type", "description")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            style="Events.Treeview",
            selectmode="browse",
        )

        self.tree.heading("id", text="#", anchor="center")
        self.tree.heading("timestamp", text="Timestamp", anchor="w")
        self.tree.heading("risk", text="Risk Level", anchor="center")
        self.tree.heading("type", text="Event Type", anchor="w")
        self.tree.heading("description", text="Description / Traceable Rationale", anchor="w")

        self.tree.column("id", width=55, minwidth=45, anchor="center", stretch=False)
        self.tree.column("timestamp", width=155, minwidth=140, anchor="w", stretch=False)
        self.tree.column("risk", width=150, minwidth=135, anchor="center", stretch=False)
        self.tree.column("type", width=190, minwidth=165, anchor="w", stretch=False)
        self.tree.column("description", width=620, minwidth=350, anchor="w", stretch=True)

        # Vertical & Horizontal Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid placement inside tree_frame for clean two-way scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Tags for high-contrast colored rows with subtle background tint
        self.tree.tag_configure("GREEN", foreground=RISK_GREEN_FG, background=RISK_GREEN_BG)
        self.tree.tag_configure("YELLOW", foreground=RISK_YELLOW_FG, background=RISK_YELLOW_BG)
        self.tree.tag_configure("ORANGE", foreground=RISK_ORANGE_FG, background=RISK_ORANGE_BG)
        self.tree.tag_configure("RED", foreground=RISK_RED_FG, background=RISK_RED_BG)

    # --- Video Stream, Vision HUD & Context Integration ---

    def _render_placeholder_frame(self, message: str) -> None:
        """Draws a standby graphic on the canvas when video is inactive."""
        w = max(VIDEO_DISPLAY_WIDTH, self.video_canvas.winfo_width())
        h = max(390, self.video_canvas.winfo_height())
        if w < 10 or h < 10:
            w, h = VIDEO_DISPLAY_WIDTH, 390

        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw tech border & standby text
        cv2.rectangle(img, (10, 10), (w - 10, h - 10), (35, 45, 60), 2)
        cv2.putText(
            img,
            "WATCHGUARD VISION // STANDBY",
            (w // 2 - 160, h // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
        )
        cv2.putText(
            img,
            message,
            (w // 2 - 190, h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 160, 175),
            1,
        )

        pil_img = Image.fromarray(img)
        self._photo_image = ImageTk.PhotoImage(image=pil_img, master=self.root)
        self.video_canvas.delete("all")
        self._canvas_img_id = self.video_canvas.create_image(0, 0, image=self._photo_image, anchor="nw")

    def _start_ai_worker(self) -> None:
        """Starts the dedicated background AI inference worker thread (Phase 5.8)."""
        self._ai_stop_event.clear()
        self._ai_worker_thread = threading.Thread(
            target=self._ai_worker_loop,
            name="WatchGuard-AI-Worker",
            daemon=True,
        )
        self._ai_worker_thread.start()

    def _ai_worker_loop(self) -> None:
        """
        Independent AI analysis worker thread (Phase 5.8).
        Executes YOLOv8 object detection, Person recognition (YuNet + SFace + PAD),
        PersonEntityFusion, and Context Engine.
        Always processes the newest available frame; discards stale frames to prevent lag accumulation.
        Publishes atomic security state snapshots for the UI.
        """
        ai_frame_count = 0
        ai_fps_timer = time.perf_counter()

        while not self._ai_stop_event.is_set():
            # Wait for a new frame signal
            if not self._ai_frame_event.wait(timeout=0.1):
                continue
            self._ai_frame_event.clear()

            # Retrieve latest unanalyzed frame atomically
            with self._ai_frame_lock:
                frame_bgr = self._ai_input_frame
                self._ai_input_frame = None

            if frame_bgr is None:
                continue

            t_start = time.perf_counter()

            # 1. Run Object Detection Analysis (Phase 2.2)
            if self.object_engine and self.object_engine.is_ready:
                obj_results = self.object_engine.process_frame(frame_bgr)
            else:
                obj_results = {"detected": False, "count": 0, "objects": []}

            # 2. Run Person Recognition Analysis (Phases 2.1 & 3.0 with Object Fusion)
            if self.person_engine and self.person_engine.is_ready:
                person_results = self.person_engine.process_frame(
                    frame_bgr,
                    detected_objects=obj_results.get("objects", []),
                )
            else:
                person_results = {"detected": False, "people": []}

            # 3. Evaluate Context & Security Intelligence (Phase 2.3)
            if self.context_engine:
                decision = self.context_engine.evaluate_context(
                    person_data=person_results,
                    object_data=obj_results,
                    camera_id=self.camera.camera_index,
                    frame_shape=frame_bgr.shape,
                )
            else:
                decision = None

            t_ai_elapsed = (time.perf_counter() - t_start) * 1000.0
            self._ai_latency_ms = t_ai_elapsed

            # Calculate AI Worker FPS
            ai_frame_count += 1
            now = time.perf_counter()
            if now - ai_fps_timer >= 1.0:
                self._ai_fps = ai_frame_count / (now - ai_fps_timer)
                ai_frame_count = 0
                ai_fps_timer = now

            # 4. Atomically publish latest completed security state
            new_state = {
                "obj_results": obj_results,
                "person_results": person_results,
                "decision": decision,
                "fps": self.camera.fps,
                "camera_status": self.camera.status,
                "camera_index": self.camera.camera_index,
                "demo_mode": self.camera.demo_mode,
                "ai_fps": self._ai_fps,
                "ai_latency_ms": self._ai_latency_ms,
            }
            with self._state_lock:
                self._latest_ai_state = new_state
                self._latest_vision_state = new_state

    def _schedule_video_update(self) -> None:
        """
        High-frequency Display/Rendering loop (~30 FPS).
        Grabs latest camera frame, submits newest frame copy to background AI worker,
        overlays latest completed AI security annotations, and renders to canvas.
        Zero blocking on AI inference.
        """
        if self._is_closed:
            return

        loop_start = time.perf_counter()

        if self.camera.is_running:
            success, frame_bgr = self.camera.get_frame()
            if success and frame_bgr is not None:
                # 1. Provide latest frame to AI Worker (drops stale unconsumed frame)
                with self._ai_frame_lock:
                    self._ai_input_frame = frame_bgr.copy()
                    self._ai_frame_event.set()

                annotated_bgr = frame_bgr.copy()

                # 2. Retrieve latest completed AI state atomically
                with self._state_lock:
                    ai_state = self._latest_ai_state

                if ai_state is not None:
                    decision = ai_state.get("decision")
                    obj_results = ai_state.get("obj_results")
                    person_results = ai_state.get("person_results")

                    # 3. Draw Zone boundaries & spatial interaction vectors
                    if self.context_engine and decision:
                        annotated_bgr = self.context_engine.draw_context_hud(annotated_bgr, decision)

                    # 4. Draw Object & Face Bounding Boxes
                    if self.object_engine and self.object_engine.is_ready and obj_results:
                        annotated_bgr = self.object_engine.draw_annotations(
                            annotated_bgr, obj_results, suppress_person_boxes=True
                        )
                    if self.person_engine and self.person_engine.is_ready and person_results:
                        annotated_bgr = self.person_engine.draw_annotations(
                            annotated_bgr,
                            person_results,
                            context_decision=decision,
                        )

                # 5. Convert to RGB and render onto Tkinter canvas
                frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                canvas_w = max(50, self.video_canvas.winfo_width())
                canvas_h = max(50, self.video_canvas.winfo_height())
                frame_resized = cv2.resize(frame_rgb, (canvas_w, canvas_h))

                pil_img = Image.fromarray(frame_resized)
                self._photo_image = ImageTk.PhotoImage(image=pil_img, master=self.root)
                if self._canvas_img_id is None:
                    self.video_canvas.delete("all")
                    self._canvas_img_id = self.video_canvas.create_image(0, 0, image=self._photo_image, anchor="nw")
                else:
                    self.video_canvas.itemconfig(self._canvas_img_id, image=self._photo_image)

                # Track display FPS
                self._display_frame_count += 1
                now = time.perf_counter()
                if now - self._display_fps_timer >= 1.0:
                    self._display_fps = self._display_frame_count / (now - self._display_fps_timer)
                    self._display_frame_count = 0
                    self._display_fps_timer = now

        # Dynamic frame rate pacing (~30 FPS = ~33.3ms total loop budget)
        elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
        target_interval_ms = 1000.0 / 30.0
        delay_ms = max(1, int(target_interval_ms - elapsed_ms))

        if not self._is_closed:
            try:
                self._after_video_id = self.root.after(delay_ms, self._schedule_video_update)
            except Exception:
                pass

    def _schedule_ui_telemetry_update(self) -> None:
        """
        Decoupled UI refresh loop running at ~10 Hz (100ms).
        Updates text labels, telemetry cards, and risk meter only when state changes.
        """
        if self._is_closed:
            return

        state = self._latest_vision_state
        if state is not None and self.camera.is_running:
            # 1. Update Object Detection Telemetry
            self._handle_object_detection_telemetry(state["obj_results"])

            # 2. Update Person Recognition Telemetry
            self._handle_person_recognition_telemetry(state["person_results"])

            # 3. Update Context Decision
            if state["decision"] is not None:
                self._handle_context_decision(state["decision"])

        if state is not None:
            # 4. Update FPS label with change-caching
            if self.camera.is_running:
                disp_fps = self._display_fps if self._display_fps > 0 else state.get("fps", 0.0)
                ai_fps_val = self._ai_fps if self._ai_fps > 0 else state.get("ai_fps", 0.0)
            else:
                disp_fps = 0.0
                ai_fps_val = 0.0
            fps_text = f"Display: {disp_fps:.1f} FPS | AI: {ai_fps_val:.1f} FPS"
            if fps_text != self._last_fps_str:
                self._last_fps_str = fps_text
                self.lbl_fps.configure(text=fps_text)

            # 5. Update Stream Status with change-caching
            if not self.camera.is_running:
                stat_text = "● STREAM OFFLINE"
                stat_fg = TEXT_MUTED
            else:
                cam_stat = state["camera_status"]
                cam_idx = state["camera_index"]
                demo_m = state["demo_mode"]
                if cam_stat == CameraStatus.REAL:
                    stat_text = f"● LIVE (Camera #{cam_idx})"
                    stat_fg = COLOR_GREEN
                elif cam_stat == CameraStatus.SYNTHETIC:
                    stat_text = f"● SIMULATION ({demo_m.upper()})"
                    stat_fg = COLOR_YELLOW
                elif cam_stat == CameraStatus.ERROR:
                    stat_text = f"● CAMERA ERROR (Device #{cam_idx})"
                    stat_fg = COLOR_RED
                else:
                    stat_text = "● STREAM OFFLINE"
                    stat_fg = TEXT_MUTED

            if stat_text != self._last_stream_status_text or stat_fg != self._last_stream_status_fg:
                self._last_stream_status_text = stat_text
                self._last_stream_status_fg = stat_fg
                self.lbl_stream_status.configure(text=stat_text, fg=stat_fg)

        # Re-schedule next decoupled UI update loop (100ms = 10 Hz)
        if not self._is_closed:
            try:
                self._after_ui_id = self.root.after(100, self._schedule_ui_telemetry_update)
            except Exception:
                pass

    def _handle_object_detection_telemetry(self, results: Dict[str, Any]) -> None:
        """Updates Object Detection telemetry card."""
        if not results.get("detected", False):
            self.item_objects.set_value("ACTIVE (0)", TEXT_MUTED)
            return

        count = results.get("count", 0)
        summary = results.get("summary", "")
        self.item_objects.set_value(f"{count} ({summary[:14]})", ACCENT_BLUE)

    def _handle_person_recognition_telemetry(self, results: Dict[str, Any]) -> None:
        pad_model_name = results.get("pad_model")
        if not pad_model_name:
            pad_mgr = getattr(self.person_engine, "pad_manager", None)
            pad_model_name = getattr(pad_mgr, "active_model_name", "Blink-EAR") if pad_mgr else "Blink-EAR"
        self.item_pad_model.set_value(str(pad_model_name)[:14], ACCENT_BLUE)

        if not results.get("detected", False):
            self.item_person.set_value("ACTIVE (Scan)", ACCENT_BLUE)
            self.item_liveness.set_value("STANDBY", TEXT_MUTED)
            self.item_access.set_value("STANDBY", TEXT_MUTED)
            self.item_pad_latency.set_value("-- ms", TEXT_MUTED)
            return

        has_unknown = results.get("has_unknown", False)
        has_authorized = results.get("has_authorized", False)
        has_verifying = results.get("has_verifying", False)
        has_spoof = results.get("has_spoof", False)
        people = results.get("people", [])

        if people:
            primary_person = people[0]
            name = primary_person.get("name", "Person")
            sim_pct = primary_person.get("match_pct", int(primary_person.get("similarity", 0) * 100))
            live_pct = int(primary_person.get("liveness_confidence", 0.90) * 100)
            is_p_spoof = primary_person.get("is_spoof", False)
            joint_dec: Optional[JointDecision] = primary_person.get("joint_decision")
            pad_res: Optional[PADResult] = primary_person.get("pad_result")

            # Multi-person summary or single-person card
            if len(people) > 1:
                spoof_cnt = sum(1 for p in people if p.get("is_spoof", False))
                auth_cnt = sum(1 for p in people if p.get("joint_decision") and p["joint_decision"].status == JointStatus.AUTHORIZED)
                
                if spoof_cnt > 0:
                    self.item_liveness.set_value(f"⚠️ {spoof_cnt} SPOOF", COLOR_RED)
                    self.item_access.set_value("SPOOF REJECTED", COLOR_RED)
                    self.item_person.set_value(f"⚠️ {len(people)}P ({auth_cnt} Auth, {spoof_cnt} Spf)", COLOR_RED)
                else:
                    self.item_liveness.set_value(f"● {auth_cnt}/{len(people)} LIVE", COLOR_GREEN if auth_cnt > 0 else COLOR_YELLOW)
                    self.item_access.set_value(f"{auth_cnt} AUTH", COLOR_GREEN if auth_cnt > 0 else COLOR_YELLOW)
                    self.item_person.set_value(f"● {len(people)} Persons", COLOR_GREEN if auth_cnt > 0 else COLOR_YELLOW)
            else:
                # Single person display
                if is_p_spoof:
                    self.item_liveness.set_value(f"⚠️ SPOOF ({live_pct}%)", COLOR_RED)
                elif joint_dec and joint_dec.status == JointStatus.LIVENESS_VERIFYING:
                    self.item_liveness.set_value(f"⏳ VERIFYING ({live_pct}%)", COLOR_YELLOW)
                elif joint_dec and joint_dec.status == JointStatus.AUTHORIZED:
                    self.item_liveness.set_value(f"● LIVE ({live_pct}%)", COLOR_GREEN)
                else:
                    self.item_liveness.set_value(f"● LIVE ({live_pct}%)", COLOR_GREEN if pad_res and pad_res.is_live else COLOR_YELLOW)

                if joint_dec:
                    self.item_access.set_value(joint_dec.status.display_label, joint_dec.status.color_hex)
                elif is_p_spoof:
                    self.item_access.set_value("SPOOF REJECTED", COLOR_RED)
                else:
                    self.item_access.set_value("ACCESS PENDING", COLOR_YELLOW)

                if is_p_spoof:
                    self.item_person.set_value(f"⛔ {name} (Spoof)", COLOR_RED)
                elif self.context_engine and self.context_engine.last_decision and self.context_engine.last_decision.decision == "AFTER_HOURS_RESTRICTED_ACCESS_DENIED":
                    self.item_person.set_value(f"{name} (Identified)", COLOR_ORANGE)
                    self.item_access.set_value("DENIED (After Hours)", COLOR_ORANGE)
                elif joint_dec and joint_dec.status == JointStatus.LIVENESS_VERIFYING:
                    self.item_person.set_value(f"⏳ {name} (Verifying...)", COLOR_YELLOW)
                elif joint_dec and joint_dec.status == JointStatus.AUTHORIZED:
                    self.item_person.set_value(f"{name} ({sim_pct}%)", COLOR_GREEN)
                elif has_unknown:
                    self.item_person.set_value("⚠️ UNKNOWN", COLOR_YELLOW)

            # Update Latency Row
            if pad_res:
                self.item_pad_latency.set_value(f"{pad_res.inference_latency_ms:.1f}ms", TEXT_PRIMARY)

    def _handle_context_decision(self, decision: SecurityDecision) -> None:
        """Updates Context Intelligence card, manages risk escalation, and logs debounced security events."""
        ctx = decision.context_summary

        # 1. Update Context Panel Rows
        p_name = ctx.get("person_name", "None")
        if decision.decision == "AFTER_HOURS_RESTRICTED_ACCESS_DENIED":
            self.ctx_item_person.set_value(f"{p_name} (Identified)", COLOR_ORANGE)
        elif ctx.get("is_authorized"):
            self.ctx_item_person.set_value(f"{p_name} (Authorized)", COLOR_GREEN)
        elif decision.decision == "LIVENESS_VERIFYING":
            self.ctx_item_person.set_value(f"{p_name} (Verifying)", COLOR_YELLOW)
        elif ctx.get("has_spoof"):
            self.ctx_item_person.set_value(f"{p_name} (Spoof)", COLOR_RED)
        elif ctx.get("is_unknown"):
            self.ctx_item_person.set_value("Unregistered", COLOR_YELLOW)
        else:
            self.ctx_item_person.set_value("None", TEXT_MUTED)

        obj_summary = ctx.get("object_summary", "None")
        self.ctx_item_object.set_value(obj_summary[:16] if obj_summary != "None" else "None", ACCENT_BLUE if obj_summary != "None" else TEXT_MUTED)

        z_name = ctx.get("zone_name", "Perimeter")
        z_color = COLOR_RED if ctx.get("zone") == "RESTRICTED" else ACCENT_BLUE
        self.ctx_item_zone.set_value(z_name[:12], z_color)

        t_str = ctx.get("time_str", "--:--")
        is_auth_h = ctx.get("is_authorized_hours", True)
        time_status = "NORMAL HOURS" if is_auth_h else "AFTER HOURS"
        if self.context_engine._simulated_time_override is not None:
            time_display = f"{t_str} [SIM {time_status}]"
        else:
            time_display = f"{t_str} ({time_status})"

        self.ctx_item_time.set_value(
            time_display,
            TEXT_PRIMARY if is_auth_h else COLOR_ORANGE,
        )

        dec_color = decision.risk_level.color_hex
        self.ctx_item_decision.set_value(f"{decision.decision} [{decision.risk_level.value}]", dec_color)

        # Update Zone State, Duration & Violation Telemetry (Phase 4.1)
        per_tracks = getattr(decision, "per_track_states", [])
        if per_tracks:
            t0 = per_tracks[0]
            occ = t0.get("zone_occupancy", "OUTSIDE")
            dur = t0.get("duration_inside_zone_seconds", 0.0)
            v_cnt = t0.get("distinct_violation_count", 0)

            if occ == "INSIDE":
                state_disp = f"INSIDE ({int(dur)}s)"
                self.ctx_item_zone_state.set_value(state_disp, COLOR_ORANGE if decision.risk_level != RiskLevel.GREEN else COLOR_GREEN)
            else:
                self.ctx_item_zone_state.set_value("OUTSIDE", TEXT_MUTED)

            v_color = COLOR_RED if v_cnt >= 3 else (COLOR_ORANGE if v_cnt > 0 else TEXT_MUTED)
            self.ctx_item_violations.set_value(f"{v_cnt} Distinct", v_color)
        else:
            self.ctx_item_zone_state.set_value("OUTSIDE", TEXT_MUTED)
            self.ctx_item_violations.set_value("0 Distinct", TEXT_MUTED)

        # Update Active Incident Telemetry
        inc = getattr(decision, "incident", None)
        if inc:
            state_str = inc.state.value if hasattr(inc.state, "value") else str(inc.state)
            inc_color = COLOR_RED if inc.risk_level in (RiskLevel.RED, "RED") else COLOR_ORANGE
            self.ctx_item_incident.set_value(f"{inc.incident_id} [{state_str}]", inc_color)
        elif decision.risk_level in (RiskLevel.ORANGE, RiskLevel.RED):
            self.ctx_item_incident.set_value(f"INCIDENT [{decision.risk_level.value}]", dec_color)
        else:
            self.ctx_item_incident.set_value("None", TEXT_MUTED)

        # Update Analytics Counters (Phase 5)
        if hasattr(self.context_engine, "security_analytics") and self.context_engine.security_analytics:
            metrics = self.context_engine.security_analytics.get_metrics("today")
            sp_val = f"{metrics.spoof_attempts}"
            sp_col = COLOR_RED if metrics.spoof_attempts > 0 else TEXT_MUTED
            self.ctx_item_analytics_spoofs.set_value(sp_val, sp_col)

            ah_val = f"{metrics.after_hours_violations}"
            ah_col = COLOR_ORANGE if metrics.after_hours_violations > 0 else TEXT_MUTED
            self.ctx_item_analytics_after_hours.set_value(ah_val, ah_col)

        # Clamp visible reason text to 2 lines max with ellipsis for stable card height
        reason_text = decision.reason
        if len(reason_text) > 115:
            display_reason = reason_text[:112] + "..."
        else:
            display_reason = reason_text

        self.lbl_ctx_reason.set_text(
            display_reason,
            fg=dec_color if decision.risk_level != RiskLevel.GREEN else TEXT_PRIMARY,
        )

        # 2. Update Master Risk Banner
        self.risk_banner.update_risk(decision.risk_level, decision.reason)

        # 3. Handle Debounced Security Event Dispatching & Evidence Snapshot
        if decision.risk_level in (RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED):
            rule_sig = f"{decision.decision}_{p_name}_{obj_summary}"
            if self.context_engine.should_trigger_alert(rule_sig):
                snap_path = None
                if decision.evidence_required and self.camera.is_running:
                    snap_path = self.camera.capture_snapshot(filename_prefix=f"evidence_{decision.decision.lower()}")

                meta_dict = {
                    "person_identity": p_name,
                    "zone": ctx.get("zone_name", "Standard Perimeter"),
                    "current_time": ctx.get("time_str", "--:--"),
                    "authorized_hours_status": "NORMAL HOURS" if ctx.get("is_authorized_hours") else "AFTER HOURS",
                    "authorized_hours_window": f"{self.context_engine.hours_start}–{self.context_engine.hours_end}",
                    "final_decision": decision.decision,
                    "risk_level": decision.risk_level.value,
                    "traceable_reason": decision.reason,
                    "is_access_granted": decision.is_access_granted,
                }

                self.logger.log_event(
                    event_type=decision.event_type,
                    risk_level=decision.risk_level,
                    description=decision.reason,
                    evidence_frame_path=snap_path,
                    person_name=p_name if p_name != "None" else None,
                    object_summary=obj_summary if obj_summary != "None" else None,
                    metadata_json=json.dumps(meta_dict),
                )

        # Also log verified authorized entry once per arrival
        elif decision.risk_level == RiskLevel.GREEN and ctx.get("is_authorized"):
            if self.person_engine.should_trigger_alert(f"auth_arrival_{p_name}"):
                self.logger.log_event(
                    event_type=EventType.AUTHORIZED_PERSON,
                    risk_level=RiskLevel.GREEN,
                    description=f"Authorized personnel arrival verified: {p_name}.",
                    person_name=p_name,
                )

    def _on_ack_incident_clicked(self) -> None:
        """Handles operator click on ACK incident button."""
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
            if inc and (hasattr(inc, "state") and inc.state.value != "RESOLVED"):
                self.context_engine.incident_manager.acknowledge_incident(inc.incident_id, operator="Dashboard Operator")
                self.ctx_item_incident.set_value(f"{inc.incident_id} [ACKNOWLEDGED]", COLOR_YELLOW)
                self.logger.log_event(
                    event_type=EventType.INCIDENT_ACKNOWLEDGED,
                    risk_level=RiskLevel.YELLOW,
                    description=f"Incident {inc.incident_id} acknowledged by Dashboard Operator.",
                )

    def _on_resolve_incident_clicked(self) -> None:
        """Handles operator click on RESOLVE incident button."""
        if self.context_engine and hasattr(self.context_engine, "incident_manager"):
            inc = self.context_engine.incident_manager.get_latest_incident()
            if inc:
                self.context_engine.incident_manager.resolve_incident(inc.incident_id, operator="Dashboard Operator")
                self.ctx_item_incident.set_value(f"{inc.incident_id} [RESOLVED]", COLOR_GREEN)
                self.logger.log_event(
                    event_type=EventType.INCIDENT_RESOLVED,
                    risk_level=RiskLevel.GREEN,
                    description=f"Incident {inc.incident_id} resolved by Dashboard Operator.",
                )

    # --- Voice Assistant UI Callbacks & Handlers ---

    def _on_voice_status_changed(self, status: str) -> None:
        """Thread-safe UI callback for voice assistant status changes."""
        def update_ui():
            if status == "LISTENING":
                self.item_voice_status.set_value("● LISTENING (Speak now...)", COLOR_YELLOW)
                self.btn_ptt.configure(bg="#D97706", text="🎙️ LISTENING...")
            elif status == "PROCESSING":
                self.item_voice_status.set_value("● PROCESSING QUERY...", ACCENT_BLUE)
                self.btn_ptt.configure(bg="#2563EB", text="⏳ PROCESSING...")
            elif status == "SPEAKING":
                self.item_voice_status.set_value("● SPEAKING RESPONSE...", COLOR_GREEN)
                self.btn_ptt.configure(bg="#2563EB", text="🎤 PTT (3.5s)")
            else:
                self.item_voice_status.set_value("● READY / STANDBY", TEXT_MUTED)
                self.btn_ptt.configure(bg="#2563EB", text="🎤 PTT (3.5s)")
        self.root.after(0, update_ui)

    def _on_ptt_clicked(self) -> None:
        """Triggers Push-to-Talk audio recording and query execution."""
        def on_complete(transcription: str, response: str):
            def update_ui():
                self.item_voice_command.set_value(f"'{transcription}'", TEXT_PRIMARY)
                self.lbl_voice_resp.set_text(response)
            self.root.after(0, update_ui)

        self.voice_assistant.listen_and_process(duration_seconds=3.5, on_complete=on_complete)

    def _on_mute_toggled(self) -> None:
        """Toggles audio mute state."""
        self.voice_assistant.mute_audio = self.mute_var.get()

    def _on_send_text_query(self) -> None:
        """Executes text-query fallback."""
        query = self.ent_voice_query.get().strip()
        if not query:
            return

        self.item_voice_command.set_value(f"'{query}'", TEXT_PRIMARY)
        response = self.voice_assistant.process_command(query)
        self.lbl_voice_resp.set_text(response)
        self.voice_assistant.speak(response)

    # --- Face Registration UI Dialog ---

    def _open_face_registration_dialog(self) -> None:
        """Opens a modal dialog to register a person's face into SQLite database."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Person Face Profile")
        dialog.geometry("480x360")
        dialog.configure(bg=CARD_BG)
        dialog.transient(self.root)
        dialog.grab_set()

        # Header
        lbl_head = tk.Label(
            dialog,
            text="PERSON FACE REGISTRATION",
            font=FONT_SUBTITLE,
            bg=CARD_HEADER_BG,
            fg=TEXT_PRIMARY,
            pady=8,
        )
        lbl_head.pack(fill="x")

        form = tk.Frame(dialog, bg=CARD_BG, padx=20, pady=14)
        form.pack(fill="both", expand=True)

        # Full Name Field
        tk.Label(form, text="Full Name:", font=FONT_BODY_BOLD, bg=CARD_BG, fg=TEXT_PRIMARY).grid(row=0, column=0, sticky="w", pady=6)
        ent_name = tk.Entry(form, font=FONT_BODY, bg="#0F172A", fg=TEXT_PRIMARY, insertbackground="#FFFFFF")
        ent_name.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=6)
        ent_name.insert(0, "Hunter")

        # Username Field
        tk.Label(form, text="Username:", font=FONT_BODY_BOLD, bg=CARD_BG, fg=TEXT_PRIMARY).grid(row=1, column=0, sticky="w", pady=6)
        ent_user = tk.Entry(form, font=FONT_BODY, bg="#0F172A", fg=TEXT_PRIMARY, insertbackground="#FFFFFF")
        ent_user.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=6)
        ent_user.insert(0, "hunter")

        # Role Dropdown
        tk.Label(form, text="Security Role:", font=FONT_BODY_BOLD, bg=CARD_BG, fg=TEXT_PRIMARY).grid(row=2, column=0, sticky="w", pady=6)
        role_var = tk.StringVar(value="Security Lead")
        combo_role = ttk.Combobox(form, textvariable=role_var, values=["Administrator", "Security Lead", "Security Officer", "Staff", "Authorized User"], state="readonly")
        combo_role.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=6)

        form.columnconfigure(1, weight=1)

        # Status feedback label
        lbl_status = tk.Label(dialog, text="Select registration capture source below:", font=FONT_SMALL, bg=CARD_BG, fg=TEXT_SECONDARY)
        lbl_status.pack(pady=4)

        btn_box = tk.Frame(dialog, bg=CARD_BG, pady=10)
        btn_box.pack()

        def register_from_live():
            name = ent_name.get().strip()
            username = ent_user.get().strip()
            role = role_var.get()
            if not name or not username:
                messagebox.showwarning("Incomplete", "Please enter both full name and username.", parent=dialog)
                return

            if not self.camera.is_running:
                messagebox.showwarning("Camera Offline", "Please start the camera stream first.", parent=dialog)
                return

            success, frame = self.camera.get_frame()
            if not success or frame is None:
                messagebox.showerror("Error", "Could not capture frame from camera.", parent=dialog)
                return

            existing = self.db.get_user_by_username(username)
            if existing:
                u_id = existing["id"]
            else:
                u_id = self.db.create_user(username=username, full_name=name, role=role, access_level=2)

            reg_ok, msg = self.person_engine.register_face_from_image(user_id=u_id, image_bgr=frame)
            if reg_ok:
                self.item_registered_faces.set_value(f"{self.person_engine.known_face_count}")
                self.logger.log_event(
                    event_type=EventType.AUTHORIZED_PERSON,
                    risk_level=RiskLevel.GREEN,
                    description=f"New face profile registered: {name} ({role})",
                    person_name=name,
                )
                messagebox.showinfo("Success", f"Face registered successfully for {name}!", parent=dialog)
                dialog.destroy()
            else:
                lbl_status.configure(text=f"❌ {msg}", fg=COLOR_RED)

        def register_from_file():
            name = ent_name.get().strip()
            username = ent_user.get().strip()
            role = role_var.get()
            if not name or not username:
                messagebox.showwarning("Incomplete", "Please enter both full name and username.", parent=dialog)
                return

            file_path = filedialog.askopenfilename(
                title="Select Person Face Image",
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")],
                parent=dialog,
            )
            if not file_path:
                return

            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "Failed to load image file.", parent=dialog)
                return

            existing = self.db.get_user_by_username(username)
            if existing:
                u_id = existing["id"]
            else:
                u_id = self.db.create_user(username=username, full_name=name, role=role, access_level=2)

            reg_ok, msg = self.person_engine.register_face_from_image(user_id=u_id, image_bgr=img, source_image_path=file_path)
            if reg_ok:
                self.item_registered_faces.set_value(f"{self.person_engine.known_face_count}")
                self.logger.log_event(
                    event_type=EventType.AUTHORIZED_PERSON,
                    risk_level=RiskLevel.GREEN,
                    description=f"New face profile registered from file: {name} ({role})",
                    person_name=name,
                )
                messagebox.showinfo("Success", f"Face registered successfully for {name}!", parent=dialog)
                dialog.destroy()
            else:
                lbl_status.configure(text=f"❌ {msg}", fg=COLOR_RED)

        tk.Button(
            btn_box,
            text="📷 Capture from Live Stream",
            font=FONT_BODY_BOLD,
            bg=BTN_START_BG,
            fg="#FFFFFF",
            padx=10,
            pady=4,
            command=register_from_live,
        ).pack(side="left", padx=5)

        tk.Button(
            btn_box,
            text="📁 Browse Image File...",
            font=FONT_BODY,
            bg=BTN_NEUTRAL_BG,
            fg=TEXT_PRIMARY,
            padx=10,
            pady=4,
            command=register_from_file,
        ).pack(side="left", padx=5)

    # --- Event Queue & Logging UI Integration ---

    def _format_risk_display(self, risk_level: str) -> str:
        """Formats risk level with clear visual emoji badge and text label."""
        r = (risk_level or "GREEN").upper()
        if r == "RED":
            return "🔴 INCIDENT (RED)"
        elif r == "ORANGE":
            return "🟠 HIGH RISK (ORANGE)"
        elif r == "YELLOW":
            return "🟡 ATTENTION (YELLOW)"
        else:
            return "🟢 NORMAL (GREEN)"

    def _on_event_logged(self, event_payload: Dict[str, Any]) -> None:
        """Callback from EventLogger: push to thread-safe queue."""
        self.event_queue.put(event_payload)

    def _schedule_event_pump(self) -> None:
        """Pumps events from background threads into the Tkinter UI safely."""
        if self._is_closed:
            return

        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self._insert_event_to_tree(event, prepend=True)
                self._update_event_count_stat()
            except queue.Empty:
                break

        if not self._is_closed:
            try:
                self._after_pump_id = self.root.after(100, self._schedule_event_pump)
            except Exception:
                pass

    def _insert_event_to_tree(self, event: Dict[str, Any], prepend: bool = True) -> None:
        """Inserts an event row into the Treeview with formatted risk and high-contrast styling."""
        event_id = event.get("id", "--")
        ts = event.get("timestamp", "")
        raw_risk = (event.get("risk_level") or "GREEN").upper()
        etype = event.get("event_type", "INFO")
        desc = event.get("description", "")

        risk_formatted = self._format_risk_display(raw_risk)
        values = (event_id, ts, risk_formatted, etype, desc)

        tag = raw_risk if raw_risk in ("GREEN", "YELLOW", "ORANGE", "RED") else "GREEN"

        if prepend:
            item = self.tree.insert("", 0, values=values, tags=(tag,))
            self.tree.see(item)
        else:
            self.tree.insert("", "end", values=values, tags=(tag,))

    def _load_recent_events(self) -> None:
        """Loads historical events from SQLite into table (newest first)."""
        self._clear_treeview()
        events = self.db.get_recent_security_events(limit=MAX_RECENT_EVENTS_DISPLAY)
        # events is ordered ORDER BY id DESC from database
        for ev in events:
            self._insert_event_to_tree(ev, prepend=False)
        self._update_event_count_stat()
        children = self.tree.get_children()
        if children:
            self.tree.see(children[0])

    def _clear_treeview(self) -> None:
        """Clears rows in the table UI."""
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _update_event_count_stat(self) -> None:
        """Updates the total events telemetry number from SQLite database."""
        total_count = self.db.get_total_event_count()
        self.item_db_records.set_value(str(total_count), ACCENT_BLUE)

    def _on_device_auto_selected(self, device_index: int) -> None:
        """Thread-safe UI update when camera manager auto-detects or switches active device."""
        def update_ui():
            self.camera_index_var.set(device_index)
            if self.camera.is_running:
                if self.camera.is_fallback_mode:
                    self.lbl_stream_status.configure(
                        text="● SIMULATION (No Camera)",
                        fg=COLOR_YELLOW,
                    )
                else:
                    self.lbl_stream_status.configure(
                        text=f"● LIVE (Camera #{device_index})",
                        fg=COLOR_GREEN,
                    )
        self.root.after(0, update_ui)

    # --- Camera & Simulation Actions ---

    def _toggle_camera(self) -> None:
        """Starts or stops the camera stream on operator button click."""
        if self.camera.is_running and self.camera.status in (CameraStatus.REAL, CameraStatus.SYNTHETIC):
            # 1. Clean stop requested by operator
            self.camera.stop()
            self.btn_toggle_cam.configure(
                text="▶ START CAMERA",
                bg=BTN_START_BG,
                activebackground=COLOR_GREEN,
            )
            self.lbl_stream_status.configure(
                text="● STREAM OFFLINE",
                fg=TEXT_MUTED,
            )
            self.lbl_fps.configure(text="FPS: 0.0")
            self.item_person.set_value("STANDBY", TEXT_MUTED)
            self.item_objects.set_value("STANDBY", TEXT_MUTED)
            self.item_liveness.set_value("STANDBY", TEXT_MUTED)
            self.item_access.set_value("STANDBY", TEXT_MUTED)
            self.item_pad_latency.set_value("-- ms", TEXT_MUTED)
            self.ctx_item_person.set_value("None", TEXT_MUTED)
            self.ctx_item_object.set_value("None", TEXT_MUTED)
            self.ctx_item_decision.set_value("STANDBY", TEXT_MUTED)
            self.lbl_ctx_reason.set_text("Camera inactive - Stream stopped.", fg=TEXT_MUTED)
            self._render_placeholder_frame("Camera Inactive - Click [Start Camera] below")
            self.logger.log_event(
                event_type=EventType.CAMERA_STOPPED,
                risk_level=RiskLevel.GREEN,
                description="Webcam stream stopped by operator.",
            )
        else:
            # 2. Start requested by operator
            cam_idx = self.camera_index_var.get()
            choice = self.demo_target_var.get()

            # Determine mode based on simulation dropdown
            if "Hunter" in choice or "Alice" in choice:
                mode = CameraMode.SYNTHETIC
                self.camera.set_demo_mode("authorized")
            elif "Unknown" in choice:
                mode = CameraMode.SYNTHETIC
                self.camera.set_demo_mode("unknown")
            else:
                mode = CameraMode.PHYSICAL
                self.camera.set_demo_mode("none")

            opened = self.camera.start(
                camera_index=cam_idx,
                mode=mode,
                allow_synthetic_fallback=False,
                auto_discover=True,
            )

            if opened and self.camera.status == CameraStatus.REAL and self.camera.is_running:
                self.btn_toggle_cam.configure(
                    text="⏹ STOP CAMERA",
                    bg=BTN_STOP_BG,
                    activebackground=COLOR_RED,
                )
                self.lbl_stream_status.configure(
                    text=f"● LIVE (Camera #{self.camera.camera_index})",
                    fg=COLOR_GREEN,
                )
                self.logger.log_event(
                    event_type=EventType.CAMERA_STARTED,
                    risk_level=RiskLevel.GREEN,
                    description=f"Physical webcam stream activated on device {self.camera.camera_index}.",
                )
            elif opened and self.camera.status == CameraStatus.SYNTHETIC and self.camera.is_running:
                self.btn_toggle_cam.configure(
                    text="⏹ STOP CAMERA",
                    bg=BTN_STOP_BG,
                    activebackground=COLOR_RED,
                )
                self.lbl_stream_status.configure(
                    text=f"● SIMULATION ({self.camera.demo_mode.upper()})",
                    fg=COLOR_YELLOW,
                )
                self.logger.log_event(
                    event_type=EventType.CAMERA_STARTED,
                    risk_level=RiskLevel.GREEN,
                    description=f"Synthetic simulator stream activated (Target: {self.camera.demo_mode}).",
                )
            else:
                self.btn_toggle_cam.configure(
                    text="▶ RETRY CAMERA",
                    bg=BTN_START_BG,
                    activebackground=COLOR_GREEN,
                )
                self.lbl_stream_status.configure(
                    text=f"● CAMERA ERROR (Device #{cam_idx})",
                    fg=COLOR_RED,
                )
                self.lbl_ctx_reason.configure(
                    text=f"Camera hardware unavailable: {self.camera.error_message or 'Failed to open physical webcam device.'}",
                    fg=COLOR_RED,
                )
                self._render_placeholder_frame(f"Camera Error - Device #{cam_idx} Failed to Open")

    def _on_camera_index_changed(self, event=None) -> None:
        """Handles manual camera index switch."""
        new_idx = self.camera_index_var.get()
        if self.camera.is_running and self.camera.status == CameraStatus.REAL:
            self.camera.switch_camera(new_idx)

    def _on_demo_target_changed(self, event=None) -> None:
        """Handles demo target selection dropdown."""
        choice = self.demo_target_var.get()
        if "Hunter" in choice:
            self.camera.set_demo_mode("authorized")
            if self.camera.is_running:
                self.lbl_stream_status.configure(
                    text=f"● SIMULATION (AUTHORIZED)",
                    fg=COLOR_YELLOW,
                )
        elif "Unknown" in choice:
            self.camera.set_demo_mode("unknown")
            if self.camera.is_running:
                self.lbl_stream_status.configure(
                    text=f"● SIMULATION (UNKNOWN)",
                    fg=COLOR_YELLOW,
                )
        else:
            self.camera.set_demo_mode("none")
            if self.camera.is_running:
                self.camera.stop()
                cam_idx = self.camera_index_var.get()
                self.camera.start(camera_index=cam_idx, mode=CameraMode.PHYSICAL, allow_synthetic_fallback=False)
                if self.camera.status == CameraStatus.REAL:
                    self.lbl_stream_status.configure(
                        text=f"● LIVE (Camera #{self.camera.camera_index})",
                        fg=COLOR_GREEN,
                    )

    def _on_scenario_changed(self, event=None) -> None:
        """Switches scenario override for Context & Security Intelligence testing."""
        choice = self.scenario_var.get()
        if "After-Hours" in choice:
            self.context_engine.set_time_override(dtime(22, 30))
            self.context_engine.set_zone_override(None)
        elif "Daytime" in choice:
            self.context_engine.set_time_override(dtime(14, 0))
            self.context_engine.set_zone_override(None)
        elif "Restricted" in choice:
            self.context_engine.set_zone_override("RESTRICTED")
            self.context_engine.set_time_override(None)
        else:
            self.context_engine.set_time_override(None)
            self.context_engine.set_zone_override(None)

    def _take_snapshot(self) -> None:
        """Captures evidence snapshot from current video stream."""
        if not self.camera.is_running:
            messagebox.showinfo("Camera Inactive", "Start the camera stream to capture a snapshot.", parent=self.root)
            return

        filepath = self.camera.capture_snapshot(filename_prefix="evidence_manual")
        if filepath:
            self.logger.log_event(
                event_type=EventType.SECURITY_ALERT,
                risk_level=RiskLevel.YELLOW,
                description=f"Manual snapshot captured and preserved: {filepath}",
                evidence_frame_path=filepath,
            )
            messagebox.showinfo("Snapshot Captured", f"Saved evidence snapshot:\n{filepath}", parent=self.root)
        else:
            messagebox.showerror("Error", "Could not capture snapshot frame.", parent=self.root)

    def _set_risk_level(self, level: RiskLevel) -> None:
        """Manually updates risk banner and logs incident for testing."""
        self.current_risk_level = level
        self.risk_banner.update_risk(level)
        self.logger.log_event(
            event_type=EventType.SECURITY_ALERT,
            risk_level=level,
            description=f"Operator manually adjusted threat level to {level.label}",
        )

    def _on_close(self) -> None:
        """Clean shutdown handler."""
        self._is_closed = True
        self._ai_stop_event.set()
        self._ai_frame_event.set()
        if self._after_video_id:
            try:
                self.root.after_cancel(self._after_video_id)
            except Exception:
                pass
        if self._after_ui_id:
            try:
                self.root.after_cancel(self._after_ui_id)
            except Exception:
                pass
        if self._after_pump_id:
            try:
                self.root.after_cancel(self._after_pump_id)
            except Exception:
                pass
        if self.camera.is_running:
            self.camera.stop()
        if self._ai_worker_thread and self._ai_worker_thread.is_alive():
            try:
                self._ai_worker_thread.join(timeout=0.5)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass
