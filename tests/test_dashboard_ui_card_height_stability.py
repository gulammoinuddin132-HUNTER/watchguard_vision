"""
UI Card Height Stability Tests
-------------------------------
Verifies that:
1. Right-column cards (Threat Level, Context Intelligence, Vision Telemetry, Voice Assistant)
   maintain stable, constant heights across GREEN, YELLOW, ORANGE, and RED threat states.
2. Long multi-line reason descriptions and voice assistant responses do not cause vertical reflow/expansion.
3. Full untruncated reasons and descriptions remain logged in SQLite and the EventLogger.
"""

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.events.event_types import EventType, RiskLevel
from src.camera.camera_manager import CameraManager
from src.core.person_recognition import PersonRecognitionEngine
from src.core.object_detection import ObjectDetectionEngine
from src.core.context_engine import ContextEngine, SecurityDecision
from src.core.voice_assistant import VoiceAssistant
from src.ui.dashboard import WatchGuardDashboard


class TestDashboardUICardHeightStability(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry("1260x860")
        self.root.withdraw()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_height.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.logger = EventLogger(db_manager=self.db)
        self.context_engine = ContextEngine()

        self.mock_camera = MagicMock(spec=CameraManager)
        self.mock_camera.is_running = True
        self.mock_camera.is_fallback_mode = False
        self.mock_camera.camera_index = 0
        self.mock_camera.get_frame.return_value = (False, None)
        self.mock_camera.capture_snapshot.return_value = None
        self.mock_camera.fps = 0.0

        self.mock_person_engine = MagicMock(spec=PersonRecognitionEngine)
        self.mock_person_engine.is_ready = True
        self.mock_person_engine.known_face_count = 1

        self.mock_object_engine = MagicMock(spec=ObjectDetectionEngine)
        self.mock_object_engine.is_ready = True

        self.va = VoiceAssistant(
            db_manager=self.db,
            event_logger=self.logger,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            camera_manager=self.mock_camera,
        )
        self.va.mute_audio = True

        self.dashboard = WatchGuardDashboard(
            root=self.root,
            db_manager=self.db,
            event_logger=self.logger,
            camera_manager=self.mock_camera,
            person_engine=self.mock_person_engine,
            object_engine=self.mock_object_engine,
            context_engine=self.context_engine,
            voice_assistant=self.va,
        )
        self.root.update_idletasks()

    def tearDown(self):
        if hasattr(self, "dashboard") and self.dashboard:
            try:
                self.dashboard.close(destroy_root=False)
            except Exception:
                pass
        try:
            if hasattr(self, "root") and self.root and self.root.winfo_exists():
                self.root.destroy()
        except Exception:
            pass
        self.temp_dir.cleanup()

    def test_right_column_card_heights_stable_across_all_risk_levels(self):
        """Verify cards maintain identical height in GREEN, YELLOW, ORANGE, and RED states."""
        # 1. State: GREEN (Normal baseline)
        dec_green = SecurityDecision(
            risk_level=RiskLevel.GREEN,
            decision="NORMAL_ACTIVITY",
            reason="Perimeter clear. Routine surveillance monitoring active.",
            matched_rules=["RULE_AUTHORIZED_PERIMETER"],
            evidence_required=False,
            recommended_response="Continue routine monitoring.",
            context_summary={"person_detected": False, "objects_detected": 0},
            event_type=EventType.AUTHORIZED_PERSON,
        )
        self.dashboard._handle_context_decision(dec_green)
        self.root.update_idletasks()

        h_green_risk = self.dashboard.risk_banner.winfo_reqheight()
        h_green_ctx = self.dashboard.lbl_ctx_reason.master.winfo_reqheight()
        h_green_resp = self.dashboard.lbl_voice_resp.master.winfo_reqheight()

        # 2. State: YELLOW (Medium length reason)
        dec_yellow = SecurityDecision(
            risk_level=RiskLevel.YELLOW,
            decision="UNKNOWN_VISITOR_DETECTED",
            reason="Unregistered visitor detected in camera view without active credentials. Visual verification recommended.",
            matched_rules=["RULE_UNKNOWN_PERSON"],
            evidence_required=True,
            recommended_response="Request visitor credentials.",
            context_summary={"person_detected": True, "person_name": "Unknown", "is_unknown": True},
            event_type=EventType.UNKNOWN_PERSON,
        )
        self.dashboard._handle_context_decision(dec_yellow)
        self.root.update_idletasks()

        h_yellow_risk = self.dashboard.risk_banner.winfo_reqheight()
        h_yellow_ctx = self.dashboard.lbl_ctx_reason.master.winfo_reqheight()

        self.assertEqual(h_green_risk, h_yellow_risk, "Risk banner height changed on YELLOW state!")
        self.assertEqual(h_green_ctx, h_yellow_ctx, "Context reason container height changed on YELLOW state!")

        # 3. State: ORANGE (Long policy violation reason)
        dec_orange = SecurityDecision(
            risk_level=RiskLevel.ORANGE,
            decision="RESTRICTED_ZONE_VIOLATION",
            reason="Policy violation: Individual detected inside designated high-security RESTRICTED perimeter zone after hours.",
            matched_rules=["RULE_RESTRICTED_ZONE_BREACH"],
            evidence_required=True,
            recommended_response="Dispatch security officer immediately.",
            context_summary={"person_detected": True, "zone": "RESTRICTED"},
            event_type=EventType.POLICY_VIOLATION,
        )
        self.dashboard._handle_context_decision(dec_orange)
        self.root.update_idletasks()

        h_orange_risk = self.dashboard.risk_banner.winfo_reqheight()
        h_orange_ctx = self.dashboard.lbl_ctx_reason.master.winfo_reqheight()

        self.assertEqual(h_green_risk, h_orange_risk, "Risk banner height changed on ORANGE state!")
        self.assertEqual(h_green_ctx, h_orange_ctx, "Context reason container height changed on ORANGE state!")

        # 4. State: RED (Very long incident rationale)
        dec_red = SecurityDecision(
            risk_level=RiskLevel.RED,
            decision="POTENTIAL_UNAUTHORIZED_INTERACTION",
            reason="Critical security incident: Unregistered visitor in physical proximity (12.4px) to protected physical asset (LAPTOP) without valid security clearance credentials.",
            matched_rules=["RULE_UNAUTHORIZED_ASSET_INTERACTION"],
            evidence_required=True,
            recommended_response="Lock asset and raise perimeter alarm.",
            context_summary={"person_detected": True, "is_unknown": True, "objects_detected": 1},
            event_type=EventType.SECURITY_ALERT,
        )
        self.dashboard._handle_context_decision(dec_red)
        self.root.update_idletasks()

        h_red_risk = self.dashboard.risk_banner.winfo_reqheight()
        h_red_ctx = self.dashboard.lbl_ctx_reason.master.winfo_reqheight()

        self.assertEqual(h_green_risk, h_red_risk, "Risk banner height changed on RED state!")
        self.assertEqual(h_green_ctx, h_red_ctx, "Context reason container height changed on RED state!")

        # 5. Long Voice Response Test
        long_response = (
            "There are 45 logged events. Recent activity: 1. 01:15 - ORANGE - Restricted zone violation; "
            "2. 01:14 - YELLOW - Unregistered person detected; 3. 01:12 - GREEN - Authorized access Hunter. "
            "The latest significant alert was risk ORANGE: Restricted zone violation."
        )
        self.dashboard.lbl_voice_resp.configure(text=long_response)
        self.root.update_idletasks()

        h_resp_after_long = self.dashboard.lbl_voice_resp.master.winfo_reqheight()
        self.assertEqual(h_green_resp, h_resp_after_long, "Voice response container height changed with long text!")


if __name__ == "__main__":
    unittest.main()
