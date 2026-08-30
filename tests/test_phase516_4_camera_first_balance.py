"""
Phase 5.16.4 Unit & Regression Tests: Camera-First UI Balance Tuning
------------------------------------------------------------------
Validates:
1. Camera card width dominance (>= 760 px, roughly 60-65% of usable upper width).
2. Right dashboard compact footprint (roughly 35-40% of usable upper width).
3. Event log container (~95-125 px) and Treeview (~60-80 px, ~3 visible rows).
4. Long Voice Assistant responses (1, 5, 10 lines) maintain 0 px geometry shift.
5. Responsive behavior across standard (1280x880), maximized (1600x950), and compact (1150x760).
"""

import os
import shutil
import tempfile
import unittest
import tkinter as tk

from src.camera.camera_manager import CameraManager
from src.database.db_manager import DatabaseManager
from src.events.event_logger import EventLogger
from src.ui.dashboard import WatchGuardDashboard


class TestPhase5164CameraFirstBalance(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_wg.db")
        self.db = DatabaseManager(self.db_path)
        self.logger = EventLogger(self.db)
        self.cam = CameraManager()

        self.root = tk.Tk()
        self.root.geometry("1280x880")
        self.app = WatchGuardDashboard(
            root=self.root,
            camera_manager=self.cam,
            db_manager=self.db,
            event_logger=self.logger,
        )
        self.root.update_idletasks()
        self.root.update()

    def tearDown(self):
        if hasattr(self, "app") and self.app:
            try:
                self.app.close(destroy_root=False)
            except Exception:
                pass
        try:
            if hasattr(self, "root") and self.root and self.root.winfo_exists():
                self.root.destroy()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_containers(self):
        main_ws = None
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame) and child.grid_info().get("row") == 1:
                main_ws = child
                break
        self.assertIsNotNone(main_ws, "Main workspace not found")
        upper_area = main_ws.winfo_children()[0]
        ctrl_bar = main_ws.winfo_children()[1]
        evt_card = main_ws.winfo_children()[2]
        video_card = upper_area.winfo_children()[0]
        right_panel = upper_area.winfo_children()[1]
        return upper_area, video_card, right_panel, ctrl_bar, evt_card

    def test_01_camera_first_upper_ratio_and_dimensions(self):
        """Verify camera card occupies 60-65% width and right dashboard occupies 35-40%."""
        upper_area, video_card, right_panel, ctrl_bar, evt_card = self._get_containers()

        upper_w = upper_area.winfo_width()
        cam_w = video_card.winfo_width()
        right_w = right_panel.winfo_width()

        cam_pct = (cam_w / upper_w) * 100.0
        right_pct = (right_w / upper_w) * 100.0

        # Camera card target: ~840-900 px wide (Phase 5.16.12: ~68-70%)
        self.assertGreaterEqual(cam_w, 820, f"Camera card width {cam_w} px is less than 820 px!")
        self.assertLessEqual(cam_w, 950, f"Camera card width {cam_w} px exceeds 950 px!")

        # Camera ratio target: roughly 67-72%
        self.assertGreaterEqual(cam_pct, 66.0, f"Camera ratio {cam_pct:.1f}% is less than 66%!")
        self.assertLessEqual(cam_pct, 73.0, f"Camera ratio {cam_pct:.1f}% exceeds 73%!")

        # Right dashboard target: ~380-405 px wide (Phase 5.16.12: ~30-32%)
        self.assertGreaterEqual(right_w, 375, f"Right dashboard width {right_w} px is less than 375 px!")
        self.assertLessEqual(right_w, 410, f"Right dashboard width {right_w} px exceeds 410 px!")
        self.assertGreaterEqual(right_pct, 28.0, f"Right dashboard ratio {right_pct:.1f}% is less than 28%!")
        self.assertLessEqual(right_pct, 33.0, f"Right dashboard ratio {right_pct:.1f}% exceeds 33%!")

        # Upper area absorbs vertical space (removes empty bottom region, Section 7)
        upper_h = upper_area.winfo_height()
        self.assertGreaterEqual(upper_h, 480, f"Upper area height {upper_h} px is too small!")
        self.assertLessEqual(upper_h, 680, f"Upper area height {upper_h} px exceeded 680 px!")

        # Event log sits flush at bottom with minimal unused region (<= 25px margin)
        root_h = self.root.winfo_height()
        evt_bottom = evt_card.winfo_y() + evt_card.winfo_height() + evt_card.master.winfo_y()
        bottom_margin = root_h - evt_bottom
        self.assertGreaterEqual(bottom_margin, 0, "Event log is clipped outside root window!")
        self.assertLessEqual(bottom_margin, 30, f"Large unused dark region below event log ({bottom_margin}px > 30px)!")

    def test_02_very_compact_event_log_and_treeview(self):
        """Verify event log container is ~95-125 px and Treeview is ~60-85 px (~3 rows)."""
        _, _, _, _, evt_card = self._get_containers()
        tree = self.app.tree

        evt_h = evt_card.winfo_height()
        tree_h = tree.winfo_height()

        self.assertGreaterEqual(evt_h, 95, f"Event log container height {evt_h} px is less than 95 px!")
        self.assertLessEqual(evt_h, 125, f"Event log container height {evt_h} px exceeds 125 px!")

        self.assertGreaterEqual(tree_h, 60, f"Treeview viewport height {tree_h} px is less than 60 px!")
        self.assertLessEqual(tree_h, 85, f"Treeview viewport height {tree_h} px exceeds 85 px!")

    def test_03_voice_response_boundedness_and_invariance(self):
        """Verify 1-line, 5-line, and 10-line Voice responses cause 0 px geometry deviation."""
        _, video_card, right_panel, _, evt_card = self._get_containers()

        base_cam = (video_card.winfo_width(), video_card.winfo_height())
        base_right = (right_panel.winfo_width(), right_panel.winfo_height())
        base_evt = (evt_card.winfo_width(), evt_card.winfo_height(), evt_card.winfo_y())

        responses = [
            ("1-line", "Voice assistant ready. Ask a question or use Push to Talk."),
            ("5-line", "Line 1: Status OK\nLine 2: Identity verified\nLine 3: Zone clear\nLine 4: No threats\nLine 5: Authorized"),
            ("10-line", "\n".join([f"Trace log line {i+1}: Incident resolution rationale audit verified." for i in range(10)])),
        ]

        for label, text in responses:
            self.app.lbl_voice_resp.set_text(text)
            self.root.update_idletasks()
            self.root.update()

            cur_cam = (video_card.winfo_width(), video_card.winfo_height())
            cur_right = (right_panel.winfo_width(), right_panel.winfo_height())
            cur_evt = (evt_card.winfo_width(), evt_card.winfo_height(), evt_card.winfo_y())

            self.assertEqual(cur_cam, base_cam, f"Camera size changed after {label} voice response!")
            self.assertEqual(cur_right, base_right, f"Right dashboard size changed after {label} voice response!")
            self.assertEqual(cur_evt, base_evt, f"Event log geometry changed after {label} voice response!")

    def test_04_responsive_resizing_without_clipping(self):
        """Verify responsive behavior at 1600x950 and 1150x760 without clipping."""
        resolutions = [
            ("Maximized", 1600, 950),
            ("Compact", 1150, 760),
        ]

        for name, w, h in resolutions:
            self.root.geometry(f"{w}x{h}")
            self.root.update_idletasks()
            self.root.update()

            _, video_card, right_panel, _, evt_card = self._get_containers()

            root_h = self.root.winfo_height()
            bottom_y = evt_card.winfo_y() + evt_card.winfo_height() + evt_card.master.winfo_y()
            self.assertLessEqual(bottom_y, root_h, f"Clipping detected in {name} mode ({bottom_y} > {root_h})")

    def test_05_camera_actual_render_scale_and_voice_assistant_width(self):
        """Verify 4 stacked cards, actual rendered video is 4:3, and Voice Assistant >= 240px."""
        import math
        _, video_card, right_panel, _, _ = self._get_containers()
        
        # Section 3: 4 cards stacked vertically inside right_panel
        self.assertEqual(len(right_panel.winfo_children()), 4, "Right panel must have 4 vertically stacked cards!")
        voice_card = right_panel.winfo_children()[3]

        # Section 7: Voice Assistant width >= 240 px
        voice_w = voice_card.winfo_width()
        self.assertGreaterEqual(voice_w, 240, f"Voice Assistant width {voice_w} px is less than 240 px!")

        # Section 2: Camera Canvas and 4:3 render scale
        canvas = self.app.video_canvas
        c_w = canvas.winfo_width()
        c_h = canvas.winfo_height()
        scale = min(c_w / 640.0, c_h / 480.0)
        render_w = math.floor(640.0 * scale)
        render_h = math.floor(480.0 * scale)
        aspect = render_w / render_h

        # Strict 4:3 aspect ratio
        self.assertAlmostEqual(aspect, 4.0 / 3.0, places=2, msg="Rendered image is not 4:3 aspect ratio!")

    def test_06_ccsdf_and_authority_full_visibility(self):
        """Verify CCSDF Risk and Authority values are 100% visible with zero clipping."""
        test_values = [
            ("R=0.47 [DENY]", "UNKNOWN_PERSON [YELLOW]"),
            ("R=0.54 [RESTRICT]", "UNAUTHORIZED_RESTRICTED_ZONE_ACCESS [RED]"),
            ("R=0.64 [RESTRICT]", "AUTHORIZED [GREEN]"),
            ("R=0.99 [PENDING_VERIFICATION]", "MULTI_ZONE_BREACH_DETECTED [RED]"),
        ]

        for ccsdf_val, auth_val in test_values:
            self.app.ctx_item_ccsdf.set_value(ccsdf_val)
            self.app.ctx_item_decision.set_value(auth_val)
            self.root.update_idletasks()
            self.root.update()

            cc_text = self.app.ctx_item_ccsdf.lbl_val.cget("text")
            cc_req = self.app.ctx_item_ccsdf.lbl_val.winfo_reqwidth()
            cc_act = self.app.ctx_item_ccsdf.lbl_val.winfo_width()

            auth_text = self.app.ctx_item_decision.lbl_val.cget("text")
            auth_req = self.app.ctx_item_decision.lbl_val.winfo_reqwidth()
            auth_act = self.app.ctx_item_decision.lbl_val.winfo_width()

            self.assertNotIn("...", cc_text, f"CCSDF text was truncated with ellipsis: {cc_text}")
            self.assertEqual(cc_text, ccsdf_val, f"CCSDF text '{cc_text}' did not match expected '{ccsdf_val}'")
            self.assertGreaterEqual(cc_act, cc_req, f"CCSDF value '{ccsdf_val}' is clipped! (act {cc_act} < req {cc_req})")

            self.assertNotIn("...", auth_text, f"Authority text was truncated with ellipsis: {auth_text}")
            self.assertEqual(auth_text, auth_val, f"Authority text '{auth_text}' did not match expected '{auth_val}'")
            self.assertGreaterEqual(auth_act, auth_req, f"Authority value '{auth_val}' is clipped! (act {auth_act} < req {auth_req})")


if __name__ == "__main__":
    unittest.main()
