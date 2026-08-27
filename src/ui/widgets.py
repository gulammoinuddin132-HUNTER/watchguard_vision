"""
Custom UI Widgets
-----------------
Reusable styled UI components for the WatchGuard Vision dashboard.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

from src.events.event_types import RiskLevel
from src.ui.theme import (
    CARD_BG,
    CARD_BORDER,
    CARD_HEADER_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    ACCENT_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_RED,
    FONT_TITLE,
    FONT_SUBTITLE,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_SMALL,
    FONT_RISK_BANNER,
)


class CardFrame(tk.Frame):
    """A sleek dark container card with an optional compact title header."""

    def __init__(self, parent, title: Optional[str] = None, *args, **kwargs):
        super().__init__(
            parent,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            *args,
            **kwargs
        )
        self.title = title
        if title:
            header_frame = tk.Frame(self, bg=CARD_HEADER_BG, height=24)
            header_frame.pack(fill="x", side="top")
            header_frame.pack_propagate(False)

            # Left accent dot
            dot = tk.Label(header_frame, text="●", font=(FONT_SMALL[0], 7),
                           bg=CARD_HEADER_BG, fg=ACCENT_BLUE)
            dot.pack(side="left", padx=(6, 3), pady=2)

            lbl = tk.Label(
                header_frame,
                text=title.upper(),
                font=FONT_SMALL,
                bg=CARD_HEADER_BG,
                fg=TEXT_PRIMARY,
            )
            lbl.pack(side="left", pady=2)


class RiskBanner(tk.Frame):
    """Dynamic visual indicator banner for current threat & risk level."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, bg=CARD_BG, *args, **kwargs)
        self._current_level: Optional[RiskLevel] = None
        self._current_desc: Optional[str] = None

        self.container = tk.Frame(
            self,
            bg=COLOR_GREEN,
            padx=8,
            pady=4,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
        )
        self.container.pack(fill="x", expand=True)

        self.icon_label = tk.Label(
            self.container,
            text="🛡️",
            font=("Segoe UI Emoji", 12),
            bg=COLOR_GREEN,
            fg="#FFFFFF",
        )
        self.icon_label.pack(side="left", padx=(0, 6))

        self.text_frame = tk.Frame(self.container, bg=COLOR_GREEN)
        self.text_frame.pack(side="left", fill="both", expand=True)

        self.title_label = tk.Label(
            self.text_frame,
            text="SYSTEM STATUS: NORMAL",
            font=FONT_BODY_BOLD,
            bg=COLOR_GREEN,
            fg="#FFFFFF",
            anchor="w",
        )
        self.title_label.pack(fill="x")

        self.desc_label = tk.Label(
            self.text_frame,
            text="No active threats or policy violations detected.",
            font=FONT_SMALL,
            bg=COLOR_GREEN,
            fg="#E0F2FE",
            anchor="w",
            height=1,
        )
        self.desc_label.pack(fill="x")

    def update_risk(self, level: RiskLevel | str, description: Optional[str] = None) -> None:
        """Updates colors and text reflecting the current risk status with fixed single-line height."""
        if isinstance(level, str):
            try:
                level = RiskLevel(level.upper())
            except ValueError:
                level = RiskLevel.GREEN

        # Truncate description to single line with ellipsis if needed
        if description:
            clean_desc = description.strip()
            if len(clean_desc) > 65:
                desc_text = clean_desc[:62] + "..."
            else:
                desc_text = clean_desc
        else:
            desc_text = {
                RiskLevel.GREEN: "No active threats or policy violations detected.",
                RiskLevel.YELLOW: "Attention required: Potential anomaly or unverified person.",
                RiskLevel.ORANGE: "High Risk: Policy violation or restricted zone activity.",
                RiskLevel.RED: "CRITICAL: Security incident in progress!",
            }.get(level, "Status updated.")

        # Flicker-free guard: only reconfigure widgets if state has changed
        if self._current_level == level and self._current_desc == desc_text:
            return

        self._current_level = level
        self._current_desc = desc_text

        icons = {
            RiskLevel.GREEN: "🛡️",
            RiskLevel.YELLOW: "⚠️",
            RiskLevel.ORANGE: "🚨",
            RiskLevel.RED: "⛔",
        }
        color = level.color_hex
        title_text = f"SYSTEM STATUS: {level.label}"

        # Update widget colors & text
        self.container.configure(bg=color)
        self.icon_label.configure(bg=color, text=icons.get(level, "🛡️"))
        self.text_frame.configure(bg=color)
        self.title_label.configure(bg=color, text=title_text)
        self.desc_label.configure(bg=color, text=desc_text)


class TelemetryItem(tk.Frame):
    """Key-value telemetry row with compact high contrast styling and change-caching."""

    def __init__(self, parent, label: str, default_value: str = "--", *args, **kwargs):
        super().__init__(parent, bg=CARD_BG, *args, **kwargs)
        self._current_value: str = default_value
        self._current_color: str = TEXT_PRIMARY

        self.lbl_key = tk.Label(
            self,
            text=label,
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
        )
        self.lbl_key.pack(side="left", padx=2, pady=1)

        self.lbl_val = tk.Label(
            self,
            text=default_value,
            font=(FONT_SMALL[0], FONT_SMALL[1], "bold"),
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="e",
        )
        self.lbl_val.pack(side="right", padx=2, pady=1)

    def set_value(self, value: str, color: Optional[str] = None) -> None:
        c = color or TEXT_PRIMARY
        # Flicker-free guard: only reconfigure if text or color changed
        if self._current_value == value and self._current_color == c:
            return
        self._current_value = value
        self._current_color = c
        self.lbl_val.configure(text=value, fg=c)


class BoundedTextViewport(tk.Frame):
    """
    Fixed-height, read-only text container with internal word wrapping.
    Guarantees zero parent geometry changes when text length changes.
    """

    def __init__(
        self,
        parent,
        height_px: int = 36,
        bg: str = CARD_BG,
        fg: str = TEXT_PRIMARY,
        font=FONT_SMALL,
        *args,
        **kwargs,
    ):
        super().__init__(parent, bg=bg, height=height_px, *args, **kwargs)
        self.pack_propagate(False)
        self.grid_propagate(False)

        self._current_text: str = ""
        self._current_fg: str = fg

        self.text_widget = tk.Text(
            self,
            bg=bg,
            fg=fg,
            font=font,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=1,
            pady=0,
            cursor="arrow",
            state="disabled",
        )
        self.text_widget.pack(fill="both", expand=True)

    def set_text(self, text: str, fg: Optional[str] = None) -> None:
        """Updates text with change-caching to prevent unnecessary redraws."""
        color = fg or self._current_fg
        if self._current_text == text and self._current_fg == color:
            return

        self._current_text = text
        self._current_fg = color

        self.text_widget.configure(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", text)
        self.text_widget.configure(fg=color, state="disabled")

    def configure(self, cnf=None, **kwargs):
        """Backwards-compatible wrapper for standard Tkinter configure / config."""
        if cnf is not None and isinstance(cnf, dict):
            kwargs.update(cnf)
        text = kwargs.pop("text", None)
        fg = kwargs.pop("fg", kwargs.pop("foreground", None))
        if text is not None or fg is not None:
            self.set_text(text if text is not None else self._current_text, fg=fg)
        if kwargs:
            super().configure(**kwargs)

    config = configure

    def cget(self, key: str):
        """Backwards-compatible wrapper for standard Tkinter cget."""
        if key in ("text", "-text"):
            return self._current_text
        elif key in ("fg", "-fg", "foreground", "-foreground"):
            return self._current_fg
        return super().cget(key)

    def __getitem__(self, key: str):
        return self.cget(key)

    def get_text(self) -> str:
        """Returns the current rendered text."""
        return self._current_text
