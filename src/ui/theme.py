"""
UI Theme & Visual Styles
------------------------
Modern Dark Security Operations Center (SOC) theme palette and font definitions.
"""

# Color Palette
BG_DARK = "#0B0F17"          # App window background
CARD_BG = "#151C28"          # Card background
CARD_BORDER = "#243247"      # Subtle card borders
CARD_HEADER_BG = "#1A2332"   # Header strip on cards

TEXT_PRIMARY = "#F1F5F9"     # High-contrast white/light text
TEXT_SECONDARY = "#94A3B8"   # Subtle gray text
TEXT_MUTED = "#64748B"       # Dim placeholder text

ACCENT_BLUE = "#38BDF8"      # Cyan / Sky blue accent
ACCENT_PURPLE = "#818CF8"    # Indigo accent

# Risk Colors
COLOR_GREEN = "#10B981"      # Normal / Secure
COLOR_YELLOW = "#F59E0B"     # Attention
COLOR_ORANGE = "#F97316"     # High Risk
COLOR_RED = "#EF4444"        # Security Incident

# High-Contrast Event Table Colors
TABLE_BG = "#131B27"         # Base table background
TABLE_BG_ALT = "#182232"     # Alternating row background
TABLE_TEXT = "#F8FAFC"       # Crisp high-contrast description text
TABLE_HEADER_BG = "#1E293B"  # Header row background
TABLE_HEADER_FG = "#FFFFFF"  # Header text

# High-Contrast Risk Foreground & Background Tints for Event Table
RISK_GREEN_FG = "#34D399"    # Bright Emerald Mint
RISK_GREEN_BG = "#0A291E"    # Deep Emerald tint
RISK_YELLOW_FG = "#FDE047"   # Bright Yellow Gold
RISK_YELLOW_BG = "#302208"   # Deep Amber tint
RISK_ORANGE_FG = "#FB923C"   # Bright Coral Orange
RISK_ORANGE_BG = "#351A0A"   # Deep Orange tint
RISK_RED_FG = "#F87171"      # Bright Salmon Crimson
RISK_RED_BG = "#381014"      # Deep Crimson tint

# Button Styles
BTN_START_BG = "#059669"
BTN_START_HOVER = "#10B981"
BTN_STOP_BG = "#DC2626"
BTN_STOP_HOVER = "#EF4444"
BTN_NEUTRAL_BG = "#1E293B"
BTN_NEUTRAL_BORDER = "#334155"

# Typography Fonts
FONT_FAMILY = "Segoe UI"
FONT_TITLE = (FONT_FAMILY, 14, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 11, "bold")
FONT_BODY = (FONT_FAMILY, 10)
FONT_BODY_BOLD = (FONT_FAMILY, 10, "bold")
FONT_SMALL = (FONT_FAMILY, 9)
FONT_MONO = ("Consolas", 9)
FONT_RISK_BANNER = (FONT_FAMILY, 13, "bold")
