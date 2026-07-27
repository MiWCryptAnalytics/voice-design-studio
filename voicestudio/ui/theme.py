"""Dark studio theme.

Sizes are computed from the application font (see metrics.py) so the UI honours
the user's Qt HiDPI and font settings instead of pinning everything to fixed
pixels.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor

from .metrics import Metrics, metrics

# Palette
BG = "#14161a"
PANEL = "#1b1e24"
PANEL_ALT = "#22262e"
BORDER = "#2e333d"
TEXT = "#e6e9ef"
TEXT_DIM = "#8d95a5"
ACCENT = "#5b9cff"
ACCENT_DIM = "#3d6dbf"
GOOD = "#4ec9a5"
WARN = "#e0a35c"
BAD = "#e0655c"
STAR = "#ffc857"

WAVE = QColor(91, 156, 255)
WAVE_PLAYED = QColor(78, 201, 165)
WAVE_BG = QColor(28, 31, 38)
PLAYHEAD = QColor(230, 233, 239)

SLOT_A = "#5b9cff"
SLOT_B = "#c98bff"


def build_qss(m: Metrics | None = None) -> str:
    m = m or metrics()

    pad_y = m.sp(0.35)
    pad_x = m.sp(0.6)
    radius = m.sp(0.35)
    radius_sm = m.sp(0.25)
    field_h = m.sp(1.25)
    bar_h = m.sp(0.35)
    check_box = m.sp(0.85)
    scroll_w = m.sp(0.55)

    return f"""
/* Base widgets inherit the application font — never pin a pixel size here,
   or the UI stops following the user's font/DPI settings. */
QWidget {{
    background: {BG};
    color: {TEXT};
}}
QLabel, QCheckBox {{ background: transparent; }}
QLabel[role="title"] {{
    font-size: {m.pt(0.92)}pt;
    font-weight: 600;
    color: {TEXT_DIM};
    letter-spacing: 1px;
}}
QLabel[role="hint"] {{ color: {TEXT_DIM}; font-size: {m.pt(0.86)}pt; }}
QLabel[role="metric"] {{ color: {TEXT_DIM}; font-size: {m.pt(0.86)}pt; }}

QFrame[role="panel"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {radius}px;
}}
QFrame[role="card"] {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {radius_sm}px;
}}
QFrame[role="card"][selected="true"] {{ border: 1px solid {ACCENT}; }}
QFrame[role="card"][slot="A"] {{ border-left: {m.sp(0.18)}px solid {SLOT_A}; }}
QFrame[role="card"][slot="B"] {{ border-left: {m.sp(0.18)}px solid {SLOT_B}; }}

QPlainTextEdit, QTextEdit, QLineEdit {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {radius_sm}px;
    padding: {pad_y}px;
    selection-background-color: {ACCENT_DIM};
}}
QPlainTextEdit:focus, QTextEdit:focus, QLineEdit:focus {{ border: 1px solid {ACCENT}; }}

QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {radius_sm}px;
    padding: {max(2, pad_y - 1)}px {pad_x}px;
    min-height: {field_h}px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
/* Disabled inputs must read as disabled — e.g. sub-talker fields while linked. */
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QLineEdit:disabled, QPlainTextEdit:disabled {{
    color: {TEXT_DIM};
    background: {PANEL};
    border-color: {PANEL_ALT};
}}
QCheckBox:disabled {{ color: {TEXT_DIM}; }}
QLabel:disabled {{ color: {TEXT_DIM}; }}
QComboBox::drop-down {{ border: none; width: {m.sp(0.9)}px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}

QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {radius_sm}px;
    padding: {pad_y}px {pad_x}px;
}}
QPushButton:hover {{ border: 1px solid {ACCENT_DIM}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}
QPushButton[role="primary"] {{
    background: {ACCENT_DIM};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {ACCENT}; }}
QPushButton[role="primary"]:disabled {{ background: {PANEL_ALT}; border-color: {BORDER}; }}
QPushButton[role="danger"]:hover {{ border: 1px solid {BAD}; color: {BAD}; }}
QPushButton[role="icon"] {{ padding: {max(1, pad_y - 2)}px {max(2, pad_x // 2)}px; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: {scroll_w}px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: {max(2, scroll_w // 2)}px;
    min-height: {m.sp(1.6)}px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: {scroll_w}px; }}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: {max(2, scroll_w // 2)}px;
    min-width: {m.sp(1.6)}px;
}}

QProgressBar {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {max(2, bar_h // 2)}px;
    height: {bar_h}px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: {max(1, bar_h // 2)}px; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}

QCheckBox {{ spacing: {m.sp(0.35)}px; }}
QCheckBox::indicator {{
    width: {check_box}px; height: {check_box}px;
    border: 1px solid {BORDER};
    border-radius: {radius_sm}px;
    background: {PANEL_ALT};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}

QToolTip {{
    background: {PANEL_ALT};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: {max(2, pad_y - 2)}px;
}}
"""
