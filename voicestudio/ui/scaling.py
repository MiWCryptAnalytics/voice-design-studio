"""Pick a sensible UI scale on HiDPI screens nobody has configured.

Qt only scales the UI when something tells it to — a Wayland compositor scale,
`Xft.dpi`, or a `QT_*` environment override. A bare X11 session on a 4K monitor
reports 96 logical DPI, so everything renders tiny and the user reaches for
`QT_SCALE_FACTOR=2`. This module makes that reach unnecessary: when nothing else
has scaled the UI, it compares the screen's *physical* DPI against the 96 DPI
baseline and enlarges the application font to match. Every size in the app
derives from that font (see metrics.py), so one font change scales the whole UI.

The decision defers to anything explicit, in this order:

1. `VOICESTUDIO_SCALE` — forced factor; `1` disables auto-scaling entirely.
2. Any Qt scaling override (`QT_SCALE_FACTOR`, `QT_FONT_DPI`, …) — the user or
   desktop already decided; do nothing on top.
3. A devicePixelRatio above 1 — the platform already scales (Wayland, or a
   properly configured X11).
4. Logical DPI above ~110 — the desktop scales through fonts (`Xft.dpi`).
5. Otherwise: factor = physical DPI / 96, in quarter steps, only when the
   screen is meaningfully HiDPI, ignoring screens whose EDID-reported size is
   nonsense (projectors and TVs routinely lie).
"""

from __future__ import annotations

import os
import sys
from typing import Mapping

# Any of these means the user or desktop already chose a scaling policy.
QT_SCALE_OVERRIDES = (
    "QT_SCALE_FACTOR",
    "QT_SCREEN_SCALE_FACTORS",
    "QT_FONT_DPI",
    "QT_ENABLE_HIGHDPI_SCALING",
    "QT_USE_PHYSICAL_DPI",
)

FORCE_ENV = "VOICESTUDIO_SCALE"

BASELINE_DPI = 96.0
# Below this logical DPI the desktop clearly hasn't configured font scaling.
CONFIGURED_LOGICAL_DPI = 110.0
# Physical DPI outside this range means the screen is lying about its size.
PLAUSIBLE_PHYSICAL_DPI = (50.0, 400.0)
# Don't bother scaling for less than this — it reads as jitter, not intent.
MIN_AUTO_FACTOR = 1.25
FACTOR_RANGE = (0.5, 3.0)


def _clamp(factor: float) -> float:
    low, high = FACTOR_RANGE
    return min(max(factor, low), high)


def pick_scale(
    logical_dpi: float,
    physical_dpi: float,
    device_pixel_ratio: float,
    env: Mapping[str, str],
) -> float:
    """The scale factor to apply to the application font. Pure — no Qt."""
    forced = env.get(FORCE_ENV, "").strip()
    if forced:
        try:
            return _clamp(float(forced))
        except ValueError:
            pass  # unparseable — fall through to auto

    if any(env.get(name) for name in QT_SCALE_OVERRIDES):
        return 1.0
    if device_pixel_ratio > 1.001:
        return 1.0
    if logical_dpi > CONFIGURED_LOGICAL_DPI:
        return 1.0
    low, high = PLAUSIBLE_PHYSICAL_DPI
    if not (low <= physical_dpi <= high):
        return 1.0

    factor = physical_dpi / BASELINE_DPI
    if factor < MIN_AUTO_FACTOR:
        return 1.0
    # Quarter steps: enough resolution to fit any screen, coarse enough that
    # two launches on the same monitor can't disagree.
    return _clamp(round(factor * 4) / 4)


def apply_auto_scale(app) -> float:
    """Detect, scale the application font, and return the factor used.

    Call once, right after the QApplication exists and before metrics/QSS are
    built. Headless platforms report no meaningful screen, so they stay at 1.0
    unless a factor is forced.
    """
    forced = os.environ.get(FORCE_ENV, "").strip()
    if not forced and app.platformName() in ("offscreen", "minimal"):
        return 1.0

    screen = app.primaryScreen()
    if screen is None:
        return 1.0

    factor = pick_scale(
        logical_dpi=float(screen.logicalDotsPerInch()),
        physical_dpi=float(screen.physicalDotsPerInch()),
        device_pixel_ratio=float(screen.devicePixelRatio()),
        env=os.environ,
    )
    if abs(factor - 1.0) < 0.01:
        return 1.0

    font = app.font()
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * factor)
    elif font.pixelSize() > 0:
        font.setPixelSize(max(1, round(font.pixelSize() * factor)))
    else:
        return 1.0
    app.setFont(font)

    geometry = screen.geometry()
    print(
        f"voicestudio: scaled UI x{factor:g} for {geometry.width()}x"
        f"{geometry.height()} at {screen.physicalDotsPerInch():.0f} DPI "
        f"({FORCE_ENV}=1 to disable, {FORCE_ENV}=<factor> to override)",
        file=sys.stderr,
    )
    return factor
