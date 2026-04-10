"""
renderers/correlation.py — Stereo correlation meter bar.

Drawn directly into the slot's main drawlist (slot_dl) below the VU meter strip.
Occupies a fixed-height band at the bottom of the meter column.

Visual:
  - Dark background spanning METER_W
  - Thin center tick (neutral reference)
  - Filled colored bar growing from center toward the current value
    Green  (+0.5 .. +1.0) : good correlation / mono-compatible
    Yellow ( 0.0 ..  0.5) : weak correlation
    Red   (-1.0 ..  0.0) : out-of-phase
  - Value decays toward 0 when not playing
"""

from __future__ import annotations
import dearpygui.dearpygui as dpg
from renderers.meters import METER_W

CORR_H     = 12    # height of the correlation bar in pixels
CORR_GAP   =  2    # gap between VU meter bottom and this bar
CORR_TOTAL = CORR_H + CORR_GAP   # total pixels consumed from slot bottom

_COL_POSITIVE = (187, 187, 187, 220)
_COL_WEAK     = (220, 200, 50,  220)
_COL_NEGATIVE = (220, 60,  50,  220)
_COL_BG     = (13,  13,  13,  255)
_COL_TICK   = (80,  80,  80,  200)

_DECAY = 0.85   # per-update smoothing toward 0 when paused


class CorrelationBar:
    """Draws a stereo correlation meter into a DPG drawlist."""

    def __init__(
        self,
        drawlist_tag: int | str,
        x_offset: int,
        slot_height: int,
    ) -> None:
        self._dl     = drawlist_tag
        self._xo     = x_offset
        self._smooth = 0.0

        x0 = x_offset
        x1 = x0 + METER_W
        y0 = slot_height - CORR_H        # top of bar
        y1 = slot_height                 # bottom of bar
        cx = x0 + METER_W // 2          # horizontal center

        # Background
        dpg.draw_rectangle(
            (x0, y0), (x1, y1),
            color=(0, 0, 0, 0), fill=_COL_BG,
            parent=self._dl,
        )
        # Center reference tick
        dpg.draw_line(
            (cx, y0), (cx, y1),
            color=_COL_TICK, thickness=1,
            parent=self._dl,
        )
        # Filled value bar (pmin/pmax updated each frame)
        self._bar = dpg.draw_rectangle(
            (cx, y0 + 2), (cx, y1 - 2),
            color=(0, 0, 0, 0), fill=_COL_POSITIVE,
            parent=self._dl,
        )

        self._x0   = x0
        self._x1   = x1
        self._cx   = cx
        self._y0   = y0
        self._y1   = y1

    def update(self, corr: float, playing: bool) -> None:
        if playing:
            # Smooth slightly to avoid jitter
            self._smooth = self._smooth * 0.6 + corr * 0.4
        else:
            self._smooth *= _DECAY

        v  = max(-1.0, min(1.0, self._smooth))
        cx = self._cx
        hw = METER_W // 2   # half width available each side

        bar_x = cx + int(v * hw)   # endpoint (right of center for positive)
        x_min = min(cx, bar_x)
        x_max = max(cx, bar_x)

        if v >= 0.5:
            color = _COL_POSITIVE
        elif v >= 0.0:
            color = _COL_WEAK
        else:
            color = _COL_NEGATIVE

        dpg.configure_item(
            self._bar,
            pmin=(x_min,        self._y0 + 2),
            pmax=(max(x_max, x_min + 1), self._y1 - 2),
            fill=color,
        )
