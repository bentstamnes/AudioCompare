"""
renderers/meters.py — GPU VU meter renderer for MixABTestGPU.

Draws peak/RMS meters using DPG draw_rectangle / draw_line calls directly
into the slot's main drawlist at a given x_offset.  No separate child
drawlist is used — this avoids DPG positioning quirks with add_drawlist
inside themed child windows.
"""

from __future__ import annotations
import math
import time
import dearpygui.dearpygui as dpg

METER_W    = 48    # total width of the meter strip in pixels
_BAR_W     = 10   # width of the RMS/peak bar (rightmost part of the strip)
_TXT_X     = 2    # left-align text inside the text zone (left of the bar)

_FLOOR_DB  = -60.0   # dBFS floor for meter display

HOLD_SECS  = 2.0     # seconds peak marker holds at max
FALL_RATE  = 0.4     # fraction per second the peak falls after hold


def _db_fraction(linear: float) -> float:
    if linear <= 0.0:
        return 0.0
    db = 20.0 * math.log10(linear)
    return max(0.0, min(1.0, (db - _FLOOR_DB) / (-_FLOOR_DB)))


def _hex_to_dpg(hex_color: str, alpha: int = 255) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


class MeterStrip:
    """Owns DPG draw items for one track's VU meter strip.

    All items are drawn directly into ``drawlist_tag`` at x=``x_offset``.
    This avoids a separate child drawlist and the DPG positioning issues
    that come with it.
    """

    def __init__(
        self,
        drawlist_tag: int | str,
        x_offset: int,
        active_color: str,
        inactive_color: str,
        height: int,
    ) -> None:
        self._dl       = drawlist_tag
        self._xo       = x_offset          # left edge of meter strip in drawlist coords
        self._clr_rms  = _hex_to_dpg(inactive_color)
        self._clr_peak = _hex_to_dpg(active_color)
        self._clr_bg   = (13, 13, 13, 255)
        self._height   = max(1, height)

        self._peak_hold      = 0.0
        self._peak_hold_time = 0.0

        x0 = self._xo
        x1 = x0 + METER_W
        bx0 = x0 + (METER_W - _BAR_W)
        h   = self._height

        # Background
        dpg.draw_rectangle(
            (x0, 0), (x1, h),
            color=(0, 0, 0, 0), fill=self._clr_bg,
            parent=self._dl,
        )
        # RMS bar (starts at full height = silent)
        self._rms_rect = dpg.draw_rectangle(
            (bx0, h), (x1, h),
            color=(0, 0, 0, 0), fill=self._clr_rms,
            parent=self._dl,
        )
        # Peak hold line
        self._peak_line = dpg.draw_line(
            (bx0, h), (x1, h),
            color=self._clr_peak, thickness=2,
            parent=self._dl,
        )
        # dB text items
        self._peak_txt = dpg.draw_text(
            (x0 + _TXT_X, 4), "",
            color=self._clr_peak, size=11,
            parent=self._dl,
        )
        self._rms_txt = dpg.draw_text(
            (x0 + _TXT_X, 16), "",
            color=self._clr_rms, size=11,
            parent=self._dl,
        )

    def update(self, rms: float, peak: float, playing: bool, height: int) -> None:
        """Recompute meter geometry and update DPG draw items."""
        if not playing:
            rms = peak = 0.0

        self._height = max(1, height)
        h = self._height

        rms_frac  = _db_fraction(rms)
        peak_frac = _db_fraction(peak)

        # Peak hold / fall
        now  = time.monotonic()
        held = self._peak_hold
        held_time = self._peak_hold_time
        if peak_frac >= held:
            self._peak_hold      = peak_frac
            self._peak_hold_time = now
            held = peak_frac
        else:
            elapsed = now - held_time
            if elapsed > HOLD_SECS:
                self._peak_hold = max(0.0, held - (elapsed - HOLD_SECS) * FALL_RATE)
                held = self._peak_hold

        x0  = self._xo
        x1  = x0 + METER_W
        bx0 = x0 + (METER_W - _BAR_W)

        rms_y  = int((1.0 - rms_frac) * h)
        peak_y = max(0, int((1.0 - held) * h))

        dpg.configure_item(
            self._rms_rect,
            pmin=(bx0, rms_y), pmax=(x1, h),
        )
        dpg.configure_item(
            self._peak_line,
            p1=(bx0, peak_y), p2=(x1, peak_y),
        )

        if playing and peak > 1e-9:
            peak_db = 20.0 * math.log10(peak)
            dpg.configure_item(self._peak_txt, text=f"{peak_db:+.1f}")
        else:
            dpg.configure_item(self._peak_txt, text="")

        if playing and rms > 1e-9:
            rms_db = 20.0 * math.log10(rms)
            dpg.configure_item(self._rms_txt, text=f"{rms_db:+.1f}")
        else:
            dpg.configure_item(self._rms_txt, text="")
