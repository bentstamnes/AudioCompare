"""
renderers/spectrum.py — FFT spectrum overlay rendered via DPG draw_rectangle calls.

Draws directly into a provided DPG drawlist (slot_dl) at a given x_offset,
using the same pattern as MeterStrip.  No separate child drawlist is created,
which eliminates all pos= positioning issues.
"""

from __future__ import annotations
import numpy as np
import dearpygui.dearpygui as dpg

SPECTRUM_DECAY    = 0.72   # bar fall multiplier per update frame
SPECTRUM_DB_FLOOR = -70.0  # dBFS floor


class SpectrumOverlay:
    """Draws FFT bars as rectangles directly into a parent DPG drawlist."""

    def __init__(
        self,
        drawlist_tag: int | str,
        x_offset: int,
        width: int,
        height: int,
        n_bands: int,
        active_color: str,
    ) -> None:
        self._dl       = drawlist_tag
        self._xo       = x_offset
        self._n_bands  = n_bands
        self._smooth: np.ndarray | None = None
        self._visible  = True
        self._width    = max(1, width)
        self._height   = max(1, height)

        h = active_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        self._bar_color_active = (r, g, b, 180)
        self._bar_color_dim    = (int(r * 0.5), int(g * 0.5), int(b * 0.5), int(180 * 0.6))
        self._bar_color        = self._bar_color_active

        self._rects: list[int | str] = []
        self._bg_rects: list[int | str] = []
        self._build_rects()

    def _build_rects(self) -> None:
        n     = self._n_bands
        bar_w = max(1.0, self._width / n)
        height = self._height
        xo     = self._xo
        for i in range(n):
            rx = xo + i * bar_w
            bg = dpg.draw_rectangle(
                (rx, height), (rx + bar_w - 1, height),
                color=(0, 0, 0, 0),
                fill=(0, 0, 0, 140),
                show=self._visible,
                parent=self._dl,
            )
            self._bg_rects.append(bg)
            tag = dpg.draw_rectangle(
                (rx, height), (rx + bar_w - 1, height),
                color=(0, 0, 0, 0),
                fill=self._bar_color,
                show=self._visible,
                parent=self._dl,
            )
            self._rects.append(tag)

    def update(self, raw_bands: np.ndarray | None, playing: bool, n_bands: int) -> None:
        if not self._visible:
            return

        if n_bands != self._n_bands:
            self._n_bands = n_bands
            self._smooth  = None
            self._rebuild_rects()

        h = self._height

        if raw_bands is not None and playing:
            fracs = np.clip(
                (raw_bands - SPECTRUM_DB_FLOOR) / (-SPECTRUM_DB_FLOOR),
                0.0, 1.0,
            ).astype(np.float32)
        else:
            fracs = np.zeros(n_bands, dtype=np.float32)

        if self._smooth is None or len(self._smooth) != n_bands:
            self._smooth = fracs.copy()
        else:
            np.maximum(fracs, self._smooth * SPECTRUM_DECAY, out=self._smooth)

        bar_w = self._width / n_bands
        xo    = self._xo
        for i, rect in enumerate(self._rects):
            bar_h = int(self._smooth[i] * h)
            rx    = xo + i * bar_w
            dpg.configure_item(self._bg_rects[i],
                               pmin=(rx,             h - bar_h),
                               pmax=(rx + bar_w - 1, h))
            dpg.configure_item(rect,
                               pmin=(rx,             h - bar_h),
                               pmax=(rx + bar_w - 1, h))

    def _rebuild_rects(self) -> None:
        for r in self._bg_rects + self._rects:
            try:
                dpg.delete_item(r)
            except Exception:
                pass
        self._rects = []
        self._bg_rects = []
        self._build_rects()

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        for r in self._bg_rects + self._rects:
            try:
                dpg.configure_item(r, show=visible)
            except Exception:
                pass

    def set_active(self, is_active: bool) -> None:
        color = self._bar_color_active if is_active else self._bar_color_dim
        if color == self._bar_color:
            return
        self._bar_color = color
        for r in self._rects:
            try:
                dpg.configure_item(r, fill=color)
            except Exception:
                pass

    def reset_smooth(self) -> None:
        self._smooth = None

    def delete(self) -> None:
        for r in self._bg_rects + self._rects:
            try:
                dpg.delete_item(r)
            except Exception:
                pass
        self._rects = []
        self._bg_rects = []
