"""
renderers/spectrum.py — FFT spectrum overlay rendered via a single RGBA texture.

Each SpectrumOverlay owns one DPG raw texture of size (height, wave_w, 4) float32.
The texture is full-pixel-width so draw_image renders it 1:1 with no bilinear
scaling — bars have crisp, sharp edges.

Computation strategy:
  - Bar heights are computed on a small (height, n_bands, 4) buffer.
  - That buffer is expanded to full (height, wave_w, 4) via a numpy gather
    (integer column→band index map) immediately before upload.
  - The gather produces a new contiguous array each call, which set_value
    holds a reference to until the next render frame.
  - This gives crisp nearest-neighbour bar edges with ~1/14 the scatter cost
    of computing directly at wave_w (128 bands vs 1800 pixels).

n_bands is user-configurable.  When it changes, only the column→band index
map is rebuilt — no texture resize.
"""

from __future__ import annotations
import numpy as np
import dearpygui.dearpygui as dpg

SPECTRUM_DECAY    = 0.72    # bar fall multiplier per update frame
SPECTRUM_DB_FLOOR = -70.0   # dBFS floor


class SpectrumOverlay:
    """Draws FFT bars as a single RGBA texture overlay in a parent DPG drawlist."""

    def __init__(
        self,
        texture_registry: int | str,
        drawlist_tag: int | str,
        x_offset: int,
        width: int,
        height: int,
        n_bands: int,
        active_color: str,
    ) -> None:
        self._reg      = texture_registry
        self._dl       = drawlist_tag
        self._xo       = x_offset
        self._n_bands  = max(1, n_bands)
        self._width    = max(1, width)
        self._height   = max(1, height)
        self._visible  = True
        self._smooth: np.ndarray | None = None

        # Small computation buffer: (height, n_bands, 4).
        # Expanded to full width at upload time via np.take into _out.
        self._band_buf = np.zeros((self._height, self._n_bands, 4), dtype=np.float32)

        # Pre-allocated full-width output buffer — reused every update() to avoid
        # a 9.6 MB per-call heap allocation.  Safe to reuse because spectrum fires
        # every other frame, so DPG always completes the GPU upload (render_frame)
        # before the next update() overwrites _out.
        self._out = np.empty((self._height, self._width, 4), dtype=np.float32)

        # Cached nearest-neighbour column→band index map.
        # Rebuilt only when n_bands changes.
        self._col_to_band: np.ndarray = self._build_col_map(self._width, self._n_bands)
        self._col_to_band_n: int = self._n_bands

        h = active_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        self._bar_color_active_f32 = np.array(
            [r / 255.0, g / 255.0, b / 255.0, 180 / 255.0], dtype=np.float32)
        self._bar_color_dim_f32 = np.array(
            [r * 0.5 / 255.0, g * 0.5 / 255.0, b * 0.5 / 255.0, 180 * 0.6 / 255.0],
            dtype=np.float32)
        self._bar_color_f32 = self._bar_color_active_f32.copy()

        # Full-width texture: draw_image maps it 1:1, no bilinear stretching.
        flat = np.zeros(self._width * self._height * 4, dtype=np.float32)
        self._tag = dpg.add_raw_texture(
            self._width, self._height, flat,
            format=dpg.mvFormat_Float_rgba,
            parent=self._reg,
        )
        self._img = dpg.draw_image(
            self._tag,
            (self._xo, 0), (self._xo + self._width, self._height),
            show=self._visible,
            parent=self._dl,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, raw_bands: np.ndarray | None, playing: bool, n_bands: int) -> None:
        if not self._visible:
            return

        n = max(1, n_bands)
        h = self._height

        # Rebuild band buffer and index map when n_bands changes.
        if n != self._n_bands:
            self._n_bands  = n
            self._band_buf = np.zeros((h, n, 4), dtype=np.float32)
            self._smooth   = None
            # _out stays the same size (wave_w × height) — no realloc needed

        if raw_bands is not None and playing and len(raw_bands) == n:
            fracs = np.clip(
                (raw_bands - SPECTRUM_DB_FLOOR) / (-SPECTRUM_DB_FLOOR),
                0.0, 1.0,
            ).astype(np.float32)
        else:
            fracs = np.zeros(n, dtype=np.float32)

        if self._smooth is None or len(self._smooth) != n:
            self._smooth = fracs.copy()
        else:
            np.maximum(fracs, self._smooth * SPECTRUM_DECAY, out=self._smooth)

        # Rebuild column→band map when n_bands changed.
        if self._col_to_band_n != n:
            self._col_to_band   = self._build_col_map(self._width, n)
            self._col_to_band_n = n

        # Bar drawing on small (H, n_bands) buffer — cheap scatter.
        bar_heights = np.minimum((self._smooth * h).astype(np.int32), h)  # (n,)
        rows        = np.arange(h, dtype=np.int32)[:, np.newaxis]          # (H, 1)
        thresholds  = (h - bar_heights)[np.newaxis, :]                      # (1, n)
        mask        = rows >= thresholds                                     # (H, n) bool

        self._band_buf[:] = 0.0
        self._band_buf[mask] = self._bar_color_f32

        # Expand to full width via nearest-neighbour gather into pre-allocated _out.
        # np.take writes directly to _out (no heap allocation).  _out.ravel() is a
        # view (C-contiguous), so set_value stores a reference at zero copy cost.
        # _out is safe to reuse next tick because spectrum fires every other frame —
        # DPG's GPU upload always happens between the write and the next overwrite.
        if dpg.does_item_exist(self._tag):
            np.take(self._band_buf, self._col_to_band, axis=1, out=self._out)
            dpg.set_value(self._tag, self._out.ravel())

    def set_visible(self, visible: bool) -> None:
        if visible == self._visible:
            return
        self._visible = visible
        if dpg.does_item_exist(self._img):
            dpg.configure_item(self._img, show=visible)

    def set_active(self, is_active: bool) -> None:
        """Switch between active (bright) and dim bar colours."""
        new_color = self._bar_color_active_f32 if is_active else self._bar_color_dim_f32
        if np.array_equal(new_color, self._bar_color_f32):
            return
        self._bar_color_f32 = new_color.copy()

    def reset_smooth(self) -> None:
        self._smooth = None

    def delete(self) -> None:
        """Delete the DPG texture.  The draw_image is a drawlist child and
        will be cleaned up when the drawlist is deleted."""
        try:
            dpg.delete_item(self._tag)
        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_col_map(width: int, n_bands: int) -> np.ndarray:
        """Return int32 array of length `width` mapping each pixel column to a band."""
        return (np.arange(width, dtype=np.int32) * n_bands // width).clip(0, n_bands - 1)
