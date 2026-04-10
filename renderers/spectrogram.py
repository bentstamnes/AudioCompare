"""
renderers/spectrogram.py — Real-time scrolling spectrogram overlay.

Draws a horizontal spectrogram directly into a parent DPG drawlist using a
dynamic texture.  Each FFT update shifts the buffer one column left and writes
the new frequency column on the right edge.

Frequency layout: low freq at bottom (y = height-1), high freq at top (y = 0).
Color encodes amplitude via a fixed heat palette:
  silence → black → dark purple → bright magenta → red → bright orange
Mutually exclusive with SpectrumOverlay — only one is shown at a time.
"""

from __future__ import annotations
import numpy as np
import dearpygui.dearpygui as dpg

_SPECTRUM_DB_FLOOR = -70.0   # dBFS floor (matches spectrum.py)

# ── Heat palette ─────────────────────────────────────────────────────────────
# Control points: (amplitude_0_to_1, R, G, B) all in 0..1 range.
# Inspired by classic spectrogram "fire" palettes.
_PALETTE_STOPS = np.array([
    [0.00,  0.00, 0.00, 0.00],   # black        — silence
    [0.18,  0.12, 0.00, 0.22],   # dark purple
    [0.38,  0.55, 0.00, 0.65],   # mid purple
    [0.55,  0.85, 0.00, 0.75],   # bright magenta/pink
    [0.70,  0.95, 0.05, 0.15],   # red-pink
    [0.83,  1.00, 0.35, 0.00],   # orange-red
    [0.93,  1.00, 0.70, 0.00],   # orange
    [1.00,  1.00, 0.95, 0.40],   # bright yellow-orange — peak
], dtype=np.float32)


def _build_lut(n: int = 256) -> np.ndarray:
    """Return (n, 3) float32 RGB LUT, index 0 = silence, n-1 = peak."""
    xs   = _PALETTE_STOPS[:, 0]
    rgb  = _PALETTE_STOPS[:, 1:]
    t    = np.linspace(0.0, 1.0, n, dtype=np.float32)
    lut  = np.empty((n, 3), dtype=np.float32)
    for ch in range(3):
        lut[:, ch] = np.interp(t, xs, rgb[:, ch])
    return lut


_LUT = _build_lut(256)   # module-level, built once


class SpectrogramStrip:
    """Scrolling spectrogram overlay drawn into a parent DPG drawlist."""

    def __init__(
        self,
        tex_registry: int | str,
        drawlist_tag: int | str,
        width: int,
        height: int,
        active_color: str,   # accepted for API compatibility, not used for palette
    ) -> None:
        self._dl      = drawlist_tag
        self._width   = max(1, width)
        self._height  = max(1, height)
        self._visible = False   # hidden by default; enabled via set_visible

        # Dim multiplier when track is not the active one
        self._brightness = 1.0   # 1.0 = active, 0.5 = inactive

        # RGBA float32 pixel buffer: shape (height, width, 4)
        self._buf = np.zeros((self._height, self._width, 4), dtype=np.float32)

        # DPG dynamic texture (registered into the app's shared texture_registry)
        self._tex = dpg.add_dynamic_texture(
            self._width, self._height, self._buf.ravel(), parent=tex_registry,
        )

        # Draw image into the drawlist (covers the full waveform area)
        self._img = dpg.draw_image(
            self._tex,
            pmin=(0, 0),
            pmax=(self._width, self._height),
            show=self._visible,
            parent=self._dl,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, raw_bands: np.ndarray | None, n_bands: int) -> None:
        """Scroll buffer left by one pixel and paint new column from raw_bands."""
        if not self._visible:
            return

        w  = self._width
        h  = self._height
        br = self._brightness

        # Scroll: shift all columns left by 1
        if w > 1:
            self._buf[:, :w - 1, :] = self._buf[:, 1:, :]

        # Build the new rightmost column
        col = np.zeros((h, 4), dtype=np.float32)

        if raw_bands is not None and len(raw_bands) > 0:
            nb = len(raw_bands)
            # Convert dB bands to 0..1 fractions
            fracs = np.clip(
                (raw_bands - _SPECTRUM_DB_FLOOR) / (-_SPECTRUM_DB_FLOOR),
                0.0, 1.0,
            ).astype(np.float32)

            # Map bands → pixel rows: band 0 → bottom row (h-1), last band → top row (0)
            row_indices  = np.arange(h, dtype=np.float32)
            band_for_row = ((h - 1 - row_indices) / h * nb).astype(np.int32)
            band_for_row = np.clip(band_for_row, 0, nb - 1)
            amp = fracs[band_for_row]   # shape: (h,)  — 0..1 per row

            # Look up RGB from LUT
            lut_idx = np.clip((amp * 255.0).astype(np.int32), 0, 255)
            rgb = _LUT[lut_idx]         # shape: (h, 3)

            col[:, 0] = rgb[:, 0] * br
            col[:, 1] = rgb[:, 1] * br
            col[:, 2] = rgb[:, 2] * br
            # Alpha: fully opaque where there is any signal, transparent at silence
            col[:, 3] = np.where(amp > 0.01, 0.92, 0.0)

        self._buf[:, w - 1, :] = col

        # Push to GPU
        dpg.set_value(self._tex, self._buf.ravel())

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        try:
            dpg.configure_item(self._img, show=visible)
        except Exception:
            pass
        if not visible:
            # Clear buffer so it starts fresh next time it becomes visible
            self._buf[:] = 0.0
            try:
                dpg.set_value(self._tex, self._buf.ravel())
            except Exception:
                pass

    def set_active(self, is_active: bool) -> None:
        self._brightness = 1.0 if is_active else 0.5

    def delete(self) -> None:
        try:
            dpg.delete_item(self._img)
        except Exception:
            pass
        try:
            dpg.delete_item(self._tex)
        except Exception:
            pass
        self._buf = np.zeros((1, 1, 4), dtype=np.float32)
