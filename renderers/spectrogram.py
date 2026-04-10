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
_SAMPLERATE        = 48000   # must match player.py SAMPLERATE
_FS_FREQ_LO        = 30.0    # Hz — bottom of fullscreen display
_FS_FREQ_HI        = 20000.0 # Hz — top of fullscreen display

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
        scroll_step: int = 2,
    ) -> None:
        self._dl          = drawlist_tag
        self._width       = max(1, width)
        self._height      = max(1, height)
        self._visible     = False   # hidden by default; enabled via set_visible
        self._scroll_step = max(1, scroll_step)

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

    def update(self, raw_bands: np.ndarray | None, n_bands: int,
               raw_bins: bool = False) -> None:
        """Scroll buffer left by one pixel and paint new column from raw_bands.

        raw_bins=False (default): raw_bands is dBFS band averages, length n_bands.
        raw_bins=True  (fullscreen): raw_bands is linear reassigned magnitudes,
            length = FFT_SIZE//2+1.  Each pixel row is mapped to its exact
            frequency via log-frequency interpolation — no band quantization.
        """
        if not self._visible:
            return

        w  = self._width
        h  = self._height
        br = self._brightness

        # Scroll: shift all columns left by scroll_step
        s = self._scroll_step
        if w > s:
            self._buf[:, :w - s, :] = self._buf[:, s:, :]
        self._buf[:, w - s:, :] = 0.0   # clear the newly exposed columns

        # Build the new rightmost column
        col = np.zeros((h, 4), dtype=np.float32)

        if raw_bands is not None and len(raw_bands) > 0:
            # t=0.0 at top row (high freq), t=1.0 at bottom row (low freq).
            # Pure log-frequency mapping — equal pixel density per octave.
            row_indices = np.arange(h, dtype=np.float32)
            t_mapped    = row_indices / max(h - 1, 1)

            if raw_bins:
                # ── Fullscreen row→bin interpolation ──────────────────────────
                # For each display row, compute its exact (fractional) FFT bin
                # and linearly interpolate the reassigned spectrum.  Fills every
                # row smoothly regardless of how many bins map to a given range.
                n_bins_r = len(raw_bands)
                n_fft_r  = (n_bins_r - 1) * 2

                # t_mapped is the log-frequency coordinate:
                #   0 = top row / FREQ_HI,  1 = bottom row / FREQ_LO.
                log_range  = np.log(_FS_FREQ_HI / _FS_FREQ_LO)
                freq_row   = _FS_FREQ_HI * np.exp(-t_mapped * log_range)
                freq_row   = np.clip(freq_row, _FS_FREQ_LO, _FS_FREQ_HI)

                # Fractional bin for each row
                bin_f  = freq_row * n_fft_r / _SAMPLERATE
                bin_f  = np.clip(bin_f, 0.0, float(n_bins_r - 1))
                bin_lo = np.clip(np.floor(bin_f).astype(np.int32), 0, n_bins_r - 1)
                bin_hi = np.clip(bin_lo + 1,                        0, n_bins_r - 1)
                frac_b = (bin_f - np.floor(bin_f)).astype(np.float32)

                # Interpolate raw linear magnitudes, THEN convert to dBFS.
                # Interpolating in linear space before log gives correct envelope.
                interp_lin = (raw_bands[bin_lo] * (1.0 - frac_b)
                              + raw_bands[bin_hi] * frac_b)
                db_row  = 20.0 * np.log10(np.maximum(interp_lin, 1e-6))
                amp     = np.clip(
                    (db_row - _SPECTRUM_DB_FLOOR) / (-_SPECTRUM_DB_FLOOR),
                    0.0, 1.0,
                ).astype(np.float32)
            else:
                # ── Normal band mode ──────────────────────────────────────────
                nb = len(raw_bands)
                # Convert dB bands to 0..1 fractions
                fracs = np.clip(
                    (raw_bands - _SPECTRUM_DB_FLOOR) / (-_SPECTRUM_DB_FLOOR),
                    0.0, 1.0,
                ).astype(np.float32)
                # band_frac 0 (top) → high band (nb-1); 1 (bottom) → low band (0)
                band_for_row = np.clip(
                    ((1.0 - t_mapped) * nb).astype(np.int32), 0, nb - 1
                )
                amp = fracs[band_for_row]   # shape: (h,)

            # Look up RGB from LUT
            lut_idx = np.clip((amp * 255.0).astype(np.int32), 0, 255)
            rgb = _LUT[lut_idx]   # shape: (h, 3)

            col[:, 0] = rgb[:, 0] * br
            col[:, 1] = rgb[:, 1] * br
            col[:, 2] = rgb[:, 2] * br
            col[:, 3] = 0.92   # always opaque; LUT maps silence to black

        # Write new column into all scroll_step newly exposed slots
        self._buf[:, w - self._scroll_step:, :] = col[:, np.newaxis, :]
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

    def get_buffer(self) -> np.ndarray:
        """Return a copy of the current RGBA pixel buffer (height, width, 4)."""
        return self._buf.copy()

    def seed_from_buffer(self, src: np.ndarray) -> None:
        """Bilinearly resize src (any height, same width, 4ch) into this buffer
        and upload to GPU.  Used to pre-populate fullscreen from slot strip."""
        sh, sw = src.shape[:2]
        dh, dw = self._height, self._width
        if sw != dw:
            # Width mismatch — just clear rather than stretch horizontally
            self._buf[:] = 0.0
        else:
            row_f  = np.linspace(0.0, sh - 1, dh, dtype=np.float32)
            row_lo = np.clip(np.floor(row_f).astype(np.int32), 0, sh - 1)
            row_hi = np.clip(row_lo + 1,                        0, sh - 1)
            frac   = row_f - np.floor(row_f)
            frac   = frac[:, np.newaxis, np.newaxis].astype(np.float32)
            self._buf[:] = src[row_lo] * (1.0 - frac) + src[row_hi] * frac
        try:
            dpg.set_value(self._tex, self._buf.ravel())
        except Exception:
            pass

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
