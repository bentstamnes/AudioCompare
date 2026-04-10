"""
renderers/waveform.py — GPU waveform texture renderer for MixABTestGPU.

Each track slot owns one WaveformTexture instance.  Every frame (or on
resize/seek) the caller passes updated parameters; the renderer writes into
a numpy RGBA array and uploads it to a Dear PyGui dynamic texture in one
call (set_value).  The texture is then displayed via add_image inside a
child-window drawlist.

Design:
  - numpy-vectorised: no Python loop over pixels.
  - Supports the played/unplayed colour split at an arbitrary pixel column.
  - Supports dim (non-active) rendering.
  - Supports an optional vertical playhead line (white, 2 px wide).
  - Texture is (re)allocated only when width or height changes.
"""

from __future__ import annotations
import numpy as np
import dearpygui.dearpygui as dpg


# RGBA float32 constant — DPG dynamic textures use 0.0–1.0 floats per channel.
_TRANSPARENT = np.zeros(4, dtype=np.float32)


def _hex_to_rgba_f32(hex_color: str, alpha: float = 1.0) -> np.ndarray:
    """Convert '#rrggbb' to a float32 RGBA array [R, G, B, A]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([r / 255.0, g / 255.0, b / 255.0, alpha], dtype=np.float32)


class WaveformTexture:
    """Owns a single DPG dynamic texture and renders waveform data into it."""

    def __init__(
        self,
        texture_registry: int | str,
        width: int,
        height: int,
        bg_color: str,
        active_color: str,
        inactive_color: str,
        dim_color: str,
    ) -> None:
        self._reg     = texture_registry
        self._width   = max(1, width)
        self._height  = max(1, height)

        self._bg       = _hex_to_rgba_f32(bg_color)
        self._active   = _hex_to_rgba_f32(active_color)
        self._inactive = _hex_to_rgba_f32(inactive_color)
        self._dim      = _hex_to_rgba_f32(dim_color)

        # Pre-allocated pixel buffer: (H, W, 4) float32
        self._buf = np.empty((self._height, self._width, 4), dtype=np.float32)

        # DPG texture tag
        flat = np.zeros(self._width * self._height * 4, dtype=np.float32)
        self._tag = dpg.add_dynamic_texture(
            self._width, self._height, flat, parent=self._reg
        )

        # Cached waveform columns: shape (W_wave,) int16 amplitudes in pixels
        self._wave_amps: np.ndarray | None = None
        self._wave_px_count: int = 0   # number of columns that have content
        self._dirty = True

    @property
    def tag(self) -> int | str:
        return self._tag

    def resize(self, width: int, height: int) -> None:
        """Resize the texture (re-allocates buffer and DPG texture)."""
        width  = max(1, width)
        height = max(1, height)
        if width == self._width and height == self._height:
            return
        self._width  = width
        self._height = height
        self._buf    = np.empty((height, width, 4), dtype=np.float32)
        # Replace DPG texture with a new one of the right size
        old_tag = self._tag
        flat = np.zeros(width * height * 4, dtype=np.float32)
        self._tag = dpg.add_dynamic_texture(width, height, flat, parent=self._reg)
        dpg.delete_item(old_tag)
        # Recompute amplitudes at new width
        self._dirty = True

    def set_wave_samples(
        self,
        samples: tuple,          # int16 mono samples at ~4000 Hz
        track_dur: float,
        max_dur: float,
        wave_peak: int,
        wave_global_peak: int,
        normalize: bool,
    ) -> None:
        """Recompute the per-column amplitude array from raw waveform samples."""
        w = self._width
        h = self._height

        if not samples or max_dur <= 0 or track_dur <= 0:
            self._wave_amps   = None
            self._wave_px_count = 0
            self._dirty = True
            return

        px_count = max(1, int(w * min(1.0, track_dur / max_dur)))
        n        = len(samples)
        mid      = h / 2.0
        divisor  = float(wave_global_peak if normalize else (wave_peak or 32768))

        # Vectorised column amplitude computation
        x_arr  = np.arange(px_count, dtype=np.float32)
        i0_arr = (x_arr / px_count * n).astype(np.int32)
        i1_arr = np.maximum(i0_arr + 1, ((x_arr + 1) / px_count * n).astype(np.int32))
        i1_arr = np.minimum(i1_arr, n)

        samps_arr = np.asarray(samples, dtype=np.int32)

        # For each column, take max(|sample|) in [i0, i1)
        # Use a simple loop-free approach: reduce by column boundaries
        # Build a running-max via np.maximum.reduceat
        abs_samps = np.abs(samps_arr)
        col_max   = np.maximum.reduceat(abs_samps, i0_arr)[:px_count]
        amps      = np.maximum(1, (col_max / divisor * mid * 0.85)).astype(np.int32)

        self._wave_amps   = amps
        self._wave_px_count = px_count
        self._dirty = True

    def render(
        self,
        is_active: bool,
        play_px: int,          # pixel column of playhead (-1 = no playhead)
        show_playhead: bool = True,
        playhead_frac: float = 0.0,  # used when play_px < 0 but we still want playhead
    ) -> None:
        """Render into the buffer and upload to GPU.

        Call every frame when parameters change.
        """
        w, h = self._width, self._height
        buf  = self._buf

        # Fill background
        buf[:] = self._bg

        amps      = self._wave_amps
        px_count  = self._wave_px_count

        if amps is not None and px_count > 0:
            mid   = h // 2
            n_col = min(px_count, w)

            if is_active and play_px >= 0:
                # Two colour regions: played / unplayed
                split = min(play_px, n_col)
                if split > 0:
                    _draw_bars(buf, amps[:split], np.arange(split, dtype=np.int32),
                               mid, h, self._active)
                if split < n_col:
                    _draw_bars(buf, amps[split:n_col],
                               np.arange(split, n_col, dtype=np.int32),
                               mid, h, self._inactive)
            else:
                # Dim colour for non-active tracks, inactive for active-but-no-playhead
                clr = self._inactive if is_active else self._dim
                _draw_bars(buf, amps[:n_col], np.arange(n_col, dtype=np.int32),
                           mid, h, clr)

        # Playhead line (2 px wide, white)
        if show_playhead and play_px >= 0:
            px = max(0, min(w - 1, play_px))
            buf[:, max(0, px - 1):px + 1, :] = [1.0, 1.0, 1.0, 0.9]

        # Upload: DPG set_value accepts a numpy array directly (much faster than tolist)
        if dpg.does_item_exist(self._tag):
            dpg.set_value(self._tag, buf.ravel())

    def delete(self) -> None:
        """Remove the DPG texture."""
        try:
            dpg.delete_item(self._tag)
        except Exception:
            pass


def _draw_bars(
    buf: np.ndarray,
    amps: np.ndarray,
    cols: np.ndarray,
    mid: int,
    h: int,
    color: np.ndarray,
) -> None:
    """Draw vertical bars into buf for the given column indices and amplitudes.

    amps[i] is the half-height in pixels for column cols[i].
    Uses numpy fancy indexing — no Python loop.
    """
    if len(amps) == 0:
        return
    tops    = np.maximum(0, mid - amps)      # shape (N,)
    bottoms = np.minimum(h - 1, mid + amps)  # shape (N,)

    # Build index arrays for a vectorised scatter
    # For each column, fill rows tops[i]..bottoms[i] with color.
    # We use a cumsum trick to avoid a per-column loop.
    # Approach: sparse mark array, then cumsum along rows.
    mark = np.zeros((h + 1, buf.shape[1]), dtype=np.int8)
    np.add.at(mark, (tops,  cols), 1)
    np.add.at(mark, (bottoms + 1, cols), -1)
    mask = np.cumsum(mark[:h], axis=0).astype(bool)  # (H, W)

    buf[mask] = color
