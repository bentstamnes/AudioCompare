"""
renderers/waveform.py — GPU waveform texture renderer for MixABTestGPU.

Each track slot owns one WaveformTexture instance.  Every frame (or on
resize/seek) the caller passes updated parameters; the renderer writes into
a numpy RGB array and uploads it to a Dear PyGui dynamic texture in one
call (set_value).  The texture is then displayed via add_image inside a
child-window drawlist.

Design:
  - numpy-vectorised: no Python loop over pixels.
  - Supports the played/unplayed colour split at an arbitrary pixel column.
  - Supports dim (non-active) rendering.
  - Supports an optional vertical playhead line (white, 2 px wide).
  - Texture is (re)allocated only when width or height changes.
  - Dirty tracking: skips rebuild+upload when nothing visible has changed.
    Inactive tracks upload once (on load/resize) and never again during playback.
  - Raw texture (add_raw_texture) instead of dynamic: skips DPG safety checks on
    every set_value call; designed for large textures updated every frame.
  - float32 RGB (3 ch, mvFormat_Float_rgb) — 25% smaller per upload vs RGBA.
    NOTE: mvFormat_Float_rgb is only valid on raw textures, not dynamic textures.
"""

from __future__ import annotations
import numpy as np
import dearpygui.dearpygui as dpg


def _hex_to_rgb_f32(hex_color: str) -> np.ndarray:
    """Convert '#rrggbb' to a float32 RGB array [R, G, B]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)


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

        self._bg       = _hex_to_rgb_f32(bg_color)
        self._active   = _hex_to_rgb_f32(active_color)
        self._inactive = _hex_to_rgb_f32(inactive_color)
        self._dim      = _hex_to_rgb_f32(dim_color)

        # Pre-allocated pixel buffer: (H, W, 3) float32 — RGB, no alpha channel
        self._buf = np.empty((self._height, self._width, 3), dtype=np.float32)

        # Raw texture: designed for large textures updated every frame.
        # mvFormat_Float_rgb = 3 floats/pixel (25% smaller than RGBA).
        # add_raw_texture skips DPG's safety checks/conversion on every set_value call.
        # NOTE: mvFormat_Float_rgb is only valid on raw textures, not dynamic textures.
        flat = np.zeros(self._width * self._height * 3, dtype=np.float32)
        self._tag = dpg.add_raw_texture(
            self._width, self._height, flat,
            format=dpg.mvFormat_Float_rgb,
            parent=self._reg,
        )

        # Cached waveform columns: shape (W_wave,) int16 amplitudes in pixels
        self._wave_amps: np.ndarray | None = None
        self._wave_px_count: int = 0   # number of columns that have content
        self._dirty = True

        # Dirty-tracking: skip render+upload when nothing changed
        self._last_play_px:   int         = -2   # -2 = sentinel "never rendered"
        self._last_is_active: bool | None = None

        # Pre-allocated scratch buffer for _draw_bars — avoids per-frame heap alloc
        self._mark = np.zeros((self._height + 1, self._width), dtype=np.int8)

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
        self._buf    = np.empty((height, width, 3), dtype=np.float32)
        self._mark   = np.zeros((height + 1, width), dtype=np.int8)
        # Replace DPG texture with a new one of the right size
        old_tag = self._tag
        flat = np.zeros(width * height * 3, dtype=np.float32)
        self._tag = dpg.add_raw_texture(
            width, height, flat,
            format=dpg.mvFormat_Float_rgb,
            parent=self._reg,
        )
        dpg.delete_item(old_tag)
        # Force full re-render next frame
        self._dirty = True
        self._last_play_px   = -2
        self._last_is_active = None

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
            self._wave_amps     = None
            self._wave_px_count = 0
            self._dirty         = True
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
        # Build a running-max via np.maximum.reduceat
        abs_samps = np.abs(samps_arr)
        col_max   = np.maximum.reduceat(abs_samps, i0_arr)[:px_count]
        amps      = np.maximum(1, (col_max / divisor * mid * 0.85)).astype(np.int32)

        self._wave_amps      = amps
        self._wave_px_count  = px_count
        self._dirty          = True
        self._last_play_px   = -2   # force re-render next frame

    def render(
        self,
        is_active: bool,
        play_px: int,          # pixel column of playhead (-1 = no playhead)
        show_playhead: bool = True,
        playhead_frac: float = 0.0,  # used when play_px < 0 but we still want playhead
    ) -> None:
        """Render into the buffer and upload to GPU.

        Skips the rebuild+upload when nothing visible has changed since the last
        call.  Inactive tracks never change during playback (no played/unplayed
        split), so they upload only once after load and on resize.
        """
        # ── Dirty check: skip if nothing changed ─────────────────────────────
        # Inactive tracks: play_px doesn't affect their appearance (single colour).
        # Active track: only the playhead split position matters.
        if not self._dirty:
            effective_px = play_px if is_active else -1
            if effective_px == self._last_play_px and is_active == self._last_is_active:
                return
        self._dirty          = False
        self._last_is_active = is_active
        self._last_play_px   = play_px if is_active else -1

        w, h = self._width, self._height
        buf  = self._buf

        # Fill background
        buf[:] = self._bg

        amps     = self._wave_amps
        px_count = self._wave_px_count

        if amps is not None and px_count > 0:
            mid   = h // 2
            n_col = min(px_count, w)

            if is_active and play_px >= 0:
                # Two colour regions: played / unplayed
                split = min(play_px, n_col)
                if split > 0:
                    _draw_bars(buf, amps[:split], np.arange(split, dtype=np.int32),
                               mid, h, self._active, self._mark)
                if split < n_col:
                    _draw_bars(buf, amps[split:n_col],
                               np.arange(split, n_col, dtype=np.int32),
                               mid, h, self._inactive, self._mark)
            else:
                # Dim colour for non-active tracks, inactive for active-but-no-playhead
                clr = self._inactive if is_active else self._dim
                _draw_bars(buf, amps[:n_col], np.arange(n_col, dtype=np.int32),
                           mid, h, clr, self._mark)

        # Playhead line (2 px wide, white)
        if show_playhead and play_px >= 0:
            px = max(0, min(w - 1, play_px))
            buf[:, max(0, px - 1):px + 1, :] = [1.0, 1.0, 1.0]

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
    mark: np.ndarray | None = None,
) -> None:
    """Draw vertical bars into buf for the given column indices and amplitudes.

    amps[i] is the half-height in pixels for column cols[i].
    Uses numpy fancy indexing — no Python loop.

    mark: pre-allocated (h+1, W) int8 scratch buffer.  If supplied it is
    zeroed and reused, avoiding a heap allocation on every call.
    """
    if len(amps) == 0:
        return
    tops    = np.maximum(0, mid - amps)      # shape (N,)
    bottoms = np.minimum(h - 1, mid + amps)  # shape (N,)

    # Sparse mark + cumsum: avoids a per-column loop.
    if mark is not None and mark.shape == (h + 1, buf.shape[1]):
        mark[:] = 0   # clear pre-allocated buffer in-place (no allocation)
    else:
        mark = np.zeros((h + 1, buf.shape[1]), dtype=np.int8)
    np.add.at(mark, (tops,  cols), 1)
    np.add.at(mark, (bottoms + 1, cols), -1)
    mask = np.cumsum(mark[:h], axis=0).astype(bool)  # (H, W)

    buf[mask] = color
