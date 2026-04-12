"""
renderers/waveform.py — GPU waveform dual-texture renderer for MixABTestGPU.

Each track slot owns one WaveformTexture instance.  On load (or resize) it
pre-builds two raw GPU textures:

  tag_played  — full waveform drawn in the "played" (active) colour on bg.
  tag_rest    — full waveform drawn in the "inactive" or "dim" colour on bg.

Per-frame: the caller adjusts pmin/pmax/uv_min/uv_max on two draw_image items
so that tag_played covers [0, play_px) and tag_rest covers [play_px, wave_w).
No texture data is transferred to the GPU during playback.

A texture re-upload only happens when:
  - set_wave_samples() is called (new waveform data or scale change).
  - set_active() is called (track switch) — rebuilds tag_rest in correct colour.
  - resize() is called (viewport resize).

Design:
  - Single CPU-side (H, W, 3) float32 scratch buffer reused for both builds.
  - numpy-vectorised bar rendering; pre-allocated mark scratch.
  - add_raw_texture + mvFormat_Float_rgb — 25% smaller than RGBA, bypasses
    DPG safety checks on set_value.
  - is_active passed at construction time to initialise rest colour correctly
    before the first set_wave_samples call.
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
    """Owns two DPG raw textures and renders waveform data into them.

    tag_played: waveform in active colour   (the "played" portion).
    tag_rest:   waveform in inactive/dim colour (the "unplayed" / non-active).

    The caller places two draw_image items in the slot drawlist and updates
    their pmin/pmax/uv_min/uv_max each frame to split at the playhead column.
    No texture upload happens during playback.
    """

    def __init__(
        self,
        texture_registry: int | str,
        width: int,
        height: int,
        bg_color: str,
        active_color: str,
        inactive_color: str,
        dim_color: str,
        is_active: bool = False,
    ) -> None:
        self._reg     = texture_registry
        self._width   = max(1, width)
        self._height  = max(1, height)

        self._bg       = _hex_to_rgb_f32(bg_color)
        self._active   = _hex_to_rgb_f32(active_color)
        self._inactive = _hex_to_rgb_f32(inactive_color)
        self._dim      = _hex_to_rgb_f32(dim_color)

        self._is_active = is_active

        # Separate CPU buffers for each texture — MUST NOT be shared.
        # dpg.set_value defers the GPU upload to render time and stores a
        # reference to the numpy array, not a copy.  If both textures share
        # one buffer, the second set_value call overwrites both.
        self._buf_played = np.empty((self._height, self._width, 3), dtype=np.float32)
        self._buf_rest   = np.empty((self._height, self._width, 3), dtype=np.float32)
        # Pre-allocated mark scratch for _draw_bars — zeroed in-place per call
        self._mark = np.zeros((self._height + 1, self._width), dtype=np.int8)

        # Cached waveform columns: shape (W,) int32 half-heights in pixels
        self._wave_amps: np.ndarray | None = None
        self._wave_px_count: int = 0

        flat = np.zeros(self._width * self._height * 3, dtype=np.float32)
        self._tag_played = dpg.add_raw_texture(
            self._width, self._height, flat,
            format=dpg.mvFormat_Float_rgb,
            parent=self._reg,
        )
        self._tag_rest = dpg.add_raw_texture(
            self._width, self._height, flat.copy(),
            format=dpg.mvFormat_Float_rgb,
            parent=self._reg,
        )

    @property
    def tag_played(self) -> int | str:
        return self._tag_played

    @property
    def tag_rest(self) -> int | str:
        return self._tag_rest

    @property
    def width(self) -> int:
        return self._width

    def set_active(self, is_active: bool) -> None:
        """Update active state and rebuild tag_rest with the correct colour.

        Call once per track-switch for the old and new active slots.
        tag_rest uses inactive colour when active, dim colour when not.
        """
        if is_active == self._is_active:
            return
        self._is_active = is_active
        self._build_upload_rest()

    def resize(self, width: int, height: int) -> None:
        """Resize both textures (re-allocates buffers and DPG textures)."""
        width  = max(1, width)
        height = max(1, height)
        if width == self._width and height == self._height:
            return
        self._width      = width
        self._height     = height
        self._buf_played = np.empty((height, width, 3), dtype=np.float32)
        self._buf_rest   = np.empty((height, width, 3), dtype=np.float32)
        self._mark       = np.zeros((height + 1, width), dtype=np.int8)

        flat = np.zeros(width * height * 3, dtype=np.float32)
        old_played = self._tag_played
        old_rest   = self._tag_rest
        self._tag_played = dpg.add_raw_texture(
            width, height, flat,
            format=dpg.mvFormat_Float_rgb,
            parent=self._reg,
        )
        self._tag_rest = dpg.add_raw_texture(
            width, height, flat.copy(),
            format=dpg.mvFormat_Float_rgb,
            parent=self._reg,
        )
        dpg.delete_item(old_played)
        dpg.delete_item(old_rest)
        self._build_upload_played()
        self._build_upload_rest()

    def set_wave_samples(
        self,
        samples: tuple,          # int16 mono samples at ~4000 Hz
        track_dur: float,
        max_dur: float,
        wave_peak: int,
        wave_global_peak: int,
        normalize: bool,
    ) -> None:
        """Recompute per-column amplitudes and upload both textures."""
        w = self._width
        h = self._height

        if not samples or max_dur <= 0 or track_dur <= 0:
            self._wave_amps     = None
            self._wave_px_count = 0
            self._build_upload_played()
            self._build_upload_rest()
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
        abs_samps = np.abs(samps_arr)
        col_max   = np.maximum.reduceat(abs_samps, i0_arr)[:px_count]
        amps      = np.maximum(1, (col_max / divisor * mid * 0.85)).astype(np.int32)

        self._wave_amps     = amps
        self._wave_px_count = px_count

        self._build_upload_played()
        self._build_upload_rest()

    def render_buffers(
        self,
        samples: tuple,
        track_dur: float,
        max_dur: float,
        wave_peak: int,
        wave_global_peak: int,
        normalize: bool,
    ) -> tuple:
        """Pure numpy: compute waveform and fill two new RGB float32 buffers.

        Allocates fresh arrays instead of using self._buf_* so this method is
        safe to call from a background thread while DPG may still hold a
        deferred reference to the previous self._buf_* arrays.

        Returns (buf_played, buf_rest, wave_amps, wave_px_count) — pass to
        upload_buffers() on the main thread so set_active() can later
        re-render the rest texture on track switches.
        """
        w, h    = self._width, self._height
        buf_p   = np.empty((h, w, 3), dtype=np.float32)
        buf_r   = np.empty((h, w, 3), dtype=np.float32)
        buf_p[:] = self._bg
        buf_r[:] = self._bg

        if not samples or max_dur <= 0 or track_dur <= 0:
            return buf_p, buf_r, None, 0

        px_count = max(1, int(w * min(1.0, track_dur / max_dur)))
        n        = len(samples)
        mid      = h / 2.0
        divisor  = float(wave_global_peak if normalize else (wave_peak or 32768))

        x_arr     = np.arange(px_count, dtype=np.float32)
        i0_arr    = (x_arr / px_count * n).astype(np.int32)
        i1_arr    = np.maximum(i0_arr + 1, ((x_arr + 1) / px_count * n).astype(np.int32))
        i1_arr    = np.minimum(i1_arr, n)
        samps_arr = np.asarray(samples, dtype=np.int32)
        abs_samps = np.abs(samps_arr)
        col_max   = np.maximum.reduceat(abs_samps, i0_arr)[:px_count]
        amps      = np.maximum(1, (col_max / divisor * mid * 0.85)).astype(np.int32)

        mid_i = int(mid)
        n_col = min(px_count, w)
        cols  = np.arange(n_col, dtype=np.int32)
        # Local mark scratch — do not use self._mark (not thread-safe)
        mark  = np.zeros((h + 1, w), dtype=np.int8)

        _draw_bars(buf_p, amps[:n_col], cols, mid_i, h, self._active, mark)
        mark[:] = 0
        clr = self._inactive if self._is_active else self._dim
        _draw_bars(buf_r, amps[:n_col], cols, mid_i, h, clr, mark)

        return buf_p, buf_r, amps, px_count

    def upload_buffers(self, buf_played: np.ndarray, buf_rest: np.ndarray,
                       wave_amps=None, wave_px_count: int = 0) -> None:
        """Upload pre-rendered buffers to GPU. Must be called from the main thread.

        Stores the new arrays as self._buf_* so DPG's deferred reference
        remains valid until the next frame render.
        Also updates self._wave_amps / _wave_px_count so set_active() can
        re-render the rest texture on subsequent track switches.
        """
        if dpg.does_item_exist(self._tag_played):
            dpg.set_value(self._tag_played, buf_played.ravel())
        if dpg.does_item_exist(self._tag_rest):
            dpg.set_value(self._tag_rest, buf_rest.ravel())
        self._buf_played    = buf_played
        self._buf_rest      = buf_rest
        if wave_amps is not None:
            self._wave_amps     = wave_amps
            self._wave_px_count = wave_px_count

    def _build_upload_played(self) -> None:
        """Build full waveform in active colour and upload to tag_played."""
        self._buf_played[:] = self._bg
        if self._wave_amps is not None and self._wave_px_count > 0:
            mid   = self._height // 2
            n_col = min(self._wave_px_count, self._width)
            _draw_bars(self._buf_played, self._wave_amps[:n_col],
                       np.arange(n_col, dtype=np.int32),
                       mid, self._height, self._active, self._mark)
        if dpg.does_item_exist(self._tag_played):
            dpg.set_value(self._tag_played, self._buf_played.ravel())

    def _build_upload_rest(self) -> None:
        """Build full waveform in inactive/dim colour and upload to tag_rest."""
        self._buf_rest[:] = self._bg
        if self._wave_amps is not None and self._wave_px_count > 0:
            mid   = self._height // 2
            n_col = min(self._wave_px_count, self._width)
            clr   = self._inactive if self._is_active else self._dim
            _draw_bars(self._buf_rest, self._wave_amps[:n_col],
                       np.arange(n_col, dtype=np.int32),
                       mid, self._height, clr, self._mark)
        if dpg.does_item_exist(self._tag_rest):
            dpg.set_value(self._tag_rest, self._buf_rest.ravel())

    def delete(self) -> None:
        """Remove both DPG textures."""
        for tag in (self._tag_played, self._tag_rest):
            try:
                dpg.delete_item(tag)
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
