"""
renderers/spectrogram.py — Pre-computed static spectrogram texture with UV scrolling.

Architecture:

  OLD (real-time scrolling): add_dynamic_texture + scroll 57MB buffer + set_value
       every tick = ~53ms per _apply_spectrum call → 31fps at 2K with 1 track.

  NEW: Full-file spectrogram computed ONCE in a background thread at load time.
       The full column count is split into tiles of _N_COLS_MAX columns each —
       each tile is a separate GPU texture (avoids the 16 384 px GPU texture limit).
       During playback: zero GPU upload — only uv_min/uv_max and pmin/pmax are
       updated via one configure_item call per visible tile per frame.

  Tiling: at HOP_SIZE=256 a 4-min track produces ~40 000 columns = 4 tiles.
  compute_static() processes one tile at a time (peak RAM ~160 MB per tile
  instead of ~655 MB for the full matrix), then returns a list of RGBA arrays.

  Why it looks like scrolling: the UV window ends at the current playhead
  position. As pos_norm advances, the window slides right → spectrogram scrolls
  left, playhead stays pinned to the right edge.

Frequency layout: low freq at bottom (row height-1), high freq at top (row 0).
Color: fixed heat palette — silence→black → purple → magenta → orange.
"""

from __future__ import annotations
import numpy as np
import dearpygui.dearpygui as dpg

# pyfftw — if installed, use FFTW3 for batch rfft (3-5× faster than numpy for
# the (n_cols, 4096) workload in compute_static).  The interface cache keeps
# the FFTW plan alive across tiles and tracks so the planning cost is paid once
# per unique array shape.  Falls back to numpy.fft transparently if not installed.
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fft
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
    # Allow FFTW to search for the best plan (takes ~0.5 s first call per shape,
    # but shapes are fixed after n_cols is capped, so this pays off immediately).
    pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
except ImportError:
    import numpy.fft as _fft  # type: ignore[assignment]

_SPECTRUM_DB_FLOOR = -70.0   # dBFS floor — also the colormap black point
_FS_FREQ_LO        = 20.0    # Hz — bottom of display
_FS_FREQ_HI        = 20000.0 # Hz — top of display

# Long FFT: 4096 @ 44100 Hz → 93 ms window, 10.7 Hz/bin.
_FFT_SIZE          = 4096

# Hop size: 256 samples @ 44100 Hz → 5.8 ms/column.
# A 100 ms kick drum spans ~17 columns — clearly connected on screen.
_HOP_SIZE          = 256

# Maximum decode sample rate.
_SR_MAX            = 48000

# Tile width — maximum GPU texture width.  Each tile is one DPG raw texture.
# GPU texture width limit is 16 384 px; 10 000 is safe on all hardware.
# compute_static() splits the full column count into tiles of this size.
_N_COLS_MAX        = 10000

# ── Heat palette ─────────────────────────────────────────────────────────────
_PALETTE_STOPS = np.array([
    [0.00,  0.00, 0.00, 0.00],
    [0.09,  0.12, 0.00, 0.22],
    [0.24,  0.55, 0.00, 0.65],
    [0.38,  0.85, 0.00, 0.75],
    [0.55,  0.95, 0.05, 0.15],
    [0.69,  1.00, 0.35, 0.00],
    [0.83,  1.00, 0.70, 0.00],
    [0.96,  1.00, 0.95, 0.40],
], dtype=np.float32)


def _build_lut(n: int = 256) -> np.ndarray:
    xs  = _PALETTE_STOPS[:, 0]
    rgb = _PALETTE_STOPS[:, 1:]
    t   = np.linspace(0.0, 1.0, n, dtype=np.float32)
    lut = np.empty((n, 3), dtype=np.float32)
    for ch in range(3):
        lut[:, ch] = np.interp(t, xs, rgb[:, ch])
    return lut


_LUT      = _build_lut(256)
_N_BINS   = _FFT_SIZE // 2 + 1
_FFT_NORM = float(_FFT_SIZE // 4)

# Reassignment windows — long FFT (4096) computed once at import
_n_arr   = np.arange(_FFT_SIZE, dtype=np.float32)
_HANN    = np.hanning(_FFT_SIZE).astype(np.float32)
_HANN_DH = np.gradient(_HANN).astype(np.float32)
_HANN_TH = (_n_arr * _HANN).astype(np.float32)

# ── Short FFT constants (transient/onset layer) ───────────────────────────────
_FFT_SIZE_S  = 1024
_N_BINS_S    = _FFT_SIZE_S // 2 + 1              # 513
_FFT_NORM_S  = float(_FFT_SIZE_S // 4)
_HANN_S      = np.hanning(_FFT_SIZE_S).astype(np.float32)
# Short layer blend weight.  Uses spectral flux onset (frame-difference) instead
# of EMA background subtraction.  Flux is sweep-immune: a sweep rising through a
# 43 Hz bin over ~4 frames produces a per-frame delta of ~0.25× peak — well below
# the rel_gate threshold of 0.40.  A genuine transient (cymbal, hi-hat, snare
# crack) produces a delta of ~1.0× in a single frame.  Tune upward to add more
# transient brightness; downward if short-layer noise becomes audible.
_SHORT_WEIGHT = 1.0

# Pre-scatter per-column relative gate (long layer only).
# Each bin must exceed this fraction of the column's peak magnitude to be
# scattered.  This is a global per-column test (not per-band), so harmonics
# that are weaker than the fundamental are suppressed even when they are the
# only content in their frequency band — something the per-band gate cannot do.
# Raise to suppress more harmonics; lower to keep more harmonic detail.
_PRE_SCATTER_GATE = 0.05

# Post-scatter per-band gate.
# Applied after scatter-accumulate: each output row must exceed this fraction
# of its frequency band's per-column maximum.  Set to 0.0 (disabled) because
# the gate cannot distinguish between a weak sustained note and noise — it
# suppresses both based purely on relative amplitude within the band.  This
# caused systematic dropout on sub-bass sustained notes (e.g. 43 Hz) whenever
# a louder signal existed elsewhere in the same frequency band.  The absolute
# magnitude gate (_MAG_GATE) and pre-scatter relative gate (_PRE_SCATTER_GATE)
# provide sufficient noise suppression without this inter-frequency competition.
_N_BANDS          = 6
_BAND_GATE_THRESH = 0.0   # disabled — see note above

# Causal group-delay gate: maximum number of samples the energy centroid may
# sit ahead of the window centre before the bin is rejected.  Tighter = fewer
# precursor ghost lines on sweeps; too tight = spurious dropout on sustained
# notes whose amplitude rises or phase-rotates within the 93 ms analysis window.
# FFT_SIZE//4 (1024 samples = 23 ms) eliminates spurious rejections on sustained
# notes (observed t_hat max = 595 samples) while still catching true sweep
# precursors whose t_hat approaches FFT_SIZE//2 = 2048.
_CAUSAL_GATE_SAMPLES = _FFT_SIZE // 4   # 1024 samples = 23 ms

# Maximum half-bandwidth in display rows for the bandwidth-aware scatter.
# Each bin scatters to rows [row_hat ± _BW_MAX_ROWS] with a triangular weight
# profile.  The nominal log-scale bandwidth of sub-bass bins can reach ~18 rows
# at height=300, which makes low-frequency lines appear too tall.  Capping at
# 1.5 rows gives lines up to 3 px wide — close to the reference look while
# still providing smooth, gap-free rendering of sustained notes.
_BW_MAX_ROWS = 1.5

# ── Log-frequency axis ────────────────────────────────────────────────────────
_LOG_RANGE = float(np.log(_FS_FREQ_HI / _FS_FREQ_LO))   # ln(20000/20) ≈ 6.9

# ── Spectral tilt ─────────────────────────────────────────────────────────────
# 9.0 dB/decade boosts high-frequency content enough to match the density of
# reference analysers on typical EDM material (which rolls off ~6-12 dB/oct).
_TILT_DB_PER_DECADE = 9.0
_TILT_F_REF         = 1000.0

# ── Frequency-dependent gate crossover ────────────────────────────────────────
# Both the causal gate and the pre-scatter gate are applied at full strength
# below _GATE_F_FULL Hz and fade linearly to zero at _GATE_F_OFF Hz.
# This keeps the low end clean (sweep-precursor rejection, harmonic suppression)
# while allowing transient mid/high content (cymbals, hi-hats, overtones) to
# pass through without being masked by loud bass hits.
_GATE_F_FULL = 500.0    # Hz — gates at 100 % below this
_GATE_F_OFF  = 4000.0   # Hz — gates fully disabled above this

_cached_row_map: dict = {}


def _get_row_map(height: int, sample_rate: int):
    key = (height, sample_rate)
    if key in _cached_row_map:
        return _cached_row_map[key]
    t         = np.arange(height, dtype=np.float32) / max(height - 1, 1)
    freq_row  = _FS_FREQ_HI * np.exp(-t * _LOG_RANGE)
    freq_row  = np.clip(freq_row, _FS_FREQ_LO, _FS_FREQ_HI)
    n_fft     = (_N_BINS - 1) * 2
    bin_f     = np.clip(freq_row * n_fft / sample_rate, 0.0, float(_N_BINS - 1))
    bin_lo    = np.clip(np.floor(bin_f).astype(np.int32), 0, _N_BINS - 1)
    bin_hi    = np.clip(bin_lo + 1,                        0, _N_BINS - 1)
    frac      = (bin_f - np.floor(bin_f)).astype(np.float32)
    result    = (bin_lo, bin_hi, frac)
    _cached_row_map[key] = result
    return result


def decode_sample_rate(native_rate: int) -> int:
    return min(native_rate, _SR_MAX)


def n_cols_for_samples(n_samp: int, wave_w: int) -> int:
    """Return total column count for a file with n_samp samples.

    Capped at wave_w * _MAX_DISPLAY_ZOOM so we never compute more FFT frames
    than the display can show at maximum zoom.  The UI offers zoom levels 0
    (full-track), 1×, 2×, 4× — at zoom=4 the visible window is wave_w/4
    texture columns wide, so wave_w*4 columns gives exactly 1 col-per-pixel
    at the highest available zoom.  Computing beyond this threshold produces no
    visible improvement because the GPU bilinear-filters surplus columns away.
    """
    _MAX_DISPLAY_ZOOM = 4          # highest zoom level offered in the UI
    fft_frames = max(1, n_samp // _HOP_SIZE)
    cap        = max(wave_w, wave_w * _MAX_DISPLAY_ZOOM)
    return min(fft_frames, cap)


def compute_static(
    samples,
    n_cols: int,
    height: int,
    sample_rate: int,
) -> list:
    """Pre-compute the full-file spectrogram as a list of (height, tile_cols, 4) float32 arrays.

    Returns a list of one or more RGBA tiles, each at most _N_COLS_MAX columns wide.
    Processing is tile-by-tile so peak RAM stays at ~160 MB per tile regardless of
    file length (avoids the ~655 MB frame matrix for a full 40 000-column file).

    The EMA background state for the short-layer onset filter is carried between
    tiles so transient detection is continuous across tile boundaries.
    """
    n_cols  = max(1, n_cols)
    height  = max(1, height)
    arr     = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    n_samp  = len(arr)

    if n_samp < _FFT_SIZE:
        return [np.zeros((height, min(n_cols, _N_COLS_MAX), 4), dtype=np.float32)]

    # All column centres evenly spaced across the file
    centres = np.linspace(
        _FFT_SIZE // 2, n_samp - _FFT_SIZE // 2 - 1,
        n_cols, dtype=np.float32,
    ).astype(np.int32)

    _MAG_GATE = 10.0 ** (_SPECTRUM_DB_FLOOR / 20.0)

    # ── Shared precomputed quantities (same for every tile) ───────────────────

    # Spectral tilt per display row
    row_t     = np.arange(height, dtype=np.float32) / max(height - 1, 1)
    freq_rows = (_FS_FREQ_HI * np.exp(-row_t * _LOG_RANGE)).astype(np.float32)
    tilt_db   = (
        _TILT_DB_PER_DECADE * np.log10(np.maximum(freq_rows, 1.0) / _TILT_F_REF)
    ).astype(np.float32)                                              # (height,)

    # Short-layer: nominal bin → log-freq row (same for all columns/tiles)
    k_s_all   = np.arange(_N_BINS_S, dtype=np.float32)
    k_s       = k_s_all[None, :]                                      # (1, N_BINS_S)
    freq_s    = np.clip(k_s * (sample_rate / float(_FFT_SIZE_S)),
                        _FS_FREQ_LO, _FS_FREQ_HI)
    t_row_s   = np.log(_FS_FREQ_HI / freq_s) / _LOG_RANGE
    row_idx_s = np.clip(
        np.round(t_row_s * (height - 1)).astype(np.int32), 0, height - 1
    )                                                                  # (1, N_BINS_S)

    # Short-layer frequency floor: suppress bins below 300 Hz.
    # The 43 Hz/bin short FFT can't resolve sub-bass harmonics; frame-to-frame
    # phase jitter on sustained bass still produces onset diffs above gate.
    freq_ok   = (k_s_all * (sample_rate / float(_FFT_SIZE_S))) >= 300.0  # (N_BINS_S,)

    # Previous-frame spectrum for spectral flux onset; carried across tiles.
    prev_frame_s = None   # shape (N_BINS_S,) after first column

    # ── Bandwidth-aware scatter precomputation ────────────────────────────────
    # Each FFT bin k nominally owns the frequency band [f_{k-0.5}, f_{k+0.5}].
    # On a log-scale display these bands map to row ranges that grow wider as
    # frequency decreases (up to ~18 rows at 300 px height, ~63 at 1080 px).
    # Instead of scattering each bin to just 2 rows via bilinear interpolation,
    # we scatter to all rows in the band with a triangular weight profile
    # centred at the reassigned frequency.  This fills inter-bin gaps at the
    # scatter step — no post-processing required.
    _k_arr    = np.arange(_N_BINS, dtype=np.float32)
    _f_lo_k   = np.maximum((_k_arr - 0.5) * (sample_rate / float(_FFT_SIZE)), _FS_FREQ_LO)
    _f_hi_k   = np.minimum((_k_arr + 0.5) * (sample_rate / float(_FFT_SIZE)), _FS_FREQ_HI)
    # clip before log so DC-and-below bins map cleanly to the bottom row
    _f_lo_k   = np.clip(_f_lo_k, _FS_FREQ_LO, _FS_FREQ_HI)
    _f_hi_k   = np.clip(_f_hi_k, _FS_FREQ_LO, _FS_FREQ_HI)
    # row_upper = row for the HIGH-frequency edge of each bin (smaller row idx)
    # row_lower = row for the LOW-frequency edge (larger row idx)
    _row_upper_k = (np.log(_FS_FREQ_HI / _f_hi_k) / _LOG_RANGE * (height - 1)).astype(np.float32)
    _row_lower_k = (np.log(_FS_FREQ_HI / _f_lo_k) / _LOG_RANGE * (height - 1)).astype(np.float32)
    # Half-bandwidth in display rows: min 0.5 (always fill ≥1 row), max _BW_MAX_ROWS
    _half_bw_k   = np.clip((_row_lower_k - _row_upper_k) / 2.0, 0.5, _BW_MAX_ROWS).astype(np.float32)

    # Per-bin gate strength: 1.0 at f ≤ _GATE_F_FULL, 0.0 at f ≥ _GATE_F_OFF
    _bin_freq_k        = _k_arr * (sample_rate / float(_FFT_SIZE))
    _gate_weight_k     = np.clip(
        (_GATE_F_OFF - _bin_freq_k) / (_GATE_F_OFF - _GATE_F_FULL), 0.0, 1.0
    ).astype(np.float32)                                               # (N_BINS,)
    # Causal limit per bin: _CAUSAL_GATE_SAMPLES at low freq → _FFT_SIZE (off) at high freq
    _causal_limit_k    = (
        _gate_weight_k * _CAUSAL_GATE_SAMPLES +
        (1.0 - _gate_weight_k) * _FFT_SIZE
    ).astype(np.float32)                                               # (N_BINS,)
    # Pre-scatter threshold per bin: _PRE_SCATTER_GATE at low freq → 0.0 at high freq
    _pre_gate_k        = (_gate_weight_k * _PRE_SCATTER_GATE).astype(np.float32)  # (N_BINS,)

    # ── Tile loop ─────────────────────────────────────────────────────────────
    tiles = []

    for t_start in range(0, n_cols, _N_COLS_MAX):
        t_end   = min(t_start + _N_COLS_MAX, n_cols)
        t_ncols = t_end - t_start
        tc      = centres[t_start:t_end]                              # (t_ncols,)

        # ── Long FFT (4096, reassigned) ───────────────────────────────────────
        starts  = np.clip(tc - _FFT_SIZE // 2, 0, n_samp - _FFT_SIZE)
        indices = starts[:, None] + np.arange(_FFT_SIZE, dtype=np.int32)[None, :]
        raw     = arr[indices]                                        # (t_ncols, FFT_SIZE)

        X_h     = _fft.rfft(raw * _HANN[None, :],    axis=1)
        X_dh    = _fft.rfft(raw * _HANN_DH[None, :], axis=1)
        X_th    = _fft.rfft(raw * _HANN_TH[None, :], axis=1)
        mag     = (np.abs(X_h) / _FFT_NORM).astype(np.float32)

        denom     = np.where(np.abs(X_h) > 1e-10, X_h, 1e-10)
        ratio     = X_dh / denom
        k_bins    = np.arange(_N_BINS, dtype=np.float32)[None, :]
        omega_hat = (k_bins - np.imag(ratio).astype(np.float32)
                     * (_FFT_SIZE / (2.0 * np.pi)))
        omega_hat = np.clip(omega_hat, 0.0, float(_N_BINS - 1))

        # Group delay gate: Re(X_th/X_h) is the energy centroid in sample position
        # within the frame [0, N-1]; subtract N/2 to get offset from window center.
        # Bins with large positive offset have their energy in the future half of the
        # window — for a frequency sweep these appear as horizontal precursor lines
        # to the left of the main sweep line.  Reject them.
        t_hat_offset = (np.real(X_th / denom) - _FFT_SIZE // 2).astype(np.float32)
        # Per-bin causal gate: tight at low frequencies (sweep precursor rejection),
        # fades to disabled above _GATE_F_OFF so transient cymbal/hi-hat bins pass.
        causal       = (t_hat_offset <= _causal_limit_k[None, :]).ravel()

        # Per-bin pre-scatter gate: full strength at low frequencies (keeps low end
        # clean when loud bass hits dominate the column peak), fades to zero above
        # _GATE_F_OFF so weak high-frequency content is never masked by the bass.
        col_peak_l = mag.max(axis=1, keepdims=True)           # (t_ncols, 1)
        pre_gate   = (mag > col_peak_l * _pre_gate_k[None, :]).ravel()

        valid     = (mag > _MAG_GATE).ravel() & causal & pre_gate

        freq_hat  = omega_hat * (sample_rate / float(_FFT_SIZE))
        freq_hat  = np.clip(freq_hat, _FS_FREQ_LO, _FS_FREQ_HI)
        t_row     = np.log(_FS_FREQ_HI / freq_hat) / _LOG_RANGE
        row_f     = (t_row * (height - 1)).astype(np.float32)            # reassigned row

        # ── Bandwidth-aware scatter ───────────────────────────────────────────
        # Each valid (col, bin) scatter point fills all display rows within its
        # nominal frequency band with a triangular weight profile centred at the
        # reassigned row.  This eliminates inter-bin gaps on the log scale
        # without any post-processing pass.
        #
        # valid index i encodes: col = i // _N_BINS, bin = i % _N_BINS
        valid_idx  = np.where(valid)[0]
        size       = height * t_ncols

        if len(valid_idx) == 0:
            out_long = np.zeros((height, t_ncols), dtype=np.float32)
        else:
            k_v        = valid_idx % _N_BINS                             # bin index
            col_v      = (valid_idx // _N_BINS).astype(np.int32)        # column index
            mag_v      = ((mag ** 1.5).ravel())[valid_idx]              # scatter magnitude
            row_hat_v  = row_f.ravel()[valid_idx]                       # reassigned centre row
            hbw_v      = _half_bw_k[k_v]                               # half-bandwidth (rows)

            # Row range to fill: centred at reassigned position, ±half_bandwidth
            fill_lo_v  = np.clip(np.floor(row_hat_v - hbw_v).astype(np.int32), 0, height - 1)
            fill_hi_v  = np.clip(np.ceil (row_hat_v + hbw_v).astype(np.int32), 0, height - 1)
            n_rep_v    = np.maximum(1, fill_hi_v - fill_lo_v + 1)      # rows per scatter point
            total_ops  = int(n_rep_v.sum())

            # Expand each scatter point to n_rep_v rows (vectorised arange trick)
            col_rep     = np.repeat(col_v,       n_rep_v)
            mag_rep     = np.repeat(mag_v,       n_rep_v)
            fill_lo_rep = np.repeat(fill_lo_v,   n_rep_v)
            row_hat_rep = np.repeat(row_hat_v,   n_rep_v)
            hbw_rep     = np.repeat(hbw_v,       n_rep_v)

            # Per-row offset within each group (0, 1, 2, ..., n_rep_v[i]-1)
            cum_n      = np.empty(len(n_rep_v) + 1, dtype=np.int64)
            cum_n[0]   = 0
            np.cumsum(n_rep_v.astype(np.int64), out=cum_n[1:])
            positions  = np.arange(total_ops, dtype=np.int64)
            grp_start  = np.repeat(cum_n[:-1], n_rep_v)
            offsets    = (positions - grp_start).astype(np.int32)

            row_rep    = np.clip(fill_lo_rep + offsets, 0, height - 1)

            # Triangular weight: 1.0 at reassigned centre, 0.0 at bandwidth edge
            dist   = np.abs(row_rep.astype(np.float32) + 0.5 - row_hat_rep)
            weight = np.maximum(0.0, 1.0 - dist / np.maximum(hbw_rep, 0.5))

            flat_rep = (row_rep.astype(np.int64) * t_ncols + col_rep.astype(np.int64))
            out_long = np.bincount(
                flat_rep,
                weights=(mag_rep * weight).astype(np.float64),
                minlength=size,
            ).reshape(height, t_ncols).astype(np.float32)

        # ── Short FFT (1024, onset layer) ─────────────────────────────────────
        # Skipped entirely when disabled (_SHORT_WEIGHT == 0.0).
        if _SHORT_WEIGHT > 0.0:
            starts_s  = np.clip(tc - _FFT_SIZE_S // 2, 0, n_samp - _FFT_SIZE_S)
            idx_s     = starts_s[:, None] + np.arange(_FFT_SIZE_S, dtype=np.int32)[None, :]
            raw_s     = arr[idx_s]                                        # (t_ncols, FFT_SIZE_S)
            X_s       = _fft.rfft(raw_s * _HANN_S[None, :], axis=1)
            mag_s     = (np.abs(X_s) / _FFT_NORM_S).astype(np.float32)  # (t_ncols, N_BINS_S)

            # Spectral flux onset: onset[t] = max(mag[t] - mag[t-1], 0).
            # Sweep-immune: a sweep crossing a 43 Hz bin over ~4 frames produces
            # a per-frame delta of ~0.25× peak (below the rel_gate of 0.40).
            # A genuine transient rises from near-zero in one frame → delta ≈ 1.0×.
            # Fully vectorised — no Python loop, no GIL hold.
            prev        = prev_frame_s if prev_frame_s is not None else np.zeros(_N_BINS_S, dtype=np.float32)
            stacked     = np.vstack([prev[None, :], mag_s])              # (t_ncols+1, N_BINS_S)
            mag_s_onset = np.maximum(stacked[1:] - stacked[:-1], 0.0).astype(np.float32)
            prev_frame_s = mag_s[-1].copy()

            freq_ok_full = np.tile(freq_ok, t_ncols)
            rel_gate     = mag_s_onset > 0.40 * mag_s
            valid_s      = (mag_s_onset > _MAG_GATE).ravel() & freq_ok_full & rel_gate.ravel()

            col_flat_s   = np.repeat(np.arange(t_ncols, dtype=np.int32), _N_BINS_S)
            row_flat_s   = np.tile(row_idx_s.ravel(), t_ncols)
            scatter_s    = mag_s_onset.ravel()

            flat_s = (row_flat_s[valid_s] * t_ncols + col_flat_s[valid_s]).astype(np.int64)
            out_short = np.bincount(
                flat_s, weights=scatter_s[valid_s].astype(np.float64), minlength=size
            ).reshape(height, t_ncols).astype(np.float32)

            out_lin = out_long + _SHORT_WEIGHT * out_short
        else:
            # ── Blend ─────────────────────────────────────────────────────────
            out_lin = out_long

        # ── Per-band per-column gate ───────────────────────────────────────────
        # Applied after blending, on the accumulated out_lin.  Each row must
        # exceed this fraction of its band's column max to survive.
        band_h = max(1, height // _N_BANDS)
        for b in range(_N_BANDS):
            r0   = b * band_h
            r1   = height if b == _N_BANDS - 1 else r0 + band_h
            band = out_lin[r0:r1, :]
            bmax = np.maximum(band.max(axis=0, keepdims=True), 1e-10)
            out_lin[r0:r1, :] *= (band > bmax * _BAND_GATE_THRESH)

        # ── dBFS + spectral tilt → LUT → RGBA ─────────────────────────────────
        db       = 20.0 * np.log10(np.maximum(out_lin, 1e-7))
        db       = db + tilt_db[:, np.newaxis]
        amp_rows = np.clip(
            (db - _SPECTRUM_DB_FLOOR) / (-_SPECTRUM_DB_FLOOR), 0.0, 1.0
        ).astype(np.float32)
        lut_idx  = np.clip((amp_rows * 255.0).astype(np.int32), 0, 255)
        rgb      = _LUT[lut_idx]

        out          = np.empty((height, t_ncols, 4), dtype=np.float32)
        out[..., :3] = rgb
        out[..., 3]  = 0.9
        tiles.append(out)

    return tiles


# ── SpectrogramStrip ─────────────────────────────────────────────────────────

class SpectrogramStrip:
    """UV-scrolling spectrogram overlay backed by one GPU texture per tile.

    compute_static() returns a list of RGBA tiles.  set_data() uploads each tile
    as a separate DPG raw texture and creates a draw_image per tile.  During
    playback, set_scroll() updates only the 1-2 tiles currently visible — all
    other tiles receive a zero-width pmin/pmax so they are invisible with no
    GPU cost.
    """

    def __init__(
        self,
        tex_registry: int | str,
        drawlist_tag: int | str,
        width: int,
        height: int,
        active_color: str,
        scroll_step: int = 2,
        x_offset: int = 0,
    ) -> None:
        self._wave_w        = max(1, width)
        self._height        = max(1, height)
        self._x_offset      = x_offset
        self._visible       = False
        self._tex_registry  = tex_registry
        self._drawlist_tag  = drawlist_tag
        self._zoom          = 1.0
        self._total_cols    = self._wave_w

        # Placeholder: one invisible tile until set_data() arrives
        flat = np.zeros(self._wave_w * self._height * 4, dtype=np.float32)
        tag  = dpg.add_raw_texture(
            self._wave_w, self._height, flat,
            format=dpg.mvFormat_Float_rgba,
            parent=tex_registry,
        )
        img  = dpg.draw_image(
            tag,
            pmin=(x_offset, 0), pmax=(x_offset, self._height),
            uv_min=(0, 0), uv_max=(0, 1),
            show=False,
            parent=drawlist_tag,
        )
        # Each entry: {'tex': tag, 'img': img, 't_start': int, 't_ncols': int}
        self._tile_items = [{'tex': tag, 'img': img, 't_start': 0, 't_ncols': self._wave_w}]

    # ── Public API ────────────────────────────────────────────────────────────

    def set_data(self, tiles) -> None:
        """Upload pre-computed tile list from compute_static().

        tiles: List[np.ndarray], each (height, tile_cols, 4) float32.
        Also accepts a single ndarray for backward compatibility.
        """
        if not isinstance(tiles, list):
            tiles = [tiles]

        # Delete old tile textures; draw_image items may already be gone if
        # _rebuild_track_area deleted the drawlist, so errors are ignored.
        for item in self._tile_items:
            try:
                dpg.delete_item(item['tex'])
            except Exception:
                pass
            try:
                dpg.delete_item(item['img'])
            except Exception:
                pass
        self._tile_items = []

        new_height = tiles[0].shape[0] if tiles else self._height
        self._height = new_height

        t_start_col = 0
        for rgba in tiles:
            h, tc, _ = rgba.shape
            flat = rgba.ravel()
            tag  = dpg.add_raw_texture(
                tc, h, flat,
                format=dpg.mvFormat_Float_rgba,
                parent=self._tex_registry,
            )
            img  = dpg.draw_image(
                tag,
                pmin=(self._x_offset, 0), pmax=(self._x_offset, h),
                uv_min=(0, 0), uv_max=(0, 1),
                show=self._visible,
                parent=self._drawlist_tag,
            )
            self._tile_items.append({
                'tex':     tag,
                'img':     img,
                't_start': t_start_col,
                't_ncols': tc,
            })
            t_start_col += tc

        self._total_cols = t_start_col

    def set_zoom(self, factor: float) -> None:
        # factor == 0.0 is the sentinel for "full track" mode: entire spectrogram
        # fits in the slot width, playhead moves across it, no scrolling.
        if factor == 0.0:
            self._zoom = 0.0
        else:
            self._zoom = max(1.0, float(factor))

    def set_scroll(self, pos_norm: float, dur_frac: float = 1.0) -> None:
        """Update visible UV windows across all tiles for the current playhead.

        zoom == 0.0 (full-track mode): all tiles are shown across the full slot
        width simultaneously; pos_norm is only used externally for the playhead
        line position — this method just ensures the full texture is always visible.

        dur_frac: track_duration / longest_track_duration (1.0 for longest track).
        Scales avail and vis so all tracks use the same time-per-pixel standard,
        keeping scrolling speeds in sync regardless of individual track durations.
        Shorter tracks show dark background past their end (no texture there).
        """
        if not self._visible or not self._tile_items:
            return

        dur_frac = max(1e-6, min(1.0, dur_frac))

        n   = self._total_cols
        w   = self._wave_w
        xo  = self._x_offset
        h   = self._height

        if self._zoom == 0.0:
            # Full-track mode: map texture across its proportional share of wave_w.
            # A track shorter than the longest gets a narrower pixel band; the rest
            # is dark background — matching the same time-per-pixel standard.
            for item in self._tile_items:
                img = item['img']
                if not dpg.does_item_exist(img):
                    continue
                t_start = item['t_start']
                t_ncols = item['t_ncols']
                # Pixel extents scaled to the track's proportional width
                px_left  = xo + round(t_start / n * dur_frac * w)
                px_right = xo + round((t_start + t_ncols) / n * dur_frac * w)
                dpg.configure_item(img,
                                   pmin=(px_left, 0), pmax=(px_right, h),
                                   uv_min=(0, 0), uv_max=(1, 1))
            return

        # Scroll mode: "now" is pinned at the right edge.
        # Keep vis_f and avail_f as floats — rounding them to integers causes each
        # track to cross its rounding threshold at a different frame, producing
        # visible jitter between strips. Only round at the final pixel position.
        vis_f      = max(1.0, w / self._zoom / dur_frac)
        avail_f    = min(pos_norm / dur_frac * n, float(n))
        avail_f    = max(0.0, avail_f)
        left_col_f = avail_f - vis_f   # negative during early playback

        for item in self._tile_items:
            img = item['img']
            if not dpg.does_item_exist(img):
                continue

            t_start = item['t_start']
            t_ncols = item['t_ncols']
            t_end   = t_start + t_ncols

            # Content columns (float) from this tile that fall in the visible window
            c_left_f  = max(max(0.0, left_col_f), float(t_start))
            c_right_f = min(avail_f, float(t_end))

            if c_left_f >= c_right_f or avail_f <= 0.0:
                # Tile not in view — zero width
                dpg.configure_item(img,
                                   pmin=(xo, 0), pmax=(xo, h),
                                   uv_min=(0, 0), uv_max=(0, 1))
                continue

            # Screen pixel extents — round only here so all strips step together
            px_left  = xo + round((c_left_f  - left_col_f) / vis_f * w)
            px_right = xo + round((c_right_f - left_col_f) / vis_f * w)
            px_left  = max(xo, px_left)
            px_right = min(xo + w, px_right)

            # UV coordinates within this tile
            uv_left  = (c_left_f  - t_start) / t_ncols
            uv_right = (c_right_f - t_start) / t_ncols

            dpg.configure_item(img,
                               pmin=(px_left, 0), pmax=(px_right, h),
                               uv_min=(uv_left, 0), uv_max=(uv_right, 1))

    def set_visible(self, visible: bool) -> None:
        if visible == self._visible:
            return
        self._visible = visible
        xo = self._x_offset
        h  = self._height
        for item in self._tile_items:
            img = item['img']
            if not dpg.does_item_exist(img):
                continue
            if visible:
                dpg.configure_item(img, show=True)
            else:
                dpg.configure_item(img, show=False,
                                   pmin=(xo, 0), pmax=(xo, h),
                                   uv_min=(0, 0), uv_max=(0, 1))

    def set_active(self, is_active: bool) -> None:
        pass

    def get_buffer(self) -> np.ndarray | None:
        return None

    def seed_from_buffer(self, src) -> None:
        pass

    def delete(self) -> None:
        """Delete all tile textures.  draw_image items are drawlist children
        and are cleaned up automatically when the drawlist is deleted."""
        for item in self._tile_items:
            try:
                dpg.delete_item(item['tex'])
            except Exception:
                pass
        self._tile_items = []
