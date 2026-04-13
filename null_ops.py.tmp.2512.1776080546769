"""
null_ops.py — high-precision null test engine for MixABTest.

Pipeline (run_null_test):
  1. Probe both files for sample rate / channel count.
  2. Decode both to float64 PCM via ffmpeg + SoXR resampler.
  3. Remove DC offset per channel.
  4. Coarse alignment: FFT cross-correlation → integer sample lag.
  5. Polarity correction: flip candidate if correlation peak is negative.
  6. Level matching: least-squares optimal gain scalar.
  7. Fine alignment: phase-slope estimation → fractional-sample delay
     applied via 257-tap Blackman-windowed sinc FIR.
  8. Compute null signal (difference).
  9. Collect metrics and build interpretation string.
 10. Downcast null to float32 for playback; discard float64 arrays.

Requires: numpy, scipy
"""

import subprocess
import sys

import numpy as np
# scipy.signal is imported lazily on first call to run_null_test() — it adds
# ~10 s to startup on some systems and is only needed when a null test runs.
_fftconvolve = None   # populated by _ensure_scipy()

try:
    from . import audio_ops
except ImportError:
    import audio_ops  # type: ignore

# ── Internal helpers ──────────────────────────────────────────────────────────

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# Maximum sample rate we'll decode at — avoids absurd RAM use on DSD/384k files.
_SR_CAP = 96_000

# Cross-correlation segment length (seconds) used for alignment search.
_ALIGN_SEG_SECS = 10.0

# Fractional-delay FIR length (odd, Blackman window).
_FIR_TAPS = 257


def _ensure_scipy():
    """Import scipy.signal on first use and cache fftconvolve."""
    global _fftconvolve
    if _fftconvolve is None:
        from scipy.signal import fftconvolve as _fc
        _fftconvolve = _fc


# ── Step 2: decode ────────────────────────────────────────────────────────────

def _decode_float64(path: str, target_sr: int, channels: int) -> np.ndarray:
    """Decode *path* to float64 PCM, shape (n_samples, channels).

    Attempts SoXR resampler first (highest quality); falls back to the ffmpeg
    default resampler if SoXR is unavailable in this build.
    Raises RuntimeError on decode failure.
    """
    ffmpeg = audio_ops._tool("ffmpeg")
    base_cmd = [
        ffmpeg, "-y",
        "-i", path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(target_sr),
    ]

    # Try SoXR first; if it fails (no output), retry without the -af flag.
    for af in (["aresample=resampler=soxr"], None):
        cmd = base_cmd + (["-af", af[0]] if af else []) + ["-f", "f64le", "-"]
        result = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
        raw = result.stdout
        n = len(raw) // 8
        if n > 0:
            data = np.frombuffer(raw[: n * 8], dtype="<f8").copy()
            return data.reshape(-1, channels)

    # Both attempts failed — report the last stderr
    raise RuntimeError(
        f"ffmpeg produced no audio output for: {path}\n"
        f"{result.stderr.decode(errors='replace')}"
    )


# ── Step 3: DC removal ────────────────────────────────────────────────────────

def _remove_dc(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0)


# ── Step 4+5: coarse alignment + polarity ────────────────────────────────────

def _coarse_align(a: np.ndarray, b: np.ndarray, sr: int):
    """Return (lag_samples: int, polarity_inverted: bool).

    Positive lag → b starts later than a (b's front must be trimmed or a padded).
    Uses a centre segment of up to _ALIGN_SEG_SECS from both signals.
    Amplitude-normalises before correlating so level differences don't bias the peak.
    """
    a_mono = a.mean(axis=1) if a.ndim > 1 else a.copy()
    b_mono = b.mean(axis=1) if b.ndim > 1 else b.copy()

    n = min(len(a_mono), len(b_mono))
    seg_len = min(n, int(sr * _ALIGN_SEG_SECS))
    start = (n - seg_len) // 2

    a_seg = a_mono[start: start + seg_len].copy()
    b_seg = b_mono[start: start + seg_len].copy()

    # Amplitude-normalise
    a_seg /= (np.abs(a_seg).max() + 1e-12)
    b_seg /= (np.abs(b_seg).max() + 1e-12)

    corr = _fftconvolve(a_seg, b_seg[::-1], mode="full")
    peak_idx = int(np.argmax(np.abs(corr)))
    lag = peak_idx - (len(b_seg) - 1)

    polarity_inverted = bool(corr[peak_idx] < 0)
    return lag, polarity_inverted


def _apply_lag(a: np.ndarray, b: np.ndarray, lag: int):
    """Trim leading samples so a and b are aligned; return equal-length pair."""
    if lag > 0:
        b = b[lag:]
    elif lag < 0:
        a = a[-lag:]
    n = min(len(a), len(b))
    return a[:n].copy(), b[:n].copy()


# ── Step 6: level matching ────────────────────────────────────────────────────

def _optimal_gain(a: np.ndarray, b: np.ndarray) -> float:
    """Least-squares scalar g minimising RMS(a − g·b).

    Derivation: d/dg ‖a − g·b‖² = 0  →  g = <a,b> / <b,b>
    """
    a_f = a.flatten()
    b_f = b.flatten()
    denom = float(np.dot(b_f, b_f))
    if denom < 1e-30:
        return 1.0
    return float(np.dot(a_f, b_f) / denom)


# ── Step 7: fractional-sample alignment ──────────────────────────────────────

def _phase_slope_delay(a_mono: np.ndarray, b_mono: np.ndarray) -> float:
    """Estimate sub-sample delay of b relative to a via weighted phase-slope regression.

    Returns delay in samples (fractional). Only meaningful when the integer part
    has already been removed (i.e. after _apply_lag).
    """
    N = len(a_mono)
    A = np.fft.rfft(a_mono)
    B = np.fft.rfft(b_mono)

    cpsd    = A * np.conj(B)
    phase   = np.angle(cpsd)
    freqs   = np.fft.rfftfreq(N)   # normalised 0..0.5
    weights = np.abs(cpsd)

    # Use mid-band frequencies where SNR is reliable
    mask = (freqs > 0.01) & (freqs < 0.49)
    if mask.sum() < 4:
        return 0.0

    ph_m  = np.unwrap(phase[mask])
    fr_m  = freqs[mask]
    wt_m  = weights[mask]

    # Weighted least-squares: phase ≈ slope * freqs  (no intercept — DC is excluded)
    denom = float(np.sum(wt_m * fr_m ** 2))
    if denom < 1e-30:
        return 0.0
    slope = float(np.sum(wt_m * fr_m * ph_m) / denom)
    return -slope / (2.0 * np.pi)   # delay in samples


def _fractional_delay_filter(delay: float) -> np.ndarray:
    """Design a Blackman-windowed sinc FIR for a fractional delay of *delay* samples."""
    half = (_FIR_TAPS - 1) / 2.0
    t = np.arange(_FIR_TAPS) - half + delay
    h = np.sinc(t) * np.blackman(_FIR_TAPS)
    h /= h.sum()
    return h.astype(np.float64)


def _apply_fractional_delay(signal: np.ndarray, delay: float) -> np.ndarray:
    """Apply sub-sample fractional delay to signal (n_samples, channels) or (n_samples,)."""
    if abs(delay) < 0.01:
        return signal
    h = _fractional_delay_filter(delay)
    if signal.ndim == 1:
        return _fftconvolve(signal, h, mode="same")
    return np.column_stack([
        _fftconvolve(signal[:, ch], h, mode="same")
        for ch in range(signal.shape[1])
    ])


# ── Step 9: metrics ───────────────────────────────────────────────────────────

def _interpret(null_floor_db: float) -> str:
    if null_floor_db < -120:
        return "Bit-perfect identical"
    if null_floor_db < -100:
        return "Dither / word-length difference only"
    if null_floor_db < -80:
        return "Excellent — minor resampling or dither artifacts"
    if null_floor_db < -60:
        return "Good — possible high-quality lossy (320 kbps MP3 / AAC)"
    if null_floor_db < -40:
        return "Audible difference — lossy codec or subtle processing"
    if null_floor_db < -20:
        return "Significant processing difference (EQ, compression, master)"
    return "Major difference — level mismatch or different content"


def _null_metrics(null: np.ndarray, ref: np.ndarray, sr: int,
                  lag: int, gain_db: float, frac_delay: float) -> dict:
    eps = 1e-10

    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    def db(linear):
        return float(20.0 * np.log10(max(abs(linear), eps)))

    null_rms  = rms(null)
    null_peak = float(np.abs(null).max())
    ref_rms   = rms(ref)
    ref_peak  = float(np.abs(ref).max())

    null_rms_db  = db(null_rms)
    null_peak_db = db(null_peak)
    ref_rms_db   = db(ref_rms)
    ref_peak_db  = db(ref_peak)
    null_floor_db = null_rms_db - ref_rms_db   # relative; negative = good
    null_crest_db = null_peak_db - null_rms_db

    # Pearson correlation between ref and candidate (= ref − null)
    r_flat = ref.flatten()
    c_flat = (ref - null).flatten()
    cov    = float(np.dot(r_flat - r_flat.mean(), c_flat - c_flat.mean()))
    std    = float(np.std(r_flat) * np.std(c_flat) * len(r_flat))
    corr   = cov / std if std > 1e-30 else 1.0

    return {
        "null_floor_db":        round(null_floor_db, 2),
        "null_rms_db":          round(null_rms_db, 2),
        "null_peak_db":         round(null_peak_db, 2),
        "null_crest_db":        round(null_crest_db, 2),
        "ref_rms_db":           round(ref_rms_db, 2),
        "ref_peak_db":          round(ref_peak_db, 2),
        "signal_correlation":   round(corr, 6),
        "lag_samples":          lag,
        "lag_ms":               round(lag / sr * 1000.0, 3),
        "gain_applied_db":      round(gain_db, 3),
        "frac_delay_samples":   round(frac_delay, 4),
        "sample_rate":          sr,
        "interpretation":       _interpret(null_floor_db),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_null_test(path_ref: str, path_cand: str,
                  progress_cb=None) -> tuple:
    """Compute a null test between *path_ref* and *path_cand*.

    Returns (metrics: dict, null_float32: np.ndarray shape (n, channels)).

    *progress_cb*, if provided, is called with a status string at key steps
    so a UI can display progress.  It must be safe to call from a non-main thread.

    Raises RuntimeError with a human-readable message on failure.
    """
    _ensure_scipy()

    def _prog(msg):
        if progress_cb:
            progress_cb(msg)

    # ── 1. Probe ──────────────────────────────────────────────────────────────
    _prog("Probing files…")
    info_ref  = audio_ops.probe_info(path_ref)
    info_cand = audio_ops.probe_info(path_cand)

    sr       = min(_SR_CAP, max(info_ref["sample_rate"], info_cand["sample_rate"]))
    channels = max(info_ref["channels"], info_cand["channels"])
    # Clamp channels to stereo — we don't need surround for a null test
    channels = min(channels, 2)

    # ── 2. Decode ─────────────────────────────────────────────────────────────
    _prog("Decoding reference…")
    ref = _decode_float64(path_ref, sr, channels)

    _prog("Decoding candidate…")
    cand = _decode_float64(path_cand, sr, channels)

    # ── 3. DC removal ─────────────────────────────────────────────────────────
    _prog("Removing DC offset…")
    ref  = _remove_dc(ref)
    cand = _remove_dc(cand)

    # ── 4+5. Coarse alignment + polarity ──────────────────────────────────────
    _prog("Aligning signals…")
    lag, polarity_inv = _coarse_align(ref, cand, sr)
    ref_a, cand_a = _apply_lag(ref, cand, lag)

    if polarity_inv:
        cand_a = -cand_a

    # ── 6. Level matching ─────────────────────────────────────────────────────
    _prog("Matching levels…")
    g = _optimal_gain(ref_a, cand_a)
    gain_db = float(20.0 * np.log10(max(abs(g), 1e-10)))
    cand_a *= g

    # ── 7. Fractional alignment ───────────────────────────────────────────────
    _prog("Fine-aligning (sub-sample)…")
    ref_mono  = ref_a.mean(axis=1)  if ref_a.ndim  > 1 else ref_a.copy()
    cand_mono = cand_a.mean(axis=1) if cand_a.ndim > 1 else cand_a.copy()
    frac_delay = _phase_slope_delay(ref_mono, cand_mono)
    # The phase-slope estimator is only reliable when the integer alignment
    # has already been applied and the residual delay is truly sub-sample.
    # For signals with significant processing differences the weighted
    # regression can return delays of many samples; feeding those to the
    # sinc FIR collapses h.sum() to near-zero, causing h /= h.sum() to
    # produce coefficients of ~1e7 and astronomical null amplitudes.
    # Cap to ±1 sample — anything larger indicates an unreliable estimate.
    if 0.01 <= abs(frac_delay) < 1.0:
        cand_a = _apply_fractional_delay(cand_a, frac_delay)
        # fftconvolve with mode='same' can shift length by 1 — re-trim
        n = min(len(ref_a), len(cand_a))
        ref_a  = ref_a[:n]
        cand_a = cand_a[:n]
    else:
        frac_delay = 0.0   # unreliable — record as zero in metrics

    # ── 8. Null ───────────────────────────────────────────────────────────────
    _prog("Computing null…")
    null_f64 = ref_a - cand_a

    # ── 9. Metrics ────────────────────────────────────────────────────────────
    _prog("Computing metrics…")
    metrics = _null_metrics(null_f64, ref_a, sr, lag, gain_db, frac_delay)

    # ── 10. Downcast to float32 for playback; free float64 arrays ─────────────
    _prog("Done.")
    null_f32 = null_f64.astype(np.float32)
    del ref, cand, ref_a, cand_a, null_f64

    return metrics, null_f32


def prepare_for_engine(null_f32: np.ndarray, src_sr: int,
                        target_sr: int = 48000) -> np.ndarray:
    """Resample *null_f32* to *target_sr* and ensure stereo, ready for AudioEngine.set_buffer().

    AudioEngine always plays at 48000 Hz stereo.  The null is computed at the
    file's native sample rate which may differ — passing a mismatched array
    causes the engine position to overshoot the buffer end immediately, producing
    silence.
    """
    _ensure_scipy()
    from scipy.signal import resample_poly
    from math import gcd

    arr = np.asarray(null_f32, dtype=np.float32)

    # ── Resample to engine sample rate ────────────────────────────────────────
    if src_sr != target_sr:
        g    = gcd(src_sr, target_sr)
        up   = target_sr // g
        down = src_sr    // g
        if arr.ndim == 1:
            arr = resample_poly(arr, up, down).astype(np.float32)
        else:
            arr = np.column_stack([
                resample_poly(arr[:, ch], up, down).astype(np.float32)
                for ch in range(arr.shape[1])
            ])

    # ── Ensure stereo (engine expects shape (n, 2)) ───────────────────────────
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    elif arr.shape[1] == 1:
        arr = np.column_stack([arr[:, 0], arr[:, 0]])

    return arr


def downsample_for_waveform(null_f32: np.ndarray, sr: int,
                             target_sr: int = 4000) -> tuple:
    """Downsample *null_f32* to *target_sr* Hz mono int16 for waveform display.

    Returns a tuple of int16 samples, matching the format of audio_ops.extract_waveform().
    """
    mono = null_f32.mean(axis=1) if null_f32.ndim > 1 else null_f32.copy()
    step = max(1, sr // target_sr)
    downsampled = mono[::step]
    # Scale to int16 range
    peak = float(np.abs(downsampled).max())
    if peak < 1e-9:
        return tuple(np.zeros(len(downsampled), dtype=np.int16))
    scaled = (downsampled / peak * 32767.0).clip(-32768, 32767).astype(np.int16)
    return tuple(int(s) for s in scaled)
