"""
player.py — sounddevice-based audio engine for MixABTest.

A single OutputStream runs continuously.  Switching tracks is just
swapping which decoded buffer the callback reads — no stream restart,
no gap, no click.
"""

import threading
from typing import Dict, List, Optional, Any, Callable

try:
    import numpy as np
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

SAMPLERATE      = 48000
CHANNELS        = 2
BLOCKSIZE       = 512    # ~10.7 ms at 48 kHz — ignored for ASIO (driver sets its own)
XFADE_FRAMES    = 512    # ~10.7 ms — fade long enough to be inaudible as a click
_WASAPI_NAME    = "Windows WASAPI"
_ASIO_NAME      = "ASIO"
_WDM_KS_NAME    = "Windows WDM-KS"   # kept for settings list; NOT used as a default

# Preferred API fallback order for first-run default (safe APIs only)
_PREFERRED_APIS = [_ASIO_NAME, _WASAPI_NAME]

# Pre-computed linear crossfade ramps — computed once at import, never re-allocated.
# Shape (XFADE_FRAMES, 1) so they broadcast directly against (frames, CHANNELS) chunks.
# fade_in + fade_out = 1.0 at every sample: no volume increase is possible.
if _HAS_SD:
    _t        = np.linspace(0.0, 1.0, XFADE_FRAMES, endpoint=False, dtype=np.float32)
    _FADE_IN  = _t.reshape(-1, 1).copy()
    _FADE_OUT = (1.0 - _t).reshape(-1, 1).copy()
    del _t

    # FFT spectrum — pre-computed once at import time; reconfigured via configure_fft().
    _FFT_SIZE   = 1024   # ~21 ms window at 48 kHz
    _N_BANDS    = 64     # log-spaced bands shown per channel
    _HANN_WIN   = np.hanning(_FFT_SIZE).astype(np.float32)
    _fft_freqs  = np.fft.rfftfreq(_FFT_SIZE, d=1.0 / SAMPLERATE)
    _band_edges = np.logspace(np.log10(20.0), np.log10(20000.0), _N_BANDS + 1)
    _BAND_BINS  = []
    for _bi in range(_N_BANDS):
        _lo = int(np.searchsorted(_fft_freqs, _band_edges[_bi]))
        _hi = int(np.searchsorted(_fft_freqs, _band_edges[_bi + 1]))
        _BAND_BINS.append((_lo, max(_hi, _lo + 1)))
    # Vectorised reduceat helpers
    _BAND_START   = np.array([lo for lo, hi in _BAND_BINS], dtype=np.intp)
    _BAND_COUNTS  = np.array([max(1, hi - lo) for lo, hi in _BAND_BINS], dtype=np.float32)
    _BAND_LAST_HI = _BAND_BINS[-1][1]   # clip mag here so final band stops at 20 kHz
    del _fft_freqs, _band_edges, _bi, _lo, _hi

    # ── Hi-res spectrogram — single 16384-point FFT, 512 log-spaced bands ────
    # 16384 pts at 48 kHz → 2.93 Hz/bin, 341 ms window.
    # 2× better low-end frequency resolution vs 8192, without the temporal
    # smearing of larger windows (65536 = 1.36 s → muddy bass transients).
    # Only used by sample_spectrum_hires(); never changed at runtime.
    _SPECTRO_FFT_SIZE  = 16384
    _SPECTRO_N_BANDS   = 512
    _SPECTRO_HANN      = np.hanning(_SPECTRO_FFT_SIZE).astype(np.float32)

    # Derivative window for frequency reassignment: h_D[n] = dh/dn
    # Centred finite difference; endpoints use one-sided (h outside window = 0).
    # Error < 2% for windows this size (Flandrin et al.).
    _spectro_h_tmp          = _SPECTRO_HANN  # alias
    _SPECTRO_H_D            = np.empty(_SPECTRO_FFT_SIZE, dtype=np.float32)
    _SPECTRO_H_D[1:-1]      = (_spectro_h_tmp[2:] - _spectro_h_tmp[:-2]) * 0.5
    _SPECTRO_H_D[0]         = _spectro_h_tmp[1] * 0.5   # h[-1] = 0 outside window
    _SPECTRO_H_D[-1]        = -_spectro_h_tmp[-2] * 0.5  # h[N] = 0 outside window
    del _spectro_h_tmp

    _spectro_freqs     = np.fft.rfftfreq(_SPECTRO_FFT_SIZE, d=1.0 / SAMPLERATE)
    _spectro_edges     = np.logspace(np.log10(20.0), np.log10(20000.0), _SPECTRO_N_BANDS + 1)
    _spectro_bins      = []
    for _bi in range(_SPECTRO_N_BANDS):
        _lo = int(np.searchsorted(_spectro_freqs, _spectro_edges[_bi]))
        _hi = int(np.searchsorted(_spectro_freqs, _spectro_edges[_bi + 1]))
        _spectro_bins.append((_lo, max(_hi, _lo + 1)))
    _SPECTRO_BAND_START   = np.array([lo for lo, hi in _spectro_bins], dtype=np.intp)
    _SPECTRO_BAND_COUNTS  = np.array([max(1, hi - lo) for lo, hi in _spectro_bins], dtype=np.float32)
    _SPECTRO_BAND_LAST_HI = _spectro_bins[-1][1]
    del _spectro_freqs, _spectro_edges, _spectro_bins, _bi, _lo, _hi
else:
    _FADE_IN = _FADE_OUT = None
    _FFT_SIZE = _N_BANDS = _HANN_WIN = _BAND_BINS = None
    _BAND_START = _BAND_COUNTS = _BAND_LAST_HI = None
    _SPECTRO_FFT_SIZE = _SPECTRO_N_BANDS = _SPECTRO_HANN = _SPECTRO_H_D = None
    _SPECTRO_BAND_START = _SPECTRO_BAND_COUNTS = _SPECTRO_BAND_LAST_HI = None


class AudioEngine:
    """Click-free audio engine backed by a single sounddevice OutputStream."""

    def __init__(self, device: Optional[int] = None) -> None:
        self._device: Optional[int] = device   # sd device index / name, or None for default
        self._buffers: Dict[int, Any] = {}       # track_id -> np.ndarray (frames, CHANNELS) float32
        self._gains: Dict[int, float] = {}       # track_id -> float32 linear gain (default 1.0)
        self._volume: float = 1.0     # master output volume 0.0–1.0
        self._muted: bool = False   # master mute
        self._active: Optional[int] = None     # track_id of the currently audible buffer
        self._prev_active: Optional[int] = None     # track_id being faded out during a crossfade
        self._xfade_remaining: int = 0        # frames left in the current crossfade
        self._pos: int = 0        # playback position in frames
        self._playing: bool = False
        self._on_end: Optional[Callable[[], None]] = None     # callable fired when the active buffer runs out
        self._stream: Any = None
        # Pre-allocated scratch buffer used only during crossfades — keeps the
        # real-time callback entirely free of heap allocations.
        if _HAS_SD:
            self._xf_scratch = np.zeros((XFADE_FRAMES, CHANNELS), dtype=np.float32)

        if _HAS_SD:
            self._open_stream()

    # ── stream ────────────────────────────────────────────────────────────────

    def _open_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # ASIO drivers control their own buffer size; passing 0 lets PortAudio
        # use the device-preferred size.  All other APIs use our fixed BLOCKSIZE.
        bs = 0 if _is_asio_device(self._device) else BLOCKSIZE
        self._stream = sd.OutputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype="float32",
            device=self._device,
            blocksize=bs,
            callback=self._callback,
        )
        self._stream.start()

    def reopen(self, device=None):
        """Switch output device; preserves position and playing state."""
        was_playing   = self._playing
        self._playing = False
        self._device  = device
        self._open_stream()
        self._playing = was_playing

    # ── audio callback (runs in a real-time thread) ───────────────────────────

    def _callback(self, outdata, frames, time_info, status):
        # All attribute reads/writes here are GIL-atomic in CPython — safe without a lock.
        if not self._playing or self._active is None:
            outdata[:] = 0
            return

        buf = self._buffers.get(self._active)
        if buf is None:
            outdata[:] = 0
            return

        pos   = self._pos
        end   = pos + frames
        avail = len(buf) - pos

        if avail <= 0:
            outdata[:] = 0
            self._playing = False
            if self._on_end:
                self._on_end()
            return

        # ── Gather new-track samples ──────────────────────────────────────────
        if avail >= frames:
            new_chunk  = buf[pos:end]   # numpy view — zero copy
            self._pos  = end
            fire_end   = False
        else:
            new_chunk = np.zeros((frames, CHANNELS), dtype=np.float32)
            new_chunk[:avail] = buf[pos:]
            self._pos     = len(buf)
            self._playing = False
            fire_end      = True

        # ── Apply per-track gain (audio normalisation) — GIL-atomic dict lookup ─
        gain_a = self._gains.get(self._active, 1.0)
        gain_p = self._gains.get(self._prev_active, 1.0)

        # ── Crossfade: both tracks mixed simultaneously, gains sum to 1.0 always ─
        # Ramps are pre-computed at import time; scratch buffer pre-allocated in
        # __init__ — this entire branch is free of heap allocations.
        xr = self._xfade_remaining
        if xr > 0:
            prev_buf = self._buffers.get(self._prev_active)
            if prev_buf is not None and pos < len(prev_buf):
                xf  = min(frames, xr)
                fi  = XFADE_FRAMES - xr          # start index into pre-computed ramps

                # Copy fading-out track into scratch (no alloc)
                p_avail = len(prev_buf) - pos
                p_copy  = min(xf, p_avail)
                self._xf_scratch[:p_copy] = prev_buf[pos : pos + p_copy]
                if p_copy < xf:
                    self._xf_scratch[p_copy:xf] = 0

                # out[:xf] = fade_in * new*gain_a + fade_out * prev*gain_p
                np.multiply(_FADE_IN[fi:fi+xf],  new_chunk[:xf],           out=outdata[:xf])
                outdata[:xf]               *= gain_a                        # in-place scalar
                np.multiply(_FADE_OUT[fi:fi+xf], self._xf_scratch[:xf],    out=self._xf_scratch[:xf])
                self._xf_scratch[:xf]      *= gain_p                        # in-place scalar
                np.add(outdata[:xf], self._xf_scratch[:xf],                 out=outdata[:xf])

                if xf < frames:
                    np.multiply(new_chunk[xf:], gain_a, out=outdata[xf:])
            else:
                np.multiply(new_chunk, gain_a, out=outdata)
            self._xfade_remaining = max(0, xr - frames)
        else:
            np.multiply(new_chunk, gain_a, out=outdata)

        # ── Master volume / mute — applied after all mixing ──────────────────
        if self._muted:
            outdata[:] = 0
        elif self._volume != 1.0:
            outdata *= self._volume

        if fire_end and self._on_end:
            self._on_end()

    # ── public API ────────────────────────────────────────────────────────────

    def set_buffer(self, track_id, data):
        """Store a decoded audio buffer (np.ndarray float32, shape=(frames, CHANNELS))."""
        self._buffers[track_id] = data

    def set_gain(self, track_id, gain):
        """Set linear playback gain for track_id (1.0 = unity, 2.0 = +6 dB, etc.)."""
        self._gains[track_id] = float(gain)

    def set_volume(self, volume):
        """Set master output volume 0.0–1.0."""
        self._volume = max(0.0, min(1.0, float(volume)))

    def set_muted(self, muted):
        """Mute or unmute master output."""
        self._muted = bool(muted)

    def remove_buffer(self, track_id):
        self._buffers.pop(track_id, None)
        self._gains.pop(track_id, None)

    def set_active(self, track_id):
        """Switch to track_id, triggering a short linear crossfade to eliminate clicks.

        Write order matters for GIL-safety: _prev_active and _xfade_remaining are
        committed before _active so the callback never sees an inconsistent state.
        """
        if track_id == self._active:
            return
        self._prev_active     = self._active   # 1. save outgoing track
        self._xfade_remaining = XFADE_FRAMES   # 2. arm the fade counter
        self._active          = track_id       # 3. swap — callback picks this up last

    def set_on_end(self, cb):
        self._on_end = cb

    def sample_spectrum(self, track_ids=None):
        """Return {track_id: np.ndarray(_N_BANDS,)} of dBFS values for specified buffers.

        If track_ids is None, compute for all loaded buffers.
        Values are in dBFS (0 = full scale, negative = quieter).
        Reads a 2048-sample Hann-windowed FFT centred on the current position.
        """
        if not _HAS_SD:
            return {}
        pos    = self._pos
        result = {}
        buffers = self._buffers if track_ids is None else {tid: self._buffers[tid] for tid in track_ids if tid in self._buffers}
        for tid, buf in list(buffers.items()):
            start = max(0, pos - _FFT_SIZE // 2)
            end   = min(len(buf), start + _FFT_SIZE)
            chunk = buf[start:end]
            n     = len(chunk)
            if n < 64:
                result[tid] = np.full(_N_BANDS, -90.0, dtype=np.float32)
                continue
            mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.copy()
            if n < _FFT_SIZE:
                padded = np.zeros(_FFT_SIZE, dtype=np.float32)
                padded[:n] = mono
                mono = padded
            mag    = np.abs(np.fft.rfft(mono * _HANN_WIN)) * (2.0 / _FFT_SIZE)
            # Vectorised band averaging — one reduceat instead of a Python loop
            sums   = np.add.reduceat(mag[:_BAND_LAST_HI], _BAND_START)
            bands  = (sums[:_N_BANDS] / _BAND_COUNTS).astype(np.float32)
            result[tid] = 20.0 * np.log10(np.maximum(bands, 1e-6))
        return result

    def sample_spectrum_hires(self, track_ids=None):
        """Return {track_id: np.ndarray(512,)} using frequency-reassigned spectrogram.

        16384-point FFT, 512 log-spaced bands.  Each bin's energy is scattered to
        its instantaneous frequency (centre of gravity) rather than its geometric
        bin centre, sharpening harmonic lines and reducing smearing.

        Algorithm (Auger-Flandrin frequency reassignment):
          X   = rfft(x * h)     — standard STFT
          X_D = rfft(x * h_D)   — STFT with derivative window
          k_hat = k + Im(X_D * conj(X)) / |X|² * N/(2π)   — reassigned bin
          Scatter |X[k]| to reassigned_amp[k_hat], then band-average.
        """
        if not _HAS_SD:
            return {}
        pos     = self._pos
        result  = {}
        buffers = self._buffers if track_ids is None else {tid: self._buffers[tid] for tid in track_ids if tid in self._buffers}
        n_bins  = _SPECTRO_FFT_SIZE // 2 + 1
        scale   = 2.0 / _SPECTRO_FFT_SIZE
        inv_2pi = _SPECTRO_FFT_SIZE / (2.0 * np.pi)
        k_src   = np.arange(n_bins, dtype=np.int32)

        for tid, buf in list(buffers.items()):
            # Skip frames where the window extends before the audio start — those
            # require zero-padding the left half, which smears the onset blob.
            if pos < _SPECTRO_FFT_SIZE // 2:
                continue
            start = pos - _SPECTRO_FFT_SIZE // 2
            end   = min(len(buf), start + _SPECTRO_FFT_SIZE)
            chunk = buf[start:end]
            n     = len(chunk)
            if n < 64:
                result[tid] = np.full(_SPECTRO_N_BANDS, -90.0, dtype=np.float32)
                continue
            mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.copy()
            if n < _SPECTRO_FFT_SIZE:
                padded = np.zeros(_SPECTRO_FFT_SIZE, dtype=np.float32)
                padded[:n] = mono
                mono = padded

            # Two FFTs — standard and derivative-windowed
            X   = np.fft.rfft(mono * _SPECTRO_HANN)
            X_D = np.fft.rfft(mono * _SPECTRO_H_D)

            # |X|² — used for reassignment weight and energy threshold
            mag_sq = X.real * X.real + X.imag * X.imag

            # Im(X_D * conj(X)) = X_D.imag*X.real - X_D.real*X.imag
            num_im = X_D.imag * X.real - X_D.real * X.imag

            # Frequency reassignment with fractional scatter.
            # Using np.round() causes nearby bins converging on the same true
            # frequency to round to different integers → multiple parallel lines.
            # Fractional scatter splits each bin's energy between floor/ceil,
            # so all bins targeting the same frequency naturally merge.
            safe       = mag_sq > 1e-10
            denom      = np.where(safe, mag_sq, 1.0)
            # Sign is MINUS: sidelobes are pushed TOWARD the true peak.
            # PLUS was wrong — it scattered sidelobes away, creating ghost lines.
            k_hat_f    = k_src - num_im / denom * inv_2pi   # fractional bin, no round
            k_lo       = np.clip(np.floor(k_hat_f).astype(np.int32), 0, n_bins - 1)
            k_hi       = np.clip(k_lo + 1,                            0, n_bins - 1)
            frac       = np.clip((k_hat_f - np.floor(k_hat_f)).astype(np.float32), 0.0, 1.0)

            amp = np.abs(X) * scale
            amp = np.where(safe, amp, 0.0)
            reassigned = np.zeros(n_bins, dtype=np.float32)
            np.add.at(reassigned, k_lo, amp * (1.0 - frac))
            np.add.at(reassigned, k_hi, amp * frac)

            # Band-average and convert to dBFS
            sums  = np.add.reduceat(reassigned[:_SPECTRO_BAND_LAST_HI], _SPECTRO_BAND_START)
            bands = (sums[:_SPECTRO_N_BANDS] / _SPECTRO_BAND_COUNTS).astype(np.float32)
            result[tid] = 20.0 * np.log10(np.maximum(bands, 1e-6))
        return result

    def sample_spectrum_fullscreen(self, track_ids=None):
        """Return {tid: np.ndarray(n_bins,)} — raw reassigned linear magnitudes.

        Returns the full reassigned amplitude spectrum without band-averaging,
        so SpectrogramStrip can map each pixel row to its exact frequency bin
        via log-frequency interpolation.  Array length = _SPECTRO_FFT_SIZE//2 + 1.
        Values are linear amplitude (not dBFS) so the strip can interpolate cleanly.
        """
        if not _HAS_SD:
            return {}
        pos     = self._pos
        result  = {}
        buffers = self._buffers if track_ids is None else {tid: self._buffers[tid] for tid in track_ids if tid in self._buffers}
        n_bins  = _SPECTRO_FFT_SIZE // 2 + 1
        scale   = 2.0 / _SPECTRO_FFT_SIZE
        inv_2pi = _SPECTRO_FFT_SIZE / (2.0 * np.pi)
        k_src   = np.arange(n_bins, dtype=np.int32)

        for tid, buf in list(buffers.items()):
            if pos < _SPECTRO_FFT_SIZE // 2:
                continue
            start = pos - _SPECTRO_FFT_SIZE // 2
            end   = min(len(buf), start + _SPECTRO_FFT_SIZE)
            chunk = buf[start:end]
            n     = len(chunk)
            if n < 64:
                result[tid] = np.zeros(n_bins, dtype=np.float32)
                continue
            mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.copy()
            if n < _SPECTRO_FFT_SIZE:
                padded = np.zeros(_SPECTRO_FFT_SIZE, dtype=np.float32)
                padded[:n] = mono
                mono = padded

            X   = np.fft.rfft(mono * _SPECTRO_HANN)
            X_D = np.fft.rfft(mono * _SPECTRO_H_D)

            mag_sq  = X.real * X.real + X.imag * X.imag
            num_im  = X_D.imag * X.real - X_D.real * X.imag
            safe    = mag_sq > 1e-10
            denom   = np.where(safe, mag_sq, 1.0)
            k_hat_f = k_src - num_im / denom * inv_2pi   # MINUS: pulls sidelobes toward true freq
            k_lo    = np.clip(np.floor(k_hat_f).astype(np.int32), 0, n_bins - 1)
            k_hi    = np.clip(k_lo + 1,                            0, n_bins - 1)
            frac    = np.clip((k_hat_f - np.floor(k_hat_f)).astype(np.float32), 0.0, 1.0)

            amp = np.abs(X) * scale
            amp = np.where(safe, amp, 0.0)
            reassigned = np.zeros(n_bins, dtype=np.float32)
            np.add.at(reassigned, k_lo, amp * (1.0 - frac))
            np.add.at(reassigned, k_hi, amp * frac)
            result[tid] = reassigned   # raw linear magnitudes, NOT dBFS
        return result

    def sample_levels(self, n_frames=1024):
        """Return {track_id: (rms, peak)} for every loaded buffer at the current position.

        Reads a window of n_frames centred on _pos from each buffer — safe to call
        from the main thread; uses a snapshot of _buffers to avoid mid-iteration
        modification by the decode threads.
        """
        if not _HAS_SD:
            return {}
        pos     = self._pos
        result  = {}
        for tid, buf in list(self._buffers.items()):   # snapshot keys+values
            start = max(0, pos - n_frames // 2)
            end   = min(len(buf), start + n_frames)
            chunk = buf[start:end]
            if len(chunk) == 0:
                result[tid] = (0.0, 0.0)
            else:
                rms  = float(np.sqrt(np.mean(chunk ** 2)))
                peak = float(np.max(np.abs(chunk)))
                result[tid] = (rms, peak)
        return result

    def sample_correlation(self, n_frames=4096):
        """Return {track_id: float} L/R correlation in [-1, +1] at current position.

        +1 = perfect mono, 0 = uncorrelated, -1 = fully out of phase.
        Returns 0.0 for mono buffers or silence.
        """
        if not _HAS_SD:
            return {}
        pos    = self._pos
        result = {}
        for tid, buf in list(self._buffers.items()):
            if buf.ndim < 2 or buf.shape[1] < 2:
                result[tid] = 0.0
                continue
            start = max(0, pos - n_frames // 2)
            end   = min(len(buf), start + n_frames)
            chunk = buf[start:end]
            if len(chunk) < 16:
                result[tid] = 0.0
                continue
            L = chunk[:, 0].astype(np.float32)
            R = chunk[:, 1].astype(np.float32)
            num  = float(np.mean(L * R))
            denom = float(np.sqrt(np.mean(L * L) * np.mean(R * R)))
            result[tid] = num / denom if denom > 1e-12 else 0.0
        return result

    # ── transport ─────────────────────────────────────────────────────────────

    def play(self):
        self._playing = True

    def pause(self):
        self._playing = False

    @property
    def playing(self):
        return self._playing

    def seek(self, secs):
        frame = int(secs * SAMPLERATE)
        buf   = self._buffers.get(self._active)
        if buf is not None:
            frame = max(0, min(frame, len(buf) - 1))
        else:
            frame = max(0, frame)
        self._pos = frame

    @property
    def position(self):
        """Current playback position in seconds."""
        return self._pos / SAMPLERATE

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self._playing = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def _is_asio_device(device_idx):
    """Return True if device_idx belongs to an ASIO host API."""
    if not _HAS_SD or device_idx is None:
        return False
    try:
        apis = sd.query_hostapis()
        dev  = sd.query_devices(device_idx)
        return _ASIO_NAME.lower() in apis[dev["hostapi"]]["name"].lower()
    except Exception:
        return False


def default_device_for_api(api_name):
    """Return the index of the first output device for *api_name*, or None."""
    if not _HAS_SD:
        return None
    try:
        apis    = sd.query_hostapis()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_output_channels"] < 1:
                continue
            if apis[d["hostapi"]]["name"] == api_name:
                return i
    except Exception:
        pass
    return None


def best_default_device():
    """Return (device_index, api_name) for the safest available API.

    Priority: ASIO → WASAPI → system default (None).
    WDM-KS is intentionally excluded — it operates at kernel level and can
    cause BSODs, particularly when ASIO4ALL is also installed.
    """
    for api_name in _PREFERRED_APIS:
        idx = default_device_for_api(api_name)
        if idx is not None:
            return idx, api_name
    return None, "Default"


def available_output_devices():
    """Return list of (display_name, device_index, api_name) for all output devices."""
    if not _HAS_SD:
        return []
    result = []
    try:
        apis    = sd.query_hostapis()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_output_channels"] < 1:
                continue
            api_name = apis[d["hostapi"]]["name"] if d["hostapi"] < len(apis) else "?"
            result.append((d["name"], i, api_name))
    except Exception:
        pass
    return result


def available_apis():
    """Return list of (api_name, api_index) that have at least one output device."""
    if not _HAS_SD:
        return []
    result = []
    try:
        apis    = sd.query_hostapis()
        devices = sd.query_devices()
        seen    = set()
        for d in devices:
            if d["max_output_channels"] < 1:
                continue
            idx = d["hostapi"]
            if idx not in seen:
                seen.add(idx)
                result.append((apis[idx]["name"], idx))
    except Exception:
        pass
    return result


def configure_fft(fft_size, n_bands):
    """Recompute FFT lookup tables for new fft_size / n_bands.

    Thread-safe on CPython: each global is replaced atomically via the GIL.
    Call this from the main thread before the next _update_spectrum tick.
    """
    global _FFT_SIZE, _N_BANDS, _HANN_WIN, _BAND_BINS
    global _BAND_START, _BAND_COUNTS, _BAND_LAST_HI
    if not _HAS_SD:
        return
    fft_freqs  = np.fft.rfftfreq(fft_size, d=1.0 / SAMPLERATE)
    band_edges = np.logspace(np.log10(20.0), np.log10(20000.0), n_bands + 1)
    bins = []
    for bi in range(n_bands):
        lo = int(np.searchsorted(fft_freqs, band_edges[bi]))
        hi = int(np.searchsorted(fft_freqs, band_edges[bi + 1]))
        bins.append((lo, max(hi, lo + 1)))
    # Commit atomically — assign precomputed arrays first, then the sizes
    # so sample_spectrum never reads mismatched values.
    _HANN_WIN     = np.hanning(fft_size).astype(np.float32)
    _BAND_BINS    = bins
    _BAND_START   = np.array([lo for lo, hi in bins], dtype=np.intp)
    _BAND_COUNTS  = np.array([max(1, hi - lo) for lo, hi in bins], dtype=np.float32)
    _BAND_LAST_HI = bins[-1][1]
    _N_BANDS      = n_bands    # update sizes last
    _FFT_SIZE     = fft_size
