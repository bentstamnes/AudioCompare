"""
audio_ops.py — minimal ffmpeg helpers for MixABTest.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
from typing import List, Optional, Any

# In a windowed frozen app (PyInstaller console=False) there is no inherited
# console, so Windows creates a visible one for every subprocess unless we
# explicitly suppress it.  This flag is Windows-only and a no-op elsewhere.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _tool(name):
    """Locate an ffmpeg tool.

    Search order:
      1. sys._MEIPASS  — PyInstaller one-folder bundle (_internal/)
      2. next to sys.executable — dev run or legacy one-file bundle
      3. PATH
      4. bare name (let the OS raise a useful error)
    """
    exe = name + (".exe" if sys.platform == "win32" else "")
    if getattr(sys, "frozen", False):
        # One-folder: everything lives in _MEIPASS (_internal/), not next to the .exe
        for base in (getattr(sys, "_MEIPASS", None), os.path.dirname(sys.executable)):
            if base:
                candidate = os.path.join(base, exe)
                if os.path.exists(candidate):
                    return candidate
    else:
        # Dev: check directory of this source file
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), exe)
        if os.path.exists(candidate):
            return candidate
    found = shutil.which(name)
    return found if found else name


def extract_waveform(path: str) -> List[int]:
    """Decode audio to mono 4000 Hz s16le PCM.

    Returns a tuple of int16 samples, or an empty tuple on failure.
    """
    try:
        cmd = [
            _tool("ffmpeg"), "-y",
            "-i", path,
            "-vn", "-ac", "1", "-ar", "4000",
            "-f", "s16le", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
        raw = result.stdout
        n   = len(raw) // 2
        return struct.unpack(f"<{n}h", raw[: n * 2]) if n else ()
    except Exception:
        return ()


def decode_audio(path: str, samplerate: int = 48000, channels: int = 2) -> Optional[Any]:
    """Decode an audio file to a float32 numpy array of shape (frames, channels).

    Returns None on failure or if numpy is unavailable.
    """
    if not _HAS_NUMPY:
        return None
    try:
        cmd = [
            _tool("ffmpeg"), "-y",
            "-i", path,
            "-vn", "-ac", str(channels), "-ar", str(samplerate),
            "-f", "f32le", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
        raw = result.stdout
        n   = len(raw) // 4   # float32 = 4 bytes per sample
        if n == 0:
            return None
        data = np.frombuffer(raw[: n * 4], dtype=np.float32).copy()
        return data.reshape(-1, channels)
    except Exception:
        return None


def analyze_loudness(data, sr: int = 48000, n_points: int = 1300):
    """Calculate integrated LUFS and short-term loudness curve (ITU-R BS.1770-4).

    Args:
        data: (N, 2) float32 numpy array
        sr:   sample rate (default 48000)
        n_points: number of curve points (one per waveform pixel column)

    Returns:
        (integrated_lufs: float, curve: np.ndarray(n_points, float32))
        integrated_lufs is -inf for silence.
    """
    if not _HAS_NUMPY:
        return (float("-inf"), None)
    try:
        from scipy.signal import sosfilt

        # K-weighting SOS coefficients for 48 kHz (ITU-R BS.1770-4)
        # Stage 1: high-shelf pre-filter (+4 dB at ~1.5 kHz)
        # Stage 2: high-pass filter (75 Hz, 2nd order)
        _KW_SOS = np.array([
            [ 1.53512485958697, -2.69169618940638,  1.19839281085285,
              1.0,             -1.69065929318241,  0.73248077421585],
            [ 1.0,             -2.0,                1.0,
              1.0,             -1.99004745483398,  0.99007225036621],
        ], dtype=np.float64)

        # Apply K-weighting to both channels
        n_ch = min(data.shape[1], 2) if data.ndim > 1 else 1
        if data.ndim == 1:
            ch = [sosfilt(_KW_SOS, data.astype(np.float64))]
        else:
            ch = [sosfilt(_KW_SOS, data[:, c].astype(np.float64)) for c in range(n_ch)]

        # Mean-square per channel, summed (stereo weight = 1.0 each per BS.1770)
        ms_sum = sum(c ** 2 for c in ch)   # shape (N,)

        # ── Integrated LUFS ───────────────────────────────────────────────────
        block  = int(0.4 * sr)    # 400 ms
        hop    = block // 4       # 75% overlap
        blocks = []
        for s in range(0, len(ms_sum) - block, hop):
            mean = float(np.mean(ms_sum[s:s + block]))
            blocks.append(-0.691 + 10.0 * np.log10(max(mean, 1e-10)))
        blocks = np.array(blocks, dtype=np.float64)

        abs_gated = blocks[blocks > -70.0]
        if len(abs_gated) == 0:
            integrated = float("-inf")
        else:
            rel_thresh = float(np.mean(abs_gated)) - 10.0
            rel_gated  = abs_gated[abs_gated > rel_thresh]
            integrated = float(np.mean(rel_gated)) if len(rel_gated) > 0 else float("-inf")

        # ── Short-term loudness curve ─────────────────────────────────────────
        win    = int(3.0 * sr)    # 3-second window
        n_samp = len(ms_sum)
        curve  = np.full(n_points, -70.0, dtype=np.float32)
        for i in range(n_points):
            center = int(i * n_samp / n_points)
            s = max(0, center - win // 2)
            e = min(n_samp, s + win)
            mean = float(np.mean(ms_sum[s:e]))
            curve[i] = -0.691 + 10.0 * np.log10(max(mean, 1e-10))

        return (integrated, curve)
    except Exception:
        return (float("-inf"), None)


def probe_duration(path):
    """Return the duration of an audio file in seconds, or 0.0 on failure."""
    try:
        cmd = [
            _tool("ffprobe"),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
        data   = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def probe_info(path: str) -> dict:
    """Return {duration, sample_rate, channels, codec_name, bits_per_sample} for an audio file.

    Falls back to safe defaults on any failure so callers never need to guard
    against None.  codec_name is the raw ffprobe codec name (e.g. "mp3", "aac",
    "flac", "pcm_s24le") or "" on failure.  bits_per_sample is 0 for lossy
    formats where the concept doesn't apply.
    """
    try:
        cmd = [
            _tool("ffprobe"),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
        data   = json.loads(result.stdout)
        duration    = float(data.get("format", {}).get("duration", 0))
        sample_rate = 48000
        channels    = 2
        codec_name  = ""
        bits_per_sample = 0
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                sample_rate = int(stream.get("sample_rate", 48000))
                channels    = int(stream.get("channels", 2))
                codec_name  = stream.get("codec_name", "")
                # bits_per_raw_sample is set for lossless PCM/FLAC; bits_per_sample
                # for some containers.  Either may be absent or 0 for lossy formats.
                bits_per_sample = int(stream.get("bits_per_raw_sample", 0) or
                                      stream.get("bits_per_sample", 0))
                break
        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "codec_name": codec_name,
            "bits_per_sample": bits_per_sample,
        }
    except Exception:
        return {"duration": 0.0, "sample_rate": 48000, "channels": 2,
                "codec_name": "", "bits_per_sample": 0}
