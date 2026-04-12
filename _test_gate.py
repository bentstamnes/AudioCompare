"""Test different _PRE_SCATTER_GATE values on Sloppy Groove, save PNGs for comparison."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_ops
import renderers.spectrogram as spmod
from renderers.spectrogram import decode_sample_rate, n_cols_for_samples
from PIL import Image

WAV    = os.path.join(os.path.dirname(__file__), "testfiles", "Sloppy Groove 85bpm.wav")
WAVE_W = 1800
HEIGHT = 300
RESULTS = os.path.join(os.path.dirname(__file__), "_test_spectro_results")

info    = audio_ops.probe_info(WAV)
sr      = decode_sample_rate(int(info.get("sample_rate", 48000)))
audio   = audio_ops.decode_audio(WAV, samplerate=sr, channels=1)
samples = audio.ravel()
n_cols  = n_cols_for_samples(len(samples), WAVE_W)
n_samp  = len(samples)
print(f"n_cols={n_cols}  sr={sr}")


def save_png(arr_rgb, path, scale_w=0):
    img = Image.fromarray(arr_rgb, "RGB")
    if scale_w and scale_w != arr_rgb.shape[1]:
        img = img.resize((scale_w, arr_rgb.shape[0]), Image.NEAREST)
    img.save(path)


def _next_prefix():
    existing = [f for f in os.listdir(RESULTS) if f.endswith("_scaled.png")]
    return f"{len(existing) + 1:03d}"


def compute_with_gate(samples, n_cols, height, sample_rate, pre_gate):
    arr    = np.asarray(samples, dtype=np.float32)
    n_samp = len(arr)
    centres = np.linspace(
        spmod._FFT_SIZE // 2, n_samp - spmod._FFT_SIZE // 2 - 1,
        n_cols, dtype=np.float32,
    ).astype(np.int32)
    _MAG_GATE = 10.0 ** (spmod._SPECTRUM_DB_FLOOR / 20.0)

    row_t     = np.arange(height, dtype=np.float32) / max(height - 1, 1)
    freq_rows = (spmod._FS_FREQ_HI * np.exp(-row_t * spmod._LOG_RANGE)).astype(np.float32)
    tilt_db   = (spmod._TILT_DB_PER_DECADE * np.log10(np.maximum(freq_rows, 1.0) / spmod._TILT_F_REF)).astype(np.float32)

    k_arr     = np.arange(spmod._N_BINS, dtype=np.float32)
    f_lo_k    = np.clip(np.maximum((k_arr - 0.5) * (sample_rate / float(spmod._FFT_SIZE)), spmod._FS_FREQ_LO), spmod._FS_FREQ_LO, spmod._FS_FREQ_HI)
    f_hi_k    = np.clip(np.minimum((k_arr + 0.5) * (sample_rate / float(spmod._FFT_SIZE)), spmod._FS_FREQ_HI), spmod._FS_FREQ_LO, spmod._FS_FREQ_HI)
    row_upper = (np.log(spmod._FS_FREQ_HI / f_hi_k) / spmod._LOG_RANGE * (height - 1)).astype(np.float32)
    row_lower = (np.log(spmod._FS_FREQ_HI / f_lo_k) / spmod._LOG_RANGE * (height - 1)).astype(np.float32)
    half_bw_k = np.clip((row_lower - row_upper) / 2.0, 0.5, spmod._BW_MAX_ROWS).astype(np.float32)

    tiles = []
    for t_start in range(0, n_cols, spmod._N_COLS_MAX):
        t_end   = min(t_start + spmod._N_COLS_MAX, n_cols)
        t_ncols = t_end - t_start
        tc      = centres[t_start:t_end]

        starts  = np.clip(tc - spmod._FFT_SIZE // 2, 0, n_samp - spmod._FFT_SIZE)
        indices = starts[:, None] + np.arange(spmod._FFT_SIZE, dtype=np.int32)[None, :]
        raw     = arr[indices]

        X_h  = spmod._fft.rfft(raw * spmod._HANN[None, :],    axis=1)
        X_dh = spmod._fft.rfft(raw * spmod._HANN_DH[None, :], axis=1)
        X_th = spmod._fft.rfft(raw * spmod._HANN_TH[None, :], axis=1)
        mag  = (np.abs(X_h) / spmod._FFT_NORM).astype(np.float32)

        denom     = np.where(np.abs(X_h) > 1e-10, X_h, 1e-10)
        ratio     = X_dh / denom
        k_bins    = np.arange(spmod._N_BINS, dtype=np.float32)[None, :]
        omega_hat = np.clip(k_bins - np.imag(ratio).astype(np.float32) * (spmod._FFT_SIZE / (2.0 * np.pi)), 0.0, float(spmod._N_BINS - 1))

        t_hat_offset = (np.real(X_th / denom) - spmod._FFT_SIZE // 2).astype(np.float32)
        causal       = (t_hat_offset <= spmod._CAUSAL_GATE_SAMPLES).ravel()

        col_peak_l = mag.max(axis=1, keepdims=True)
        pg         = (mag > col_peak_l * pre_gate).ravel() if pre_gate > 0.0 else np.ones(mag.size, dtype=bool)
        valid      = (mag > _MAG_GATE).ravel() & causal & pg

        freq_hat = np.clip(omega_hat * (sample_rate / float(spmod._FFT_SIZE)), spmod._FS_FREQ_LO, spmod._FS_FREQ_HI)
        row_f    = (np.log(spmod._FS_FREQ_HI / freq_hat) / spmod._LOG_RANGE * (height - 1)).astype(np.float32)

        valid_idx = np.where(valid)[0]
        size      = height * t_ncols

        if len(valid_idx) == 0:
            out_long = np.zeros((height, t_ncols), dtype=np.float32)
        else:
            k_v        = valid_idx % spmod._N_BINS
            col_v      = (valid_idx // spmod._N_BINS).astype(np.int32)
            mag_v      = ((mag ** 1.5).ravel())[valid_idx]
            row_hat_v  = row_f.ravel()[valid_idx]
            hbw_v      = half_bw_k[k_v]

            fill_lo_v  = np.clip(np.floor(row_hat_v - hbw_v).astype(np.int32), 0, height - 1)
            fill_hi_v  = np.clip(np.ceil (row_hat_v + hbw_v).astype(np.int32), 0, height - 1)
            n_rep_v    = np.maximum(1, fill_hi_v - fill_lo_v + 1)
            total_ops  = int(n_rep_v.sum())

            col_rep     = np.repeat(col_v,     n_rep_v)
            mag_rep     = np.repeat(mag_v,     n_rep_v)
            fill_lo_rep = np.repeat(fill_lo_v, n_rep_v)
            row_hat_rep = np.repeat(row_hat_v, n_rep_v)
            hbw_rep     = np.repeat(hbw_v,     n_rep_v)

            cum_n     = np.empty(len(n_rep_v) + 1, dtype=np.int64)
            cum_n[0]  = 0
            np.cumsum(n_rep_v.astype(np.int64), out=cum_n[1:])
            positions = np.arange(total_ops, dtype=np.int64)
            grp_start = np.repeat(cum_n[:-1], n_rep_v)
            offsets   = (positions - grp_start).astype(np.int32)
            row_rep   = np.clip(fill_lo_rep + offsets, 0, height - 1)

            dist   = np.abs(row_rep.astype(np.float32) + 0.5 - row_hat_rep)
            weight = np.maximum(0.0, 1.0 - dist / np.maximum(hbw_rep, 0.5))

            flat_rep = (row_rep.astype(np.int64) * t_ncols + col_rep.astype(np.int64))
            out_long = np.bincount(flat_rep, weights=(mag_rep * weight).astype(np.float64), minlength=size).reshape(height, t_ncols).astype(np.float32)

        band_h = max(1, height // spmod._N_BANDS)
        for b in range(spmod._N_BANDS):
            r0 = b * band_h; r1 = height if b == spmod._N_BANDS - 1 else r0 + band_h
            band = out_long[r0:r1, :]; bmax = np.maximum(band.max(axis=0, keepdims=True), 1e-10)
            out_long[r0:r1, :] *= (band > bmax * spmod._BAND_GATE_THRESH)

        db       = 20.0 * np.log10(np.maximum(out_long, 1e-7)) + tilt_db[:, np.newaxis]
        amp_rows = np.clip((db - spmod._SPECTRUM_DB_FLOOR) / (-spmod._SPECTRUM_DB_FLOOR), 0.0, 1.0).astype(np.float32)
        lut_idx  = np.clip((amp_rows * 255.0).astype(np.int32), 0, 255)
        rgb      = spmod._LUT[lut_idx]
        out = np.empty((height, t_ncols, 4), dtype=np.float32)
        out[..., :3] = rgb; out[..., 3] = 0.9
        tiles.append(out)
    return tiles


for gate in [0.05, 0.01, 0.0]:
    t0    = time.perf_counter()
    tiles = compute_with_gate(samples, n_cols, HEIGHT, sr, gate)
    dt    = time.perf_counter() - t0
    full  = np.concatenate(tiles, axis=1)
    rgb   = (np.clip(full[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
    prefix = _next_prefix()
    label  = f"gate{gate:.2f}"
    path   = os.path.join(RESULTS, f"{prefix}_groove_{label}_scaled.png")
    img = Image.fromarray(rgb, "RGB")
    img = img.resize((WAVE_W, HEIGHT), Image.NEAREST)
    img.save(path)
    print(f"gate={gate:.2f}  t={dt:.2f}s  -> {path}")

print("Done.")
