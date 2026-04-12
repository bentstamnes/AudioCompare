#!/usr/bin/env python3
"""
_test_spectro.py — Headless spectrogram gap analysis tool.

Decodes a WAV file, runs compute_static(), and saves two PNGs:
  _test_spectro_native.png  — 1:1 column-per-pixel (unscaled, no gap hiding)
  _test_spectro_scaled.png  — rescaled to WAVE_W px wide for quick visual scan

Prints gap statistics for the target frequency row to stdout.
Highlights gap columns as red pixels on the target row in both PNGs.

Usage:
    python _test_spectro.py [wav] [wave_w] [height] [target_hz]

Defaults:
    wav       = testfiles/subsustain.wav
    wave_w    = 1800
    height    = 300
    target_hz = 43.0
"""

import sys, os, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import audio_ops
from renderers.spectrogram import (
    compute_static, decode_sample_rate, n_cols_for_samples,
    _FS_FREQ_LO, _FS_FREQ_HI,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WAV       = os.path.join(HERE, "testfiles", "subsustain.wav")
DEFAULT_WAVE_W    = 1800
DEFAULT_HEIGHT    = 300
DEFAULT_TARGET_HZ = 43.0

# Rows around target frequency to consider as "signal present"
ANALYSIS_HALF_WIDTH = 5
# Post-LUT brightness below this = gap (0–1 float)
GAP_THRESHOLD = 0.05

RESULTS_DIR = os.path.join(HERE, "_test_spectro_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _next_run_prefix() -> str:
    """Return zero-padded run number based on existing files in RESULTS_DIR."""
    existing = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_scaled.png")]
    return f"{len(existing) + 1:03d}"

_LOG_RANGE = math.log(_FS_FREQ_HI / _FS_FREQ_LO)


def freq_to_row(freq_hz: float, height: int) -> int:
    """Log-scale frequency → pixel row (row 0 = top = high freq)."""
    freq_hz = max(freq_hz, _FS_FREQ_LO)
    t = math.log(_FS_FREQ_HI / freq_hz) / _LOG_RANGE
    return int(round(t * (height - 1)))


def save_png(arr_rgb_uint8: np.ndarray, path: str, scale_w: int = 0) -> None:
    """Save (H, W, 3) uint8 array as PNG, optionally rescaling width."""
    try:
        from PIL import Image
        img = Image.fromarray(arr_rgb_uint8, "RGB")
        if scale_w > 0 and scale_w != arr_rgb_uint8.shape[1]:
            img = img.resize((scale_w, arr_rgb_uint8.shape[0]), Image.NEAREST)
        img.save(path)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h, w = arr_rgb_uint8.shape[:2]
        fig_w = max(6, (scale_w if scale_w else w) / 100)
        fig_h = max(2, h / 100)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        disp = arr_rgb_uint8
        if scale_w > 0 and scale_w != w:
            from PIL import Image as _I
            disp = np.array(_I.fromarray(disp).resize((scale_w, h), _I.NEAREST))
        ax.imshow(disp, aspect="auto", interpolation="nearest")
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()


def analyze_gaps(full_rgba: np.ndarray, target_row: int, half_w: int, threshold: float):
    """
    Returns (gap_mask, runs) where:
      gap_mask  — bool array (n_cols,), True = gap column
      runs      — list of (start_col, length) tuples
    """
    r0 = max(0, target_row - half_w)
    r1 = min(full_rgba.shape[0] - 1, target_row + half_w)
    window   = full_rgba[r0:r1 + 1, :, :3]          # (window_h, n_cols, 3)
    col_max  = window.max(axis=(0, 2))               # (n_cols,)
    gap_mask = col_max < threshold

    runs = []
    cols = np.where(gap_mask)[0]
    if len(cols):
        rs, rl = cols[0], 1
        for i in range(1, len(cols)):
            if cols[i] == cols[i - 1] + 1:
                rl += 1
            else:
                runs.append((rs, rl))
                rs, rl = cols[i], 1
        runs.append((rs, rl))

    return gap_mask, runs


def main():
    wav       = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV
    wave_w    = int(sys.argv[2])   if len(sys.argv) > 2 else DEFAULT_WAVE_W
    height    = int(sys.argv[3])   if len(sys.argv) > 3 else DEFAULT_HEIGHT
    target_hz = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_TARGET_HZ

    print(f"=== Spectrogram gap test ===")
    print(f"File:       {wav}")
    print(f"Resolution: {wave_w} × {height}")
    print(f"Target:     {target_hz} Hz")

    # ── Decode ────────────────────────────────────────────────────────────────
    info = audio_ops.probe_info(wav)
    sr   = decode_sample_rate(int(info.get("sample_rate", 48000)))
    print(f"Decoding at {sr} Hz ...")
    audio = audio_ops.decode_audio(wav, samplerate=sr, channels=1)
    if audio is None:
        print("ERROR: decode_audio returned None")
        sys.exit(1)

    samples = audio.ravel()
    n_samp  = len(samples)
    n_cols  = n_cols_for_samples(n_samp, wave_w)
    dur_s   = n_samp / sr
    print(f"Duration:   {dur_s:.2f}s  |  samples={n_samp}  |  n_cols={n_cols}")

    # ── Compute ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("Running compute_static() ...")
    tiles = compute_static(samples, n_cols, height, sr)
    dt = time.perf_counter() - t0
    print(f"Computed in {dt:.2f}s")

    full = np.concatenate(tiles, axis=1)   # (height, total_cols, 4) float32
    total_cols = full.shape[1]
    print(f"Output:     {total_cols} columns × {height} rows")

    # ── Gap analysis ──────────────────────────────────────────────────────────
    target_row = freq_to_row(target_hz, height)
    print(f"\nTarget row: {target_row}  ({target_hz} Hz, rows "
          f"{max(0, target_row - ANALYSIS_HALF_WIDTH)}–"
          f"{min(height-1, target_row + ANALYSIS_HALF_WIDTH)})")

    gap_mask, runs = analyze_gaps(full, target_row, ANALYSIS_HALF_WIDTH, GAP_THRESHOLD)

    n_gaps    = gap_mask.sum()
    gap_pct   = 100.0 * n_gaps / total_cols
    run_sizes = [r[1] for r in runs]

    print(f"\n--- Gap report ---")
    print(f"Total cols:   {total_cols}")
    print(f"Gap cols:     {n_gaps}  ({gap_pct:.1f}%)")
    print(f"Gap runs:     {len(runs)}")
    if runs:
        print(f"Largest gap:  {max(run_sizes)} cols  "
              f"({max(run_sizes)/total_cols*100:.1f}% of track)")
        print(f"Mean gap:     {sum(run_sizes)/len(run_sizes):.1f} cols")
        print(f"\nFirst 15 runs:")
        for start, length in runs[:15]:
            pct = 100.0 * start / total_cols
            dur = length / total_cols * dur_s
            print(f"  col {start:6d} ({pct:5.1f}%)  len={length:4d}  ({dur*1000:.0f} ms)")
        if len(runs) > 15:
            print(f"  ... {len(runs) - 15} more runs")
    else:
        print("  *** NO GAPS — line is continuous ***")

    # ── Build annotated RGB image ─────────────────────────────────────────────
    # Base: spectrogram RGB
    rgb = (np.clip(full[..., :3], 0.0, 1.0) * 255).astype(np.uint8)

    # Draw a dim cyan guide line at the target row (not in gap columns)
    # and bright red on gap columns so they're immediately visible
    for col in range(total_cols):
        if gap_mask[col]:
            rgb[target_row, col] = [255, 0, 0]      # red = gap
        else:
            # Brighten the target row slightly so we can see where signal IS
            existing = rgb[target_row, col].astype(np.int32)
            rgb[target_row, col] = np.clip(existing + [0, 60, 60], 0, 255).astype(np.uint8)

    # ── Save PNGs (numbered so history is preserved) ──────────────────────────
    prefix   = _next_run_prefix()
    out_nat  = os.path.join(RESULTS_DIR, f"{prefix}_native.png")
    out_scl  = os.path.join(RESULTS_DIR, f"{prefix}_scaled.png")

    save_png(rgb, out_nat)
    print(f"\nSaved (native):  {out_nat}  ({total_cols}×{height})")

    save_png(rgb, out_scl, scale_w=wave_w)
    print(f"Saved (scaled):  {out_scl}  ({wave_w}×{height})")

    # Summary score (lower = better)
    print(f"\nGAP SCORE: {gap_pct:.2f}%  ({n_gaps}/{total_cols} cols)")


if __name__ == "__main__":
    main()
