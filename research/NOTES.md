# Spectrogram Research Notes

Notes extracted from the PDFs in this folder. Written so future sessions don't need to re-parse the PDFs.

---

## Why reassignment? (the problem)

Standard `|rfft|²` spectrogram smears energy across FFT bins. At 44.1kHz with FFT_SIZE=2048, each bin is ~21.6 Hz wide. A 50 Hz bass tone smears across 2-3 bins — visible as a thick blurry line. The Heisenberg uncertainty principle means you can't fix this by changing window size alone: longer window = better frequency resolution but worse time resolution.

**Reassignment solves this** by using phase information (normally discarded) to relocate energy to its true center of gravity in time-frequency space. Sinusoids and harmonics sharpen to thin lines. Noise scatters (cleans the noise floor too).

---

## The method (practical implementation)

For each frame (one column in the texture), compute **three FFTs** instead of one:

```python
n    = np.arange(FFT_SIZE, dtype=np.float32)
h    = np.hanning(FFT_SIZE).astype(np.float32)          # standard Hann window
dh   = np.gradient(h).astype(np.float32)                # time-derivative of Hann
th   = (n * h).astype(np.float32)                       # time-ramp × Hann

X_h  = rfft(frame * h)     # standard — shape (n_bins,)
X_dh = rfft(frame * dh)    # derivative window
X_th = rfft(frame * th)    # ramp window
```

Compute reassigned frequency bin for each bin k:

```python
# Instantaneous frequency (in fractional bins)
# SIGN IS SUBTRACT — verified empirically with np.gradient(hann) as dh.
# When freq > bin_center, Im(X_dh/X_h) is negative, so subtracting pushes bin up.
omega_hat = k - np.imag(X_dh / X_h) * FFT_SIZE / (2 * np.pi)
```

The magnitude at bin k gets painted at `omega_hat` instead of k. Bins where `|X_h|` is below a noise threshold are skipped.

Group delay (`t_hat`) is also computable but for a *static* pre-computed spectrogram it has no visual effect — we only care about frequency placement.

---

## Scatter-accumulate implementation

After computing `omega_hat` (shape `(n_cols, n_bins)`):

1. Clamp to valid bin range `[0, n_bins-1]`
2. Map fractional bins → display rows via the log-frequency row map (same as current `_get_row_map`)
3. Use `np.add.at(out_col, row_idx, magnitude)` to accumulate energy at reassigned rows

**Key:** this is an *accumulate* not an *assign* — multiple bins can reassign to the same row, and their energies add. This is correct behaviour (energy is conserved).

---

## Noise threshold / masking

Bins where `|X_h[k]|` is below a threshold (e.g. 1% of max for that frame) have unreliable phase estimates — their `omega_hat` is noise-driven and scatters randomly. Skip these by masking before the scatter step:

```python
mag = np.abs(X_h)
mask = mag > (mag.max(axis=1, keepdims=True) * 0.01)
# only scatter where mask is True
```

---

## Sources

- **0903.pdf** — Fitz & Fulop, "A Unified Theory of Time-Frequency Reassignment", arXiv:0903.3080 (2009). Best mathematical treatment; derivation of ω̂ and t̂.
- **document.pdf** — Flandrin, Auger, Chassande-Mottin, "Time-Frequency reassignment: from principles to algorithms", CRC Press 2002. HAL:hal-00414583. Centre-of-gravity geometric interpretation.
- **Reassigned Spectrogram.pdf** — Cornell ECE overview (Land). Practical worked examples with bird calls and fish sounds. Shows color_1_dot rendering mode.
- **Reassignment method - Wikipedia.pdf** — Accessible summary; acoustic bass example (73.4 Hz fundamental, Kaiser window).

---

## Current implementation status (as of 2026-04-11)

`renderers/spectrogram.py` → `compute_static()` uses standard `|rfft|²`. Correct frequency mapping and UV-scroll architecture are in place (prerequisite work done). Reassignment is the next step — only `compute_static()` needs to change; rendering path is untouched.

**The output of `compute_static` is an `(height, n_cols, 4)` float32 RGBA array.** Shape does not change with reassignment. `SpectrogramStrip.set_data()` and all rendering code are unaffected.

---

## Expected visual result on "Stonebank - Onyx" (44.1kHz WAV, ~50 Hz fundamental)

- Sub-bass lines (50-100 Hz) currently: thick blurry horizontal bands
- After reassignment: thin sharp lines, individual harmonics distinguishable
- Noise floor: visually cleaner (scattered noise contributes less to any single pixel)
