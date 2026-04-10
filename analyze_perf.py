"""
analyze_perf.py -- Post-run analysis of MixABTestGPU performance logs.

Usage:
    python analyze_perf.py                  # reads latest CSV in logs/
    python analyze_perf.py path/to/file.csv # reads specific file
    python analyze_perf.py --all            # summarises every CSV in logs/

Output: a table of mean / p95 / max per section, plus a breakdown showing
what fraction of total tick time each section accounts for.

Interpretation guide (printed at the end):
  tick_mean_ms  > 8ms  at 60fps -> frame budget exceeded -> stutter risk
  waveform      > 50% of tick  -> texture upload is the bottleneck
  meters        > 20% of tick  -> sample_levels / correlation too expensive
  ctrl_spacer   > 5%  of tick  -> DPG query overhead, worth caching
"""

from __future__ import annotations
import csv
import glob
import os
import sys


# Columns that represent per-section mean values (order must match SECTIONS in perf_tracker.py)
_SECTIONS = (
    "drain_queue",
    "ctrl_spacer",
    "dot_click",
    "frame",
    "waveform",
    "meters",
    "spectrum_dispatch",
)

_FRAME_BUDGET_MS = 1000.0 / 60.0   # ~16.67 ms at 60 fps


def _latest_csv(log_dir: str) -> str | None:
    pattern = os.path.join(log_dir, "perf_*.csv")
    files   = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(col, "").strip()
        if v:
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _p(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s   = sorted(data)
    idx = max(0, int(pct / 100.0 * (len(s) - 1)))
    return s[idx]


def _fmt(v: float) -> str:
    return f"{v:7.3f}"


def _bar(frac: float, width: int = 20) -> str:
    filled = max(0, min(width, int(frac * width)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def analyze(path: str, verbose: bool = True) -> dict:
    rows = _load_csv(path)
    if not rows:
        print(f"  (empty file: {path})")
        return {}

    fps_vals  = _floats(rows, "fps")
    tick_vals = _floats(rows, "tick_mean_ms")
    tick_p95  = _floats(rows, "tick_p95_ms")
    tick_max  = _floats(rows, "tick_max_ms")

    mean_fps      = sum(fps_vals) / len(fps_vals)   if fps_vals  else 0.0
    mean_tick     = sum(tick_vals) / len(tick_vals)  if tick_vals else 0.0
    mean_tick_p95 = sum(tick_p95)  / len(tick_p95)   if tick_p95  else 0.0
    max_tick      = max(tick_max)                     if tick_max  else 0.0

    section_means: dict[str, float] = {}
    section_maxs:  dict[str, float] = {}
    for s in _SECTIONS:
        vals = _floats(rows, f"{s}_mean_ms")
        mxs  = _floats(rows, f"{s}_max_ms")
        section_means[s] = sum(vals) / len(vals) if vals else 0.0
        section_maxs[s]  = max(mxs)              if mxs  else 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {os.path.basename(path)}")
        print(f"{'='*60}")
        print(f"  Duration     : {_floats(rows, 'elapsed_s')[-1]:.1f}s  ({len(rows)} intervals)")
        print(f"  Avg FPS      : {mean_fps:.1f}")

        budget_pct = mean_tick / _FRAME_BUDGET_MS * 100
        flag = " <<<" if mean_tick > _FRAME_BUDGET_MS * 0.5 else ""
        print(f"  Tick mean    : {mean_tick:.3f} ms  ({budget_pct:.0f}% of 16.67ms budget){flag}")
        print(f"  Tick p95     : {mean_tick_p95:.3f} ms")
        print(f"  Tick max     : {max_tick:.3f} ms")
        print()

        print(f"  {'Section':<22} {'mean_ms':>8} {'max_ms':>8}  {'% of tick':>9}  Share")
        print(f"  {'-'*22} {'-'*8} {'-'*8}  {'-'*9}  {'-'*22}")
        for s in _SECTIONS:
            smean = section_means[s]
            smax  = section_maxs[s]
            frac  = smean / mean_tick if mean_tick > 0 else 0.0
            bar   = _bar(frac, 20)
            flag  = " <--" if frac > 0.3 else ""
            print(f"  {s:<22} {_fmt(smean):>8} {_fmt(smax):>8}  {frac*100:8.1f}%  {bar}{flag}")

        print()
        if mean_tick > _FRAME_BUDGET_MS * 0.75:
            print("  [!] Mean tick exceeds 75% of frame budget -- stutter very likely.")
        elif mean_tick > _FRAME_BUDGET_MS * 0.5:
            print("  [!] Mean tick exceeds 50% of frame budget -- stutter risk at higher loads.")
        else:
            print("  [ok] Mean tick within frame budget.")

        top = max(_SECTIONS, key=lambda s: section_means[s])
        if section_means[top] > 0:
            print(f"  [!] Largest section: '{top}'  ({section_means[top]:.3f} ms mean)")

    return {
        "path": path,
        "mean_fps": mean_fps,
        "mean_tick_ms": mean_tick,
        "tick_p95_ms": mean_tick_p95,
        "max_tick_ms": max_tick,
        "sections": section_means,
    }


def main() -> None:
    args = sys.argv[1:]
    log_dir = os.path.join(os.path.dirname(__file__), "logs")

    if not args:
        path = _latest_csv(log_dir)
        if path is None:
            print(f"No perf_*.csv files found in {log_dir}/")
            print("Start the app, enable Settings > Perf logging, then run again.")
            return
        analyze(path)

    elif args == ["--all"]:
        pattern = os.path.join(log_dir, "perf_*.csv")
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"No perf_*.csv files in {log_dir}/")
            return
        results = []
        for f in files:
            r = analyze(f, verbose=False)
            if r:
                results.append(r)
        # Summary comparison table
        print(f"\n{'='*70}")
        print(f"  Comparison: {len(results)} runs")
        print(f"{'='*70}")
        print(f"  {'File':<32} {'fps':>5} {'tick_mean':>10} {'tick_p95':>9} {'tick_max':>9}")
        print(f"  {'-'*32} {'-'*5} {'-'*10} {'-'*9} {'-'*9}")
        for r in results:
            name = os.path.basename(r["path"])[:32]
            print(f"  {name:<32} {r['mean_fps']:5.1f} {r['mean_tick_ms']:10.3f} "
                  f"{r['tick_p95_ms']:9.3f} {r['max_tick_ms']:9.3f}")

    else:
        for arg in args:
            if os.path.isfile(arg):
                analyze(arg)
            else:
                print(f"File not found: {arg}")


if __name__ == "__main__":
    main()
