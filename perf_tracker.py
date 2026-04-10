"""
perf_tracker.py — Lightweight per-frame performance tracker for MixABTestGPU.

Accumulates per-frame timings for up to one second, then flushes one CSV row.
Zero overhead when disabled — every public method short-circuits on the first check.

Tracked sections (instrumented in app.py):
  drain_queue       — _drain_queue + _drain_dnd_queue
  ctrl_spacer       — _update_ctrl_spacer (DPG measurement queries)
  dot_click         — _check_dot_click
  frame             — total _update_frame (includes waveform + meters + spectrum)
  waveform          — tex.render loop + dpg.set_value uploads inside _update_frame
  meters            — sample_levels + meter/corr bar updates
  spectrum_dispatch — _update_spectrum call (thread dispatch only, not compute)

Usage:
  tracker = PerfTracker()
  path = tracker.enable(log_dir)   # start; returns CSV path
  # each frame:
  tracker.begin_tick()
  tracker.begin_section("waveform")
  ...
  tracker.end_section("waveform")
  tracker.end_tick()
  # when done:
  tracker.disable()
"""

from __future__ import annotations
import csv
import os
import time
from datetime import datetime


SECTIONS = (
    "drain_queue",
    "ctrl_spacer",
    "dot_click",
    "frame",
    "waveform",
    "meters",
    "spectrum_dispatch",
)

_FLUSH_INTERVAL = 1.0   # seconds between CSV rows


def _p95(lst: list) -> float:
    """Return the 95th-percentile value from a list of floats."""
    if not lst:
        return 0.0
    s = sorted(lst)
    idx = max(0, int(0.95 * (len(s) - 1)))
    return s[idx]


class PerfTracker:
    """Per-frame performance tracker.  All methods are no-ops when disabled."""

    def __init__(self) -> None:
        self._enabled: bool = False
        self._f = None
        self._writer = None
        self._log_path: str = ""

        self._start_wall: float = 0.0
        self._flush_wall: float = 0.0

        self._tick_times: list = []
        self._section_times: dict = {s: [] for s in SECTIONS}

        self._tick_start:    float = 0.0
        self._section_start: dict  = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def log_path(self) -> str:
        return self._log_path

    def enable(self, log_dir: str) -> str:
        """Open a new CSV log file and start recording.  Returns the file path."""
        if self._enabled:
            self.disable()
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"perf_{stamp}.csv")
        self._f = open(path, "w", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.writer(self._f)

        header = ["elapsed_s", "fps", "tick_mean_ms", "tick_p95_ms", "tick_max_ms"]
        for s in SECTIONS:
            header += [f"{s}_mean_ms", f"{s}_max_ms"]
        header.append("frames")
        self._writer.writerow(header)

        now = time.perf_counter()
        self._start_wall = now
        self._flush_wall = now
        self._tick_times.clear()
        for s in SECTIONS:
            self._section_times[s].clear()
        self._log_path = path
        self._enabled = True
        return path

    def disable(self) -> None:
        """Flush any remaining data and close the log file."""
        if not self._enabled:
            return
        self._enabled = False
        self._flush()
        if self._f:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None
        self._writer = None

    # ── Per-frame instrumentation ──────────────────────────────────────────────

    def begin_tick(self) -> None:
        if not self._enabled:
            return
        self._tick_start = time.perf_counter()

    def end_tick(self) -> None:
        if not self._enabled:
            return
        now = time.perf_counter()
        self._tick_times.append((now - self._tick_start) * 1000.0)
        if now - self._flush_wall >= _FLUSH_INTERVAL:
            self._flush()
            self._flush_wall = now

    def begin_section(self, name: str) -> None:
        if not self._enabled:
            return
        self._section_start[name] = time.perf_counter()

    def end_section(self, name: str) -> None:
        if not self._enabled:
            return
        t = self._section_start.get(name)
        if t is not None:
            self._section_times[name].append((time.perf_counter() - t) * 1000.0)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        if not self._writer or not self._tick_times:
            return
        now     = time.perf_counter()
        elapsed = now - self._start_wall
        fps     = len(self._tick_times)

        tt  = self._tick_times
        mn  = sum(tt) / len(tt)
        mx  = max(tt)
        p95 = _p95(tt)

        row: list = [
            f"{elapsed:.1f}",
            fps,
            f"{mn:.3f}",
            f"{p95:.3f}",
            f"{mx:.3f}",
        ]
        for s in SECTIONS:
            st = self._section_times[s]
            if st:
                smn = sum(st) / len(st)
                smx = max(st)
                row += [f"{smn:.3f}", f"{smx:.3f}"]
            else:
                row += ["", ""]
        row.append(fps)

        self._writer.writerow(row)

        self._tick_times.clear()
        for s in SECTIONS:
            self._section_times[s].clear()
