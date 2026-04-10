"""
_test_perf.py — Automated performance measurement for MixABTestGPU.

Designed to be run by Claude Code via Bash to get before/after numbers.
Prints the full analysis table to stdout, saves CSV to logs/.

Usage:
    python _test_perf.py                            # default window size, 4 test tracks
    python _test_perf.py --width 2560 --height 1440 # stress-test at 2K
    python _test_perf.py --no-files                 # baseline: no tracks loaded
    python _test_perf.py --label "before dirty-fix" # adds label to output header

Exit code: 0 = ok, 1 = failed / no CSV written.

Frame schedule:
    0  - 179  : warmup — files loading, decoding, first render
    180 - 1379 : MEASURE (20s at 60fps) — perf logged, playback running
    1380+      : flush, analyze, exit
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dearpygui.dearpygui as dpg
from app import MixABTestGPU

_HERE       = os.path.dirname(os.path.abspath(__file__))
_TESTFILES  = [
    os.path.join(_HERE, "testfiles", "Stage Frights PREVIEW.mp3"),
    os.path.join(_HERE, "testfiles", "Pex_L_-_Reflections_DEMO.mp3"),
    os.path.join(_HERE, "testfiles", "Pex L - Stuck To Me Now MASTER.mp3"),
    os.path.join(_HERE, "testfiles", "Pex L - Only You.mp3"),
]

WARMUP_FRAMES   = 180   # 3 s at 60 fps — files load + decode
MEASURE_FRAMES  = 1200  # 20 s of measurement
EXIT_FRAME      = WARMUP_FRAMES + MEASURE_FRAMES + 60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MixABTestGPU perf benchmark")
    p.add_argument("--width",    type=int, default=0, help="Viewport width  (0 = use saved prefs)")
    p.add_argument("--height",   type=int, default=0, help="Viewport height (0 = use saved prefs)")
    p.add_argument("--no-files", action="store_true",  help="Don't load test files (pure overhead baseline)")
    p.add_argument("--label",    type=str, default="",  help="Free-text label printed in header")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.label:
        print(f"\n=== {args.label} ===", flush=True)

    # ── Create app ────────────────────────────────────────────────────────────
    app = MixABTestGPU()

    # ── Load test files via thread (same pattern as _test_launch.py) ─────────
    if not args.no_files:
        def _load() -> None:
            time.sleep(1.0)   # wait for tick_ready / first frame
            for f in _TESTFILES:
                app._post(app._add_track, f)
                time.sleep(0.1)
        threading.Thread(target=_load, daemon=True).start()

    # ── Render loop state ─────────────────────────────────────────────────────
    frame_count    = 0
    resize_done    = False
    perf_started   = False
    perf_stopped   = False
    csv_path: str  = ""

    dpg.set_exit_callback(app._on_close)

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        frame_count += 1

        # Always drive _tick so we measure real workload
        if app._tick_ready:
            app._tick()

        # ── Viewport resize (frame 3 — after first-frame setup fires) ─────────
        if frame_count == 3 and args.width > 0 and args.height > 0 and not resize_done:
            dpg.set_viewport_width(args.width)
            dpg.set_viewport_height(args.height)
            resize_done = True
            print(f"[frame {frame_count}] viewport -> {args.width}x{args.height}", flush=True)

        # ── Start perf + playback after warmup ────────────────────────────────
        if frame_count == WARMUP_FRAMES and not perf_started:
            perf_started = True
            if not args.no_files:
                app._engine.seek(0)
                app._engine.play()
                app._playing = True
            csv_path = app._perf.enable(app._perf_log_dir)
            w = dpg.get_viewport_width()
            h = dpg.get_viewport_height()
            print(f"[frame {frame_count}] perf logging started  "
                  f"viewport={w}x{h}  tracks={len(app._tracks)}", flush=True)

        # ── Stop measurement ──────────────────────────────────────────────────
        if frame_count == WARMUP_FRAMES + MEASURE_FRAMES and not perf_stopped:
            perf_stopped = True
            app._perf.disable()
            print(f"[frame {frame_count}] perf logging stopped -> {os.path.basename(csv_path)}", flush=True)

        # ── Exit ──────────────────────────────────────────────────────────────
        if frame_count >= EXIT_FRAME:
            dpg.stop_dearpygui()

    # ── Inline analysis (before destroy — destroy can block on GPU cleanup) ──
    if not csv_path or not os.path.isfile(csv_path):
        # Fall back to most recent file in logs/
        candidates = sorted(glob.glob(os.path.join(_HERE, "logs", "perf_*.csv")))
        csv_path = candidates[-1] if candidates else ""

    if csv_path and os.path.isfile(csv_path):
        from analyze_perf import analyze
        analyze(csv_path)
        exit_code = 0
    else:
        print("ERROR: no CSV file was written.", flush=True)
        exit_code = 1

    # Destroy context on a daemon thread — GPU teardown can stall the main
    # thread long enough for Windows to show "Not Responding".  Results are
    # already printed, so we exit as soon as the analysis is done.
    threading.Thread(target=dpg.destroy_context, daemon=True).start()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
