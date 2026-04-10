"""
_test_sweep.py — Load audiosweep.mp3, enable fullscreen spectrogram,
play, and screenshot after 19 seconds for reassignment diagnostics.
All setup driven from the render loop (no background thread timing).
"""
import sys, os, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import MixABTestGPU
import dearpygui.dearpygui as dpg

_HERE     = os.path.dirname(os.path.abspath(__file__))
SWEEP     = os.path.join(_HERE, "testfiles", "sweeptest.wav")
OUT_PATH  = os.path.join(_HERE, "screenshot_sweep.png")

app = MixABTestGPU()
dpg.set_exit_callback(app._on_close)

# State machine — advance through setup steps in the render loop
# so each step only fires after DPG is fully running.
STATE          = 0
TRACK_LOADED   = False
screenshot_done = False

# Load file from background thread (correct pattern per CLAUDE.md)
def _load():
    time.sleep(1.0)
    app._post(app._add_track, SWEEP)

threading.Thread(target=_load, daemon=True).start()

frame  = 0
# 19 seconds of playback at ~60fps = 1140 frames; allow 5s for decode = 300 frames startup
PLAY_FRAME       = 300   # frame to start playing (after decode)
SCREENSHOT_FRAME = PLAY_FRAME + 780   # play_frame + 13s

while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()
    app._tick()
    frame += 1

    # Step 1: once track is loaded and play frame reached, enable fullscreen + play
    if frame == PLAY_FRAME:
        if app._tracks:
            app._set_fft_type("spectrogram")
            app._toggle_spectrogram_fullscreen()
            app.toggle_play()
            print(f"[frame {frame}] Started playback + fullscreen", flush=True)
        else:
            # Track not loaded yet — push screenshot frame later
            PLAY_FRAME       += 60
            SCREENSHOT_FRAME += 60
            print(f"[frame {frame}] Track not ready, retrying at {PLAY_FRAME}", flush=True)

    # Step 2: screenshot
    if frame == SCREENSHOT_FRAME and not screenshot_done:
        dpg.output_frame_buffer(OUT_PATH)
        print(f"[frame {frame}] Screenshot saved: {OUT_PATH}", flush=True)
        screenshot_done = True

    if frame >= SCREENSHOT_FRAME + 10:
        dpg.stop_dearpygui()

dpg.destroy_context()
print("Done.", flush=True)
