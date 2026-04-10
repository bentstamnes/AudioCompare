import sys, os, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import MixABTestGPU
import dearpygui.dearpygui as dpg

app = MixABTestGPU()

_HERE = os.path.dirname(os.path.abspath(__file__))
files = [
    os.path.join(_HERE, "testfiles", "Stage Frights PREVIEW.mp3"),
    os.path.join(_HERE, "testfiles", "Pex_L_-_Reflections_DEMO.mp3"),
    os.path.join(_HERE, "testfiles", "Pex L - Stuck To Me Now MASTER.mp3"),
    os.path.join(_HERE, "testfiles", "Pex L - Only You.mp3"),
]

def _load_after_delay():
    time.sleep(0.5)
    for f in files:
        app._post(app._add_track, f)
        time.sleep(0.1)

threading.Thread(target=_load_after_delay, daemon=True).start()
dpg.set_exit_callback(app._on_close)

frame_count = 0
SCREENSHOT_FRAME = 450  # 5 seconds at 60fps
while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()
    frame_count += 1
    if frame_count == SCREENSHOT_FRAME:
        out = os.path.join(_HERE, "screenshot_test.png")
        dpg.output_frame_buffer(out)
        print(f"Saved: {out}", flush=True)
    if frame_count >= SCREENSHOT_FRAME + 5:
        dpg.stop_dearpygui()

dpg.destroy_context()
