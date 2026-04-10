import sys, os, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import MixABTestGPU
import dearpygui.dearpygui as dpg

app = MixABTestGPU()
_HERE = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(_HERE, "testfiles", "Stage Frights PREVIEW.mp3")]
log = open(os.path.join(_HERE, "_debug_out.txt"), "w")

def _load():
    time.sleep(0.5)
    app._post(app._add_track, files[0])

threading.Thread(target=_load, daemon=True).start()
dpg.set_exit_callback(app._on_close)

frame_count = 0
while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()
    frame_count += 1
    if frame_count == 200:
        for tag in [app._tag_play_btn, app._tag_time_lbl, app._tag_mute_btn, app._tag_vol_slider, app._tag_dots_group]:
            st = dpg.get_item_state(tag)
            log.write(f"{tag}: rect_min={st.get('rect_min')} rect_size={st.get('rect_size')}\n")
        log.flush()
        dpg.output_frame_buffer(os.path.join(_HERE, "screenshot_test.png"))
    if frame_count >= 205:
        dpg.stop_dearpygui()

log.close()
dpg.destroy_context()
