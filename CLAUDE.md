# MixABTestGPU — Claude context

## Project overview

MixABTestGPU is a Windows desktop audio comparison tool built with Dear PyGui 2.2 (GPU/OpenGL, 60fps vsync). Users load up to 4 audio files and compare them by switching between them during playback. Features: waveform display, peak/RMS meters, stereo correlation meter, LUFS loudness overlay, FFT spectrum overlay, and a null-test mode (REF vs candidate difference signal).

**Main files:**
- `app.py` — UI + application logic (MixABTestGPU class)
- `player.py` — AudioEngine (sounddevice/PortAudio)
- `renderers/waveform.py` — numpy → DPG dynamic texture waveform renderer
- `renderers/meters.py` — DPG drawlist VU meter strip
- `renderers/correlation.py` — stereo correlation bar
- `renderers/spectrum.py` — DPG FFT spectrum overlay
- `audio_ops.py` — ffmpeg decode, waveform extraction, loudness analysis
- `null_ops.py` — null-test computation
- `win_drop.py` — Win32 IDropTarget for OS-level Explorer drag-and-drop
- `_test_launch.py` — automated screenshot test script

**Key architectural decisions:**
- DPG dynamic texture (`add_dynamic_texture` + `set_value`) for waveforms — numpy-vectorised RGBA rendering uploaded to GPU each frame
- No Tkinter anywhere; all UI is DPG
- OS drag-and-drop via Win32 `RegisterDragDrop` / `IDropTarget` (ctypes); HWND found via `FindWindowW("MixABTest GPU")` after `show_viewport()`
- Thread-safe message queue (`queue.Queue`) for audio/waveform decode results posted to main thread
- DPG item creation outside context managers: always use `parent=` parameter explicitly
- Prefs file: `%APPDATA%\MixABTestGPU\prefs.json`

DPG's render loop runs on the main thread. Audio playback runs on a sounddevice/PortAudio real-time thread.

---

## User preferences

- Allows all bash commands — do not ask permission per-command.
- Prefers full implementations, not stubs or skeleton code.
- Dear PyGui 2.2 is the chosen framework — no Tkinter hybrid.
- 60fps target, `vsync=True`.
- Distribution via PyInstaller `.exe` is required.
- Explorer drag-and-drop is a hard requirement.


---

## Architecture decisions

### Spectrum overlay — active vs inactive dimming

When "FFT show active only" is off, inactive tracks show their spectrum at 50% RGB + 60% alpha (≈50% perceived brightness). `SpectrumOverlay` stores `_bar_color_active` and `_bar_color_dim`. `_apply_spectrum` calls `ov.set_active(t["id"] == active_id)` before `ov.update()` for each visible track. `set_active` short-circuits if the color is already correct.

### Analyzer layout — fixed reserved space

`wave_w` is always `vw - METER_W`, regardless of whether the VU meter is currently visible. The right-side column is permanently reserved for the meter and correlation bar.

`MeterStrip` and `CorrelationBar` are always created for every slot in `_build_slot_window`, with initial visibility set via `set_visible()`. Toggling them only calls `set_visible()` — no texture reallocation, no rebuild.

**Rationale:** Rebuilding on toggle causes audio dropouts (GIL stall). Dual textures doubles update work. 48 px of fixed reserved space is acceptable.

**Rule:** If adding new right-side analyzer overlays, follow the same pattern — always create them, always use `wave_w = vw - METER_W`, toggle with `set_visible()`. Do not conditionally subtract analyzer widths from `wave_w`.

---

## Audio rules

### Never call `_rebuild_track_area()` from Settings toggles

Calling `_rebuild_track_area()` during playback causes audible pops/cracks. The sounddevice audio callback is a Python callable on a PortAudio real-time thread — it needs the GIL. Heavy DPG item deletion/creation on the main thread holds the GIL long enough to starve the callback.

**Rule:** Never call `_rebuild_track_area()` from a Settings-menu toggle. Use `set_visible()` / `dpg.configure_item(show=...)` on existing draw items instead. `_rebuild_track_area()` is only acceptable on viewport resize and track add/remove.

### Never call `_log()` from background threads

`_log()` writes to a `buffering=1` TextIOWrapper. Calling it from decode/wave/spectrum threads causes GIL + IO lock contention that can starve the background thread's `_post()` call — the message never lands in `_msg_queue`, so callbacks like `_on_decode_done` are never delivered.

**Rule:** Only call `_log()` from the main DPG thread. Background methods must post logging: `self._post(_log, "message")`.

**Rule:** Never add `_log()` calls to investigate a problem without removing them in the same edit. Every diagnostic log added has previously caused regressions.

### Always wrap background thread bodies in try/except

`ThreadPoolExecutor.submit()` and `threading.Thread` both silently swallow exceptions. A crash in `_wave_thread` or `_decode_thread` causes silent failure with no log entry.

```python
def _decode_thread(self, track_id, path):
    try:
        ...
        self._post(self._on_decode_done, track_id, data, audio_peak)
    except Exception:
        import traceback
        self._post(_log, f"_decode_thread EXCEPTION: {traceback.format_exc()}")
```

**Special case — `_spectrum_thread`:** if it crashes before calling `_post(self._apply_spectrum, ...)`, the `_spectrum_pending` flag stays `True` forever. No more spectrum updates fire (the rest of the app works fine). The fix is try/except that resets `_spectrum_pending = False` directly (GIL-atomic) and posts the traceback. A rebuild (`_rebuild_track_area`) also resets `_spectrum_pending`, which is why removing a track appears to fix it.

### Use `ThreadPoolExecutor(max_workers=1)` for decode

Running 4+ concurrent ffmpeg decode processes (each ~110 MB of raw PCM) causes I/O contention. Use a serial executor for decode; wave extraction can use plain `threading.Thread`.

---

## DPG rules and gotchas

### Per-frame tick — use the Python render loop, not `set_frame_callback`

`dpg.set_frame_callback(frame + 1, self._tick)` with self-rescheduling silently stops firing after ~9–36 seconds (DPG 2.2 bug). Use the Python render loop instead:

```python
def run(self):
    dpg.set_exit_callback(self._on_close)
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        if self._tick_ready:
            self._tick()
    dpg.destroy_context()
```

Set `self._tick_ready = True` in `_on_first_frame`. One-shot `set_frame_callback` for frame 1 is fine; it's repeated re-registration that breaks.

### `dpg.get_frame_count()` is unreliable from the render loop

Returns 0 or a fixed value when called from the manual Python render loop. Use a plain Python counter for frame-skip throttling (e.g. `self._spectrum_frame_skip`).

### Drain queue re-entry — never `_post()` for retries inside `_drain_queue` callbacks

`_drain_queue` loops `while not self._msg_queue.empty()`. If a callback inside that loop calls `_post()` to retry itself, it re-enqueues immediately and `_drain_queue` never exits — render loop freezes.

**Fix:** Use `dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: ...)` for retries. Reserve `_post` for background threads posting results to the main thread.

### Slot rebuild safety — queue flushing, generation counters, `does_item_exist` guards

When `_rebuild_track_area()` runs:
1. **Flush** (don't drain) the queue — discard stale callbacks without executing them
2. **Bump** `self._slot_generation += 1`
3. Deferred callbacks must check generation and bail if stale
4. Any `dpg.set_value` / `dpg.configure_item` must be guarded with `dpg.does_item_exist()`

Deleted DPG items free C++ pointers. Calling `set_value` on a freed tag causes a hard segfault with no Python traceback.

### Drawlist positioning — no `pos=(0,0)`, no nested drawlists

- `add_drawlist(pos=(0,0))` is treated as "no override" — use `pos=(0,1)` to force top-of-window placement.
- `add_drawlist(parent=another_drawlist)` throws error 1004 — only draw primitives are valid drawlist children.
- Meter items draw directly into `slot_dl` with coordinate offsets (`x_offset=wave_w`). No nested meter drawlist.

**Correct item order in `_build_slot_window`:**
1. Invisible click button at `pos=(0,0)` — bottom Z, receives clicks
2. `slot_dl` drawlist (waveform + meter items at `x_offset=wave_w`)
3. Spectrum drawlist at `pos=(0,1)` — must be added here while cursor is at y≈0
4. Name text at `pos=(8,4)`
5. X remove hint drawn into `slot_dl`

### Theme inheritance — bind to items, not containers

- Global `mvAll` theme overrides `mvButton`-scoped item themes for `Text` color. For colored dots, use `mvThemeCol_Button` (background), not `mvThemeCol_Text`.
- `bind_item_theme(container, theme)` cascades to ALL children — `mvButton` settings in the window theme override individually-bound button themes.
- `FramePadding` on a parent window kills small child buttons (negative content area → invisible). Apply `FramePadding` only to specific items via direct `bind_item_theme`.
- Slider height: bind a per-item theme with `mvSliderInt` component scope, not `mvAll`, to avoid double-apply from window theme cascade.

### Click zones — coordinate math, not buttons over drawlists

Buttons placed over a drawlist area compete with other click handlers and stop working silently. Instead, draw the visual using `dpg.draw_text()` into the drawlist and detect clicks via `dpg.get_mouse_pos(local=True)` in the slot's existing click callback.

### Control bar vertical alignment

Horizontal groups top-align all items. To center a text label vertically:
```python
with dpg.group():
    dpg.add_dummy(height=(btn_h - 13) // 2)  # 13 = Consolas font height
    dpg.add_text("0:00 / 0:00", tag=self._tag_time_lbl)
```
Drawlists that are `btn_h` tall with content at `cy = btn_h // 2` align naturally — do NOT add a dummy wrapper (double-counts the offset).

### Layout measurements — no hardcoded pixel distances

Never hardcode pixel offsets that depend on font size, padding, or window chrome. Use `dpg.get_item_rect_min()`, `dpg.get_item_rect_size()`, `dpg.get_item_rect_max()` to measure at runtime. Only use constants for things explicitly given a fixed `width=`/`height=`.

### Fonts — default font is ASCII only

DPG 2.2's default font (Proggy Clean) renders characters above ASCII as `?`. Use ASCII substitutes in menus: `"* "` not `"\u2713"`. Emoji (🔇 🔊) and geometric shapes (▶ ⏸) work only when Segoe UI Emoji / Segoe UI Symbol are loaded and bound to the specific item.

---

## Testing

### Automated screenshot test (`_test_launch.py`)

Test files: `testfiles/` (4 MP3s). Run with:
```
cd "d:/Dropbox/NorthBent/MixABTestGPU/mixabtestgpu" && timeout 35 python _test_launch.py
```

Key constraints:
- Don't use `set_frame_callback` to load files — the app already uses frame 1. Use a background thread posting to `_msg_queue` with `time.sleep(1.0)` delay.
- Run the render loop manually (don't call `app.run()`).
- `dpg.output_frame_buffer()` only works inside the render loop — not from callbacks or threads.
- `SCREENSHOT_FRAME = 450` (~7.5s at 60fps) is sufficient for all 4 tracks and waveforms to load.

---

## Collaboration preferences

- Propose visual or behavioral changes before implementing — don't alter existing behavior without approval.
- If a bug report is ambiguous, ask one clarifying question rather than assuming and diagnosing the wrong thing.
- When hitting a fork with non-obvious trade-offs, state 2–3 options briefly with pros/cons and ask the user to choose.
- After any rename, grep for the old name before finishing — missed references cause silent DPG failures (blank slots, no traceback).
- Present findings concisely; don't spiral through every option internally before asking.
