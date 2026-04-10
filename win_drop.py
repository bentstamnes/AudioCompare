"""
win_drop.py — Windows Explorer file drag-and-drop for DPG via WM_DROPFILES.

Subclasses the SDL2 window using SetWindowLongPtrW with properly typed
64-bit argtypes/restype, then calls DragAcceptFiles.

The subclass proc receives WM_DROPFILES, extracts paths, and pushes them
to a queue.Queue.  The app's per-frame tick drains the queue on the main
thread so DPG is never called from inside the message hook.
"""

from __future__ import annotations
import sys
import ctypes
import ctypes.wintypes as wt

_ENABLED = sys.platform == "win32"

if _ENABLED:
    user32  = ctypes.windll.user32
    shell32 = ctypes.windll.shell32

    WM_DROPFILES = 0x0233
    GWL_WNDPROC  = -4

    # On 64-bit Windows all handles/pointers are 8 bytes.
    # Use c_uint64 for WPARAM/LPARAM to avoid sign/overflow issues.
    WNDPROC_TYPE = ctypes.WINFUNCTYPE(
        ctypes.c_int64,
        wt.HWND, wt.UINT, ctypes.c_uint64, ctypes.c_uint64,
    )

    # Properly typed SetWindowLongPtrW: returns INT_PTR (c_int64 on 64-bit)
    _SetWindowLongPtrW = user32.SetWindowLongPtrW
    _SetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_int64]
    _SetWindowLongPtrW.restype  = ctypes.c_int64

    # Properly typed CallWindowProcW
    _CallWindowProcW = user32.CallWindowProcW
    _CallWindowProcW.argtypes = [
        ctypes.c_int64, wt.HWND, wt.UINT, ctypes.c_uint64, ctypes.c_uint64,
    ]
    _CallWindowProcW.restype = ctypes.c_int64

    # Module-level state
    _orig_wndproc  = None   # int (the original proc pointer/atom)
    _hook_proc_obj = None   # keep WNDPROC_TYPE object alive
    _drop_queue    = None

    def _wndproc_hook(hwnd, msg, wparam, lparam):
        if msg == WM_DROPFILES:
            hDrop = ctypes.c_void_p(wparam)
            count = shell32.DragQueryFileW(hDrop, 0xFFFFFFFF, None, 0)
            paths = []
            buf   = ctypes.create_unicode_buffer(260)
            for i in range(count):
                shell32.DragQueryFileW(hDrop, i, buf, 260)
                paths.append(buf.value)
            shell32.DragFinish(ctypes.c_void_p(wparam))
            if _drop_queue is not None and paths:
                _drop_queue.put(paths)
            return 0
        return _CallWindowProcW(_orig_wndproc, hwnd, msg, wparam, lparam)

    def find_viewport_hwnd(title: str) -> int:
        return user32.FindWindowW(None, title)

    def setup_drop(hwnd: int, drop_queue) -> bool:
        global _orig_wndproc, _hook_proc_obj, _drop_queue
        if not hwnd:
            return False

        _drop_queue = drop_queue

        # Allow WM_DROPFILES through UIPI (needed if app is elevated)
        try:
            user32.ChangeWindowMessageFilterEx(hwnd, WM_DROPFILES, 1, None)
            user32.ChangeWindowMessageFilterEx(hwnd, 0x0049, 1, None)
        except Exception:
            pass

        shell32.DragAcceptFiles(hwnd, True)

        hook = WNDPROC_TYPE(_wndproc_hook)
        _hook_proc_obj = hook  # prevent GC

        # Pass the hook as a c_int64 (LONG_PTR) via ctypes.cast
        hook_ptr = ctypes.cast(hook, ctypes.c_void_p).value
        prev = _SetWindowLongPtrW(hwnd, GWL_WNDPROC, hook_ptr)
        _orig_wndproc = prev

        return True

    def teardown_drop(hwnd: int) -> None:
        global _orig_wndproc, _hook_proc_obj, _drop_queue
        if not hwnd or _orig_wndproc is None:
            return
        _SetWindowLongPtrW(hwnd, GWL_WNDPROC, _orig_wndproc)
        shell32.DragAcceptFiles(hwnd, False)
        _orig_wndproc  = None
        _hook_proc_obj = None
        _drop_queue    = None

    def register_drop_target(hwnd: int, callback, drop_queue) -> bool:
        return setup_drop(hwnd, drop_queue)

    def unregister_drop_target(hwnd: int) -> None:
        teardown_drop(hwnd)

else:
    def find_viewport_hwnd(title: str) -> int:
        return 0

    def setup_drop(hwnd: int, drop_queue) -> bool:
        return False

    def teardown_drop(hwnd: int) -> None:
        pass

    def register_drop_target(hwnd: int, callback, drop_queue) -> bool:
        return False

    def unregister_drop_target(hwnd: int) -> None:
        pass
