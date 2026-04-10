"""
MixABTest GPU — entry point.

Usage:
    python main.py

ASIO support: same DLL drop-in approach as the original MixABTest.
Place libportaudio64bit-asio.dll next to main.py.
"""

import json as _json
import os
import struct as _struct
import sys

_HERE = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ── ASIO DLL (opt-in, same logic as original) ─────────────────────────────────
_PREFS_FILE = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), "MixABTestGPU", "prefs.json"
) if sys.platform == "win32" else os.path.join(
    os.path.expanduser("~"), ".config", "MixABTestGPU", "prefs.json"
)

_saved_api = ""
try:
    with open(_PREFS_FILE) as _f:
        _saved_api = _json.load(_f).get("audio_api_name", "")
except Exception:
    pass

if _saved_api == "ASIO":
    _bits = _struct.calcsize("P") * 8
    _candidates = (
        ["libportaudio64bit-asio.dll", "libportaudio32bit-asio.dll"]
        if _bits == 64
        else ["libportaudio32bit-asio.dll", "libportaudio64bit-asio.dll"]
    )
    try:
        os.add_dll_directory(_HERE)
    except AttributeError:
        pass
    os.environ["PATH"] = _HERE + os.pathsep + os.environ.get("PATH", "")
    for _dll_name in _candidates:
        _dll_path = os.path.join(_HERE, _dll_name)
        if os.path.exists(_dll_path):
            os.environ["PORTAUDIO_DLL_PATH"] = _dll_path
            break

# ── Redirect stderr to log file so exceptions are visible ─────────────────────
import traceback as _traceback
_log_path = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), "MixABTestGPU", "error.log"
)
try:
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)
    _log_f = open(_log_path, "w", encoding="utf-8", buffering=1)
    sys.stderr = _log_f
except Exception:
    pass

# ── Launch ────────────────────────────────────────────────────────────────────
from app import MixABTestGPU


def main():
    try:
        files = [a for a in sys.argv[1:] if os.path.isfile(a)]
        app = MixABTestGPU(initial_files=files)
        app.run()
    except Exception:
        _traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
