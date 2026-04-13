# build.spec — cross-platform PyInstaller spec for MixABTestGPU
# Produces a one-folder build on Windows (MixABTestGPU/) and a .app bundle on macOS.
import sys

block_cipher = None
is_win = sys.platform == "win32"

binaries = [
    ("ffmpeg.exe" if is_win else "ffmpeg",   "."),
    ("ffprobe.exe" if is_win else "ffprobe", "."),
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=[
        ("appicon.ico", "."),
        ("appicon.png", "."),
    ],
    hiddenimports=[
        "pyfftw",
        "pyfftw.interfaces",
        "pyfftw.interfaces.numpy_fft",
        "pyfftw.interfaces.cache",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MixABTestGPU",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="appicon.ico" if is_win else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="MixABTestGPU",
)

# macOS: wrap the collected folder in a .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="MixABTestGPU.app",
        icon="appicon.png",
        bundle_identifier="com.bentstamnes.mixabtestgpu",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleName": "MixABTestGPU",
        },
    )
