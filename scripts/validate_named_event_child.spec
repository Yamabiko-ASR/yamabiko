# -*- mode: python ; coding: utf-8 -*-
# Minimal onedir windowed build mirroring installer/yamabiko.spec's windowed mode.
a = Analysis(
    ['validate_named_event_child.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='probe_child',
    console=False,          # windowed (runw.exe) — matches yamabiko.exe
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name='probe_child',
)
