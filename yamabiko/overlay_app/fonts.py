"""Bundled UI fonts: BIZ UDPGothic (Regular, Bold), BIZ UDPMincho (Regular,
Bold) and Klee One (SemiBold).

SIL OFL 1.1, shipped unmodified under ``yamabiko/display/static/fonts`` with
their license texts. Qt registers them once per process, right after the
QApplication exists and before any widget builds a QFont from them; the
browser/OBS page loads the same files through ``@font-face`` in overlay.css.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QFontDatabase

from .. import paths

_log = logging.getLogger(__name__)

FONT_FILES = (
    "BIZUDPGothic-Regular.ttf",
    "BIZUDPGothic-Bold.ttf",
    "KleeOne-SemiBold.ttf",
    "BIZUDPMincho-Regular.ttf",
    "BIZUDPMincho-Bold.ttf",
)


def bundled_fonts_dir() -> Path:
    """Directory holding the bundled font files (dev checkout or frozen bundle)."""
    return paths.bundle_data_dir() / "display" / "static" / "fonts"


def register_bundled_fonts(fonts_dir: Path | None = None) -> list[str]:
    """Register every bundled font with Qt and return the sorted family names.

    A missing or unreadable file is logged and skipped, so a damaged bundle
    falls back to system fonts instead of failing to start.
    """
    root = fonts_dir if fonts_dir is not None else bundled_fonts_dir()
    families: set[str] = set()
    for name in FONT_FILES:
        path = root / name
        if not path.is_file():
            _log.warning("overlay.fonts.missing file=%s", path)
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            _log.warning("overlay.fonts.unreadable file=%s", path)
            continue
        families.update(QFontDatabase.applicationFontFamilies(font_id))
    registered = sorted(families)
    _log.info("overlay.fonts.registered families=%s", ",".join(registered))
    return registered
