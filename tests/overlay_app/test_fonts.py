"""Bundled UI fonts register with Qt under the family names the UI uses."""
import logging

import pytest
from PySide6.QtGui import QFont, QFontInfo

from yamabiko.overlay_app.fonts import FONT_FILES, register_bundled_fonts


@pytest.mark.real
def test_bundled_fonts_register_and_resolve_by_family(qapp):
    # Real QFontDatabase on the shipped files. Seeded defects: a wrong file
    # name or directory (nothing registers), or a missing Bold file (Bold
    # resolves to a synthesized or other face) -> red.
    families = register_bundled_fonts()
    assert {"BIZ UDPGothic", "BIZ UDPMincho", "Klee One"} <= set(families)

    for family in ("BIZ UDPGothic", "BIZ UDPMincho"):
        bold = QFont(family)
        bold.setWeight(QFont.Weight.Bold)
        bold_info = QFontInfo(bold)
        assert bold_info.family() == family
        assert int(bold_info.weight()) == int(QFont.Weight.Bold)
        assert bold_info.styleName() == "Bold"

    klee = QFont("Klee One")
    klee.setWeight(QFont.Weight.DemiBold)
    assert QFontInfo(klee).family() == "Klee One"


def test_register_bundled_fonts_skips_missing_files(qapp, tmp_path, caplog):
    # A bundle without the fonts must still start (system fonts take over).
    # Seeded defect: call addApplicationFont blindly or raise on a missing
    # file -> exception or no warning lines, red.
    caplog.set_level(logging.WARNING, logger="yamabiko.overlay_app.fonts")
    assert register_bundled_fonts(tmp_path) == []
    assert caplog.text.count("overlay.fonts.missing") == len(FONT_FILES)
