"""Real tier: the saved series cover (phase-5 L11) through real Qt image I/O.

Hard rule 3: save_cover_file (QSaveFile + QPixmap.save) and load_cover_file
(QPixmap.load) are new PySide6 file call sites. This test writes a real PNG
into a real folder from one palette session and reads it back in a second
one; nothing is mocked. No window is shown, so it runs in the real-deps
phase (the overlay test package's offscreen platform is enough for image
I/O).
"""
from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")

pytestmark = pytest.mark.real


def test_real_saved_cover_comes_back_in_a_new_session(qtbot, tmp_path, monkeypatch):
    """A restored series shows its poster at boot: the poster one palette
    saved for its active series is what a fresh palette's cover_for returns,
    ahead of the AniList color it also knows.

    Seeded defects: no saved-file step in cover_for -> the color comes back
    -> red; save in a format load_cover_file cannot read -> red.
    """
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QPixmap

    from yamabiko.gazetteer.anilist import SeriesHit
    from yamabiko.gazetteer.gazetteer import cover_path
    from yamabiko.overlay_app.series_palette import SeriesPalette

    # Real tier: the autouse app-data isolation exempts it, so point the
    # gazetteer folder at this test's tmp dir explicitly.
    monkeypatch.setenv("YAMABIKO_APPDATA", str(tmp_path / "Yamabiko"))

    source = QPixmap(10, 14)
    source.fill(QColor("#123456"))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    source.save(buf, "PNG")
    buf.close()

    first = SeriesPalette()
    qtbot.addWidget(first)
    first._on_cover_ready(21, bytes(data))
    first.set_current_series(21, "Frieren (2023)")
    assert cover_path(21).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    second = SeriesPalette()
    qtbot.addWidget(second)
    second._note_hit(SeriesHit(21, "Sousou no Frieren", "Frieren", None, 2023, "TV",
                               color="#e4a15d"))
    cover = second.cover_for(21)
    assert isinstance(cover, QPixmap), f"cover_for gave {cover!r}, not the saved poster"
    assert (cover.width(), cover.height()) == (10, 14)
    assert cover.toImage().pixelColor(5, 7) == QColor("#123456")
