"""Unit coverage for the native-renderer sentinel log in add_subtitle.

OverlayWindow.add_subtitle emits an ``overlay.subtitle.rendered`` INFO line
the moment the real overlay puts a subtitle on screen. That line is the
observable the frozen default-launch E2E
(tests/test_real_frozen_e2e.py::
test_frozen_default_launch_translates_wav_and_overlay_renders_subtitle)
greps for in main-overlay.log to prove the SHIPPED Qt renderer -- a separate
process from the pipeline -- actually rendered the translated subtitle, not
just that a WS frame left the pipeline.

The marker is CONTENT-FREE by default: it carries only en_len /
en_ascii_letters stats, never the EN (or JP) text. Default logs must carry
no transcript content -- default_config.yaml ships
``logging.transcript_log_enabled: false`` and bug_report.yml promises users
their diagnostics zip holds no dialogue. The raw EN rides along as
``en=<repr>`` only behind that same consent flag (plumbed in by
build_overlay), mirroring main.py's drop-line / latency.log gating.

This file pins both halves of that contract (content-free default, raw text
under consent) plus the exact literal tokens the frozen E2E's grep/split
depend on, so production drift that would break the expensive frozen tier
turns this fast tier red first.
"""
from __future__ import annotations

import logging

import pytest
from PySide6.QtWidgets import QApplication

from yamabiko.overlay_app.overlay_window import OverlayWindow
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS

_RENDER_LOGGER = "yamabiko.overlay_app.overlay_window"
_JP = "水が飲みたい"
_EN = "I want to drink water"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _render_markers(caplog):
    return [
        r for r in caplog.records
        if r.name == _RENDER_LOGGER
        and r.getMessage().startswith("overlay.subtitle.rendered")
    ]


def test_add_subtitle_render_marker_is_content_free_by_default(qapp, caplog):
    """Default construction (transcript consent OFF -- the shipped default)
    -> add_subtitle emits exactly one ``overlay.subtitle.rendered`` INFO
    record on the overlay_window logger carrying ONLY content-free stats:
    en_len (character count) and en_ascii_letters (Latin-letter count), and
    NEVER the EN or JP text. main-overlay.log ships in the collect-diagnostics
    zip, so an ungated marker would upload hours of translated dialogue.

    Seeded-defect drill: revert add_subtitle's sentinel to the ungated
    ``_log.info("overlay.subtitle.rendered en=%r", en)`` -> the EN text
    appears in the message (and the stat tokens vanish) -> red.
    """
    w = OverlayWindow(style=dict(STYLE_DEFAULTS))
    try:
        with caplog.at_level(logging.INFO, logger=_RENDER_LOGGER):
            w.add_subtitle(_JP, _EN, jp_ruby=[])

        markers = _render_markers(caplog)
        assert len(markers) == 1, (
            f"expected exactly one overlay.subtitle.rendered marker; got "
            f"{[r.getMessage() for r in caplog.records if r.name == _RENDER_LOGGER]!r}"
        )
        assert markers[0].levelno == logging.INFO, markers[0].levelno
        msg = markers[0].getMessage()
        # Consent is off: no transcript content, in either language.
        assert _EN not in msg, (
            f"render marker leaked the EN translation without transcript "
            f"consent: {msg!r}"
        )
        assert _JP not in msg, (
            f"render marker leaked the JP source text without transcript "
            f"consent: {msg!r}"
        )
        # The content-free stats must be the TRUE stats of the rendered EN --
        # they are what the frozen E2E asserts render proof against.
        assert int(msg.split("en_len=", 1)[1].split()[0]) == len(_EN), msg
        expected_latin = sum(1 for c in _EN if c.isascii() and c.isalpha())
        assert (
            int(msg.split("en_ascii_letters=", 1)[1].split()[0])
            == expected_latin
        ), msg
    finally:
        w.close()


def test_render_marker_pins_the_frozen_e2e_filter_and_split_tokens(qapp, caplog):
    """Pin the EXACT literal tokens the frozen default-launch E2E's grep and
    split depend on, by applying the E2E's own string operations to the
    record add_subtitle just produced. The E2E (tests/test_real_frozen_e2e.py::
    test_frozen_default_launch_translates_wav_and_overlay_renders_subtitle)
    runs the shipped default (consent off) against a fresh bundle and does:

      * filter:  ``"overlay.subtitle.rendered en_len=" in line``
      * split:   ``int(line.split("en_ascii_letters=", 1)[1].split()[0]) > 0``

    Any production rename of those tokens would break the expensive frozen
    tier silently until a release-gate bundle run; this test replays the
    identical operations at the unit tier so the drift goes red in seconds.
    Both consent states are covered: the consent-on line appends `` en=<repr>``
    AFTER the stats, so the E2E's filter/split must keep working on a host
    where the user opted in.

    Seeded-defect drill (run 2026-07-11: seeded red, reverted green): rename
    the ``en_ascii_letters=`` token in add_subtitle's marker to ``ascii=``
    -> the split target vanishes -> red. Renaming ``en_len=`` (filter token)
    or the ``overlay.subtitle.rendered`` prefix trips the same asserts.
    """
    for consent, en in ((False, _EN), (True, _EN + " opted-in")):
        caplog.clear()
        w = OverlayWindow(style=dict(STYLE_DEFAULTS), transcript_enabled=consent)
        try:
            with caplog.at_level(logging.INFO, logger=_RENDER_LOGGER):
                w.add_subtitle(_JP, en, jp_ruby=[])
            markers = _render_markers(caplog)
            assert len(markers) == 1, (
                f"consent={consent}: expected exactly one render marker; got "
                f"{[r.getMessage() for r in caplog.records if r.name == _RENDER_LOGGER]!r}"
            )
            line = markers[0].getMessage()

            # E2E filter token, verbatim.
            assert "overlay.subtitle.rendered en_len=" in line, (
                f"consent={consent}: the frozen E2E filters main-overlay.log "
                f"lines on the literal 'overlay.subtitle.rendered en_len='; "
                f"this marker would never match: {line!r}"
            )
            # E2E split + parse, verbatim (must not raise, must be a real int).
            latin = int(line.split("en_ascii_letters=", 1)[1].split()[0])
            assert latin == sum(
                1 for c in en if c.isascii() and c.isalpha()
            ), (
                f"consent={consent}: en_ascii_letters must be the Latin-letter "
                f"count of the rendered EN (the E2E's render proof is "
                f"latin > 0); got {latin} from {line!r}"
            )
            assert latin > 0, line  # the E2E's exact predicate on a real EN
        finally:
            w.close()


def test_add_subtitle_render_marker_carries_en_text_only_under_transcript_consent(
    qapp, caplog,
):
    """transcript_enabled=True (the logging.transcript_log_enabled opt-in,
    plumbed to the window by build_overlay) -> the same single marker
    additionally carries the raw EN text as ``en=<repr>`` for accuracy
    debugging -- the same opt-in shape as latency.log's raw jp/en fields.

    Seeded-defect drill: drop the consent branch in add_subtitle (always log
    the stats-only form) -> the ``en='...'`` token never appears -> red.
    """
    w = OverlayWindow(style=dict(STYLE_DEFAULTS), transcript_enabled=True)
    try:
        with caplog.at_level(logging.INFO, logger=_RENDER_LOGGER):
            w.add_subtitle(_JP, _EN, jp_ruby=[])

        markers = _render_markers(caplog)
        assert len(markers) == 1, (
            f"expected exactly one overlay.subtitle.rendered marker; got "
            f"{[r.getMessage() for r in caplog.records if r.name == _RENDER_LOGGER]!r}"
        )
        assert f"en={_EN!r}" in markers[0].getMessage(), (
            f"transcript consent is ON but the render marker did not carry "
            f"the raw EN text: {markers[0].getMessage()!r}"
        )
    finally:
        w.close()
