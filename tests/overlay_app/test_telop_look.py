"""telop_look: the style dict -> resolved overlay look (Qt-free)."""
from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
from yamabiko.overlay_app.telop_look import look_from_style


def test_watch_and_learn_typography_from_defaults():
    """Spec sizes at 100% scale from the default points: Watch is BIZ
    UDPGothic Bold 21 px over Bahnschrift SemiBold 31 px; Learn swaps the
    emphasis to Klee One SemiBold 33 px over Segoe UI Variable Text 21 px,
    with vermilion furigana on paper and amber in the echo.

    Seeded defects: drop the 4/3 point conversion (jp px 16), or keep the
    user's jp_font in Learn -> red."""
    watch = look_from_style(STYLE_DEFAULTS)
    assert (watch.jp.family, watch.jp.weight, watch.jp.px) == ("BIZ UDPGothic", 700, 21)
    assert (watch.en.family, watch.en.weight, watch.en.px) == ("Bahnschrift", 600, 31)
    assert watch.kana.px == 11 and watch.kana_plate[3] == 153  # 0.5 em, ink at 60%
    assert watch.bar == (0x0E, 0x10, 0x14, 237)

    learn = look_from_style({**STYLE_DEFAULTS, "mode": "learn"})
    assert (learn.jp.family, learn.jp.weight, learn.jp.px) == ("Klee One", 600, 33)
    assert (learn.en.family, learn.en.weight, learn.en.px) == ("Segoe UI Variable Text", 400, 21)
    assert learn.kana_plate == (0xC2, 0x3B, 0x2E, 255)
    assert learn.kana_echo == (0xF0, 0xBF, 0x5A, 255)
