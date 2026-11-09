"""Resolved look of the "Telop, echoing" subtitle overlay.

look_from_style() turns an overlay style dict (config ``overlay.style``) into
the fonts, pixel sizes, colors and paddings the Qt painter uses. Qt-free, so
it is testable without a QApplication. The browser/OBS page
(``display/static/overlay.css`` and ``overlay.js``) carries the same numbers
by hand; keep the two in sync.

``jp_size`` and ``en_size`` are points, their unit in the Qt overlay since
0.1 (1 pt = 4/3 px at the 96-dpi logical resolution Qt and browsers share).
The approved design's pixel sizes are at 1920 x 1080, 100 % scale; the
defaults 16 pt and 23 pt give its 21 px and 31 px.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .style_defaults import STYLE_CHOICES, STYLE_DEFAULTS, parse_hex_rgb

RGBA = tuple[int, int, int, int]

PT_TO_PX = 4 / 3

# Fixed colors of the approved design (not settings).
PAPER: RGBA = (0xF3, 0xEE, 0xE3, 255)            # strip behind the newest Japanese
INK: RGBA = (0x15, 0x17, 0x1B, 255)              # Japanese on the paper strip
BANNER_RED: RGBA = (0xD2, 0x45, 0x2F, 255)       # brand vermilion
BANNER_PAUSED: RGBA = (0x5B, 0x5F, 0x66, 255)
BANNER_TEXT: RGBA = (255, 255, 255, 255)
OUTLINE: RGBA = (0, 0, 0, 255)
LEARN_KANA: RGBA = (0xC2, 0x3B, 0x2E, 255)       # Learn furigana on paper
LEARN_ECHO_KANA: RGBA = (0xF0, 0xBF, 0x5A, 255)  # Learn furigana in the echo

# Banner (px at 100 % scale). Its width is also every line's left gutter.
BANNER_WIDTH = 38
BANNER_FONT = "BIZ UDPGothic"
BANNER_FONT_PX = 18
BANNER_FONT_WEIGHT = 700
BANNER_EN_FONT = "Bahnschrift"
BANNER_EN_FONT_PX = 15
BANNER_EN_FONT_WEIGHT = 600
BANNER_PAD = 12        # above and below the banner content
BANNER_GAP = 8         # between pause glyph, Japanese and English
PAUSE_BAR_W = 4
PAUSE_BAR_GAP = 3
PAUSE_GLYPH_H = 14

# Notice line (plan phase 4, Decisions 6 and 40): the お知らせ banner and a
# bar between the newest line and the echo while a problem lasts, then the
# recovery line in the same slot for NOTICE_RECOVERY_MS.
NOTICE_LABEL = "お知らせ"
NOTICE_BANNER: RGBA = (0xE0, 0xA0, 0x30, 255)      # amber: a problem
NOTICE_INK: RGBA = (0x1F, 0x24, 0x21, 255)         # the problem label; the recovery banner
NOTICE_RECOVERY_TEXT: RGBA = PAPER                  # the recovery label (#f3eee3)
NOTICE_TEXT: RGBA = (255, 255, 255, 255)           # the text on the bar
NOTICE_FONT_PX = 24
NOTICE_FONT_WEIGHT = 600
NOTICE_ROW_PX = 30
NOTICE_PAD_TOP = 9
NOTICE_PAD_SIDE = 20
NOTICE_PAD_BOTTOM = 11
NOTICE_BAR_H = NOTICE_PAD_TOP + NOTICE_ROW_PX + NOTICE_PAD_BOTTOM
NOTICE_GAP = 8          # under each notice, and between the echo and the top one
NOTICE_RECOVERY_MS = 5000

# Faces.
WATCH_JP_WEIGHT = 700
WATCH_KANA_WEIGHT = 400
WATCH_EN_WEIGHT = 600
LEARN_JP_FONT = "Klee One"
LEARN_JP_WEIGHT = 600
LEARN_EN_FONT = "Segoe UI Variable Text"
LEARN_EN_WEIGHT = 400

# Echo: size and opacity by age (0 = newest). Older ages reuse the last
# entry; the stack holds max_lines lines.
AGE_SCALE = (1.0, 0.84, 0.70)
AGE_OPACITY = (1.0, 0.80, 0.55)

# Motion (ms).
RESHAPE_MS = 350       # plates/banner reshape, echo glide and resize
WIPE_MS = 280          # the newest line wipes in from the left
WIPE_EN_DELAY_MS = 40
BANNER_TINT_MS = 300   # vermilion <-> paused grey

# Echo text: 2 px black outline (the text drawn at the eight 2 px offsets,
# as the mockup's CSS does) over a soft shadow (the black text box-blurred
# by a SHADOW_BOX_PX downscale/upscale, alpha SHADOW_ALPHA), approximating
# CSS `0 0 7px rgba(0,0,0,.85)`.
OUTLINE_PX = 2
SHADOW_BOX_PX = 6
SHADOW_ALPHA = 0.85
IMAGE_MARGIN = 9       # transparent margin around a rendered block

GENRE_BANNER: dict[str, tuple[str, str]] = {
    "variety": ("バラエティ", "Variety"),
    "news": ("ニュース", "News"),
    "anime": ("アニメ", "Anime"),
    "drama": ("ドラマ", "Drama"),
}
PAUSED_BANNER: tuple[str, str] = ("一時停止", "Paused")


@dataclass(frozen=True)
class FontSpec:
    family: str
    weight: int   # CSS weight, 100..900
    px: int


@dataclass(frozen=True)
class Pad:
    top: int
    side: int
    bottom: int


@dataclass(frozen=True)
class TelopLook:
    mode: str              # "watch" | "learn"
    alignment: str         # "left" | "center"
    echo: str              # "priority" | "both"
    banner_en: bool        # the banner also shows English, set sideways
    jp: FontSpec           # base glyphs (kanji and kana)
    kana: FontSpec         # furigana
    jp_line_height: float  # row height as a multiple of jp.px
    en: FontSpec
    en_line_height: float
    jp_pad: Pad            # newest line, on the paper strip
    en_pad: Pad            # newest line, on the dark bar
    echo_jp_pad: Pad
    echo_en_pad: Pad
    jp_plate: RGBA         # newest Japanese on paper
    kana_plate: RGBA
    en_plate: RGBA         # newest English on the bar
    jp_echo: RGBA          # echo fills
    kana_echo: RGBA
    en_echo: RGBA
    paper: RGBA
    bar: RGBA
    max_lines: int
    hold_ms: int
    fade_in_ms: int
    fade_out_ms: int
    notice_family: str     # notice text: the Watch English face in both modes

    def shows_jp(self, age: int) -> bool:
        """The newest line shows both languages; a "priority" echo keeps
        Japanese only in Learn."""
        return age == 0 or self.echo == "both" or self.mode == "learn"

    def shows_en(self, age: int) -> bool:
        return age == 0 or self.echo == "both" or self.mode == "watch"

    def text_key(self) -> tuple:
        """Everything that moves a glyph; text layouts are cached on it."""
        return (self.mode, self.alignment, self.echo, self.jp, self.kana,
                self.jp_line_height, self.en, self.en_line_height,
                self.jp_pad, self.en_pad, self.echo_jp_pad, self.echo_en_pad)

    def paint_key(self) -> tuple:
        """Text colors; rendered block images are cached on it too."""
        return (self.jp_plate, self.kana_plate, self.en_plate,
                self.jp_echo, self.kana_echo, self.en_echo)


def round_half_up(value: float) -> int:
    """JS Math.round semantics, so Qt and the browser page agree on px."""
    return int(math.floor(value + 0.5))


def age_scale(age: int) -> float:
    return AGE_SCALE[min(max(age, 0), len(AGE_SCALE) - 1)]


def age_opacity(age: int) -> float:
    return AGE_OPACITY[min(max(age, 0), len(AGE_OPACITY) - 1)]


def _num(style: dict, key: str) -> float:
    value = style.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(STYLE_DEFAULTS[key])
    try:
        if not math.isfinite(value):
            return float(STYLE_DEFAULTS[key])
    except OverflowError:
        return float(STYLE_DEFAULTS[key])
    return float(value)


def _family(style: dict, key: str) -> str:
    value = style.get(key)
    return value if isinstance(value, str) and value.strip() else STYLE_DEFAULTS[key]


def _choice(style: dict, key: str) -> str:
    value = style.get(key)
    return value if value in STYLE_CHOICES[key] else STYLE_DEFAULTS[key]


def _rgb(style: dict, key: str) -> tuple[int, int, int]:
    return parse_hex_rgb(style.get(key)) or parse_hex_rgb(STYLE_DEFAULTS[key])


def _px(points: float) -> int:
    return max(1, round_half_up(points * PT_TO_PX))


def _pad(top: float, side: float, bottom: float, k: float) -> Pad:
    return Pad(round_half_up(top * k), round_half_up(side * k), round_half_up(bottom * k))


def look_from_style(style: dict | None) -> TelopLook:
    """Resolve ``style`` (a clamped or raw overlay style dict) to the look.
    Garbage values fall back to STYLE_DEFAULTS rather than raising."""
    s = dict(style or {})
    mode = _choice(s, "mode")
    jp_pt = _num(s, "jp_size")
    en_pt = _num(s, "en_size")
    k = max(0.0, _num(s, "padding")) / 10
    tr, tg, tb = _rgb(s, "text_color")
    br, bgreen, bb = _rgb(s, "bg_color")
    alpha = int(min(255, max(0, round_half_up(_num(s, "bg_opacity")))))
    text: RGBA = (tr, tg, tb, 255)
    if mode == "learn":
        jp = FontSpec(LEARN_JP_FONT, LEARN_JP_WEIGHT,
                      max(1, round_half_up(en_pt * PT_TO_PX * 33 / 31)))
        kana = FontSpec(LEARN_JP_FONT, LEARN_JP_WEIGHT, max(1, round_half_up(jp.px * 0.46)))
        jp_line_height = 1.9
        en = FontSpec(LEARN_EN_FONT, LEARN_EN_WEIGHT, _px(jp_pt))
        jp_pad = _pad(4, 22, 2, k)
        en_pad = _pad(6, 22, 8, k)
        kana_plate, kana_echo = LEARN_KANA, LEARN_ECHO_KANA
        en_plate = en_echo = (tr, tg, tb, round_half_up(255 * 0.88))
    else:
        jp_family = _family(s, "jp_font")
        jp = FontSpec(jp_family, WATCH_JP_WEIGHT, _px(jp_pt))
        kana = FontSpec(jp_family, WATCH_KANA_WEIGHT, max(1, round_half_up(jp.px * 0.5)))
        jp_line_height = 1.75
        en = FontSpec(_family(s, "en_font"), WATCH_EN_WEIGHT, _px(en_pt))
        jp_pad = _pad(3, 18, 2, k)
        en_pad = _pad(9, 20, 11, k)
        kana_plate = (INK[0], INK[1], INK[2], round_half_up(255 * 0.6))
        kana_echo = (tr, tg, tb, round_half_up(255 * 0.85))
        en_plate = en_echo = text
    return TelopLook(
        mode=mode,
        alignment=_choice(s, "alignment"),
        echo=_choice(s, "echo"),
        banner_en=_choice(s, "banner") == "jp_en",
        jp=jp, kana=kana, jp_line_height=jp_line_height,
        en=en, en_line_height=1.2,
        jp_pad=jp_pad, en_pad=en_pad,
        echo_jp_pad=Pad(0, jp_pad.side, 0),
        echo_en_pad=Pad(round_half_up(1 * k), en_pad.side, round_half_up(3 * k)),
        jp_plate=INK, kana_plate=kana_plate, en_plate=en_plate,
        jp_echo=text, kana_echo=kana_echo, en_echo=en_echo,
        paper=PAPER, bar=(br, bgreen, bb, alpha),
        max_lines=max(1, int(_num(s, "max_lines"))),
        hold_ms=max(0, int(_num(s, "hold_ms"))),
        fade_in_ms=max(0, int(_num(s, "fade_in_ms"))),
        fade_out_ms=max(0, int(_num(s, "fade_out_ms"))),
        notice_family=_family(s, "en_font"),
    )
