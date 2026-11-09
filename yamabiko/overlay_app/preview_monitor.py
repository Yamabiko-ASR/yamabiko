"""The Settings program monitor: a 1:1 crop of a screen's lower-left corner.

A night street photo ("Nakasu Yatai Stalls (19979437930)" by Yoshikazu
TAKADA, CC BY 2.0; assets/LICENSE-preview-night.txt) with the REAL telop
SubtitleStack over it at the overlay's own sizes: an echo line above the
newest line, plates and the banner, never animated. The stack sits where
the mockup puts it: 28 px from the left, 22 px from the bottom (Center
alignment: the monitor's center). A line's width cap is the overlay's real
cap (screen width x max_width_pct) bounded by the room the crop leaves, so a
long line wraps at the monitor's edge like the mockup.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRectF, QSize
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import paths
from .style_defaults import coerce_max_width_pct
from .telop_look import GENRE_BANNER
from .telop_stack import SubtitleStack

_log = logging.getLogger(__name__)

MONITOR_W = 564
MONITOR_H = 232
# A short Settings window shrinks the monitor down to this: the Overlay
# switcher beside it needs about 199 px for its four rows.
MONITOR_MIN_H = 200
_LEFT = 28
_BOTTOM = 22
_RADIUS = 4.0
_FALLBACK = "#14181f"
_PHOTO_FOCUS_Y = 0.72  # the mockup's background-position: center 72%

# The mockup's lines: the echo above, the newest line with its furigana.
SAMPLE_ECHO = (
    "えっ、本当にそれ食べるの？", "Wait, you're really going to eat that?",
    [{"text": "えっ、"}, {"text": "本当", "reading": "ほんとう"}, {"text": "にそれ"},
     {"text": "食", "reading": "た"}, {"text": "べるの？"}],
)
SAMPLE_NEWEST = (
    "いや、これがこの町の名物なんですよ。", "Come on, this is what the town is famous for.",
    [{"text": "いや、これがこの"}, {"text": "町", "reading": "まち"}, {"text": "の"},
     {"text": "名物", "reading": "めいぶつ"}, {"text": "なんですよ。"}],
)


def photo_path() -> Path:
    return paths.bundle_data_dir() / "overlay_app" / "assets" / "preview-night.jpg"


def photo_source(photo: QPixmap, width: float, height: float) -> QRectF:
    """The part of ``photo`` that covers a ``width`` x ``height`` box the way
    the mockups' ``background: center 72% / cover`` does (Settings' monitor,
    setup's tone sample)."""
    pw, ph = photo.width(), photo.height()
    scale = max(width / pw, height / ph)
    sw, sh = width / scale, height / scale
    return QRectF((pw - sw) / 2.0, (ph - sh) * _PHOTO_FOCUS_Y, sw, sh)


class PreviewMonitor(QWidget):
    """The 564 px program monitor (see the module docstring). ``rebuild``
    restyles the stack and reloads the two sample lines."""

    def __init__(self, style: dict, parent: QWidget | None = None, *,
                 banner: tuple[str, str] = GENRE_BANNER["variety"],
                 screen_width: Optional[int] = None) -> None:
        super().__init__(parent)
        # 564 x 232; a short Settings window shrinks it through fit_height.
        self.setFixedSize(MONITOR_W, MONITOR_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Subtitle preview")
        self._screen_width = screen_width
        self._style = dict(style)
        self._photo = QPixmap(str(photo_path()))
        if self._photo.isNull():
            _log.warning("settings.preview.photo_missing path=%s", photo_path())
        self.stack = SubtitleStack(self, style=style)
        self.stack.set_reduce_motion(True)
        self.stack.set_banner(banner[0], banner[1], False)
        self._sample_ids: list[int] = []

    def photo_loaded(self) -> bool:
        return not self._photo.isNull()

    def fit_height(self, height: int) -> bool:
        """Make the monitor ``height`` px tall, clamped to MONITOR_MIN_H ..
        MONITOR_H, its width keeping the 564 : 232 ratio. The Settings
        window calls this from its resize handling, so a short window takes
        height from the monitor before its panels scroll. True when the
        size changed."""
        height = max(MONITOR_MIN_H, min(MONITOR_H, int(height)))
        size = QSize(round(height * MONITOR_W / MONITOR_H), height)
        if size == self.minimumSize():
            return False
        self.setFixedSize(size)
        return True

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(MONITOR_W, MONITOR_H)

    def set_banner(self, jp: str, en: str) -> None:
        self.stack.set_banner(jp, en, False)

    def rebuild(self, style: dict) -> None:
        self._style = dict(style)
        # Drop the old samples BEFORE the restyle: each remove_line re-places
        # the lines left, which would re-measure them under the new style
        # only to throw them away.
        for sample_id in self._sample_ids:
            self.stack.remove_line(sample_id, fade=False)
        self._sample_ids = []
        self.stack.apply_style(style, retarget=False)
        self._place()
        samples = [SAMPLE_ECHO, SAMPLE_NEWEST][-min(2, self.stack.look().max_lines):]
        for _jp, en, ruby in samples:
            self._sample_ids.append(self.stack.add_line(ruby, en, hold_ms=None, instant=True))

    def _cap(self, centered: bool) -> int:
        screen_width = self._screen_width
        if screen_width is None:
            screen = QGuiApplication.primaryScreen()
            screen_width = screen.geometry().width() if screen is not None else 1920
        pct = coerce_max_width_pct(self._style.get("max_width_pct"))
        room = self.width() - (2 * _LEFT if centered else _LEFT)
        return max(1, min(int(screen_width * pct / 100), room))

    def _place(self) -> None:
        self.stack.setGeometry(self.rect())
        centered = self.stack.look().alignment == "center"
        anchor_x = self.width() / 2 if centered else _LEFT
        self.stack.set_anchor(anchor_x, self.height() - _BOTTOM, self._cap(centered),
                              self.width(), 0, self.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._place()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), _RADIUS, _RADIUS)
        p.setClipPath(clip)
        if self._photo.isNull():
            p.fillRect(self.rect(), QColor(_FALLBACK))
        else:
            p.drawPixmap(QRectF(self.rect()), self._photo,
                         photo_source(self._photo, self.width(), self.height()))
        p.end()
