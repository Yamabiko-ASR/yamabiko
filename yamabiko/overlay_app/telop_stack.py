"""SubtitleStack: the "Telop, echoing" subtitle overlay, painted.

Two layers (spec Surface 1). The anchored layer (the vertical banner, the
paper strip behind the newest Japanese, the dark bar behind the newest
English) belongs to the stack, never to a line: it reshapes to fit the
newest line. The text layer has one entry per line: the newest sits on the
plates and wipes in from the left; older lines rise into the echo as
outlined text (age 1: 84 % size, 80 % opacity; age 2: 70 %, 55 %) and leave
once more than max_lines are on screen.

One widget paints everything, so the same class serves the full-screen
overlay window and the Settings preview strip. The owner supplies geometry
(set_anchor); text, style, banner, lifetimes (hold timers) and motion live
here. Motion is tweened on one clock and repainted through a 16 ms timer
that only runs while something moves; each tick repaints only the union of
the previous and current painted bounds, so a full-screen translucent
window does not re-upload itself every frame. Reduced motion lands every
change at once.

Notices (plan phase 4): while a problem lasts, a notice entry (the amber
お知らせ banner and a bar with the problem line) sits between the newest
line and the echo; when it clears, the recovery line (ink banner) takes
its slot for a few seconds. set_notices() holds them, one per problem kind,
translator above audio; they never age and do not count toward max_lines.
The owner (OverlayWindow) decides what shows; the stack places, fades and
paints them.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from PySide6.QtCore import QElapsedTimer, QEasingCurve, QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter
from PySide6.QtWidgets import QWidget

from .style_defaults import STYLE_DEFAULTS
from .telop_banner import BannerContent, banner_content, paint_banner
from .telop_look import (
    BANNER_PAUSED, BANNER_RED, BANNER_TINT_MS, BANNER_WIDTH, GENRE_BANNER,
    IMAGE_MARGIN, NOTICE_BANNER, NOTICE_BAR_H, NOTICE_FONT_PX, NOTICE_FONT_WEIGHT,
    NOTICE_GAP, NOTICE_INK, NOTICE_LABEL, NOTICE_PAD_SIDE, NOTICE_PAD_TOP,
    NOTICE_RECOVERY_TEXT, NOTICE_ROW_PX, NOTICE_TEXT, RESHAPE_MS, WIPE_EN_DELAY_MS,
    WIPE_MS, FontSpec, TelopLook, age_opacity, look_from_style,
)
from .telop_text import MIN_TEXT_WIDTH, Block, LineLayout, layout_line, make_font, render_block


def _bezier(x1: float, y1: float, x2: float, y2: float) -> QEasingCurve:
    curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
    curve.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return curve


GLIDE = _bezier(0.2, 0.7, 0.2, 1.0)   # reshape and echo glide (ease-out)
WIPE = _bezier(0.3, 0.6, 0.2, 1.0)    # entrance wipe
EASE = _bezier(0.25, 0.1, 0.25, 1.0)  # CSS "ease": opacity, tint, crossfade


def _lerp(a, b, t: float):
    if isinstance(a, QRectF):
        return QRectF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t,
                      a.width() + (b.width() - a.width()) * t,
                      a.height() + (b.height() - a.height()) * t)
    return a + (b - a) * t


class Anim:
    """One tweened value (a float or a QRectF) on the stack's clock (ms)."""

    __slots__ = ("start", "end", "t0", "ms", "curve")

    def __init__(self, value) -> None:
        self.start = value
        self.end = value
        self.t0 = 0.0
        self.ms = 0
        self.curve = GLIDE

    def get(self, now: float):
        if self.ms <= 0 or now >= self.t0 + self.ms:
            return self.end
        t = (now - self.t0) / self.ms
        if t <= 0.0:
            return self.start
        return _lerp(self.start, self.end, self.curve.valueForProgress(t))

    def to(self, value, now: float, ms: int, curve: QEasingCurve = GLIDE,
           delay: float = 0.0) -> None:
        """Move from the current value to ``value``; ``ms <= 0`` snaps."""
        self.start = self.get(now)
        self.end = value
        self.t0 = now + delay
        self.ms = ms
        self.curve = curve

    def snap(self, value) -> None:
        self.start = value
        self.end = value
        self.ms = 0

    def active(self, now: float) -> bool:
        return self.ms > 0 and now < self.t0 + self.ms


@dataclass(eq=False)
class _BlockView:
    kind: str
    rect: Anim                    # block box, widget coordinates
    wipe: Anim                    # 0..1 of the box revealed from the left
    xfade: Anim                   # 0..1 from ``prev`` to ``image``
    layout: Optional[Block] = None
    image: Optional[QImage] = None
    prev: Optional[QImage] = None


@dataclass(eq=False)
class _Line:
    id: int
    segments: list
    en: str
    persistent: bool
    opacity: Anim
    hold: Optional[QTimer] = None
    layout: Optional[LineLayout] = None
    layout_key: Optional[tuple] = None
    blocks: dict = field(default_factory=dict)   # kind -> _BlockView
    rect: Optional[QRectF] = None                # target line box
    placed: bool = False


@dataclass(eq=False)
class _Notice:
    kind: str                          # problem kind: "translator" | "audio"
    text: str                          # the whole line; the bar shows it elided
    recovery: bool                     # the recovery line (ink banner)
    opacity: Anim
    rect: Optional[Anim] = None        # entry box (banner + bar); None until placed
    shown: str = ""                    # the text as drawn
    bar_w: int = 0
    key: Optional[tuple] = None        # (family, text width cap) it was measured for
    font: Optional[QFont] = None
    image: Optional[QImage] = None     # the text on a transparent bar-sized image


@dataclass(frozen=True)
class LineBox:
    id: int
    age: int
    rect: QRect
    blocks: dict                  # kind -> QRect, target block boxes


@dataclass(frozen=True)
class NoticeBox:
    kind: str
    text: str                     # as drawn (elided at the line cap)
    recovery: bool
    rect: QRect                   # banner + bar, target
    banner: QRect
    bar: QRect


@dataclass(frozen=True)
class StackFrame:
    """Where everything settles (the target layout, not mid-animation)."""
    lines: tuple                  # tuple[LineBox, ...], oldest first
    jp_plate: Optional[QRect]
    en_plate: Optional[QRect]
    banner: Optional[QRect]
    banner_text: tuple            # (jp, en)
    paused: bool
    notices: tuple = ()           # tuple[NoticeBox, ...], top first


def place_lines(sizes: list, anchor_x: float, bottom_y: float, alignment: str,
                bounds_w: float, top: float, avail_bottom: float,
                reach: float = 0.0, gaps: Optional[list] = None) -> list:
    """Entry boxes ``(x, y, w, h)`` for ``sizes`` (top first, the one on the
    anchor last).

    The last entry's bottom sits on ``bottom_y`` and the others stack
    upward; ``gaps[i]`` is the space between entry ``i`` and the one under
    it (none by default: subtitle lines stack with no gap). Left: x is the
    anchor; Center: each entry is centered on it. x is clamped on-screen.
    When the stack, or the banner rising ``reach`` from the newest line's
    bottom, is taller than the space above the anchor it is pushed down
    until its top is on-screen (below a top-docked taskbar); one taller
    than the whole available area keeps the newest line on the anchor
    instead.
    """
    gaps = list(gaps) if gaps is not None else [0.0] * len(sizes)
    stack_h = max(sum(h for _, h in sizes) + sum(gaps[:-1]), reach)
    y = bottom_y
    if top + stack_h <= avail_bottom:
        y = max(bottom_y, top + stack_h)
    placed = []
    for i in range(len(sizes) - 1, -1, -1):
        w, h = sizes[i]
        x = anchor_x - w / 2 if alignment == "center" else anchor_x
        x = float(math.floor(max(0.0, min(x, bounds_w - w)) + 0.5))
        placed.append((x, y - h, float(w), float(h)))
        y -= h + (gaps[i - 1] if i > 0 else 0.0)
    placed.reverse()
    return placed


def _mix(a: tuple, b: tuple, t: float) -> QColor:
    return QColor(*(round(x + (y - x) * t) for x, y in zip(a, b)))


def _notice_text_image(notice: _Notice, dpr: float) -> QImage:
    """The notice text in white on a transparent image the bar's size. On a
    transparent layer the text is grayscale-antialiased like the plate text;
    drawn straight onto the opaque bar it would take ClearType's fringes."""
    image = QImage(max(1, math.ceil(notice.bar_w * dpr)), max(1, math.ceil(NOTICE_BAR_H * dpr)),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(notice.font)
    painter.setPen(QColor(*NOTICE_TEXT))
    fm = QFontMetricsF(notice.font)
    # The em box centered in the 30 px row, as the browser sets the line.
    base = NOTICE_PAD_TOP + (NOTICE_ROW_PX - (fm.ascent() + fm.descent())) / 2 + fm.ascent()
    painter.drawText(QPointF(NOTICE_PAD_SIDE, base), notice.shown)
    painter.end()
    return image


class SubtitleStack(QWidget):
    """Paints the telop subtitle stack: plates, text and banner."""

    def __init__(self, parent: QWidget | None = None, *, style: dict | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        if clock is None:
            elapsed = QElapsedTimer()
            elapsed.start()

            def clock() -> float:
                return float(elapsed.elapsed())
        self._clock = clock
        self._look: TelopLook = look_from_style(style if style is not None else STYLE_DEFAULTS)
        self._text_key = self._look.text_key()
        self._paint_key = self._look.paint_key()
        self._reduce_motion = False
        self._lines: list[_Line] = []      # oldest first; the last is the newest
        self._ghosts: list[_Line] = []     # leaving lines, fading out
        self._ids = itertools.count(1)
        self._anchor = (0.0, 0.0)
        self._cap = 1
        self._bounds = (0, 0, 0)           # (width, top, available bottom)
        self._banner_text = GENRE_BANNER["variety"]
        self._paused = False
        # The state's banner content, and the content painted: they differ
        # only while the banner fades out (it keeps what it showed).
        self._banner: BannerContent = banner_content(*self._banner_text, False,
                                                     self._look.banner_en)
        self._banner_paint = self._banner
        # Notices (set_notices), top first, and those fading out after a
        # change; their banner always reads お知らせ.
        self._notices: list[_Notice] = []
        self._notice_ghosts: list[_Notice] = []
        self._notice_banner: BannerContent = banner_content(NOTICE_LABEL, "", False, False)
        self._jp_plate = Anim(QRectF())
        self._en_plate = Anim(QRectF())
        self._jp_plate_on = False
        self._en_plate_on = False
        self._plates_opacity = Anim(0.0)
        self._banner_rect = Anim(QRectF())
        self._banner_opacity = Anim(0.0)
        self._banner_tint = Anim(0.0)      # 0 vermilion .. 1 paused grey
        self._frame = StackFrame((), None, None, None, self._banner_text, False)
        self._dpr = 0.0
        self._painted = QRect()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    # -- public ---------------------------------------------------------

    def look(self) -> TelopLook:
        return self._look

    def frame(self) -> StackFrame:
        return self._frame

    def set_reduce_motion(self, on: bool) -> None:
        self._reduce_motion = bool(on)
        if self._reduce_motion:
            self._retarget(motion=False)   # lands anything in flight now

    def apply_style(self, style: dict, *, retarget: bool = True) -> None:
        """Adopt a new style. With ``retarget`` False the caller re-places
        the stack itself (OverlayWindow moves the anchor in the same step)."""
        look = look_from_style(style)
        self._look = look
        self._banner = banner_content(*self._banner_text, self._paused, look.banner_en)
        if look.paint_key() != self._paint_key:
            self._paint_key = look.paint_key()
            for line in self._lines + self._ghosts:
                for view in line.blocks.values():
                    view.image = None
                    view.prev = None
        text_key = look.text_key()
        if text_key != self._text_key:
            # The lines on screen move to a new layout (a Watch/Learn switch,
            # say): each gets a full hold from now so it can be read there.
            for line in self._lines:
                if line.hold is not None:
                    line.hold.start(look.hold_ms)
        self._text_key = text_key
        self._trim()
        if retarget:
            self._retarget(motion=True)

    def set_anchor(self, anchor_x: float, bottom_y: float, cap_px: int, bounds_w: int,
                   top: int, avail_bottom: int, *, animate: bool = False) -> None:
        """Place the stack. ``anchor_x`` is the stack's left edge (Left) or
        center (Center); the newest line's bottom sits on ``bottom_y``;
        ``cap_px`` caps a line's width, banner gutter included."""
        self._anchor = (float(anchor_x), float(bottom_y))
        self._cap = max(1, int(cap_px))
        self._bounds = (int(bounds_w), int(top), int(avail_bottom))
        self._retarget(motion=animate)

    def set_banner(self, jp: str, en: str, paused: bool) -> None:
        self._paused = bool(paused)
        self._banner_text = (jp, en)
        self._banner = banner_content(jp, en, self._paused, self._look.banner_en)
        self._retarget(motion=True)

    def add_line(self, segments: list, en: str, *, hold_ms: int | None = None,
                 instant: bool = False) -> int:
        """Add a line as the newest and return its id. ``hold_ms`` None keeps
        it until remove_line (the placement and preview samples); otherwise
        it fades out after ``hold_ms``. ``instant`` lands this line at once,
        with no wipe, and the plates and banner with it; older and fading
        lines still move."""
        line = _Line(next(self._ids), list(segments), str(en), hold_ms is None, Anim(1.0))
        self._lines.append(line)
        if hold_ms is not None:
            # Child of the stack, stopped and deleted in _release: an evicted
            # line's timer can never fire later.
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda line_id=line.id: self.remove_line(line_id))
            timer.start(max(0, int(hold_ms)))
            line.hold = timer
        self._trim()
        self._retarget(motion=True, entering=line, instant=instant)
        if self._ghosts and line.rect is not None:
            # Every leaving line (pushed past max_lines, or fading after its
            # hold) rises with the stack by the new line's height.
            now = self._clock()
            rise = line.rect.height()
            for ghost in self._ghosts:
                for view in ghost.blocks.values():
                    view.rect.to(view.rect.end.translated(0, -rise), now, RESHAPE_MS, GLIDE)
        return line.id

    def remove_line(self, line_id: int, *, fade: bool = True) -> None:
        """Remove a line. ``fade`` False takes it away at once (a sample
        hidden), and the plates and banner too when no line is left (paused,
        the banner stays); the rest of the stack still moves."""
        line = next((ln for ln in self._lines if ln.id == line_id), None)
        if line is None:
            return
        self._release(line)
        self._lines.remove(line)
        if fade:
            self._ghost(line, self._look.fade_out_ms)
        self._retarget(motion=True, instant=not fade)

    def set_notices(self, notices: list) -> None:
        """Show ``notices``: ``(kind, text, recovery)`` tuples, top first, at
        most one per kind (the problem lines and recovery lines). An entry
        that stays the same stays put; one that changes or goes fades out
        where it is while its replacement fades in (plan Decision 42)."""
        old = {(n.kind, n.text, n.recovery): n for n in self._notices}
        kept: list[_Notice] = []
        for kind, text, recovery in notices:
            key = (str(kind), str(text), bool(recovery))
            kept.append(old.pop(key, None) or _Notice(*key, Anim(0.0)))
        for notice in old.values():
            self._ghost_notice(notice)
        self._notices = kept
        self._retarget(motion=True)

    # -- lifetimes ------------------------------------------------------

    def _trim(self) -> None:
        """Evict the oldest non-persistent lines past max_lines; they leave
        as ghosts that fade over RESHAPE_MS."""
        real = [ln for ln in self._lines if not ln.persistent]
        while len(real) > self._look.max_lines:
            old = real.pop(0)
            self._release(old)
            self._lines.remove(old)
            self._ghost(old, RESHAPE_MS)

    @staticmethod
    def _release(line: _Line) -> None:
        if line.hold is not None:
            line.hold.stop()
            line.hold.deleteLater()
            line.hold = None

    def _ghost(self, line: _Line, ms: int) -> None:
        if self._reduce_motion or ms <= 0 or not line.blocks:
            return
        line.opacity.to(0.0, self._clock(), ms, EASE)
        self._ghosts.append(line)

    def _ghost_notice(self, notice: _Notice) -> None:
        """A notice that changed or went fades out where it is."""
        ms = self._look.fade_out_ms
        if self._reduce_motion or ms <= 0 or notice.rect is None:
            return
        notice.opacity.to(0.0, self._clock(), ms, EASE)
        self._notice_ghosts.append(notice)

    # -- layout ---------------------------------------------------------

    def _retarget(self, *, motion: bool, entering: Optional[_Line] = None,
                  instant: bool = False) -> None:
        """Re-place everything. Without ``motion`` (or with reduced motion)
        nothing stays in flight. ``entering`` is a newly added line: it
        wipes in, or with ``instant`` lands at once. ``instant`` without
        ``entering`` is a line removed at once."""
        now = self._clock()
        animate = motion and not self._reduce_motion
        if not animate:
            self._ghosts.clear()
            self._notice_ghosts.clear()
        look = self._look
        max_box = max(1, self._cap - BANNER_WIDTH)
        count = len(self._lines)
        shown: list[_Line] = []
        for index, line in enumerate(self._lines):
            age = count - 1 - index
            key = (age, self._text_key, max_box)
            if line.layout_key != key:
                line.layout = layout_line(line.segments, line.en, look, age, max_box)
                line.layout_key = key
            if line.layout.blocks:
                shown.append(line)
            else:
                line.blocks.clear()
        self._measure_notices(max_box)
        anchor_x, bottom_y = self._anchor
        bounds_w, top, avail_bottom = self._bounds
        newest = shown[-1] if shown else None
        natural = self._banner.natural_height
        # One column, top first: the echo lines, the notices, then what sits
        # on the anchor (the newest line, or the paused banner when paused
        # with no lines). gaps[i] is the space under entry i (Decision 40).
        sizes = [(BANNER_WIDTH + ln.layout.width, ln.layout.height) for ln in shown[:-1]]
        gaps = [0.0] * len(sizes)
        if self._notices and gaps:
            gaps[-1] = float(NOTICE_GAP)   # the echo stays clear of the top notice
        notice_h = max(NOTICE_BAR_H, self._notice_banner.natural_height)
        for notice in self._notices:
            sizes.append((BANNER_WIDTH + notice.bar_w, notice_h))
            gaps.append(float(NOTICE_GAP))
        base = None
        rise = 0.0
        if newest is not None:
            base = (BANNER_WIDTH + newest.layout.width, newest.layout.height)
            # The bottom notice clears the newest line's banner where a long
            # name grows it past the line.
            rise = max(0.0, natural - newest.layout.height)
        elif self._paused:
            base = (BANNER_WIDTH, natural)
        if base is not None:
            if self._notices:
                gaps[-1] += rise
            sizes.append(base)
            gaps.append(0.0)
        # The banner grows upward from the newest line: the clamp keeps it
        # on screen too.
        placed = place_lines(sizes, anchor_x, bottom_y, look.alignment, bounds_w, top,
                             avail_bottom, natural if newest is not None else 0.0, gaps)
        echo_count = max(0, len(shown) - 1)
        line_boxes = placed[:echo_count] + (placed[-1:] if newest is not None else [])
        notice_boxes = placed[echo_count:echo_count + len(self._notices)]
        paused_rect = QRectF(*placed[-1]) if newest is None and base is not None else None
        boxes = []
        for line, (x, y, w, h) in zip(shown, line_boxes):
            line.rect = QRectF(x, y, w, h)
            arriving = line is entering
            self._retarget_blocks(line, x + BANNER_WIDTH, y, now,
                                  animate and not (arriving and instant), arriving)
            target = age_opacity(line.layout.age)
            if arriving or not line.placed or not animate:
                line.opacity.snap(target)
            else:
                line.opacity.to(target, now, RESHAPE_MS, EASE)
            line.placed = True
            boxes.append(LineBox(line.id, line.layout.age, line.rect.toRect(),
                                 {k: v.rect.end.toRect() for k, v in line.blocks.items()}))
        self._retarget_notices(notice_boxes, now, animate)
        # The plates and banner land and leave with an instant line (the
        # sample) when it is the newest line, or when removing it leaves
        # none; under newer lines they follow those as usual.
        land = instant and (newest is None if entering is None else newest is entering)
        self._retarget_plates(newest, now, animate and not land)
        self._retarget_banner(newest, now, animate and not land, paused_rect)
        if land:
            for anim in self._anchored_anims():
                anim.snap(anim.end)
        if not animate:
            # Without motion nothing stays in flight: fades, tints and
            # reshapes started earlier land now too.
            for anim in self._anims():
                anim.snap(anim.end)
        plates = self._plates_opacity.end > 0.0   # settled, like the banner
        self._frame = StackFrame(
            lines=tuple(boxes),
            jp_plate=self._jp_plate.end.toRect() if plates and self._jp_plate_on else None,
            en_plate=self._en_plate.end.toRect() if plates and self._en_plate_on else None,
            banner=self._banner_rect.end.toRect() if self._banner_opacity.end > 0.0 else None,
            banner_text=self._banner_text,
            paused=self._paused,
            notices=tuple(self._notice_box(notice) for notice in self._notices),
        )
        self._kick()

    def _measure_notices(self, max_box: int) -> None:
        """Size each notice's bar for the line cap: it hugs its text, which is
        elided (never wrapped) to fit."""
        family = self._look.notice_family
        cap = max(MIN_TEXT_WIDTH, max_box - 2 * NOTICE_PAD_SIDE)
        key = (family, cap)
        for notice in self._notices:
            if notice.key == key:
                continue
            font = make_font(FontSpec(family, NOTICE_FONT_WEIGHT, NOTICE_FONT_PX))
            fm = QFontMetricsF(font)
            notice.shown = fm.elidedText(notice.text, Qt.TextElideMode.ElideRight, cap)
            notice.bar_w = math.ceil(fm.horizontalAdvance(notice.shown) - 1e-6) + 2 * NOTICE_PAD_SIDE
            notice.font = font
            notice.key = key
            notice.image = None

    def _retarget_notices(self, boxes: list, now: float, animate: bool) -> None:
        for notice, (x, y, w, h) in zip(self._notices, boxes):
            target = QRectF(x, y, w, h)
            if notice.rect is None:
                # Appearing: placed at once; only the fade-in runs.
                notice.rect = Anim(target)
                notice.opacity.snap(0.0)
                notice.opacity.to(1.0, now, self._look.fade_in_ms if animate else 0, EASE)
            else:
                notice.rect.to(target, now, RESHAPE_MS if animate else 0, GLIDE)

    @staticmethod
    def _notice_box(notice: _Notice) -> NoticeBox:
        rect = notice.rect.end
        return NoticeBox(
            notice.kind, notice.shown, notice.recovery, rect.toRect(),
            QRectF(rect.x(), rect.y(), BANNER_WIDTH, rect.height()).toRect(),
            QRectF(rect.x() + BANNER_WIDTH, rect.y() + rect.height() - NOTICE_BAR_H,
                   notice.bar_w, NOTICE_BAR_H).toRect(),
        )

    def _retarget_blocks(self, line: _Line, x0: float, y0: float, now: float,
                         animate: bool, arriving: bool) -> None:
        targets: dict = {}
        y = y0
        for block in line.layout.blocks:
            targets[block.kind] = (block, QRectF(x0, y, block.width, block.height))
            y += block.height
        for kind in [k for k in line.blocks if k not in targets]:
            del line.blocks[kind]   # e.g. Japanese leaving a Watch echo line
        for kind, (block, rect) in targets.items():
            view = line.blocks.get(kind)
            if view is None:
                view = _BlockView(kind, Anim(rect), Anim(1.0), Anim(1.0))
                line.blocks[kind] = view
                if arriving and animate:
                    view.wipe.snap(0.0)
                    view.wipe.to(1.0, now, WIPE_MS, WIPE,
                                 WIPE_EN_DELAY_MS if kind == "en" else 0.0)
            else:
                view.rect.to(rect, now, RESHAPE_MS if animate else 0, GLIDE)
            if view.layout is not block:
                if animate and view.image is not None:
                    view.prev = view.image
                    view.xfade.snap(0.0)
                    view.xfade.to(1.0, now, RESHAPE_MS, EASE)
                view.image = None          # rendered lazily at paint
                view.layout = block
            if not animate:
                view.wipe.snap(1.0)
                view.xfade.snap(1.0)
                view.prev = None

    @staticmethod
    def _aim(anim: Anim, was_on: bool, target: QRectF, appearing: bool, now: float,
             animate: bool) -> None:
        if appearing or not was_on:
            anim.snap(target)
        else:
            anim.to(target, now, RESHAPE_MS if animate else 0, GLIDE)

    def _retarget_plates(self, newest: Optional[_Line], now: float, animate: bool) -> None:
        if newest is None:
            self._plates_opacity.to(0.0, now, self._look.fade_out_ms if animate else 0, EASE)
            return
        # Appearing from hidden; plates still fading out reshape from where
        # they are instead.
        appearing = self._plates_opacity.get(now) <= 0.0
        jp = newest.blocks.get("jp")
        en = newest.blocks.get("en")
        if jp is not None:
            self._aim(self._jp_plate, self._jp_plate_on, jp.rect.end, appearing, now, animate)
        if en is not None:
            self._aim(self._en_plate, self._en_plate_on, en.rect.end, appearing, now, animate)
        self._jp_plate_on = jp is not None
        self._en_plate_on = en is not None
        if self._plates_opacity.end != 1.0:
            self._plates_opacity.to(1.0, now, self._look.fade_in_ms if animate else 0, EASE)

    def _retarget_banner(self, newest: Optional[_Line], now: float, animate: bool,
                         paused_rect: Optional[QRectF] = None) -> None:
        if newest is None and not self._paused:
            # Fading out keeps the content and tint it showed: an unpause
            # with no lines fades the grey 一時停止, never the genre.
            self._banner_opacity.to(0.0, now, self._look.fade_out_ms if animate else 0, EASE)
            return
        self._banner_paint = self._banner
        tint = 1.0 if self._paused else 0.0
        natural = self._banner.natural_height
        if newest is not None:
            line = newest.rect
            height = max(line.height(), natural)
            target = QRectF(line.x(), line.y() + line.height() - height, BANNER_WIDTH, height)
        else:
            # Paused with no lines: _retarget placed it on the anchor like a
            # line of the banner's size (on screen at a top anchor too), under
            # any notices.
            target = paused_rect
        if self._banner_opacity.get(now) <= 0.0:
            # Appearing from hidden: it shows the new state from the start.
            self._banner_rect.snap(target)
            self._banner_tint.snap(tint)
        else:
            # Shown, or still fading out: it reshapes and retints from what
            # it shows.
            self._banner_rect.to(target, now, RESHAPE_MS if animate else 0, GLIDE)
            if self._banner_tint.end != tint:
                self._banner_tint.to(tint, now, BANNER_TINT_MS if animate else 0, EASE)
        if self._banner_opacity.end != 1.0:
            self._banner_opacity.to(1.0, now, self._look.fade_in_ms if animate else 0, EASE)

    # -- painting -------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        now = self._clock()
        dpr = self.devicePixelRatioF()
        if dpr != self._dpr:
            self._dpr = dpr
            for line in self._lines + self._ghosts:
                for view in line.blocks.values():
                    view.image = None
                    view.prev = None
            for notice in self._notices + self._notice_ghosts:
                notice.image = None
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        look = self._look
        plates = self._plates_opacity.get(now)
        if plates > 0.0:
            painter.setOpacity(plates)
            if self._jp_plate_on:
                painter.fillRect(self._jp_plate.get(now), QColor(*look.paper))
            if self._en_plate_on:
                painter.fillRect(self._en_plate.get(now), QColor(*look.bar))
        for line in self._ghosts + self._lines:
            self._paint_line(painter, line, now)
        banner = self._banner_opacity.get(now)
        if banner > 0.0:
            painter.setOpacity(banner)
            paint_banner(painter, self._banner_rect.get(now), self._banner_paint,
                         _mix(BANNER_RED, BANNER_PAUSED, self._banner_tint.get(now)))
        for notice in self._notice_ghosts + self._notices:
            self._paint_notice(painter, notice, now)
        painter.end()

    def _paint_line(self, painter: QPainter, line: _Line, now: float) -> None:
        opacity = line.opacity.get(now)
        if opacity <= 0.0:
            return
        for view in line.blocks.values():
            block = view.layout
            if block is None or block.width <= 0 or block.height <= 0:
                continue
            if view.image is None:
                view.image = render_block(block, self._look, self._dpr or 1.0)
            rect = view.rect.get(now)
            mx = IMAGE_MARGIN * rect.width() / block.width
            my = IMAGE_MARGIN * rect.height() / block.height
            target = rect.adjusted(-mx, -my, mx, my)
            painter.save()
            wipe = view.wipe.get(now)
            if wipe < 1.0:
                painter.setClipRect(QRectF(target.x(), target.y(),
                                           mx + rect.width() * wipe, target.height()))
            fade = view.xfade.get(now)
            if view.prev is not None and fade < 1.0:
                painter.setOpacity(opacity * (1.0 - fade))
                painter.drawImage(target, view.prev)
                painter.setOpacity(opacity * fade)
            else:
                view.prev = None
                painter.setOpacity(opacity)
            painter.drawImage(target, view.image)
            painter.restore()

    def _paint_notice(self, painter: QPainter, notice: _Notice, now: float) -> None:
        """The bar (the English bar color, Decision 25), the text, then the
        banner: amber with an ink label, or ink with a paper label for the
        recovery line."""
        opacity = notice.opacity.get(now)
        if opacity <= 0.0 or notice.rect is None or notice.font is None:
            return
        rect = notice.rect.get(now)
        # From the animated rect, not bar_w: a re-measure sets bar_w at once
        # while the rect glides, and the bar follows the glide like a block.
        bar = QRectF(rect.x() + BANNER_WIDTH, rect.y() + rect.height() - NOTICE_BAR_H,
                     rect.width() - BANNER_WIDTH, NOTICE_BAR_H)
        if notice.image is None:
            notice.image = _notice_text_image(notice, self._dpr or 1.0)
        tint, ink = ((NOTICE_INK, NOTICE_RECOVERY_TEXT) if notice.recovery
                     else (NOTICE_BANNER, NOTICE_INK))
        painter.save()
        painter.setOpacity(opacity)
        painter.fillRect(bar, QColor(*self._look.bar))
        painter.drawImage(bar, notice.image)
        paint_banner(painter, QRectF(rect.x(), rect.y(), BANNER_WIDTH, rect.height()),
                     self._notice_banner, QColor(*tint), QColor(*ink))
        painter.restore()

    # -- clock ----------------------------------------------------------

    def _paint_bounds(self, now: float) -> QRect:
        bounds = QRectF()
        if self._plates_opacity.get(now) > 0.0:
            if self._jp_plate_on:
                bounds = bounds.united(self._jp_plate.get(now))
            if self._en_plate_on:
                bounds = bounds.united(self._en_plate.get(now))
        reach = 2 * IMAGE_MARGIN
        for line in self._ghosts + self._lines:
            if line.opacity.get(now) <= 0.0:
                continue
            for view in line.blocks.values():
                bounds = bounds.united(view.rect.get(now).adjusted(-reach, -reach, reach, reach))
        if self._banner_opacity.get(now) > 0.0:
            bounds = bounds.united(self._banner_rect.get(now))
        for notice in self._notice_ghosts + self._notices:
            if notice.rect is not None and notice.opacity.get(now) > 0.0:
                bounds = bounds.united(notice.rect.get(now))
        if bounds.isNull():
            return QRect()
        return bounds.toAlignedRect().adjusted(-2, -2, 2, 2)

    def _anchored_anims(self) -> tuple:
        return (self._jp_plate, self._en_plate, self._plates_opacity,
                self._banner_rect, self._banner_opacity, self._banner_tint)

    def _anims(self):
        yield from self._anchored_anims()
        for line in self._ghosts + self._lines:
            yield line.opacity
            for view in line.blocks.values():
                yield from (view.rect, view.wipe, view.xfade)
        for notice in self._notice_ghosts + self._notices:
            yield notice.opacity
            if notice.rect is not None:
                yield notice.rect

    def _active(self, now: float) -> bool:
        return any(anim.active(now) for anim in self._anims())

    def _tick(self) -> None:
        now = self._clock()
        self._ghosts = [g for g in self._ghosts if g.opacity.active(now)]
        self._notice_ghosts = [g for g in self._notice_ghosts if g.opacity.active(now)]
        bounds = self._paint_bounds(now)
        dirty = bounds.united(self._painted)
        self._painted = bounds
        if not dirty.isEmpty():
            self.update(dirty)
        if not self._active(now):
            self._timer.stop()

    def _kick(self) -> None:
        self._tick()
        if self._active(self._clock()) and not self._timer.isActive():
            self._timer.start()
