"""The startup error window ("Report card", design record Surface 6).

A 720 px window titled "Yamabiko": a 44 px mast (the landscape strip behind
the painted caption), a left column with the title in BIZ UDPMincho, the
message, "What to try" as one dash item per sentence and, for a missing audio
device, "Open Settings"; a right "Report" panel with the hanko seal, the facts
(Yamabiko, Windows, GPU, Error), "Recent log" with "Open log folder" and the
full-width "Copy report"; a footer with the privacy note and Close.

Built from the Settings chrome pieces (``window_chrome``, ``controls``,
``landscape``, ``seal``, ``theme``, ``fonts``). Imported lazily by
``error_dialog.show_startup_error``: the watchdog imports that module before
any Qt exists. Everything shown or copied comes from
``yamabiko.ui.error_report`` (redacted).
"""
from __future__ import annotations

import ctypes
import logging
import os
import re
import time

from PySide6.QtCore import QCoreApplication, QEventLoop, QPoint, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import (
    QAction, QColor, QFont, QFontMetricsF, QGuiApplication, QIcon, QKeySequence, QPainter,
    QPainterPath, QPen, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument,
    QTextLayout, QTextLine, QTextOption,
)
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy,
    QTextBrowser, QVBoxLayout, QWidget,
)

from yamabiko import paths
from yamabiko.overlay_app.controls import LinkButton, _ClickOnEnter, _PaintedButton, mincho_font, ui_font
from yamabiko.overlay_app.fonts import register_bundled_fonts
from yamabiko.overlay_app.landscape import paint_strip
from yamabiko.overlay_app.seal import Seal
from yamabiko.overlay_app.theme import palette_for
from yamabiko.overlay_app.window_chrome import (
    CAPTION_H, ClientAreaFit, DialogCaption, add_window_menu, apply_window_qss,
    expand_client_area, nonclient_event, set_border_color, take_foreground,
)
from yamabiko.ui.error_dialog import StartupError
from yamabiko.ui.error_report import (
    LogLine, ReportFacts, collect_facts, condense_log, redact_user, report_text,
)
from yamabiko.ui.qt_bootstrap import ensure_qapplication

_log = logging.getLogger(__name__)

WINDOW_W = 720
MAST_H = 44
PANEL_W = 412
FOOTER_H = 48
LOG_W, LOG_H = 382, 109
# Inline code in the "What to try" items (the models case): Consolas on the
# field tone, never broken across lines.
CODE_CHIPS = ("state.json", "%LOCALAPPDATA%\\Yamabiko")
FOOTER_NOTE = "Nothing is sent automatically, and your user name is left out."
FACT_LABELS = ("Yamabiko", "Windows", "GPU", "Error")
# The Error value as shown (the copied report keeps the whole line): the
# exception's own type name without its dotted module path, on three lines
# at most.
_MODULE_PATH = re.compile(r"^(?:[A-Za-z_]\w*\.)+(?=[A-Z]\w*(?::|$))")
ERROR_LINES = 3
_WORD_JOINER = "\u2060"  # no line break on either side; zero width, not drawn

# Keep the QApplication alive for the life of the process if we had to
# construct one ad hoc; otherwise Qt's C++ side can reference a deleted
# Python object during window teardown.
_owned_app: "object | None" = None


def mono_font(px: int = 12) -> QFont:
    """The report's monospace face (the log, the Error value, code chips)."""
    font = QFont()
    font.setFamilies(["Consolas", "Cascadia Mono", "Courier New"])
    font.setPixelSize(px)
    return font


def _weighted(font: QFont, weight: QFont.Weight) -> QFont:
    font = QFont(font)
    font.setWeight(weight)
    return font


def _utf16_len(text: str) -> int:
    """``text``'s length in UTF-16 units, the unit of Qt's text positions."""
    return len(text.encode("utf-16-le")) // 2


def _join_chips(text: str, chips: tuple[str, ...]) -> tuple[str, list[tuple[int, int]]]:
    """``text`` with each code chip's characters joined by word joiners (no
    line break inside it), and the chips' (start, length) in that display
    text, in UTF-16 units (QTextLayout's positions). A chip followed by a
    backslash is the start of a longer path ("%LOCALAPPDATA%\\Yamabiko\\logs"
    in a hand-built message), not the chip, and stays plain text."""
    if not chips:
        return text, []
    pattern = re.compile("(?:" + "|".join(re.escape(chip) for chip in chips) + r")(?!\\)")
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    shown = 0
    position = 0
    for match in pattern.finditer(text):
        plain = text[position:match.start()]
        joined = _WORD_JOINER.join(match.group())
        parts += [plain, joined]
        shown += _utf16_len(plain)
        spans.append((shown, _utf16_len(joined)))
        shown += _utf16_len(joined)
        position = match.end()
    parts.append(text[position:])
    return "".join(parts), spans


class TextBlock(QWidget):
    """Wrapped text painted on a fixed line box (the mockup's CSS line
    height, which QLabel can't do). ``wrap``: "greedy"; "balance" (the
    narrowest width that keeps the greedy line count, CSS text-wrap:
    balance); "pretty" (no lone last word when a narrower width avoids it at
    the same line count). Long words break anywhere. ``indent`` insets the
    text; ``marker`` paints the muted 7 x 1.5 px dash at (+1, +9) before it;
    ``chips`` are substrings drawn in Consolas 12 on the field tone, never
    broken (not where a backslash follows: that is a longer path).
    ``top_pad`` adds space above the first line. ``max_lines`` caps the
    lines shown, the last one ending in an ellipsis where text is cut."""

    def __init__(self, text: str, font: QFont, line_height: int, palette: dict, *,
                 color: str = "ink", wrap: str = "greedy", indent: int = 0,
                 marker: bool = False, chips: tuple[str, ...] = (), top_pad: int = 0,
                 max_lines: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._line_height = line_height
        self._palette = palette
        self._color = color
        self._wrap = wrap
        self._indent = indent
        self._marker = marker
        self._chips = chips
        self._top_pad = top_pad
        self._max_lines = max_lines
        self._option = QTextOption()
        self._option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setFont(font)
        self.set_text(text)
        # Preferred, not Fixed: a vertical box gives a height-for-width item
        # heightForWidth() but caps a Fixed one at its one-line size hint.
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        """Show ``text``: re-wrapped at the current width, the accessible name
        and the height-for-width following (setup's live lines)."""
        self._text = text
        self._display, self._chip_spans = _join_chips(text, self._chips)
        # Qt's line positions count UTF-16 units: slice these, not the str
        # (an emoji is one code point but two units).
        self._units = self._display.encode("utf-16-le")
        self._widths: dict[int, tuple[float, int]] = {}
        self.setAccessibleName(" ".join(text.split()))
        self.updateGeometry()
        self.update()

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.update()

    def _formats(self) -> list:
        ranges = []
        for start, length in self._chip_spans:
            fmt = QTextCharFormat()
            fmt.setFont(mono_font(12))
            fmt.setBackground(QColor(self._palette["field"]))
            item = QTextLayout.FormatRange()
            item.start, item.length, item.format = start, length, fmt
            ranges.append(item)
        return ranges

    def _layout(self, width: float, text: str | None = None) -> QTextLayout:
        layout = QTextLayout(self._display if text is None else text, self.font())
        layout.setTextOption(self._option)
        layout.setFormats(self._formats())
        layout.beginLayout()
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(1.0, width))
        layout.endLayout()
        return layout

    def _shown(self, width: float) -> str:
        """The display text as shown at wrap ``width``: past ``max_lines``
        lines, the last line holds as much of the rest as fits before an
        ellipsis."""
        if self._max_lines is None:
            return self._display
        layout = self._layout(width)
        if layout.lineCount() <= self._max_lines:
            return self._display
        start = 2 * layout.lineAt(self._max_lines - 1).textStart()
        rest = self._units[start:].decode("utf-16-le")
        return self._units[:start].decode("utf-16-le") + QFontMetricsF(self.font()).elidedText(
            rest, Qt.TextElideMode.ElideRight, width)

    def _line_text(self, line: QTextLine, units: bytes | None = None) -> str:
        """The display text of one laid-out ``line`` (word joiners kept) of
        ``units`` (the whole display text by default)."""
        units = self._units if units is None else units
        start = 2 * line.textStart()
        return units[start: start + 2 * line.textLength()].decode("utf-16-le")

    def _last_line_is_one_word(self, layout: QTextLayout) -> bool:
        return " " not in self._line_text(layout.lineAt(layout.lineCount() - 1)).strip()

    def _fit(self, width: int) -> tuple[float, int]:
        """The wrap width and line count for an available ``width``."""
        if width in self._widths:
            return self._widths[width]
        avail = float(max(1, width - self._indent))
        greedy = self._layout(avail)
        count = greedy.lineCount()
        chosen = avail
        if self._wrap == "balance" and count > 1:
            low, high = 1.0, avail
            while high - low > 0.5:
                middle = (low + high) / 2.0
                if self._layout(middle).lineCount() <= count:
                    high = middle
                else:
                    low = middle
            chosen = high
        elif self._wrap == "pretty" and count > 1 and self._last_line_is_one_word(greedy):
            trial = avail - 1.0
            while trial > avail * 0.6:
                layout = self._layout(trial)
                if layout.lineCount() != count:
                    break
                if not self._last_line_is_one_word(layout):
                    chosen = trial
                    break
                trial -= 1.0
        if self._max_lines is not None:
            count = min(count, self._max_lines)
        self._widths[width] = (chosen, count)
        return chosen, count

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt override)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        return self._top_pad + self._fit(width)[1] * self._line_height

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        natural = int(QFontMetricsF(self.font()).horizontalAdvance(self._display)) + 1 + self._indent
        return QSize(natural, self.heightForWidth(natural))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(0, self._top_pad + self._line_height)

    def lines(self) -> list[str]:
        """The wrapped lines shown at the current width (for tests and the
        accessible text), without the chips' word joiners."""
        width = self._fit(self.width())[0]
        shown = self._shown(width)
        layout = self._layout(width, shown)
        units = shown.encode("utf-16-le")
        return [
            self._line_text(layout.lineAt(index), units).replace(_WORD_JOINER, "").strip()
            for index in range(layout.lineCount())
        ]

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        if self._marker:
            p.fillRect(QRectF(1, self._top_pad + 9, 7, 1.5), QColor(self._palette["muted"]))
        width = self._fit(self.width())[0]
        layout = self._layout(width, self._shown(width))
        for index in range(layout.lineCount()):
            line = layout.lineAt(index)
            top = self._top_pad + index * self._line_height
            line.setPosition(QPointF(self._indent, top + (self._line_height - line.height()) / 2.0))
        p.setPen(QColor(self._palette[self._color]))
        layout.draw(p, QPointF(0, 0))
        p.end()


class CopyReportButton(_ClickOnEnter, _PaintedButton):
    """The Report panel's full-width 36 px primary action: accent fill, the
    copy icon and "Copy report" (14 px semibold, on-accent); once copied, the
    field tone with a 1.5 px accent border, the accent check and "Copied.
    Paste it into a message to the developer." (13 px, ink) until the window
    closes. Its text and accessible name stay "Copy report" (UI Automation
    finds and invokes it by that name); the copied sentence is its
    accessible description."""

    TEXT = "Copy report"
    COPIED_TEXT = "Copied. Paste it into a message to the developer."
    _COPIED_LEAD = "Copied."
    _ICON = 14.0
    _GAP = 8.0

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(palette, parent)
        self.setText(self.TEXT)
        self.setAccessibleName(self.TEXT)
        self.setFixedHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._copied = False

    def copied(self) -> bool:
        return self._copied

    def set_copied(self) -> None:
        self._copied = True
        self.setAccessibleDescription(self.COPIED_TEXT)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(LOG_W, 36)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        rect = QRectF(self.rect())
        if self._copied:
            p.setPen(QPen(self._color("accent"), 1.5))
            p.setBrush(self._color("field"))
            p.drawRoundedRect(rect.adjusted(0.75, 0.75, -0.75, -0.75), 2.0, 2.0)
            lead_font = _weighted(ui_font(13), QFont.Weight.DemiBold)
            rest_font = ui_font(13)
            rest = self.COPIED_TEXT[len(self._COPIED_LEAD):]
            lead_w = QFontMetricsF(lead_font).horizontalAdvance(self._COPIED_LEAD)
            rest_w = QFontMetricsF(rest_font).horizontalAdvance(rest)
            x = (rect.width() - (self._ICON + self._GAP + lead_w + rest_w)) / 2.0
            self._paint_check(p, QPointF(x, (rect.height() - self._ICON) / 2.0))
            baseline = self._baseline(rest_font)
            p.setPen(self._color("ink"))
            p.setFont(lead_font)
            p.drawText(QPointF(x + self._ICON + self._GAP, baseline), self._COPIED_LEAD)
            p.setFont(rest_font)
            p.drawText(QPointF(x + self._ICON + self._GAP + lead_w, baseline), rest)
            ring = self._color("accent")
        else:
            fill = self._color("accent_hover" if self.isEnabled() and self.underMouse() else "accent")
            p.setPen(QPen(fill, 1.0))
            p.setBrush(fill)
            p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)
            font = _weighted(ui_font(14), QFont.Weight.DemiBold)
            text_w = QFontMetricsF(font).horizontalAdvance(self.TEXT)
            x = (rect.width() - (self._ICON + self._GAP + text_w)) / 2.0
            self._paint_copy(p, QPointF(x, (rect.height() - self._ICON) / 2.0))
            p.setPen(self._color("on_accent"))
            p.setFont(font)
            p.drawText(QPointF(x + self._ICON + self._GAP, self._baseline(font)), self.TEXT)
            ring = self._color("on_accent")
        if self.ring_visible():
            p.setPen(QPen(ring, 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(3.0, 3.0, -3.0, -3.0), 2.0, 2.0)
        p.end()

    def _baseline(self, font: QFont) -> float:
        metrics = QFontMetricsF(font)
        return (self.height() - metrics.height()) / 2.0 + metrics.ascent()

    def _paint_copy(self, p: QPainter, origin: QPointF) -> None:
        """The mockup's copy icon (14 box, 1.3 px): the front page and the
        back page's corner."""
        p.save()
        p.translate(origin)
        pen = QPen(self._color("on_accent"), 1.3)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(4.5, 4.5, 7.5, 8.0), 1.0, 1.0)
        back = QPainterPath(QPointF(2.0, 9.5))
        back.lineTo(2.0, 2.5)
        back.arcTo(QRectF(2.0, 1.5, 2.0, 2.0), 180.0, -90.0)
        back.lineTo(9.0, 1.5)
        p.drawPath(back)
        p.restore()

    def _paint_check(self, p: QPainter, origin: QPointF) -> None:
        """The mockup's check (14 box, 1.7 px, round caps and joins)."""
        p.save()
        p.translate(origin)
        pen = QPen(self._color("accent"), 1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        check = QPainterPath(QPointF(2.5, 7.5))
        check.lineTo(5.5, 10.5)
        check.lineTo(11.5, 3.5)
        p.drawPath(check)
        p.restore()


class LogView(QTextBrowser):
    """The Report panel's "Recent log": the tail's records condensed to
    "HH:MM:SS name LEVEL message" in Consolas 12 on 16 px lines, colored
    (time muted, name ink, INFO muted, WARNING ink bold, ERROR accent, an
    exception's final line accent bold), each record one block with a 2
    character hanging indent; read-only, focusable, opens scrolled to the
    newest line. Its box (field, 1 px edge, radius 2) is the window QSS's
    ``QTextBrowser#recentLog``; the insets keep a clean 6 px strip on top."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setObjectName("recentLog")
        self.setAccessibleName("Recent log")
        self.setFixedSize(LOG_W, LOG_H)
        self.setOpenLinks(False)
        self.setFont(mono_font(12))
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Inside the 1 px border: a 6 px top strip, six whole 16 px lines,
        # 5 px under them (107 = 6 + 96 + 5); text 9 px in, 4 px off the right.
        self.setViewportMargins(9, 6, 4, 5)
        self.document().setDocumentMargin(0)
        self.document().setDefaultFont(mono_font(12))

    def _block_format(self) -> QTextBlockFormat:
        two_ch = 2.0 * QFontMetricsF(mono_font(12)).horizontalAdvance("0")
        block = QTextBlockFormat()
        block.setLeftMargin(two_ch)
        block.setTextIndent(-two_ch)
        block.setLineHeight(16.0, QTextBlockFormat.LineHeightTypes.FixedHeight.value)
        return block

    def _char(self, token: str, *, bold: bool = False) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setFont(mono_font(12))
        fmt.setForeground(QColor(self._palette[token]))
        fmt.setFontWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
        return fmt

    def _level_format(self, level: str) -> QTextCharFormat:
        if level == "WARNING":
            return self._char("ink", bold=True)
        if level in ("ERROR", "CRITICAL"):
            return self._char("accent")
        return self._char("muted")

    def set_lines(self, lines: list[str]) -> None:
        """Show the raw tail ``lines`` condensed and colored."""
        document = QTextDocument(self)
        document.setDocumentMargin(0)
        document.setDefaultFont(mono_font(12))
        cursor = QTextCursor(document)
        for index, line in enumerate(condense_log(lines)):
            if index:
                cursor.insertBlock(self._block_format())
            else:
                cursor.setBlockFormat(self._block_format())
            self._insert(cursor, line)
        self.setDocument(document)
        self.scroll_to_newest()

    def _insert(self, cursor: QTextCursor, line: LogLine) -> None:
        if line.kind == "exception":
            cursor.insertText(line.text, self._char("accent", bold=True))
        elif line.kind == "text":
            cursor.insertText(line.text, self._char("ink"))
        else:
            cursor.insertText(line.time, self._char("muted"))
            cursor.insertText(" ", self._char("ink"))
            cursor.insertText(line.name, self._char("ink"))
            cursor.insertText(" ", self._char("ink"))
            cursor.insertText(line.level, self._level_format(line.level))
            if line.text:
                cursor.insertText(" " + line.text, self._char("ink"))

    def scroll_to_newest(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def copy_selection(self) -> None:
        """Copy the selected lines as plain text through copy_to_clipboard,
        flushed like Copy report's text so it outlives the pipeline's
        os._exit(43); Qt's own copy would leave with the process."""
        text = self.textCursor().selection().toPlainText()
        if text:
            copy_to_clipboard(text)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def context_menu(self, pos: QPoint) -> QMenu:
        """The standard context menu with its Copy (object name "edit-copy")
        taken off Qt's copy and put on copy_selection."""
        menu = self.createStandardContextMenu(pos)
        copy = menu.findChild(QAction, "edit-copy")
        if copy is not None:
            copy.triggered.disconnect()
            copy.triggered.connect(self.copy_selection)
        return menu

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt override)
        menu = self.context_menu(event.pos())
        menu.exec(event.globalPos())
        menu.deleteLater()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self.scroll_to_newest()


class ErrorMast(QWidget):
    """The 44 px mast: the landscape strip across it and the 32 px caption
    (app mark, "Yamabiko", close) on top of it. The caption drags the
    window."""

    def __init__(self, title: str, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)  # paints every pixel of its rect
        self._palette = palette
        self.setFixedHeight(MAST_H)
        self.caption = DialogCaption(title, palette, self)

    def apply_palette(self, palette: dict) -> None:
        self._palette = palette
        self.caption.apply_palette(palette)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.caption.setGeometry(0, 0, self.width(), CAPTION_H)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        paint_strip(p, QRectF(self.rect()), self._palette)
        p.end()


# OleFlushClipboard's failure while another process holds the clipboard open,
# as OleDLL reports it (OSError.winerror, the signed HRESULT 0x800401D0).
_CLIPBRD_E_CANT_OPEN = -2147221040
_FLUSH_RETRY_S = 0.5


def _flush_clipboard() -> None:
    """OleFlushClipboard, tried again for up to _FLUSH_RETRY_S while another
    process holds the clipboard open. Clipboard listeners (TextInputHost,
    explorer, the clipboard service ...) open it as soon as a copy lands
    (seen holding it through GetOpenClipboardWindow), so the first flush can
    fail with CLIPBRD_E_CANT_OPEN (frozen E2E: 2 of 7 runs). Events are
    pumped between tries, so a listener that asks this thread to render the
    text is served rather than kept waiting; user input waits, so a click
    can't re-enter the copy. A flush that took more than one try is logged
    (INFO)."""
    deadline = time.monotonic() + _FLUSH_RETRY_S
    tries = 0
    while True:
        tries += 1
        try:
            ctypes.OleDLL("ole32").OleFlushClipboard()
        except OSError as exc:
            if getattr(exc, "winerror", None) != _CLIPBRD_E_CANT_OPEN or time.monotonic() >= deadline:
                raise
        else:
            if tries > 1:
                _log.info("clipboard flush succeeded after %d tries", tries)
            return
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, 10)
        time.sleep(0.01)


def copy_to_clipboard(text: str) -> bool:
    """Put ``text`` on the clipboard so it outlives this process. Qt's
    Windows clipboard renders text on request (OLE delayed rendering); the
    pipeline leaves through os._exit(43) after the window closes, which took
    the text with it (probe: a child's setText + os._exit left no
    CF_UNICODETEXT). OleFlushClipboard renders it now, retried while another
    process holds the clipboard open (_flush_clipboard). False (logged) when
    the text didn't make it; never raises."""
    try:
        QGuiApplication.clipboard().setText(text)
        if QGuiApplication.platformName() == "windows":
            _flush_clipboard()
    except Exception:
        _log.warning("could not put the report on the clipboard", exc_info=True)
        return False
    return True


class _Rule(QWidget):
    """A 1 px rule in the line tone."""

    def __init__(self, palette: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setFixedHeight(1)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self._palette["line"]))
        p.end()


def _label(text: str, *, muted: bool = False, font: QFont | None = None) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setFont(font if font is not None else ui_font(13))
    if muted:
        label.setObjectName("muted")
    return label


class ErrorWindow(QDialog):
    """The startup error window for ``err`` with the facts collected when it
    opened, in ``palette`` (MIST or NIGHT). Close, the caption's close and
    Esc end it."""

    def __init__(self, err: StartupError, facts: ReportFacts, palette: dict) -> None:
        super().__init__(None)
        self.setObjectName("compactDialog")  # the window QSS's ground and 2 px buttons
        self.setWindowTitle("Yamabiko")
        icon = paths.app_icon_path()
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        expand_client_area(self)
        self.setFont(ui_font(13))
        self._err = err
        self._facts = facts
        self._palette = palette
        self.settings_button: LinkButton | None = None
        self.settings_window = None
        self._values: dict[str, TextBlock] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._mast = ErrorMast("Yamabiko", palette, self)
        self._mast.caption.closeRequested.connect(self.reject)
        root.addWidget(self._mast)
        body = QHBoxLayout()
        body.setContentsMargins(22, 8, 20, 14)
        body.setSpacing(24)
        body.addLayout(self._build_left(), 1)
        self._panel = self._build_panel()
        body.addWidget(self._panel)
        root.addLayout(body)
        root.addWidget(self._build_footer())
        self._fit = ClientAreaFit(self, root)
        self.setFixedWidth(WINDOW_W)
        apply_window_qss(self, palette)

    # --- building -------------------------------------------------------

    def _build_left(self) -> QVBoxLayout:
        left = QVBoxLayout()
        left.setContentsMargins(0, 8, 0, 0)
        left.setSpacing(0)
        p = self._palette
        # The title as the error gives it (factory copy or a device name: the
        # bare-name rule would turn "Ann's AirPods" into "<user>'s AirPods");
        # the message and the What to try sentences redacted, as the report
        # copies them (a hand-built message can carry exception text with a
        # path, ruling I2).
        self.title_label = TextBlock(self._err.title, mincho_font(21), 28, p, wrap="balance")
        left.addWidget(self.title_label)
        left.addSpacing(8)
        self._message = TextBlock(redact_user(self._err.message), ui_font(14), 20, p,
                                  wrap="pretty")
        left.addWidget(self._message)
        self._items: list[TextBlock] = []
        items = [redact_user(sentence) for sentence in self._err.what_to_try()]
        if items:
            left.addSpacing(20)
            heading = _label("What to try", font=mincho_font(15))
            heading.setFixedHeight(20)
            left.addWidget(heading)
            left.addSpacing(6)
            for index, sentence in enumerate(items):
                if index:
                    left.addSpacing(8)
                item = TextBlock(sentence, ui_font(13), 19, p, wrap="pretty", indent=15,
                                 marker=True, chips=CODE_CHIPS)
                self._items.append(item)
                left.addWidget(item)
            if self._err.settings_page:
                # The text button under the last item, at the items' text
                # (13 px semibold, 24 px tall, 3 px below the item).
                self.settings_button = LinkButton("Open Settings", p, height=24)
                self.settings_button.setFont(_weighted(ui_font(13), QFont.Weight.DemiBold))
                self.settings_button.clicked.connect(self.open_settings)
                left.addSpacing(3)
                row = QHBoxLayout()
                row.setContentsMargins(15, 0, 0, 0)
                row.addWidget(self.settings_button)
                row.addStretch(1)
                left.addLayout(row)
        left.addStretch(1)
        return left

    def _build_panel(self) -> QFrame:
        p = self._palette
        panel = QFrame()
        panel.setObjectName("reportPanel")
        panel.setFixedWidth(PANEL_W)
        rows = QVBoxLayout(panel)
        # The QSS border (1 px) already insets the contents: 14, 10, 14, 14 inside it.
        rows.setContentsMargins(14, 10, 14, 14)
        rows.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(10)
        report = _label("Report", font=mincho_font(15))
        report.setFixedHeight(20)
        head.addWidget(report, 0, Qt.AlignVCenter)
        head.addWidget(_label("For the developer", muted=True), 0, Qt.AlignVCenter)
        head.addStretch(1)
        head.addWidget(Seal(22), 0, Qt.AlignVCenter)
        head_box = QWidget()
        head_box.setFixedHeight(24)
        head_box.setLayout(head)
        rows.addWidget(head_box)
        rows.addSpacing(8)

        facts = QGridLayout()
        facts.setContentsMargins(0, 0, 0, 0)
        facts.setHorizontalSpacing(10)
        facts.setVerticalSpacing(2)
        facts.setColumnMinimumWidth(0, 62)
        values = (self._facts.version, self._facts.windows, self._facts.gpu, self._facts.error)
        for row, (label, value) in enumerate(zip(FACT_LABELS, values)):
            name = _label(label, muted=True)
            name.setFixedHeight(18)
            name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            facts.addWidget(name, row, 0, Qt.AlignTop)
            if label == "Error":
                shown = TextBlock(_MODULE_PATH.sub("", value), mono_font(12), 17, p,
                                  top_pad=1, max_lines=ERROR_LINES)
            else:
                shown = TextBlock(value, ui_font(13), 18, p)
            self._values[label] = shown
            facts.addWidget(shown, row, 1)
        facts.setColumnStretch(1, 1)
        rows.addLayout(facts)
        rows.addSpacing(8)
        rows.addWidget(_Rule(p))
        rows.addSpacing(4)

        caption = QHBoxLayout()
        caption.setContentsMargins(0, 0, 0, 0)
        caption.addWidget(_label("Recent log", muted=True), 0, Qt.AlignVCenter)
        caption.addStretch(1)
        self.open_log_button = LinkButton("Open log folder", p, height=24)
        self.open_log_button.setFont(ui_font(13))
        self.open_log_button.clicked.connect(self.open_log_folder)
        caption.addWidget(self.open_log_button, 0, Qt.AlignVCenter)
        caption_box = QWidget()
        caption_box.setFixedHeight(24)
        caption_box.setLayout(caption)
        rows.addWidget(caption_box)
        rows.addSpacing(3)

        self.log_view = LogView(p)
        self.log_view.set_lines(self._facts.log.lines)
        rows.addWidget(self.log_view)
        rows.addSpacing(10)
        self.copy_button = CopyReportButton(p)
        self.copy_button.clicked.connect(self._copy_report)
        rows.addWidget(self.copy_button)
        rows.addStretch(1)
        return panel

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("footer")  # the window QSS's 1 px line on top
        footer.setFixedHeight(FOOTER_H)
        row = QHBoxLayout(footer)
        row.setContentsMargins(22, 1, 20, 0)
        row.setSpacing(12)
        row.addWidget(_label(FOOTER_NOTE, muted=True), 0, Qt.AlignVCenter)
        row.addStretch(1)
        self.close_button = QPushButton("Close")
        self.close_button.setFont(ui_font(13))
        self.close_button.clicked.connect(self.reject)
        row.addWidget(self.close_button, 0, Qt.AlignVCenter)
        return footer

    # --- behavior -------------------------------------------------------

    def fact_value(self, label: str) -> str:
        """The value shown for a fact ("Yamabiko", "Windows", "GPU", "Error")."""
        return self._values[label].text()

    def copied(self) -> bool:
        return self.copy_button.copied()

    def _copy_report(self) -> None:
        if copy_to_clipboard(report_text(self._err, self._facts)):
            self.copy_button.set_copied()

    def open_log_folder(self) -> None:
        """Open the folder of the log the tail came from (the logs folder
        when there was none). Best effort: a missing or locked folder is
        logged, never raised."""
        tail = self._facts.log.path
        folder = tail.parent if tail is not None else paths.logs_dir()
        try:
            paths.ensure_dirs()
            os.startfile(folder)
        except OSError:
            _log.warning("could not open the log folder", exc_info=True)

    def open_settings(self) -> None:
        """Open Settings at ``err.settings_page`` in this process as its own
        top-level window, as the tray opens it: unowned, so it keeps its
        13 px fonts (owned by this styled window, its labels and buttons
        took the 9 pt app font: probe smoke3), and not blocked, because
        this window runs without modality (run_error_window). Seeded with
        the user's saved style the way the tray seeds it: every Apply writes
        all of overlay.style, so a defaults seed would reset a customized
        style. Live-apply signals stay unconnected: the choice applies the
        next time Yamabiko starts. A second click raises the open window."""
        page = self._err.settings_page
        existing = self.settings_window
        if existing is not None and existing.isVisible():
            existing.show_page(page)
            if existing.isMinimized():
                existing.showNormal()
            existing.raise_()
            existing.activateWindow()
            return
        from yamabiko import config
        from yamabiko.overlay_app.settings_window import SettingsWindow

        try:
            style = dict(config.load_overlay_config().style)
        except Exception:
            # Unreachable in practice: the device-gone error comes after the
            # pipeline loaded this same config. Editing over a config that
            # can't be read would save the default style over the user's.
            _log.warning("could not read the saved settings; Settings stays closed",
                         exc_info=True)
            return
        window = SettingsWindow(style, None, config_path=paths.user_overrides_path(), page=page)
        self.settings_window = window
        window.show()
        window.raise_()
        window.activateWindow()

    def wait_for_settings(self) -> None:
        """Return once a Settings window this window opened is closed (it
        stays open after this window closes; the process must outlive it)."""
        window = self.settings_window
        if window is None or not window.isVisible():
            return
        loop = QEventLoop()
        window.previewEnded.connect(loop.quit)  # emitted by its closeEvent
        loop.exec()

    def refit_height(self) -> None:
        """Fix the height to the content at the window's width."""
        layout = self.layout()
        layout.activate()
        if layout.hasHeightForWidth():
            self.setFixedHeight(layout.totalHeightForWidth(WINDOW_W))
        else:
            self.setFixedHeight(layout.totalSizeHint().height())

    def center_on(self, area: QRect) -> None:
        """Center the fitted window in ``area`` (a screen's work area), its
        top-left kept inside it."""
        x = area.x() + max(0, (area.width() - self.width()) // 2)
        y = area.y() + max(0, (area.height() - self.height()) // 2)
        self.move(x, y)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.refit_height()
        if not event.spontaneous():
            # Here, before the native show: a move after it would lay the
            # frame out on the window menu's margins (setVisible) and widen
            # the window.
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                self.center_on(screen.availableGeometry())
        super().showEvent(event)
        set_border_color(self, self._palette["frame"])

    def setVisible(self, visible: bool) -> None:  # noqa: N802 (Qt override)
        super().setVisible(visible)
        if visible:
            # After the native show, never in showEvent: Qt 6.11 keeps two
            # frame margins for an expanded client area, WM_NCCALCSIZE's
            # (8 device px a side at 150 %) and its own (6), which it reads
            # the geometry back with on each WM_MOVE/WM_SIZE. The window
            # menu's frame change (SWP_FRAMECHANGED) installs the first, and
            # a native resize before the next move or size lays the frame out
            # on them: 4 device px wider than Qt reads back (DWM 1084 px, Qt
            # 723). The show itself applies the height fitted while hidden,
            # after showEvent (probe .superpowers/sdd/probes/p6f3_geom.py,
            # QT_LOGGING_RULES=qt.qpa.window=true).
            add_window_menu(self, min_max=False)

    def nativeEvent(self, event_type, message):  # noqa: N802 (Qt override)
        result = nonclient_event(event_type, message, self._palette["ground"])
        if result is not None:
            return True, result
        return super().nativeEvent(event_type, message)


def saved_theme() -> str:
    """overlay.theme from user_overrides.yaml ("system" when absent or
    invalid). Read directly: load_full_config raises in exactly the config
    errors this window reports, and writes files. Setup reads it the same
    way (wizard.run_wizard)."""
    import yaml

    try:
        with open(paths.user_overrides_path(), encoding="utf-8") as handle:
            value = yaml.safe_load(handle)["overlay"]["theme"]
    except Exception:
        return "system"
    return value if value in ("system", "light", "dark") else "system"


def run_error_window(err: StartupError) -> None:
    """Show the window for ``err`` (centered on the primary screen's work
    area) and return when it is closed, together with a Settings window it
    opened. Reads the facts first, then creates the QApplication when none
    exists and registers the bundled fonts (BIZ UDPMincho)."""
    global _owned_app
    # Before anything here logs (the font registration does): the log tail
    # then ends with show_startup_error's "startup error dialog:" line.
    facts = collect_facts(err)
    app = ensure_qapplication()
    _owned_app = app  # prevent GC (no-op if Qt already owned one)
    register_bundled_fonts()
    palette = palette_for(saved_theme())  # after the QApplication: the system probe asks Qt
    window = ErrorWindow(err, facts, palette)
    try:
        # The pipeline (a background child of the watchdog) shows this
        # window: without the foreground it opens behind the user's apps.
        window.show()
        window.raise_()
        take_foreground(window)
        window.activateWindow()
        # show() and a local loop, not exec(): exec() in place of show() shows
        # the window application-modal, which blocks the Settings window Open
        # Settings opens (probe smoke5: its Cancel click never arrived).
        # Nothing else shares this process's screen, so nothing needs blocking.
        loop = QEventLoop()
        window.finished.connect(loop.quit)
        loop.exec()
        window.wait_for_settings()
    finally:
        # Hidden before the last references go, even when something above
        # raised: past this call only each window's ClientAreaFit cycle keeps
        # it, and a garbage collection inside a later paint (the watchdog
        # shows queued errors in a row) would free a shown one mid-paint.
        window.hide()
        if window.settings_window is not None:
            window.settings_window.hide()
