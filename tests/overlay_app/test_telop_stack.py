"""SubtitleStack: the telop overlay's layout, echo, banner and motion."""
import pytest

from yamabiko.overlay_app.style_defaults import STYLE_DEFAULTS
from yamabiko.overlay_app.telop_look import BANNER_PAUSED, BANNER_RED, BANNER_WIDTH, PAPER

SEGS_A = [{"text": "ちょっと"}, {"text": "待", "reading": "ま"}, {"text": "って。"}]
SEGS_B = [{"text": "いや、これがこの"}, {"text": "町", "reading": "まち"},
          {"text": "の"}, {"text": "名物", "reading": "めいぶつ"}, {"text": "なんですよ。"}]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_stack(qapp):
    from yamabiko.overlay_app.fonts import register_bundled_fonts
    from yamabiko.overlay_app.telop_stack import SubtitleStack

    register_bundled_fonts()
    made = []

    def _make(style=None, clock=None, anchor_x=100, cap=1200):
        stack = SubtitleStack(style={**STYLE_DEFAULTS, "en_font": "BIZ UDPGothic", **(style or {})},
                              clock=clock)
        stack.resize(1920, 1080)
        stack.set_anchor(anchor_x, 950, cap, 1920, 0, 1080)
        made.append(stack)
        return stack

    yield _make
    for stack in made:
        stack.deleteLater()


def test_left_alignment_stacks_lines_on_left_anchor_with_banner_gutter(make_stack):
    """Left: every line starts at the anchor, newest at the bottom; the
    banner sits in the 38 px gutter beside the newest line (same left, same
    bottom, at least its height); the paper strip and dark bar fit the
    newest line's Japanese and English blocks, each hugging its own text.

    Seeded defects: centre lines in Left, or place the plates from the
    oldest line -> red."""
    stack = make_stack()
    stack.add_line(SEGS_A, "Wait a moment.", instant=True)
    stack.add_line(SEGS_B, "Come on, this is what the town is famous for.", instant=True)
    frame = stack.frame()
    newest, older = frame.lines[-1], frame.lines[0]
    assert (newest.age, older.age) == (0, 1)
    assert newest.rect.x() == 100 and older.rect.x() == 100
    assert newest.rect.bottom() + 1 == 950 and older.rect.bottom() < newest.rect.top()
    assert frame.banner.x() == 100 and frame.banner.width() == BANNER_WIDTH
    assert frame.banner.bottom() == newest.rect.bottom()
    assert frame.banner.height() >= newest.rect.height()
    assert frame.jp_plate == newest.blocks["jp"] and frame.en_plate == newest.blocks["en"]
    assert frame.jp_plate.x() == 100 + BANNER_WIDTH
    assert frame.jp_plate.width() != frame.en_plate.width()  # ragged, not shared


def test_center_alignment_centers_lines_and_plates_share_the_line_width(make_stack):
    """Center: each line's box (gutter included) is centered on the anchor
    and the strip and bar share the line's full width.

    Seeded defect: skip the shared-width widening -> plate widths differ ->
    red."""
    stack = make_stack({"alignment": "center"}, anchor_x=960)
    stack.add_line(SEGS_A, "Wait a moment.", instant=True)
    stack.add_line(SEGS_B, "Come on, this is what the town is famous for.", instant=True)
    frame = stack.frame()
    for box in frame.lines:
        assert abs(box.rect.center().x() - 960) <= 1
    assert frame.jp_plate.width() == frame.en_plate.width()
    assert frame.banner.x() == frame.lines[-1].rect.x()


def test_echo_ages_shrink_and_leave_past_max_lines(make_stack):
    """Four lines with max_lines 3: the oldest leaves; ages run 2, 1, 0
    oldest to newest, and older lines are shorter (84% / 70% sizes).

    Seeded defect: never trim -> four lines -> red."""
    stack = make_stack({"echo": "both"})
    for i in range(4):
        stack.add_line(SEGS_A, f"Line {i}.", hold_ms=60000, instant=True)
    ages = [box.age for box in stack.frame().lines]
    assert ages == [2, 1, 0]
    heights = [box.rect.height() for box in stack.frame().lines]
    assert heights[0] < heights[1] < heights[2]


def test_priority_echo_keeps_one_language_and_both_keeps_two(make_stack):
    """Echo "priority" keeps English in Watch and Japanese in Learn; "both"
    keeps both. The newest line always shows both.

    Seeded defect: swap shows_jp / shows_en -> red."""
    for style, expected in (
        ({"mode": "watch", "echo": "priority"}, {"en"}),
        ({"mode": "learn", "echo": "priority"}, {"jp"}),
        ({"mode": "watch", "echo": "both"}, {"jp", "en"}),
    ):
        stack = make_stack(style)
        stack.add_line(SEGS_A, "Wait a moment.", instant=True)
        stack.add_line(SEGS_B, "This is it.", instant=True)
        older, newest = stack.frame().lines
        assert set(older.blocks) == expected, style
        assert set(newest.blocks) == {"jp", "en"}


def test_banner_grows_upward_past_the_newest_line(make_stack):
    """A series name longer than the newest line grows the banner upward:
    same bottom as the line, taller than it.

    Seeded defect: clamp the banner to the line height -> red."""
    stack = make_stack()
    stack.set_banner("とんでもスキルで異世界放浪メシ", "Campfire Cooking", False)
    stack.add_line([{"text": "はい。"}], "Yes.", instant=True)
    frame = stack.frame()
    newest = frame.lines[-1]
    assert frame.banner.bottom() == newest.rect.bottom()
    assert frame.banner.height() > newest.rect.height()


def test_paused_banner_stays_at_anchor_without_lines(make_stack):
    """Paused with no lines: the banner stays at the anchor (left edge in
    Left, bottom on the anchor). Not paused and no lines: no banner. While
    hidden it takes the new state at once: it appears grey (never vermilion
    fading to grey), and unpaused with no lines it fades out still showing
    the paused banner (grey, pause glyph), never flashing the genre. A line
    arriving while it is still fading out turns it back from the grey it
    shows: the tint fades to vermilion, it does not jump.

    Seeded defects: hide the banner whenever there are no lines; tween the
    tint while the banner appears; swap to the genre for the fade-out; treat
    a banner still fading out as hidden (its opacity target is 0) -> red."""
    now = [0.0]
    stack = make_stack(clock=lambda: now[0])
    assert stack.frame().banner is None
    stack.set_banner("一時停止", "Paused", True)
    frame = stack.frame()
    assert frame.paused and frame.banner is not None
    assert frame.banner.x() == 100 and frame.banner.bottom() + 1 == 950
    assert frame.jp_plate is None and frame.en_plate is None
    rect = frame.banner

    def white(image):   # the banner's white content (pause glyph, text)
        return {(x, y) for y in range(rect.top(), rect.bottom() + 1)
                for x in range(rect.left(), rect.right() + 1)
                if (c := image.pixelColor(x, y)).alpha() and min(c.red(), c.green(), c.blue()) > 200}

    now[0] = 200.0   # the 160 ms fade-in is done; a 300 ms tint fade would not be
    assert _pixel_at(stack, rect.x() + 1, rect.y() + 1) == BANNER_PAUSED
    shown = white(_render(stack))
    now[0] = 1000.0
    stack.set_banner("バラエティ", "Variety", False)
    assert stack.frame().banner is None
    now[0] = 1100.0  # mid fade-out
    r, g, b, a = _pixel_at(stack, rect.x() + 1, rect.y() + 1)
    assert a > 0 and max(abs(r - 0x5B), abs(g - 0x5F), abs(b - 0x66)) <= 3
    # Same content: only antialiased edges may flip at partial opacity (about
    # 15 %); the genre in its place differs on more pixels than 一時停止 has.
    assert len(white(_render(stack)) ^ shown) <= len(shown) // 4
    stack.add_line([{"text": "はい。"}], "Yes.")
    now[0] = 1150.0  # 50 ms into the 300 ms tint fade back to vermilion
    r, g, b, a = _pixel_at(stack, rect.x() + 1, rect.bottom() - 1)
    assert a > 0 and abs(r - 0x5B) < abs(r - BANNER_RED[0])


def _render(stack):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage, QRegion
    from PySide6.QtWidgets import QWidget

    image = QImage(stack.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    stack.render(image, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return image


def _pixel_at(stack, x, y):
    c = _render(stack).pixelColor(x, y)
    return (c.red(), c.green(), c.blue(), c.alpha())


def _paper_at(stack, x, y):
    return _pixel_at(stack, x, y) == PAPER


def test_plates_reshape_over_time_unless_reduce_motion(make_stack):
    """A new, wider line: with motion the paper strip starts at the old
    line's width and reaches the new width after the 350 ms reshape; with
    reduced motion it is there at once. The same reshape runs when the line
    arrives while the strip is still fading out after the last line left.
    Turning reduced motion on while the plates and banner fade in lands them
    at full opacity at once.

    Seeded defects: snap plates always (motion lost), ignore
    set_reduce_motion, treat a strip still fading out as hidden (its
    opacity target is 0, so it jumps to the new width), or leave in-flight
    fade-ins running when it turns on -> red."""
    now = [0.0]
    for reduced in (False, True):
        now[0] = 0.0
        stack = make_stack(clock=lambda: now[0])
        stack.set_reduce_motion(reduced)
        stack.add_line([{"text": "はい。"}], "Yes.")
        now[0] = 2000.0
        stack.add_line(SEGS_B, "Come on, this is what the town is famous for.")
        plate = stack.frame().jp_plate
        probe = (plate.right() - 4, plate.center().y())
        assert _paper_at(stack, *probe) is reduced
        now[0] = 2400.0
        assert _paper_at(stack, *probe)
    now[0] = 0.0
    stack = make_stack(clock=lambda: now[0])
    only = stack.add_line([{"text": "はい。"}], "Yes.", hold_ms=60000)
    now[0] = 2000.0
    stack.remove_line(only)          # its hold ended: the strip fades out
    now[0] = 2100.0
    stack.add_line(SEGS_B, "Come on, this is what the town is famous for.")
    plate = stack.frame().jp_plate
    probe = (plate.right() - 4, plate.center().y())
    assert _pixel_at(stack, *probe)[3] == 0   # still the short line's width
    now[0] = 2500.0
    assert _paper_at(stack, *probe)
    now[0] = 0.0
    stack = make_stack(clock=lambda: now[0])
    stack.add_line(SEGS_A, "Wait a moment.")
    now[0] = 60.0    # the 160 ms fade-in is in flight
    frame = stack.frame()
    probe = (frame.jp_plate.right() - 4, frame.jp_plate.center().y())
    assert not _paper_at(stack, *probe)
    stack.set_reduce_motion(True)
    assert _paper_at(stack, *probe)
    assert _pixel_at(stack, frame.banner.x() + 1, frame.banner.y() + 1) == BANNER_RED


def test_last_line_expiry_fades_out_plates_and_banner_unless_paused(make_stack, qapp):
    """When the only line's hold time ends, the plates and banner go with
    it; while paused the banner stays.

    Seeded defect: keep the plates after the last line -> red."""
    from PySide6.QtTest import QTest

    for paused in (False, True):
        stack = make_stack({"hold_ms": 500, "fade_out_ms": 0})
        if paused:
            stack.set_banner("一時停止", "Paused", True)
        stack.add_line(SEGS_A, "Wait a moment.", hold_ms=50, instant=True)
        QTest.qWait(150)
        frame = stack.frame()
        assert frame.lines == () and frame.jp_plate is None
        assert (frame.banner is not None) is paused


def test_mode_switch_gives_the_line_on_screen_a_fresh_full_hold(make_stack, qapp):
    """Switching Watch/Learn re-lays out the line on screen and restarts its
    hold_ms from the switch: it is still there after its first countdown
    would have ended, and it leaves once the fresh one ends.

    Seeded defect: leave the hold timer running from the line's arrival in
    apply_style -> the line vanishes soon after the switch -> red."""
    from PySide6.QtTest import QTest

    style = {**STYLE_DEFAULTS, "en_font": "BIZ UDPGothic", "hold_ms": 1000}
    stack = make_stack({**style, "mode": "watch"})
    line = stack.add_line(SEGS_A, "Wait a moment.", hold_ms=1000, instant=True)
    QTest.qWait(600)
    stack.apply_style({**style, "mode": "learn"})
    QTest.qWait(600)   # 1200 ms since it arrived: past its first countdown
    assert [box.id for box in stack.frame().lines] == [line]
    QTest.qWait(600)   # 1200 ms since the switch: past the fresh one
    assert stack.frame().lines == ()


def test_fading_lines_rise_with_the_stack_when_a_new_line_arrives(make_stack):
    """A line whose hold time ended fades where it is; when a new line
    arrives during that fade it rises with the stack by the new line's
    height (as lines pushed past max_lines do), so the rising echo never
    slides over it.

    Seeded defect: rise only the lines evicted by this add -> the fading
    line stays put and nothing is painted where it should have risen ->
    red."""
    now = [0.0]
    stack = make_stack({"fade_out_ms": 2000}, clock=lambda: now[0])
    first = stack.add_line(SEGS_A, "Wait a moment.", hold_ms=60000)
    now[0] = 1000.0
    stack.add_line(SEGS_B, "Come on, this is what the town is famous for.", hold_ms=60000)
    now[0] = 2000.0
    fading = stack.frame().lines[0].rect
    stack.remove_line(first)   # what its hold timer does
    now[0] = 2100.0
    stack.add_line([{"text": "はい。"}], "Yes.", hold_ms=60000)
    rise = stack.frame().lines[-1].rect.height()
    risen = fading.translated(0, -rise)
    assert not risen.intersects(stack.frame().lines[0].rect)  # clear of the echo
    now[0] = 2600.0            # the glide is done, the fade is not
    image = _render(stack)
    assert any(image.pixelColor(x, y).alpha() > 0
               for y in range(risen.top() + 2, risen.bottom() - 1)
               for x in range(risen.left() + BANNER_WIDTH, risen.right()))


def test_sample_shows_and_hides_without_landing_live_motion(make_stack):
    """The persistent sample (placement, the Settings live preview) lands at
    once, but showing or hiding it lands nothing else: a subtitle wiping in
    keeps wiping, and a line fading out after its hold keeps fading. On an
    empty stack the strip, bar and banner land and leave with the sample:
    paper behind it at once, nothing left 1 ms after it is hidden.

    Seeded defects: retarget without motion for an instant add or an
    unfaded removal -> the wipe lands at once and the fading line vanishes
    -> red. Move the plates and banner with motion when the instant sample
    is the newest line, or when hiding it leaves no line -> no paper behind
    the sample at first, an empty strip fading out after -> red."""
    now = [0.0]
    stack = make_stack({"fade_out_ms": 2000}, clock=lambda: now[0])
    leaving = stack.add_line(SEGS_A, "Wait a moment.", hold_ms=60000)
    now[0] = 1000.0
    stack.remove_line(leaving)   # its hold ended: it fades out over 2 s
    stack.add_line(SEGS_B, "Come on, this is what the town is famous for.", hold_ms=60000)
    now[0] = 1100.0              # the new line is wiping in
    sample = stack.add_line([{"text": "サンプル"}], "Sample", instant=True)
    stack.remove_line(sample, fade=False)
    newest = stack.frame().lines[-1].rect
    en = stack.frame().lines[-1].blocks["en"]
    tail = (en.left() + en.width() * 7 // 10, en.right())   # the English's last 30 %

    def bright(image):   # English text over the dark bar, in the tail
        return sum(1 for y in range(en.top(), en.bottom() + 1) for x in range(*tail)
                   if (c := image.pixelColor(x, y)).alpha() == 255
                   and min(c.red(), c.green(), c.blue()) > 200)

    assert bright(_render(stack)) == 0
    now[0] = 1500.0              # the wipe and the rise are done, the fade is not
    image = _render(stack)
    assert bright(image) > 0
    assert any(image.pixelColor(x, y).alpha() > 0
               for y in range(0, newest.top())
               for x in range(newest.left() + BANNER_WIDTH, newest.right()))

    now[0] = 0.0
    stack = make_stack(clock=lambda: now[0])
    sample = stack.add_line(SEGS_A, "Wait a moment.", instant=True)
    frame = stack.frame()
    plate = frame.jp_plate
    assert _paper_at(stack, plate.right() - 4, plate.center().y())
    assert _pixel_at(stack, frame.banner.x() + 1, frame.banner.y() + 1) == BANNER_RED
    now[0] = 1000.0
    stack.remove_line(sample, fade=False)
    now[0] = 1001.0
    image = _render(stack)
    assert not any(image.pixelColor(x, y).alpha() > 0
                   for r in (frame.jp_plate, frame.en_plate, frame.banner)
                   for y in range(r.top(), r.bottom() + 1)
                   for x in range(r.left(), r.right() + 1))


TRANSLATOR_LINE = "The translator isn't responding. Restarting it\u2026"
AUDIO_LINE = "Speakers was disconnected."


def test_notices_sit_above_the_newest_lines_banner_with_the_echo_above_them(make_stack):
    """Decisions 6 and 40: a notice entry is the 38 px お知らせ banner plus
    its bar; the bottom one sits 8 px above the newest line's banner where
    a long series name grows it past the line; two notices stack
    translator over audio, 8 px apart; the echo line moves above the top
    notice, 8 px clear. Left: every entry starts at the anchor.

    Seeded defects: put the notices on top of the echo, ignore the
    banner's rise, drop a gap, or reverse the kinds -> red."""
    stack = make_stack()
    stack.set_banner("薬屋のひとりごと", "The Apothecary Diaries", False)
    stack.add_line(SEGS_A, "Wait a moment.", instant=True)
    stack.add_line([{"text": "はい。"}], "Yes.", instant=True)
    stack.set_notices([("translator", TRANSLATOR_LINE, False), ("audio", AUDIO_LINE, False)])
    frame = stack.frame()
    echo, newest = frame.lines
    top, low = frame.notices
    assert (top.kind, low.kind) == ("translator", "audio")
    assert (top.text, low.text) == (TRANSLATOR_LINE, AUDIO_LINE)
    assert frame.banner.height() > newest.rect.height()      # the long name rises
    assert low.rect.bottom() + 1 + 8 == frame.banner.top()
    assert top.rect.bottom() + 1 + 8 == low.rect.top()
    assert echo.rect.bottom() + 1 + 8 == top.rect.top()
    for notice in (top, low):
        assert notice.rect.x() == 100 and notice.banner.width() == BANNER_WIDTH
        assert notice.bar.x() == 100 + BANNER_WIDTH and notice.bar.height() == 50
        assert notice.banner.bottom() == notice.bar.bottom() == notice.rect.bottom()
        assert notice.banner.height() > notice.bar.height()  # お知らせ rises past its bar


def test_notice_takes_the_newest_line_position_without_lines_and_sits_above_the_paused_banner(make_stack):
    """Decision 6: with no subtitle on screen the notice sits at the
    newest-line position (bottom on the anchor); while paused it sits 8 px
    above the paused banner, so the pause stays visible. Center: the
    notice is centered on the anchor.

    Seeded defects: hide notices when no line is on screen, or draw them
    over the paused banner -> red."""
    stack = make_stack()
    stack.set_notices([("audio", AUDIO_LINE, False)])
    notice = stack.frame().notices[0]
    assert notice.rect.x() == 100 and notice.rect.bottom() + 1 == 950
    stack.set_banner("一時停止", "Paused", True)
    frame = stack.frame()
    assert frame.banner.bottom() + 1 == 950
    assert frame.notices[0].rect.bottom() + 1 + 8 == frame.banner.top()
    center = make_stack(style={"alignment": "center"}, anchor_x=960)
    center.set_notices([("audio", AUDIO_LINE, False)])
    rect = center.frame().notices[0].rect
    assert abs(rect.x() + rect.width() / 2 - 960) <= 1


def test_notice_paints_amber_with_white_text_and_the_recovery_in_ink(make_stack):
    """Decisions 5 and 6: the problem line has the amber #e0a030 banner with
    the お知らせ label in ink and white text on the dark bar (whatever the
    subtitle text color); the recovery line has the ink #1f2421 banner with
    a #f3eee3 label. A long line is elided with … inside the line cap,
    never wrapped.

    Seeded defects: the subtitle banner's tint, the label in white, the
    text in the subtitle text color, or an unelided line -> red."""
    stack = make_stack(style={"text_color": "#ffe066"}, cap=600)
    stack.set_reduce_motion(True)
    stack.set_notices([("translator", TRANSLATOR_LINE + " " + TRANSLATOR_LINE, False)])
    notice = stack.frame().notices[0]
    assert notice.text.endswith("\u2026") and len(notice.text) < 2 * len(TRANSLATOR_LINE)
    assert notice.rect.width() <= 600
    image = _render(stack)

    def region(rect):
        return [image.pixelColor(x, y) for y in range(rect.top(), rect.bottom() + 1)
                for x in range(rect.left(), rect.right() + 1)]

    banner, bar = notice.banner, notice.bar
    assert _pixel_at(stack, banner.x() + 1, banner.y() + 1) == (0xE0, 0xA0, 0x30, 255)
    assert any(c.alpha() == 255 and c.red() < 0x60 for c in region(banner))   # ink label
    white = [c for c in region(bar) if c.alpha() and min(c.red(), c.green(), c.blue()) > 240]
    assert len(white) > 100
    stack.set_notices([("translator", "Translations are back.", True)])
    recovery = stack.frame().notices[0].banner
    image = _render(stack)
    assert _pixel_at(stack, recovery.x() + 1, recovery.y() + 1) == (0x1F, 0x24, 0x21, 255)
    assert any((c.red(), c.green(), c.blue(), c.alpha()) == (0xF3, 0xEE, 0xE3, 255)
               for c in region(recovery))


def test_notice_fades_in_and_out_unless_reduce_motion(make_stack):
    """Decision 25: a notice fades in over fade_in_ms and, when it goes,
    fades out where it is over fade_out_ms; with reduced motion it appears
    and goes at once.

    Seeded defects: no ghost for a removed notice, or fades under reduced
    motion -> red."""
    now = [0.0]
    stack = make_stack(clock=lambda: now[0])
    stack.set_notices([("audio", AUDIO_LINE, False)])
    box = stack.frame().notices[0].banner
    x, y = box.x() + 1, box.y() + 1
    assert _pixel_at(stack, x, y)[3] == 0
    now[0] = 1000.0
    assert _pixel_at(stack, x, y) == (0xE0, 0xA0, 0x30, 255)
    stack.set_notices([])
    now[0] = 1000.0 + STYLE_DEFAULTS["fade_out_ms"] / 2
    assert 0 < _pixel_at(stack, x, y)[3] < 255
    now[0] = 2000.0
    assert _pixel_at(stack, x, y)[3] == 0
    stack.set_reduce_motion(True)
    stack.set_notices([("audio", AUDIO_LINE, False)])
    assert _pixel_at(stack, x, y) == (0xE0, 0xA0, 0x30, 255)
    stack.set_notices([])
    assert _pixel_at(stack, x, y)[3] == 0
