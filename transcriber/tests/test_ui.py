import io
import logging

from transcriber.ui import Console, face, face_state, fmt_clock, level_bar


class Capture(io.StringIO):
    def __init__(self, encoding):
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self):
        return self._encoding


def plain_console(unicode_ok=True):
    out = Capture("utf-8" if unicode_ok else "ascii")
    return Console(stream=out, color=False, live=False), out


def test_level_bar_spans_floor_to_ceiling():
    assert level_bar(-70, 10) == "░" * 10
    assert level_bar(-20, 10) == "█" * 10
    assert level_bar(-40, 10).count("█") == 5
    assert level_bar(-40, 10, unicode_ok=False) == "#####....."


def test_face_state_priority():
    assert face_state("loading", True, True, True) == "loading"
    assert face_state("stopping", True, False, False) == "stopping"
    assert face_state("running", True, True, True) == "speech"
    assert face_state("running", False, True, True) == "thinking"
    assert face_state("running", False, False, True) == "worried"
    assert face_state("running", False, False, False) == "listening"


def test_face_blinks_only_briefly_and_has_ascii_fallback():
    assert face("listening", 1.0) != face("listening", 4.05)
    assert face("listening", 1.0) == face("listening", 2.0)
    assert face("speech", 0.0, unicode_ok=False).isascii()


def test_fmt_clock():
    assert fmt_clock(75) == "1:15"
    assert fmt_clock(3725) == "1:02:05"


def test_banner_and_lines_are_plain_without_a_terminal():
    console, out = plain_console()
    console.banner("Emote  v1", ["small.en", "session x"])
    console.transcript("00:00:01.000", "hello there")
    console.dropped("00:00:02.000", "you", "no_speech_prob 0.63 > 0.6")
    text = out.getvalue()
    assert "\x1b" not in text
    assert "╭─────╮" in text and "Emote  v1" in text and "small.en" in text
    assert "00:00:01.000  hello there" in text
    assert '~ "you"  [no_speech_prob 0.63 > 0.6]' in text


def test_ascii_banner_when_terminal_is_not_utf8():
    console, out = plain_console(unicode_ok=False)
    console.banner("Emote", [])
    assert out.getvalue().isascii()


def test_status_line_reports_state_and_counts():
    console, _ = plain_console()
    console.update(phase="running", speaking=True, level_db=-30.0, kept=3, dropped=1, queued=2, started=0.0)
    line = console.render_status(65.0, 120)
    assert "hearing you" in line and "3 heard · 1 dropped · 2 waiting" in line and "1:05" in line
    console.update(speaking=False, worried=True)
    assert "can't hear you well" in console.render_status(65.0, 120)


def test_logging_goes_through_the_console_and_is_restored():
    console, out = plain_console()
    root = logging.getLogger()
    original = logging.StreamHandler(io.StringIO())
    original.setLevel(logging.WARNING)
    root.addHandler(original)
    try:
        console.attach_logging(original)
        logging.getLogger("t").warning("mic is quiet")
        logging.getLogger("t").info("not shown")
        assert "! mic is quiet" in out.getvalue() and "not shown" not in out.getvalue()
        console.detach_logging()
        assert original in root.handlers
    finally:
        root.removeHandler(original)
