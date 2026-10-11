"""Terminal front end: a small listening mascot, a live status line, styled transcript lines.

Standard library only. On a terminal the bottom line is redrawn ten times a second
(mascot face, mic level, counters) while transcript lines and warnings scroll above
it. When stdout is not a terminal, or NO_COLOR is set, everything degrades to plain
text with no status line, so logs and pipes stay clean.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass

CORAL = "38;5;209"
GREY = "38;5;245"
GREEN = "38;5;78"
YELLOW = "38;5;221"
RED = "38;5;203"
BOLD = "1"
DIM = "2"

SPRITE = (" ╭─────╮ ", "(│ ◕ ◕ │)", " ╰──◡──╯ ")
SPRITE_ASCII = (" .-----. ", "(| o o |)", " '--v--' ")

FACES = {
    "loading": ("( ◔_◔ )", "( ◔_◔ )"),
    "listening": ("( ◕‿◕ )", "( -‿- )"),
    "speech": ("( ◕o◕ )", "( ^o^ )"),
    "thinking": ("( ◔_◔ )", "( ◔.◔ )"),
    "worried": ("( ◕﹏◕ )", "( ◕﹏◕ )"),
    "stopping": ("( -_- )", "( -_- )"),
}
FACES_ASCII = {
    "loading": "( o_o )", "listening": "( ^_^ )", "speech": "( ^o^ )",
    "thinking": "( o.o )", "worried": "( o_O )", "stopping": "( -_- )",
}
LABELS = {
    "loading": "waking up", "listening": "listening", "speech": "hearing you",
    "thinking": "transcribing", "worried": "can't hear you well", "stopping": "wrapping up",
}
BAR_FLOOR_DB = -60.0
BAR_CEIL_DB = -20.0


def dbfs(x: float) -> float:
    return -120.0 if x <= 0 else 20 * math.log10(x)


def level_bar(db: float, width: int = 10, unicode_ok: bool = True) -> str:
    frac = min(1.0, max(0.0, (db - BAR_FLOOR_DB) / (BAR_CEIL_DB - BAR_FLOOR_DB)))
    filled = round(frac * width)
    full, empty = ("█", "░") if unicode_ok else ("#", ".")
    return full * filled + empty * (width - filled)


def face_state(phase: str, speaking: bool, busy: bool, worried: bool) -> str:
    if phase in ("loading", "stopping"):
        return phase
    if speaking:
        return "speech"
    if busy:
        return "thinking"
    return "worried" if worried else "listening"


def face(state: str, now: float, unicode_ok: bool = True) -> str:
    if not unicode_ok:
        return FACES_ASCII[state]
    open_eyes, alt = FACES[state]
    if state == "listening":
        return alt if (now % 4.0) < 0.15 else open_eyes
    if state == "speech":
        return alt if int(now * 4) % 2 else open_eyes
    if state == "thinking":
        return alt if int(now * 2) % 2 else open_eyes
    return open_eyes


def fmt_clock(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class Status:
    phase: str = "loading"     # loading | running | stopping
    detail: str = ""           # shown after the label while loading
    speaking: bool = False
    busy: bool = False
    worried: bool = False
    level_db: float = -120.0
    kept: int = 0
    dropped: int = 0
    queued: int = 0
    started: float | None = None


class Console:
    def __init__(self, stream=None, color: bool | None = None, live: bool | None = None) -> None:
        self._out = stream or sys.stdout
        tty = bool(getattr(self._out, "isatty", lambda: False)())
        no_color = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
        self.color = (tty and not no_color) if color is None else color
        self.live = (tty and not no_color) if live is None else live
        self.unicode_ok = (getattr(self._out, "encoding", "") or "").lower().replace("-", "").startswith("utf")
        self.status = Status()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._drawn = False
        self._swap: tuple[logging.Handler, logging.Handler] | None = None

    # --- styling ---------------------------------------------------------------
    def style(self, text: str, *codes: str) -> str:
        if not self.color or not codes:
            return text
        return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"

    # --- permanent lines ---------------------------------------------------------
    def banner(self, title: str, rows: list[str]) -> None:
        sprite = SPRITE if self.unicode_ok else SPRITE_ASCII
        lines = []
        for i in range(max(len(sprite), len(rows) + 1)):
            art = sprite[i] if i < len(sprite) else " " * len(sprite[0])
            if i == 0:
                text = title
            else:
                text = self.style(rows[i - 1], GREY) if i - 1 < len(rows) else ""
            lines.append(f"{self.style(art, CORAL)}  {text}")
        self._emit("\n" + "\n".join(lines) + "\n")

    def note(self, text: str) -> None:
        self._emit(self.style(f"  {text}", GREY))

    def alert(self, level: int, text: str) -> None:
        if level >= logging.ERROR:
            mark, code = "✗" if self.unicode_ok else "x", RED
        else:
            mark, code = "!", YELLOW
        body = text.replace("\n", "\n    ")
        self._emit(self.style(f"  {mark} {body}", code))

    def transcript(self, stamp: str, text: str) -> None:
        self._emit(f"{self.style(stamp, GREY)}  {self.style(text, BOLD)}")

    def dropped(self, stamp: str, text: str, reason: str) -> None:
        shown = f'"{text}"' if text else "(nothing recognised)"
        self._emit(self.style(f"{stamp}  ~ {shown}  [{reason}]", GREY, DIM))

    def _emit(self, text: str) -> None:
        with self._lock:
            self._clear_status()
            self._out.write(text + "\n")
            self._draw_status()
            self._out.flush()

    # --- live status line --------------------------------------------------------
    def update(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self.status, key, value)

    def render_status(self, now: float, width: int) -> str:
        s = self.status
        state = face_state(s.phase, s.speaking, s.busy, s.worried)
        mood = {"speech": GREEN, "worried": YELLOW, "stopping": GREY}.get(state, CORAL)
        label = LABELS[state] + (f" {s.detail}" if s.phase == "loading" and s.detail else "")
        if state == "thinking":
            label += "." * (1 + int(now * 3) % 3)
        parts = [self.style(face(state, now, self.unicode_ok), mood), self.style(label.ljust(20), BOLD)]
        if s.phase == "running" or s.phase == "stopping":
            bar_color = GREY
            if s.speaking:
                bar_color = GREEN if s.level_db > -35 else YELLOW if s.level_db > -42 else RED
            db = f"{s.level_db:4.0f} dB" if s.level_db > -119 else "  -- dB"
            parts.append(self.style(f"{level_bar(s.level_db, 10, self.unicode_ok)} {db}", bar_color))
            counts = f"{s.kept} heard · {s.dropped} dropped"
            if s.queued:
                counts += f" · {s.queued} waiting"
            parts.append(self.style(counts, GREY))
            if s.started is not None:
                parts.append(self.style(fmt_clock(now - s.started), GREY))
        line = "  ".join(parts)
        return line if not self.color else _truncate_ansi(line, width - 1)

    def _clear_status(self) -> None:
        if self.live and self._drawn:
            self._out.write("\r\x1b[2K")
            self._drawn = False

    def _draw_status(self) -> None:
        if self.live and self._thread is not None:
            width = shutil.get_terminal_size((80, 24)).columns
            self._out.write("\r\x1b[2K" + self.render_status(time.monotonic(), width))
            self._drawn = True

    def start(self) -> None:
        if not self.live or self._thread is not None:
            return
        self._out.write("\x1b[?25l")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ui", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                self._draw_status()
                self._out.flush()

    def stop(self) -> None:
        thread, self._thread = self._thread, None
        self._stop.set()
        if thread is not None:
            thread.join(timeout=1.0)
        with self._lock:
            self._clear_status()
            if self.live:
                self._out.write("\x1b[?25h")
            self._out.flush()
        self.detach_logging()

    # --- logging -----------------------------------------------------------------
    def attach_logging(self, handler: logging.Handler | None) -> None:
        """Swap the console log handler for one that prints above the status line."""
        if handler is None:
            return
        root = logging.getLogger()
        replacement = _AlertHandler(self)
        replacement.setLevel(handler.level)
        root.removeHandler(handler)
        root.addHandler(replacement)
        self._swap = (handler, replacement)

    def detach_logging(self) -> None:
        if self._swap is not None:
            original, replacement = self._swap
            root = logging.getLogger()
            root.removeHandler(replacement)
            root.addHandler(original)
        self._swap = None


class _AlertHandler(logging.Handler):
    def __init__(self, console: Console) -> None:
        super().__init__()
        self._console = console

    def emit(self, record: logging.LogRecord) -> None:
        self._console.alert(record.levelno, record.getMessage())


def _truncate_ansi(text: str, width: int) -> str:
    out, visible, i = [], 0, 0
    while i < len(text) and visible < width:
        if text[i] == "\x1b":
            j = text.find("m", i)
            out.append(text[i : j + 1])
            i = j + 1
            continue
        out.append(text[i])
        visible += 1
        i += 1
    return "".join(out) + "\x1b[0m"
