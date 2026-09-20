"""Short rolling memory of recent transcript text used as Whisper's prompt.

Kept deliberately small: a long prompt makes Whisper loop and hallucinate.
"""

from __future__ import annotations

from collections import deque


class ContextWindow:
    def __init__(self, enabled: bool = True, max_chars: int = 400) -> None:
        self.enabled = enabled
        self.max_chars = max_chars
        self._segments: deque[str] = deque()

    def add(self, text: str) -> None:
        if not self.enabled:
            return
        text = (text or "").strip()
        if not text:
            return
        self._segments.append(text)
        while len(self._segments) > 1 and len(" ".join(self._segments)) > self.max_chars:
            self._segments.popleft()

    def get_prompt(self) -> str:
        if not self.enabled or not self._segments:
            return ""
        prompt = " ".join(self._segments)
        if len(prompt) > self.max_chars:
            prompt = prompt[-self.max_chars :].lstrip()
        return prompt

    def clear(self) -> None:
        self._segments.clear()
