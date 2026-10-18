"""Flush speakable phrases so TTS can start before the LLM finishes."""
from __future__ import annotations

import re

_END = re.compile(r"([.!?]+)(\s+|$)")
_CLAUSE = re.compile(r"[,;:—–]\s+")


def _nwords(s: str) -> int:
    return len((s or "").split())


class SentenceBuffer:
    def __init__(
        self,
        first_min_words: int = 5,
        first_max_words: int = 14,
        later_min_words: int = 8,
        later_max_words: int = 25,
    ):
        self.buf = ""
        self.first = True
        self.first_min_words = first_min_words
        self.first_max_words = first_max_words
        self.later_min_words = later_min_words
        self.later_max_words = later_max_words

    def push(self, piece: str) -> list[str]:
        if not piece:
            return []
        self.buf += piece
        out: list[str] = []
        while True:
            got = self._take_one()
            if not got:
                break
            out.append(got)
        return out

    def flush(self) -> str:
        text = _clean(self.buf)
        self.buf = ""
        self.first = False
        return text

    def _emit(self, end: int) -> str:
        chunk = _clean(self.buf[:end])
        self.buf = self.buf[end:]
        self.first = False
        return chunk

    def _take_one(self) -> str:
        if self.first:
            return self._take_first()
        return self._take_later()

    def _take_first(self) -> str:
        # Wait for a real sentence of 5–12 words before using commas.
        m = _END.search(self.buf)
        if m:
            n = _nwords(self.buf[: m.end()])
            if self.first_min_words <= n <= self.first_max_words + 2:
                return self._emit(m.end())
            if n > self.first_max_words + 2:
                end = self._clause_end(self.buf[: m.end()], 8, self.first_max_words)
                if end:
                    return self._emit(end)
                end = _complete_word_end(self.buf, self.first_max_words)
                if end:
                    return self._emit(end)
            return ""
        # Hard-split only after first_max+4 so a 13–16 word sentence can land on its period.
        if _nwords(self.buf) >= self.first_max_words + 4:
            end = self._clause_end(self.buf, 8, self.first_max_words + 2)
            if end:
                return self._emit(end)
            end = _complete_word_end(self.buf, self.first_max_words)
            if end:
                return self._emit(end)
        return ""

    def _take_later(self) -> str:
        m = _END.search(self.buf)
        if m:
            n = _nwords(self.buf[: m.end()])
            if self.later_min_words <= n <= self.later_max_words:
                return self._emit(m.end())
            if n > self.later_max_words:
                end = self._clause_end(self.buf[: m.end()], 10, self.later_max_words)
                if end:
                    return self._emit(end)
                end = _complete_word_end(self.buf, self.later_max_words)
                if end:
                    return self._emit(end)
            return ""
        if _nwords(self.buf) >= self.later_max_words:
            end = self._clause_end(self.buf, 10, self.later_max_words)
            if end:
                return self._emit(end)
            end = _complete_word_end(self.buf, self.later_max_words)
            if end:
                return self._emit(end)
        return ""

    def _clause_end(self, text: str, lo: int, hi: int) -> int:
        best = 0
        for m in _CLAUSE.finditer(text):
            n = _nwords(text[: m.end()])
            if lo <= n <= hi:
                best = m.end()
        return best


def _complete_word_end(buf: str, n: int) -> int:
    """Index after n complete words. 0 if the nth word is still growing."""
    raw = _split_n_words(buf, n)
    if _nwords(raw) < n or not raw:
        return 0
    if len(raw) < len(buf):
        nxt = buf[len(raw)]
        if not nxt.isspace() and nxt not in ".!?,;:—–":
            return 0
        return len(raw)
    if raw[-1].isalnum():
        return 0
    return len(raw)


def _split_n_words(buf: str, n: int) -> str:
    if n <= 0:
        return ""
    i = 0
    while i < len(buf) and buf[i].isspace():
        i += 1
    seen = 0
    while i < len(buf) and seen < n:
        while i < len(buf) and not buf[i].isspace():
            i += 1
        seen += 1
        if seen < n:
            while i < len(buf) and buf[i].isspace():
                i += 1
    return buf[:i]


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[\s,;:—–-]+", "", s)
    return s
