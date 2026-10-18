"""Conservative, context-aware STT vocabulary correction.

Applied after Parakeet and before routing/LLM. Not an ASR model change.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_VOCAB_PATH = Path(__file__).with_name("stt_vocab.json")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ARITHMETIC_WORDS = re.compile(r"^(?P<prefix>what is )?(?P<left>[a-z -]+?)\s+(?P<op>times|multiplied by|plus|minus|divided by)\s+(?P<right>[a-z -?]+)$", re.I)
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


@dataclass(frozen=True)
class VocabTerm:
    canonical: str
    aliases: tuple[str, ...]
    context: tuple[str, ...]
    min_ratio: float


@dataclass(frozen=True)
class Correction:
    term: str
    original_span: str
    similarity: float
    source: str


def load_vocab(path: Path | None = None) -> list[VocabTerm]:
    data = json.loads((path or _VOCAB_PATH).read_text(encoding="utf-8"))
    terms: list[VocabTerm] = []
    for item in data.get("terms") or []:
        canonical = str(item.get("canonical") or "").strip()
        if not canonical:
            continue
        aliases = tuple(
            " ".join(str(alias).lower().split())
            for alias in (item.get("aliases") or [])
            if str(alias).strip()
        )
        context = tuple(str(word).lower() for word in (item.get("context") or []) if str(word).strip())
        min_ratio = float(item.get("min_ratio") or 0.8)
        terms.append(VocabTerm(canonical=canonical, aliases=aliases, context=context, min_ratio=min_ratio))
    return terms


def _norm(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def _has_context(sentence_norm: str, term: VocabTerm, skip_span_norm: str) -> bool:
    remainder = sentence_norm
    if skip_span_norm:
        remainder = remainder.replace(skip_span_norm, " ", 1)
    padded = f" {remainder} "
    canonical_norm = _norm(term.canonical)
    if canonical_norm and f" {canonical_norm} " in padded:
        return True
    for hint in term.context:
        if hint and hint in remainder:
            return True
    return False


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _windows(words: list[re.Match[str]], max_n: int = 3) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for n in range(1, max_n + 1):
        for i in range(0, len(words) - n + 1):
            start = words[i].start()
            end = words[i + n - 1].end()
            out.append((start, end, " ".join(w.group(0) for w in words[i : i + n])))
    return out


def apply_stt_vocab(text: str, terms: list[VocabTerm] | None = None) -> tuple[str, list[Correction]]:
    """Return possibly corrected text and the list of applied corrections."""

    original = text or ""
    if not original.strip():
        return original, []
    vocab = terms if terms is not None else load_vocab()
    if not vocab:
        return original, []

    words = list(_WORD_RE.finditer(original))
    if not words:
        return original, []
    sentence_norm = _norm(original)
    used: set[tuple[int, int]] = set()
    replacements: list[tuple[int, int, str, Correction]] = []

    for start, end, span in sorted(_windows(words), key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < used_end and end > used_start for used_start, used_end in used):
            continue
        span_norm = _norm(span)
        if not span_norm:
            continue
        best: tuple[VocabTerm, float, str] | None = None
        for term in vocab:
            canonical_norm = _norm(term.canonical)
            if span_norm == canonical_norm:
                continue
            if canonical_norm and canonical_norm in span_norm.split():
                continue
            source = ""
            score = 0.0
            if span_norm in {_norm(alias) for alias in term.aliases}:
                source = "alias"
                score = 1.0
            else:
                score = _similarity(span_norm.replace(" ", ""), canonical_norm.replace(" ", ""))
                if score >= term.min_ratio:
                    source = "fuzzy"
                else:
                    alias_score = max(
                        (_similarity(span_norm, _norm(alias)) for alias in term.aliases),
                        default=0.0,
                    )
                    if alias_score >= max(0.86, term.min_ratio):
                        source = "alias-fuzzy"
                        score = alias_score
            if not source:
                continue
            if not _has_context(sentence_norm, term, span_norm):
                continue
            if best is None or score > best[1]:
                best = (term, score, source)
        if best is None:
            continue
        term, score, source = best
        used.add((start, end))
        replacements.append(
            (
                start,
                end,
                term.canonical,
                Correction(term=term.canonical, original_span=span, similarity=score, source=source),
            )
        )

    if not replacements:
        return original, []
    replacements.sort(key=lambda item: item[0])
    pieces: list[str] = []
    cursor = 0
    applied: list[Correction] = []
    for start, end, canonical, correction in replacements:
        pieces.append(original[cursor:start])
        pieces.append(canonical)
        applied.append(correction)
        cursor = end
    pieces.append(original[cursor:])
    return "".join(pieces), applied


def normalize_application_text(text: str) -> str:
    """Conservatively normalize spoken integers only inside an arithmetic utterance."""
    match = _ARITHMETIC_WORDS.match(" ".join((text or "").strip().rstrip("?").split()))
    if not match:
        return text
    def number(words: str) -> int | None:
        total = current = 0
        for word in words.lower().replace("-", " ").split():
            if word in _ONES: current += _ONES[word]
            elif word in _TENS: current += _TENS[word]
            elif word == "hundred" and current: current *= 100
            elif word == "and": continue
            else: return None
        return total + current if current or words.strip().lower() == "zero" else None
    left, right = number(match.group("left")), number(match.group("right"))
    if left is None or right is None:
        return text
    op = {"times": "times", "multiplied by": "times", "plus": "plus", "minus": "minus", "divided by": "divided by"}[match.group("op").lower()]
    return f"{match.group('prefix') or ''}{left} {op} {right}".strip()


def correction_log_fields(original: str, corrected: str, applied: list[Correction]) -> dict[str, Any]:
    return {
        "stt_vocab_applied": bool(applied),
        "stt_vocab_term": ",".join(item.term for item in applied) if applied else "",
        "stt_vocab_similarity": round(min((item.similarity for item in applied), default=0.0), 3),
        "stt_vocab_original": original,
        "stt_vocab_corrected": corrected,
    }
