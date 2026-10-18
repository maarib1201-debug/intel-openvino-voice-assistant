"""Cheap sentence-level guard for unsupported factual specificity and causal chains."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .sentences import SentenceBuffer


_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_ERA = re.compile(r"\b(?:\d{1,2}(?:st|nd|rd|th) century|(?:18|19|20)\d0s)\b", re.IGNORECASE)
_INSTITUTION = re.compile(
    r"\b(?:university|institute|observatory|laborator(?:y|ies)|research center|"
    r"corporation|ministry|government agency|department)\b",
    re.IGNORECASE,
)
_ATTRIBUTION = re.compile(
    r"\b(?:invented|discovered|created|developed|founded|designed|introduced|"
    r"identified|established)\b(?:.{0,45}\b(?:by|at|in)\b)?",
    re.IGNORECASE,
)
_PERSON = re.compile(
    r"\b(?:Dr\.?\s+)?[A-Z][a-z]{2,}(?:\s+(?:van|von|de|del))?\s+[A-Z][a-z]{2,}\b"
)
_PROPER_PHRASE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b")
_CERTAINTY = re.compile(
    r"\b(?:established (?:scientific |historical )?records?|factual correction|"
    r"definitively|specifically|proven fact|without (?:any )?doubt|exact(?:ly)?|guarantees?)\b",
    re.IGNORECASE,
)
_HISTORICAL_QUERY = re.compile(
    r"\b(?:who|when|what happened|invent(?:ed|or)?|discover(?:ed|y)?|incident|origin)\b",
    re.IGNORECASE,
)
_NAMED_ORG = re.compile(
    r"\b(?:U\.?S\.?\s+)?(?:Navy|Army|Air Force|Marines?|Police|Agency|Authority)"
    r"(?:\s+[A-Z]{2,}|\s+[A-Z][a-z]+){0,3}\b"
)
_ACRONYM_ORG = re.compile(r"\b(?:[A-Z]\.){2,}")
_NAMED_LOCATION = re.compile(r"\b(?:in|at|from)\s+[A-Z][A-Za-z]+,\s*[A-Z][A-Za-z]+\b")
_SPECULATIVE_DEFINITION = re.compile(
    r"^[A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,3}\s+(?:is|refers to)\s+"
    r"(?:an?\s+)?(?:specialized|advanced|proprietary|novel|experimental)\b",
    re.IGNORECASE,
)
_QUANTIFIED_DETAIL = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x|[- ]?bit|[- ]?byte|KB|MB|GB|TB|GHz|MHz|%|percent|per second|/s)\b",
    re.IGNORECASE,
)
_CONDITIONAL_GUARANTEE = re.compile(
    r"\beven if\b.{0,70}\b(?:safe|secure|protected|cannot be (?:read|accessed|stolen|broken))\b",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"\b(?:I(?:'m| am) not (?:sure|certain|familiar)|I (?:do not|don't) know|"
    r"I (?:cannot|can't) verify|not enough (?:reliable )?information|"
    r"may be fictional|appears to be fictional|does not appear to be real)\b",
    re.IGNORECASE,
)
_MECHANISM_QUERY = re.compile(
    r"\b(?:how (?:does|do|did|is|are|can)|why (?:does|do|did|is|are)|"
    r"explain how|what causes|what makes)\b",
    re.IGNORECASE,
)
_ENTITY_DEFINITION_QUERY = re.compile(
    r"\b(?i:what is|explain|tell me about)\s+(?i:the\s+)?"
    r"[A-Z][A-Za-z-]{4,}\s+[a-z][a-z-]{3,}",
)
_CAUSAL = re.compile(
    r"\b(?:because|therefore|thereby|by|through|which|then|caus(?:e|es|ed|ing)|"
    r"creat(?:e|es|ed|ing)|convert(?:s|ed|ing)?|draw(?:s|n|ing)?|"
    r"forc(?:e|es|ed|ing)|push(?:es|ed|ing)?|allow(?:s|ed|ing)?|"
    r"ensur(?:e|es|ed|ing)|result(?:s|ed|ing)?|lead(?:s|ing)? to|"
    r"rel(?:y|ies|ied|ying) on|calculat(?:e|es|ed|ing)|determin(?:e|es|ed|ing)|"
    r"triangulat(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)

FALLBACK_FIRST = "I'm not confident enough about that mechanism to give you a reliable answer."
FALLBACK_LATER = "I'm not confident enough about the more specific details to go further."
FALLBACK_FACTS_FIRST = "I'm not confident enough about those specific facts to give you a reliable answer."
FALLBACK_FACTS_LATER = "I'm not confident enough about those additional details to go further."


def _normalized(text: str) -> str:
    return " ".join((text or "").lower().split())


@dataclass(frozen=True)
class GuardDecision:
    text: str
    triggered: bool = False
    reason: str | None = None
    score: int = 0


class FactualRiskGuard:
    """Reviews complete sentences; it does not call a model or external service."""

    def __init__(
        self,
        user: str,
        *,
        prior_context: str = "",
        enabled: bool = True,
        grounded: bool = False,
        specificity_threshold: int = 2,
        causal_threshold: int = 3,
    ) -> None:
        self.enabled = bool(enabled) and not grounded
        self.context = _normalized(f"{prior_context} {user}")
        self.mechanism_query = bool(_MECHANISM_QUERY.search(user or ""))
        self.historical_query = bool(_HISTORICAL_QUERY.search(user or ""))
        self.entity_definition_query = bool(_ENTITY_DEFINITION_QUERY.search(user or ""))
        self.needs_extended_prefix = (
            self.mechanism_query or self.historical_query or self.entity_definition_query
        )
        self.specificity_threshold = max(1, int(specificity_threshold))
        self.causal_threshold = max(2, int(causal_threshold))
        self.safe_sentences = 0
        self.triggered = False
        self.reason: str | None = None

    def inspect(self, sentence: str) -> GuardDecision:
        text = " ".join((sentence or "").split()).strip()
        if not text or not self.enabled:
            return GuardDecision(text=text)
        if self.triggered:
            return GuardDecision(text="", triggered=True, reason=self.reason)
        if _UNCERTAINTY.search(text):
            self.safe_sentences += 1
            return GuardDecision(text=text)

        specificity = self._specificity_score(text)
        if specificity >= self.specificity_threshold:
            return self._trigger("unsupported-specificity", specificity)

        causal = self._causal_score(text) if self.mechanism_query else 0
        if causal >= self.causal_threshold and len(text.split()) >= 12:
            return self._trigger("dense-causal-chain", causal)

        self.safe_sentences += 1
        return GuardDecision(text=text, score=max(specificity, causal))

    def _trigger(self, reason: str, score: int) -> GuardDecision:
        self.triggered = True
        self.reason = reason
        if reason == "unsupported-specificity":
            fallback = FALLBACK_FACTS_LATER if self.safe_sentences else FALLBACK_FACTS_FIRST
        else:
            fallback = FALLBACK_LATER if self.safe_sentences else FALLBACK_FIRST
        return GuardDecision(text=fallback, triggered=True, reason=reason, score=score)

    def _specificity_score(self, text: str) -> int:
        lower = _normalized(text)
        score = 0
        years = _YEAR.findall(text)
        if any(year not in self.context for year in years):
            score += 2
        eras = _ERA.findall(text)
        if any(_normalized(era) not in self.context for era in eras):
            score += 2

        institution = bool(_INSTITUTION.search(text))
        attribution = bool(_ATTRIBUTION.search(text))
        person = any(_normalized(name) not in self.context for name in _PERSON.findall(text))
        proper = any(_normalized(name) not in self.context for name in _PROPER_PHRASE.findall(text))
        certainty = bool(_CERTAINTY.search(text))

        if institution and lower not in self.context:
            score += 1
        if attribution and (institution or person or proper):
            score += 1
        if person and attribution and (self.historical_query or institution or bool(years) or bool(eras)):
            score += 1
        if certainty:
            score += 1
        if proper and certainty:
            score += 1
        if _CONDITIONAL_GUARANTEE.search(text):
            score += 2
        if self.historical_query and (
            _NAMED_ORG.search(text) or _NAMED_LOCATION.search(text) or _ACRONYM_ORG.search(text)
        ):
            score += 2
        if self.historical_query and attribution:
            score += 2
        if _SPECULATIVE_DEFINITION.search(text):
            score += 2
        if len(_QUANTIFIED_DETAIL.findall(text)) >= 2:
            score += 2
        return score

    @staticmethod
    def _causal_score(text: str) -> int:
        markers = len(_CAUSAL.findall(text))
        # Multiple clauses are where the tiny model most often invents hidden steps.
        if text.count(",") >= 2:
            markers += 1
        return markers


def guard_complete_text(
    user: str,
    text: str,
    *,
    prior_context: str = "",
    specificity_threshold: int = 2,
    causal_threshold: int = 3,
) -> tuple[str, str | None]:
    """Apply the same policy to saved benchmark output."""

    guard = FactualRiskGuard(
        user,
        prior_context=prior_context,
        specificity_threshold=specificity_threshold,
        causal_threshold=causal_threshold,
    )
    buffer = (
        SentenceBuffer(first_max_words=18, later_max_words=28)
        if guard.needs_extended_prefix
        else SentenceBuffer()
    )
    sentences = buffer.push(text or "")
    tail = buffer.flush()
    if tail:
        sentences.append(tail)
    output: list[str] = []
    for sentence in sentences:
        decision = guard.inspect(sentence)
        if decision.text:
            output.append(decision.text)
        if decision.triggered:
            break
    if not sentences and text:
        decision = guard.inspect(text)
        if decision.text:
            output.append(decision.text)
    return " ".join(output).strip(), guard.reason
