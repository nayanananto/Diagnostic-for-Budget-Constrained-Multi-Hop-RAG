"""Lightweight evidence verification for the MVP."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import RetrievalRun
from .text import has_negation, lexical_overlap


@dataclass
class VerificationResult:
    support_score: float
    has_conflict: bool
    decision: str
    rationale: str


@dataclass
class EvidenceVerifier:
    support_threshold: float = 0.08

    def verify(self, query: str, run: RetrievalRun) -> VerificationResult:
        if not run.hits:
            return VerificationResult(0.0, False, "abstain", "no evidence retrieved")

        overlaps = [lexical_overlap(query, hit.text) for hit in run.hits]
        support = max(overlaps) if overlaps else 0.0
        negated = [has_negation(hit.text) for hit in run.hits[:5]]
        has_conflict = any(negated) and not all(negated)
        if has_conflict:
            return VerificationResult(support, True, "conflict", "retrieved evidence contains negated and non-negated claims")
        if support < self.support_threshold:
            return VerificationResult(support, False, "abstain", "retrieved evidence has low lexical support")
        return VerificationResult(support, False, "answer", "retrieved evidence appears sufficient")

