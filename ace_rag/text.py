"""Small text utilities used by the offline MVP.

These are intentionally conservative. Later experiments can replace sentence
splitting, entity extraction, and claim extraction with stronger NLP/LLM tools
without changing the rest of the pipeline.
"""

from __future__ import annotations

import re
import string
from collections import Counter


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+|$)){1,5}")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words)


def split_sentences(text: str) -> list[tuple[str, tuple[int, int]]]:
    sentences: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    for part in _SENTENCE_RE.split(text.strip()):
        sent = part.strip()
        if not sent:
            continue
        start = text.find(sent, cursor)
        if start < 0:
            start = cursor
        end = start + len(sent)
        sentences.append((sent, (start, end)))
        cursor = end
    if not sentences and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        sentences.append((stripped, (start, start + len(stripped))))
    return sentences


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        ent = " ".join(match.group(0).split()).strip()
        if len(ent) <= 2 or ent.lower() in {"the", "in", "on"}:
            continue
        if ent not in seen:
            entities.append(ent)
            seen.add(ent)
    return entities


def cosine_from_counters(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def lexical_overlap(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def has_negation(text: str) -> bool:
    toks = set(tokenize(text))
    return bool(toks & {"not", "no", "never", "none", "cannot", "isnt", "wasnt", "didnt"})

