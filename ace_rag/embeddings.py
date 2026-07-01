"""Embedding backends."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .text import cosine_from_counters, tokenize


class TextEmbedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str]): ...

    def score(self, query: str, texts: list[str]) -> list[float]: ...


@dataclass
class LexicalEmbedder:
    """Dependency-free cosine scorer over token counters."""

    name: str = "lexical-counter"

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> list[Counter[str]]:
        return [Counter(tokenize(t)) for t in texts]

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = Counter(tokenize(query))
        return [cosine_from_counters(q, t) for t in self.encode(texts)]


class SentenceTransformerEmbedder:
    """Sentence-transformers backend for real experiments."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.name = f"sentence-transformers:{model_name}"
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]):
        print(f"[embed] encoding {len(texts)} texts on device={self.model.device}", flush=True)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

    def score(self, query: str, texts: list[str]) -> list[float]:
        import numpy as np

        q = self.encode([query])
        x = self.encode(texts)
        return list(np.dot(x, q[0]).astype(float))


def build_embedder(name: str, model_name: str | None = None, device: str | None = None) -> TextEmbedder:
    if name == "lexical":
        return LexicalEmbedder()
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name or "BAAI/bge-small-en-v1.5", device=device)
    raise ValueError(f"unknown embedder '{name}'")
