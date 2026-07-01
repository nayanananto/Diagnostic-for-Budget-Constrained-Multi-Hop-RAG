"""Semantic representation compression methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Compressor(Protocol):
    name: str

    def fit(self, vectors: Any) -> None: ...

    def transform(self, vectors: Any) -> Any: ...


@dataclass
class IdentityCompressor:
    name: str = "identity"

    def fit(self, vectors: Any) -> None:
        return None

    def transform(self, vectors: Any) -> Any:
        return vectors


@dataclass
class TruncateCompressor:
    dims: int = 128

    @property
    def name(self) -> str:
        return f"truncate:{self.dims}"

    def fit(self, vectors: Any) -> None:
        return None

    def transform(self, vectors: Any) -> Any:
        try:
            return vectors[:, : self.dims]
        except Exception:
            return vectors


@dataclass
class PCACompressor:
    dims: int = 128
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = None
        self.effective_dims = self.dims

    @property
    def name(self) -> str:
        return f"pca:{self.dims}"

    def fit(self, vectors: Any) -> None:
        from sklearn.decomposition import PCA

        self.effective_dims = max(1, min(self.dims, vectors.shape[0], vectors.shape[1]))
        self.model = PCA(n_components=self.effective_dims, random_state=self.random_state)
        self.model.fit(vectors)

    def transform(self, vectors: Any) -> Any:
        if self.model is None:
            raise RuntimeError("PCACompressor must be fit before transform")
        return self.model.transform(vectors)


@dataclass
class BinarySignCompressor:
    """Sign-only binary projection for compact retrieval experiments."""

    name: str = "binary-sign"

    def fit(self, vectors: Any) -> None:
        return None

    def transform(self, vectors: Any) -> Any:
        import numpy as np

        return np.where(vectors >= 0, 1, -1).astype("int8")


def build_compressor(name: str, dims: int | None = None) -> Compressor:
    if name == "identity":
        return IdentityCompressor()
    if name == "truncate":
        return TruncateCompressor(dims or 128)
    if name == "pca":
        return PCACompressor(dims or 128)
    if name == "binary":
        return BinarySignCompressor()
    raise ValueError(f"unknown compressor '{name}'")
