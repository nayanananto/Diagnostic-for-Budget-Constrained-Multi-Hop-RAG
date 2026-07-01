"""Baseline and ACE graph retrievers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .compression import Compressor, IdentityCompressor
from .embeddings import TextEmbedder
from .schema import CorpusDataset, EvidenceGraph, RetrievalHit, RetrievalRun
from .text import extract_entities, lexical_overlap, tokenize

_BRIDGE_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "did", "does", "do", "that",
    "this", "these", "those", "as", "at", "from", "it", "its", "their", "his",
    "her", "he", "she", "they", "them", "same", "into", "than", "then", "also",
}


def _cosine_scores(matrix, query_vector) -> list[float]:
    import numpy as np

    matrix = matrix.astype("float32")
    qv = query_vector.astype("float32")
    denom = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(qv)), 1e-12)
    scores = np.dot(matrix, qv) / np.maximum(denom, 1e-12)
    return [float(x) for x in scores]


def _cosine_score_matrix(matrix, query_vectors):
    import numpy as np

    matrix = matrix.astype("float32")
    queries = query_vectors.astype("float32")
    matrix_norm = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    query_norm = np.maximum(np.linalg.norm(queries, axis=1), 1e-12)
    return np.dot(queries, matrix.T) / np.outer(query_norm, matrix_norm)


def _top_indices(scores, k: int) -> list[int]:
    import numpy as np

    arr = np.asarray(scores, dtype="float32")
    if len(arr) == 0:
        return []
    k = min(k, len(arr))
    if k == len(arr):
        idx = np.argsort(-arr)
    else:
        idx = np.argpartition(-arr, k - 1)[:k]
        idx = idx[np.argsort(-arr[idx])]
    return [int(i) for i in idx]


@dataclass
class ChunkRetriever:
    embedder: TextEmbedder
    top_k: int = 5

    def fit(self, dataset: CorpusDataset) -> None:
        self.dataset = dataset
        self.doc_ids = list(dataset.documents)
        self.texts = [dataset.documents[d].text for d in self.doc_ids]
        self.embedder.fit(self.texts)
        try:
            import numpy as np

            vectors = self.embedder.encode(self.texts)
            self.doc_vectors = vectors if isinstance(vectors, np.ndarray) else None
        except Exception:
            self.doc_vectors = None

    def retrieve(self, query: str, qid: str = "") -> RetrievalRun:
        if getattr(self, "doc_vectors", None) is not None:
            try:
                qv = self.embedder.encode([query])[0]
                scores = _cosine_scores(self.doc_vectors, qv)
            except Exception:
                scores = self.embedder.score(query, self.texts)
        else:
            scores = self.embedder.score(query, self.texts)
        return self._build_run_from_scores(query, qid, scores)

    def retrieve_many(self, queries: list[tuple[str, str]]) -> list[RetrievalRun]:
        if getattr(self, "doc_vectors", None) is None:
            return [self.retrieve(text, qid) for qid, text in queries]
        try:
            q_vectors = self.embedder.encode([text for _, text in queries])
            score_matrix = _cosine_score_matrix(self.doc_vectors, q_vectors)
            return [
                self._build_run_from_scores(text, qid, [float(x) for x in row])
                for (qid, text), row in zip(queries, score_matrix)
            ]
        except Exception:
            return [self.retrieve(text, qid) for qid, text in queries]

    def _build_run_from_scores(self, query: str, qid: str, scores: list[float]) -> RetrievalRun:
        top = _top_indices(scores, self.top_k)
        ranked = [(self.doc_ids[i], self.texts[i], float(scores[i])) for i in top]
        hits = [
            RetrievalHit(
                node_id=f"passage::{doc_id}",
                node_type="passage",
                text=text,
                score=float(score),
                source_doc_id=doc_id,
                expanded_doc_ids=[doc_id],
            )
            for doc_id, text, score in ranked
        ]
        return RetrievalRun(
            qid=qid,
            query=query,
            hits=hits,
            retrieved_doc_ids=[h.source_doc_id or "" for h in hits],
            diagnostics={
                "retrieved_nodes": len(hits),
                "expanded_docs": len(hits),
                "evidence_tokens": sum(len(tokenize(hit.text)) for hit in hits),
            },
        )


@dataclass
class BM25Retriever:
    top_k: int = 5

    def fit(self, dataset: CorpusDataset) -> None:
        from rank_bm25 import BM25Okapi

        self.dataset = dataset
        self.doc_ids = list(dataset.documents)
        self.texts = [dataset.documents[d].text for d in self.doc_ids]
        self._bm25 = BM25Okapi([tokenize(text) for text in self.texts])

    def retrieve(self, query: str, qid: str = "") -> RetrievalRun:
        scores = self._bm25.get_scores(tokenize(query))
        return self._build_run_from_scores(query, qid, scores)

    def retrieve_many(self, queries: list[tuple[str, str]]) -> list[RetrievalRun]:
        return [self.retrieve(text, qid) for qid, text in queries]

    def _build_run_from_scores(self, query: str, qid: str, scores) -> RetrievalRun:
        top = _top_indices(scores, self.top_k)
        ranked = [(self.doc_ids[i], self.texts[i], float(scores[i])) for i in top]
        hits = [
            RetrievalHit(
                node_id=f"passage::{doc_id}",
                node_type="passage",
                text=text,
                score=float(score),
                source_doc_id=doc_id,
                expanded_doc_ids=[doc_id],
            )
            for doc_id, text, score in ranked
        ]
        return RetrievalRun(
            qid=qid,
            query=query,
            hits=hits,
            retrieved_doc_ids=[h.source_doc_id or "" for h in hits],
            diagnostics={
                "retrieved_nodes": len(hits),
                "expanded_docs": len(hits),
                "evidence_tokens": sum(len(tokenize(hit.text)) for hit in hits),
                "retriever": "bm25",
            },
        )


@dataclass
class HybridChunkRetriever:
    embedder: TextEmbedder
    top_k: int = 5
    alpha: float = 0.5

    def fit(self, dataset: CorpusDataset) -> None:
        from rank_bm25 import BM25Okapi

        self.dataset = dataset
        self.doc_ids = list(dataset.documents)
        self.texts = [dataset.documents[d].text for d in self.doc_ids]
        self._bm25 = BM25Okapi([tokenize(text) for text in self.texts])
        self.embedder.fit(self.texts)
        try:
            import numpy as np

            vectors = self.embedder.encode(self.texts)
            self.doc_vectors = vectors if isinstance(vectors, np.ndarray) else None
        except Exception:
            self.doc_vectors = None

    def retrieve(self, query: str, qid: str = "") -> RetrievalRun:
        return self.retrieve_many([(qid, query)])[0]

    def retrieve_many(self, queries: list[tuple[str, str]]) -> list[RetrievalRun]:
        dense_rows = None
        if getattr(self, "doc_vectors", None) is not None:
            try:
                q_vectors = self.embedder.encode([text for _, text in queries])
                dense_rows = _cosine_score_matrix(self.doc_vectors, q_vectors)
            except Exception:
                dense_rows = None

        runs: list[RetrievalRun] = []
        for row_idx, (qid, query) in enumerate(queries):
            dense_scores = (
                [float(x) for x in dense_rows[row_idx]]
                if dense_rows is not None
                else self.embedder.score(query, self.texts)
            )
            bm25_scores = [float(x) for x in self._bm25.get_scores(tokenize(query))]
            scores = _fuse_scores(dense_scores, bm25_scores, self.alpha)
            runs.append(self._build_run_from_scores(query, qid, scores))
        return runs

    def _build_run_from_scores(self, query: str, qid: str, scores: list[float]) -> RetrievalRun:
        top = _top_indices(scores, self.top_k)
        ranked = [(self.doc_ids[i], self.texts[i], float(scores[i])) for i in top]
        hits = [
            RetrievalHit(
                node_id=f"passage::{doc_id}",
                node_type="passage",
                text=text,
                score=float(score),
                source_doc_id=doc_id,
                expanded_doc_ids=[doc_id],
            )
            for doc_id, text, score in ranked
        ]
        return RetrievalRun(
            qid=qid,
            query=query,
            hits=hits,
            retrieved_doc_ids=[h.source_doc_id or "" for h in hits],
            diagnostics={
                "retrieved_nodes": len(hits),
                "expanded_docs": len(hits),
                "evidence_tokens": sum(len(tokenize(hit.text)) for hit in hits),
                "retriever": "hybrid",
                "hybrid_alpha": self.alpha,
            },
        )


@dataclass
class ACEGraphRetriever:
    embedder: TextEmbedder
    compressor: Compressor = field(default_factory=IdentityCompressor)
    top_k_nodes: int = 8
    max_expanded_docs: int = 5
    conflict_expand: bool = True

    def fit(self, graph: EvidenceGraph) -> None:
        self.graph = graph
        self.index_nodes = [n for n in graph.nodes.values() if n.node_type in {"claim", "entity"}]
        self.index_texts = [n.text for n in self.index_nodes]
        self.embedder.fit(self.index_texts)
        try:
            import numpy as np

            vectors = self.embedder.encode(self.index_texts)
            self.compressor.fit(vectors)
            compressed = self.compressor.transform(vectors)
            self.compressed_vectors = compressed if isinstance(compressed, np.ndarray) else None
        except Exception:
            self.compressed_vectors = None

    def retrieve(self, query: str, qid: str = "") -> RetrievalRun:
        scores = self._score_query(query)
        return self._build_run_from_scores(query, qid, scores)

    def retrieve_many(self, queries: list[tuple[str, str]]) -> list[RetrievalRun]:
        if self.compressed_vectors is None:
            return [self.retrieve(text, qid) for qid, text in queries]
        try:
            q_vectors = self.embedder.encode([text for _, text in queries])
            q_vectors = self.compressor.transform(q_vectors)
            score_matrix = _cosine_score_matrix(self.compressed_vectors, q_vectors)
            return [
                self._build_run_from_scores(text, qid, [float(x) for x in row])
                for (qid, text), row in zip(queries, score_matrix)
            ]
        except Exception:
            return [self.retrieve(text, qid) for qid, text in queries]

    def _build_run_from_scores(self, query: str, qid: str, scores: list[float]) -> RetrievalRun:
        top = _top_indices(scores, self.top_k_nodes)
        ranked = [(self.index_nodes[i], float(scores[i])) for i in top]

        hits: list[RetrievalHit] = []
        expanded_doc_ids: list[str] = []
        seen_docs: set[str] = set()
        conflict_count = 0

        for node, score in ranked:
            source_passages = self.graph.source_passages_for(node.node_id)
            node_doc_ids = [p.source_doc_id for p in source_passages if p.source_doc_id]

            if self.conflict_expand:
                for neighbor in self.graph.neighbors(node.node_id, {"contradicts", "rev:contradicts"}):
                    conflict_count += 1
                    for passage in self.graph.source_passages_for(neighbor.node_id):
                        if passage.source_doc_id:
                            node_doc_ids.append(passage.source_doc_id)

            for doc_id in node_doc_ids:
                if doc_id and doc_id not in seen_docs and len(expanded_doc_ids) < self.max_expanded_docs:
                    expanded_doc_ids.append(doc_id)
                    seen_docs.add(doc_id)

            hits.append(
                RetrievalHit(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    text=node.text,
                    score=float(score),
                    source_doc_id=node.source_doc_id,
                    expanded_doc_ids=list(dict.fromkeys(d for d in node_doc_ids if d)),
                    metadata={
                        "entities": node.metadata.get("entities", []),
                        "query_overlap": lexical_overlap(query, node.text),
                    },
                )
            )
            if len(expanded_doc_ids) >= self.max_expanded_docs:
                break

        return RetrievalRun(
            qid=qid,
            query=query,
            hits=hits,
            retrieved_doc_ids=expanded_doc_ids,
            diagnostics={
                "retrieved_nodes": len(hits),
                "expanded_docs": len(expanded_doc_ids),
                "evidence_tokens": sum(len(tokenize(hit.text)) for hit in hits),
                "conflict_edges_seen": conflict_count,
                "compressor": self.compressor.name,
            },
        )

    def _score_query(self, query: str) -> list[float]:
        if self.compressed_vectors is None:
            return self.embedder.score(query, self.index_texts)
        try:
            q = self.embedder.encode([query])
            q = self.compressor.transform(q)
            return _cosine_scores(self.compressed_vectors, q[0])
        except Exception:
            return self.embedder.score(query, self.index_texts)


@dataclass
class BridgeACEGraphRetriever(ACEGraphRetriever):
    """Two-hop ACE retrieval for harder composed multi-hop questions.

    The first pass finds seed evidence. Bridge terms/entities from the seed
    evidence are then used to rerank graph nodes for a second-hop retrieval.
    This is intentionally lightweight and keeps the base ACE expansion logic.
    """

    bridge_seed_nodes: int = 12
    bridge_terms: int = 10
    bridge_weight: float = 0.35
    bridge_bonus: float = 0.12

    def retrieve(self, query: str, qid: str = "") -> RetrievalRun:
        base_scores = self._score_query(query)
        combined_scores, terms = self._bridge_scores(query, base_scores)
        run = self._build_run_from_scores(query, qid, combined_scores)
        run.diagnostics.update(
            {
                "ace_retriever": "bridge",
                "bridge_terms": terms,
                "bridge_seed_nodes": self.bridge_seed_nodes,
            }
        )
        return run

    def retrieve_many(self, queries: list[tuple[str, str]]) -> list[RetrievalRun]:
        # Bridge terms are query-specific, so each query needs its own second-hop pass.
        return [self.retrieve(text, qid) for qid, text in queries]

    def _bridge_scores(self, query: str, base_scores: list[float]) -> tuple[list[float], list[str]]:
        seed_indices = _top_indices(base_scores, self.bridge_seed_nodes)
        terms = self._extract_bridge_terms(query, seed_indices)
        if not terms:
            return base_scores, []

        bridge_query = f"{query} {' '.join(terms)}"
        bridge_scores = self._score_query(bridge_query)
        term_set = {term.lower() for term in terms}

        combined: list[float] = []
        for node, base, bridge in zip(self.index_nodes, base_scores, bridge_scores):
            score = (1.0 - self.bridge_weight) * float(base) + self.bridge_weight * float(bridge)
            text = node.text.lower()
            if any(term in text for term in term_set):
                score += self.bridge_bonus
            source_doc_id = node.source_doc_id
            if source_doc_id:
                source_passages = self.graph.source_passages_for(node.node_id)
                source_text = " ".join(p.text.lower() for p in source_passages)
                if any(term in source_text for term in term_set):
                    score += self.bridge_bonus * 0.5
            combined.append(score)
        return combined, terms

    def _extract_bridge_terms(self, query: str, seed_indices: list[int]) -> list[str]:
        query_terms = _content_terms(query)
        candidates: dict[str, float] = {}

        for rank, idx in enumerate(seed_indices):
            node = self.index_nodes[idx]
            rank_weight = 1.0 / (rank + 1)
            texts = [node.text]
            for passage in self.graph.source_passages_for(node.node_id):
                texts.append(passage.text)

            for text in texts:
                for entity in extract_entities(text):
                    normalized = entity.lower()
                    if normalized not in query.lower() and len(normalized) > 2:
                        candidates[entity] = candidates.get(entity, 0.0) + 2.0 * rank_weight
                for term in _content_terms(text):
                    if term not in query_terms:
                        candidates[term] = candidates.get(term, 0.0) + rank_weight

            for entity in node.metadata.get("entities", []):
                normalized = str(entity).lower()
                if normalized not in query.lower():
                    candidates[str(entity)] = candidates.get(str(entity), 0.0) + 2.5 * rank_weight

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        return [term for term, _ in ranked[: self.bridge_terms]]


def _content_terms(text: str) -> set[str]:
    return {tok for tok in tokenize(text) if len(tok) > 2 and tok not in _BRIDGE_STOPWORDS}


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [1.0 for _ in scores]
    return [(score - lo) / (hi - lo) for score in scores]


def _fuse_scores(dense_scores: list[float], bm25_scores: list[float], alpha: float) -> list[float]:
    dense_norm = _minmax(dense_scores)
    bm25_norm = _minmax(bm25_scores)
    return [
        alpha * dense + (1.0 - alpha) * bm25
        for dense, bm25 in zip(dense_norm, bm25_norm)
    ]
