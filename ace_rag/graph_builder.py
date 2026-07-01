"""Evidence graph construction."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .schema import CorpusDataset, EvidenceGraph, EvidenceNode
from .text import extract_entities, has_negation, lexical_overlap, split_sentences


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def build_evidence_graph(dataset: CorpusDataset, max_claims_per_doc: int | None = None) -> EvidenceGraph:
    graph = EvidenceGraph()
    entity_to_claims: dict[str, list[str]] = defaultdict(list)

    for doc in dataset.documents.values():
        passage_id = f"passage::{doc.doc_id}"
        graph.add_node(
            EvidenceNode(
                node_id=passage_id,
                node_type="passage",
                text=doc.text,
                source_doc_id=doc.doc_id,
                source_span=(0, len(doc.text)),
                metadata={"title": doc.title, **doc.metadata},
            )
        )

        sentences = split_sentences(doc.text)
        if max_claims_per_doc:
            sentences = sentences[:max_claims_per_doc]
        for idx, (sentence, span) in enumerate(sentences):
            claim_id = _stable_id("claim", f"{doc.doc_id}:{idx}:{sentence}")
            entities = extract_entities(sentence)
            graph.add_node(
                EvidenceNode(
                    node_id=claim_id,
                    node_type="claim",
                    text=sentence,
                    source_doc_id=doc.doc_id,
                    source_span=span,
                    metadata={"entities": entities, "sentence_index": idx},
                )
            )
            graph.add_edge(claim_id, passage_id, "derived_from")

            for entity in entities:
                ent_id = _stable_id("entity", entity.lower())
                if ent_id not in graph.nodes:
                    graph.add_node(
                        EvidenceNode(
                            node_id=ent_id,
                            node_type="entity",
                            text=entity,
                            metadata={"canonical": entity.lower()},
                        )
                    )
                graph.add_edge(claim_id, ent_id, "mentions")
                entity_to_claims[entity.lower()].append(claim_id)

    _add_lightweight_conflict_edges(graph, entity_to_claims)
    return graph


def _add_lightweight_conflict_edges(graph: EvidenceGraph, entity_to_claims: dict[str, list[str]]) -> None:
    """Add cheap contradiction edges for the MVP.

    This is not meant to be the final paper method. It is a deterministic
    placeholder that lets the experimental harness exercise conflict-aware
    retrieval before an NLI/LLM verifier is added.
    """

    seen_pairs: set[tuple[str, str]] = set()
    for claim_ids in entity_to_claims.values():
        for i, left_id in enumerate(claim_ids):
            left = graph.nodes[left_id]
            for right_id in claim_ids[i + 1 :]:
                right = graph.nodes[right_id]
                key = tuple(sorted((left_id, right_id)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if has_negation(left.text) == has_negation(right.text):
                    continue
                if lexical_overlap(left.text, right.text) >= 0.45:
                    graph.add_edge(left_id, right_id, "contradicts", weight=0.8)

