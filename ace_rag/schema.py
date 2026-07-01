"""Shared data structures for ACE-RAG experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    answers: list[str]
    gold_doc_ids: set[str] = field(default_factory=set)


@dataclass
class CorpusDataset:
    name: str
    documents: dict[str, Document]
    questions: list[Question]

    def summary(self) -> str:
        gold = sum(len(q.gold_doc_ids) for q in self.questions)
        avg_gold = gold / max(1, len(self.questions))
        return (
            f"{self.name}: {len(self.documents)} docs, "
            f"{len(self.questions)} questions, {avg_gold:.2f} gold docs/question"
        )


@dataclass
class EvidenceNode:
    node_id: str
    node_type: str
    text: str
    source_doc_id: str | None = None
    source_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0


@dataclass
class EvidenceGraph:
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: list[EvidenceEdge] = field(default_factory=list)
    adjacency: dict[str, list[EvidenceEdge]] = field(default_factory=dict)

    def add_node(self, node: EvidenceNode) -> None:
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(node.node_id, [])

    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0) -> None:
        edge = EvidenceEdge(source=source, target=target, edge_type=edge_type, weight=weight)
        self.edges.append(edge)
        self.adjacency.setdefault(source, []).append(edge)
        reverse = EvidenceEdge(source=target, target=source, edge_type=f"rev:{edge_type}", weight=weight)
        self.adjacency.setdefault(target, []).append(reverse)

    def nodes_by_type(self, node_type: str) -> list[EvidenceNode]:
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def neighbors(self, node_id: str, edge_types: set[str] | None = None) -> Iterable[EvidenceNode]:
        for edge in self.adjacency.get(node_id, []):
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            target = self.nodes.get(edge.target)
            if target is not None:
                yield target

    def source_passages_for(self, node_id: str) -> list[EvidenceNode]:
        node = self.nodes[node_id]
        if node.node_type == "passage":
            return [node]
        passages: list[EvidenceNode] = []
        seen: set[str] = set()
        for edge in self.adjacency.get(node_id, []):
            target = self.nodes.get(edge.target)
            if target and target.node_type == "passage" and target.node_id not in seen:
                passages.append(target)
                seen.add(target.node_id)
        return passages


@dataclass
class RetrievalHit:
    node_id: str
    score: float
    node_type: str
    text: str
    source_doc_id: str | None = None
    expanded_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRun:
    qid: str
    query: str
    hits: list[RetrievalHit]
    retrieved_doc_ids: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)

