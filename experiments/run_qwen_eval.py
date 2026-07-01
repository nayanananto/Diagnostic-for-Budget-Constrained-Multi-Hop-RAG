"""Stage-2 answer generation evaluation with a free local Qwen reader."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ace_rag.compression import build_compressor
from ace_rag.datasets import load_dataset
from ace_rag.embeddings import build_embedder
from ace_rag.generator import HuggingFaceGenerator
from ace_rag.graph_builder import build_evidence_graph
from ace_rag.metrics import evaluate_retrieval, exact_match, token_f1
from ace_rag.retriever import (
    ACEGraphRetriever,
    BM25Retriever,
    BridgeACEGraphRetriever,
    ChunkRetriever,
    HybridChunkRetriever,
)
from ace_rag.schema import CorpusDataset, RetrievalHit, RetrievalRun
from ace_rag.text import lexical_overlap, split_sentences, tokenize

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "did", "does", "do", "that",
    "this", "these", "those", "as", "at", "from", "it", "its", "their", "his",
    "her", "he", "she", "they", "them", "same",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa", choices=["toy", "hotpotqa", "musique_local"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--musique-path", default=None)
    parser.add_argument("--methods", default="chunk,ace_graph")
    parser.add_argument("--embedder", default="sentence-transformers", choices=["lexical", "sentence-transformers"])
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embed-device", default="cuda")
    parser.add_argument("--compressor", default="truncate", choices=["identity", "truncate", "pca", "binary"])
    parser.add_argument("--compress-dims", type=int, default=320)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--top-k-nodes", type=int, default=48)
    parser.add_argument("--max-expanded-docs", type=int, default=5)
    parser.add_argument("--ace-retriever", default="standard", choices=["standard", "bridge"])
    parser.add_argument("--bridge-seed-nodes", type=int, default=12)
    parser.add_argument("--bridge-terms", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.35)
    parser.add_argument("--reader-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reader-device", default="cuda")
    parser.add_argument("--reader-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--reader-context",
        default="sources",
        choices=["sources", "hits", "snippets", "packed_snippets", "focused_packed"],
        help="Use expanded source passages or raw retrieval hits as reader evidence.",
    )
    parser.add_argument("--snippet-window", type=int, default=1, help="Neighbor sentences on each side of a matched sentence.")
    parser.add_argument("--max-snippets", type=int, default=8, help="Maximum snippets shown to the reader per question.")
    parser.add_argument("--max-snippet-tokens", type=int, default=80, help="Maximum tokens per snippet.")
    parser.add_argument("--packed-token-budget", type=int, default=240, help="Total token budget for packed snippets.")
    parser.add_argument("--out-dir", default="cloud_results")
    return parser.parse_args()


def load_from_args(args: argparse.Namespace) -> CorpusDataset:
    if args.dataset == "toy":
        return load_dataset("toy")
    if args.dataset == "musique_local":
        if not args.musique_path:
            raise SystemExit("--musique-path is required for musique_local")
        return load_dataset("musique_local", path=args.musique_path, limit=args.limit)
    return load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)


def retrieve_chunk(dataset: CorpusDataset, args: argparse.Namespace) -> list[RetrievalRun]:
    retriever = ChunkRetriever(
        build_embedder(args.embedder, args.embedding_model, device=args.embed_device),
        top_k=args.top_k,
    )
    print("[chunk] fitting/indexing", flush=True)
    retriever.fit(dataset)
    print("[chunk] retrieving", flush=True)
    return retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])


def retrieve_bm25(dataset: CorpusDataset, args: argparse.Namespace) -> list[RetrievalRun]:
    retriever = BM25Retriever(top_k=args.top_k)
    print("[bm25] fitting/indexing", flush=True)
    retriever.fit(dataset)
    print("[bm25] retrieving", flush=True)
    return retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])


def retrieve_hybrid(dataset: CorpusDataset, args: argparse.Namespace) -> list[RetrievalRun]:
    retriever = HybridChunkRetriever(
        build_embedder(args.embedder, args.embedding_model, device=args.embed_device),
        top_k=args.top_k,
        alpha=args.hybrid_alpha,
    )
    print("[hybrid] fitting/indexing", flush=True)
    retriever.fit(dataset)
    print("[hybrid] retrieving", flush=True)
    return retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])


def retrieve_ace(dataset: CorpusDataset, args: argparse.Namespace) -> list[RetrievalRun]:
    print("[ace] building evidence graph", flush=True)
    graph = build_evidence_graph(dataset)
    print(f"[ace] graph built: nodes={len(graph.nodes)} edges={len(graph.edges)}", flush=True)
    retriever_cls = BridgeACEGraphRetriever if args.ace_retriever == "bridge" else ACEGraphRetriever
    kwargs = {}
    if args.ace_retriever == "bridge":
        kwargs = {
            "bridge_seed_nodes": args.bridge_seed_nodes,
            "bridge_terms": args.bridge_terms,
            "bridge_weight": args.bridge_weight,
        }
    retriever = retriever_cls(
        embedder=build_embedder(args.embedder, args.embedding_model, device=args.embed_device),
        compressor=build_compressor(args.compressor, args.compress_dims),
        top_k_nodes=args.top_k_nodes,
        max_expanded_docs=args.max_expanded_docs,
        conflict_expand=True,
        **kwargs,
    )
    print("[ace] fitting/indexing embeddings", flush=True)
    retriever.fit(graph)
    print("[ace] retrieving", flush=True)
    return retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])


def generation_metrics(dataset: CorpusDataset, runs: list[RetrievalRun], predictions: list[str]) -> dict[str, float]:
    by_qid = {q.qid: q for q in dataset.questions}
    em = 0.0
    f1 = 0.0
    evidence_tokens = 0
    n = 0
    for run, pred in zip(runs, predictions):
        q = by_qid[run.qid]
        if not q.answers:
            continue
        em += exact_match(pred, q.answers)
        f1 += token_f1(pred, q.answers)
        evidence_tokens += sum(len(tokenize(hit.text)) for hit in run.hits)
        n += 1
    n = max(1, n)
    return {
        "qwen_em": round(em / n, 4),
        "qwen_f1": round(f1 / n, 4),
        "avg_reader_evidence_tokens": round(evidence_tokens / n, 2),
    }


def materialize_reader_context(
    dataset: CorpusDataset,
    runs: list[RetrievalRun],
    mode: str,
    snippet_window: int = 1,
    max_snippets: int = 8,
    max_snippet_tokens: int = 80,
    packed_token_budget: int = 240,
) -> list[RetrievalRun]:
    """Convert retriever output into the evidence form shown to the reader.

    `hits` preserves the raw retrieval units. `sources` expands retrieved document
    ids into source passages, which is the intended ACE-RAG generation path.
    """

    if mode == "hits":
        return runs
    if mode == "snippets":
        return [
            _snippet_context_run(dataset, run, snippet_window, max_snippets, max_snippet_tokens)
            for run in runs
        ]
    if mode == "packed_snippets":
        return [
            _packed_snippet_context_run(
                dataset,
                run,
                snippet_window=snippet_window,
                max_snippets=max_snippets,
                max_snippet_tokens=max_snippet_tokens,
                token_budget=packed_token_budget,
            )
            for run in runs
        ]
    if mode == "focused_packed":
        return [
            _focused_packed_context_run(
                dataset,
                run,
                snippet_window=snippet_window,
                max_snippets=max_snippets,
                max_snippet_tokens=max_snippet_tokens,
                token_budget=packed_token_budget,
            )
            for run in runs
        ]

    reader_runs: list[RetrievalRun] = []
    for run in runs:
        hits: list[RetrievalHit] = []
        for idx, doc_id in enumerate(run.retrieved_doc_ids):
            doc = dataset.documents.get(doc_id)
            if doc is None:
                continue
            hits.append(
                RetrievalHit(
                    node_id=f"reader_passage::{doc_id}",
                    node_type="passage",
                    text=doc.text,
                    score=float(len(run.retrieved_doc_ids) - idx),
                    source_doc_id=doc_id,
                    expanded_doc_ids=[doc_id],
                    metadata={"title": doc.title},
                )
            )
        reader_runs.append(
            RetrievalRun(
                qid=run.qid,
                query=run.query,
                hits=hits,
                retrieved_doc_ids=list(run.retrieved_doc_ids),
                diagnostics={**run.diagnostics, "reader_context": mode},
            )
        )
    return reader_runs


def _candidate_doc_ids(run: RetrievalRun) -> list[str]:
    candidate_doc_ids: list[str] = []
    for hit in run.hits:
        for doc_id in hit.expanded_doc_ids:
            if doc_id not in candidate_doc_ids:
                candidate_doc_ids.append(doc_id)
        if hit.source_doc_id and hit.source_doc_id not in candidate_doc_ids:
            candidate_doc_ids.append(hit.source_doc_id)
    for doc_id in run.retrieved_doc_ids:
        if doc_id not in candidate_doc_ids:
            candidate_doc_ids.append(doc_id)
    return candidate_doc_ids


def _snippet_context_run(
    dataset: CorpusDataset,
    run: RetrievalRun,
    snippet_window: int,
    max_snippets: int,
    max_snippet_tokens: int,
) -> RetrievalRun:
    hits: list[RetrievalHit] = []
    seen_texts: set[str] = set()

    candidate_doc_ids = _candidate_doc_ids(run)

    for hit in run.hits:
        for doc_id in candidate_doc_ids:
            doc = dataset.documents.get(doc_id)
            if doc is None:
                continue
            snippet = _best_snippet_for_hit(doc.text, hit.text, run.query, snippet_window, max_snippet_tokens)
            key = " ".join(snippet.lower().split())
            if not snippet or key in seen_texts:
                continue
            seen_texts.add(key)
            hits.append(
                RetrievalHit(
                    node_id=f"reader_snippet::{doc_id}::{len(hits)}",
                    node_type="snippet",
                    text=snippet,
                    score=hit.score,
                    source_doc_id=doc_id,
                    expanded_doc_ids=[doc_id],
                    metadata={"title": doc.title, "from_hit": hit.node_id},
                )
            )
            if len(hits) >= max_snippets:
                return RetrievalRun(
                    qid=run.qid,
                    query=run.query,
                    hits=hits,
                    retrieved_doc_ids=list(dict.fromkeys(h.source_doc_id or "" for h in hits if h.source_doc_id)),
                    diagnostics={**run.diagnostics, "reader_context": "snippets"},
                )

    return RetrievalRun(
        qid=run.qid,
        query=run.query,
        hits=hits,
        retrieved_doc_ids=list(dict.fromkeys(h.source_doc_id or "" for h in hits if h.source_doc_id)),
        diagnostics={**run.diagnostics, "reader_context": "snippets"},
    )


def _packed_snippet_context_run(
    dataset: CorpusDataset,
    run: RetrievalRun,
    snippet_window: int,
    max_snippets: int,
    max_snippet_tokens: int,
    token_budget: int,
) -> RetrievalRun:
    candidates: list[tuple[float, str, str, str]] = []
    candidate_doc_ids = _candidate_doc_ids(run)
    doc_rank = {doc_id: idx for idx, doc_id in enumerate(candidate_doc_ids)}

    for doc_id in candidate_doc_ids:
        doc = dataset.documents.get(doc_id)
        if doc is None:
            continue
        sentences = split_sentences(doc.text)
        if not sentences:
            continue
        related_hits = [
            hit for hit in run.hits
            if hit.source_doc_id == doc_id or doc_id in hit.expanded_doc_ids
        ] or run.hits
        for idx, (sentence, _) in enumerate(sentences):
            hit_overlap = max((lexical_overlap(sentence, hit.text) for hit in related_hits), default=0.0)
            query_overlap = lexical_overlap(sentence, run.query)
            rank_bonus = 0.05 / (1 + doc_rank.get(doc_id, 0))
            score = hit_overlap + 0.75 * query_overlap + rank_bonus
            if score <= 0:
                continue
            left = max(0, idx - snippet_window)
            right = min(len(sentences), idx + snippet_window + 1)
            snippet = " ".join(sent for sent, _ in sentences[left:right]).strip()
            snippet = _clip_text_tokens(snippet, max_snippet_tokens)
            if snippet:
                candidates.append((score, doc_id, doc.title, snippet))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[RetrievalHit] = []
    seen_texts: set[str] = set()
    per_doc: dict[str, int] = {}
    used_tokens = 0

    # First pass encourages evidence coverage across documents.
    for max_per_doc in (1, 2, 999):
        for score, doc_id, title, snippet in candidates:
            if per_doc.get(doc_id, 0) >= max_per_doc:
                continue
            key = " ".join(snippet.lower().split())
            if key in seen_texts:
                continue
            n_tokens = len(tokenize(snippet))
            if used_tokens + n_tokens > token_budget and selected:
                continue
            selected.append(
                RetrievalHit(
                    node_id=f"reader_packed_snippet::{doc_id}::{len(selected)}",
                    node_type="packed_snippet",
                    text=snippet,
                    score=float(score),
                    source_doc_id=doc_id,
                    expanded_doc_ids=[doc_id],
                    metadata={"title": title},
                )
            )
            seen_texts.add(key)
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            used_tokens += n_tokens
            if len(selected) >= max_snippets or used_tokens >= token_budget:
                return _packed_run(run, selected, token_budget)

    return _packed_run(run, selected, token_budget)


def _focused_packed_context_run(
    dataset: CorpusDataset,
    run: RetrievalRun,
    snippet_window: int,
    max_snippets: int,
    max_snippet_tokens: int,
    token_budget: int,
) -> RetrievalRun:
    """Pack snippets with query-term coverage and source coverage.

    This is a deliberately small improvement over `_packed_snippet_context_run`:
    it still uses lexical scores, but it greedily prefers snippets that add new
    question terms and cover new retrieved documents under the same token budget.
    """

    query_terms = _content_terms(run.query)
    candidates: list[dict[str, Any]] = []
    candidate_doc_ids = _candidate_doc_ids(run)
    doc_rank = {doc_id: idx for idx, doc_id in enumerate(candidate_doc_ids)}

    for doc_id in candidate_doc_ids:
        doc = dataset.documents.get(doc_id)
        if doc is None:
            continue
        sentences = split_sentences(doc.text)
        if not sentences:
            continue
        related_hits = [
            hit for hit in run.hits
            if hit.source_doc_id == doc_id or doc_id in hit.expanded_doc_ids
        ] or run.hits
        for idx, (sentence, _) in enumerate(sentences):
            left = max(0, idx - snippet_window)
            right = min(len(sentences), idx + snippet_window + 1)
            snippet = " ".join(sent for sent, _ in sentences[left:right]).strip()
            snippet = _clip_text_tokens(snippet, max_snippet_tokens)
            if not snippet:
                continue
            q_overlap = lexical_overlap(snippet, run.query)
            hit_overlap = max((lexical_overlap(snippet, hit.text) for hit in related_hits), default=0.0)
            title_overlap = lexical_overlap(doc.title, run.query) if doc.title else 0.0
            rank_bonus = 0.05 / (1 + doc_rank.get(doc_id, 0))
            base_score = 1.15 * q_overlap + 0.65 * hit_overlap + 0.25 * title_overlap + rank_bonus
            if base_score <= 0:
                continue
            candidates.append(
                {
                    "base_score": base_score,
                    "doc_id": doc_id,
                    "title": doc.title,
                    "snippet": snippet,
                    "terms": _content_terms(snippet),
                    "tokens": len(tokenize(snippet)),
                }
            )

    selected: list[RetrievalHit] = []
    seen_texts: set[str] = set()
    seen_docs: set[str] = set()
    covered_terms: set[str] = set()
    used_tokens = 0

    while candidates and len(selected) < max_snippets and used_tokens < token_budget:
        best_idx = -1
        best_score = -1.0
        for idx, candidate in enumerate(candidates):
            snippet = candidate["snippet"]
            key = " ".join(snippet.lower().split())
            if key in seen_texts:
                continue
            if any(lexical_overlap(snippet, hit.text) > 0.8 for hit in selected):
                continue
            if used_tokens + candidate["tokens"] > token_budget and selected:
                continue
            new_terms = candidate["terms"] & query_terms - covered_terms
            coverage_bonus = len(new_terms) / max(1, len(query_terms))
            doc_bonus = 0.15 if candidate["doc_id"] not in seen_docs else 0.0
            score = candidate["base_score"] + 0.55 * coverage_bonus + doc_bonus
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx < 0:
            break
        candidate = candidates.pop(best_idx)
        snippet = candidate["snippet"]
        selected.append(
            RetrievalHit(
                node_id=f"reader_focused_packed::{candidate['doc_id']}::{len(selected)}",
                node_type="focused_packed_snippet",
                text=snippet,
                score=float(best_score),
                source_doc_id=candidate["doc_id"],
                expanded_doc_ids=[candidate["doc_id"]],
                metadata={"title": candidate["title"]},
            )
        )
        seen_texts.add(" ".join(snippet.lower().split()))
        seen_docs.add(candidate["doc_id"])
        covered_terms |= candidate["terms"] & query_terms
        used_tokens += candidate["tokens"]

    return RetrievalRun(
        qid=run.qid,
        query=run.query,
        hits=selected,
        retrieved_doc_ids=list(dict.fromkeys(h.source_doc_id or "" for h in selected if h.source_doc_id)),
        diagnostics={**run.diagnostics, "reader_context": "focused_packed", "packed_token_budget": token_budget},
    )


def _packed_run(run: RetrievalRun, hits: list[RetrievalHit], token_budget: int) -> RetrievalRun:
    return RetrievalRun(
        qid=run.qid,
        query=run.query,
        hits=hits,
        retrieved_doc_ids=list(dict.fromkeys(h.source_doc_id or "" for h in hits if h.source_doc_id)),
        diagnostics={**run.diagnostics, "reader_context": "packed_snippets", "packed_token_budget": token_budget},
    )


def _best_snippet_for_hit(
    document_text: str,
    hit_text: str,
    query: str,
    snippet_window: int,
    max_snippet_tokens: int,
) -> str:
    sentences = split_sentences(document_text)
    if not sentences:
        return " ".join(tokenize(document_text)[:max_snippet_tokens])

    best_idx = 0
    best_score = -1.0
    for idx, (sentence, _) in enumerate(sentences):
        score = lexical_overlap(sentence, hit_text) + 0.5 * lexical_overlap(sentence, query)
        if score > best_score:
            best_idx = idx
            best_score = score

    left = max(0, best_idx - snippet_window)
    right = min(len(sentences), best_idx + snippet_window + 1)
    snippet = " ".join(sentence for sentence, _ in sentences[left:right])
    tokens = snippet.split()
    if len(tokens) > max_snippet_tokens:
        snippet = " ".join(tokens[:max_snippet_tokens])
    return snippet.strip()


def _clip_text_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max_tokens]).strip()


def _content_terms(text: str) -> set[str]:
    return {tok for tok in tokenize(text) if len(tok) > 2 and tok not in _STOPWORDS}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(path: Path, method: str, dataset: CorpusDataset, runs: list[RetrievalRun], predictions: list[str]) -> None:
    by_qid = {q.qid: q for q in dataset.questions}
    with path.open("a", encoding="utf-8") as f:
        for run, pred in zip(runs, predictions):
            q = by_qid[run.qid]
            f.write(
                json.dumps(
                    {
                        "method": method,
                        "qid": q.qid,
                        "question": q.text,
                        "gold_answers": q.answers,
                        "prediction": pred,
                        "retrieved_doc_ids": run.retrieved_doc_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    dataset = load_from_args(args)
    print(dataset.summary(), flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = {m.strip() for m in args.methods.split(",") if m.strip()}
    runs_by_method: dict[str, list[RetrievalRun]] = {}
    if "chunk" in requested:
        runs_by_method["chunk"] = retrieve_chunk(dataset, args)
    if "bm25" in requested:
        runs_by_method["bm25"] = retrieve_bm25(dataset, args)
    if "hybrid" in requested:
        runs_by_method["hybrid"] = retrieve_hybrid(dataset, args)
    if "ace_graph" in requested:
        runs_by_method["ace_graph"] = retrieve_ace(dataset, args)

    print("[reader] loading Qwen reader", flush=True)
    reader = HuggingFaceGenerator(
        model_name=args.reader_model,
        device=args.reader_device,
        batch_size=args.reader_batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    rows: list[dict[str, Any]] = []
    pred_path = out_dir / (
        f"{args.dataset}_qwen_predictions_limit{args.limit}_"
        f"{args.reader_model.split('/')[-1]}_{args.compressor}{args.compress_dims}.jsonl"
    )
    if pred_path.exists():
        pred_path.unlink()

    for method, runs in runs_by_method.items():
        print(f"[reader] generating method={method} context={args.reader_context}", flush=True)
        reader_runs = materialize_reader_context(
            dataset,
            runs,
            args.reader_context,
            snippet_window=args.snippet_window,
            max_snippets=args.max_snippets,
            max_snippet_tokens=args.max_snippet_tokens,
            packed_token_budget=args.packed_token_budget,
        )
        predictions = reader.answer_many([(run.query, run) for run in reader_runs])
        row: dict[str, Any] = {
            "method": method,
            "reader_model": args.reader_model,
            "retriever_compressor": f"{args.compressor}:{args.compress_dims}" if method == "ace_graph" else "none",
            "ace_retriever": args.ace_retriever if method == "ace_graph" else "none",
            "hybrid_alpha": args.hybrid_alpha if method == "hybrid" else "",
            "reader_context": args.reader_context,
        }
        row.update(evaluate_retrieval(runs, dataset.questions, k_values=(1, 2, 5)))
        row.update(generation_metrics(dataset, reader_runs, predictions))
        rows.append(row)
        save_predictions(pred_path, method, dataset, reader_runs, predictions)

    metrics_path = out_dir / (
        f"{args.dataset}_qwen_eval_limit{args.limit}_{args.reader_model.split('/')[-1]}_"
        f"{args.ace_retriever}_{args.compressor}{args.compress_dims}_metrics.csv"
    )
    write_csv(metrics_path, rows)
    for row in rows:
        print(", ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
