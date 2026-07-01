"""Run the ACE-RAG MVP against a baseline chunk retriever."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ace_rag.compression import build_compressor
from ace_rag.datasets import load_dataset
from ace_rag.embeddings import build_embedder
from ace_rag.generator import ExtractiveGenerator
from ace_rag.graph_builder import build_evidence_graph
from ace_rag.metrics import evaluate_retrieval, exact_match, token_f1
from ace_rag.retriever import (
    ACEGraphRetriever,
    BM25Retriever,
    BridgeACEGraphRetriever,
    ChunkRetriever,
    HybridChunkRetriever,
)
from ace_rag.schema import CorpusDataset, RetrievalRun
from ace_rag.text import tokenize
from ace_rag.verifier import EvidenceVerifier


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional YAML config file.")
    pre_args, _ = pre_parser.parse_known_args()
    defaults: dict[str, Any] = {}
    if pre_args.config:
        import yaml

        config_path = Path(pre_args.config)
        defaults = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("--dataset", default="toy", choices=["toy", "hotpotqa", "musique_local", "ragbench"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--musique-path", default=None)
    parser.add_argument("--ragbench-subset", default=None)
    parser.add_argument("--embedder", default="lexical", choices=["lexical", "sentence-transformers"])
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default=None, help="Embedding device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--compressor", default="identity", choices=["identity", "truncate", "pca", "binary"])
    parser.add_argument("--compress-dims", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--top-k-nodes", type=int, default=8)
    parser.add_argument("--max-expanded-docs", type=int, default=5)
    parser.add_argument("--ace-retriever", default="standard", choices=["standard", "bridge"])
    parser.add_argument("--bridge-seed-nodes", type=int, default=12)
    parser.add_argument("--bridge-terms", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.35)
    parser.add_argument(
        "--methods",
        default="chunk,ace_graph,ace_graph_no_conflict_expand",
        help="Comma-separated methods: chunk,bm25,hybrid,ace_graph,ace_graph_no_conflict_expand",
    )
    parser.add_argument("--no-save-runs", action="store_true", help="Write metrics CSV only, not detailed runs JSON.")
    parser.add_argument("--out-dir", default="results")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def load_from_args(args: argparse.Namespace) -> CorpusDataset:
    if args.dataset == "toy":
        return load_dataset("toy")
    if args.dataset == "hotpotqa":
        return load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)
    if args.dataset == "musique_local":
        if not args.musique_path:
            raise SystemExit("--musique-path is required for musique_local")
        return load_dataset("musique_local", path=args.musique_path, limit=args.limit)
    if args.dataset == "ragbench":
        return load_dataset("ragbench", split=args.split, subset=args.ragbench_subset, limit=args.limit)
    raise ValueError(args.dataset)


def score_answers(dataset: CorpusDataset, runs: list[RetrievalRun]) -> dict[str, float]:
    generator = ExtractiveGenerator()
    by_qid = {q.qid: q for q in dataset.questions}
    em_total = 0.0
    f1_total = 0.0
    n = 0
    for run in runs:
        q = by_qid[run.qid]
        if not q.answers:
            continue
        pred = generator.answer(q.text, run)
        em_total += exact_match(pred, q.answers)
        f1_total += token_f1(pred, q.answers)
        n += 1
    if n == 0:
        return {}
    return {"extractive_em": round(em_total / n, 4), "extractive_f1": round(f1_total / n, 4)}


def verify_runs(runs: list[RetrievalRun]) -> dict[str, float]:
    verifier = EvidenceVerifier()
    counts = {"answer": 0, "conflict": 0, "abstain": 0}
    for run in runs:
        result = verifier.verify(run.query, run)
        counts[result.decision] = counts.get(result.decision, 0) + 1
    total = max(1, len(runs))
    return {f"verifier_{key}_rate": round(value / total, 4) for key, value in counts.items()}


def budget_metrics(runs: list[RetrievalRun]) -> dict[str, float]:
    total = max(1, len(runs))
    evidence_tokens = [
        run.diagnostics.get("evidence_tokens", sum(len(tokenize(hit.text)) for hit in run.hits))
        for run in runs
    ]
    expanded_docs = [len(run.retrieved_doc_ids) for run in runs]
    hits = [len(run.hits) for run in runs]
    return {
        "avg_evidence_tokens": round(sum(evidence_tokens) / total, 2),
        "avg_expanded_docs": round(sum(expanded_docs) / total, 2),
        "avg_hits": round(sum(hits) / total, 2),
    }


def run_chunk(dataset: CorpusDataset, args: argparse.Namespace) -> tuple[dict[str, Any], list[RetrievalRun]]:
    retriever = ChunkRetriever(build_embedder(args.embedder, args.embedding_model, device=args.device), top_k=args.top_k)
    print("[chunk] fitting/indexing", flush=True)
    retriever.fit(dataset)
    print("[chunk] retrieving", flush=True)
    runs = retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])
    metrics = evaluate_retrieval(runs, dataset.questions, k_values=(1, 2, 5))
    metrics.update(score_answers(dataset, runs))
    metrics.update(verify_runs(runs))
    metrics.update(budget_metrics(runs))
    metrics.update({"method": "chunk", "retrieved_unit": "passage", "compressor": "none"})
    return metrics, runs


def run_bm25(dataset: CorpusDataset, args: argparse.Namespace) -> tuple[dict[str, Any], list[RetrievalRun]]:
    retriever = BM25Retriever(top_k=args.top_k)
    print("[bm25] fitting/indexing", flush=True)
    retriever.fit(dataset)
    print("[bm25] retrieving", flush=True)
    runs = retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])
    metrics = evaluate_retrieval(runs, dataset.questions, k_values=(1, 2, 5))
    metrics.update(score_answers(dataset, runs))
    metrics.update(verify_runs(runs))
    metrics.update(budget_metrics(runs))
    metrics.update({"method": "bm25", "retrieved_unit": "passage", "compressor": "none"})
    return metrics, runs


def run_hybrid(dataset: CorpusDataset, args: argparse.Namespace) -> tuple[dict[str, Any], list[RetrievalRun]]:
    retriever = HybridChunkRetriever(
        build_embedder(args.embedder, args.embedding_model, device=args.device),
        top_k=args.top_k,
        alpha=args.hybrid_alpha,
    )
    print("[hybrid] fitting/indexing", flush=True)
    retriever.fit(dataset)
    print("[hybrid] retrieving", flush=True)
    runs = retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])
    metrics = evaluate_retrieval(runs, dataset.questions, k_values=(1, 2, 5))
    metrics.update(score_answers(dataset, runs))
    metrics.update(verify_runs(runs))
    metrics.update(budget_metrics(runs))
    metrics.update(
        {
            "method": "hybrid",
            "retrieved_unit": "passage",
            "compressor": "none",
            "hybrid_alpha": args.hybrid_alpha,
        }
    )
    return metrics, runs


def run_ace(dataset: CorpusDataset, args: argparse.Namespace, conflict_expand: bool = True) -> tuple[dict[str, Any], list[RetrievalRun]]:
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
        embedder=build_embedder(args.embedder, args.embedding_model, device=args.device),
        compressor=build_compressor(args.compressor, args.compress_dims),
        top_k_nodes=args.top_k_nodes,
        max_expanded_docs=args.max_expanded_docs,
        conflict_expand=conflict_expand,
        **kwargs,
    )
    print("[ace] fitting/indexing embeddings", flush=True)
    retriever.fit(graph)
    print("[ace] retrieving", flush=True)
    runs = retriever.retrieve_many([(q.qid, q.text) for q in dataset.questions])
    metrics = evaluate_retrieval(runs, dataset.questions, k_values=(1, 2, 5))
    metrics.update(score_answers(dataset, runs))
    metrics.update(verify_runs(runs))
    metrics.update(budget_metrics(runs))
    metrics.update(
        {
            "method": "ace_graph" if conflict_expand else "ace_graph_no_conflict_expand",
            "retrieved_unit": "claim/entity",
            "compressor": retriever.compressor.name,
            "ace_retriever": args.ace_retriever,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
        }
    )
    return metrics, runs


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_to_jsonable(run: RetrievalRun) -> dict[str, Any]:
    return {
        "qid": run.qid,
        "query": run.query,
        "retrieved_doc_ids": run.retrieved_doc_ids,
        "diagnostics": run.diagnostics,
        "hits": [
            {
                "node_id": hit.node_id,
                "node_type": hit.node_type,
                "score": hit.score,
                "source_doc_id": hit.source_doc_id,
                "expanded_doc_ids": hit.expanded_doc_ids,
                "text": hit.text,
            }
            for hit in run.hits
        ],
    }


def main() -> None:
    args = parse_args()
    dataset = load_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(dataset.summary())
    requested_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    rows: list[dict[str, Any]] = []
    all_runs: dict[str, list[dict[str, Any]]] = {}

    if "chunk" in requested_methods:
        print("running method=chunk", flush=True)
        chunk_metrics, chunk_runs = run_chunk(dataset, args)
        rows.append(chunk_metrics)
        all_runs["chunk"] = [run_to_jsonable(run) for run in chunk_runs]

    if "bm25" in requested_methods:
        print("running method=bm25", flush=True)
        bm25_metrics, bm25_runs = run_bm25(dataset, args)
        rows.append(bm25_metrics)
        all_runs["bm25"] = [run_to_jsonable(run) for run in bm25_runs]

    if "hybrid" in requested_methods:
        print("running method=hybrid", flush=True)
        hybrid_metrics, hybrid_runs = run_hybrid(dataset, args)
        rows.append(hybrid_metrics)
        all_runs["hybrid"] = [run_to_jsonable(run) for run in hybrid_runs]

    if "ace_graph" in requested_methods:
        print("running method=ace_graph", flush=True)
        ace_metrics, ace_runs = run_ace(dataset, args, conflict_expand=True)
        rows.append(ace_metrics)
        all_runs["ace_graph"] = [run_to_jsonable(run) for run in ace_runs]

    if "ace_graph_no_conflict_expand" in requested_methods:
        print("running method=ace_graph_no_conflict_expand", flush=True)
        no_conflict_metrics, no_conflict_runs = run_ace(dataset, args, conflict_expand=False)
        rows.append(no_conflict_metrics)
        all_runs["ace_graph_no_conflict_expand"] = [run_to_jsonable(run) for run in no_conflict_runs]

    if not rows:
        raise SystemExit(f"No valid methods requested: {args.methods}")

    prefix = (
        f"{args.dataset}_{args.embedder}_{args.compressor}"
        f"_{args.ace_retriever}_dims{args.compress_dims}_nodes{args.top_k_nodes}_docs{args.max_expanded_docs}"
    )
    metrics_path = out_dir / f"{prefix}_metrics.csv"
    runs_path = out_dir / f"{prefix}_runs.json"
    write_metrics(metrics_path, rows)
    if not args.no_save_runs:
        runs_path.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")

    for row in rows:
        compact = ", ".join(f"{k}={v}" for k, v in row.items())
        print(compact)
    print(f"wrote {metrics_path}")
    if not args.no_save_runs:
        print(f"wrote {runs_path}")


if __name__ == "__main__":
    main()
