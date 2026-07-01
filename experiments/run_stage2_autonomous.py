"""Autonomous Stage-2 Kaggle pipeline for Qwen answer evaluation.

This script starts from the current project state:
  - HotpotQA retrieval signal is established.
  - Qwen generation exposed that evidence materialization is the bottleneck.

It retrieves once, loads Qwen once, then evaluates several reader-context
policies in a single run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

from ace_rag.datasets import load_dataset
from ace_rag.generator import HuggingFaceGenerator
from ace_rag.metrics import evaluate_retrieval
from experiments.run_qwen_eval import (
    generation_metrics,
    materialize_reader_context,
    retrieve_ace,
    retrieve_bm25,
    retrieve_chunk,
    retrieve_hybrid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa", choices=["toy", "hotpotqa", "musique_local"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--musique-path", default=None)
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
    parser.add_argument("--baselines", default="chunk,bm25,hybrid")
    parser.add_argument("--packed-budgets", default="160,220,280,340")
    parser.add_argument("--focused-budgets", default="220,280")
    parser.add_argument("--snippet-window", type=int, default=1)
    parser.add_argument("--max-snippets", type=int, default=8)
    parser.add_argument("--max-snippet-tokens", type=int, default=80)
    parser.add_argument("--out-dir", default="cloud_results")
    return parser.parse_args()


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


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


def append_predictions(path: Path, policy: str, reader_runs, predictions, dataset) -> None:
    by_qid = {q.qid: q for q in dataset.questions}
    with path.open("a", encoding="utf-8") as f:
        for run, pred in zip(reader_runs, predictions):
            q = by_qid[run.qid]
            f.write(
                json.dumps(
                    {
                        "policy": policy,
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


def evaluate_policy(policy: dict[str, Any], dataset, base_runs, reader, pred_path: Path) -> dict[str, Any]:
    method = policy["method"]
    context = policy["context"]
    print(f"[stage2] materializing policy={policy['name']}", flush=True)
    reader_runs = materialize_reader_context(
        dataset,
        base_runs,
        context,
        snippet_window=policy.get("snippet_window", 1),
        max_snippets=policy.get("max_snippets", 8),
        max_snippet_tokens=policy.get("max_snippet_tokens", 80),
        packed_token_budget=policy.get("packed_token_budget", 240),
    )
    print(f"[stage2] generating policy={policy['name']}", flush=True)
    predictions = reader.answer_many([(run.query, run) for run in reader_runs])
    append_predictions(pred_path, policy["name"], reader_runs, predictions, dataset)

    row: dict[str, Any] = {
        "policy": policy["name"],
        "method": method,
        "reader_context": context,
        "reader_model": reader.model_name,
        "ace_retriever": policy.get("ace_retriever", "none" if method != "ace_graph" else ""),
        "compressor": policy.get("compressor", "none" if method != "ace_graph" else ""),
        "hybrid_alpha": policy.get("hybrid_alpha", ""),
        "packed_token_budget": policy.get("packed_token_budget", ""),
    }
    row.update(evaluate_retrieval(base_runs, dataset.questions, k_values=(1, 2, 5)))
    row.update(generation_metrics(dataset, reader_runs, predictions))
    print("[stage2] " + ", ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return row


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "toy":
        dataset = load_dataset("toy")
    elif args.dataset == "musique_local":
        if not args.musique_path:
            raise SystemExit("--musique-path is required for musique_local")
        dataset = load_dataset("musique_local", path=args.musique_path, limit=args.limit)
    else:
        dataset = load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)
    print(dataset.summary(), flush=True)

    requested_baselines = {name.strip() for name in args.baselines.split(",") if name.strip()}
    valid_baselines = {"chunk", "bm25", "hybrid"}
    unknown_baselines = requested_baselines - valid_baselines
    if unknown_baselines:
        raise SystemExit(f"Unknown baselines requested: {sorted(unknown_baselines)}")

    baseline_runs: dict[str, Any] = {}
    if "chunk" in requested_baselines:
        print("[stage2] retrieving chunk baseline", flush=True)
        baseline_runs["chunk"] = retrieve_chunk(dataset, args)
        cleanup_cuda()
    if "bm25" in requested_baselines:
        print("[stage2] retrieving BM25 baseline", flush=True)
        baseline_runs["bm25"] = retrieve_bm25(dataset, args)
        cleanup_cuda()
    if "hybrid" in requested_baselines:
        print("[stage2] retrieving hybrid baseline", flush=True)
        baseline_runs["hybrid"] = retrieve_hybrid(dataset, args)
        cleanup_cuda()
    if not baseline_runs:
        raise SystemExit("At least one baseline is required. Use --baselines chunk,bm25,hybrid")

    print("[stage2] retrieving ACE graph", flush=True)
    ace_runs = retrieve_ace(dataset, args)
    cleanup_cuda()

    print("[stage2] loading Qwen reader once", flush=True)
    reader = HuggingFaceGenerator(
        model_name=args.reader_model,
        device=args.reader_device,
        batch_size=args.reader_batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    budgets = [int(x.strip()) for x in args.packed_budgets.split(",") if x.strip()]
    focused_budgets = [int(x.strip()) for x in args.focused_budgets.split(",") if x.strip()]
    policies: list[dict[str, Any]] = []
    for baseline in ("chunk", "bm25", "hybrid"):
        if baseline not in baseline_runs:
            continue
        extra = {"hybrid_alpha": args.hybrid_alpha} if baseline == "hybrid" else {}
        policies.extend(
            [
                {"name": f"{baseline}_sources", "method": baseline, "context": "sources", "base": baseline, **extra},
                {
                    "name": f"{baseline}_packed_280",
                    "method": baseline,
                    "context": "packed_snippets",
                    "base": baseline,
                    "packed_token_budget": 280,
                    **extra,
                },
            ]
        )
    policies.append(
        {
            "name": "ace_sources" if args.ace_retriever == "standard" else "ace_bridge_sources",
            "method": "ace_graph",
            "context": "sources",
            "base": "ace",
            "ace_retriever": args.ace_retriever,
            "compressor": f"{args.compressor}:{args.compress_dims}",
        },
    )
    for budget in budgets:
        policies.append(
            {
                "name": f"ace_packed_{budget}" if args.ace_retriever == "standard" else f"ace_bridge_packed_{budget}",
                "method": "ace_graph",
                "context": "packed_snippets",
                "base": "ace",
                "ace_retriever": args.ace_retriever,
                "compressor": f"{args.compressor}:{args.compress_dims}",
                "packed_token_budget": budget,
                "snippet_window": args.snippet_window,
                "max_snippets": args.max_snippets,
                "max_snippet_tokens": args.max_snippet_tokens,
            }
        )
    for budget in focused_budgets:
        policies.append(
            {
                "name": f"ace_focused_{budget}" if args.ace_retriever == "standard" else f"ace_bridge_focused_{budget}",
                "method": "ace_graph",
                "context": "focused_packed",
                "base": "ace",
                "ace_retriever": args.ace_retriever,
                "compressor": f"{args.compressor}:{args.compress_dims}",
                "packed_token_budget": budget,
                "snippet_window": args.snippet_window,
                "max_snippets": args.max_snippets,
                "max_snippet_tokens": args.max_snippet_tokens,
            }
        )

    metrics_path = out_dir / f"{args.dataset}_stage2_autonomous_{args.ace_retriever}_limit{args.limit}_metrics.csv"
    pred_path = out_dir / f"{args.dataset}_stage2_autonomous_{args.ace_retriever}_limit{args.limit}_predictions.jsonl"
    if pred_path.exists():
        pred_path.unlink()

    rows: list[dict[str, Any]] = []
    runs_by_base = {**baseline_runs, "ace": ace_runs}
    for policy in policies:
        base_runs = runs_by_base[policy["base"]]
        rows.append(evaluate_policy(policy, dataset, base_runs, reader, pred_path))
        write_csv(metrics_path, rows)

    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
