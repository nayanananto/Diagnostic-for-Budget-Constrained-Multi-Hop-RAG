"""Stage-3 adaptive evidence routing evaluation.

This runner tests the next research hypothesis after fixed ACE policies:
different questions need different compact evidence policies. It evaluates:

  - fixed compact baselines;
  - a rule-based router that uses retrieval/context features only;
  - an oracle router upper bound that chooses the best candidate answer per item.

The oracle is not deployable. It is a diagnostic for whether routing has enough
headroom to be worth improving.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ace_rag.datasets import load_dataset
from ace_rag.generator import HuggingFaceGenerator
from ace_rag.metrics import evaluate_retrieval, exact_match, token_f1
from ace_rag.schema import CorpusDataset, RetrievalRun
from ace_rag.text import lexical_overlap, tokenize
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
    parser.add_argument("--dataset", default="hotpotqa", choices=["toy", "hotpotqa", "musique_local", "ragbench"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--ragbench-subset", default=None)
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
    parser.add_argument("--snippet-window", type=int, default=1)
    parser.add_argument("--max-snippets", type=int, default=8)
    parser.add_argument("--max-snippet-tokens", type=int, default=80)
    parser.add_argument("--chunk-budget", type=int, default=280)
    parser.add_argument("--bm25-budget", type=int, default=280)
    parser.add_argument("--hybrid-budget", type=int, default=280)
    parser.add_argument("--ace-packed-budget", type=int, default=280)
    parser.add_argument(
        "--ace-focused-budget",
        type=int,
        default=0,
        help="Use 0 for default: 220 for standard ACE, 280 for bridge ACE.",
    )
    parser.add_argument(
        "--router-candidates",
        default="",
        help="Optional comma-separated candidate policy names. Empty means all compact candidates.",
    )
    parser.add_argument("--router-ace-margin", type=float, default=0.03)
    parser.add_argument("--router-hybrid-margin", type=float, default=0.04)
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


def load_from_args(args: argparse.Namespace) -> CorpusDataset:
    if args.dataset == "toy":
        return load_dataset("toy")
    if args.dataset == "musique_local":
        if not args.musique_path:
            raise SystemExit("--musique-path is required for musique_local")
        return load_dataset("musique_local", path=args.musique_path, limit=args.limit)
    if args.dataset == "ragbench":
        return load_dataset("ragbench", split=args.split, subset=args.ragbench_subset, limit=args.limit)
    return load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)


def build_candidate_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    ace_prefix = "ace_bridge" if args.ace_retriever == "bridge" else "ace"
    focused_budget = args.ace_focused_budget
    if focused_budget <= 0:
        focused_budget = 280 if args.ace_retriever == "bridge" else 220

    policies: list[dict[str, Any]] = [
        {
            "name": f"chunk_packed_{args.chunk_budget}",
            "method": "chunk",
            "context": "packed_snippets",
            "base": "chunk",
            "packed_token_budget": args.chunk_budget,
            "compressor": "none",
            "ace_retriever": "none",
        },
        {
            "name": f"bm25_packed_{args.bm25_budget}",
            "method": "bm25",
            "context": "packed_snippets",
            "base": "bm25",
            "packed_token_budget": args.bm25_budget,
            "compressor": "none",
            "ace_retriever": "none",
        },
        {
            "name": f"hybrid_packed_{args.hybrid_budget}",
            "method": "hybrid",
            "context": "packed_snippets",
            "base": "hybrid",
            "packed_token_budget": args.hybrid_budget,
            "compressor": "none",
            "ace_retriever": "none",
            "hybrid_alpha": args.hybrid_alpha,
        },
        {
            "name": f"{ace_prefix}_packed_{args.ace_packed_budget}",
            "method": "ace_graph",
            "context": "packed_snippets",
            "base": "ace",
            "packed_token_budget": args.ace_packed_budget,
            "compressor": f"{args.compressor}:{args.compress_dims}",
            "ace_retriever": args.ace_retriever,
        },
        {
            "name": f"{ace_prefix}_focused_{focused_budget}",
            "method": "ace_graph",
            "context": "focused_packed",
            "base": "ace",
            "packed_token_budget": focused_budget,
            "compressor": f"{args.compressor}:{args.compress_dims}",
            "ace_retriever": args.ace_retriever,
        },
    ]

    requested = {name.strip() for name in args.router_candidates.split(",") if name.strip()}
    if not requested:
        return policies
    by_name = {policy["name"]: policy for policy in policies}
    missing = requested - set(by_name)
    if missing:
        raise SystemExit(f"Unknown router candidates: {sorted(missing)}. Available: {sorted(by_name)}")
    return [policy for policy in policies if policy["name"] in requested]


def materialize_policy(dataset: CorpusDataset, base_runs: list[RetrievalRun], policy: dict[str, Any], args: argparse.Namespace) -> list[RetrievalRun]:
    return materialize_reader_context(
        dataset,
        base_runs,
        policy["context"],
        snippet_window=args.snippet_window,
        max_snippets=args.max_snippets,
        max_snippet_tokens=args.max_snippet_tokens,
        packed_token_budget=policy["packed_token_budget"],
    )


def fixed_policy_row(
    policy: dict[str, Any],
    dataset: CorpusDataset,
    base_runs: list[RetrievalRun],
    reader_runs: list[RetrievalRun],
    predictions: list[str],
    reader_model: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy": policy["name"],
        "router_type": "fixed",
        "method": policy["method"],
        "reader_context": policy["context"],
        "reader_model": reader_model,
        "ace_retriever": policy.get("ace_retriever", "none"),
        "compressor": policy.get("compressor", "none"),
        "hybrid_alpha": policy.get("hybrid_alpha", ""),
        "packed_token_budget": policy.get("packed_token_budget", ""),
        "candidate_policies": "",
        "selection_counts": "",
    }
    row.update(evaluate_retrieval(base_runs, dataset.questions, k_values=(1, 2, 5)))
    row.update(generation_metrics(dataset, reader_runs, predictions))
    return row


def router_row(
    name: str,
    router_type: str,
    dataset: CorpusDataset,
    selected_runs: list[RetrievalRun],
    selected_predictions: list[str],
    selected_policies: list[str],
    candidate_names: list[str],
    reader_model: str,
) -> dict[str, Any]:
    counts = Counter(selected_policies)
    row: dict[str, Any] = {
        "policy": name,
        "router_type": router_type,
        "method": "adaptive_router",
        "reader_context": "routed",
        "reader_model": reader_model,
        "ace_retriever": "",
        "compressor": "",
        "hybrid_alpha": "",
        "packed_token_budget": "",
        "candidate_policies": "|".join(candidate_names),
        "selection_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
    }
    row.update(evaluate_retrieval(selected_runs, dataset.questions, k_values=(1, 2, 5)))
    row.update(generation_metrics(dataset, selected_runs, selected_predictions))
    return row


def token_count(run: RetrievalRun) -> int:
    return sum(len(tokenize(hit.text)) for hit in run.hits)


def retrieval_confidence(run: RetrievalRun, dataset: CorpusDataset) -> float:
    if not run.hits:
        return 0.0
    top_score = float(run.hits[0].score)
    second_score = float(run.hits[1].score) if len(run.hits) > 1 else 0.0
    score_margin = max(0.0, min(1.0, (top_score - second_score) / max(abs(top_score), 1e-6)))
    hit_overlap = max((lexical_overlap(run.query, hit.text) for hit in run.hits), default=0.0)
    title_overlap = 0.0
    for doc_id in run.retrieved_doc_ids:
        doc = dataset.documents.get(doc_id)
        if doc is not None:
            title_overlap = max(title_overlap, lexical_overlap(run.query, doc.title))
    doc_coverage = min(1.0, len(set(run.retrieved_doc_ids)) / 5.0)
    return 0.45 * hit_overlap + 0.25 * title_overlap + 0.15 * score_margin + 0.15 * doc_coverage


def jaccard(left: list[str], right: list[str]) -> float:
    lset = set(left)
    rset = set(right)
    if not lset and not rset:
        return 0.0
    return len(lset & rset) / max(1, len(lset | rset))


def pick_existing(names: list[str], prefix: str, contains: str = "") -> str | None:
    for name in names:
        if name.startswith(prefix) and contains in name:
            return name
    return None


def choose_rule_policy(
    qid: str,
    dataset: CorpusDataset,
    candidate_names: list[str],
    candidate_outputs: dict[str, dict[str, Any]],
    base_by_name: dict[str, dict[str, RetrievalRun]],
    args: argparse.Namespace,
) -> str:
    chunk_name = pick_existing(candidate_names, "chunk_", "packed")
    bm25_name = pick_existing(candidate_names, "bm25_", "packed")
    hybrid_name = pick_existing(candidate_names, "hybrid_", "packed")
    ace_focused_name = pick_existing(candidate_names, "ace_", "focused")
    ace_packed_name = pick_existing(candidate_names, "ace_", "packed")

    fallback = chunk_name or hybrid_name or ace_packed_name or ace_focused_name or candidate_names[0]
    chunk_run = base_by_name.get("chunk", {}).get(qid)
    hybrid_run = base_by_name.get("hybrid", {}).get(qid)
    ace_run = base_by_name.get("ace", {}).get(qid)

    chunk_conf = retrieval_confidence(chunk_run, dataset) if chunk_run is not None else 0.0
    hybrid_conf = retrieval_confidence(hybrid_run, dataset) if hybrid_run is not None else 0.0
    ace_conf = retrieval_confidence(ace_run, dataset) if ace_run is not None else 0.0

    agreement = jaccard(ace_run.retrieved_doc_ids, chunk_run.retrieved_doc_ids) if ace_run and chunk_run else 0.0
    ace_reader = candidate_outputs.get(ace_focused_name or ace_packed_name or "", {}).get("reader_by_qid", {}).get(qid)
    chunk_reader = candidate_outputs.get(chunk_name or "", {}).get("reader_by_qid", {}).get(qid)
    ace_tokens = token_count(ace_reader) if ace_reader is not None else 10**9
    chunk_tokens = token_count(chunk_reader) if chunk_reader is not None else 10**9

    # Hybrid can rescue cases where dense and lexical evidence looks stronger than chunk alone.
    if hybrid_name and hybrid_conf >= chunk_conf + args.router_hybrid_margin and hybrid_conf >= ace_conf:
        return hybrid_name

    # BM25 is rarely the best fixed policy here, but keep it as a fallback for lexical-heavy questions.
    bm25_run = base_by_name.get("bm25", {}).get(qid)
    bm25_conf = retrieval_confidence(bm25_run, dataset) if bm25_run is not None else 0.0
    if bm25_name and bm25_conf >= max(chunk_conf, hybrid_conf, ace_conf) + args.router_hybrid_margin:
        return bm25_name

    # ACE is selected only when its retrieval signal is clearly better than chunk.
    # The earlier permissive rule over-selected focused ACE and degraded results.
    if ace_focused_name and ace_conf >= chunk_conf + args.router_ace_margin and ace_tokens <= 0.85 * chunk_tokens:
        return ace_focused_name

    if ace_packed_name and ace_conf >= chunk_conf + args.router_ace_margin:
        return ace_packed_name

    # When ACE and chunk agree on evidence, focused ACE is a safe compact choice
    # only if ACE is still at least slightly stronger and substantially cheaper.
    if (
        ace_focused_name
        and agreement >= 0.5
        and ace_conf >= chunk_conf + (args.router_ace_margin / 2.0)
        and ace_tokens <= 0.75 * chunk_tokens
    ):
        return ace_focused_name

    return fallback


def choose_oracle_policy(
    qid: str,
    question_answers: list[str],
    candidate_names: list[str],
    candidate_outputs: dict[str, dict[str, Any]],
) -> str:
    best_name = candidate_names[0]
    best_key = (-1.0, -1.0, 0)
    for name in candidate_names:
        pred = candidate_outputs[name]["prediction_by_qid"][qid]
        run = candidate_outputs[name]["reader_by_qid"][qid]
        f1 = token_f1(pred, question_answers)
        em = exact_match(pred, question_answers)
        key = (f1, em, -token_count(run))
        if key > best_key:
            best_key = key
            best_name = name
    return best_name


def build_selected(
    dataset: CorpusDataset,
    candidate_names: list[str],
    candidate_outputs: dict[str, dict[str, Any]],
    select_fn,
) -> tuple[list[RetrievalRun], list[str], list[str]]:
    selected_runs: list[RetrievalRun] = []
    selected_predictions: list[str] = []
    selected_policies: list[str] = []
    for question in dataset.questions:
        chosen = select_fn(question)
        if chosen not in candidate_names:
            raise ValueError(f"router selected unavailable policy {chosen!r}")
        selected_policies.append(chosen)
        selected_runs.append(candidate_outputs[chosen]["reader_by_qid"][question.qid])
        selected_predictions.append(candidate_outputs[chosen]["prediction_by_qid"][question.qid])
    return selected_runs, selected_predictions, selected_policies


def write_predictions(
    path: Path,
    dataset: CorpusDataset,
    candidate_outputs: dict[str, dict[str, Any]],
    router_outputs: dict[str, tuple[list[RetrievalRun], list[str], list[str]]],
) -> None:
    by_qid = {q.qid: q for q in dataset.questions}
    with path.open("w", encoding="utf-8") as f:
        for policy_name, output in candidate_outputs.items():
            for run in output["reader_runs"]:
                pred = output["prediction_by_qid"][run.qid]
                question = by_qid[run.qid]
                f.write(
                    json.dumps(
                        {
                            "policy": policy_name,
                            "router_type": "fixed",
                            "qid": run.qid,
                            "question": question.text,
                            "gold_answers": question.answers,
                            "prediction": pred,
                            "retrieved_doc_ids": run.retrieved_doc_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        for router_name, (runs, predictions, selected_policies) in router_outputs.items():
            for run, pred, selected_policy in zip(runs, predictions, selected_policies):
                question = by_qid[run.qid]
                f.write(
                    json.dumps(
                        {
                            "policy": router_name,
                            "router_type": "adaptive",
                            "selected_policy": selected_policy,
                            "qid": run.qid,
                            "question": question.text,
                            "gold_answers": question.answers,
                            "prediction": pred,
                            "retrieved_doc_ids": run.retrieved_doc_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def write_router_features(
    path: Path,
    dataset: CorpusDataset,
    candidate_names: list[str],
    candidate_outputs: dict[str, dict[str, Any]],
    base_by_name: dict[str, dict[str, RetrievalRun]],
    rule_policies: list[str],
    oracle_policies: list[str],
) -> None:
    """Persist routing features plus gold-derived labels for offline analysis."""

    rule_by_qid = {q.qid: policy for q, policy in zip(dataset.questions, rule_policies)}
    oracle_by_qid = {q.qid: policy for q, policy in zip(dataset.questions, oracle_policies)}
    rows: list[dict[str, Any]] = []
    for question in dataset.questions:
        qid = question.qid
        chunk_run = base_by_name.get("chunk", {}).get(qid)
        bm25_run = base_by_name.get("bm25", {}).get(qid)
        hybrid_run = base_by_name.get("hybrid", {}).get(qid)
        ace_run = base_by_name.get("ace", {}).get(qid)
        row: dict[str, Any] = {
            "qid": qid,
            "question": question.text,
            "rule_policy": rule_by_qid[qid],
            "oracle_policy": oracle_by_qid[qid],
            "chunk_confidence": retrieval_confidence(chunk_run, dataset) if chunk_run else 0.0,
            "bm25_confidence": retrieval_confidence(bm25_run, dataset) if bm25_run else 0.0,
            "hybrid_confidence": retrieval_confidence(hybrid_run, dataset) if hybrid_run else 0.0,
            "ace_confidence": retrieval_confidence(ace_run, dataset) if ace_run else 0.0,
            "ace_chunk_doc_jaccard": jaccard(ace_run.retrieved_doc_ids, chunk_run.retrieved_doc_ids)
            if ace_run and chunk_run
            else 0.0,
        }
        for name in candidate_names:
            run = candidate_outputs[name]["reader_by_qid"][qid]
            pred = candidate_outputs[name]["prediction_by_qid"][qid]
            row[f"{name}_tokens"] = token_count(run)
            row[f"{name}_doc_count"] = len(set(run.retrieved_doc_ids))
            row[f"{name}_em"] = exact_match(pred, question.answers)
            row[f"{name}_f1"] = token_f1(pred, question.answers)
        rows.append(row)
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_args(args)
    print(dataset.summary(), flush=True)

    print("[stage3] retrieving chunk baseline", flush=True)
    chunk_runs = retrieve_chunk(dataset, args)
    cleanup_cuda()

    print("[stage3] retrieving BM25 baseline", flush=True)
    bm25_runs = retrieve_bm25(dataset, args)
    cleanup_cuda()

    print("[stage3] retrieving hybrid baseline", flush=True)
    hybrid_runs = retrieve_hybrid(dataset, args)
    cleanup_cuda()

    print("[stage3] retrieving ACE graph", flush=True)
    ace_runs = retrieve_ace(dataset, args)
    cleanup_cuda()

    base_runs_by_name = {
        "chunk": chunk_runs,
        "bm25": bm25_runs,
        "hybrid": hybrid_runs,
        "ace": ace_runs,
    }
    base_by_name = {name: {run.qid: run for run in runs} for name, runs in base_runs_by_name.items()}

    print("[stage3] loading Qwen reader once", flush=True)
    reader = HuggingFaceGenerator(
        model_name=args.reader_model,
        device=args.reader_device,
        batch_size=args.reader_batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    policies = build_candidate_policies(args)
    candidate_outputs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for policy in policies:
        print(f"[stage3] materializing policy={policy['name']}", flush=True)
        base_runs = base_runs_by_name[policy["base"]]
        reader_runs = materialize_policy(dataset, base_runs, policy, args)
        print(f"[stage3] generating policy={policy['name']}", flush=True)
        predictions = reader.answer_many([(run.query, run) for run in reader_runs])
        candidate_outputs[policy["name"]] = {
            "policy": policy,
            "reader_runs": reader_runs,
            "reader_by_qid": {run.qid: run for run in reader_runs},
            "predictions": predictions,
            "prediction_by_qid": {run.qid: pred for run, pred in zip(reader_runs, predictions)},
        }
        row = fixed_policy_row(policy, dataset, base_runs, reader_runs, predictions, reader.model_name)
        rows.append(row)
        print("[stage3] " + ", ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    candidate_names = [policy["name"] for policy in policies]

    rule_runs, rule_predictions, rule_policies = build_selected(
        dataset,
        candidate_names,
        candidate_outputs,
        lambda question: choose_rule_policy(
            question.qid,
            dataset,
            candidate_names,
            candidate_outputs,
            base_by_name,
            args,
        ),
    )
    rule_row = router_row(
        "router_rule",
        "rule",
        dataset,
        rule_runs,
        rule_predictions,
        rule_policies,
        candidate_names,
        reader.model_name,
    )
    rows.append(rule_row)
    print("[stage3] " + ", ".join(f"{k}={v}" for k, v in rule_row.items()), flush=True)

    oracle_runs, oracle_predictions, oracle_policies = build_selected(
        dataset,
        candidate_names,
        candidate_outputs,
        lambda question: choose_oracle_policy(
            question.qid,
            question.answers,
            candidate_names,
            candidate_outputs,
        ),
    )
    oracle_row = router_row(
        "router_oracle",
        "oracle",
        dataset,
        oracle_runs,
        oracle_predictions,
        oracle_policies,
        candidate_names,
        reader.model_name,
    )
    rows.append(oracle_row)
    print("[stage3] " + ", ".join(f"{k}={v}" for k, v in oracle_row.items()), flush=True)

    metrics_path = out_dir / f"{args.dataset}_stage3_router_{args.ace_retriever}_limit{args.limit}_metrics.csv"
    pred_path = out_dir / f"{args.dataset}_stage3_router_{args.ace_retriever}_limit{args.limit}_predictions.jsonl"
    features_path = out_dir / f"{args.dataset}_stage3_router_{args.ace_retriever}_limit{args.limit}_router_features.csv"
    write_csv(metrics_path, rows)
    write_predictions(
        pred_path,
        dataset,
        candidate_outputs,
        {
            "router_rule": (rule_runs, rule_predictions, rule_policies),
            "router_oracle": (oracle_runs, oracle_predictions, oracle_policies),
        },
    )
    write_router_features(
        features_path,
        dataset,
        candidate_names,
        candidate_outputs,
        base_by_name,
        rule_policies,
        oracle_policies,
    )
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {pred_path}", flush=True)
    print(f"wrote {features_path}", flush=True)


if __name__ == "__main__":
    main()
