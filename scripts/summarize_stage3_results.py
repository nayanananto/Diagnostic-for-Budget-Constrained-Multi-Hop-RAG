"""Summarize Stage-3 ACE-RAG metrics across Kaggle runs.

The script intentionally uses only the Python standard library so it can run in
Kaggle, Colab, or a fresh local checkout without installing pandas.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ("qwen_em", "qwen_f1", "avg_reader_evidence_tokens", "recall@5", "all_gold@5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", help="Directory containing Kaggle/Colab result folders.")
    parser.add_argument("--out-dir", default="analysis_results")
    parser.add_argument("--include-latest", action="store_true", help="Also read kaggle_results/latest.")
    return parser.parse_args()


def infer_run_fields(path: Path) -> dict[str, str]:
    text = str(path).replace("\\", "/")
    name = path.parent.name
    fields = {
        "source_file": text,
        "job": name,
        "dataset": "unknown",
        "limit": "",
        "seed": "",
        "analysis_group": "standard",
    }
    if "hotpotqa" in text.lower():
        fields["dataset"] = "hotpotqa"
    elif "musique" in text.lower():
        fields["dataset"] = "musique"

    seed_match = re.search(r"seed(\d+)", text)
    if seed_match:
        fields["seed"] = seed_match.group(1)
    limit_match = re.search(r"limit(\d+)", text)
    if limit_match:
        fields["limit"] = limit_match.group(1)
    budget_match = re.search(r"budget(\d+)", text)
    if budget_match:
        fields["analysis_group"] = f"budget_{budget_match.group(1)}"
    return fields


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(results_root: Path, include_latest: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("*metrics.csv")):
        normalized = str(path).replace("\\", "/")
        if not include_latest and "/latest/" in normalized:
            continue
        if "stage3_router" not in path.name:
            continue
        run_fields = infer_run_fields(path)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merged = dict(run_fields)
                merged.update(row)
                for metric in METRICS:
                    merged[metric] = to_float(str(merged.get(metric, "")))
                rows.append(merged)
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one metric row per logical run/policy.

    Kaggle publishes are cumulative: a later result commit often contains the
    metrics from earlier jobs as well as the new job. Without this pass, seed
    aggregates silently over-count older runs.
    """

    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        logical_run = row.get("seed") or row.get("job") or row.get("source_file", "")
        key = (
            str(row.get("dataset", "")),
            str(row.get("limit", "")),
            str(row.get("analysis_group", "")),
            str(logical_run),
            str(row.get("policy", "")),
            str(row.get("reader_model", "")),
        )
        previous = by_key.get(key)
        if previous is None or str(row.get("source_file", "")) > str(previous.get("source_file", "")):
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def sem(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) <= 1:
        return 0.0
    return stdev(clean) / math.sqrt(len(clean))


def metric_summary(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    return mean(clean), sem(clean)


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


def build_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["limit"], row["analysis_group"], row["policy"])].append(row)

    aggregate: list[dict[str, Any]] = []
    for (dataset, limit, analysis_group, policy), group_rows in sorted(grouped.items()):
        seeds = sorted({row.get("seed", "") for row in group_rows if row.get("seed", "")})
        out: dict[str, Any] = {
            "dataset": dataset,
            "limit": limit,
            "analysis_group": analysis_group,
            "policy": policy,
            "n_runs": len(group_rows),
            "seeds": ",".join(seeds),
            "router_type": group_rows[0].get("router_type", ""),
            "method": group_rows[0].get("method", ""),
            "reader_context": group_rows[0].get("reader_context", ""),
        }
        for metric in METRICS:
            metric_mean, metric_sem = metric_summary([float(row[metric]) for row in group_rows])
            out[f"{metric}_mean"] = round(metric_mean, 6)
            out[f"{metric}_sem"] = round(metric_sem, 6)
        aggregate.append(out)
    return aggregate


def build_deltas(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["limit"], row["analysis_group"], row["policy"]): row for row in aggregate}
    rows: list[dict[str, Any]] = []
    for dataset, limit, analysis_group in sorted(
        {(row["dataset"], row["limit"], row["analysis_group"]) for row in aggregate}
    ):
        policies = {
            row["policy"]
            for row in aggregate
            if row["dataset"] == dataset and row["limit"] == limit and row["analysis_group"] == analysis_group
        }
        budgets = sorted(
            {
                match.group(1)
                for policy in policies
                for match in [re.match(r"chunk_packed_(\d+)$", policy)]
                if match
            },
            key=int,
        )
        comparisons: list[tuple[str, str]] = []
        for budget in budgets:
            comparisons.extend(
                [
                    (f"ace_packed_{budget}", f"chunk_packed_{budget}"),
                    (f"ace_focused_{budget}", f"chunk_packed_{budget}"),
                ]
            )
        for chunk_policy in sorted([policy for policy in policies if policy.startswith("chunk_packed_")]):
            comparisons.extend(
                [
                    ("router_rule", chunk_policy),
                    ("router_oracle", chunk_policy),
                ]
            )
        if analysis_group == "standard" and "chunk_packed_280" in policies:
            comparisons.extend(
                [
                    ("ace_focused_220", "chunk_packed_280"),
                ]
            )
        seen: set[tuple[str, str]] = set()
        for left, right in comparisons:
            if (left, right) in seen:
                continue
            seen.add((left, right))
            left_row = by_key.get((dataset, limit, analysis_group, left))
            right_row = by_key.get((dataset, limit, analysis_group, right))
            if left_row is None or right_row is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "limit": limit,
                    "analysis_group": analysis_group,
                    "left_policy": left,
                    "right_policy": right,
                    "delta_qwen_f1": round(float(left_row["qwen_f1_mean"]) - float(right_row["qwen_f1_mean"]), 6),
                    "delta_qwen_em": round(float(left_row["qwen_em_mean"]) - float(right_row["qwen_em_mean"]), 6),
                    "delta_tokens": round(
                        float(left_row["avg_reader_evidence_tokens_mean"])
                        - float(right_row["avg_reader_evidence_tokens_mean"]),
                        6,
                    ),
                    "left_runs": left_row["n_runs"],
                    "right_runs": right_row["n_runs"],
                }
            )
    return rows


def write_markdown(path: Path, aggregate: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage-3 Result Summary",
        "",
        "## Aggregates",
        "",
        "| dataset | limit | group | policy | runs | F1 mean | EM mean | tokens mean |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {dataset} | {limit} | {group} | {policy} | {n_runs} | {qwen_f1_mean:.4f} | {qwen_em_mean:.4f} | {tokens:.2f} |".format(
                dataset=row["dataset"],
                limit=row["limit"],
                group=row["analysis_group"],
                policy=row["policy"],
                n_runs=row["n_runs"],
                qwen_f1_mean=float(row["qwen_f1_mean"]),
                qwen_em_mean=float(row["qwen_em_mean"]),
                tokens=float(row["avg_reader_evidence_tokens_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Key Deltas",
            "",
            "| dataset | limit | group | comparison | delta F1 | delta EM | delta tokens |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        lines.append(
            "| {dataset} | {limit} | {group} | {left} vs {right} | {f1:.4f} | {em:.4f} | {tokens:.2f} |".format(
                dataset=row["dataset"],
                limit=row["limit"],
                group=row["analysis_group"],
                left=row["left_policy"],
                right=row["right_policy"],
                f1=float(row["delta_qwen_f1"]),
                em=float(row["delta_qwen_em"]),
                tokens=float(row["delta_tokens"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(results_root, args.include_latest)
    deduped_rows = dedupe_rows(rows)
    aggregate = build_aggregate(deduped_rows)
    deltas = build_deltas(aggregate)

    write_csv(out_dir / "stage3_raw_rows.csv", rows)
    write_csv(out_dir / "stage3_deduped_rows.csv", deduped_rows)
    write_csv(out_dir / "stage3_aggregate.csv", aggregate)
    write_csv(out_dir / "stage3_policy_deltas.csv", deltas)
    write_markdown(out_dir / "stage3_summary.md", aggregate, deltas)

    print(f"read {len(rows)} metric rows")
    print(f"kept {len(deduped_rows)} deduplicated rows")
    print(f"wrote {out_dir / 'stage3_summary.md'}")


if __name__ == "__main__":
    main()
