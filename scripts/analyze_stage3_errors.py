"""Question-level error analysis for Stage-3 ACE-RAG runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing a Stage-3 predictions jsonl file.")
    parser.add_argument("--left-policy", default="ace_packed_280")
    parser.add_argument("--right-policy", default="chunk_packed_280")
    parser.add_argument("--out-dir", default="analysis_results/stage3_errors")
    parser.add_argument("--max-examples", type=int, default=25)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        answer_tokens = normalize(answer).split()
        if not answer_tokens:
            continue
        pred_counts = Counter(pred_tokens)
        answer_counts = Counter(answer_tokens)
        overlap = sum((pred_counts & answer_counts).values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def exact_match(prediction: str, answers: list[str]) -> float:
    norm_pred = normalize(prediction)
    return float(any(norm_pred == normalize(answer) for answer in answers))


def find_file(run_dir: Path, suffix: str) -> Path:
    matches = sorted(run_dir.rglob(f"*{suffix}"), key=lambda path: len(str(path)))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} file found under {run_dir}")
    return matches[0]


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    by_policy_qid: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            policy = item["policy"]
            qid = item["qid"]
            answers = item.get("gold_answers", [])
            item["em"] = exact_match(item.get("prediction", ""), answers)
            item["f1"] = token_f1(item.get("prediction", ""), answers)
            by_policy_qid[f"{policy}\t{qid}"] = item
    return by_policy_qid


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


def compare_policies(
    predictions: dict[str, dict[str, Any]],
    left_policy: str,
    right_policy: str,
) -> list[dict[str, Any]]:
    qids = sorted(
        {
            key.split("\t", 1)[1]
            for key in predictions
            if key.startswith(f"{left_policy}\t") or key.startswith(f"{right_policy}\t")
        }
    )
    rows: list[dict[str, Any]] = []
    for qid in qids:
        left = predictions.get(f"{left_policy}\t{qid}")
        right = predictions.get(f"{right_policy}\t{qid}")
        if left is None or right is None:
            continue
        delta_f1 = float(left["f1"]) - float(right["f1"])
        delta_em = float(left["em"]) - float(right["em"])
        rows.append(
            {
                "qid": qid,
                "question": left.get("question", ""),
                "gold_answers": " | ".join(left.get("gold_answers", [])),
                "left_policy": left_policy,
                "right_policy": right_policy,
                "left_prediction": left.get("prediction", ""),
                "right_prediction": right.get("prediction", ""),
                "left_f1": round(float(left["f1"]), 6),
                "right_f1": round(float(right["f1"]), 6),
                "delta_f1": round(delta_f1, 6),
                "left_em": int(float(left["em"])),
                "right_em": int(float(right["em"])),
                "delta_em": int(delta_em),
                "left_docs": " | ".join(left.get("retrieved_doc_ids", [])),
                "right_docs": " | ".join(right.get("retrieved_doc_ids", [])),
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]], max_examples: int) -> None:
    wins = [row for row in rows if float(row["delta_f1"]) > 0]
    losses = [row for row in rows if float(row["delta_f1"]) < 0]
    ties = [row for row in rows if float(row["delta_f1"]) == 0]
    lines = [
        "# Stage-3 Error Analysis",
        "",
        f"Compared `{rows[0]['left_policy']}` against `{rows[0]['right_policy']}`." if rows else "No comparable rows.",
        "",
        f"- Left wins: {len(wins)}",
        f"- Right wins: {len(losses)}",
        f"- Ties: {len(ties)}",
        "",
        "## Largest Left Wins",
        "",
    ]
    for row in sorted(wins, key=lambda item: float(item["delta_f1"]), reverse=True)[:max_examples]:
        lines.extend(
            [
                f"### {row['qid']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold: {row['gold_answers']}",
                "",
                f"Left prediction: {row['left_prediction']}",
                "",
                f"Right prediction: {row['right_prediction']}",
                "",
            ]
        )
    lines.extend(["## Largest Right Wins", ""])
    for row in sorted(losses, key=lambda item: float(item["delta_f1"]))[:max_examples]:
        lines.extend(
            [
                f"### {row['qid']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold: {row['gold_answers']}",
                "",
                f"Left prediction: {row['left_prediction']}",
                "",
                f"Right prediction: {row['right_prediction']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    wins = [row for row in rows if float(row["delta_f1"]) > 0]
    losses = [row for row in rows if float(row["delta_f1"]) < 0]
    ties = [row for row in rows if float(row["delta_f1"]) == 0]
    summary = [
        {
            "left_policy": rows[0]["left_policy"] if rows else "",
            "right_policy": rows[0]["right_policy"] if rows else "",
            "questions": len(rows),
            "left_wins": len(wins),
            "right_wins": len(losses),
            "ties": len(ties),
            "mean_delta_f1": round(sum(float(row["delta_f1"]) for row in rows) / max(1, len(rows)), 6),
        }
    ]
    write_csv(path, summary)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = find_file(run_dir, "_predictions.jsonl")
    predictions = read_predictions(pred_path)
    rows = compare_policies(predictions, args.left_policy, args.right_policy)
    write_csv(out_dir / "policy_comparison_examples.csv", rows)
    write_summary(out_dir / "policy_comparison_summary.csv", rows)
    write_markdown(out_dir / "policy_comparison_examples.md", rows, args.max_examples)
    print(f"read {pred_path}")
    print(f"compared {len(rows)} questions")
    print(f"wrote {out_dir / 'policy_comparison_examples.md'}")


if __name__ == "__main__":
    main()
