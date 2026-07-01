"""Label-style analysis for RAGBench PubMedQA Stage-3 predictions.

RAGBench PubMedQA stores long explanatory answers, while small instruction
readers often answer with short labels such as "yes", "no", or "unknown".
Standard token F1 therefore understates whether the reader chose the right
PubMedQA-style decision. This script adds a conservative secondary metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("yes", "no", "maybe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing a Stage-3 predictions jsonl file.")
    parser.add_argument("--baseline-policy", default="chunk_packed_160")
    parser.add_argument("--out-dir", default="analysis_results/pubmedqa_labels")
    return parser.parse_args()


def find_predictions(run_dir: Path) -> Path:
    matches = sorted(run_dir.rglob("*_predictions.jsonl"), key=lambda path: len(str(path)))
    if not matches:
        raise FileNotFoundError(f"No *_predictions.jsonl found under {run_dir}")
    return matches[0]


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    return " ".join(text.split())


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text


def label_prediction(text: str) -> str | None:
    norm = clean(text)
    if not norm:
        return None
    first = norm.split()[0]
    if first in {"yes", "yeah", "true"}:
        return "yes"
    if first in {"no", "false"}:
        return "no"
    if first in {"unknown", "unclear", "maybe", "inconclusive", "undetermined"}:
        return "maybe"
    if norm.startswith(("answer unknown", "cannot determine", "not enough information", "insufficient information")):
        return "maybe"
    if "unknown" in norm or "cannot determine" in norm or "insufficient information" in norm:
        return "maybe"
    return None


def label_gold(answer: str) -> tuple[str | None, str]:
    """Return a conservative label and the rule that produced it."""

    raw_first = first_sentence(answer)
    first = clean(raw_first)
    full = clean(answer)
    if not first:
        return None, "empty"

    if first.startswith("yes "):
        return "yes", "starts_yes"
    if first == "yes":
        return "yes", "starts_yes"
    if first.startswith("no "):
        return "no", "starts_no"
    if first == "no":
        return "no", "starts_no"

    maybe_patterns = [
        "cannot be definitively stated",
        "cannot be determined",
        "not clear",
        "unclear",
        "inconclusive",
        "insufficient evidence",
        "further research is needed",
        "more research is needed",
    ]
    if any(pattern in full for pattern in maybe_patterns):
        return "maybe", "uncertainty_phrase"

    no_patterns = [
        "does not",
        "do not",
        "did not",
        "is not",
        "are not",
        "was not",
        "were not",
        "has not",
        "have not",
        "may not",
        "not necessarily",
        "no significant",
        "no evidence",
        "not play",
        "not support",
        "not associated",
        "not a negative",
    ]
    if any(pattern in first for pattern in no_patterns):
        return "no", "negation_phrase"

    yes_patterns = [
        "does alter",
        "do influence",
        "does influence",
        "do play",
        "does play",
        "is associated",
        "are associated",
        "was associated",
        "were associated",
        "has an impact",
        "have an impact",
        "suggest that",
        "suggests that",
        "showed that",
        "shows that",
        "found that",
    ]
    if any(pattern in first for pattern in yes_patterns):
        return "yes", "affirmation_phrase"

    return None, "ambiguous"


def macro_f1(golds: list[str], preds: list[str]) -> float:
    scores: list[float] = []
    for label in LABELS:
        tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


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


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def analyze(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labeled_rows: list[dict[str, Any]] = []
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        answers = row.get("gold_answers") or []
        gold_label = None
        gold_rule = "missing"
        if answers:
            gold_label, gold_rule = label_gold(str(answers[0]))
        pred_label = label_prediction(str(row.get("prediction", "")))
        labeled = {
            "policy": row.get("policy", ""),
            "router_type": row.get("router_type", ""),
            "qid": row.get("qid", ""),
            "question": row.get("question", ""),
            "gold_label": gold_label or "",
            "gold_rule": gold_rule,
            "prediction_label": pred_label or "",
            "is_scorable": int(gold_label is not None and pred_label is not None),
            "label_correct": int(gold_label == pred_label) if gold_label is not None and pred_label is not None else "",
            "prediction": row.get("prediction", ""),
            "gold_answer": answers[0] if answers else "",
        }
        labeled_rows.append(labeled)
        by_policy[str(row.get("policy", ""))].append(labeled)

    summary_rows: list[dict[str, Any]] = []
    for policy, policy_rows in sorted(by_policy.items()):
        scorable = [row for row in policy_rows if row["is_scorable"]]
        golds = [str(row["gold_label"]) for row in scorable]
        preds = [str(row["prediction_label"]) for row in scorable]
        correct = sum(int(row["label_correct"]) for row in scorable)
        pred_counts = Counter(row["prediction_label"] or "unparsed" for row in policy_rows)
        gold_counts = Counter(row["gold_label"] or "ambiguous" for row in policy_rows)
        summary_rows.append(
            {
                "policy": policy,
                "questions": len(policy_rows),
                "scorable_questions": len(scorable),
                "gold_label_coverage": round(len(scorable) / max(1, len(policy_rows)), 4),
                "label_accuracy": round(correct / max(1, len(scorable)), 4),
                "label_macro_f1": round(macro_f1(golds, preds), 4) if scorable else 0.0,
                "gold_yes": gold_counts["yes"],
                "gold_no": gold_counts["no"],
                "gold_maybe": gold_counts["maybe"],
                "gold_ambiguous": gold_counts["ambiguous"],
                "pred_yes": pred_counts["yes"],
                "pred_no": pred_counts["no"],
                "pred_maybe": pred_counts["maybe"],
                "pred_unparsed": pred_counts["unparsed"],
            }
        )
    return summary_rows, labeled_rows


def add_baseline_deltas(summary_rows: list[dict[str, Any]], baseline_policy: str) -> None:
    baseline = next((row for row in summary_rows if row["policy"] == baseline_policy), None)
    if baseline is None:
        for row in summary_rows:
            row["delta_label_accuracy_vs_baseline"] = ""
            row["delta_label_macro_f1_vs_baseline"] = ""
        return
    base_acc = float(baseline["label_accuracy"])
    base_f1 = float(baseline["label_macro_f1"])
    for row in summary_rows:
        row["baseline_policy"] = baseline_policy
        row["delta_label_accuracy_vs_baseline"] = round(float(row["label_accuracy"]) - base_acc, 4)
        row["delta_label_macro_f1_vs_baseline"] = round(float(row["label_macro_f1"]) - base_f1, 4)


def write_markdown(path: Path, pred_path: Path, summary_rows: list[dict[str, Any]], baseline_policy: str) -> None:
    lines = [
        "# PubMedQA Label Analysis",
        "",
        f"Predictions: `{pred_path}`",
        "",
        "This is a secondary analysis for RAGBench PubMedQA because the run stores long explanatory gold answers while the reader usually emits short yes/no/unknown decisions.",
        "",
        "| Policy | Scorable | Accuracy | Macro F1 | Delta F1 vs Baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {scorable}/{questions} | {acc:.4f} | {f1:.4f} | {delta} |".format(
                policy=row["policy"],
                scorable=row["scorable_questions"],
                questions=row["questions"],
                acc=float(row["label_accuracy"]),
                f1=float(row["label_macro_f1"]),
                delta=row.get("delta_label_macro_f1_vs_baseline", ""),
            )
        )
    lines.extend(
        [
            "",
            f"Baseline policy: `{baseline_policy}`.",
            "",
            "Rows with ambiguous gold explanations are excluded from label accuracy and macro F1.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = find_predictions(run_dir)
    rows = read_rows(pred_path)
    summary_rows, labeled_rows = analyze(rows)
    add_baseline_deltas(summary_rows, args.baseline_policy)

    write_csv(out_dir / "pubmedqa_label_summary.csv", summary_rows)
    write_csv(out_dir / "pubmedqa_label_examples.csv", labeled_rows)
    write_markdown(out_dir / "pubmedqa_label_summary.md", pred_path, summary_rows, args.baseline_policy)
    print(f"read {pred_path}")
    print(f"wrote {out_dir / 'pubmedqa_label_summary.md'}")


if __name__ == "__main__":
    main()
