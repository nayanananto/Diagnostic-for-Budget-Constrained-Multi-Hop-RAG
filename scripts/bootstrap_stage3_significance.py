"""Paired bootstrap significance checks for Stage-3 policy comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing a Stage-3 predictions jsonl file.")
    parser.add_argument("--left-policy", required=True)
    parser.add_argument("--right-policy", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="analysis_results/stage3_bootstrap")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    norm_pred = normalize(prediction)
    return float(any(norm_pred == normalize(answer) for answer in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        answer_tokens = normalize(answer).split()
        if not answer_tokens:
            continue
        overlap = 0
        remaining = answer_tokens[:]
        for token in pred_tokens:
            if token in remaining:
                overlap += 1
                remaining.remove(token)
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def find_predictions(run_dir: Path) -> Path:
    matches = sorted(run_dir.rglob("*_predictions.jsonl"), key=lambda path: len(str(path)))
    if not matches:
        raise FileNotFoundError(f"No *_predictions.jsonl found under {run_dir}")
    return matches[0]


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            policy = item["policy"]
            qid = item["qid"]
            answers = item.get("gold_answers", [])
            rows[f"{policy}\t{qid}"] = {
                "qid": qid,
                "f1": token_f1(item.get("prediction", ""), answers),
                "em": exact_match(item.get("prediction", ""), answers),
            }
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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


def bootstrap(
    paired: list[dict[str, float]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(paired)
    observed_f1 = mean(row["left_f1"] - row["right_f1"] for row in paired)
    observed_em = mean(row["left_em"] - row["right_em"] for row in paired)
    f1_samples: list[float] = []
    em_samples: list[float] = []
    for _ in range(samples):
        draw = [paired[rng.randrange(n)] for _ in range(n)]
        f1_samples.append(mean(row["left_f1"] - row["right_f1"] for row in draw))
        em_samples.append(mean(row["left_em"] - row["right_em"] for row in draw))
    return {
        "questions": n,
        "observed_delta_f1": round(observed_f1, 6),
        "delta_f1_ci_low": round(percentile(f1_samples, 0.025), 6),
        "delta_f1_ci_high": round(percentile(f1_samples, 0.975), 6),
        "f1_positive_rate": round(sum(value > 0 for value in f1_samples) / max(1, len(f1_samples)), 6),
        "observed_delta_em": round(observed_em, 6),
        "delta_em_ci_low": round(percentile(em_samples, 0.025), 6),
        "delta_em_ci_high": round(percentile(em_samples, 0.975), 6),
        "em_positive_rate": round(sum(value > 0 for value in em_samples) / max(1, len(em_samples)), 6),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = find_predictions(run_dir)
    predictions = read_predictions(pred_path)

    qids = sorted(
        {
            key.split("\t", 1)[1]
            for key in predictions
            if key.startswith(f"{args.left_policy}\t") or key.startswith(f"{args.right_policy}\t")
        }
    )
    paired: list[dict[str, float]] = []
    for qid in qids:
        left = predictions.get(f"{args.left_policy}\t{qid}")
        right = predictions.get(f"{args.right_policy}\t{qid}")
        if left is None or right is None:
            continue
        paired.append(
            {
                "left_f1": float(left["f1"]),
                "right_f1": float(right["f1"]),
                "left_em": float(left["em"]),
                "right_em": float(right["em"]),
            }
        )

    summary = bootstrap(paired, args.samples, args.seed)
    summary.update(
        {
            "left_policy": args.left_policy,
            "right_policy": args.right_policy,
            "samples": args.samples,
            "seed": args.seed,
            "predictions_file": str(pred_path),
        }
    )
    write_csv(out_dir / "bootstrap_summary.csv", [summary])
    print(f"read {pred_path}")
    print(f"compared {summary['questions']} questions")
    print(f"wrote {out_dir / 'bootstrap_summary.csv'}")


if __name__ == "__main__":
    main()
