"""Offline transfer analysis for Stage-3 router feature files.

This script does not rerun retrieval or Qwen. It trains lightweight routers on
one Stage-3 feature CSV and evaluates selected-policy answer metrics on another
feature CSV. Gold outcome columns are used only as training targets/evaluation
labels, not as deploy-time features.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


META_COLUMNS = {"qid", "question", "rule_policy", "oracle_policy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def candidate_names(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    names = [col[: -len("_f1")] for col in rows[0] if col.endswith("_f1")]
    return sorted(names)


def feature_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    cols: list[str] = []
    for col in rows[0]:
        if col in META_COLUMNS:
            continue
        if col.endswith("_em") or col.endswith("_f1"):
            continue
        cols.append(col)
    return cols


def as_float(value: str) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


def row_features(row: dict[str, str], cols: list[str]) -> list[float]:
    return [as_float(row[col]) for col in cols]


def policy_metrics(rows: list[dict[str, str]], selected: list[str], name: str, router_type: str) -> dict[str, Any]:
    ems: list[float] = []
    f1s: list[float] = []
    tokens: list[float] = []
    for row, policy in zip(rows, selected):
        ems.append(as_float(row[f"{policy}_em"]))
        f1s.append(as_float(row[f"{policy}_f1"]))
        tokens.append(as_float(row[f"{policy}_tokens"]))
    counts = Counter(selected)
    return {
        "policy": name,
        "router_type": router_type,
        "qwen_em": round(sum(ems) / max(1, len(ems)), 4),
        "qwen_f1": round(sum(f1s) / max(1, len(f1s)), 4),
        "avg_reader_evidence_tokens": round(sum(tokens) / max(1, len(tokens)), 2),
        "selection_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
    }


def fixed_and_existing_rows(rows: list[dict[str, str]], candidates: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        out.append(policy_metrics(rows, [candidate] * len(rows), candidate, "fixed_offline"))
    out.append(policy_metrics(rows, [row["rule_policy"] for row in rows], "router_rule", "rule_offline"))
    out.append(policy_metrics(rows, [row["oracle_policy"] for row in rows], "router_oracle", "oracle_offline"))
    return out


def train_oracle_classifier(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    cols: list[str],
    random_state: int,
) -> list[str]:
    from sklearn.tree import DecisionTreeClassifier

    x_train = [row_features(row, cols) for row in train_rows]
    y_train = [row["oracle_policy"] for row in train_rows]
    x_test = [row_features(row, cols) for row in test_rows]
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, class_weight="balanced", random_state=random_state)
    clf.fit(x_train, y_train)
    return list(clf.predict(x_test))


def train_f1_regressor(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    candidates: list[str],
    cols: list[str],
    random_state: int,
) -> list[str]:
    from sklearn.tree import DecisionTreeRegressor

    candidate_to_idx = {name: idx for idx, name in enumerate(candidates)}
    x_train: list[list[float]] = []
    y_train: list[float] = []
    for row in train_rows:
        base = row_features(row, cols)
        for candidate in candidates:
            one_hot = [0.0] * len(candidates)
            one_hot[candidate_to_idx[candidate]] = 1.0
            x_train.append(base + one_hot)
            y_train.append(as_float(row[f"{candidate}_f1"]))

    reg = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=15,
        random_state=random_state,
    )
    reg.fit(x_train, y_train)

    selected: list[str] = []
    for row in test_rows:
        base = row_features(row, cols)
        best_policy = candidates[0]
        best_key = (-1.0, 0.0)
        for candidate in candidates:
            one_hot = [0.0] * len(candidates)
            one_hot[candidate_to_idx[candidate]] = 1.0
            pred_f1 = float(reg.predict([base + one_hot])[0])
            key = (pred_f1, -as_float(row[f"{candidate}_tokens"]))
            if key > best_key:
                best_key = key
                best_policy = candidate
        selected.append(best_policy)
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    print(f"reading train features: {args.train_features}", flush=True)
    train_rows = read_rows(args.train_features)
    print(f"reading test features: {args.test_features}", flush=True)
    test_rows = read_rows(args.test_features)
    candidates = candidate_names(train_rows)
    test_candidates = candidate_names(test_rows)
    if candidates != test_candidates:
        raise SystemExit(f"candidate mismatch: train={candidates}, test={test_candidates}")

    cols = feature_columns(train_rows)
    rows = fixed_and_existing_rows(test_rows, candidates)

    print("training oracle-label tree", flush=True)
    clf_selected = train_oracle_classifier(train_rows, test_rows, cols, args.random_state)
    rows.append(policy_metrics(test_rows, clf_selected, "router_learned_oracle_tree", "learned_offline"))

    print("training F1 regression tree", flush=True)
    reg_selected = train_f1_regressor(train_rows, test_rows, candidates, cols, args.random_state)
    rows.append(policy_metrics(test_rows, reg_selected, "router_learned_f1_tree", "learned_offline"))

    for row in rows:
        print(", ".join(f"{key}={value}" for key, value in row.items()), flush=True)

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out_path, rows)
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
