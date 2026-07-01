"""Pooled multi-seed paired bootstrap over density-router per-question CSVs.

Pools several per-question CSVs (e.g. one per seed) by keying each row on
(file_index, qid), so paired differences are taken within a seed and then
pooled. Reports paired bootstrap (F1 and EM) for the standard packer contrasts
and the per-file best fixed policy.

Usage:
    python scripts/pool_multiseed_bootstrap.py \
        seed42_per_question.csv seed13_per_question.csv seed7_per_question.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

FIXED = [
    "chunk_packed_160", "chunk_focused_160", "chunk_mmr_160", "chunk_submod_160",
    "ace_focused_160", "ace_mmr_160", "ace_submod_160",
]
CONTRASTS = [
    ("chunk_submod_160", "chunk_mmr_160"),
    ("chunk_submod_160", "chunk_focused_160"),
    ("chunk_submod_160", "chunk_packed_160"),
    ("chunk_mmr_160", "chunk_focused_160"),
    ("ace_submod_160", "ace_focused_160"),
    ("chunk_submod_160", "ace_submod_160"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_question_csvs", nargs="+", type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    pool: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for fi, path in enumerate(args.per_question_csvs):
        for r in csv.DictReader(path.open(encoding="utf-8")):
            pool[(fi, r["qid"])][r["policy"]] = r
    keys = list(pool)
    print(f"pooled {len(keys)} (file,qid) instances from {len(args.per_question_csvs)} file(s)\n")

    def paired(left, right, metric, n, seed=0):
        rng = random.Random(seed)
        diffs = [
            float(pool[k][left][metric]) - float(pool[k][right][metric])
            for k in keys if left in pool[k] and right in pool[k]
        ]
        if not diffs:
            return None
        m = len(diffs)
        obs = sum(diffs) / m
        boots = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
        lo, hi = boots[int(.025 * n)], boots[int(.975 * n)]
        p = 2 * min(sum(b <= 0 for b in boots), sum(b >= 0 for b in boots)) / n
        return obs, lo, hi, min(p, 1.0), m

    print("=== Pooled paired bootstrap (left - right) ===")
    for left, right in CONTRASTS:
        for metric in ("f1", "em"):
            res = paired(left, right, metric, args.boot)
            if res is None:
                continue
            obs, lo, hi, p, m = res
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"{star} {left:>18} - {right:<18} {metric.upper():>3}: {obs:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] p={p:.3f} (n={m})")
        print()

    print("=== Per-file best fixed policy (by F1) ===")
    n_files = len(args.per_question_csvs)
    for fi in range(n_files):
        means = {}
        for pol in FIXED:
            vals = [float(pool[k][pol]["f1"]) for k in keys if k[0] == fi and pol in pool[k]]
            if vals:
                means[pol] = sum(vals) / len(vals)
        if not means:
            continue
        best = max(means, key=means.get)
        print(f"  file {fi}: best={best} (F1={means[best]:.4f})")


if __name__ == "__main__":
    main()
