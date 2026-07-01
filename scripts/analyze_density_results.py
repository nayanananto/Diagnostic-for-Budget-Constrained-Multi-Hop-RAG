"""Stage-4 analysis: paired bootstrap + answer-density mediation.

Given a ``*_per_question.csv`` written by ``experiments.run_density_router``,
this reports:

  1. Paired bootstrap (EM and F1) for a set of policy contrasts. The headline
     check is ``chunk_submod`` vs the heuristic packers and vs the MMR baseline.
  2. The mediation analysis: correlation of each context-quality / retrieval
     feature with answer F1/EM, pooled over all policy-question rows, plus the
     conditional F1 given the answer is (not) present in the reader context.

Usage:
    python scripts/analyze_density_results.py <per_question.csv> \
        [--contrasts chunk_submod_160:chunk_mmr_160,chunk_submod_160:chunk_focused_160]

The point of (2) is the thesis claim that ``ans_in_context`` predicts answer
quality better than document ``recall@k`` does.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_pol: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pol[r["policy"]][r["qid"]] = r
    qids = sorted({r["qid"] for r in rows})
    return rows, by_pol, qids


def paired_boot(by_pol, qids, left, right, key, n=10000, seed=0):
    rng = random.Random(seed)
    L = {q: float(by_pol[left][q][key]) for q in qids if q in by_pol[left]}
    R = {q: float(by_pol[right][q][key]) for q in qids if q in by_pol[right]}
    common = [q for q in qids if q in L and q in R]
    diffs = [L[q] - R[q] for q in common]
    if not diffs:
        return 0.0, 0.0, 0.0, 1.0
    m = len(diffs)
    obs = sum(diffs) / m
    boots = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    lo, hi = boots[int(0.025 * n)], boots[int(0.975 * n)]
    p = 2 * min(sum(b <= 0 for b in boots), sum(b >= 0 for b in boots)) / n
    return obs, lo, hi, min(p, 1.0)


def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_question_csv", type=Path)
    ap.add_argument("--contrasts", default="")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    rows, by_pol, qids = load(args.per_question_csv)
    policies = list(by_pol)
    print(f"{len(qids)} questions; policies: {policies}\n")

    if args.contrasts:
        contrasts = [tuple(c.split(":")) for c in args.contrasts.split(",")]
    else:
        # Auto-build sensible contrasts if the expected policy names are present.
        def pick(sub):
            return next((p for p in policies if sub in p), None)
        cand = [
            (pick("chunk_submod"), pick("chunk_mmr")),
            (pick("chunk_submod"), pick("chunk_focused")),
            (pick("chunk_submod"), pick("chunk_packed")),
            (pick("chunk_mmr"), pick("chunk_focused")),
            (pick("ace_submod"), pick("ace_focused")),
        ]
        contrasts = [(a, b) for a, b in cand if a and b]

    print("=== Paired bootstrap (left - right) ===")
    for left, right in contrasts:
        if left not in by_pol or right not in by_pol:
            print(f"  [skip] {left} or {right} not in results")
            continue
        for key in ("f1", "em"):
            obs, lo, hi, p = paired_boot(by_pol, qids, left, right, key, n=args.boot)
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"{star} {left:>20} - {right:<20} {key.upper():>3}: {obs:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p={p:.3f}")
        print()

    feats = ["ans_in_context", "gold_token_density", "gold_doc_reader_cov", "base_recall@5", "base_all_gold@5"]
    feats = [f for f in feats if f in rows[0]]
    f1s = [float(r["f1"]) for r in rows]
    ems = [float(r["em"]) for r in rows]
    print("=== Feature -> answer-quality correlation (pooled) ===")
    for feat in feats:
        xs = [float(r[feat]) for r in rows]
        print(f"  corr({feat:>20}, F1) = {pearson(xs, f1s):+.4f}   corr(.,EM) = {pearson(xs, ems):+.4f}")

    print("\n=== Conditional F1 on mediator presence (pooled) ===")
    for flag in ("ans_in_context", "base_all_gold@5"):
        if flag not in rows[0]:
            continue
        on = [float(r["f1"]) for r in rows if float(r[flag]) >= 0.5]
        off = [float(r["f1"]) for r in rows if float(r[flag]) < 0.5]
        mon = sum(on) / len(on) if on else 0.0
        moff = sum(off) / len(off) if off else 0.0
        print(f"  {flag:>18}: F1 when=1: {mon:.4f} (n={len(on)})   when=0: {moff:.4f} (n={len(off)})   gap={mon - moff:+.4f}")


if __name__ == "__main__":
    main()
