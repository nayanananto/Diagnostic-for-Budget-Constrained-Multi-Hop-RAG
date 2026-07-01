"""Mechanism decomposition: WHY does one packer beat another?

Pairs two policies per question (default chunk_submod vs chunk_focused) and
decomposes the F1 gain by the answer-in-context transition:

    gained_ans (0->1) | lost_ans (1->0) | both_have (1->1) | neither (0->0)

The thesis claim is that a principled packer's gain is concentrated in the
``gained_ans`` bucket -- i.e. it wins by flipping the answer INTO the reader
context, driven by better complementary gold-document coverage rather than by
raising raw gold-token density. The bucket contributions sum to the overall
mean F1 delta, so the decomposition is exact.

Usage:
    python scripts/decompose_packer_winloss.py <per_question.csv> \
        [--left chunk_submod_160] [--right chunk_focused_160]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_question_csv", type=Path)
    ap.add_argument("--left", default="chunk_submod_160")
    ap.add_argument("--right", default="chunk_focused_160")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.per_question_csv.open(encoding="utf-8")))
    by_pol: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pol[r["policy"]][r["qid"]] = r

    L, R = args.left, args.right
    if L not in by_pol or R not in by_pol:
        raise SystemExit(f"policies not found. available: {list(by_pol)}")
    qids = [q for q in by_pol[L] if q in by_pol[R]]

    def f(p, q, k):
        return float(by_pol[p][q][k])

    buckets: dict[str, list[float]] = {
        "gained_ans (0->1)": [], "lost_ans (1->0)": [], "both_have (1->1)": [], "neither (0->0)": [],
    }
    for q in qids:
        a_sub, a_foc = f(L, q, "ans_in_context"), f(R, q, "ans_in_context")
        d_f1 = f(L, q, "f1") - f(R, q, "f1")
        if a_sub > 0.5 and a_foc < 0.5:
            buckets["gained_ans (0->1)"].append(d_f1)
        elif a_sub < 0.5 and a_foc > 0.5:
            buckets["lost_ans (1->0)"].append(d_f1)
        elif a_sub > 0.5 and a_foc > 0.5:
            buckets["both_have (1->1)"].append(d_f1)
        else:
            buckets["neither (0->0)"].append(d_f1)

    overall = sum(f(L, q, "f1") - f(R, q, "f1") for q in qids) / len(qids)
    print(f"Pairing {L} vs {R} over {len(qids)} questions")
    print(f"Overall mean F1 delta = {overall:+.4f}\n")
    print("=== F1 gain decomposed by answer-in-context transition ===")
    tot = 0.0
    for name, ds in buckets.items():
        n = len(ds)
        contrib = sum(ds) / len(qids)
        tot += contrib
        mean = sum(ds) / n if n else 0.0
        print(f"  {name:>18}: n={n:3d}  meanF1delta={mean:+.4f}  contrib={contrib:+.4f}")
    print(f"  {'TOTAL':>18}: contrib sum = {tot:+.4f}")

    gained = len(buckets["gained_ans (0->1)"])
    lost = len(buckets["lost_ans (1->0)"])
    print(f"\nAnswer-in-context flips: gained {gained}, lost {lost}, NET {gained - lost:+d}")

    gqs = [q for q in qids if f(L, q, "ans_in_context") > 0.5 and f(R, q, "ans_in_context") < 0.5]
    if gqs:
        dcov = sum(f(L, q, "gold_doc_reader_cov") - f(R, q, "gold_doc_reader_cov") for q in gqs) / len(gqs)
        dden = sum(f(L, q, "gold_token_density") - f(R, q, "gold_token_density") for q in gqs) / len(gqs)
        print(f"On the {len(gqs)} flipped-in questions: mean delta reader-coverage={dcov:+.4f}, token-density={dden:+.4f}")

    both_L = sum(1 for q in qids if f(L, q, "gold_doc_reader_cov") >= 0.999)
    both_R = sum(1 for q in qids if f(R, q, "gold_doc_reader_cov") >= 0.999)
    print(f"All gold docs in reader context: {L}={both_L}, {R}={both_R}")


if __name__ == "__main__":
    main()
