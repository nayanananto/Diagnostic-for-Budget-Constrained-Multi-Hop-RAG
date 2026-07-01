"""Incremental validity of answer-in-context (AiC) over retrieval recall.

Pre-empts the reviewer critique "answer-in-context is tautological with
correctness / it is just recall in disguise." We test, on the HotpotQA
per-question density CSVs, whether AiC predicts answer F1/EM *controlling for
retrieval*, and -- the cleanest version -- whether AiC still predicts quality
*among questions where retrieval already succeeded* (all gold retrieved). The
latter isolates the contribution of the packing step, orthogonal to retrieval.

All inference is cluster-robust on question id (each question appears once per
packing policy, so the pooled rows are not independent). The conditional-means
contrast uses a question-clustered bootstrap.

Usage:
    python scripts/incremental_validity_aic.py seed42_per_question.csv [seed13.csv seed7.csv]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for fi, p in enumerate(paths):
        df = pd.read_csv(p)
        df = df.rename(
            columns={
                "ans_in_context": "aic",
                "base_recall@5": "recall5",
                "base_all_gold@5": "allgold5",
                "gold_doc_reader_cov": "reader_cov",
                "gold_token_density": "tok_density",
            }
        )
        df["seed_idx"] = fi
        df["cluster"] = df["seed_idx"].astype(str) + ":" + df["qid"].astype(str)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Exclude the oracle/router pseudo-policies if present; keep fixed packers only.
    out = out[~out["policy"].str.contains("oracle|router", case=False, na=False)].copy()
    return out


def corr_block(df: pd.DataFrame) -> None:
    print("=== Simple correlations with outcomes (pooled policy x question rows) ===")
    feats = ["aic", "reader_cov", "allgold5", "recall5", "tok_density"]
    print(f"{'feature':>12}  {'r(F1)':>7}  {'r(EM)':>7}")
    for f in feats:
        rf = df[f].corr(df["f1"])
        re = df[f].corr(df["em"])
        print(f"{f:>12}  {rf:+.3f}  {re:+.3f}")
    print(f"\n  corr(aic, recall5) = {df['aic'].corr(df['recall5']):+.3f}  "
          f"(if this were ~1.0, AiC would be redundant with recall)")
    print(f"  n rows = {len(df)}, n questions = {df['cluster'].nunique()}\n")


def incremental_ols(df: pd.DataFrame) -> None:
    print("=== Incremental validity: OLS F1 ~ recall + AiC (cluster-robust on qid) ===")
    d = df.copy()
    d["z_recall"] = zscore(d["recall5"])
    d["z_aic"] = zscore(d["aic"])
    cl = d["cluster"]

    m_r = smf.ols("f1 ~ z_recall", data=d).fit(cov_type="cluster", cov_kwds={"groups": cl})
    m_ra = smf.ols("f1 ~ z_recall + z_aic", data=d).fit(cov_type="cluster", cov_kwds={"groups": cl})

    print(f"  recall only:      R^2 = {m_r.rsquared:.4f}")
    print(f"  recall + AiC:     R^2 = {m_ra.rsquared:.4f}   (delta R^2 = {m_ra.rsquared - m_r.rsquared:+.4f})")
    print("\n  Standardized coefficients in the joint model (cluster-robust p):")
    for name in ["z_recall", "z_aic"]:
        b = m_ra.params[name]
        p = m_ra.pvalues[name]
        ci = m_ra.conf_int().loc[name]
        print(f"    {name:>9}: beta={b:+.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.2e}")

    # Partial correlation of AiC with F1 controlling for recall.
    rx = smf.ols("z_aic ~ z_recall", data=d).fit().resid
    ry = smf.ols("f1 ~ z_recall", data=d).fit().resid
    pr = np.corrcoef(rx, ry)[0, 1]
    print(f"\n  partial corr(AiC, F1 | recall) = {pr:+.3f}\n")


def conditional_on_retrieval(df: pd.DataFrame, success_col: str, label: str, boot: int = 10000) -> None:
    print(f"=== Orthogonal-to-retrieval test: among questions where {label} ===")
    sub = df[df[success_col] >= 0.999].copy()
    n_rows = len(sub)
    if n_rows == 0:
        print("  (no rows meet the success condition)\n")
        return
    aic1 = sub[sub["aic"] >= 0.5]
    aic0 = sub[sub["aic"] < 0.5]
    frac_aic = sub["aic"].mean()
    print(f"  rows with {label}: {n_rows}  (of which AiC=1: {len(aic1)} = {frac_aic:.1%}; AiC=0: {len(aic0)})")
    print(f"  -> retrieval success does NOT guarantee answer-in-context: {1 - frac_aic:.1%} of "
          f"retrieval-successful rows still drop the answer in packing")
    if len(aic1) == 0 or len(aic0) == 0:
        print("  (one AiC cell empty; cannot contrast)\n")
        return
    # Precompute per-cluster numpy arrays once for a fast clustered bootstrap.
    clusters = sub["cluster"].unique().tolist()
    cl_idx = {c: i for i, c in enumerate(clusters)}
    sub = sub.assign(_ci=sub["cluster"].map(cl_idx))
    aic_mask = (sub["aic"].values >= 0.5)
    by_cluster: dict[int, dict[str, np.ndarray]] = {}
    for ci, g in sub.groupby("_ci"):
        gm = g["aic"].values >= 0.5
        by_cluster[ci] = {
            "f1_1": g["f1"].values[gm], "f1_0": g["f1"].values[~gm],
            "em_1": g["em"].values[gm], "em_0": g["em"].values[~gm],
        }
    n_cl = len(clusters)
    rng = np.random.default_rng(0)
    sample_mat = rng.integers(0, n_cl, size=(boot, n_cl))
    for metric in ("f1", "em"):
        m1 = sub[metric].values[aic_mask].mean()
        m0 = sub[metric].values[~aic_mask].mean()
        obs = m1 - m0
        a1 = [by_cluster[i][f"{metric}_1"] for i in range(n_cl)]
        a0 = [by_cluster[i][f"{metric}_0"] for i in range(n_cl)]
        diffs = np.empty(boot)
        for b in range(boot):
            picks = sample_mat[b]
            v1 = np.concatenate([a1[i] for i in picks])
            v0 = np.concatenate([a0[i] for i in picks])
            diffs[b] = (v1.mean() if v1.size else np.nan) - (v0.mean() if v0.size else np.nan)
        diffs = np.sort(diffs[~np.isnan(diffs)])
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        ratio = (m1 / m0) if m0 > 0 else float("inf")
        print(f"  {metric.upper()}: AiC=1 {m1:.3f} vs AiC=0 {m0:.3f}  "
              f"diff {obs:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}]  ({ratio:.1f}x)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_question_csvs", nargs="+", type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    df = load(args.per_question_csvs)
    print(f"\nLoaded {len(args.per_question_csvs)} file(s); {len(df)} fixed-policy rows, "
          f"{df['cluster'].nunique()} (seed,question) clusters.\n")
    corr_block(df)
    incremental_ols(df)
    conditional_on_retrieval(df, "allgold5", "ALL gold paragraphs were retrieved (all_gold@5=1)", args.boot)
    conditional_on_retrieval(df, "recall5", "retrieval recall@5 = 1", args.boot)


if __name__ == "__main__":
    main()
