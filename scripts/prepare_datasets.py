"""Dataset preparation helper.

This script intentionally does not hide dataset downloads behind the experiment
runner. Use it to validate external dataset access before launching expensive
runs.
"""

from __future__ import annotations

import argparse

from ace_rag.datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["toy", "hotpotqa", "musique_local", "ragbench"], default="toy")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--musique-path")
    parser.add_argument("--ragbench-subset")
    args = parser.parse_args()

    if args.dataset == "musique_local":
        ds = load_dataset("musique_local", path=args.musique_path, limit=args.limit)
    elif args.dataset == "ragbench":
        ds = load_dataset("ragbench", split=args.split, subset=args.ragbench_subset, limit=args.limit)
    elif args.dataset == "hotpotqa":
        ds = load_dataset("hotpotqa", split=args.split, limit=args.limit, seed=args.seed)
    else:
        ds = load_dataset("toy")
    print(ds.summary())
    print("sample question:", ds.questions[0])


if __name__ == "__main__":
    main()

