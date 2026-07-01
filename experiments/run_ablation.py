"""Run the core ACE-RAG ablation grid."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    base = [sys.executable, "-m", "experiments.run_mvp", "--dataset", "toy", "--embedder", "lexical"]
    grids = [
        ["--compressor", "identity", "--max-expanded-docs", "5"],
        ["--compressor", "identity", "--max-expanded-docs", "2"],
        ["--compressor", "truncate", "--compress-dims", "64", "--max-expanded-docs", "5"],
    ]
    for args in grids:
        subprocess.run(base + args, check=True)


if __name__ == "__main__":
    main()

