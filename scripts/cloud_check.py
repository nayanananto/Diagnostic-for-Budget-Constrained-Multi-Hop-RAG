"""Print cloud runtime diagnostics for Kaggle/Colab runs."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    print("python:", platform.python_version())
    print("platform:", platform.platform())
    print("cwd:", os.getcwd())
    print("kaggle:", bool(os.environ.get("KAGGLE_URL_BASE") or os.path.exists("/kaggle")))
    print("colab:", bool(os.environ.get("COLAB_RELEASE_TAG") or os.path.exists("/content")))
    print("nvidia-smi:", shutil.which("nvidia-smi") or "not found")
    if shutil.which("nvidia-smi"):
        print(run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]))
    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        print("cuda_device_count:", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            print(f"cuda:{idx}:", torch.cuda.get_device_name(idx))
    except Exception as exc:
        print("torch unavailable:", exc)


if __name__ == "__main__":
    main()

