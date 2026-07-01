# Running ACE-RAG on Kaggle or Google Colab

## Recommendation

Use **Kaggle first** for this project.

Reason: the current experiments are batch-style retrieval/embedding jobs, and Kaggle makes it easy to keep outputs under `/kaggle/working`. A single **P100** or **T4** is enough for the first HotpotQA/MuSiQue runs. Kaggle's dual T4 option will not automatically double speed unless we add multi-process embedding; the current runner uses one process/model instance.

Practical choice:

1. **Kaggle P100** if available.
2. **Kaggle T4 x2** if P100 is not available, or if P100 throws a CUDA kernel-image error; it will likely behave like one T4 until multi-GPU embedding is added.
3. **Colab T4** if Kaggle quota is exhausted or Colab is more convenient.

Colab's own FAQ says GPU availability and usage limits vary over time, so do not design the first experiments around a specific Colab GPU.

## CUDA Kernel-Image Error

If you see:

```text
CUDA error: no kernel image is available for execution on the device
cudaErrorNoKernelImageForDevice
```

This is usually a PyTorch/CUDA wheel mismatch with the selected GPU architecture. It is common on older **P100/Pascal** runtimes when the installed CUDA kernels were not built for that device.

Fastest fix:

1. Stop the Kaggle session.
2. Change accelerator from **P100** to **T4 x2**.
3. Restart the notebook from the first cell.

If you must stay on P100, try reinstalling a compatible PyTorch build before installing the project dependencies:

```python
!python -m pip uninstall -y torch torchvision torchaudio xformers
!python -m pip install -q --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
!python -m pip install -q -r requirements-cloud.txt
!python -m scripts.cloud_check
```

Emergency CPU fallback, only to verify code correctness:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

Then run the toy test. Do not use CPU for large dense HotpotQA runs unless you are only debugging.

## Kaggle Setup

1. Zip and upload the `ace_rag_research` folder as a Kaggle Dataset, or upload it directly into a notebook session.
2. In notebook settings:
   - Accelerator: `GPU P100` or `GPU T4 x2`
   - Internet: `On`
3. Run the cells from [notebooks/ACE_RAG_Cloud_Quickstart.ipynb](notebooks/ACE_RAG_Cloud_Quickstart.ipynb).

For staged experiment runs, use [notebooks/ACE_RAG_Kaggle_Runbook.ipynb](notebooks/ACE_RAG_Kaggle_Runbook.ipynb). It has separate cells for smoke testing, the HotpotQA baseline, ACE-only tuning, compression sweeps, metrics display, and result zipping.

Minimal Kaggle cells if you do not use the notebook:

```python
import glob
import os
import shutil
from pathlib import Path

candidates = glob.glob("/kaggle/input/**/ace_rag_research", recursive=True)
if candidates:
    src = Path(candidates[0])
    dst = Path("/kaggle/working/ace_rag_research")
    if not dst.exists():
        shutil.copytree(src, dst)
else:
    dst = Path("/kaggle/working/ace_rag_research")

os.chdir(dst)
print(Path.cwd())
```

```python
!python -m pip install -q -r requirements-cloud.txt
!python -m scripts.cloud_check
```

```python
!python -m experiments.run_mvp --config configs/toy.yaml --out-dir cloud_results
```

```python
!python -m experiments.run_mvp \
  --config configs/hotpotqa.yaml \
  --limit 200 \
  --out-dir cloud_results
```

```python
!zip -r ace_rag_cloud_results.zip cloud_results
```

## Colab Setup

1. Runtime > Change runtime type > T4 GPU.
2. Upload `ace_rag_research.zip` or clone/copy the folder into `/content`.
3. Run:

```python
from google.colab import files
uploaded = files.upload()  # upload ace_rag_research.zip
```

```python
!unzip -q ace_rag_research.zip -d /content
%cd /content/ace_rag_research
!python -m pip install -q -r requirements-cloud.txt
!python -m scripts.cloud_check
```

```python
!python -m experiments.run_mvp --config configs/toy.yaml --out-dir cloud_results
!python -m experiments.run_mvp --config configs/hotpotqa.yaml --limit 200 --out-dir cloud_results
```

## First Real Run

Start with:

```bash
python -m experiments.run_mvp --config configs/hotpotqa.yaml --limit 200 --out-dir cloud_results
```

Then scale:

```bash
python -m experiments.run_mvp --config configs/hotpotqa.yaml --limit 1000 --out-dir cloud_results
```

For paper-grade runs, do not jump straight to `limit 1000`. First confirm:

- dataset downloads work;
- GPU is detected;
- results CSV is written;
- ACE graph and chunk RAG both complete;
- `avg_evidence_tokens` and `all_gold@5` look sane.

## Faster Tuning Runs

After one full run has produced the chunk baseline, use `--methods ace_graph` for tuning. This avoids rerunning chunk RAG and the no-conflict variant every time.

```bash
python -m experiments.run_mvp \
  --config configs/hotpotqa.yaml \
  --limit 1000 \
  --compressor identity \
  --top-k-nodes 48 \
  --max-expanded-docs 6 \
  --methods ace_graph \
  --out-dir cloud_results
```

The runner now batches query embeddings, so GPU use should appear in bursts during embedding. Retrieval/ranking and graph expansion are still CPU-heavy, so GPU utilization will not stay high for the whole run.

## Free Qwen Reader Evaluation

Use this after retrieval tuning. Start small because local generation is much slower than retrieval.

```bash
python -m experiments.run_qwen_eval \
  --dataset hotpotqa \
  --limit 50 \
  --methods chunk,ace_graph \
  --compressor truncate \
  --compress-dims 320 \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --reader-context sources \
  --out-dir cloud_results
```

If that works, scale to `--limit 100`, then `--limit 200`.

For the intended ACE reader format, use snippets instead of full source passages:

```bash
python -m experiments.run_qwen_eval \
  --dataset hotpotqa \
  --limit 200 \
  --methods chunk,ace_graph \
  --compressor truncate \
  --compress-dims 320 \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --reader-context snippets \
  --snippet-window 1 \
  --max-snippets 8 \
  --max-snippet-tokens 80 \
  --out-dir cloud_results
```

## Autonomous Stage-2 Pipeline

For the current research stage, prefer the autonomous runner:

```bash
python -m experiments.run_stage2_autonomous \
  --dataset hotpotqa \
  --limit 200 \
  --baselines chunk,bm25,hybrid \
  --compressor truncate \
  --compress-dims 320 \
  --packed-budgets 160,220,280,340 \
  --focused-budgets 220,280 \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --out-dir cloud_results
```

It runs the main Stage-2 policies in one pass and loads Qwen only once.

## Stage-3 Adaptive Router

Use this after fixed-policy Stage-2 results show mixed winners. It evaluates compact fixed policies, a rule-based router, and an oracle router upper bound.

HotpotQA router check:

```bash
python -m experiments.run_stage3_router \
  --dataset hotpotqa \
  --split validation \
  --limit 1000 \
  --seed 42 \
  --embed-device cuda \
  --compressor truncate \
  --compress-dims 320 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --ace-retriever standard \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --out-dir cloud_results_stage3_hotpotqa
```

Inspect:

```bash
cat cloud_results_stage3_hotpotqa/*stage3_router*metrics.csv
```

MuSiQue bridge router check:

```bash
python -m experiments.run_stage3_router \
  --dataset musique_local \
  --musique-path data/raw/musique.jsonl \
  --limit 500 \
  --embed-device cuda \
  --compressor truncate \
  --compress-dims 320 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --ace-retriever bridge \
  --bridge-seed-nodes 12 \
  --bridge-terms 10 \
  --bridge-weight 0.35 \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --ace-focused-budget 280 \
  --out-dir cloud_results_stage3_musique
```

Inspect:

```bash
cat cloud_results_stage3_musique/*stage3_router*metrics.csv
```

For the next thesis-quality baseline check, run a tighter version first:

```bash
python -m experiments.run_stage2_autonomous \
  --dataset hotpotqa \
  --split validation \
  --limit 1000 \
  --seed 42 \
  --baselines chunk,bm25,hybrid \
  --embed-device cuda \
  --compressor truncate \
  --compress-dims 320 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --packed-budgets 280 \
  --focused-budgets 220 \
  --out-dir cloud_results_baselines_hotpotqa
```

Then inspect:

```bash
cat cloud_results_baselines_hotpotqa/*stage2_autonomous*metrics.csv
```

## Bridge-Aware MuSiQue Commands

Retrieval-only check:

```bash
python -m experiments.run_mvp \
  --dataset musique_local \
  --musique-path data/raw/musique.jsonl \
  --limit 300 \
  --embedder sentence-transformers \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --device cuda \
  --compressor truncate \
  --compress-dims 320 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --ace-retriever bridge \
  --bridge-seed-nodes 12 \
  --bridge-terms 10 \
  --bridge-weight 0.35 \
  --methods chunk,ace_graph \
  --no-save-runs \
  --out-dir cloud_results_musique_bridge
```

Stage-2 Qwen check:

```bash
python -m experiments.run_stage2_autonomous \
  --dataset musique_local \
  --musique-path data/raw/musique.jsonl \
  --limit 200 \
  --baselines chunk,bm25,hybrid \
  --embed-device cuda \
  --compressor truncate \
  --compress-dims 320 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --ace-retriever bridge \
  --bridge-seed-nodes 12 \
  --bridge-terms 10 \
  --bridge-weight 0.35 \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --packed-budgets 280 \
  --focused-budgets 280 \
  --snippet-window 1 \
  --max-snippets 8 \
  --max-snippet-tokens 80 \
  --out-dir cloud_results_musique_bridge
```
