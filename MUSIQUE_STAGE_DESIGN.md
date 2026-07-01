# MuSiQue Stage Design

Purpose:

Use MuSiQue as the first serious external validation dataset after HotpotQA. HotpotQA showed that ACE graph retrieval plus packed evidence can work, but MuSiQue is harder and more explicitly designed to require connected multi-hop reasoning. This makes it a stronger test of whether ACE-RAG is genuinely useful rather than overfit to HotpotQA-style evidence.

Primary paper question:

> Does ACE-RAG's graph retrieval and packed evidence strategy generalize from HotpotQA to harder composed multi-hop questions?

## Why MuSiQue Matters

MuSiQue was built to reduce shortcut-solving in multi-hop QA. The dataset composes connected single-hop questions into harder multi-hop questions, so systems often need to retrieve and combine multiple pieces of evidence. This directly matches the ACE-RAG thesis:

- graph retrieval should help connect evidence;
- packed snippets should help the reader see the right reasoning context;
- lower evidence budgets should still preserve answer quality if the graph is doing useful selection.

Reference anchor:

- MuSiQue paper: https://arxiv.org/abs/2108.00573

## Dataset Setup

Use a local MuSiQue JSON/JSONL export.

Expected file location on Kaggle:

```text
data/raw/musique.jsonl
```

The loader accepts common MuSiQue-style fields:

- `id` or `qid`
- `question`
- `answer` or `answers`
- `paragraphs`, `contexts`, or `context`
- paragraph fields such as:
  - `title`
  - `paragraph_text`
  - `text`
  - `is_supporting`

The existing loader is:

```text
ace_rag.datasets.load_musique_local
```

## Experimental Stages

### Stage M0: Loader Smoke Test

Goal:

Confirm the local MuSiQue file parses correctly and produces questions, documents, and gold supporting documents.

Command:

```bash
python -m scripts.prepare_datasets \
  --dataset musique_local \
  --musique-path data/raw/musique.jsonl \
  --limit 5
```

Pass criteria:

- examples load without schema errors;
- questions are non-empty;
- documents are non-empty;
- at least some questions have gold supporting document IDs.

### Stage M1: Retrieval-Only Check

Goal:

Compare chunk retrieval against ACE graph retrieval before paying the Qwen generation cost.

Command:

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
  --methods chunk,ace_graph \
  --no-save-runs \
  --out-dir cloud_results_musique
```

Decision:

If ACE retrieval collapses badly on MuSiQue, do not run Qwen yet. Improve retrieval or graph construction first.

### Stage M2: Qwen Stage-2 Autonomous Run

Goal:

Run the same end-to-end comparison as HotpotQA:

- chunk full sources;
- chunk packed snippets;
- ACE full sources;
- ACE packed snippets;
- ACE focused packed snippets.

Start small:

```bash
python -m experiments.run_stage2_autonomous \
  --dataset musique_local \
  --musique-path data/raw/musique.jsonl \
  --limit 200 \
  --embed-device cuda \
  --compressor truncate \
  --compress-dims 320 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --reader-device cuda \
  --reader-batch-size 4 \
  --packed-budgets 220,280 \
  --focused-budgets 220,280 \
  --snippet-window 1 \
  --max-snippets 8 \
  --max-snippet-tokens 80 \
  --out-dir cloud_results_musique
```

Then scale:

```bash
python -m experiments.run_stage2_autonomous \
  --dataset musique_local \
  --musique-path data/raw/musique.jsonl \
  --limit 500 \
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
  --snippet-window 1 \
  --max-snippets 8 \
  --max-snippet-tokens 80 \
  --out-dir cloud_results_musique
```

## Main Comparisons

The important comparisons are:

1. **ACE packed vs chunk sources**
   - Does ACE approach or beat full chunk RAG with fewer reader tokens?

2. **ACE packed vs chunk packed**
   - Is the graph doing better selection than packing snippets from chunk retrieval?

3. **ACE packed vs ACE sources**
   - Does evidence packing improve or preserve answer quality while reducing context?

4. **ACE focused vs ACE packed**
   - Does coverage-aware packing improve over the simpler packed policy?

5. **Bridge ACE vs standard ACE**
   - Does first-hop bridge evidence help on harder composed multi-hop questions?

## Go/No-Go Criteria

Good MuSiQue outcome:

- ACE packed or focused packed is close to chunk sources in answer quality;
- ACE uses clearly less reader context than chunk sources;
- ACE packed beats or matches chunk packed at similar budget.

Mixed but useful outcome:

- ACE saves context but trails chunk packed;
- this means the graph representation helps efficiency but packing/reranking needs improvement.

Bad outcome:

- ACE retrieval coverage drops sharply;
- Qwen answer quality collapses even with source passages;
- this means MuSiQue needs stronger graph construction or retrieval expansion before generation experiments.

## Expected Risk

MuSiQue is harder than HotpotQA. It may expose weaknesses in the current sentence-level claim extraction and simple lexical evidence packing. If results are weak, the likely fix is not a bigger Qwen model first. The likely fix is better evidence graph construction and sub-question/entity-aware retrieval.

## How This Fits The Thesis

If MuSiQue supports the HotpotQA trend, the thesis becomes much stronger:

> ACE-RAG generalizes to harder connected multi-hop QA and can reduce reader context without sacrificing answer quality.

If MuSiQue does not support the trend, it still gives a useful thesis insight:

> Graph retrieval works on HotpotQA-style evidence, but harder composed reasoning requires better graph construction or explicit reasoning-aware packing.

## Bridge-Aware ACE Retrieval

The first MuSiQue runs showed that simply retrieving more evidence is not enough. The bounded next method is bridge-aware ACE retrieval:

```text
question
-> first-hop ACE retrieval
-> extract bridge entities and terms from first-hop evidence
-> rerank graph nodes with a bridge-expanded query
-> pack evidence for Qwen
```

Retrieval-only bridge check:

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

Stage-2 bridge check:

```bash
python -m experiments.run_stage2_autonomous \
  --dataset musique_local \
  --musique-path data/raw/musique.jsonl \
  --limit 200 \
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
