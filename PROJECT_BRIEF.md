# ACE-RAG Project Brief

Working title:

**ACE-RAG: Adaptive Compressed Evidence Graphs for Faithful and Efficient Retrieval-Augmented Generation**

## Core Research Idea

Standard RAG retrieves flat text chunks. This is inefficient because many retrieved chunks contain redundant or irrelevant text, and it is brittle because the model receives evidence without explicit structure, provenance, or conflict awareness.

ACE-RAG retrieves from an **Adaptive Compressed Evidence Graph** instead of directly retrieving only full chunks. Documents are transformed into typed evidence units:

- passage nodes;
- claim nodes;
- entity nodes;
- source/provenance links;
- optional contradiction/conflict links.

The system retrieves compact claim/entity evidence first, then expands only to the source passages needed for generation. The central goal is to preserve answer-relevant evidence while reducing retrieved evidence budget.

## Main Hypothesis

RAG can become more efficient and potentially more faithful if retrieval operates over structured evidence units rather than flat chunks.

The strongest current version of the claim is:

> ACE graph retrieval preserves most multi-hop evidence coverage while using substantially less retrieved evidence than chunk-based RAG.

## Current Status

The project has completed **Stage 1: retrieval and evidence-budget validation**.

Implemented:

- project scaffold;
- HotpotQA loader;
- toy dataset;
- evidence graph builder;
- dense embedding retrieval;
- chunk RAG baseline;
- ACE graph retriever;
- simple compression methods;
- Kaggle-ready runner;
- result logging;
- Qwen-based Stage 2 script.

The strongest early finding is qualitative:

> On HotpotQA, ACE graph retrieval keeps most of the chunk baseline's evidence coverage while using much less evidence text.

Compression status:

- PCA compression hurt retrieval quality.
- Full-dimension truncation matched identity, as expected.
- moderate truncation looks promising.
- more aggressive truncation degraded retrieval.

This means the **graph evidence allocation** idea is currently stronger than the **compressed embedding** idea. Compression remains important, but it needs careful tuning or a better learned compression method.

## Current Best Method Direction

Use ACE graph retrieval with:

- BGE-small embeddings;
- claim/entity node retrieval;
- source passage expansion;
- a moderate truncation setting if it preserves retrieval quality;
- small evidence budget for generation.

Do not over-focus on conflict expansion yet. HotpotQA is not a conflict-heavy dataset, so conflict-specific claims need a later benchmark.

## Current Caveats

The project is not yet paper-ready.

Missing pieces:

- real LLM answer generation evaluation;
- faithfulness/citation evaluation;
- stronger baselines;
- MuSiQue or another harder multi-hop dataset;
- conflict-heavy benchmark;
- stronger verifier;
- more rigorous ablations;
- statistical confidence over larger runs.

## Next Stage

The current next validation stage is **MuSiQue**.

HotpotQA now has a credible Stage-2 signal. MuSiQue should test whether the result generalizes to harder connected multi-hop questions. The MuSiQue plan is in:

```text
MUSIQUE_STAGE_DESIGN.md
```

## Longer-Term Paper Plan

A credible paper needs three pillars:

1. **Representation contribution**
   - Typed evidence graph retrieval rather than flat chunk retrieval.

2. **Efficiency contribution**
   - Lower retrieved evidence budget and smaller semantic representations.

3. **Trustworthiness contribution**
   - Better faithfulness, citation support, ambiguity handling, or conflict handling.

The current experiments support the first two partially. The third still needs work.

## Where To Look

- `RESEARCH_LOG.md`: qualitative research history.
- `RESULTS.md`: important numeric summaries.
- `CLOUD_RUN.md`: Kaggle/Colab instructions.
- `experiments/run_mvp.py`: retrieval-stage experiments.
- `experiments/run_qwen_eval.py`: Qwen generation-stage experiments.
- `notebooks/ACE_RAG_Kaggle_Runbook.ipynb`: Kaggle runbook.
