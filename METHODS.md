# Experimental Method Plan

## Hypothesis

ACE-RAG should improve evidence coverage and faithfulness per token by retrieving compressed claim/entity nodes first, then adaptively expanding to source passages only when needed.

## Independent Variables

- Retrieval representation:
  - flat passages
  - claim/entity graph
  - claim/entity graph with conflict expansion
- Compression:
  - none
  - truncation
  - PCA
  - binary sign compression
- Retrieval budget:
  - top-k passages
  - top-k graph nodes
  - max expanded documents

## Dependent Variables

- `recall@k`: fraction of gold evidence documents retrieved.
- `all_gold@k`: whether all required supporting documents were retrieved.
- `avg_evidence_tokens`: retrieved evidence tokens before generation.
- `extractive_em` / `extractive_f1`: offline proxy reader quality.
- verifier decision rates:
  - answer
  - conflict
  - abstain

## MVP Baselines

1. **Chunk RAG**
   - Retrieve top-k full passages/documents.
   - No graph, no compression.

2. **ACE Graph**
   - Retrieve compressed claim/entity evidence units.
   - Expand to source passages.
   - Expand contradiction neighbors when present.

3. **ACE Graph Without Conflict Expansion**
   - Same graph retrieval, but contradiction edges are ignored at retrieval time.
   - This isolates the value of conflict-aware retrieval.

## Paper-Grade Baselines To Add

- BM25 RAG.
- Dense vector RAG with FAISS.
- Hybrid BM25 + dense RAG.
- Graph RAG without compressed retrieval.
- Context compression baseline such as LLMLingua/xRAG-style compression.
- Long-context RAG using all retrieved documents.

## Required Ablations

- Graph vs no graph.
- Compression vs no compression.
- Claim nodes vs entity nodes.
- Conflict edges removed.
- Adaptive expansion removed.
- Different max expanded documents.
- Different compressed dimensions.
- Small vs strong reader LLM.

## Dataset Order

1. Toy dataset for deterministic smoke tests.
2. HotpotQA validation slice for multi-hop evidence retrieval.
3. MuSiQue local export for harder multi-hop reasoning.
4. RAGBench for faithfulness-oriented RAG labels.
5. Later conflict datasets for contradiction-heavy evaluation.

## Go/No-Go Gate

Scale the project only if ACE-RAG shows at least one of these on HotpotQA/MuSiQue:

- higher `all_gold@5` than chunk RAG at similar or lower evidence-token budget;
- similar retrieval quality with meaningfully lower `avg_evidence_tokens`;
- better conflict detection/abstention on conflict-heavy examples.

