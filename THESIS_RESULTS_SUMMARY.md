# Thesis Results Summary

Working title:

**ACE-RAG: Graph-Based Retrieval with Budget-Aware Evidence Packing for Efficient Multi-Hop RAG**

## Plain Summary

The project started with the idea that RAG should not retrieve only flat text chunks. Instead, documents can be converted into a graph of smaller evidence units such as claims, entities, and source passages. Retrieval can then happen over compact graph evidence, and the system can pack only the most useful evidence into the reader context.

The experiments so far show that this idea is promising on HotpotQA under a reader-context budget. With the stronger Qwen reader, packed ACE is now the best deployable method on the larger HotpotQA aggregate, and the focused ACE variant is strongest when the context budget is tight. Full chunk-source reading remains an important reference point, but the core result is now clearer: ACE-RAG is most useful when the reader must operate under a restricted evidence budget.

The same method is more fragile on MuSiQue, which is a harder composed multi-hop dataset. A bridge-aware ACE variant now gives the best answer-quality result among the tested budgeted policies, but the absolute scores remain low and retrieval coverage is not a clean win over chunk retrieval. This points to the next research problem: better multi-hop graph construction or reasoning-aware retrieval.

The first RAGBench PubMedQA cross-dataset check produced a useful but cautious signal. Raw exact-match and token-F1 are not meaningful there because the stored gold answers are long medical explanations while the reader emits short yes/no/unknown decisions. A conservative label-style analysis gives ACE a small advantage over chunk packing, but this should be treated as tentative until the pipeline preserves the original PubMedQA labels.

The RAGBench CovidQA check is stronger. Under the same tight context budget, ACE focused and ACE packed both beat packed chunk on answer F1, and paired bootstrap intervals stay positive. ExpertQA gives a smaller but still positive result under the same comparison. EManual is directionally positive in aggregate, but repeated question IDs make the paired check weaker. TechQA is mixed: focused ACE is slightly better in aggregate, while packed ACE is slightly worse and the paired intervals cross zero. These results provide the first non-HotpotQA support for the budgeted ACE claim, with CovidQA and ExpertQA as the cleaner evidence.

## Stage-4 Update (2026-06-28): The Packing Objective Is the Lever — Candidate New Headline

A controlled 2×2 of {chunk, ACE} representation × {focused heuristic, principled submodular} packer, at a fixed 160-token reader budget on HotpotQA-500 with the Qwen2.5-3B reader, produced the project's first *statistically clean* method win — but it relocates the contribution.

**Status (empirical arc complete, Stages 4-7): HotpotQA win CONFIRMED across 3 seeds {42,13,7} with the MMR baseline; chunk_submod best fixed policy every seed, all key contrasts paired-bootstrap significant. Scope mapped on 4 datasets: the packer win is SPECIFIC to HotpotQA's conjunction of multi-hop structure + sufficient retrieval + binding-but-not-extreme budget. It does NOT transfer to single-pass RAGBench (no multi-hop) or MuSiQue (retrieval-bottlenecked, all_gold@5=0.18). The answer-in-context DIAGNOSTIC, however, generalizes across all four (r=0.39-0.54 vs recall ~0.31). The honest contribution = a general diagnostic + a conditional packer win + a mechanistic "when does packing help" analysis. Next step is writing, not more runs.**

What was found (3-seed pooled paired bootstrap, 1,500 instances):

- A new **budgeted-submodular evidence packer** (monotone submodular objective: relevance + query-term coverage + saturated representativeness + concave source-diversity; cost-scaled greedy with the Lin–Bilmes singleton fallback for a constant-factor guarantee) applied to **chunk** retrieval beats every other packer: **vs MMR EM +4.6 / F1 +4.2 pts (p<0.001); vs focused EM +2.9 (p=0.004) / F1 +2.2 (p=0.029); vs packed EM +5.3 / F1 +5.1 (p<0.001)** — at *fewer* reader tokens, so it is not a "more context" effect.
- **The "it's just MMR" objection is empirically dead:** plain MMR (redundancy-aware reranking) is *significantly worse than the focused heuristic* (−2.0 F1, p=0.007). The ordering is **submod > focused > packed > mmr**. Only the full submodular objective wins; mere redundancy reduction hurts.
- A new diagnostic, **answer-in-context** (is a gold answer a contiguous token run in the reader context?), predicts answer quality far better than retrieval recall: **corr(ans_in_context, F1)=0.50** vs recall@5=0.31; conditional F1 is 0.60 when the answer is in context vs 0.12 when not. This is the long-sought explanation of the "ACE wins with lower recall" paradox: under a budget, what matters is whether the answer survives into the reader context, not how many gold documents were retrieved.
- **Mechanism** (exact per-question decomposition): 81% of the gain comes from 37 questions where the submodular packer newly placed the answer into context (+0.39 F1 each), achieved through better complementary multi-hop gold-document coverage (all gold docs in context on 289 vs 256 questions), not through raw token density.
- **Honest twist (significant, 3 seeds):** the packer helps **chunk**, not **ACE** — ace_submod is *worse* than ace_focused (−2.1 F1, p=0.021), and under the submod packer chunk beats ACE (+4.5 F1, p<0.001). ACE already compresses/de-duplicates evidence at the graph level, so submodular redundancy-reduction has little to exploit there. The lever is the packing *objective*, and it matters most for redundant chunk evidence; graph compression and principled packing are partial substitutes.

Consequence for the paper: the strongest available framing shifts from "graph-compressed evidence beats chunk under a budget" (fragile across datasets) to **"budget-constrained multi-hop RAG is bottlenecked by answer-in-context, and a principled submodular packer that maximizes complementary gold-evidence coverage delivers a clean, significant, mechanistically-explained gain."** ACE becomes one representation in a controlled factorial. The large oracle gap (F1 0.570 vs best fixed 0.448) shows real per-question routing headroom over the packer grid. See `RESULTS.md` (Stage-4 section) and `RESEARCH_LOG.md` (2026-06-28 entries) for the full tables, bootstrap, and decomposition.

### Scope (Stage-5, honest boundary)

The packer win is **not universal**, and Stages 5-7 mapped exactly where it holds. Principled submodular packing helps only when three conditions co-occur:

1. **Multi-hop complementary structure.** RAGBench CovidQA (n=246) / ExpertQA (n=203) are single-pass with all-context-gold; `chunk_submod` does *not* beat `chunk_focused` (CovidQA −0.010 F1 p=0.30; ExpertQA +0.005 p=0.15) and ACE regains its edge. No complementarity → no gain.
2. **Retrieval that surfaces the evidence.** MuSiQue is genuinely multi-hop but retrieval-bottlenecked (all_gold@5=0.18, ans_in_context≈0.20); `chunk_submod` − `chunk_focused` is +0.011 F1 (p=0.34, n.s.) and naive packing is just as good. The packer cannot assemble what retrieval never surfaced.
3. **Budget pressure that is binding but not extreme.** The HotpotQA budget curve is an inverted-U: the submod−focused gap is significant only at ~160, ≈0 at 96 (no room for complementarity) and 224 (everything fits). My "advantage grows monotonically as budget tightens" prediction was falsified.

HotpotQA is where all three hold, and there the win is large, significant, and 3-seed robust; `chunk_submod` also beats naive packing at *every* budget and reaches the heuristic's best quality at ~30% lower token cost. Crucially, the **answer-in-context diagnostic generalizes even where the packer does not** — corr(ans_in_context, F1) = 0.50 (HotpotQA), 0.54 (MuSiQue), 0.39 (CovidQA), all well above retrieval recall (~0.31). So the diagnostic is a genuine dataset-independent mediator, not an artifact of the packer. This separates the contribution into a **general** part (the diagnostic + the "when does packing help" analysis) and a **conditional** part (the packer's significant win where the conditions hold).

## Main Positive Finding

On HotpotQA, ACE graph retrieval combined with packed evidence is competitive under a comparable reader-context budget and is stronger than the tested lexical and hybrid packed controls. Its advantage over packed chunk retrieval is not stable across the current robustness checks.

The best current method is:

```text
ACE graph retrieval + truncate320 embeddings + packed snippets
```

This method:

- beats packed lexical and hybrid retrieval controls with Qwen;
- is competitive with packed chunk retrieval, but does not consistently beat it;
- uses much less reader context than full chunk-source RAG;
- does not beat full chunk-source RAG on absolute answer quality.

The clearest defensible result is now budgeted competitiveness rather than stable dominance over chunk packing.

## Retrieval Finding

Before generation, ACE graph retrieval already showed a strong efficiency pattern.

Compared with chunk retrieval, ACE usually retrieved slightly less gold evidence but used far fewer evidence tokens. This means the graph retrieval layer is doing useful evidence allocation, but it is not perfect.

The retrieval-only result supports the first part of the thesis:

> Structured evidence retrieval can reduce retrieval budget while preserving much of the useful evidence.

## Generation Finding

The first Qwen generation experiments showed that retrieval quality alone is not enough.

Three reader-context styles were tested:

1. **Raw graph hits**
   - very compact;
   - too fragmentary for Qwen;
   - weak answer quality.

2. **Full source passages**
   - better answer quality;
   - too much context;
   - loses the efficiency benefit.

3. **Packed snippets**
   - best tradeoff;
   - enough context for Qwen;
   - much more efficient than full source passages.

This changed the thesis direction. The important contribution is not only graph retrieval. It is:

> graph retrieval plus budget-aware evidence packing.

## Compression Finding

Simple PCA compression was not strong enough.

Truncation was more useful. Moderate truncation preserved the retrieval behavior well, while aggressive truncation hurt. This suggests that compressed semantic representations are viable, but naive compression has limits.

The current best compression setting is a moderate truncation setup rather than PCA.

## MuSiQue Finding

MuSiQue was used as a harder stress test.

The original ACE method did not transfer cleanly. Both chunk RAG and ACE-RAG struggled, and simple retrieval expansion did not fix the problem. The later bridge-aware ACE variant improved the result: with packed evidence, it gave the best fixed-policy answer quality among the tested chunk, lexical, hybrid, and ACE policies while keeping context compact. The corrected Stage-3 router run preserved the fixed ACE advantage, but the hand-written router did not transfer well to MuSiQue.

This means MuSiQue should currently be framed as a stress-test result:

> Bridge-aware ACE can improve budgeted generation on harder composed multi-hop QA, but the retrieval problem is not fully solved and the absolute answer quality remains low.

This is not a failure of the thesis. It is useful evidence about where the current method helps and where it still needs stronger reasoning-aware retrieval.

## Current Claims

### Claim 1

ACE graph retrieval can reduce evidence budget while preserving much of the evidence coverage on HotpotQA.

### Claim 2

Packed evidence is necessary for generation. Raw graph evidence is too small, while full source passages are too large.

### Claim 3

On HotpotQA, ACE graph retrieval with packed snippets can outperform packed chunk, lexical, and hybrid RAG controls with a free local Qwen reader. The focused ACE variant gives a lower-context alternative. This corrected result holds across three sampled seeds with the smaller Qwen reader and across three larger Qwen three-billion runs.

The stronger-reader budget curve shows that the advantage grows when the evidence budget is tighter. Both compact-budget settings are positive across the completed three-seed aggregate, and the tightest condition is the cleanest result. This is directly aligned with the thesis: compressed graph evidence is most valuable under context pressure.

The extreme-budget stress check is now replicated. The relative ACE advantage persists below the compact range, but absolute answer quality drops enough that those settings are better treated as boundary analysis than as a recommended operating point.

Paired bootstrap analysis supports the same interpretation: compact-budget comparisons separate more clearly from chunk retrieval than the standard-budget comparison. This strengthens the paper's framing around budgeted evidence compression.

### Claim 4

On MuSiQue, bridge-aware ACE with packed evidence can outperform chunk, lexical, and hybrid controls on answer quality under a compact context budget, but it does not yet solve the harder retrieval problem. The current hand-written router should not be claimed as a cross-dataset solution.

### Claim 5

On RAGBench, the evidence is now mixed but improving. PubMedQA exposed an answer-format mismatch, CovidQA and ExpertQA give cleaner positive results for ACE under a tight context budget, EManual gives a weaker directional result, and TechQA is mixed. This supports the direction but prevents an over-broad transfer claim.

### Claim 6

Adaptive evidence routing has strong headroom. The oracle router shows that different questions benefit from different compact evidence policies. The conservative rule router is competitive on the larger HotpotQA run, but the first learned-router transfer check and the corrected MuSiQue router run did not reliably beat the best fixed ACE policy. Routing should therefore be framed as promising evidence and analysis, while fixed packed ACE remains the central deployed method.

## Current Best Method

The strongest current pipeline is:

```text
documents
-> sentence-level claim/entity graph
-> dense retrieval over graph nodes
-> moderate embedding truncation
-> source-linked packed snippets
-> Qwen reader
```

## Current Limitations

- Claim extraction is still simple sentence splitting.
- Entity extraction is heuristic.
- Contradiction detection is heuristic and not central yet.
- Qwen evaluation uses a small open model.
- Stronger-reader validation is now positive on the larger HotpotQA setting, but it still needs stronger cross-dataset support.
- MuSiQue exposes a retrieval weakness.
- The current router features are not strong enough to close the oracle gap.
- Faithfulness and citation metrics are not yet fully developed.
- The current evidence packer is useful but still simple.
- The budget-curve result is now replicated, but it is still limited to HotpotQA and should be paired with clear cross-dataset limitations.
- Extremely small context budgets are useful for stress analysis, but they are not currently a practical primary setting because answer quality drops sharply.
- The standard-budget advantage is weaker than the compact-budget advantage, so the paper should not oversell ACE as a universally better retriever at every budget.
- RAGBench PubMedQA currently needs a label-aware evaluation path because free-form explanatory gold answers distort exact-match and token-F1.
- RAGBench CovidQA and ExpertQA give positive transfer evidence, EManual is only directional because duplicate question IDs weaken the paired analysis, and TechQA is mixed. The cross-dataset story is useful but should be framed carefully.

## What Comes Next

The next technical step is continued cross-dataset validation and deeper error analysis. HotpotQA now has multi-seed positive results with both the smaller Qwen reader and Qwen three-billion at the larger setting. MuSiQue has a corrected fixed-ACE positive result with the smaller reader, but still needs a stronger-reader rerun if the dataset is available in the active Kaggle environment. RAGBench should continue with another subset, while PubMedQA should eventually be rerun with explicit label preservation.

Recommended writing artifacts remain:

1. Method draft.
2. Final experiment table draft.
3. Related work outline.
4. Limitations section.
5. Future work section on reasoning-aware graph retrieval.
