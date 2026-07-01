# ACE-RAG Research Log

Project:

**ACE-RAG: Adaptive Compressed Evidence Graphs for Faithful and Efficient Retrieval-Augmented Generation**

This log records important implementation decisions, experiment outputs, failures, and current interpretations. Dates are in Bangladesh time unless noted otherwise.

## 2026-06-24: Project Scaffold

Created the first ACE-RAG experimental scaffold under:

```text
ace_rag_research/
```

Implemented:

- dataset loaders for `toy`, `hotpotqa`, `musique_local`, and `ragbench`;
- evidence graph construction with passage, claim, and entity nodes;
- edges: `derived_from`, `mentions`, and lightweight `contradicts`;
- chunk retrieval baseline;
- ACE graph retriever;
- compressors: `identity`, `truncate`, `pca`, and `binary`;
- offline extractive answer proxy;
- lightweight verifier;
- metrics for retrieval, answer proxy quality, verifier behavior, and evidence-token budget.

Initial toy smoke test:

| Method | Recall@5 | All-Gold@5 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: |
| Chunk RAG | 1.0000 | 1.0000 | 85.00 |
| ACE Graph | 1.0000 | 1.0000 | 53.25 |

Interpretation:

The local toy run confirmed the pipeline works and showed the expected first signal: ACE graph retrieval can preserve gold-document coverage while reducing retrieved evidence tokens.

## 2026-06-24: Kaggle/Colab Cloud Setup

Created cloud support:

- `requirements-cloud.txt`
- `scripts/cloud_check.py`
- `CLOUD_RUN.md`
- `notebooks/ACE_RAG_Cloud_Quickstart.ipynb`

Initial recommendation:

- Use Kaggle first.
- Prefer T4 x2 over P100 if P100 throws CUDA kernel errors.

Observed error:

```text
CUDA error: no kernel image is available for execution on the device
cudaErrorNoKernelImageForDevice
```

Interpretation:

This was treated as a PyTorch/CUDA/GPU architecture mismatch, most likely from using an incompatible CUDA wheel on Kaggle P100/Pascal. The practical recommendation became: use **Kaggle T4 x2**.

## 2026-06-24: First HotpotQA Limit 200 Run

Configuration:

```text
dataset: hotpotqa
limit: 200
embedding: BAAI/bge-small-en-v1.5
compressor: pca:128
top_k_nodes: 12
max_expanded_docs: 5
```

Output:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| Chunk RAG | 0.8525 | 0.7100 | 0.0475 | 402.20 |
| ACE Graph, PCA128 | 0.7950 | 0.6550 | 0.0715 | 135.71 |

Interpretation:

ACE achieved much lower token cost but lost too much retrieval quality under PCA128. This suggested:

- graph evidence allocation is promising;
- PCA compression is too lossy at 128 dimensions;
- identity/no-compression ACE should be tested next.

## 2026-06-24: HotpotQA Limit 200 Identity and PCA256

Identity result:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| Chunk RAG | 0.8525 | 0.7100 | 0.0475 | 402.20 |
| ACE Graph, Identity | 0.8375 | 0.7000 | 0.0721 | 135.97 |

PCA256 result:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| ACE Graph, PCA256 | 0.8225 | 0.6750 | 0.0689 | 240.26 |

Interpretation:

The strongest early signal came from ACE identity:

- only -1.5 recall@5 points versus chunk RAG;
- only -1.0 all_gold@5 point;
- about 66% fewer evidence tokens;
- higher extractive F1 proxy.

PCA256 improved over PCA128 but still degraded retrieval and used more evidence tokens.

## 2026-06-24/25: HotpotQA Limit 1000

Chunk baseline:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| Chunk RAG | 0.8560 | 0.7230 | 0.0499 | 408.94 |

ACE identity, `top_k_nodes=48`, `max_expanded_docs=5`:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| ACE Graph, Identity | 0.8310 | 0.6840 | 0.0845 | 136.63 |

Interpretation:

This is the current strongest Stage-1 result:

- ACE uses about 66.6% fewer evidence tokens than chunk RAG;
- recall@5 drops by 2.5 points;
- all_gold@5 drops by 3.9 points;
- extractive F1 proxy improves from 0.0499 to 0.0845.

Research claim supported so far:

> ACE graph retrieval preserves most HotpotQA evidence coverage while using about one-third of the evidence tokens.

## 2026-06-25: Runtime Optimization

Observed Kaggle behavior:

```text
GPU usage: near 0%
CPU usage: high
```

Interpretation:

The run was not purely GPU-bound. The heavy stages were:

- evidence graph construction;
- Python graph traversal;
- ranking over many graph nodes;
- repeated methods in the same command;
- saving large run JSONs.

Implemented optimizations:

- explicit `--device cuda`;
- progress markers:
  - `[ace] building evidence graph`
  - `[ace] fitting/indexing embeddings`
  - `[embed] encoding ... on device=cuda`
  - `[ace] retrieving`
- `--methods ace_graph` to avoid rerunning chunk/no-conflict variants;
- `--no-save-runs` to skip large JSON output during tuning;
- batched query embedding;
- top-k selection with `argpartition` instead of full sorting.

Created:

- `notebooks/ACE_RAG_Kaggle_Runbook.ipynb`
- `ace_rag_research_kaggle_ready_v2.zip`

Important packaging issue:

The first Kaggle zip used Windows-style backslashes in archive names and Kaggle rejected it:

```text
contains a forbidden character in name ('\')
```

Fixed by creating `ace_rag_research_kaggle_ready_v2.zip` with POSIX `/` archive paths.

## 2026-06-25: Budget Tuning, Docs 6

Configuration:

```text
limit: 1000
compressor: identity
top_k_nodes: 48
max_expanded_docs: 6
methods: ace_graph
device: cuda
no_save_runs: true
```

Output:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| ACE Graph, Identity, docs=6 | 0.8310 | 0.6840 | 0.0861 | 166.25 |

Comparison to docs=5:

| Config | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| docs=5 | 0.8310 | 0.6840 | 0.0845 | 136.63 |
| docs=6 | 0.8310 | 0.6840 | 0.0861 | 166.25 |

Interpretation:

Increasing `max_expanded_docs` from 5 to 6 did not improve retrieval coverage. It increased tokens by about 30 and gave only a tiny extractive F1 gain. Current best budget remains `max_expanded_docs=5`.

## 2026-06-25: Candidate Tuning, Nodes 64

Configuration:

```text
limit: 1000
compressor: identity
top_k_nodes: 64
max_expanded_docs: 5
methods: ace_graph
device: cuda
no_save_runs: true
```

Output:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens | Avg Hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACE Graph, Identity, nodes=64 | 0.8310 | 0.6840 | 0.0845 | 136.68 | 9.25 |

Comparison to nodes=48:

| Config | Recall@5 | All-Gold@5 | Avg Evidence Tokens | Avg Hits |
| --- | ---: | ---: | ---: | ---: |
| nodes=48 | 0.8310 | 0.6840 | 136.63 | 9.25 |
| nodes=64 | 0.8310 | 0.6840 | 136.68 | 9.25 |

Interpretation:

Increasing `top_k_nodes` from 48 to 64 did not improve retrieval. The run appears dominated by early document expansion and `max_expanded_docs=5`, not by candidate-node scarcity.

## 2026-06-25: Truncation Compression, 384 Dimensions

Configuration:

```text
limit: 1000
compressor: truncate
compress_dims: 384
top_k_nodes: 48
max_expanded_docs: 5
methods: ace_graph
device: cuda
no_save_runs: true
```

Output:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| ACE Graph, Truncate384 | 0.8310 | 0.6840 | 0.0845 | 136.68 |

Interpretation:

`truncate:384` matches identity. Since `BAAI/bge-small-en-v1.5` has 384-dimensional embeddings, this is effectively a no-op check. It confirms that the truncation path is functioning and does not alter results at full dimension.

Next important compression tests:

```text
truncate:256
truncate:128
```

Target:

- `truncate:256` should preserve most identity performance:
  - recall@5 >= 0.825
  - all_gold@5 >= 0.675
- `truncate:128` can drop more, but should ideally stay near recall@5 >= 0.80.

## 2026-06-25: Truncation Compression, 256 Dimensions

Configuration:

```text
limit: 1000
compressor: truncate
compress_dims: 256
top_k_nodes: 48
max_expanded_docs: 5
methods: ace_graph
device: cuda
no_save_runs: true
```

Output:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens | Avg Hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACE Graph, Truncate256 | 0.8165 | 0.6610 | 0.0826 | 136.29 | 9.08 |

Comparison:

| Config | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| Identity / full 384 dims | 0.8310 | 0.6840 | 0.0845 | 136.63 |
| Truncate256 | 0.8165 | 0.6610 | 0.0826 | 136.29 |

Interpretation:

`truncate:256` gives 33% smaller dense vectors but loses:

- 1.45 recall@5 points versus identity;
- 2.3 all_gold@5 points versus identity;
- slight extractive F1.

This is not catastrophic, but it is weaker than the target threshold. For the current HotpotQA setup, naive 256-dimensional truncation loses more evidence coverage than desired.

Next useful compression test:

```text
truncate:320
```

This tests whether a smaller 16.7% compression retains identity-level retrieval better than 256.

## Current Best Result

Best Stage-1 config:

```text
dataset: HotpotQA
limit: 1000
model: BAAI/bge-small-en-v1.5
method: ACE Graph
compressor: identity
top_k_nodes: 48
max_expanded_docs: 5
```

Best Stage-1 comparison:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| Chunk RAG | 0.8560 | 0.7230 | 0.0499 | 408.94 |
| ACE Graph | 0.8310 | 0.6840 | 0.0845 | 136.63 |

Current claim:

> On a 1000-example HotpotQA slice, ACE graph retrieval keeps most of chunk RAG's evidence coverage while reducing evidence-token budget by about two-thirds and improving a simple extractive answer proxy.

Current caveats:

- The answer generator is still an extractive proxy, not a real LLM generator.
- The verifier is heuristic, not an NLI/LLM verifier.
- PCA compression underperforms.
- Conflict expansion does not matter on HotpotQA because it is not a conflict-heavy dataset.
- We still need MuSiQue, RAGBench, and a conflict benchmark before this becomes paper-grade evidence.

## 2026-06-25: Qwen Reader Context Experiments

The first local Qwen reader experiments showed that raw graph hits were too fragmentary for answer generation. Feeding full source passages improved answer quality but removed much of ACE-RAG's efficiency advantage. A snippet-based reader context recovered most of the source-passage answer quality while reducing the reader context, but the token savings were still not strong enough for a clean final claim.

Interpretation:

ACE-RAG's retrieval stage is promising, but the generation stage needs a better evidence materialization policy. The next direction is to tune snippet selection and budget rather than simply switching to a larger reader model.

## 2026-06-25: Autonomous Stage-2 Pipeline

Built a Kaggle-oriented autonomous Stage-2 runner. It retrieves chunk and ACE evidence once, loads the local Qwen reader once, and evaluates multiple source and packed-snippet reader policies in one job. This should replace manual snippet sweeps and give a cleaner comparison of evidence packing strategies.

## 2026-06-25: Packed Evidence Becomes Promising

The autonomous Stage-2 run showed that budget-aware packed snippets are meaningfully better than naive snippets. The best ACE packed policy came close to the full chunk-source reader while using much less reader context. It also outperformed the packed chunk control at a similar budget.

Interpretation:

This is the strongest generation-stage signal so far. The research direction should now shift from ad hoc snippet tuning to scaling and validating packed evidence. The next question is whether the packed-evidence result holds on a larger HotpotQA slice and then on a harder multi-hop dataset.

## 2026-06-25: Packed Evidence Scaling Check

The larger HotpotQA check showed that packed snippets remain useful for reducing reader context, but the advantage of ACE packed snippets over the packed chunk control did not hold. This weakens the generation-stage claim in its current form.

Interpretation:

The graph retrieval layer is still promising for evidence selection, but the packed reader evidence needs a stronger selection policy. The next method step should be a better evidence packer or reranker, not simply scaling the same packed-snippet policy.

## 2026-06-25: Focused Evidence Packing

Implemented a bounded improvement to evidence packing. The new focused packer keeps the same retrieval pipeline but prefers snippets that add new question-term coverage and cover distinct retrieved documents under a fixed budget. This is a single method improvement for comparison against the previous packed-snippet policy, not a broad search over new mechanisms.

The focused packer improved the ACE generation result on the larger HotpotQA slice. The lower focused budget was better than the larger one, suggesting that better evidence selection matters more than simply adding more context. This is now the strongest ACE generation-stage method, though the packed chunk control remains a serious baseline.

## 2026-06-25: Stage-2 Limit-1000 Validation

The larger HotpotQA run produced the strongest generation-stage result so far. ACE with packed evidence outperformed the chunk-source reader and the packed chunk control while using much less context than full chunk sources. This gives the project its first credible end-to-end RAG signal, not just a retrieval-efficiency signal.

Interpretation:

The best current story is no longer focused packing. The strongest policy is ACE graph retrieval followed by packed snippet evidence at the moderate budget. This should now be validated on a different HotpotQA slice before moving to a new dataset.

## 2026-06-25: HotpotQA Seed Validation

The reported seed-validation run preserved the same overall pattern as the main HotpotQA Stage-2 run. ACE graph retrieval with packed evidence remained the strongest end-to-end policy among the tested settings, while using less reader context than full chunk sources.

Interpretation:

This strengthens the HotpotQA result. The next validation should move to MuSiQue rather than continuing to tune HotpotQA.

## 2026-06-25: MuSiQue Stress Test

MuSiQue is substantially harder for both chunk and ACE retrieval. The first MuSiQue Stage-2 run showed that ACE retrieval still reduces context, but evidence coverage is weaker and the HotpotQA advantage does not transfer cleanly. The focused packed variant was the best ACE generation policy on the first MuSiQue slice, but the result should be treated as a stress-test signal rather than a confirmed win.

Interpretation:

The project now has a clear split: HotpotQA supports the ACE packed-evidence story, while MuSiQue exposes the need for stronger multi-hop retrieval expansion. The next MuSiQue step should be a bounded retrieval-expansion test rather than more reader-context tuning.

## 2026-06-25: MuSiQue Expansion Test

The bounded MuSiQue expansion test did not help. Increasing ACE's graph-node and expanded-document budget increased reader context but did not improve evidence coverage or answer quality. This suggests the current MuSiQue bottleneck is not simply insufficient expansion depth.

Interpretation:

Stop MuSiQue tuning for this method version. Treat MuSiQue as evidence that harder composed multi-hop QA requires better graph construction or reasoning-aware retrieval, not just more retrieved context.

## 2026-06-25: Bridge-Aware ACE Retrieval

Implemented a bounded bridge-aware ACE retriever for MuSiQue-style composed questions. It performs first-hop graph retrieval, extracts bridge terms from the first-hop evidence, and uses those terms for a second-hop graph reranking pass. This tests one clear hypothesis: MuSiQue needs better bridge evidence discovery, not just more retrieval expansion.

The MuSiQue bridge run produced a modest positive result. Bridge-aware ACE improved evidence coverage compared with standard ACE and gave the strongest MuSiQue Qwen result so far, though the absolute answer quality remains low. This supports bridge-aware retrieval as a useful direction, but MuSiQue is still a hard stress test rather than a clean win.

## 2026-06-25: MuSiQue Bridge Validation

The larger MuSiQue bridge validation preserved the same weak-positive pattern. Bridge-aware ACE with packed evidence outperformed the chunk baselines on the Qwen answer metric while using less context than full chunk sources, even though its retrieval coverage remained lower than chunk retrieval.

Interpretation:

This is enough MuSiQue evidence for the current phase. The right framing is that bridge-aware ACE improves the method's behavior on a harder dataset, but MuSiQue remains a stress test with low absolute answer quality. Stop experiments here and consolidate the thesis.

## Baseline Expansion

Added standard lexical retrieval and dense plus lexical hybrid retrieval baselines. Future answer-generation tables can now compare ACE against stronger retrieval controls in the same run, which is necessary before making any publication-level claim.

## Baseline Validation Finding

The stronger baseline run changed the HotpotQA story. ACE with packed evidence beats the packed chunk and packed hybrid controls under a similar context budget, but it does not beat full source chunk reading. The thesis claim should focus on budget-constrained answer quality and evidence efficiency, not absolute best accuracy.

## MuSiQue Baseline Validation Finding

The stronger MuSiQue baseline run is a meaningful positive result. Bridge-aware ACE with packed evidence produced the best answer-quality result among the tested chunk, lexical, hybrid, and ACE policies while keeping the reader context compact. The retrieval problem is still not fully solved, so this should be framed as better budgeted generation on a hard stress test rather than a complete MuSiQue solution.

## HotpotQA Robustness Check

The second HotpotQA seed weakened the earlier clean win. ACE with packed evidence remained competitive and beat the lexical and hybrid packed controls, but it did not beat packed chunk retrieval on this slice. The HotpotQA claim should now be framed as mixed but promising budgeted evidence compression, not a stable superiority claim over packed chunk RAG.

## Additional HotpotQA Robustness Check

Another HotpotQA seed showed the same mixed pattern. ACE remained stronger than the lexical and hybrid packed controls, and the focused ACE setting used much less context than packed chunk, but fixed ACE still did not consistently beat packed chunk retrieval. This supports moving to an adaptive router instead of continuing fixed-policy tuning.

## Adaptive Router Implementation

Implemented a Stage-3 adaptive evidence router. It compares compact fixed policies, a simple retrieval-feature router, and an oracle router upper bound in one run. The purpose is to test whether choosing evidence strategy per question can overcome the weakness of fixed ACE policies.

## Adaptive Router Validation Finding

The first adaptive-router validation produced the right diagnostic but not yet a deployable router result. The oracle router showed that different questions really do prefer different compact evidence policies, so routing is worth pursuing. The hand-written rule router failed because it selected focused ACE too often, which confirmed that fixed heuristics are too brittle for the final thesis claim.

Follow-up implementation:

The Qwen reader now uses decoder-safe left padding. The rule router is now conservative and only chooses ACE when its retrieval signal is clearly stronger than chunk retrieval. Stage-3 also writes a per-question router feature file so the next routing step can be learned or calibrated from observed policy behavior instead of guessed manually.

## Corrected Stage-3 Reader Finding

The corrected Qwen reader changed the HotpotQA interpretation. Once decoder padding was fixed, focused ACE became the strongest compact fixed policy, and the conservative router became competitive with it. This makes the adaptive compressed-evidence direction stronger, but it still needs robustness checks across seeds and datasets before it can support a serious publication claim.

## Corrected HotpotQA Robustness Finding

The corrected reader result held up on another HotpotQA seed. ACE with compact packed evidence remained stronger than the compact baselines on answer quality, while the focused variant preserved the lower-context advantage. The rule router remained competitive but did not consistently beat the best fixed ACE policy. The next method work should therefore focus on learned or calibrated routing, not more manual routing rules.

## Offline Router Transfer Finding

The first learned-router transfer check did not produce a deployable router improvement. With the current retrieval-confidence features, the learned router mostly collapses toward focused ACE and does not reliably beat the best fixed compact ACE policy. This is useful evidence: the oracle gap is real, but closing it needs better deploy-time signals such as question decomposition, answerability confidence, policy agreement, or stronger graph-specific uncertainty features.

## Three-Seed Corrected HotpotQA Finding

The corrected HotpotQA result now holds across multiple sampled seeds. The rule router has the best average answer quality among deployable policies, while focused ACE remains the strongest compact fixed-policy option. This materially strengthens the thesis: the method is no longer just a single-run positive result. The remaining gap is cross-dataset validation and a stronger explanation of why compact graph evidence improves answer quality despite lower raw retrieval coverage than chunk retrieval.

## Corrected MuSiQue Stage-3 Finding

The corrected MuSiQue router run finished cleanly. Bridge-aware ACE remained the strongest fixed compact policy on answer quality, but the hand-written router was not strong enough on this harder dataset because it routed too often to chunk-style evidence. A small offline threshold sweep did not give a robust rule change across HotpotQA and MuSiQue, so the right next step is not more manual router tuning. The next bounded validation is to keep retrieval and routing fixed and test whether the MuSiQue finding holds with a stronger free Qwen reader.

## Qwen Three-Billion HotpotQA Validation

The first stronger-reader Kaggle run completed through the GitHub result loop. On HotpotQA, ACE packed became the best deployable policy, while focused ACE stayed very close with a substantially smaller evidence budget. This strengthens the central thesis claim because the ACE advantage is not only an artifact of the smaller Qwen reader. The rule router still underperformed the best fixed ACE option, so the current paper should emphasize compact graph evidence first and adaptive routing as a promising but unfinished extension.

## Qwen Three-Billion HotpotQA Robustness

The stronger-reader result replicated across additional HotpotQA samples. Across the completed Qwen three-billion runs, fixed ACE with packed evidence has the best average deployable answer quality, and focused ACE is nearly tied while using much less context. The router still does not consistently beat fixed ACE, but the oracle remains far higher, so routing should be presented as future method headroom rather than the main finished contribution.

## Qwen Three-Billion HotpotQA Scaling

The first larger Qwen three-billion HotpotQA run completed successfully. At the larger sample size, fixed ACE packed evidence remained stronger than packed chunk retrieval on answer quality, while the focused lower-budget ACE setting became more of an efficiency tradeoff than a near-tie. This is an important refinement: the headline method should be ACE packed evidence for quality, with focused ACE as the compact variant.

## Qwen Three-Billion HotpotQA Scaling Robustness

The second larger HotpotQA run replicated the fixed ACE packed result. Across the larger Qwen three-billion samples completed so far, ACE packed has the best deployable answer quality, while focused ACE remains the compact variant. This is now strong enough to treat larger-reader HotpotQA as a core positive result, pending the final larger-sample seed.

## Next Runs

Run next:

```bash
python -m experiments.run_mvp \
  --config configs/hotpotqa.yaml \
  --limit 1000 \
  --compressor truncate \
  --compress-dims 320 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --methods ace_graph \
  --device cuda \
  --no-save-runs \
  --out-dir cloud_results
```

Then:

```bash
python -m experiments.run_mvp \
  --config configs/hotpotqa.yaml \
  --limit 1000 \
  --compressor truncate \
  --compress-dims 128 \
  --top-k-nodes 48 \
  --max-expanded-docs 5 \
  --methods ace_graph \
  --device cuda \
  --no-save-runs \
  --out-dir cloud_results
```

## Kaggle Automation And Analysis Hygiene

Added a safer Kaggle GitHub publishing path and a de-duplicated Stage-3 summarizer. This matters because Kaggle result folders are cumulative, so repeated files can otherwise make aggregate results look stronger or weaker than they really are. The thesis evidence should be based on logical runs, not repeated copies from later result publishes.

## Larger HotpotQA Validation Completed

The stronger Qwen reader has now completed the larger HotpotQA validation across the planned seeds. The result supports the main fixed packed ACE claim and makes the HotpotQA side of the thesis substantially more stable. Routing remains useful as evidence of headroom, but fixed packed ACE is still the cleanest headline method.

## Question-Level Error Analysis Added

Added a script for comparing Stage-3 policies question by question. This gives the paper a way to discuss where ACE helps and where chunk retrieval still wins, instead of relying only on aggregate scores.

## Budget Curve Pilot

The tighter-budget HotpotQA result is now replicated across the planned seeds. Both compact budgets support the context-pressure hypothesis, with the tightest budget showing the cleanest advantage. This is one of the strongest paper-facing results so far.

## Extreme Budget Boundary Check

The very-small-budget result is now replicated across the planned seeds. ACE still has a relative advantage when context is aggressively restricted, but absolute answer quality falls sharply. This should be treated as boundary analysis rather than a recommended deployment setting.

## Paired Bootstrap Check

Added paired bootstrap analysis over saved predictions. The check supports the budget-pressure framing: compact-budget comparisons separate more clearly from chunk retrieval than the standard-budget comparison.

## Cross-Dataset Validation Step

Moved the pipeline from HotpotQA budget sweeps to cross-dataset validation. The next Kaggle control job prefers MuSiQue when the dataset is mounted, and otherwise tries a small RAGBench validation run. This is meant to test whether the budgeted ACE finding transfers beyond HotpotQA.

## Budget Summary Grouping Fix

Updated the summarizer so budget-curve runs are grouped separately from the standard runs. This avoids mixing policies that happen to share the same name but come from different evidence-budget experiments.

## PubMedQA Cross-Dataset Check

The first RAGBench PubMedQA run exposed an evaluation mismatch. The saved gold answers are long medical explanations, while the reader usually gives short yes, no, or unknown-style answers. A conservative label-style analysis gives ACE a small positive signal over chunk packing, but this should be treated as tentative until the pipeline preserves the original PubMedQA labels or another RAGBench subset confirms the pattern.

## CovidQA Cross-Dataset Check

The RAGBench CovidQA run gives the first cleaner cross-dataset support for the compact ACE claim. Chunk retrieval found more gold documents, but ACE produced better reader answers under the same tight context budget. This strengthens the thesis because it shows the HotpotQA budget-pressure result is not isolated to one dataset.

## ExpertQA Cross-Dataset Check

The RAGBench ExpertQA run gives a smaller but still positive cross-dataset signal. The task is open-ended and the absolute answer scores are low, but ACE again produced better reader answers than packed chunk under the same tight context budget. This adds useful transfer evidence beyond HotpotQA and CovidQA.

## EManual Cross-Dataset Check

The RAGBench EManual run is directionally positive but weaker than CovidQA and ExpertQA. ACE packed has a better aggregate answer score than packed chunk, but the dataset slice contains repeated question identifiers, so the paired analysis is less decisive. This should be used as supporting trend evidence, not as a headline result.

## TechQA Cross-Dataset Check

The RAGBench TechQA run is mixed. Focused ACE is slightly better than packed chunk in aggregate, but packed ACE is slightly worse and the paired intervals are not clean. This is useful negative evidence: the method is not universally better across every domain, so the paper should keep the claim focused on budgeted evidence compression with clear transfer limits.

## 2026-06-28: Stage-4 — Principled Submodular Packer and Evidence-Density Diagnostics

**Motivation:** The existing heuristic packers (packed, focused) select snippets by a hand-tuned lexical score without any formal objective or approximation guarantee. They also cannot explain *why* lower document recall at the retrieval stage still produces better reader answers under a budget. Stage-4 addresses both gaps.

**New methodology (two linked contributions):**

1. **Budgeted-submodular evidence packer** (`ace_rag/evidence_packer.py`). Frames reader-context construction as budgeted monotone submodular maximization. The objective combines four normalized terms: modular relevance (dominant), query-term set coverage, saturated-facility-location representativeness, and concave-over-groups source diversity. Selection uses cost-scaled (per-token) greedy maximization with the Lin & Bilmes (2011) single-element singleton fallback, which gives a constant-factor approximation guarantee under a hard token budget. Critically, the candidate generation is identical to the focused heuristic packer, so the only thing that differs is the selection objective and algorithm — making the comparison a clean controlled experiment.

2. **Evidence-density diagnostics** (`ace_rag/context_quality.py`). New per-question metrics measuring what actually matters under a fixed context budget: `answer_in_context` (is any gold answer a contiguous token run in the reader context?), `gold_token_density` (fraction of reader tokens from gold documents), `gold_doc_reader_coverage` (reader-level gold coverage after packing, as opposed to retrieval recall). These quantify the proposed mediation: ACE wins with lower recall because its packed context has higher answer-bearing density.

3. **2×2 factorial runner** (`experiments/run_density_router.py`). Evaluates {chunk, ACE} × {focused, submodular} at a fixed shared budget plus an oracle upper bound over all four. Emits both standard answer metrics and the new density columns, and writes a per-question CSV for mediation analysis (does density predict F1 better than recall@5?).

**Code delivery mechanism:** New files are shipped via `control/overlay/` in the GitHub repo. The updated `control/main.py` applies the overlay onto the frozen Kaggle project zip at runtime before the job runs, so no zip re-upload is required.

**Local de-risking result (HotpotQA-60, CPU, extractive reader):**

| Policy | ans_in_ctx | gold_tok_density | F1 | distractors |
|---|---|---|---|---|
| chunk_packed_160 | 0.600 | 0.439 | 0.061 | 1.87 |
| chunk_focused_160 | 0.683 | 0.547 | 0.063 | 1.47 |
| chunk_submod_160 | 0.650 | 0.463 | **0.079** | 2.13 |
| ace_focused_160 | 0.650 | 0.494 | 0.070 | 1.92 |
| ace_submod_160 | 0.617 | 0.437 | 0.071 | 2.20 |
| oracle (mixed) | 0.750 | 0.477 | **0.089** | 1.92 |

The density gate (submod ≥ focused on ans_in_context and gold_token_density) did not hold for ACE on this small CPU slice. However, `chunk_submod_160` produced the best F1 of all fixed policies (+26% relative over chunk_focused), and `ace_submod_160` F1 essentially ties `ace_focused_160`. The oracle gap (0.089 vs 0.071 best fixed) confirms that per-question policy mixing is valuable. The interpretation is that the submodular diversity objective selects more *complementary* evidence, which can improve generative reading even when it brings in more distractor snippets. The density-mediation story needs refinement: complementary coverage, not raw gold-token density, may be the operative mechanism. The full signal (500 questions, Qwen LLM reader) is awaited from Kaggle.

**Kaggle job queued:** `stage4-density-submodular-hotpotqa-v1` (limit=500, budget=160, Qwen2.5-3B-Instruct).

## 2026-06-28: Stage-4 RESULT (HotpotQA-500, Qwen2.5-3B) — the packer is the lever, and it reframes the thesis

The Kaggle job completed. This is the most important result of the project so far, and it shifts the headline.

**Fixed-policy table (500 questions, budget 160, Qwen2.5-3B-Instruct reader):**

| Policy | recall@5 | EM | F1 | ans_in_ctx | gold_tok_density | gold_doc_cov | distractors |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunk_packed | 0.874 | 0.316 | 0.401 | 0.590 | 0.400 | 0.685 | 2.01 |
| chunk_focused | 0.874 | 0.322 | 0.412 | 0.632 | 0.509 | 0.733 | 1.60 |
| **chunk_submod** | 0.874 | **0.370** | **0.448** | 0.640 | 0.457 | **0.779** | 2.20 |
| ace_focused | 0.859 | 0.320 | 0.421 | 0.630 | 0.452 | 0.768 | 2.13 |
| ace_submod | 0.859 | 0.308 | 0.401 | 0.612 | 0.428 | 0.758 | 2.31 |
| oracle (mixed) | — | **0.460** | **0.570** | 0.684 | 0.491 | 0.788 | 1.96 |

**Headline (paired bootstrap, 10k resamples, same 500 questions):**

- chunk_submod − chunk_focused: **EM +0.048 (p=0.008)**, F1 +0.035 (p=0.039).
- chunk_submod − chunk_packed: **EM +0.054 (p=0.006)**, F1 +0.047 (p=0.011).
- chunk_submod − ace_focused: EM +0.050 (p=0.005), F1 +0.027 (p=0.137).
- The win uses *fewer* reader tokens (145 vs 152), so it is not a "more context" artifact.

**Mediation (pooled over all 2,500 policy-question rows):**

- corr(ans_in_context, F1) = **+0.50** — far above gold_token_density (+0.26), reader gold-coverage (+0.33), and the retrieval signals recall@5 (+0.31) and all_gold@5 (+0.32).
- Conditional: F1 = 0.596 when the answer is present in context vs 0.123 when it is not (gap +0.47). The answer-in-context metric we introduced predicts answer quality *better than document recall*.

**Interpretation — three findings:**

1. The principled budgeted-submodular packer significantly beats both heuristic packers on the project's strongest dataset, at equal-or-lower token cost. This is the first *statistically clean* method win of the project.
2. `ans_in_context` is the dominant mediator and beats retrieval recall as a predictor of answer quality. This is the long-sought explanation for the "ACE wins with lower recall" paradox: what matters under a budget is whether the answer survives into the reader context, not how many gold documents were retrieved.
3. The honest twist: the packer helps **chunk**, not **ACE** (ace_submod − ace_focused is −0.020 F1, n.s.), and ace_focused − chunk_packed is *not* significant on this slice. Likely because ACE already compresses/de-duplicates evidence at the graph level, so submodular redundancy-reduction has less to exploit and only adds distractors. The lever is the packing objective, and it matters most for redundant chunk evidence.

**Strategic consequence:** the paper's center of gravity moves from "ACE graph beats chunk under budget" (fragile across datasets) to "budget-constrained RAG is bottlenecked by answer-in-context, and a principled submodular packer that maximizes complementary gold-evidence coverage delivers a clean, significant gain." ACE becomes one representation in a controlled factorial rather than the headline. The oracle gap (F1 0.570 vs best fixed 0.448) shows large routing headroom over the 2×N packer grid.

## 2026-06-28: Stage-4b queued — MMR baseline + multi-seed robustness

Two reviewer objections must be pre-empted before the headline is credible: "submodular is just MMR" and "it is one seed." Added `mmr_select`/`pack_mmr_run` (Maximal Marginal Relevance, Carbonell & Goldstein 1998) as a packer over the *identical* candidate set, so MMR-vs-submod isolates the selection rule. Expanded the runner to a 2×3 factorial {chunk, ACE} × {focused, mmr, submod} plus the chunk_packed anchor and an oracle over all six packers. Queued `stage4b-mmr-multiseed-hotpotqa-v1`: the full factorial at budget 160 for seeds {42, 13, 7} in one Kaggle session. Decisive checks: does chunk_submod's win replicate across seeds, and does submod ≥ MMR?

## 2026-06-28: Stage-4 mechanism decomposition (offline, seed 42) — the win is value-asymmetric answer-in-context flips

Decomposed the chunk_submod − chunk_focused F1 gain (+0.0353) per question by the answer-in-context transition (exact; bucket contributions sum to the total). Script: `scripts/decompose_packer_winloss.py`.

| Bucket | n | mean F1 delta | contribution |
|---|---:|---:|---:|
| gained_ans (0→1) | 37 | +0.387 | +0.0286 |
| lost_ans (1→0) | 33 | −0.113 | −0.0075 |
| both_have (1→1) | 283 | +0.010 | +0.0056 |
| neither (0→0) | 147 | +0.029 | +0.0086 |

Findings:

- **81% of the aggregate gain comes from the 37 questions where submod newly placed a gold answer into the reader context** (+0.387 F1 each). This is the mediation hypothesis confirmed at the per-question level: the packer wins by getting the answer in, and when it does, F1 jumps ~0.39.
- The flips are nearly balanced in count (37 gained vs 33 lost, net +4) but strongly **asymmetric in value** (+0.387 vs −0.113). Submod is not uniformly better at answer-in-context; it *reshuffles* which answers reach the reader, and the reshuffle is net-positive because its gains are high-value.
- Mechanism behind the flips: on the 37 flipped-in questions, submod raised reader gold-document coverage by +0.28 (vs +0.12 token-density), and across all questions it gets *all* gold docs into context on 289 vs 256 questions (+33). So the driver is **complementary multi-hop coverage**, not raw density — consistent with the aggregate gold_doc_reader_cov gap (0.779 vs 0.733).

Implication: the value-asymmetric flips are exactly the structure a per-question packer router could exploit (keep submod's 37 wins, avoid its 33 losses). This — together with the large oracle gap (F1 0.570 vs 0.448) — makes a learned focused/MMR/submod router the natural Stage-5, but only after the packer win is shown robust (Stage-4b) and general (cross-dataset). Routing has historically underperformed in this project, so it stays a motivated extension, not a promised headline.

## 2026-06-28: Stage-4b RESULT — headline CONFIRMED across 3 seeds, and MMR loses to the heuristic

The multi-seed factorial (seeds {42, 13, 7}, HotpotQA-500, Qwen2.5-3B, budget 160) completed. The headline holds and the MMR control resolves cleanly. Cross-seed mean F1 / EM:

| Policy | mean F1 | mean EM | per-seed F1 (42/13/7) |
|---|---:|---:|---|
| **chunk_submod** | **0.451** | **0.359** | .448 / .455 / .451 |
| chunk_focused | 0.429 | 0.331 | .412 / .431 / .444 |
| chunk_mmr | 0.410 | 0.318 | .390 / .425 / .414 |
| chunk_packed | 0.400 | 0.306 | .401 / .407 / .393 |
| ace_focused | 0.428 | 0.328 | .421 / .432 / .429 |
| ace_mmr | 0.405 | 0.311 | .408 / .399 / .408 |
| ace_submod | 0.406 | 0.317 | .401 / .399 / .420 |
| oracle (mixed) | 0.601 | 0.487 | .598 / .609 / .596 |

Pooled multi-seed paired bootstrap (1,500 (seed,qid) instances, 10k resamples):

| Contrast | ΔF1 [95% CI] | ΔEM [95% CI] |
|---|---|---|
| chunk_submod − chunk_mmr | **+0.042 [+0.021, +0.063]** | **+0.046 [+0.025, +0.067]** |
| chunk_submod − chunk_focused | +0.022 [+0.002, +0.041] | +0.029 [+0.009, +0.048] |
| chunk_submod − chunk_packed | +0.051 [+0.030, +0.072] | +0.053 [+0.031, +0.075] |
| chunk_mmr − chunk_focused | **−0.020 [−0.034, −0.005]** | −0.017 [−0.031, −0.003] |
| ace_submod − ace_focused | −0.021 [−0.039, −0.003] | −0.011 [−0.029, +0.007] |
| chunk_submod − ace_submod | +0.045 [+0.026, +0.063] | +0.043 [+0.024, +0.061] |

Findings:

1. **chunk_submod is the best fixed policy on all three seeds**, and beats every other packer with all intervals excluding zero (EM, and F1 except ace_submod-EM). The win is robust, not a seed artifact.
2. **The MMR objection is dead — empirically.** Plain redundancy-aware reranking (MMR) is *significantly worse than the focused heuristic* (−0.020 F1, p=0.007) and far below submod (−0.042 F1, p<0.001). The full ordering is **submod > focused > packed > mmr**. So the win is not "just redundancy reduction": MMR does exactly that and loses. The submodular objective's joint relevance + query-coverage + facility-location representativeness + concave diversity, optimized per reader token with the Lin–Bilmes guarantee, is what delivers the gain.
3. **The honest twist is now significant:** submod *hurts* ACE (−0.021 F1, p=0.021), and under the submod packer chunk beats ACE (+0.045 F1, p<0.001). Graph compression and principled packing are partial substitutes; the best single system is chunk_submod, and adding the graph representation on top of a good packer is counterproductive on HotpotQA.
4. The oracle (F1 0.601, EM 0.487) sits ~0.15 F1 above the best fixed policy on every seed — large, stable routing headroom over the packer grid.

This is the project's first fully clean, multi-seed, baseline-complete method result. The contribution is crystallized: **a principled budgeted-submodular evidence packer + the answer-in-context diagnostic that explains it.** Remaining gaps for A*: cross-dataset generality (next), budget-curve (mechanistically predicted), and an optional learned router toward the oracle.

## 2026-06-28: Stage-5 queued — cross-dataset generality of the packer

Generality is the biggest remaining gap and the historical weak spot (the old ACE-vs-chunk story was mixed across RAGBench). Queued the full 2×3 factorial on RAGBench CovidQA and ExpertQA at budget 160 (`stage5-crossdataset-packer-v1`). Note: in RAGBench every provided context is gold, so the density metrics saturate; the decisive question is whether chunk_submod still beats chunk_{focused,mmr,packed} on F1/EM under domain shift and free-form answers. If the packer win transfers where the representation win did not, that is the strongest possible argument for the reframing.

## 2026-06-28: Stage-5 RESULT — the packer win does NOT transfer to RAGBench (honest scope boundary)

The cross-dataset factorial completed (RAGBench CovidQA n=246, ExpertQA n=203, budget 160, Qwen2.5-3B). Best fixed F1 by policy:

| Policy | CovidQA F1 | ExpertQA F1 |
|---|---:|---:|
| chunk_packed | 0.106 | 0.023 |
| chunk_focused | **0.126** | 0.030 |
| chunk_mmr | 0.114 | 0.026 |
| chunk_submod | 0.116 | **0.036** |
| ace_focused | 0.134 | 0.034 |
| ace_mmr | **0.134** | 0.035 |
| ace_submod | 0.133 | 0.035 |

Paired bootstrap (chunk_submod − chunk_focused): CovidQA **−0.010 F1 (p=0.30)**; ExpertQA **+0.005 F1 (p=0.15)**. Neither significant. chunk_submod does beat chunk_packed and chunk_mmr on ExpertQA (p≤0.01) but not chunk_focused. EM is ≈0 on both (RAGBench answers are long/free-form), and absolute F1 is low, so these are noisy.

Honest reading:

1. **The clean HotpotQA result `submod > focused` does NOT generalize to RAGBench.** On CovidQA the focused heuristic is nominally better and ACE beats chunk (the *old* ACE story); on ExpertQA submod is nominally best but at noise level. The broad claim "the packer is the lever everywhere" is too strong and is now falsified.
2. **This is consistent with the mechanism, not a contradiction of it.** Submod won on HotpotQA by assembling *complementary multi-hop* evidence amid distractors. RAGBench CovidQA/ExpertQA are largely single-document-grounded with all-context-gold structure — there is little complementary multi-hop coverage to exploit, so submod ≈ focused, exactly as the mechanism predicts. (Supporting: ans_in_context still correlates with F1 at 0.39 on CovidQA; on ExpertQA the metric is degenerate because long answers never appear verbatim.)
3. The correct claim is therefore **conditional and scoped**: principled submodular packing helps specifically for *budget-constrained, distractor-heavy, multi-hop* QA. This is a sharper and more defensible thesis than unconditional dominance, and it converts the RAGBench negative into a scope-defining boundary.

Decisive follow-up: the scope claim makes two falsifiable predictions — (a) on HotpotQA the submod advantage should *grow as the budget tightens* (more pressure → packing matters more), and (b) on a *harder* multi-hop dataset (MuSiQue) submod should win by a *larger* margin than on HotpotQA. Stage-6 tests (a) directly with a budget curve and (b) opportunistically if MuSiQue is still mounted on Kaggle.

## 2026-06-28: Stage-6 queued — budget curve (mechanism test) + MuSiQue if available

Queued `stage6-budgetcurve-musique-v1`. Runs the full 2×3 factorial on HotpotQA seed 42 at budgets {96, 128, 224} (completing a {96,128,160,224} curve with the Stage-4b 160 point); and, if a MuSiQue mount is found under `/kaggle/input`, also runs the MuSiQue factorial at budget 160. Prediction under test: chunk_submod − chunk_focused increases monotonically as budget shrinks.

## 2026-06-28: Stage-6 RESULT — budget-curve prediction FALSIFIED, but a cleaner efficiency story emerges (MuSiQue not mounted)

The budget curve completed (HotpotQA seed 42, budgets 96/128/224; 160 from Stage-4b). MuSiQue was **not** mounted on Kaggle, so only the HotpotQA arm ran. chunk_submod − chunk_focused by budget (paired bootstrap, 500 q):

| budget | submod tok | ΔF1 | p | ΔEM | p |
|---:|---:|---:|---:|---:|---:|
| 96 | 83 | +0.004 | 0.81 | −0.002 | 0.96 |
| 128 | 115 | +0.014 | 0.45 | +0.020 | 0.30 |
| 160 | 145 | **+0.035** | **0.036** | **+0.048** | **0.008** |
| 224 | 204 | +0.017 | 0.26 | +0.018 | 0.29 |

**My monotonicity prediction is falsified.** The submod-vs-focused advantage is *not* largest at the tightest budget; it is an **inverted-U peaking at ~160** and statistically significant only there. Mechanistic reading: at budget 96 only ~2–3 snippets fit, leaving no room to assemble complementary multi-hop evidence, so submod ≈ focused; at 224 nearly everything important fits regardless of selection, so the heuristic catches up; at the intermediate "Goldilocks" budget selection is binding *and* complementarity fits, maximising the principled packer's edge. (Caveat: budgets 96/128/224 are single-seed; the inverted-U shape needs multi-seed confirmation, though the point-estimate spread — +0.004 vs +0.035 vs +0.017 — looks like an effect-size difference, not just power.)

Two cleaner facts survive and are arguably better than the original prediction:

1. **chunk_submod beats naive chunk_packed at every budget** (ΔF1 +0.044/+0.051/+0.047/+0.055; p = 0.022/0.009/0.009/0.002 at 96/128/160/224). Robust, budget-independent dominance over naive packing. (vs MMR: significant at 96 and 160, marginal at 128, n.s. at 224.)
2. **Token efficiency / iso-quality:** submod reaches its answer-quality plateau (EM ≈ 0.37) by budget 160, while focused needs ~224 to approach it. Paired test submod@160 vs focused@224: ΔF1 +0.005 (p=0.80), ΔEM +0.018 (p=0.33) — statistically equal quality at **~30% fewer reader tokens** (145 vs 215). The principled packer buys the heuristic's best quality at a 30% smaller context.

Honest synthesis after Stages 4–6: the contribution is **a principled budgeted-submodular packer that (i) robustly beats naive packing across budgets, (ii) significantly beats the strong focused heuristic and MMR at intermediate budgets on multi-hop QA, and (iii) matches the heuristic's best quality at ~30% lower token cost** — with the gain mediated by answer-in-context (not recall) and confined to distractor-heavy multi-hop retrieval (no RAGBench transfer). The sweeping "packer is the lever everywhere / advantage grows under pressure" stories are both falsified; the scoped, mechanistic version is what the data supports.

Open items: (a) MuSiQue is the key untested axis (harder multi-hop should widen the gap) but is not currently mounted on Kaggle — needs a `musique.jsonl` input added. (b) The oracle sits ~0.15 F1 above best-fixed at every budget; whether a cheap per-question packer router can capture any of that is the next thing to check (feasibility first, on existing data, before spending a Kaggle cycle).

## 2026-06-28: Stage-7 RESULT — MuSiQue does NOT confirm the prediction (retrieval-bottlenecked), but the DIAGNOSTIC generalizes

User mounted MuSiQue; the 2×3 factorial ran at budgets 160 and 96 (limit 500, standard retrieval, Qwen2.5-3B). Best fixed F1 / key contrasts at budget 160:

| Policy | F1 | EM | ans_in_ctx |
|---|---:|---:|---:|
| chunk_packed | 0.137 | 0.088 | 0.224 |
| chunk_focused | 0.124 | 0.070 | 0.216 |
| chunk_mmr | 0.123 | 0.064 | 0.192 |
| chunk_submod | 0.134 | 0.080 | 0.178 |
| ace_focused | **0.150** | 0.084 | 0.216 |
| ace_mmr | 0.148 | 0.084 | 0.208 |
| ace_submod | 0.141 | 0.082 | 0.194 |
| oracle | 0.251 | 0.162 | 0.252 |

Paired bootstrap (budget 160): chunk_submod − chunk_focused **+0.011 F1 (p=0.34)**, − chunk_packed −0.003 (p=0.80), − chunk_mmr +0.011 (p=0.31); ace_focused − chunk_packed +0.013 (p=0.36). **None significant.** Budget 96 shows the same small/null pattern (chunk_submod 0.105 vs focused 0.097 vs packed 0.100).

**The prediction is falsified.** The submod advantage did *not* widen on harder multi-hop; it is not significant on MuSiQue, and naive chunk_packed is just as good. The cause is visible in retrieval: **all_gold@5 = 0.184** (only ~18% of questions have all gold paragraphs retrieved) and ans_in_context ≈ 0.20. MuSiQue is **retrieval-bottlenecked** — the packer cannot assemble complementary evidence that retrieval never surfaced, so packing strategy barely matters. (Consistent with the long-standing MuSiQue finding that its bottleneck is multi-hop retrieval.)

**The silver lining is significant:** corr(ans_in_context, F1) = **+0.54 on MuSiQue** — the strongest of any dataset (HotpotQA 0.50, CovidQA 0.39). So the *diagnostic* generalizes even where the *packer* does not, which proves answer-in-context is not an artifact of the submodular packer — it is a genuine, dataset-independent mediator of RAG answer quality under budget.

**Synthesis across all four datasets — when does principled packing help?** It requires the conjunction of: (i) multi-hop complementary structure (absent on RAGBench → no gain), (ii) retrieval that actually surfaces that evidence (fails on MuSiQue, all_gold@5=0.18 → packing floored), and (iii) budget pressure that is binding but not extreme (the HotpotQA inverted-U). HotpotQA is the setting where all three hold, and there the win is large, significant, and 3-seed robust. This is a narrow but precise and mechanistically-complete scope. The contribution therefore splits cleanly into a **general** part (the answer-in-context diagnostic + the "when does packing help" analysis, validated on 4 datasets) and a **conditional** part (the submodular packer's significant win, demonstrated where the conditions hold). Further packing experiments have diminishing returns; the empirical arc is complete and the next step is writing.

## Stage-8: MuSiQue Retrieval-Unlock Ablation (limit 500, Qwen2.5-3B, top-k 12 / nodes 64 / expand 8)

Targeted ablation: Stage-7's null result was attributed to retrieval bottleneck (all_gold@5=0.18). Stage-8 tests whether widening retrieval (top-k 5→12, top-k-nodes 48→64, max-expanded-docs 5→8) unlocks the packer advantage. Budgets {160, 240}.

**Retrieval coverage — the decisive check:**

| Metric | Stage-7 | Stage-8 | Change |
| --- | ---: | ---: | ---: |
| chunk recall@5 | 0.506 | 0.506 | 0 |
| chunk all_gold@5 | 0.184 | **0.184** | **0** |
| ace recall@5 | 0.481 | 0.481 | 0 |
| ace all_gold@5 | 0.142 | 0.142 | 0 |

**Zero change.** Tripling the retrieval candidates moved all_gold@5 by zero basis points.

Budget 160 packer comparison:

| Policy | F1 | EM | ans_in_ctx | gold_doc_cov |
| --- | ---: | ---: | ---: | ---: |
| chunk_packed | 0.1101 | 0.064 | 0.184 | 0.248 |
| chunk_focused | 0.1244 | 0.074 | 0.190 | 0.340 |
| chunk_mmr | 0.1258 | 0.078 | 0.188 | 0.296 |
| chunk_submod | 0.1275 | 0.076 | 0.168 | 0.363 |
| **ace_focused** | **0.1419** | **0.086** | **0.214** | **0.381** |
| ace_mmr | 0.1194 | 0.068 | 0.188 | 0.343 |
| ace_submod | 0.1321 | 0.082 | 0.186 | 0.350 |
| oracle | 0.2584 | 0.170 | 0.250 | 0.367 |

Budget 240 packer comparison:

| Policy | F1 | EM | ans_in_ctx | gold_doc_cov |
| --- | ---: | ---: | ---: | ---: |
| chunk_packed | 0.1398 | 0.082 | 0.238 | 0.337 |
| chunk_focused | 0.1502 | 0.086 | 0.250 | 0.381 |
| chunk_mmr | 0.1428 | 0.082 | 0.230 | 0.336 |
| chunk_submod | 0.1434 | 0.086 | 0.222 | 0.406 |
| **ace_focused** | **0.1580** | **0.098** | **0.260** | **0.432** |
| ace_mmr | 0.1472 | 0.090 | 0.252 | 0.392 |
| ace_submod | 0.1442 | 0.084 | 0.246 | 0.391 |
| oracle | 0.2924 | 0.190 | 0.304 | 0.401 |

Key packer gap (descriptive, single-run — scope confirmation, no bootstrap needed):
- chunk_submod − chunk_focused: **+0.003 F1** at b160, **−0.007 F1** at b240
- Both smaller than Stage-7's already-null +0.011; no packer advantage regardless of budget

Interpretation: **The MuSiQue retrieval bottleneck is qualitative, not a depth problem.** BAAI/bge-small-en-v1.5 cannot navigate 2–4 hop compositional chains regardless of how many candidates are retrieved; the failure is in the embedding model's inability to surface connected evidence, not in the retrieval depth. This makes the Stage-7 scope condition precisely falsifiable: we tried the obvious fix (more retrieval) and it changed nothing. The packer advantage therefore requires a qualitatively different retriever (iterative multi-hop, chain-of-thought retrieval, or a model with explicit multi-hop capacity) — a clean and specific scope statement for the paper.

ace_focused remains the best fixed policy on MuSiQue at both budgets (F1 0.142 at b160, 0.158 at b240), consistent across Stages 7 and 8.

---

## Stage-9 — HotpotQA larger-reader scaling test (Run A), 2026-06-29

**Question.** The most dangerous reviewer objection to the Stage-4b headline is: *"a stronger reader simply absorbs the redundancy your packer removes, so the packing win is an artifact of using a weak (3B) reader."* This stage tests it directly. We re-ran the **exact** Stage-4b HotpotQA 2×3 factorial — same data (HotpotQA-500), same retrieval (top-k 5 / nodes 48 / expand 5), same budget (160) — changing **only the reader**: Qwen2.5-3B → **Qwen2.5-7B-Instruct**, sharded across the 2×T4 box via `device_map="auto"` (7B fp16 ≈ 14 GB does not fit one 15 GB T4). Seeds {42, 13}. Commit `a87ff64` → results `de2eecc`.

Pooled 2-seed paired bootstrap (10k resamples, `scripts/pool_multiseed_bootstrap.py`, n=1000):

| Contrast (chunk) | F1 Δ | 95% CI | p | EM Δ | p |
| --- | ---: | :---: | ---: | ---: | ---: |
| submod − focused | **−0.010** | [−0.035, +0.015] | **0.45** | −0.008 | 0.54 |
| submod − packed | **+0.054** | [+0.027, +0.081] | **<0.001** | +0.044 | 0.001 |
| mmr − focused | −0.032 | [−0.051, −0.014] | 0.001 | −0.025 | 0.006 |
| ace_submod − ace_focused | −0.020 | [−0.043, +0.003] | 0.088 | −0.026 | 0.023 |

Per-seed best fixed policy: **seed 42 → chunk_submod (F1 0.396)**, **seed 13 → chunk_focused (F1 0.407)** — the headline policy *flips with the seed*.

**Finding — the submod-vs-best-heuristic win VANISHES at 7B.** At 3B, `chunk_submod − chunk_focused` was +4.8–5.4 EM (p<0.01), same sign on all three seeds. At 7B it is −0.8 EM / −1.0 F1 with a CI straddling zero symmetrically (p≈0.45–0.54) and a per-seed sign flip. This is a textbook null: the narrower gap between the principled packer and the best heuristic closes once the reader is strong enough.

**But the mechanism is intact — this is a scope boundary, not a refutation.** The density diagnostics show submod *still* packs strictly more gold evidence than focused at 7B (mean all_gold_reader 0.58 vs 0.52; gold_doc_reader_cov 0.78 vs 0.74, **both seeds**). The packer is doing exactly what it claims; the 7B reader simply no longer *needs* the extra completeness — it extracts the answer from the slightly-less-complete focused context anyway. Consistent with this, the advantage **survives where the density gap is large** (`submod − packed` +0.054 F1, p<0.001 — naive packing is much sparser, all_gold_reader 0.45) and **closes where the density gap is small** (vs the already-good focused heuristic). The two secondary orderings replicate the 3B result unchanged: MMR is significantly worse than focused (p=0.001), and submod still hurts ACE (p=0.023).

**Interpretation — reader capability mediates the packer's benefit.** A stronger reader raises the evidence-density "floor" required to answer, absorbing *small* density advantages (submod vs focused) but not *large* ones (submod vs naive packing). The answer-in-context diagnostic explains **both** regimes with a single variable: the 3B win and the 7B null are the same density edge passed through readers of different sensitivity. This is the third axis of the scope map, alongside multi-hop structure (RAGBench lacks it) and surfacing retrieval (MuSiQue lacks it): **the reader must be the bottleneck** for evidence packing to convert density into accuracy.

**Paper-inclusion status: OPEN (user decision).** This neither cleanly strengthens the raw "packer wins" headline nor refutes the contribution; it converts the headline into a reader-conditional claim and promotes the answer-in-context diagnostic to the unifying explanation. Recommended treatment (B): include as a reader-scale scope boundary and reframe the positive claim as reader-conditional, keeping `submod − packed` (p<0.001 at 7B) as the surviving positive-at-scale result. Alternative (A): omit from main results, one limitations sentence. Empirical record is complete either way; the decision affects only the paper's framing.

**Empirical arc CLOSED (5 positive/scoping stages + 1 reader-scale boundary). All further experimental capacity should go to writing.**

## Stage-10b — 2WikiMultiHopQA full factorial (Run B), 2026-06-29

**Question.** Does the HotpotQA submod-vs-best-heuristic win **replicate on a second multi-hop dataset** that meets all stated scope conditions? 2WikiMultiHopQA is genuinely multi-hop (compositional/comparison/bridge/inference) and, unlike MuSiQue, its retrieval clears our gate. A pre-registered `--retrieval-only` gate (Stage-10a, commit `08b8719` → `8367ac0`) returned chunk **recall@5 = 0.718, all-gold@5 = 0.43** (MuSiQue was 0.184; HotpotQA ~0.76), so the gold evidence is surfaced. We then ran the **exact** Stage-4b factorial — same retrieval (top-k 5 / nodes 48 / expand 5), budget 160, **Qwen2.5-3B** reader, seeds {42,13,7}, 500 questions — changing only the dataset (2Wiki dev, `load_2wiki_local`). Deliberately 3B, not 7B, to isolate dataset-transfer from the Stage-9 reader-scale null. Commit `c1cd98e` → results `036be8d`.

Pooled 3-seed paired bootstrap (10k resamples, `scripts/pool_multiseed_bootstrap.py`, n=1500):

| Contrast (chunk) | F1 Δ | 95% CI | p | EM Δ | p |
| --- | ---: | :---: | ---: | ---: | ---: |
| submod − focused | **−0.008** | [−0.027, +0.012] | **0.44** | −0.006 | 0.57 |
| submod − packed | +0.016 | [−0.004, +0.036] | 0.12 | +0.013 | 0.20 |
| submod − mmr | +0.016 | [−0.003, +0.036] | 0.10 | +0.017 | 0.09 |
| mmr − focused | −0.024 | [−0.038, −0.010] | **0.001** | −0.023 | 0.001 |
| ace_submod − ace_focused | −0.002 | [−0.020, +0.017] | 0.85 | −0.002 | 0.86 |

Per-seed best fixed policy (by F1): **seed 42 → ace_mmr (0.294)**, **seed 13 → chunk_submod (0.267)**, **seed 7 → ace_submod (0.286)** — submod best on only 1 of 3.

**Finding — a clean null for the packer.** Not just the headline (`submod − focused` p=0.44): even `submod − packed`, which held at p<0.001 on HotpotQA *and* survived the 7B reader, is only directional here (p=0.12). The single significant packer-related contrast is the baseline ordering `mmr < focused` (p=0.001), which replicates HotpotQA. On mean F1 `chunk_submod` (0.266) sits *below* the focused/ACE cluster (~0.273–0.275); `ace_focused` is the best fixed policy (0.275 F1 / 0.251 EM).

**The reason is the most informative part — a within-method dissociation.** `chunk_submod` packs strictly more gold than `chunk_focused` (gold-doc reader coverage **+0.054**, all three seeds) yet raises answer-in-context by **−0.007** and F1 by **−0.008**. Coverage rises; answer-in-context does not; accuracy follows answer-in-context, not coverage. Mechanism: on 2Wiki's compositional questions the answer-bearing doc is typically the one focused already ranks first; the *extra* gold submod assembles is bridging evidence that scaffolds the reasoning but does not contain the answer string — so it lifts coverage without lifting answer-in-context. This is the **interventional** counterpart to the §3.3 observational incremental-validity result: move coverage but not answer-in-context, and accuracy does not move.

**The diagnostic survives strongest here.** corr(answer-in-context, F1) = **+0.55** pooled (n=10,500) — tied-highest of any dataset; conditional F1 **0.56** (answer in context) vs **0.08** (not), gap +0.48. The diagnostic does not merely correlate; on 2Wiki it correctly *predicts the packer's own null*.

**Scope-map consequence — a 5th condition.** 2Wiki meets conditions 1–4 (multi-hop ✓, retrieval surfaces evidence ✓ at all-gold@5=0.43, binding budget ✓, 3B reader ✓) and still shows no win, exposing a fifth necessary condition: **the assembled gold must contain the answer** (extractable/span answers, not composed across documents). Like condition 4 (reader scale), this is "mechanism operates but does not convert," not "mechanism cannot operate."

**Paper-inclusion: DECIDED (user-approved 2026-06-29) — include tightly as diagnostic mediation, not as a failed replication.** PAPER_DRAFT updated: new §6.6 (Condition 5), synthesis → §6.7 with a 5th row, §3.4 table + prose add the 2Wiki interventional row, abstract/§1/conclusion move "four"→"five" conditions and "four"→"five" datasets, §2 datasets paragraph adds 2Wiki [Ho 2020], Appendix D adds the 2Wiki reference table. The honest cost (HotpotQA-3B is now the lone packer win across 5 multi-hop settings) is accepted because the paper is diagnostic-led post-Stage-9 and the dissociation is genuine interventional support for the lead contribution. `submod − packed` claims remain HotpotQA-scoped (significant at both reader scales there; only directional on 2Wiki).

**Empirical arc: 5 positive/scoping stages + reader-scale boundary (Stage-9) + second-dataset boundary (Stage-10b). Closed. All remaining capacity → writing.**

## Paper reframe + Stage-11 reader ladder, 2026-06-29 (post Stage-10b)

**Decision (user, 2026-06-29):** rather than carry 2Wiki as a 5th scope condition (breadth), **focus the paper on the reader-scale axis (depth)**. Two moves:

1. **Stage-10b (2Wiki) DEMOTED in the paper** — from a full "Condition 5 + Appendix D" to a **single interventional paragraph in §3.4** supporting the diagnostic, plus a compact Appendix D data reference. The empirical finding is unchanged (clean packer null; coverage↑/AiC-flat/F1-flat dissociation); only the framing changed. Rationale: the dissociation is the cleanest *interventional* evidence for the answer-in-context diagnostic (complements §3.3's observational incremental validity), but as a *scope condition* it cost the paper a "lone-dataset" vulnerability and added breadth where the paper wants depth. So it now lives attached to the diagnostic (Contribution 1), not the scope map (Contribution 3). Scope map reverts to **four conditions**. NOT hiding — the result is fully reported in §3.4/Appendix D/RESULTS.md, just scoped to where it carries signal.

2. **Reader-scale ladder launched (Stage-11, commit aeaa462).** Turns Condition 4 (§6.5) from a single 7B point into a ladder: 3B (§5) + 7B-fp16 (Stage-9) + **7B-4bit (quantization control)** + **14B-4bit**, with **32B-4bit** as a separate follow-up (Stage-12, likely single-seed). Exact §5 HotpotQA factorial (top-k 5 / nodes 48 / expand 5, budget 160, seeds {42,13}), only the reader changes. The 7B-4bit bridge isolates scale from quantization (14B/32B need nf4 to fit 2×T4). main.py pip-installs bitsandbytes>=0.43 (not in frozen-zip reqs); overlay already ships the load_in_4bit generator+runner. Per-seed *metrics.csv skip-logic = resumable across the 9h session limit. **Expected payoff:** a quantitative "curation stops paying off beyond ~N-B reader" threshold — a focused, citable deepening of the strongest scope boundary. Decision rule when results land: if 7B-4bit ≈ 7B-fp16 on submod−focused, quantization is not confounding and the ladder trend is a clean scale effect.

**Paper draft state:** §6.5 reframed to "reader-scale ladder" with 3B/7B rungs reported and a marker for 14B/32B (`<!-- ladder: ... Stage-11/12 -->`); abstract/§1/conclusion updated to "ladder" language + "four conditions"; §3.4 holds the 2Wiki interventional paragraph; "five datasets" retained for the diagnostic's external validity (HotpotQA, MuSiQue, 2Wiki, CovidQA, ExpertQA-degenerate).

## Stage-11 — HotpotQA reader-scale ladder: 7B-4bit (quantization control) + 14B-4bit, 2026-06-29

Exact §5 HotpotQA-500 2×3 factorial (top-k 5 / nodes 48 / expand 5, budget 160), seeds {42,13}, only the reader changes. Two new rungs: **Qwen2.5-7B-Instruct in 4-bit** (nf4 + double-quant; the quantization control against the Stage-9 7B-fp16 run) and **Qwen2.5-14B-Instruct in 4-bit**. Run completed on the (now-expired) GPU account and pushed as commit `e126e4c` (status complete; all four `*metrics.csv` + `*per_question.csv` present, 500 q × 7 policies each). Pooled 2-seed paired bootstrap (10k resamples, vectorized, keyed on (file,qid); n=1000).

**The ladder — `chunk_submod − chunk_focused` (packer's edge over the best heuristic), F1:**

| Reader | ΔF1 | 95% CI | p | ΔEM | best fixed policy | verdict |
|---|---|---|---|---|---|---|
| Qwen2.5-3B (§5) | **+0.035** | [+0.001, +0.069] | **0.04** | +0.048 | chunk_submod | **submod wins** |
| Qwen2.5-7B fp16 (Stage-9) | −0.010 | [−0.035, +0.015] | 0.45 | −0.008 | chunk_focused | null |
| Qwen2.5-7B 4bit (bridge) | −0.008 | [−0.032, +0.017] | 0.55 | −0.006 | chunk_focused (both seeds) | null |
| Qwen2.5-14B 4bit | **−0.029** | [−0.052, −0.006] | **0.013** | −0.027 (p=0.027) | chunk_focused / ace_focused | **focused wins** |

**Two clean reads:**

1. **Quantization is not a confound.** 7B-4bit (`submod − focused` −0.008 F1, p=0.55) ≈ 7B-fp16 (−0.010 F1, p=0.45) — same sign, same magnitude, same null, and the same best fixed policy (chunk_focused on both seeds). The bridge does exactly its job: the 14B-4bit result below is a genuine **scale** effect, not a 4-bit artifact.

2. **The packer's edge over the best heuristic is monotone-decreasing in reader scale, and crosses zero by ~7B.** +0.035 (3B, sig+) → ≈0 (7B, null, both precisions) → **−0.029 (14B, sig−)**. At 14B the *simple focused heuristic significantly beats* the submodular packer (p=0.013 F1, p=0.027 EM). The scope boundary "the reader must be the bottleneck" (condition 4) is now pinned with a threshold *and a reversal*, not just a vanishing point.

**`chunk_submod − chunk_packed` (vs naive packing) stays significantly positive at every rung:** +0.054 (3B) / +0.054 (7B-fp16) / **+0.055 (7B-4bit, p<0.001)** / **+0.044 (14B-4bit, p=0.001)**. The submodular *objective* robustly beats dumping text into the budget at all reader scales; it is specifically the edge over the *smart* focused heuristic that scale-erodes and reverses. Secondary orderings replicate up the ladder: `mmr < focused` (7B p=0.001, 14B p<0.001); `submod hurts ace` strengthens with scale (ace_submod−ace_focused: 7B −0.020 p=0.076 → 14B −0.041 p<0.001).

**Mechanism (why 14B reverses, not just nulls).** At 14B, `chunk_submod` still packs *more* gold (gold-doc reader coverage 0.779 vs focused 0.733) and equal-or-higher answer-in-context (0.64 vs 0.63) — but also more distractor docs (≈2.2 vs 1.6) and more snippets (≈4.6 vs 3.3). The capable 14B reader already extracts the answer from focused's cleaner pack, so submod's extra bridging evidence buys no accuracy while its extra distractors cost a little. Curation's overhead exceeds its benefit once the reader is strong enough to navigate the focused pack unaided — the same answer-in-context logic as the 3B win, now running the other way. Oracle F1 climbs with scale (3B → 7B ≈0.55 → 14B ≈0.60), so headroom remains; it is simply not realized by *this* packer at this reader scale.

**Provenance note:** this is the *complete* Stage-11 (not a partial). The GPU account expired afterward; results were recovered from the pushed `e126e4c`. The new (backup) GPU account's loop, on a fresh `/kaggle/working`, will **re-run Stage-11 from scratch** unless `control/main.py` is advanced — the per-seed skip-markers live in the (now gone) old session's working dir, not in the repo. Stage-12 (32B) decision pending: the 3-rung ladder is already monotone + significant, so 32B is confirmatory, weighed against the backup account's limited credits.
