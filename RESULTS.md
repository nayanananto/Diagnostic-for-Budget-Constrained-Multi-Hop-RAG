# Current Smoke-Test Results

Command:

```bash
python -m experiments.run_mvp --config configs\toy.yaml
```

Output file:

```text
results/toy_lexical_identity_dims128_nodes8_docs5_metrics.csv
```

## Toy Dataset Result

| Method | Recall@5 | All-Gold@5 | Avg Evidence Tokens | Avg Expanded Docs | Verifier Conflict Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chunk RAG | 1.0000 | 1.0000 | 85.00 | 5.00 | 0.25 |
| ACE Graph | 1.0000 | 1.0000 | 53.25 | 3.50 | 0.25 |
| ACE Graph, No Conflict Expansion | 1.0000 | 1.0000 | 53.25 | 3.50 | 0.25 |

Interpretation:

The offline toy run confirms the harness is working and shows the desired first signal: ACE graph retrieval preserves gold-document coverage while reducing retrieved evidence tokens. This is not a research result yet; the next meaningful test is HotpotQA/MuSiQue with dense embeddings and PCA or binary compression.

## HotpotQA Stage-1 Result

Best current 1000-example HotpotQA result:

| Method | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| Chunk RAG | 0.8560 | 0.7230 | 0.0499 | 408.94 |
| ACE Graph, Identity | 0.8310 | 0.6840 | 0.0845 | 136.63 |

Interpretation:

ACE graph retrieval preserves most evidence coverage while using about one-third of the retrieved evidence tokens. This is the strongest early signal so far.

Important tuning results:

| Config | Recall@5 | All-Gold@5 | Extractive F1 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: | ---: |
| ACE identity, nodes=48, docs=5 | 0.8310 | 0.6840 | 0.0845 | 136.63 |
| ACE identity, nodes=48, docs=6 | 0.8310 | 0.6840 | 0.0861 | 166.25 |
| ACE identity, nodes=64, docs=5 | 0.8310 | 0.6840 | 0.0845 | 136.68 |
| ACE truncate384, nodes=48, docs=5 | 0.8310 | 0.6840 | 0.0845 | 136.68 |
| ACE truncate256, nodes=48, docs=5 | 0.8165 | 0.6610 | 0.0826 | 136.29 |

Current best setting remains:

```text
compressor=identity
top_k_nodes=48
max_expanded_docs=5
```

Next compression tests:

```text
truncate:320
truncate:128
```

## Qwen Stage-2 Reader Results

HotpotQA limit-200 with `Qwen/Qwen2.5-1.5B-Instruct`:

| Reader Context | Method | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: |
| full sources | Chunk RAG | 0.1600 | 0.2619 | 402.20 |
| full sources | ACE Graph | 0.1400 | 0.2385 | 419.90 |
| snippets | Chunk RAG | 0.1400 | 0.2509 | 377.13 |
| snippets | ACE Graph | 0.1200 | 0.2329 | 346.35 |

Interpretation:

Snippet context is better than raw graph-hit context for Qwen. It preserves much of the source-passage answer quality while reducing reader tokens, but it does not yet restore the strong token-efficiency advantage seen in retrieval-only experiments.

## Autonomous Stage-2 Packed Evidence

HotpotQA limit-200 with `Qwen/Qwen2.5-1.5B-Instruct`:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.1600 | 0.2619 | 402.20 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.1250 | 0.2337 | 268.31 |
| ace_sources | ACE Graph | sources | 0.1400 | 0.2385 | 419.90 |
| ace_packed_160 | ACE Graph | packed snippets | 0.1350 | 0.2227 | 150.24 |
| ace_packed_220 | ACE Graph | packed snippets | 0.1450 | 0.2440 | 208.94 |
| ace_packed_280 | ACE Graph | packed snippets | 0.1300 | 0.2530 | 266.75 |
| ace_packed_340 | ACE Graph | packed snippets | 0.1100 | 0.2375 | 318.31 |

Interpretation:

`ace_packed_280` is the strongest Stage-2 policy so far. It comes close to the full chunk-source reader while using substantially less reader context, and it beats the packed chunk control at nearly the same budget. This supports the emerging claim that graph retrieval plus budget-aware evidence packing is stronger than retrieval alone.

HotpotQA limit-500 follow-up:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.1560 | 0.2677 | 408.11 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.1560 | 0.2720 | 268.69 |
| ace_sources | ACE Graph | sources | 0.1340 | 0.2494 | 424.91 |
| ace_packed_220 | ACE Graph | packed snippets | 0.1480 | 0.2233 | 209.69 |
| ace_packed_280 | ACE Graph | packed snippets | 0.1460 | 0.2449 | 267.53 |
| ace_focused_220 | ACE Graph | focused packed | 0.1580 | 0.2509 | 209.10 |
| ace_focused_280 | ACE Graph | focused packed | 0.1340 | 0.2358 | 266.16 |

Interpretation:

The limit-500 result showed that naive ACE packed snippets were weaker than the packed chunk control. The focused packer improved ACE substantially at a lower budget and became the strongest ACE generation-stage policy so far. However, the packed chunk control remains a strong baseline and must be retained in future experiments.

HotpotQA limit-1000 follow-up:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.1350 | 0.2460 | 408.90 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.1390 | 0.2395 | 269.19 |
| ace_sources | ACE Graph | sources | 0.1220 | 0.2382 | 418.02 |
| ace_packed_280 | ACE Graph | packed snippets | 0.1440 | 0.2543 | 267.88 |
| ace_focused_220 | ACE Graph | focused packed | 0.1320 | 0.2249 | 208.66 |

Interpretation:

The limit-1000 result is the strongest end-to-end result so far. `ace_packed_280` outperforms both the full chunk-source reader and the packed chunk control while using much less reader context than full chunk sources. This makes ACE graph retrieval plus packed evidence the current best method direction.

HotpotQA seed-validation result:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.1350 | 0.2460 | 408.90 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.1390 | 0.2395 | 269.19 |
| ace_sources | ACE Graph | sources | 0.1220 | 0.2382 | 418.02 |
| ace_packed_280 | ACE Graph | packed snippets | 0.1440 | 0.2543 | 267.88 |
| ace_focused_220 | ACE Graph | focused packed | 0.1320 | 0.2249 | 208.66 |

Interpretation:

The seed-validation table preserves the same main conclusion: `ace_packed_280` is the strongest tested policy and uses substantially less reader context than full chunk-source RAG.

HotpotQA baseline-rich validation:

| Policy | Method | Context | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.8545 | 0.7210 | 0.1380 | 0.2463 | 403.17 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.8545 | 0.7210 | 0.1100 | 0.2012 | 269.52 |
| bm25_sources | BM25 RAG | sources | 0.6370 | 0.3620 | 0.1240 | 0.2140 | 409.17 |
| bm25_packed_280 | BM25 RAG | packed snippets | 0.6370 | 0.3620 | 0.0970 | 0.1844 | 268.98 |
| hybrid_sources | Hybrid RAG | sources | 0.7890 | 0.5980 | 0.1280 | 0.2195 | 402.41 |
| hybrid_packed_280 | Hybrid RAG | packed snippets | 0.7890 | 0.5980 | 0.1010 | 0.1913 | 268.72 |
| ace_sources | ACE Graph | sources | 0.8245 | 0.6680 | 0.1120 | 0.2060 | 415.92 |
| ace_packed_280 | ACE Graph | packed snippets | 0.8245 | 0.6680 | 0.1150 | 0.2151 | 267.46 |
| ace_focused_220 | ACE Graph | focused packed | 0.8245 | 0.6680 | 0.1290 | 0.2105 | 208.71 |

Interpretation:

The stronger baseline run narrows the claim. Full chunk-source RAG remains the strongest answer-quality setting, but it uses substantially more reader context. Under a comparable packed context budget, `ace_packed_280` outperforms packed chunk, packed BM25, and packed hybrid controls while using slightly less reader context. ACE retrieval also beats the hybrid baseline on evidence coverage, though it does not beat chunk retrieval coverage. This supports a budget-constrained RAG claim rather than an unconditional accuracy claim.

HotpotQA seed-7 robustness validation:

| Policy | Method | Context | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.8625 | 0.7410 | 0.1450 | 0.2475 | 400.38 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.8625 | 0.7410 | 0.1390 | 0.2381 | 269.45 |
| bm25_sources | BM25 RAG | sources | 0.6390 | 0.3720 | 0.1150 | 0.2095 | 406.18 |
| bm25_packed_280 | BM25 RAG | packed snippets | 0.6390 | 0.3720 | 0.1080 | 0.1948 | 268.78 |
| hybrid_sources | Hybrid RAG | sources | 0.7945 | 0.6120 | 0.1320 | 0.2385 | 397.37 |
| hybrid_packed_280 | Hybrid RAG | packed snippets | 0.7945 | 0.6120 | 0.1110 | 0.2123 | 267.83 |
| ace_sources | ACE Graph | sources | 0.8115 | 0.6530 | 0.1280 | 0.2411 | 405.98 |
| ace_packed_280 | ACE Graph | packed snippets | 0.8115 | 0.6530 | 0.1350 | 0.2320 | 267.18 |
| ace_focused_220 | ACE Graph | focused packed | 0.8115 | 0.6530 | 0.1330 | 0.2111 | 208.34 |

Interpretation:

The seed-7 run makes the HotpotQA result mixed rather than a clean replication. `ace_packed_280` remains stronger than packed BM25 and packed hybrid controls, and it uses slightly less context than packed chunk, but packed chunk has higher Qwen F1 on this seed. Full source chunk remains the strongest absolute answer-quality policy. The safer HotpotQA claim is that ACE is a competitive compact-evidence method with wins against lexical and hybrid packed controls, while chunk packing remains a strong baseline that must be reported honestly.

HotpotQA additional robustness validation:

| Policy | Method | Context | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.8530 | 0.7190 | 0.1290 | 0.2468 | 399.61 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.8530 | 0.7190 | 0.1260 | 0.2384 | 269.51 |
| bm25_sources | BM25 RAG | sources | 0.6455 | 0.3840 | 0.1040 | 0.2055 | 403.09 |
| bm25_packed_280 | BM25 RAG | packed snippets | 0.6455 | 0.3840 | 0.1190 | 0.2109 | 269.50 |
| hybrid_sources | Hybrid RAG | sources | 0.8095 | 0.6340 | 0.1300 | 0.2336 | 398.20 |
| hybrid_packed_280 | Hybrid RAG | packed snippets | 0.8095 | 0.6340 | 0.1250 | 0.2160 | 268.46 |
| ace_sources | ACE Graph | sources | 0.8110 | 0.6470 | 0.1200 | 0.2217 | 414.26 |
| ace_packed_280 | ACE Graph | packed snippets | 0.8110 | 0.6470 | 0.1230 | 0.2221 | 267.15 |
| ace_focused_220 | ACE Graph | focused packed | 0.8110 | 0.6470 | 0.1350 | 0.2250 | 208.60 |

Interpretation:

This additional HotpotQA seed confirms the mixed robustness pattern. Chunk packing remains the strongest packed baseline on answer F1, while ACE remains better than packed BM25 and packed hybrid. The focused ACE policy is the best ACE setting on this slice and uses much less context than packed chunk, but it still does not beat packed chunk on F1. This reinforces the need for an adaptive router rather than a fixed ACE policy.

## MuSiQue Stage Results

MuSiQue retrieval-only limit-300:

| Method | Recall@5 | All-Gold@5 | Avg Evidence Tokens |
| --- | ---: | ---: | ---: |
| Chunk RAG | 0.5450 | 0.2333 | 399.27 |
| ACE Graph | 0.5017 | 0.1700 | 142.72 |

MuSiQue Stage-2 limit-200 with `Qwen/Qwen2.5-1.5B-Instruct`:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.0350 | 0.0895 | 375.35 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.0150 | 0.0829 | 260.39 |
| ace_sources | ACE Graph | sources | 0.0350 | 0.0821 | 374.92 |
| ace_packed_220 | ACE Graph | packed snippets | 0.0050 | 0.0770 | 205.03 |
| ace_packed_280 | ACE Graph | packed snippets | 0.0100 | 0.0766 | 259.86 |
| ace_focused_220 | ACE Graph | focused packed | 0.0100 | 0.0740 | 203.53 |
| ace_focused_280 | ACE Graph | focused packed | 0.0100 | 0.0977 | 255.91 |

Interpretation:

MuSiQue is much harder than HotpotQA. The first MuSiQue result does not cleanly reproduce the HotpotQA ACE win. The best ACE policy is focused packed evidence at a moderate budget, but the bigger issue is retrieval coverage. The next MuSiQue experiment should test slightly larger ACE retrieval expansion rather than additional reader packing variants.

MuSiQue expanded ACE retrieval test:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.0350 | 0.0895 | 375.35 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.0150 | 0.0829 | 260.39 |
| ace_sources | ACE Graph | sources | 0.0300 | 0.0824 | 612.34 |
| ace_packed_280 | ACE Graph | packed snippets | 0.0050 | 0.0623 | 268.30 |
| ace_focused_280 | ACE Graph | focused packed | 0.0250 | 0.0750 | 266.22 |

Interpretation:

The expanded ACE retrieval setting did not improve MuSiQue. It increased source context substantially and degraded packed-evidence results. This suggests the current ACE graph method should not be further tuned on MuSiQue without a stronger multi-hop retrieval design.

MuSiQue bridge-aware ACE Stage-2:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.0350 | 0.0895 | 375.35 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.0150 | 0.0829 | 260.39 |
| ace_bridge_sources | ACE Bridge | sources | 0.0300 | 0.0789 | 388.38 |
| ace_bridge_packed_280 | ACE Bridge | packed snippets | 0.0150 | 0.0943 | 256.24 |
| ace_bridge_focused_280 | ACE Bridge | focused packed | 0.0150 | 0.0982 | 250.41 |

Interpretation:

Bridge-aware ACE gives the strongest MuSiQue result so far, but only modestly. It improves over the standard MuSiQue ACE setting and beats the chunk baselines on Qwen F1 while using less reader context than chunk sources. The absolute scores remain low, so MuSiQue should still be framed as a hard stress test and motivation for stronger reasoning-aware graph retrieval.

MuSiQue bridge-aware ACE limit-500 validation:

| Policy | Method | Context | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.0320 | 0.0893 | 412.44 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.0260 | 0.0752 | 256.02 |
| ace_bridge_sources | ACE Bridge | sources | 0.0360 | 0.0990 | 428.30 |
| ace_bridge_packed_280 | ACE Bridge | packed snippets | 0.0360 | 0.0993 | 252.74 |
| ace_bridge_focused_280 | ACE Bridge | focused packed | 0.0220 | 0.0832 | 246.21 |

Interpretation:

The limit-500 bridge validation keeps the weak-positive MuSiQue pattern. `ace_bridge_packed_280` is the best policy by Qwen F1 and uses much less context than chunk sources. Retrieval coverage is still lower than chunk retrieval, so the result should be framed carefully: bridge-aware ACE helps generation under a context budget, but the harder multi-hop retrieval problem is not fully solved.

MuSiQue baseline-rich bridge validation:

| Policy | Method | Context | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_sources | Chunk RAG | sources | 0.5060 | 0.1840 | 0.0320 | 0.0893 | 412.44 |
| chunk_packed_280 | Chunk RAG | packed snippets | 0.5060 | 0.1840 | 0.0260 | 0.0752 | 256.02 |
| bm25_sources | BM25 RAG | sources | 0.3830 | 0.0800 | 0.0240 | 0.0607 | 406.99 |
| bm25_packed_280 | BM25 RAG | packed snippets | 0.3830 | 0.0800 | 0.0160 | 0.0532 | 256.49 |
| hybrid_sources | Hybrid RAG | sources | 0.4990 | 0.1620 | 0.0380 | 0.0873 | 403.70 |
| hybrid_packed_280 | Hybrid RAG | packed snippets | 0.4990 | 0.1620 | 0.0360 | 0.0820 | 255.67 |
| ace_bridge_sources | ACE Bridge | sources | 0.4810 | 0.1800 | 0.0340 | 0.0982 | 427.86 |
| ace_bridge_packed_280 | ACE Bridge | packed snippets | 0.4810 | 0.1800 | 0.0400 | 0.1065 | 251.49 |
| ace_bridge_focused_280 | ACE Bridge | focused packed | 0.4810 | 0.1800 | 0.0240 | 0.0836 | 245.23 |

Interpretation:

This is the strongest MuSiQue result so far. `ace_bridge_packed_280` gives the best Qwen EM and F1 among chunk, BM25, hybrid, and ACE policies while using less reader context than packed chunk and packed hybrid. Retrieval coverage is still not a clean win: chunk retrieval has slightly better recall and all-gold coverage, while ACE bridge is close on all-gold coverage and stronger at turning retrieved evidence into answers. The defensible claim is that bridge-aware ACE improves budgeted generation on MuSiQue, not that it solves MuSiQue retrieval.

## HotpotQA Stage-3 Adaptive Router

Stage-3 HotpotQA validation with fixed compact policies, a rule router, and an oracle router.
These numbers supersede the earlier Stage-3 reader run because the earlier run used right-padding with a decoder-only Qwen model, which produced unreliable batched generation.

| Policy | Router Type | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | fixed | 0.8545 | 0.7210 | 0.3130 | 0.4157 | 269.52 |
| bm25_packed_280 | fixed | 0.6370 | 0.3620 | 0.2620 | 0.3616 | 268.98 |
| hybrid_packed_280 | fixed | 0.7890 | 0.5980 | 0.2760 | 0.3833 | 268.72 |
| ace_packed_280 | fixed | 0.8245 | 0.6680 | 0.3110 | 0.4131 | 267.46 |
| ace_focused_220 | fixed | 0.8245 | 0.6680 | 0.3240 | 0.4374 | 208.71 |
| router_rule | rule | 0.8225 | 0.6580 | 0.3220 | 0.4350 | 236.87 |
| router_oracle | oracle | 0.8200 | 0.6580 | 0.4630 | 0.5960 | 221.50 |

Interpretation:

The corrected reader run gives a much stronger HotpotQA result. Focused ACE beats compact chunk retrieval on answer quality while using substantially less context. The conservative rule router nearly matches the best fixed ACE result, but it does not yet beat it. The oracle router remains the strongest diagnostic signal: per-question policy choice has large headroom. The next publishable step is cross-seed validation and then a calibrated or learned router trained on the exported routing features.

## HotpotQA Stage-3 Seed-7 Robustness

The same corrected Stage-3 setup was rerun on a different HotpotQA sample.

| Policy | Router Type | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | fixed | 0.8625 | 0.7410 | 0.3260 | 0.4335 | 269.45 |
| bm25_packed_280 | fixed | 0.6390 | 0.3720 | 0.2850 | 0.3832 | 268.78 |
| hybrid_packed_280 | fixed | 0.7945 | 0.6120 | 0.3020 | 0.4002 | 267.83 |
| ace_packed_280 | fixed | 0.8115 | 0.6530 | 0.3420 | 0.4496 | 267.18 |
| ace_focused_220 | fixed | 0.8115 | 0.6530 | 0.3310 | 0.4240 | 208.34 |
| router_rule | rule | 0.8275 | 0.6750 | 0.3300 | 0.4324 | 235.74 |
| router_oracle | oracle | 0.8080 | 0.6470 | 0.4940 | 0.6136 | 222.31 |

Interpretation:

The corrected HotpotQA result is robust across these two seeds. ACE packed is the best compact fixed policy on seed seven, and focused ACE remains a strong lower-context variant. The rule router is not yet better than the best fixed policy, but oracle routing again shows large headroom. This supports the next method step: learn or calibrate the router rather than hand-tuning it.

## HotpotQA Stage-3 Seed-13 Robustness

The corrected Stage-3 setup was rerun on a third HotpotQA sample.

| Policy | Router Type | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | fixed | 0.8570 | 0.7340 | 0.3320 | 0.4297 | 268.00 |
| bm25_packed_280 | fixed | 0.6365 | 0.3870 | 0.2660 | 0.3628 | 268.29 |
| hybrid_packed_280 | fixed | 0.7985 | 0.6170 | 0.3070 | 0.4060 | 268.10 |
| ace_packed_280 | fixed | 0.8225 | 0.6740 | 0.3340 | 0.4377 | 267.25 |
| ace_focused_220 | fixed | 0.8225 | 0.6740 | 0.3280 | 0.4306 | 208.45 |
| router_rule | rule | 0.8300 | 0.6830 | 0.3470 | 0.4444 | 235.16 |
| router_oracle | oracle | 0.8150 | 0.6590 | 0.4790 | 0.5995 | 220.62 |

Interpretation:

Seed thirteen strengthens the adaptive-router story: the conservative rule router is the best non-oracle policy on answer quality while using less context than the fixed packed policies. The fixed ACE policies remain strong, and oracle routing again shows substantial headroom.

## HotpotQA Stage-3 Three-Seed Aggregate

Corrected Stage-3 aggregate over three HotpotQA samples:

| Policy | Mean Qwen EM | Mean Qwen F1 | Mean Reader Tokens | Mean Recall@5 | Mean All-Gold@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | 0.3237 | 0.4263 | 268.99 | 0.8580 | 0.7320 |
| bm25_packed_280 | 0.2710 | 0.3692 | 268.68 | 0.6375 | 0.3737 |
| hybrid_packed_280 | 0.2950 | 0.3965 | 268.22 | 0.7940 | 0.6090 |
| ace_packed_280 | 0.3290 | 0.4335 | 267.30 | 0.8195 | 0.6650 |
| ace_focused_220 | 0.3277 | 0.4307 | 208.50 | 0.8195 | 0.6650 |
| router_rule | 0.3330 | 0.4373 | 235.92 | 0.8267 | 0.6720 |
| router_oracle | 0.4787 | 0.6030 | 221.48 | 0.8143 | 0.6547 |

Interpretation:

Across three corrected HotpotQA runs, the rule router has the best mean answer quality among deployable policies, and focused ACE gives the best compact fixed-policy efficiency. The retrieval metrics still favor chunk retrieval on raw gold-document coverage, but answer quality favors ACE-style compact evidence. This is the strongest HotpotQA evidence so far.

## Offline Router Transfer

The exported router feature files were used for offline transfer tests without rerunning retrieval or Qwen.

Train on corrected seed forty-two features, test on corrected seed seven:

| Policy | Type | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: |
| ace_packed_280 | fixed | 0.3420 | 0.4496 | 267.18 |
| ace_focused_220 | fixed | 0.3310 | 0.4240 | 208.34 |
| router_rule | rule | 0.3300 | 0.4324 | 235.74 |
| router_learned_f1_tree | learned | 0.3340 | 0.4268 | 207.83 |
| router_oracle | oracle | 0.4940 | 0.6136 | 222.31 |

Train on corrected seed seven features, test on corrected seed forty-two:

| Policy | Type | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: |
| ace_focused_220 | fixed | 0.3240 | 0.4374 | 208.71 |
| ace_packed_280 | fixed | 0.3110 | 0.4131 | 267.46 |
| router_rule | rule | 0.3220 | 0.4350 | 236.87 |
| router_learned_f1_tree | learned | 0.3230 | 0.4372 | 208.37 |
| router_oracle | oracle | 0.4630 | 0.5960 | 221.50 |

Interpretation:

The current learned router does not transfer beyond the best fixed ACE policy. It mostly learns to choose focused ACE, which is efficient but not enough to close the oracle gap. This is an important negative result: the router needs stronger deploy-time features, not just a better classifier over the current retrieval-confidence features.

## MuSiQue Corrected Stage-3 Router

Corrected Stage-3 MuSiQue bridge run with the decoder-safe Qwen reader:

| Policy | Router Type | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | fixed | 0.4896 | 0.3000 | 0.0350 | 0.1034 | 269.25 |
| bm25_packed_280 | fixed | 0.3367 | 0.1800 | 0.0100 | 0.0523 | 266.06 |
| hybrid_packed_280 | fixed | 0.4688 | 0.2700 | 0.0500 | 0.1001 | 267.77 |
| ace_bridge_packed_280 | fixed | 0.4550 | 0.2800 | 0.0600 | 0.1188 | 265.24 |
| ace_bridge_focused_280 | fixed | 0.4550 | 0.2800 | 0.0550 | 0.1180 | 262.77 |
| router_rule | rule | 0.4446 | 0.2600 | 0.0400 | 0.1049 | 266.46 |
| router_oracle | oracle | 0.4062 | 0.2450 | 0.1150 | 0.2154 | 252.95 |

Interpretation:

Bridge-aware ACE remains the best fixed compact MuSiQue policy under the corrected Qwen reader, despite lower raw recall than chunk retrieval. The hand-written router is not strong enough on MuSiQue because it selects chunk-style evidence too often. The oracle result is still much higher, so adaptive routing has headroom, but manual threshold tuning is not the right path. A small offline sweep over current retrieval-confidence features did not find a simple rule that improves MuSiQue without hurting HotpotQA seeds. The next controlled experiment is a stronger open Qwen reader with retrieval and routing held fixed.

## HotpotQA Qwen 3B Validation

Kaggle Stage-3 HotpotQA validation with `Qwen/Qwen2.5-3B-Instruct`, seed 42, limit 500:

| Policy | Router Type | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | fixed | 0.8740 | 0.7580 | 0.3240 | 0.4194 | 269.28 |
| bm25_packed_280 | fixed | 0.6580 | 0.3960 | 0.2480 | 0.3394 | 269.15 |
| hybrid_packed_280 | fixed | 0.8080 | 0.6280 | 0.2860 | 0.3792 | 268.61 |
| ace_packed_280 | fixed | 0.8590 | 0.7260 | 0.3460 | 0.4489 | 266.50 |
| ace_focused_220 | fixed | 0.8590 | 0.7260 | 0.3420 | 0.4452 | 208.95 |
| router_rule | rule | 0.8420 | 0.6920 | 0.3320 | 0.4267 | 235.59 |
| router_oracle | oracle | 0.8380 | 0.6920 | 0.4860 | 0.6085 | 222.37 |

Interpretation:

The stronger Qwen reader improves the HotpotQA story. `ace_packed_280` is the best deployable policy on answer quality, and `ace_focused_220` is almost tied while using much less evidence context. Chunk retrieval still has stronger raw retrieval coverage, but ACE turns a slightly lower retrieval set into better answers. The rule router is not yet reliable enough to beat fixed ACE, while the oracle result again shows large per-question routing headroom.

## HotpotQA Qwen 3B Three-Seed Aggregate

Kaggle Stage-3 HotpotQA validation with `Qwen/Qwen2.5-3B-Instruct`, limit 500, seeds 42, 7, and 13:

| Policy | Mean Qwen EM | Mean Qwen F1 | Mean Reader Tokens | Mean Recall@5 | Mean All-Gold@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | 0.3227 | 0.4270 | 268.90 | 0.8750 | 0.7613 |
| bm25_packed_280 | 0.2680 | 0.3677 | 268.78 | 0.6533 | 0.4013 |
| hybrid_packed_280 | 0.3080 | 0.4081 | 268.16 | 0.8080 | 0.6327 |
| ace_packed_280 | 0.3413 | 0.4447 | 266.91 | 0.8427 | 0.7047 |
| ace_focused_220 | 0.3360 | 0.4438 | 208.51 | 0.8427 | 0.7047 |
| router_rule | 0.3213 | 0.4293 | 235.01 | 0.8367 | 0.6900 |
| router_oracle | 0.4793 | 0.6077 | 220.79 | 0.8337 | 0.6900 |

Interpretation:

The stronger-reader HotpotQA result now holds across multiple sampled runs. Fixed ACE packed evidence gives the best average deployable answer quality, while focused ACE is almost tied at a much lower context budget. Chunk retrieval has the highest raw gold-document coverage, but lower answer quality, which supports the central argument that graph-structured compressed evidence can be more useful to the reader than simply maximizing document coverage. The hand-written router remains weaker than fixed ACE on average, so it should not be the headline method.

## HotpotQA Qwen 3B Limit-1000 Scaling

Kaggle Stage-3 HotpotQA validation with `Qwen/Qwen2.5-3B-Instruct`, seed 42, limit 1000:

| Policy | Router Type | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Evidence Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | fixed | 0.8545 | 0.7210 | 0.3080 | 0.4123 | 269.52 |
| bm25_packed_280 | fixed | 0.6370 | 0.3620 | 0.2540 | 0.3502 | 268.98 |
| hybrid_packed_280 | fixed | 0.7890 | 0.5980 | 0.2810 | 0.3820 | 268.72 |
| ace_packed_280 | fixed | 0.8245 | 0.6680 | 0.3300 | 0.4338 | 267.46 |
| ace_focused_220 | fixed | 0.8245 | 0.6680 | 0.3120 | 0.4170 | 208.71 |
| router_rule | rule | 0.8225 | 0.6580 | 0.3060 | 0.4150 | 236.87 |
| router_oracle | oracle | 0.8170 | 0.6520 | 0.4730 | 0.5972 | 223.16 |

Interpretation:

The larger Qwen three-billion run preserves the main fixed-ACE result. `ace_packed_280` gives the strongest deployable answer quality while using a similar context budget to packed chunk retrieval. Focused ACE is still more efficient than packed ACE, but at this larger sample size it is better framed as a lower-budget tradeoff rather than the best-quality variant. The router remains below fixed packed ACE, while the oracle continues to show substantial routing headroom.

## HotpotQA Qwen 3B Limit-1000 Partial Aggregate

Kaggle Stage-3 HotpotQA validation with `Qwen/Qwen2.5-3B-Instruct`, limit 1000, seeds 42 and 7:

| Policy | Mean Qwen EM | Mean Qwen F1 | Mean Reader Tokens | Mean Recall@5 | Mean All-Gold@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_280 | 0.3185 | 0.4218 | 269.49 | 0.8585 | 0.7310 |
| bm25_packed_280 | 0.2675 | 0.3631 | 268.88 | 0.6380 | 0.3670 |
| hybrid_packed_280 | 0.2955 | 0.3956 | 268.27 | 0.7917 | 0.6050 |
| ace_packed_280 | 0.3380 | 0.4390 | 267.32 | 0.8180 | 0.6605 |
| ace_focused_220 | 0.3245 | 0.4225 | 208.53 | 0.8180 | 0.6605 |
| router_rule | 0.3190 | 0.4237 | 236.31 | 0.8250 | 0.6665 |
| router_oracle | 0.4870 | 0.6041 | 222.39 | 0.8155 | 0.6520 |

Interpretation:

The larger Qwen three-billion result has replicated across the first two larger samples. `ace_packed_280` remains the best deployable method by answer quality. The result is especially useful because chunk retrieval still has stronger raw gold-document coverage, yet lower answer quality. This reinforces the paper's distinction between evidence coverage and reader-useful evidence organization.

## Stage-3 Aggregate Source Of Truth

Multi-seed Stage-3 tables should be generated with:

```bash
python scripts/summarize_stage3_results.py <kaggle_results_dir> --out-dir results/stage3_summary_current
```

The summarizer de-duplicates cumulative Kaggle result publishes before computing means and deltas. This avoids over-counting earlier runs that are copied into later `colab_results` folders.

## HotpotQA Qwen 3B Limit-1000 Three-Seed Aggregate

Kaggle Stage-3 HotpotQA validation with `Qwen/Qwen2.5-3B-Instruct`, limit 1000, seeds 42, 7, and 13. These values are generated by the de-duplicated Stage-3 summarizer.

| Policy | Mean Qwen EM | Mean Qwen F1 | Mean Reader Tokens |
| --- | ---: | ---: | ---: |
| chunk_packed_280 | 0.3217 | 0.4232 | 268.99 |
| bm25_packed_280 | 0.2687 | 0.3637 | 268.68 |
| hybrid_packed_280 | 0.2987 | 0.3988 | 268.22 |
| ace_packed_280 | 0.3367 | 0.4358 | 267.30 |
| ace_focused_220 | 0.3320 | 0.4302 | 208.50 |
| router_rule | 0.3307 | 0.4347 | 235.92 |
| router_oracle | 0.4843 | 0.6015 | 221.45 |

Interpretation:

The larger-sample stronger-reader result now supports the same central HotpotQA claim. Packed ACE is the best fixed deployable policy, improving answer quality over packed chunk at essentially the same context budget. Focused ACE gives a strong lower-budget variant. The rule router is now competitive on this larger aggregate, but it still should be framed carefully because its cross-dataset behavior is weaker than the fixed ACE result. The oracle gap remains large, which supports routing as a future direction rather than the main deployed contribution.

## HotpotQA Qwen 3B Budget Curve

Kaggle Stage-3 HotpotQA validation with `Qwen/Qwen2.5-3B-Instruct`, limit 1000. The table below reports the three-seed average for the tighter context budgets.

| Budget Group | Policy | Qwen EM | Qwen F1 | Avg Reader Tokens |
| --- | --- | ---: | ---: | ---: |
| 160 | chunk_packed_160 | 0.2947 | 0.3837 | 150.97 |
| 160 | ace_packed_160 | 0.3160 | 0.4095 | 150.46 |
| 160 | ace_focused_160 | 0.3190 | 0.4144 | 149.62 |
| 220 | chunk_packed_220 | 0.3083 | 0.4058 | 210.52 |
| 220 | ace_packed_220 | 0.3273 | 0.4234 | 209.66 |
| 220 | ace_focused_220 | 0.3320 | 0.4302 | 208.50 |

Interpretation:

The budget curve strengthens the core compression argument. When the reader context budget is tighter, ACE improves answer quality over packed chunk at the same approximate token budget. This now holds across three seeds for both tight-budget settings. The advantage is larger than the standard 280-token comparison, which supports the claim that graph-compressed evidence is most useful under context pressure.

Question-level comparison supports the same pattern most clearly at the tightest budget. This gives the paper a sharper claim: ACE is not just another retrieval variant; it is most useful when the system must fit evidence into a restricted reader context.

## HotpotQA Qwen 3B Extreme Budget Boundary

Kaggle Stage-3 HotpotQA validation with `Qwen/Qwen2.5-3B-Instruct`, limit 1000. This is a replicated stress check over three seeds, not the recommended operating point.

| Budget Group | Policy | Qwen EM | Qwen F1 | Avg Reader Tokens |
| --- | --- | ---: | ---: | ---: |
| 80 | chunk_packed_80 | 0.2333 | 0.3054 | 71.81 |
| 80 | ace_packed_80 | 0.2583 | 0.3404 | 71.70 |
| 80 | ace_focused_80 | 0.2637 | 0.3449 | 71.06 |
| 120 | chunk_packed_120 | 0.2683 | 0.3503 | 111.04 |
| 120 | ace_packed_120 | 0.2880 | 0.3747 | 110.73 |
| 120 | ace_focused_120 | 0.3000 | 0.3883 | 110.05 |

Interpretation:

The boundary check shows that ACE keeps a relative advantage even under severe context pressure, but absolute answer quality drops sharply at very small budgets. This is useful for the paper because it identifies a practical operating range: tight budgets make ACE more valuable, but pushing the context too low still harms the reader. The best paper-facing operating range is therefore the compact budget range, while the extreme-budget table should be used as stress analysis.

## Paired Bootstrap Check

Paired bootstrap checks on the latest HotpotQA Qwen 3B seed compare ACE against chunk on the same questions.

| Setting | Left Policy | Right Policy | Delta F1 | 95% CI |
| --- | --- | --- | ---: | --- |
| budget 80 | ace_focused_80 | chunk_packed_80 | 0.0378 | [0.0130, 0.0637] |
| budget 120 | ace_focused_120 | chunk_packed_120 | 0.0396 | [0.0156, 0.0629] |
| budget 160 | ace_focused_160 | chunk_packed_160 | 0.0398 | [0.0162, 0.0660] |
| budget 220 | ace_focused_220 | chunk_packed_220 | 0.0437 | [0.0212, 0.0677] |
| standard 280 | ace_packed_280 | chunk_packed_280 | 0.0034 | [-0.0218, 0.0299] |

Interpretation:

The paired bootstrap reinforces the budget-pressure finding. On the latest seed, the compact-budget ACE gains are clearly positive, while the standard 280-token comparison is weaker and not separated from zero on that seed. This supports framing the paper around budgeted evidence compression rather than claiming a universal retriever improvement at every context length.

## RAGBench PubMedQA Cross-Dataset Check

Kaggle cross-dataset validation with `Qwen/Qwen2.5-3B-Instruct`, RAGBench PubMedQA, limit 200, budget 160:

| Policy | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.8860 | 0.5550 | 0.0000 | 0.0104 | 147.34 |
| ace_packed_160 | 0.7980 | 0.3450 | 0.0000 | 0.0127 | 144.69 |
| ace_focused_160 | 0.7980 | 0.3450 | 0.0000 | 0.0139 | 144.16 |
| router_rule | 0.6560 | 0.0650 | 0.0000 | 0.0098 | 146.50 |
| router_oracle | 0.6360 | 0.0200 | 0.0000 | 0.0179 | 139.57 |

Interpretation:

The raw answer metrics are not reliable for this PubMedQA slice because the stored gold answers are long explanatory paragraphs, while the Qwen reader usually emits short yes/no/unknown decisions. This makes exact match and token F1 artificially low for all methods.

Secondary label-style analysis on the scorable subset:

| Policy | Label Accuracy | Label Macro F1 | Delta Macro F1 vs Chunk |
| --- | ---: | ---: | ---: |
| chunk_packed_160 | 0.1917 | 0.1458 | 0.0000 |
| ace_packed_160 | 0.2295 | 0.1702 | 0.0244 |
| ace_focused_160 | 0.2101 | 0.1576 | 0.0118 |
| router_rule | 0.1967 | 0.1505 | 0.0047 |
| router_oracle | 0.2821 | 0.2034 | 0.0576 |

This is tentative cross-dataset support rather than a headline result. It suggests compact ACE can help even when raw retrieval coverage is lower, but the gold-label extraction is heuristic because the original run did not preserve PubMedQA labels. The next cross-dataset run should use another RAGBench subset, and a future PubMedQA rerun should preserve the original yes/no/maybe label if the source dataset exposes it.

## RAGBench CovidQA Cross-Dataset Check

Kaggle cross-dataset validation with `Qwen/Qwen2.5-3B-Instruct`, RAGBench CovidQA, limit 200, budget 160:

| Policy | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.7462 | 0.4050 | 0.0050 | 0.1100 | 146.32 |
| ace_packed_160 | 0.6250 | 0.2400 | 0.0050 | 0.1390 | 148.36 |
| ace_focused_160 | 0.6250 | 0.2400 | 0.0050 | 0.1469 | 147.90 |
| router_rule | 0.4375 | 0.0250 | 0.0050 | 0.1286 | 146.79 |
| router_oracle | 0.4600 | 0.0300 | 0.0050 | 0.1857 | 144.28 |

Paired bootstrap against packed chunk:

| Comparison | Delta Qwen F1 | 95% CI |
| --- | ---: | --- |
| ace_packed_160 vs chunk_packed_160 | 0.0291 | [0.0111, 0.0485] |
| ace_focused_160 vs chunk_packed_160 | 0.0370 | [0.0164, 0.0588] |

Interpretation:

CovidQA gives the cleanest cross-dataset support so far. Chunk retrieval still has higher raw gold-document coverage, but compact ACE produces better reader answers under the same tight context budget, and the paired bootstrap is positive. This matches the HotpotQA budget-pressure result and strengthens the claim that graph-compressed evidence can be more reader-useful than flat chunk evidence when context is limited.

## RAGBench ExpertQA Cross-Dataset Check

Kaggle cross-dataset validation with `Qwen/Qwen2.5-3B-Instruct`, RAGBench ExpertQA, limit 200, budget 160:

| Policy | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.7860 | 0.5800 | 0.0000 | 0.0244 | 152.98 |
| ace_packed_160 | 0.7256 | 0.4900 | 0.0000 | 0.0347 | 154.81 |
| ace_focused_160 | 0.7256 | 0.4900 | 0.0000 | 0.0331 | 154.90 |
| router_rule | 0.5083 | 0.1700 | 0.0000 | 0.0321 | 153.99 |
| router_oracle | 0.5148 | 0.1650 | 0.0000 | 0.0517 | 152.69 |

Paired bootstrap against packed chunk:

| Comparison | Delta Qwen F1 | 95% CI |
| --- | ---: | --- |
| ace_packed_160 vs chunk_packed_160 | 0.0104 | [0.0027, 0.0187] |
| ace_focused_160 vs chunk_packed_160 | 0.0088 | [0.0012, 0.0173] |

Interpretation:

ExpertQA gives a smaller but still positive cross-dataset result. Again, chunk retrieval has stronger raw evidence coverage, but ACE gives better reader answers under the same tight context budget. The absolute F1 values are low because ExpertQA answers are long and open-ended, but the paired comparison is still useful because all policies are judged on the same questions and answer format.

## RAGBench EManual Cross-Dataset Check

Kaggle cross-dataset validation with `Qwen/Qwen2.5-3B-Instruct`, RAGBench EManual, budget 160:

| Policy | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.5556 | 0.1970 | 0.0000 | 0.0675 | 138.61 |
| ace_packed_160 | 0.5152 | 0.1364 | 0.0000 | 0.0761 | 151.94 |
| ace_focused_160 | 0.5152 | 0.1364 | 0.0000 | 0.0702 | 152.64 |
| router_rule | 0.4242 | 0.0909 | 0.0000 | 0.0716 | 143.47 |
| router_oracle | 0.4470 | 0.0909 | 0.0000 | 0.1090 | 149.82 |

Paired bootstrap against packed chunk after de-duplicating repeated question IDs:

| Comparison | Delta Qwen F1 | 95% CI |
| --- | ---: | --- |
| ace_packed_160 vs chunk_packed_160 | 0.0087 | [-0.0092, 0.0265] |
| ace_focused_160 vs chunk_packed_160 | 0.0027 | [-0.0153, 0.0197] |

Interpretation:

EManual is directionally positive but not statistically clean. The aggregate table shows ACE packed above packed chunk, but the saved RAGBench examples contain repeated question IDs, so the paired bootstrap operates on fewer unique questions and the interval crosses zero. This should be reported as supporting trend evidence rather than a decisive transfer result.

## RAGBench TechQA Cross-Dataset Check

Kaggle cross-dataset validation with `Qwen/Qwen2.5-3B-Instruct`, RAGBench TechQA, budget 160:

| Policy | Recall@5 | All-Gold@5 | Qwen EM | Qwen F1 | Avg Reader Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.4340 | 0.0200 | 0.0000 | 0.0155 | 157.72 |
| ace_packed_160 | 0.4420 | 0.0150 | 0.0000 | 0.0125 | 153.03 |
| ace_focused_160 | 0.4420 | 0.0150 | 0.0000 | 0.0194 | 153.87 |
| router_rule | 0.2820 | 0.0000 | 0.0000 | 0.0135 | 154.54 |
| router_oracle | 0.2700 | 0.0000 | 0.0000 | 0.0292 | 149.32 |

Paired bootstrap against packed chunk:

| Comparison | Delta Qwen F1 | 95% CI |
| --- | ---: | --- |
| ace_focused_160 vs chunk_packed_160 | 0.0023 | [-0.0056, 0.0090] |
| ace_packed_160 vs chunk_packed_160 | -0.0035 | [-0.0119, 0.0033] |

Interpretation:

TechQA is mixed. Focused ACE is slightly above packed chunk in aggregate, but the paired interval crosses zero, while packed ACE is slightly worse than chunk. This weakens any broad claim that ACE always transfers, but it does not contradict the tighter claim: ACE helps on several budgeted settings, while domain and answer format still matter.

## Stage-4: Principled Submodular Packing (HotpotQA-500, Qwen2.5-3B)

The packing-objective experiment. A clean 2×2 of {chunk, ACE} × {focused heuristic, submodular} at a shared 160-token budget, plus the chunk_packed anchor and an oracle over the packers. Seed 42, 500 questions.

| Policy | Recall@5 | EM | F1 | ans_in_ctx | gold_tok_density | reader_gold_cov | Avg Reader Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.874 | 0.316 | 0.4007 | 0.590 | 0.400 | 0.685 | 151.3 |
| chunk_focused_160 | 0.874 | 0.322 | 0.4124 | 0.632 | 0.509 | 0.733 | 152.3 |
| **chunk_submod_160** | 0.874 | **0.370** | **0.4477** | 0.640 | 0.457 | **0.779** | 145.4 |
| ace_focused_160 | 0.859 | 0.320 | 0.4211 | 0.630 | 0.452 | 0.768 | 150.0 |
| ace_submod_160 | 0.859 | 0.308 | 0.4007 | 0.612 | 0.428 | 0.758 | 147.4 |
| router_oracle_packer | — | **0.460** | **0.5702** | 0.684 | 0.491 | 0.788 | 141.9 |

Paired bootstrap (10k resamples, same 500 questions):

| Comparison | Delta EM | 95% CI | Delta F1 | 95% CI |
| --- | ---: | --- | ---: | --- |
| chunk_submod vs chunk_focused | +0.048 | [+0.014, +0.082] | +0.035 | [+0.002, +0.069] |
| chunk_submod vs chunk_packed | +0.054 | [+0.016, +0.090] | +0.047 | [+0.011, +0.083] |
| chunk_submod vs ace_focused | +0.050 | [+0.016, +0.084] | +0.027 | [-0.008, +0.062] |
| ace_submod vs ace_focused | -0.012 | [-0.044, +0.020] | -0.020 | [-0.053, +0.011] |

Mediation (pooled over all 2,500 policy-question rows), correlation with F1:

| Feature | corr with F1 | corr with EM |
| --- | ---: | ---: |
| ans_in_context | **+0.50** | **+0.46** |
| gold_doc_reader_cov | +0.33 | +0.29 |
| base_all_gold@5 (retrieval) | +0.32 | +0.27 |
| base_recall@5 (retrieval) | +0.31 | +0.27 |
| gold_token_density | +0.26 | +0.23 |

Conditional F1: 0.596 when a gold answer is present in the reader context vs 0.123 when it is not (gap +0.47).

Interpretation:

1. The budgeted-submodular packer is the first statistically clean method win of the project: it beats both heuristic packers on EM (p<0.01) at equal-or-lower token cost.
2. `ans_in_context` predicts answer quality better than document recall, explaining the recall paradox: under a budget, what matters is whether the answer survives into the reader context.
3. The packer helps chunk evidence, not ACE (ace_submod is not better than ace_focused). ACE already compresses evidence at the graph level, leaving less redundancy for submodular selection to exploit. This reframes the contribution around the packing objective rather than the graph representation.

Robustness (MMR baseline + seeds {13, 7}) is queued as Stage-4b.

## Stage-4b: Multi-Seed Confirmation + MMR Baseline (HotpotQA-500, Qwen2.5-3B)

Full 2x3 factorial {chunk, ACE} x {focused, mmr, submod} + chunk_packed anchor + oracle, seeds {42, 13, 7}, budget 160. Cross-seed mean:

| Policy | mean F1 | mean EM | F1 by seed (42/13/7) |
| --- | ---: | ---: | --- |
| chunk_packed | 0.400 | 0.306 | .401 / .407 / .393 |
| chunk_focused | 0.429 | 0.331 | .412 / .431 / .444 |
| chunk_mmr | 0.410 | 0.318 | .390 / .425 / .414 |
| **chunk_submod** | **0.451** | **0.359** | .448 / .455 / .451 |
| ace_focused | 0.428 | 0.328 | .421 / .432 / .429 |
| ace_mmr | 0.405 | 0.311 | .408 / .399 / .408 |
| ace_submod | 0.406 | 0.317 | .401 / .399 / .420 |
| oracle (mixed) | 0.601 | 0.487 | .598 / .609 / .596 |

Pooled multi-seed paired bootstrap (1,500 (seed,qid) instances, 10k resamples):

| Contrast | Delta F1 [95% CI] | Delta EM [95% CI] |
| --- | --- | --- |
| chunk_submod vs chunk_mmr | +0.042 [+0.021, +0.063] | +0.046 [+0.025, +0.067] |
| chunk_submod vs chunk_focused | +0.022 [+0.002, +0.041] | +0.029 [+0.009, +0.048] |
| chunk_submod vs chunk_packed | +0.051 [+0.030, +0.072] | +0.053 [+0.031, +0.075] |
| chunk_mmr vs chunk_focused | -0.020 [-0.034, -0.005] | -0.017 [-0.031, -0.003] |
| ace_submod vs ace_focused | -0.021 [-0.039, -0.003] | -0.011 [-0.029, +0.007] |
| chunk_submod vs ace_submod | +0.045 [+0.026, +0.063] | +0.043 [+0.024, +0.061] |

Interpretation: chunk_submod is the best fixed policy on all three seeds; the packer ordering is **submod > focused > packed > mmr**. Plain MMR is significantly worse than the focused heuristic, so the gain is the submodular objective, not generic redundancy reduction. Submod significantly hurts ACE, and under submod chunk beats ACE — graph compression and principled packing are partial substitutes. The oracle sits ~0.15 F1 above the best fixed policy on every seed.

## Stage-5: Cross-Dataset Transfer — Negative / Scope Boundary (RAGBench, Qwen2.5-3B)

Full 2x3 factorial at budget 160 on RAGBench CovidQA (n=246) and ExpertQA (n=203), split=test. EM is ~0 on both (long free-form gold answers), so F1 only:

| Policy | CovidQA F1 | ExpertQA F1 |
| --- | ---: | ---: |
| chunk_packed | 0.106 | 0.023 |
| chunk_focused | **0.126** | 0.030 |
| chunk_mmr | 0.114 | 0.026 |
| chunk_submod | 0.116 | **0.036** |
| ace_focused | 0.134 | 0.034 |
| ace_mmr | **0.134** | 0.035 |
| ace_submod | 0.133 | 0.035 |
| oracle (mixed) | 0.214 | 0.069 |

Paired bootstrap, chunk_submod vs chunk_focused: CovidQA -0.010 F1 [95% CI -0.029, +0.009], p=0.30; ExpertQA +0.005 F1 [-0.002, +0.014], p=0.15. Neither significant. On ExpertQA chunk_submod does beat chunk_packed (+0.013, p=0.002) and chunk_mmr (+0.010, p=0.009), but not chunk_focused.

Interpretation: the clean HotpotQA result `submod > focused` does **not** transfer to RAGBench single-pass QA. On CovidQA the focused heuristic is nominally best among chunk packers and ACE beats chunk (the old representation story). This is consistent with the mechanism — submod's gain comes from complementary multi-hop coverage amid distractors, which RAGBench's single-document, all-context-gold structure lacks — so the honest claim is conditional: principled submodular packing helps specifically for budget-constrained, distractor-heavy, multi-hop QA. (ans_in_context still correlates with F1 at 0.39 on CovidQA; it is degenerate on ExpertQA where long answers never appear verbatim.) Stage-6 tests the budget-pressure and harder-multi-hop predictions of this scoped claim.

## Stage-6: HotpotQA Budget Curve (seed 42, Qwen2.5-3B)

chunk_submod − chunk_focused by reader-token budget (paired bootstrap, 500 q; 160 from Stage-4b):

| budget | submod F1 | focused F1 | ΔF1 | p | ΔEM | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.373 | 0.369 | +0.004 | 0.81 | -0.002 | 0.96 |
| 128 | 0.425 | 0.411 | +0.014 | 0.45 | +0.020 | 0.30 |
| 160 | 0.448 | 0.412 | **+0.035** | **0.04** | **+0.048** | **0.01** |
| 224 | 0.460 | 0.443 | +0.017 | 0.26 | +0.018 | 0.29 |

vs naive packed (all significant): ΔF1 +0.044/+0.051/+0.047/+0.055 at 96/128/160/224 (p=0.022/0.009/0.009/0.002). submod@160 vs focused@224: ΔF1 +0.005 (p=0.80), ΔEM +0.018 (p=0.33) — equal quality at ~30% fewer tokens (145 vs 215).

Interpretation: the submod-vs-focused advantage is an **inverted-U peaking at ~160**, not monotonic in budget (prediction falsified). At 96 only 2-3 snippets fit (no room for complementarity); at 224 everything fits (heuristic catches up). But submod beats naive packing at every budget, and is token-efficient (matches the heuristic's best quality at 30% lower cost). Single-seed for 96/128/224.

## Stage-7: MuSiQue Harder-Multi-Hop (limit 500, Qwen2.5-3B)

Full 2x3 factorial at budgets 160 and 96. Budget 160:

| Policy | F1 | EM | ans_in_ctx |
| --- | ---: | ---: | ---: |
| chunk_packed | 0.137 | 0.088 | 0.224 |
| chunk_focused | 0.124 | 0.070 | 0.216 |
| chunk_mmr | 0.123 | 0.064 | 0.192 |
| chunk_submod | 0.134 | 0.080 | 0.178 |
| ace_focused | **0.150** | 0.084 | 0.216 |
| ace_mmr | 0.148 | 0.084 | 0.208 |
| ace_submod | 0.141 | 0.082 | 0.194 |
| oracle (mixed) | 0.251 | 0.162 | 0.252 |

Retrieval: recall@5=0.506, **all_gold@5=0.184**. Paired bootstrap (budget 160): chunk_submod − chunk_focused +0.011 F1 (p=0.34), − chunk_packed −0.003 (p=0.80), − chunk_mmr +0.011 (p=0.31); ace_focused − chunk_packed +0.013 (p=0.36). None significant. Budget 96 same null pattern.

Interpretation: the prediction that a harder multi-hop set would *widen* the submod gap is **falsified** — the gap is not significant on MuSiQue and naive packing is just as good. The cause is retrieval: only ~18% of questions have all gold paragraphs retrieved (all_gold@5=0.18), so the packer cannot assemble complementary evidence that was never surfaced; MuSiQue is retrieval-bottlenecked. Crucially the diagnostic still holds: **corr(ans_in_context, F1) = 0.54** (highest of any dataset), so answer-in-context is a genuine dataset-independent mediator, not an artifact of the packer. Net: principled packing helps only where multi-hop structure, sufficient retrieval, and binding-but-not-extreme budget co-occur (HotpotQA); the diagnostic generalizes everywhere.

## Stage-8: MuSiQue Retrieval-Unlock Ablation

**Kaggle commit:** `09bab7a` (2026-06-28). Top-k 12, top-k-nodes 64, max-expanded-docs 8 (vs Stage-7: 5/48/5). Budgets {160, 240}, limit 500.

**Retrieval coverage (both budgets, unchanged from Stage-7):**

| Metric | Stage-7 | Stage-8 |
| --- | ---: | ---: |
| chunk recall@5 | 0.506 | 0.506 |
| **chunk all_gold@5** | **0.184** | **0.184** |
| ace recall@5 | 0.481 | 0.481 |
| **ace all_gold@5** | **0.142** | **0.142** |

Budget 160:

| Policy | F1 | EM | ans_in_ctx | gold_doc_cov | Avg tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.1101 | 0.064 | 0.184 | 0.248 | 151.5 |
| chunk_focused_160 | 0.1244 | 0.074 | 0.190 | 0.340 | 151.8 |
| chunk_mmr_160 | 0.1258 | 0.078 | 0.188 | 0.296 | 152.2 |
| chunk_submod_160 | 0.1275 | 0.076 | 0.168 | 0.363 | 148.1 |
| **ace_focused_160** | **0.1419** | **0.086** | **0.214** | **0.381** | 150.5 |
| ace_mmr_160 | 0.1194 | 0.068 | 0.188 | 0.343 | 150.4 |
| ace_submod_160 | 0.1321 | 0.082 | 0.186 | 0.350 | 147.7 |
| oracle (mixed) | 0.2584 | 0.170 | 0.250 | 0.367 | 142.1 |

Budget 240:

| Policy | F1 | EM | ans_in_ctx | gold_doc_cov | Avg tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_240 | 0.1398 | 0.082 | 0.238 | 0.337 | 231.7 |
| chunk_focused_240 | 0.1502 | 0.086 | 0.250 | 0.381 | 231.9 |
| chunk_mmr_240 | 0.1428 | 0.082 | 0.230 | 0.336 | 231.7 |
| chunk_submod_240 | 0.1434 | 0.086 | 0.222 | 0.406 | 221.8 |
| **ace_focused_240** | **0.1580** | **0.098** | **0.260** | **0.432** | 227.9 |
| ace_mmr_240 | 0.1472 | 0.090 | 0.252 | 0.392 | 228.7 |
| ace_submod_240 | 0.1442 | 0.084 | 0.246 | 0.391 | 223.5 |
| oracle (mixed) | 0.2924 | 0.190 | 0.304 | 0.401 | 216.1 |

Key packer gap (descriptive): chunk_submod − chunk_focused = **+0.003 F1** (b160), **−0.007 F1** (b240). Both smaller than Stage-7's already-null +0.011. No packer advantage at any budget.

**Verdict:** The MuSiQue retrieval bottleneck is qualitative, not a depth problem. Tripling retrieval breadth moved all_gold@5 by zero. The embedding model (BAAI/bge-small-en-v1.5) cannot navigate 2–4 hop compositional chains regardless of candidate pool size. This is a precisely falsifiable scope statement: richer retrieval was tried, it failed, so the condition for the packer win is "retrieval that qualitatively surfaces multi-hop gold evidence" — not achievable in this pipeline on MuSiQue. ace_focused is the best fixed policy on MuSiQue at both budgets and across both Stages 7 and 8.

---

## Stage-9 — HotpotQA larger-reader scaling (Run A): does the packer win survive a 7B reader?

Exact Stage-4b HotpotQA-500 2×3 factorial (top-k 5 / nodes 48 / expand 5, budget 160), reader swapped 3B → **Qwen2.5-7B-Instruct** (`device_map="auto"` across 2×T4), seeds {42, 13}. Tests the robustness objection "a stronger reader absorbs the redundancy the packer removes."

Pooled 2-seed paired bootstrap (10k, n=1000):

| Contrast | F1 Δ | 95% CI | p | EM Δ | p | vs 3B (Stage-4b) |
| --- | ---: | :---: | ---: | ---: | ---: | --- |
| chunk_submod − chunk_focused | −0.010 | [−0.035, +0.015] | **0.45** | −0.008 | 0.54 | was +4.8–5.4 EM, p<0.01 → **VANISHED** |
| chunk_submod − chunk_packed | **+0.054** | [+0.027, +0.081] | **<0.001** | +0.044 | 0.001 | **survives at 7B** |
| chunk_mmr − chunk_focused | −0.032 | [−0.051, −0.014] | 0.001 | −0.025 | 0.006 | replicates (mmr worse) |
| ace_submod − ace_focused | −0.020 | [−0.043, +0.003] | 0.088 | −0.026 | 0.023 | replicates (submod hurts ace) |

Budget 160, pooled means across seeds {42,13}:

| Policy | F1 | EM | ans_in_ctx | gold_doc_cov | all_gold_reader |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.332 | 0.259 | 0.583 | 0.685 | 0.447 |
| chunk_focused_160 | 0.396 | 0.311 | 0.638 | 0.741 | 0.526 |
| chunk_mmr_160 | 0.363 | 0.286 | 0.597 | 0.680 | 0.420 |
| chunk_submod_160 | 0.386 | 0.303 | 0.634 | **0.779** | **0.582** |
| ace_focused_160 | 0.391 | 0.303 | 0.624 | 0.760 | 0.553 |
| ace_mmr_160 | 0.366 | 0.290 | 0.600 | 0.719 | 0.487 |
| ace_submod_160 | 0.371 | 0.277 | 0.610 | 0.750 | 0.541 |
| oracle (mixed) | 0.574 | 0.461 | 0.704 | 0.780 | 0.588 |

Per-seed best fixed policy flips: **seed 42 → chunk_submod (F1 0.396)**, **seed 13 → chunk_focused (F1 0.407)**.

**Verdict:** The submod-vs-best-heuristic win **does not survive a 7B reader** (null, seed sign-flip). But it is a *scope boundary, not a refutation*: submod still packs the densest gold evidence at 7B (gold_doc_cov 0.779, all_gold_reader 0.582 — highest of any fixed policy, both seeds), and it still significantly beats *naive* packing (+0.054 F1, p<0.001). The 7B reader absorbs the *small* density edge over the focused heuristic but not the *large* edge over naive packing. **Reader capability mediates the benefit:** the answer-in-context diagnostic explains the 3B win and the 7B null as the same density advantage passed through readers of different sensitivity. Reader size is the third scope axis (with multi-hop structure and surfacing retrieval): the reader must be the bottleneck for packing to convert density into accuracy.

**Paper inclusion: OPEN (user decision).** Recommended (B): include as a reader-scale scope boundary, reframe the headline as reader-conditional, keep submod−packed as the positive-at-scale result, promote the diagnostic to the unifying explanation. Alternative (A): omit, one limitations sentence.

**Empirical arc CLOSED.**

---

## Stage-10b — 2WikiMultiHopQA full factorial (Run B), Qwen2.5-3B, seeds {42,13,7}

Retrieval gate (Stage-10a): chunk recall@5 = 0.718, **all-gold@5 = 0.43** (clears the ≥0.40 bar; MuSiQue was 0.18). Budget 160, pooled means across seeds {42,13,7}:

| Policy | F1 | EM | ans_in_ctx | gold_doc_cov |
| --- | ---: | ---: | ---: | ---: |
| chunk_packed_160 | 0.250 | 0.229 | 0.348 | 0.569 |
| chunk_focused_160 | 0.273 | 0.249 | 0.399 | 0.593 |
| chunk_mmr_160 | 0.250 | 0.226 | 0.365 | 0.547 |
| chunk_submod_160 | 0.266 | 0.243 | 0.392 | **0.647** |
| ace_focused_160 | **0.275** | **0.251** | **0.427** | 0.659 |
| ace_mmr_160 | 0.274 | 0.247 | 0.422 | 0.640 |
| ace_submod_160 | 0.273 | 0.249 | 0.411 | 0.650 |
| oracle (mixed) | 0.438 | 0.391 | 0.440 | 0.637 |

Pooled 3-seed paired bootstrap (n=1500): `submod − focused` −0.008 F1 [−0.027,+0.012] **p=0.44**; `submod − packed` +0.016 [−0.004,+0.036] p=0.12; `mmr − focused` −0.024 [−0.038,−0.010] p=0.001. Per-seed best fixed policy: ace_mmr / chunk_submod / ace_submod.

**Verdict:** A **clean null for the packer** on a second multi-hop dataset that *clears* the retrieval gate. No significant advantage of submod over any chunk heuristic; even `submod − packed` (significant on HotpotQA at both reader scales) is only directional (p=0.12). The informative part is a **within-method dissociation**: `chunk_submod` packs +0.054 more gold-doc coverage than `chunk_focused` but −0.007 answer-in-context and −0.008 F1 — coverage and answer-in-context move in opposite directions and accuracy follows answer-in-context. On 2Wiki the extra gold the packer assembles is bridging evidence that does not contain the answer span (compositional answers). The diagnostic is **strongest here** (corr 0.55; conditional F1 0.56 vs 0.08) and correctly predicts the packer's own null — clean *interventional* support for the §3.3 incremental-validity result. This adds a **5th scope condition** (the assembled gold must contain the answer) and makes HotpotQA-3B the lone packer win across the five multi-hop settings tested.

**Paper inclusion: DECIDED — included** as §6.6 (Condition 5) / Appendix D, framed as diagnostic mediation. The diagnostic-led narrative absorbs the null as a strength; the packer's positive claims stay HotpotQA-scoped. *(Superseded by the 2026-06-29 reframe: 2Wiki demoted to a single interventional paragraph in §3.4; scope map reverts to four conditions. Finding unchanged.)*

## Stage-11 — HotpotQA reader-scale ladder (7B-4bit bridge + 14B-4bit), commit e126e4c

Exact §5 HotpotQA-500 factorial, seeds {42,13}, only the reader changes. Complete (4 rungs × all metrics; recovered from pushed commit `e126e4c` after the GPU account expired).

`chunk_submod − chunk_focused` (packer edge over the best heuristic), pooled 2-seed paired bootstrap (n=1000):

| Reader | ΔF1 | 95% CI | p | best fixed policy |
|---|---|---|---|---|
| 3B (§5) | +0.035 | [+0.001, +0.069] | **0.04** | chunk_submod |
| 7B fp16 (Stage-9) | −0.010 | [−0.035, +0.015] | 0.45 | chunk_focused |
| 7B 4bit (bridge) | −0.008 | [−0.032, +0.017] | 0.55 | chunk_focused |
| 14B 4bit | −0.029 | [−0.052, −0.006] | **0.013** | chunk_focused / ace_focused |

**Verdict:** the packer's edge over the smart focused heuristic is **monotone-decreasing in reader scale and reverses by 14B** (+0.035 sig+ → null at 7B → −0.029 sig− at 14B). (1) **Quantization ruled out:** 7B-4bit ≈ 7B-fp16 (both null, ≈−0.01, same best policy) → the 14B reversal is a scale effect, not a 4-bit artifact. (2) **`submod − packed` stays significantly positive at every rung** (+0.054/+0.054/+0.055/+0.044, p≤0.001) — the submodular objective always beats naive packing; only the edge over the *smart* heuristic erodes. Mechanism: at 14B, submod still packs more gold + equal answer-in-context but more distractors; the capable reader extracts fine from focused's cleaner pack, so curation's overhead exceeds its benefit. Condition 4 (reader must be the bottleneck) now pinned with a threshold (~7B crossover) and a reversal.

**Empirical arc CLOSED** (reader ladder = depth on condition 4; 32B optional confirmatory rung pending credit decision).
