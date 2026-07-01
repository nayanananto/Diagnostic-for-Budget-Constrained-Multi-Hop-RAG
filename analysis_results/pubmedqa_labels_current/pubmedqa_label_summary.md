# PubMedQA Label Analysis

Predictions: `C:\Users\Admin\Desktop\da_project\_tmp_thesis_code_repo\kaggle_results\latest\colab_results\stage3_cross_ragbench_pubmedqa_budget160_limit200_qwen3b\ragbench_stage3_router_standard_limit200_predictions.jsonl`

This is a secondary analysis for RAGBench PubMedQA because the run stores long explanatory gold answers while the reader usually emits short yes/no/unknown decisions.

| Policy | Scorable | Accuracy | Macro F1 | Delta F1 vs Baseline |
| --- | ---: | ---: | ---: | ---: |
| ace_focused_160 | 119/200 | 0.2101 | 0.1576 | 0.0118 |
| ace_packed_160 | 122/200 | 0.2295 | 0.1702 | 0.0244 |
| chunk_packed_160 | 120/200 | 0.1917 | 0.1458 | 0.0 |
| router_oracle | 117/200 | 0.2821 | 0.2034 | 0.0576 |
| router_rule | 122/200 | 0.1967 | 0.1505 | 0.0047 |

Baseline policy: `chunk_packed_160`.

Rows with ambiguous gold explanations are excluded from label accuracy and macro F1.