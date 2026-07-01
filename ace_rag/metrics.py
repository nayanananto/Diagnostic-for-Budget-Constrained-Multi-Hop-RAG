"""Experiment metrics."""

from __future__ import annotations

from collections import Counter

from .schema import Question, RetrievalRun
from .text import normalize_answer, tokenize


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(answer) for answer in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = tokenize(normalize_answer(prediction))
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = tokenize(normalize_answer(answer))
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def retrieval_recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    return len(set(retrieved_doc_ids[:k]) & gold_doc_ids) / len(gold_doc_ids)


def all_gold_retrieved_at_k(retrieved_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    return float(gold_doc_ids.issubset(set(retrieved_doc_ids[:k])))


def evaluate_retrieval(runs: list[RetrievalRun], questions: list[Question], k_values: tuple[int, ...] = (1, 2, 5)) -> dict[str, float]:
    by_qid = {q.qid: q for q in questions}
    totals: dict[str, float] = {}
    n = 0
    for run in runs:
        q = by_qid[run.qid]
        for k in k_values:
            totals[f"recall@{k}"] = totals.get(f"recall@{k}", 0.0) + retrieval_recall_at_k(run.retrieved_doc_ids, q.gold_doc_ids, k)
            totals[f"all_gold@{k}"] = totals.get(f"all_gold@{k}", 0.0) + all_gold_retrieved_at_k(run.retrieved_doc_ids, q.gold_doc_ids, k)
        n += 1
    return {key: round(value / max(1, n), 4) for key, value in totals.items()}

