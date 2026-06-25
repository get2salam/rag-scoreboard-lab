"""Metric implementations for retrieval/ranking evaluation.

The module is intentionally dependency-free so scores are reproducible in CI,
local notebooks, and lightweight interview demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


Qrels = Mapping[str, set[str]]
Run = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class QueryScore:
    """Scores for one query at a chosen cutoff."""

    query_id: str
    relevant_total: int
    retrieved: int
    hits: int
    precision: float
    recall: float
    reciprocal_rank: float

    @property
    def hit(self) -> float:
        """Return 1.0 when at least one relevant document was retrieved."""

        return 1.0 if self.hits else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    """Per-query scores plus macro averages."""

    cutoff: int
    queries: tuple[QueryScore, ...]

    @property
    def query_count(self) -> int:
        return len(self.queries)

    @property
    def mean_precision(self) -> float:
        return _mean(score.precision for score in self.queries)

    @property
    def mean_recall(self) -> float:
        return _mean(score.recall for score in self.queries)

    @property
    def mean_reciprocal_rank(self) -> float:
        return _mean(score.reciprocal_rank for score in self.queries)

    @property
    def hit_rate(self) -> float:
        return _mean(score.hit for score in self.queries)


def precision_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """Compute precision@k for one ranked list."""

    if k <= 0:
        raise ValueError("k must be positive")
    top_k = list(retrieved[:k])
    if not top_k:
        return 0.0
    return _hit_count(relevant, top_k) / k


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """Compute recall@k for one ranked list."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return _hit_count(relevant, retrieved[:k]) / len(relevant)


def reciprocal_rank_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """Return reciprocal rank of the first relevant document at cutoff k."""

    if k <= 0:
        raise ValueError("k must be positive")
    for index, document_id in enumerate(retrieved[:k], start=1):
        if document_id in relevant:
            return 1.0 / index
    return 0.0


def evaluate_run(qrels: Qrels, run: Run, k: int) -> EvaluationResult:
    """Evaluate a run against qrels, sorted by query id for stable output.

    Only queries present in qrels are scored. Missing run entries are treated as
    empty rankings, which makes incomplete systems visible instead of crashing.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    query_scores: list[QueryScore] = []
    for query_id in sorted(qrels):
        relevant = qrels[query_id]
        retrieved = list(run.get(query_id, ()))
        top_k = retrieved[:k]
        hits = _hit_count(relevant, top_k)
        query_scores.append(
            QueryScore(
                query_id=query_id,
                relevant_total=len(relevant),
                retrieved=len(top_k),
                hits=hits,
                precision=precision_at_k(relevant, retrieved, k),
                recall=recall_at_k(relevant, retrieved, k),
                reciprocal_rank=reciprocal_rank_at_k(relevant, retrieved, k),
            )
        )
    return EvaluationResult(cutoff=k, queries=tuple(query_scores))


def _hit_count(relevant: set[str], retrieved: Sequence[str]) -> int:
    return sum(1 for document_id in retrieved if document_id in relevant)


def _mean(values) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
