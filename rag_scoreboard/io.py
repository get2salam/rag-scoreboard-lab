"""File loading and rendering helpers for the RAG scoreboard CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import EvaluationResult, Qrels, Run


def load_qrels(path: Path) -> Qrels:
    """Load qrels JSON mapping query ids to relevant document ids."""

    data = _load_json_object(path)
    qrels: dict[str, set[str]] = {}
    for query_id, document_ids in data.items():
        if not isinstance(document_ids, list) or not all(
            isinstance(item, str) for item in document_ids
        ):
            raise ValueError(
                f"qrels[{query_id!r}] must be a list of relevant document id strings"
            )
        qrels[str(query_id)] = set(document_ids)
    return qrels


def load_run(path: Path) -> Run:
    """Load run JSON mapping query ids to ranked document ids."""

    data = _load_json_object(path)
    run: dict[str, list[str]] = {}
    for query_id, document_ids in data.items():
        if not isinstance(document_ids, list) or not all(
            isinstance(item, str) for item in document_ids
        ):
            raise ValueError(f"run[{query_id!r}] must be a ranked list of document ids")
        run[str(query_id)] = document_ids
    return run


def render_markdown_table(result: EvaluationResult) -> str:
    """Render per-query scores and macro averages as a compact table."""

    lines = [
        f"# RAG Scoreboard (k={result.cutoff})",
        "",
        "| query | relevant | retrieved@k | hits | precision@k | recall@k | rr@k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score in result.queries:
        lines.append(
            "| {query} | {relevant} | {retrieved} | {hits} | {precision:.3f} | "
            "{recall:.3f} | {rr:.3f} |".format(
                query=score.query_id,
                relevant=score.relevant_total,
                retrieved=score.retrieved,
                hits=score.hits,
                precision=score.precision,
                recall=score.recall,
                rr=score.reciprocal_rank,
            )
        )
    lines.extend(
        [
            "",
            "## Macro averages",
            f"- queries: {result.query_count}",
            f"- precision@{result.cutoff}: {result.mean_precision:.3f}",
            f"- recall@{result.cutoff}: {result.mean_recall:.3f}",
            f"- mrr@{result.cutoff}: {result.mean_reciprocal_rank:.3f}",
            f"- hit_rate@{result.cutoff}: {result.hit_rate:.3f}",
        ]
    )
    return "\n".join(lines)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data
