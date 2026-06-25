"""Command-line interface for deterministic RAG retrieval evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_qrels, load_run, render_markdown_table
from .metrics import evaluate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-scoreboard",
        description="Evaluate ranked retrieval results with deterministic RAG metrics.",
    )
    parser.add_argument("--qrels", required=True, type=Path, help="JSON qrels file")
    parser.add_argument("--run", required=True, type=Path, help="JSON ranked run file")
    parser.add_argument(
        "--k", default=3, type=int, help="cutoff for precision, recall, MRR, and hit rate"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        qrels = load_qrels(args.qrels)
        run = load_run(args.run)
        result = evaluate_run(qrels, run, args.k)
    except ValueError as exc:
        parser.error(str(exc))

    print(render_markdown_table(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
