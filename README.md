# RAG Scoreboard Lab

Minimal retrieval-evaluation scoreboard for ranking search results with deterministic metrics.

## Problem

RAG prototypes often show impressive generated answers while hiding whether retrieval is reliable. This project focuses on the retrieval layer: given a set of relevant document ids (`qrels`) and a ranked search run, it computes stable metrics that make ranking quality easy to compare in CI or local experiments.

## Features

- Dependency-free Python implementation using the standard library.
- Deterministic retrieval metrics:
  - precision@k
  - recall@k
  - mean reciprocal rank (MRR@k)
  - hit rate@k
- JSON fixture data for repeatable demos and tests.
- CLI that prints a Markdown scoreboard suitable for README snippets, pull requests, or experiment notes.
- `unittest` coverage for metrics, missing-result handling, deterministic rendering, and the CLI.
- GitHub Actions workflow for automatic test verification.

## Project structure

```text
rag-scoreboard-lab/
├── rag_scoreboard/
│   ├── __init__.py
│   ├── __main__.py
│   ├── io.py
│   └── metrics.py
├── tests/
│   ├── fixtures/
│   │   ├── qrels.json
│   │   └── run.json
│   └── test_scoreboard.py
├── .github/workflows/test.yml
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quickstart

Requires Python 3.11+.

Run the fixture evaluation:

```bash
python -m rag_scoreboard --qrels tests/fixtures/qrels.json --run tests/fixtures/run.json --k 3
```

Run the tests:

```bash
python -m unittest discover -s tests
```

## Example input

`tests/fixtures/qrels.json` defines the relevant documents for each query:

```json
{
  "q1": ["doc-a", "doc-c"],
  "q2": ["doc-e"],
  "q3": ["doc-h", "doc-i"]
}
```

`tests/fixtures/run.json` defines the retrieved ranking for each query:

```json
{
  "q1": ["doc-b", "doc-a", "doc-c", "doc-d"],
  "q2": ["doc-x", "doc-y", "doc-z"],
  "q3": ["doc-i", "doc-h", "doc-j"]
}
```

## Example output

```text
# RAG Scoreboard (k=3)

| query | relevant | retrieved@k | hits | precision@k | recall@k | rr@k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q1 | 2 | 3 | 2 | 0.667 | 1.000 | 0.500 |
| q2 | 1 | 3 | 0 | 0.000 | 0.000 | 0.000 |
| q3 | 2 | 3 | 2 | 0.667 | 1.000 | 1.000 |

## Macro averages
- queries: 3
- precision@3: 0.444
- recall@3: 0.667
- mrr@3: 0.500
- hit_rate@3: 0.667
```

## Verification command

```bash
python -m unittest discover -s tests
```

Expected result: all tests pass without network access or third-party dependencies.
