import json
import subprocess
import sys
import unittest
from pathlib import Path

from rag_scoreboard.io import load_qrels, load_run, render_markdown_table
from rag_scoreboard.metrics import (
    evaluate_run,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class MetricTests(unittest.TestCase):
    def test_precision_recall_and_rr_at_k(self):
        relevant = {"doc-a", "doc-c"}
        retrieved = ["doc-b", "doc-a", "doc-c", "doc-d"]

        self.assertAlmostEqual(precision_at_k(relevant, retrieved, 3), 2 / 3)
        self.assertAlmostEqual(recall_at_k(relevant, retrieved, 3), 1.0)
        self.assertAlmostEqual(reciprocal_rank_at_k(relevant, retrieved, 3), 0.5)

    def test_evaluate_run_returns_stable_macro_scores(self):
        qrels = load_qrels(FIXTURE_DIR / "qrels.json")
        run = load_run(FIXTURE_DIR / "run.json")

        result = evaluate_run(qrels, run, k=3)

        self.assertEqual([score.query_id for score in result.queries], ["q1", "q2", "q3"])
        self.assertEqual(result.query_count, 3)
        self.assertAlmostEqual(result.mean_precision, 4 / 9)
        self.assertAlmostEqual(result.mean_recall, 2 / 3)
        self.assertAlmostEqual(result.mean_reciprocal_rank, 0.5)
        self.assertAlmostEqual(result.hit_rate, 2 / 3)

    def test_missing_run_entry_is_scored_as_empty_ranking(self):
        result = evaluate_run({"q1": {"doc-a"}}, {}, k=5)

        self.assertEqual(result.queries[0].retrieved, 0)
        self.assertEqual(result.queries[0].precision, 0.0)
        self.assertEqual(result.queries[0].recall, 0.0)
        self.assertEqual(result.queries[0].reciprocal_rank, 0.0)

    def test_non_positive_k_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_run({"q1": {"doc-a"}}, {"q1": ["doc-a"]}, k=0)


class CliTests(unittest.TestCase):
    def test_cli_prints_markdown_scoreboard(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_scoreboard",
                "--qrels",
                str(FIXTURE_DIR / "qrels.json"),
                "--run",
                str(FIXTURE_DIR / "run.json"),
                "--k",
                "3",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("# RAG Scoreboard (k=3)", completed.stdout)
        self.assertIn("| q1 | 2 | 3 | 2 | 0.667 | 1.000 | 0.500 |", completed.stdout)
        self.assertIn("- precision@3: 0.444", completed.stdout)
        self.assertIn("- mrr@3: 0.500", completed.stdout)

    def test_renderer_output_is_deterministic(self):
        result = evaluate_run(load_qrels(FIXTURE_DIR / "qrels.json"), load_run(FIXTURE_DIR / "run.json"), k=3)
        first = render_markdown_table(result)
        second = render_markdown_table(result)

        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first), json.dumps(second))


if __name__ == "__main__":
    unittest.main()
