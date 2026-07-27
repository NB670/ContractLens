"""Tests for the consolidated Checkpoint 4 benchmark runner.

Runs against the committed CUAD sample with the offline default backends
(rule classifier + hashing embedder), on a small `limit` slice for speed, and
checks the --out artifact is written as valid JSON.
"""

import json

from scripts.run_benchmarks import run, write_results


def test_run_benchmarks_smoke_hashing_rule():
    summary = run(classifier_name="rule", retrieval_backend="hashing", k=5, limit=40)

    assert summary["dataset"] == "cuad_sample.json"
    assert summary["records_scored"] == 40

    macro = summary["classification"]["macro_avg"]
    for key in ("precision", "recall", "f1"):
        assert 0.0 <= macro[key] <= 1.0

    ret = summary["retrieval"]
    assert ret["backend"] == "hashing"
    assert ret["k"] == 5
    for key in ("recall_at_k", "success_at_k", "mrr"):
        assert 0.0 <= ret[key] <= 1.0
    assert ret["queries"] >= 0


def test_run_benchmarks_reports_per_category_without_macro_row():
    summary = run(limit=60)
    per_category = summary["classification"]["per_category"]
    assert per_category, "expected at least one scored category"
    assert "macro_avg" not in per_category


def test_write_results_produces_valid_json_file(tmp_path):
    summary = run(limit=20)
    out_path = tmp_path / "benchmark.json"

    write_results(summary, out_path)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded == summary
