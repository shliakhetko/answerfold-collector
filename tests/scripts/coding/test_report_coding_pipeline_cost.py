import json

import pytest

from scripts.coding.report_coding_pipeline_cost import build_report


def _write_cache_entry(cache_dir, stage, key, *, model_returned, tokens_input, tokens_output, error=None):
    stage_dir = cache_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"{key}.json").write_text(
        json.dumps(
            {
                "error": error,
                "model_returned": model_returned,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
            }
        ),
        encoding="utf-8",
    )


def test_missing_cache_dir_returns_empty_report(tmp_path):
    report = build_report(tmp_path / "does_not_exist")
    assert report.by_stage == {}
    assert report.total_cost_usd() == 0.0


def test_groups_by_stage_and_model_and_computes_cost(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    _write_cache_entry(
        cache_dir, "a1_extractors", "aaa",
        model_returned="claude-haiku-4-5-20251001", tokens_input=1_000_000, tokens_output=1_000_000,
    )
    _write_cache_entry(
        cache_dir, "a1_extractors", "bbb",
        model_returned="gemini-3.5-flash-lite", tokens_input=1_000_000, tokens_output=1_000_000,
    )
    _write_cache_entry(
        cache_dir, "a4_entity_classification", "ccc",
        model_returned="claude-haiku-4-5-20251001", tokens_input=1_000_000, tokens_output=1_000_000,
    )

    report = build_report(cache_dir)

    assert set(report.by_stage) == {"a1_extractors", "a4_entity_classification"}
    a1 = report.by_stage["a1_extractors"]
    assert a1.n_calls == 2
    assert a1.n_errors == 0
    assert a1.cost_usd() == pytest.approx((1.00 + 5.00) + (0.10 + 0.40))
    assert report.total_cost_usd() == pytest.approx(a1.cost_usd() + report.by_stage["a4_entity_classification"].cost_usd())


def test_errored_entry_counted_but_not_priced_when_tokens_missing(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    _write_cache_entry(
        cache_dir, "a1_extractors", "err",
        model_returned=None, tokens_input=None, tokens_output=None, error="TimeoutError: simulated",
    )

    report = build_report(cache_dir)

    a1 = report.by_stage["a1_extractors"]
    assert a1.n_calls == 1
    assert a1.n_errors == 1
    assert a1.cost_usd() == 0.0


def test_batch_cost_is_half_standard_cost(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    _write_cache_entry(
        cache_dir, "a1_extractors", "aaa",
        model_returned="claude-haiku-4-5-20251001", tokens_input=1_000_000, tokens_output=1_000_000,
    )

    report = build_report(cache_dir)

    a1 = report.by_stage["a1_extractors"]
    assert a1.cost_usd_if_batch() == pytest.approx(a1.cost_usd() / 2)
