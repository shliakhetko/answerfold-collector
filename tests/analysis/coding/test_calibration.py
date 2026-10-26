import json

from analysis.coding.calibration import (
    apply_overrides,
    compute_agreement,
    measure_actual_token_cost,
    sample_calibration_runs,
    sample_low_confidence_urls,
    sample_spot_check,
)


def test_sample_calibration_runs_is_deterministic_given_seed():
    run_rows = [{"run_id": f"run-{i}"} for i in range(100)]
    a = sample_calibration_runs(run_rows, n=40, seed=42)
    b = sample_calibration_runs(run_rows, n=40, seed=42)
    assert a == b
    assert len(a) == 40
    assert len(set(a)) == 40  # no duplicates


def test_sample_calibration_runs_caps_at_available_runs():
    run_rows = [{"run_id": f"run-{i}"} for i in range(10)]
    result = sample_calibration_runs(run_rows, n=40, seed=1)
    assert len(result) == 10


def test_sample_spot_check_splits_by_vote_count():
    mentions_raw_rows = [
        {"run_id": "r1", "surface_form": "Goodcall", "votes": ["a", "b"]},  # 2-of-3 disagreement case
        {"run_id": "r2", "surface_form": "Vapi", "votes": ["a", "b", "c"]},  # full agreement, not in 2of3 pool
    ]
    single_vote_groups = [{"run_id": "r3", "extractor": "gemini", "surface_form": "Rosie"}]

    sample = sample_spot_check(
        mentions_raw_rows, single_vote_groups, n_2of3=5, n_1of3_dropped=5, n_random=5, seed=1
    )

    assert sample.two_of_three == [mentions_raw_rows[0]]
    assert sample.one_of_three_dropped == single_vote_groups
    assert set(r["run_id"] for r in sample.random_slice) == {"r1", "r2"}


def test_sample_spot_check_caps_at_available_pool_size():
    mentions_raw_rows = [{"run_id": "r1", "surface_form": "Goodcall", "votes": ["a", "b"]}]
    sample = sample_spot_check(mentions_raw_rows, [], n_2of3=10, n_1of3_dropped=10, n_random=10, seed=1)
    assert len(sample.two_of_three) == 1
    assert sample.one_of_three_dropped == []
    assert len(sample.random_slice) == 1


def test_sample_low_confidence_urls_takes_bottom_10_percent():
    url_registry_rows = [{"url_normalized": f"u{i}", "classifier_confidence": str(i / 10)} for i in range(10)]
    result = sample_low_confidence_urls(url_registry_rows, fraction=0.10)
    assert len(result) == 1
    assert result[0]["url_normalized"] == "u0"  # lowest confidence


def test_sample_low_confidence_urls_excludes_missing_confidence():
    url_registry_rows = [{"url_normalized": "u1", "classifier_confidence": ""}, {"url_normalized": "u2", "classifier_confidence": "0.5"}]
    result = sample_low_confidence_urls(url_registry_rows, fraction=1.0)
    assert result == [{"url_normalized": "u2", "classifier_confidence": "0.5"}]


def test_sample_low_confidence_urls_empty_input():
    assert sample_low_confidence_urls([]) == []


def _spot_row(url_normalized, confidence, model_class, model_pub, human_class="", human_pub="", note=""):
    return {
        "url_normalized": url_normalized,
        "classifier_confidence": confidence,
        "model_url_class": model_class,
        "model_publisher_is_in_category": model_pub,
        "human_class": human_class,
        "human_publisher_is_in_category": human_pub,
        "note": note,
    }


def test_apply_overrides_only_below_threshold_with_human_label():
    rows = [
        _spot_row("u1", "0.45", "other", "False", human_class="forum", human_pub="False"),  # below, reviewed -> override
        _spot_row("u2", "0.45", "other", "False"),  # below, NOT reviewed -> no override
        _spot_row("u3", "0.95", "vendor page", "True", human_class="forum", human_pub="False"),  # >= threshold -> no override (calibration only)
    ]
    overrides = apply_overrides(rows, threshold=0.70)
    assert overrides == {"u1": ("forum", False)}


def test_apply_overrides_empty_confidence_is_skipped():
    rows = [_spot_row("u1", "", "other", "False", human_class="forum", human_pub="False")]
    assert apply_overrides(rows, threshold=0.70) == {}


def test_compute_agreement_splits_by_band_and_counts_only_reviewed():
    rows = [
        _spot_row("u1", "0.45", "other", "False", human_class="other", human_pub="False"),  # below, agree
        _spot_row("u2", "0.45", "other", "False", human_class="forum", human_pub="True"),  # below, disagree both
        _spot_row("u3", "0.45", "other", "False"),  # below, unreviewed -> excluded
        _spot_row("u4", "0.95", "vendor page", "True", human_class="vendor page", human_pub="True"),  # above, agree
    ]
    result = compute_agreement(rows, threshold=0.70)
    below = result["below_threshold"]
    above = result["above_threshold_calibration"]
    assert below["n_sampled"] == 3
    assert below["n_reviewed"] == 2
    assert below["class_agree_rate"] == 0.5
    assert below["publisher_agree_rate"] == 0.5
    assert below["both_agree_rate"] == 0.5
    assert above["n_sampled"] == 1
    assert above["n_reviewed"] == 1
    assert above["class_agree_rate"] == 1.0


def test_compute_agreement_empty_band_reports_no_reviewed():
    result = compute_agreement([], threshold=0.70)
    assert result["below_threshold"] == {
        "n_sampled": 0, "n_reviewed": 0, "class_agree_rate": None, "publisher_agree_rate": None, "both_agree_rate": None,
    }


def _write_cache_entry(cache_dir, name, tokens_input, tokens_output):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{name}.json").write_text(
        json.dumps({"tokens_input": tokens_input, "tokens_output": tokens_output, "mentions": []})
    )


def test_measure_actual_token_cost_reads_from_cache_entries(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    _write_cache_entry(cache_dir / "anthropic", "a", 100, 40)
    _write_cache_entry(cache_dir / "openai", "b", 120, 30)
    _write_cache_entry(cache_dir / "gemini", "c", 90, 20)

    summary = measure_actual_token_cost(cache_dir, n=None)

    assert summary.n_cache_entries == 3
    assert summary.total_tokens_input == 310
    assert summary.total_tokens_output == 90
    assert summary.mean_tokens_input == 310 / 3


def test_measure_actual_token_cost_caps_at_n(tmp_path):
    cache_dir = tmp_path / "llm_cache" / "anthropic"
    for i in range(5):
        _write_cache_entry(cache_dir, f"entry-{i}", 10, 5)
    summary = measure_actual_token_cost(tmp_path / "llm_cache", n=2)
    assert summary.n_cache_entries == 2


def test_measure_actual_token_cost_empty_cache_dir(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir()
    summary = measure_actual_token_cost(cache_dir)
    assert summary == measure_actual_token_cost(cache_dir)
    assert summary.n_cache_entries == 0
    assert summary.mean_tokens_input is None
