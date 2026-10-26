from analysis.aggregate.pricing import compute_cost_usd


def test_known_model_computes_expected_cost():
    # gpt-4o: $2.50/1M input, $10.00/1M output (analysis/aggregate/pricing.py)
    cost = compute_cost_usd("gpt-4o-2024-08-06", 1_000_000, 1_000_000)
    assert cost == 2.50 + 10.00


def test_unknown_model_returns_none():
    assert compute_cost_usd("some-future-model", 100, 100) is None


def test_missing_tokens_returns_none():
    assert compute_cost_usd("gpt-4o", None, 100) is None
    assert compute_cost_usd("gpt-4o", 100, None) is None


def test_none_model_returns_none():
    assert compute_cost_usd(None, 100, 100) is None


def test_nan_inputs_do_not_raise():
    # google_ai_mode rows have no tokens/model at all -- once loaded and
    # read back through pandas (notebooks/aggregate.ipynb), an all-NULL SQL
    # column becomes float('nan'), not None. Must degrade to None, not raise.
    nan = float("nan")
    assert compute_cost_usd(nan, 100, 100) is None
    assert compute_cost_usd("gpt-4o", nan, 100) is None
    assert compute_cost_usd("gpt-4o", 100, nan) is None


def test_gemini_flash_lite_versioned_model_returned_matches_via_substring():
    # The "-latest" alias this repo pins (analysis/coding/extractors/models.py)
    # resolves to a model_returned with the version number spliced into the
    # middle, e.g. "gemini-3.5-flash-lite" -- confirmed against real cache
    # data. A plain prefix check would miss this; substring match catches it.
    cost = compute_cost_usd("gemini-3.5-flash-lite", 1_000_000, 1_000_000)
    assert cost == 0.10 + 0.40


def test_claude_haiku_and_gpt_5_mini_match():
    assert compute_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == 1.00 + 5.00
    assert compute_cost_usd("gpt-5-mini-2025-08-07", 1_000_000, 1_000_000) == 0.25 + 2.00


def test_batch_pricing_is_half_of_standard():
    standard = compute_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
    batch = compute_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000, batch=True)
    assert batch == standard / 2
