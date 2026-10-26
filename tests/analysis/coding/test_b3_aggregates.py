import pytest

from analysis.coding.b1_per_run_company import PerRunCompanyRow
from analysis.coding.b3_aggregates import (
    ELIGIBILITY_FLOOR,
    apply_eligibility_floor,
    build_cell_rates,
    day_to_day_variance,
    top_quartile_membership_per_day,
    top_ten_domains_by_citation_share,
    vertical_vs_engine_spread,
    within_day_variance,
)

UNBRANDED_INTENTS = {1, 2, 3, 4, 5, 6}


def _row(run_id, entity_id, mentioned, recommended=False):
    return PerRunCompanyRow(
        run_id=run_id, entity_id=entity_id, mentioned=mentioned, recommended=recommended,
        own_domain_cited=False, linked=False, mention_without_own_citation=False, first_mention_position=None,
    )


def _meta(engine, intent_id, day_index=1):
    return {"engine": engine, "intent_id": intent_id, "day_index": day_index}


# --------------------------------------------------------------------------
# apply_eligibility_floor -- the spec-precise part.
# --------------------------------------------------------------------------

def test_eligibility_floor_requires_exact_minimum_mentions():
    assert ELIGIBILITY_FLOOR == 5
    rows = [_row(f"run-{i}", "goodcall", mentioned=True) for i in range(5)]
    run_meta = {f"run-{i}": _meta("openai_chatgpt", intent_id=1) for i in range(5)}
    report = apply_eligibility_floor(rows, run_meta, UNBRANDED_INTENTS)
    assert report.qualifying_by_engine == {"openai_chatgpt": {"goodcall"}}
    assert report.qualifying_count("openai_chatgpt") == 1


def test_eligibility_floor_one_short_does_not_qualify():
    rows = [_row(f"run-{i}", "goodcall", mentioned=True) for i in range(4)]
    run_meta = {f"run-{i}": _meta("openai_chatgpt", intent_id=1) for i in range(4)}
    report = apply_eligibility_floor(rows, run_meta, UNBRANDED_INTENTS)
    assert report.qualifying_by_engine == {}


def test_eligibility_floor_ignores_branded_intent_mentions():
    # 5 mentions, but all in the branded intent (7) -- must not count toward
    # the unbranded-only floor.
    rows = [_row(f"run-{i}", "goodcall", mentioned=True) for i in range(5)]
    run_meta = {f"run-{i}": _meta("openai_chatgpt", intent_id=7) for i in range(5)}
    report = apply_eligibility_floor(rows, run_meta, UNBRANDED_INTENTS)
    assert report.qualifying_by_engine == {}


def test_eligibility_floor_is_per_engine_not_pooled():
    # 3 mentions on each of two engines -- neither clears the floor alone,
    # and they must not be pooled together.
    rows = (
        [_row(f"a-{i}", "goodcall", mentioned=True) for i in range(3)]
        + [_row(f"b-{i}", "goodcall", mentioned=True) for i in range(3)]
    )
    run_meta = {f"a-{i}": _meta("openai_chatgpt", 1) for i in range(3)}
    run_meta.update({f"b-{i}": _meta("google_gemini", 1) for i in range(3)})
    report = apply_eligibility_floor(rows, run_meta, UNBRANDED_INTENTS)
    assert report.qualifying_by_engine == {}


def test_eligibility_floor_ignores_not_mentioned_rows():
    rows = [_row(f"run-{i}", "goodcall", mentioned=(i < 5)) for i in range(10)]
    run_meta = {f"run-{i}": _meta("openai_chatgpt", 1) for i in range(10)}
    report = apply_eligibility_floor(rows, run_meta, UNBRANDED_INTENTS)
    assert report.qualifying_by_engine == {"openai_chatgpt": {"goodcall"}}


# --------------------------------------------------------------------------
# build_cell_rates
# --------------------------------------------------------------------------

def test_cell_rates_only_includes_qualifying_entities():
    rows = [_row(f"run-{i}", "goodcall", mentioned=True, recommended=True) for i in range(5)]
    rows += [_row("run-other", "vapi", mentioned=True)]  # never clears the floor
    run_meta = {f"run-{i}": _meta("openai_chatgpt", 1) for i in range(5)}
    run_meta["run-other"] = _meta("openai_chatgpt", 1)
    eligibility = apply_eligibility_floor(rows, run_meta, UNBRANDED_INTENTS)

    cells = build_cell_rates(rows, run_meta, eligibility)
    assert {c.entity_id for c in cells} == {"goodcall"}

    # restrict_to_eligible=False keeps the below-floor entity too, incl its
    # zero-mention-rate reality.
    unrestricted = build_cell_rates(rows, run_meta, eligibility, restrict_to_eligible=False)
    assert {c.entity_id for c in unrestricted} == {"goodcall", "vapi"}
    vapi = next(c for c in unrestricted if c.entity_id == "vapi")
    assert vapi.n_runs == 1 and vapi.mention_rate == 1.0
    cell = cells[0]
    assert cell.n_runs == 5
    assert cell.mention_rate == 1.0
    assert cell.recommendation_rate == 1.0


def test_cell_rates_mixed_mention_and_recommendation():
    rows = [_row(f"run-{i}", "goodcall", mentioned=True, recommended=(i < 2)) for i in range(5)]
    run_meta = {f"run-{i}": _meta("openai_chatgpt", 1) for i in range(5)}
    eligibility = apply_eligibility_floor(rows, run_meta, UNBRANDED_INTENTS)

    cells = build_cell_rates(rows, run_meta, eligibility)
    assert cells[0].mention_rate == 1.0
    assert cells[0].recommendation_rate == 2 / 5


# --------------------------------------------------------------------------
# variance / spread / quartile / domain share -- lighter-touch sanity checks
# (first-pass definitions, see module docstring).
# --------------------------------------------------------------------------

def test_within_day_variance_zero_when_perfectly_consistent():
    rows = [_row(f"run-{i}", "goodcall", mentioned=True) for i in range(3)]
    run_meta = {f"run-{i}": _meta("openai_chatgpt", 1, day_index=1) for i in range(3)}
    assert within_day_variance(rows, run_meta, "goodcall", 1, "openai_chatgpt") == 0.0


def test_within_day_variance_nonzero_when_reps_disagree():
    rows = [_row("run-0", "goodcall", mentioned=True), _row("run-1", "goodcall", mentioned=False)]
    run_meta = {"run-0": _meta("openai_chatgpt", 1, day_index=1), "run-1": _meta("openai_chatgpt", 1, day_index=1)}
    assert within_day_variance(rows, run_meta, "goodcall", 1, "openai_chatgpt") == 0.25


def test_day_to_day_variance_zero_when_every_day_has_same_rate():
    rows = [
        _row("d1-a", "goodcall", mentioned=True), _row("d1-b", "goodcall", mentioned=False),
        _row("d2-a", "goodcall", mentioned=True), _row("d2-b", "goodcall", mentioned=False),
    ]
    run_meta = {
        "d1-a": _meta("openai_chatgpt", 1, day_index=1), "d1-b": _meta("openai_chatgpt", 1, day_index=1),
        "d2-a": _meta("openai_chatgpt", 1, day_index=2), "d2-b": _meta("openai_chatgpt", 1, day_index=2),
    }
    assert day_to_day_variance(rows, run_meta, "goodcall", 1, "openai_chatgpt") == 0.0


def test_vertical_vs_engine_spread_computes_max_minus_min():
    from analysis.coding.b3_aggregates import CellRates

    cells = [
        CellRates("goodcall", 1, "openai_chatgpt", 5, mention_rate=1.0, recommendation_rate=1.0),
        CellRates("goodcall", 1, "google_gemini", 5, mention_rate=0.4, recommendation_rate=0.4),
        CellRates("goodcall", 2, "openai_chatgpt", 5, mention_rate=0.2, recommendation_rate=0.2),
    ]
    spread = vertical_vs_engine_spread(cells, "goodcall")
    # openai_chatgpt's mean across its 2 intents is (1.0+0.2)/2=0.6; google_gemini
    # has only 1 intent so its mean is just 0.4. engine_spread = 0.6 - 0.4.
    assert spread.engine_spread == pytest.approx(0.6 - 0.4)
    assert spread.intent_spread is not None


def test_top_quartile_membership_false_with_too_few_entities():
    rows = [_row("run-1", "a", mentioned=True), _row("run-2", "b", mentioned=True)]
    run_meta = {"run-1": _meta("x", 1, day_index=1), "run-2": _meta("x", 1, day_index=1)}
    membership = top_quartile_membership_per_day(rows, run_meta)
    assert membership == {(1, "a"): False, (1, "b"): False}


def test_top_ten_domains_by_citation_share():
    sources_by_run = {
        "r1": [{"domain": "goodcall.com"}, {"domain": "goodcall.com"}],
        "r2": [{"domain": "vapi.ai"}],
    }
    result = top_ten_domains_by_citation_share(sources_by_run)
    assert result[0] == ("goodcall.com", 2 / 3)
    assert result[1] == ("vapi.ai", 1 / 3)


def test_top_ten_domains_empty_when_no_sources():
    assert top_ten_domains_by_citation_share({}) == []
