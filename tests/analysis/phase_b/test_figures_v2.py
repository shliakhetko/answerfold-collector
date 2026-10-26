import pytest

from analysis.phase_b.figures import v2
from analysis.phase_b.reading import FrozenInputs


def _mention(run_id, surface_form, is_recommended=False):
    return {
        "run_id": run_id, "surface_form": surface_form, "evidence_span": "x",
        "is_recommended": is_recommended, "votes": [],
    }


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _raw_run(
    run_id, *, engine="openai_chatgpt", intent_id=1, paraphrase_id=1, day_index=1, rep_index=1,
    error=None, raw_response_text="some text",
):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": paraphrase_id,
        "day_index": day_index, "rep_index": rep_index, "error": error,
        "raw_response_text": raw_response_text, "is_manual": None,
    }


def _inputs(raw_runs, mentions_raw, entities):
    return FrozenInputs(
        entities=entities, alt_domains_rows=[], entity_verification_rows=[],
        mentions_raw=mentions_raw, raw_runs=raw_runs, sources_by_run={},
    )


def _unit_week(entity_id, engine, intent_id, paraphrase_id, mentioned_by_day, *, reps=3):
    """mentioned_by_day: day_index -> number of the `reps` reps that
    mention entity_id (0..reps). Produces reps usable runs per day."""
    raw_runs, mentions = [], []
    for day_index, n_mentioned in mentioned_by_day.items():
        for rep_index in range(1, reps + 1):
            run_id = f"{entity_id}-{engine}-i{intent_id}-p{paraphrase_id}-d{day_index}-r{rep_index}"
            raw_runs.append(
                _raw_run(
                    run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id,
                    day_index=day_index, rep_index=rep_index,
                )
            )
            if rep_index <= n_mentioned:
                mentions.append(_mention(run_id, entity_id))
    return raw_runs, mentions


def _qualifying_floor_runs(entity_id, engine):
    """5 additional usable, mentioned runs across 5 distinct unbranded
    intents/days -- pads any entity over the >=5-mention eligibility floor
    without disturbing the specific unit under test."""
    raw_runs, mentions = [], []
    for i, intent_id in enumerate(range(1, 6), start=1):
        run_id = f"{entity_id}-{engine}-floor-i{intent_id}"
        raw_runs.append(_raw_run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=5, day_index=1, rep_index=i))
        mentions.append(_mention(run_id, entity_id))
    return raw_runs, mentions


def test_zero_cells_is_untestable():
    result = v2.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_constant_rate_every_day_gives_zero_excess_drift():
    # same rate (2/3) every day for 3 days -- between-day variance is 0, so
    # excess_drift must floor at exactly 0 regardless of within-day variance.
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 2, 2: 2, 3: 2})
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    unit = result.notes["per_unit"][0]
    assert unit["excess_drift_points"] == pytest.approx(0.0)
    assert unit["sensitivity_excess_drift_points"] == pytest.approx(0.0)


def test_big_between_day_swing_matches_hand_computed_value():
    # day 1: 0/3 mentioned (rate 0), day 2: 3/3 mentioned (rate 1) -- a
    # maximal swing. within-day variance is 0 on both days (rate*(1-rate)
    # is 0 at both 0 and 1), so excess_drift == sqrt(between_var) exactly.
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 0, 2: 3})
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    unit = result.notes["per_unit"][0]
    # daily rates {0.0, 1.0} -> population variance = 0.25 -> sqrt = 0.5 -> *100 = 50 points
    assert unit["between_day_variance"] == pytest.approx(0.25)
    assert unit["primary_within_day_variance"] == pytest.approx(0.0)
    assert unit["excess_drift_points"] == pytest.approx(50.0)


def test_sensitivity_never_below_primary():
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 1, 2: 2, 3: 3})
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    unit = result.notes["per_unit"][0]
    assert unit["sensitivity_excess_drift_points"] >= unit["excess_drift_points"] - 1e-9


def test_engine_value_and_sensitivity_present_when_enough_qualify():
    # 10 distinct companies, each clearing the eligibility floor and each
    # contributing one well-formed 2-day unit -- enough to clear
    # MIN_QUALIFYING_FOR_V2 and get a real, non-None engine value.
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        week_runs, week_mentions = _unit_week(entity_id, engine, 1, 1, {1: 1, 2: 2})
        floor_runs, floor_mentions = _qualifying_floor_runs(entity_id, engine)
        raw_runs += week_runs + floor_runs
        mentions += week_mentions + floor_mentions
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = v2.compute(_inputs(raw_runs, mentions, entities))
    assert result.value[engine] is not None
    assert result.notes["sensitivity"][engine] is not None
    assert engine not in result.notes["untestable_engines"]


def test_two_usable_rep_day_handled_without_crashing_or_assuming_three():
    # day 2 only has 2 usable reps (mirrors the real google_gemini/i6/p4/d7
    # gap) -- must not crash, and must not silently treat it as 3.
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 1, 2: 1, 3: 2})
    # drop one of day 2's reps to leave only 2 usable
    dropped_run_id = "acme-openai_chatgpt-i1-p1-d2-r2"
    raw_runs = [r for r in raw_runs if r["run_id"] != dropped_run_id]
    mentions = [m for m in mentions if m["run_id"] != dropped_run_id]
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    unit = result.notes["per_unit"][0]
    assert unit["reps_used_by_day"][2] == 2
    assert unit["daily_rates"][2] == pytest.approx(0.5)  # 1 of 2, not assumed out of 3


def test_engine_below_min_qualifying_is_untestable_not_whole_figure():
    # only 1 qualifying company in this engine -- below MIN_QUALIFYING_FOR_V2
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 1, 2: 2})
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    assert result.status == "ok"
    assert result.value["openai_chatgpt"] is None
    assert "openai_chatgpt" in result.notes["untestable_engines"]


def test_unit_with_one_day_excluded_as_fewer_than_2_days():
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 2})
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    assert result.n_excluded.get("fewer_than_2_days", 0) >= 1
    unit_keys = {(u["intent_id"], u["paraphrase_id"]) for u in result.notes["per_unit"]}
    assert (1, 1) not in unit_keys  # the day-1-only (intent 1, paraphrase 1) unit never enters per_unit


def test_all_zero_and_all_hundred_units_flagged_separately():
    raw_runs, mentions = [], []
    rr, mm = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 0, 2: 0})  # never mentioned
    raw_runs += rr; mentions += mm
    rr, mm = _unit_week("acme", "openai_chatgpt", 1, 2, {1: 3, 2: 3})  # always mentioned
    raw_runs += rr; mentions += mm
    rr, mm = _unit_week("acme", "openai_chatgpt", 1, 3, {1: 1, 2: 2})  # a genuinely mixed unit
    raw_runs += rr; mentions += mm
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs, mentions, entities))
    units = {u["paraphrase_id"]: u for u in result.notes["per_unit"]}
    assert units[1]["all_zero"] is True and units[1]["all_hundred"] is False
    assert units[2]["all_hundred"] is True and units[2]["all_zero"] is False
    assert units[3]["all_zero"] is False and units[3]["all_hundred"] is False


def test_signed_excess_drift_can_be_negative_before_floor():
    # rates hovering near 0.5 (high within-day variance) with almost no
    # day-to-day movement (low between-day variance) -- within > between,
    # so the unfloored difference is negative even though the primary,
    # floored metric stays pinned at 0.
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 5, 2: 5, 3: 6}, reps=10)
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    unit = result.notes["per_unit"][0]
    assert unit["raw_variance_diff"] < 0
    assert unit["signed_excess_drift_points"] < 0
    assert unit["excess_drift_points"] == pytest.approx(0.0)


def test_share_alert_worthy_matches_hand_count(monkeypatch):
    # build_daily_panel's run x qualifying-entity cross product means a
    # second qualifying entity gets an all-zero "phantom" unit for every
    # grid cell the first entity used (it was never mentioned there,
    # which is itself a legitimate all-zero unit) -- lowering the floor to
    # 2 keeps that multiplication small and the hand count tractable.
    monkeypatch.setattr(v2, "MIN_QUALIFYING_FOR_V2", 2)
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for paraphrase_id in range(21, 29):  # 8 constant-rate units -- 0 excess drift each
        rr, mm = _unit_week("acme", engine, 1, paraphrase_id, {1: 3, 2: 3})
        raw_runs += rr; mentions += mm
    for paraphrase_id in (29, 30):  # 2 units with a maximal 0 -> 1 swing -- 50pts each
        rr, mm = _unit_week("acme", engine, 1, paraphrase_id, {1: 0, 2: 3})
        raw_runs += rr; mentions += mm
    entities.append(_entity("acme", "https://acme.com", "acme"))
    # one filler entity (paraphrase_id=5 internally -- no collision with 21-30)
    # to clear MIN_QUALIFYING_FOR_V2=2 without a same-grid unit of its own.
    filler_runs, filler_mentions = _qualifying_floor_runs("filler", engine)
    raw_runs += filler_runs; mentions += filler_mentions
    entities.append(_entity("filler", "https://filler.com", "filler"))

    result = v2.compute(_inputs(raw_runs, mentions, entities))
    # 10 real units from acme + 10 all-zero phantom copies for filler (never
    # mentioned in any of acme's runs) = 20.
    assert result.n_included[engine] == 20
    assert result.notes["n_alert_worthy_by_engine"][engine] == 2
    assert result.notes["share_alert_worthy_by_engine"][engine] == pytest.approx(0.1)


def test_nonzero_median_excludes_all_zero_but_reflects_real_drift(monkeypatch):
    monkeypatch.setattr(v2, "MIN_QUALIFYING_FOR_V2", 2)
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for paraphrase_id in range(21, 26):  # 5 all-zero units
        rr, mm = _unit_week("acme", engine, 1, paraphrase_id, {1: 0, 2: 0})
        raw_runs += rr; mentions += mm
    rr, mm = _unit_week("acme", engine, 1, 26, {1: 0, 2: 3})  # 1 unit with real drift (50pts)
    raw_runs += rr; mentions += mm
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", engine)  # clears acme's own floor
    raw_runs += floor_runs; mentions += floor_mentions
    entities.append(_entity("acme", "https://acme.com", "acme"))
    filler_runs, filler_mentions = _qualifying_floor_runs("filler", engine)
    raw_runs += filler_runs; mentions += filler_mentions
    entities.append(_entity("filler", "https://filler.com", "filler"))

    result = v2.compute(_inputs(raw_runs, mentions, entities))
    # acme: 5 all-zero + 1 real-drift = 6 counted units. filler gets 6
    # all-zero phantom copies of the same grid (never mentioned there) -> 12.
    assert result.n_included[engine] == 12
    assert result.notes["n_all_zero_by_engine"][engine] == 11  # 5 (acme) + 6 (filler's copies)
    assert result.notes["median_excess_drift_nonzero_only"][engine] == pytest.approx(50.0)
    assert result.notes["sensitivity_explanation"]


def test_noise_only_crossing_probability_is_zero_for_all_zero_unit():
    # pooled_rate=0 -> Binomial(n, 0) is always 0 -> between_var and
    # within_var are always exactly 0 -> excess drift is always exactly 0
    # -> this unit can NEVER cross the 5pt threshold under the null.
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 0, 2: 0, 3: 0})
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    unit = result.notes["per_unit"][0]
    assert unit["noise_only_crossing_probability"] == pytest.approx(0.0)


def test_noise_only_crossing_probability_in_valid_range():
    raw_runs, mentions = _unit_week("acme", "openai_chatgpt", 1, 1, {1: 1, 2: 2, 3: 0})
    floor_runs, floor_mentions = _qualifying_floor_runs("acme", "openai_chatgpt")
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = v2.compute(_inputs(raw_runs + floor_runs, mentions + floor_mentions, entities))
    unit = result.notes["per_unit"][0]
    assert 0.0 <= unit["noise_only_crossing_probability"] <= 1.0


def test_share_alert_worthy_null_present_and_deterministic():
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        week_runs, week_mentions = _unit_week(entity_id, engine, 1, 1, {1: 1, 2: 2})
        floor_runs, floor_mentions = _qualifying_floor_runs(entity_id, engine)
        raw_runs += week_runs + floor_runs
        mentions += week_mentions + floor_mentions
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    inputs = _inputs(raw_runs, mentions, entities)

    result1 = v2.compute(inputs)
    result2 = v2.compute(inputs)
    null1 = result1.notes["share_alert_worthy_null_by_engine"][engine]
    null2 = result2.notes["share_alert_worthy_null_by_engine"][engine]
    assert null1 is not None
    assert 0.0 <= null1 <= 1.0
    assert null1 == null2


def test_noise_comparison_fields_and_label_rule():
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        week_runs, week_mentions = _unit_week(entity_id, engine, 1, 1, {1: 1, 2: 2})
        floor_runs, floor_mentions = _qualifying_floor_runs(entity_id, engine)
        raw_runs += week_runs + floor_runs
        mentions += week_mentions + floor_mentions
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    result = v2.compute(_inputs(raw_runs, mentions, entities))
    cmp_ = result.notes["noise_comparison_by_engine"][engine]
    for key in (
        "null_share_alert_worthy", "observed_share_alert_worthy", "ratio_observed_to_null",
        "observed_percentile_in_null", "null_p95_share", "label",
    ):
        assert key in cmp_
    assert 0.0 <= cmp_["observed_percentile_in_null"] <= 1.0
    # label rule: real only when observed strictly exceeds null p95
    expected = "real" if cmp_["observed_share_alert_worthy"] > cmp_["null_p95_share"] else "consistent_with_noise"
    assert cmp_["label"] == expected


def test_noise_comparison_label_helper_never_real_when_not_above_p95():
    import numpy as np
    counts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)  # null shares 0.0..0.9 of 10 units
    at_p95 = v2._noise_comparison(0.85, counts, 10)
    assert at_p95["label"] == "consistent_with_noise"
    above = v2._noise_comparison(0.95, counts, 10)
    assert above["label"] == "real"
