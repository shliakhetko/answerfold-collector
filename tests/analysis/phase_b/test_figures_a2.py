import pytest

from analysis.phase_b.figures import a2
from analysis.phase_b.reading import FrozenInputs


def _mention(run_id, surface_form):
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": False, "votes": []}


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _raw_run(run_id, *, engine, intent_id=1, paraphrase_id=1, day_index, rep_index, error=None):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": paraphrase_id,
        "day_index": day_index, "rep_index": rep_index, "error": error,
        "raw_response_text": "some text", "is_manual": None,
    }


def _inputs(raw_runs, mentions_raw, entities):
    return FrozenInputs(
        entities=entities, alt_domains_rows=[], entity_verification_rows=[],
        mentions_raw=mentions_raw, raw_runs=raw_runs, sources_by_run={},
    )


def _week_runs(entity_id, engine, mentioned_by_day, *, reps=3):
    """mentioned_by_day: day_index (1..7) -> number of the `reps` reps
    that mention entity_id, single (intent 1, paraphrase 1) unit."""
    raw_runs, mentions = [], []
    for day_index, n_mentioned in mentioned_by_day.items():
        for rep_index in range(1, reps + 1):
            run_id = f"{entity_id}-{engine}-d{day_index}-r{rep_index}"
            raw_runs.append(_raw_run(run_id, engine=engine, day_index=day_index, rep_index=rep_index))
            if rep_index <= n_mentioned:
                mentions.append(_mention(run_id, entity_id))
    return raw_runs, mentions


VARYING_PATTERN = {1: 0, 2: 1, 3: 2, 4: 3, 5: 1, 6: 2, 7: 0}  # 9 mentions total -- clears the floor


def test_zero_cells_is_untestable():
    result = a2.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_identical_engine_patterns_give_perfect_correlation():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        for engine in (e1, e2):
            rr, mm = _week_runs(entity_id, engine, VARYING_PATTERN)
            raw_runs += rr
            mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    assert result.value[pair_key] == pytest.approx(1.0)
    assert result.notes["matrix"][e1][e2] == pytest.approx(1.0)
    assert result.value["mean"] == pytest.approx(1.0)


def test_pair_below_floor_is_untestable_not_a_crash():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for i in range(3):  # below MIN_QUALIFYING_FOR_PAIR
        entity_id = f"co{i}"
        for engine in (e1, e2):
            rr, mm = _week_runs(entity_id, engine, VARYING_PATTERN)
            raw_runs += rr
            mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    assert result.value[pair_key] is None
    assert pair_key in result.notes["untestable_pairs"]
    assert result.value["mean"] is None  # no computable pairs at all


def test_mean_computed_only_over_computable_pairs():
    e1, e2, e3 = "anthropic_claude", "google_ai_mode", "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    # 10 companies shared by e1 and e2, identical pattern -> r=1.0
    for i in range(10):
        entity_id = f"co{i}"
        for engine in (e1, e2):
            rr, mm = _week_runs(entity_id, engine, VARYING_PATTERN)
            raw_runs += rr
            mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    # e3 only has 3 companies at all -- both pairs touching e3 fall below the floor
    for i in range(3):
        entity_id = f"e3only{i}"
        rr, mm = _week_runs(entity_id, e3, VARYING_PATTERN)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    assert result.value[f"{e1}|{e2}"] == pytest.approx(1.0)
    assert result.value[f"{e1}|{e3}"] is None
    assert result.value[f"{e2}|{e3}"] is None
    assert result.value["mean"] == pytest.approx(1.0)  # only the one computable pair counted


ANTI_PATTERN = {d: 3 - v for d, v in VARYING_PATTERN.items()}  # exact mirror -> perfect negative correlation


def _ten_company_anti_correlated_pair(e1, e2):
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        rr, mm = _week_runs(entity_id, e1, VARYING_PATTERN)
        raw_runs += rr
        mentions += mm
        rr, mm = _week_runs(entity_id, e2, ANTI_PATTERN)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    return raw_runs, mentions, entities


def test_cluster_bootstrap_deterministic_given_fixed_seed():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = _ten_company_anti_correlated_pair(e1, e2)
    inputs = _inputs(raw_runs, mentions, entities)

    result1 = a2.compute(inputs)
    result2 = a2.compute(inputs)
    pair_key = f"{e1}|{e2}"
    ci1 = result1.notes["ci_by_pair"][pair_key]
    ci2 = result2.notes["ci_by_pair"][pair_key]
    assert ci1["ci_low"] == ci2["ci_low"]
    assert ci1["ci_high"] == ci2["ci_high"]


def test_fisher_z_reference_present_alongside_cluster_ci():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = _ten_company_anti_correlated_pair(e1, e2)

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    assert result.notes["ci_by_pair"][pair_key]["ci_low"] is not None
    assert result.notes["fisher_z_ci_by_pair_reference"][pair_key]["ci_low"] is not None
    assert result.notes["ci_by_pair"][pair_key]["n_companies"] == 10


def test_cluster_ci_excluding_zero_marks_pair_unexplained():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = _ten_company_anti_correlated_pair(e1, e2)

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    assert result.notes["ci_by_pair"][pair_key]["excludes_zero"] is True
    assert pair_key in result.notes["unexplained_pairs"]


def test_white_noise_lag1_reference_exposed():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = _ten_company_anti_correlated_pair(e1, e2)

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    assert result.notes["white_noise_lag1_reference"] == pytest.approx(-0.5)
    assert a2.WHITE_NOISE_LAG1_REFERENCE == pytest.approx(-0.5)


def test_ci_excludes_zero_for_strong_negative_correlation():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        rr, mm = _week_runs(entity_id, e1, VARYING_PATTERN)
        raw_runs += rr
        mentions += mm
        rr, mm = _week_runs(entity_id, e2, ANTI_PATTERN)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    assert result.value[pair_key] == pytest.approx(-1.0)
    ci = result.notes["ci_by_pair"][pair_key]
    assert ci["excludes_zero"] is True
    assert ci["ci_high"] < 0.0


def test_fisher_ci_none_below_n_floor():
    result = a2._fisher_ci(0.5, 3)
    assert result["ci_low"] is None
    assert result["ci_high"] is None
    assert result["excludes_zero"] is None


def test_within_engine_autocorrelation_present_and_deterministic():
    engine = "anthropic_claude"
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        rr, mm = _week_runs(entity_id, engine, VARYING_PATTERN)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    inputs = _inputs(raw_runs, mentions, entities)

    result1 = a2.compute(inputs)
    result2 = a2.compute(inputs)
    auto1 = result1.notes["within_engine_delta_autocorrelation"][engine]
    auto2 = result2.notes["within_engine_delta_autocorrelation"][engine]
    assert auto1["lag1_r"] is not None
    assert auto1["lag1_r"] == pytest.approx(auto2["lag1_r"])
    assert auto1["n_pairs"] == auto2["n_pairs"]


def test_engine_level_correlation_capped_at_six_and_notes_only():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    for i in range(10):
        entity_id = f"co{i}"
        for engine in (e1, e2):
            rr, mm = _week_runs(entity_id, engine, VARYING_PATTERN)
            raw_runs += rr
            mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    engine_level = result.notes["engine_level_correlation"][pair_key]
    assert engine_level["n"] <= 6
    assert engine_level["r"] == pytest.approx(1.0)
    assert "engine_level_correlation" not in result.value


def test_verdict_relabeled_no_evidence_of_shared_movement():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = [], [], []
    # constructed so the two delta sequences are exactly orthogonal (dot
    # product 0): pattern_a's rate deltas are [1,-1,1,-1,1,-1] (out of a
    # 0-3 count), pattern_b's are proportional to [1,-1,1,1,-1,1] -- three
    # sign matches, three mismatches, so r == 0 exactly.
    pattern_a = {1: 0, 2: 3, 3: 0, 4: 3, 5: 0, 6: 3, 7: 0}
    pattern_b = {1: 0, 2: 1, 3: 0, 4: 1, 5: 2, 6: 1, 7: 2}
    for i in range(10):
        entity_id = f"co{i}"
        rr, mm = _week_runs(entity_id, e1, pattern_a)
        raw_runs += rr
        mentions += mm
        rr, mm = _week_runs(entity_id, e2, pattern_b)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    assert abs(result.value[pair_key]) < a2.SHARED_MOVEMENT_AT_OR_ABOVE
    assert result.notes["verdict"] == a2.NO_EVIDENCE_OF_SHARED_MOVEMENT


def test_summary_pairs_agree_with_table_numbers_to_three_decimals():
    e1, e2 = "anthropic_claude", "openai_chatgpt"
    raw_runs, mentions, entities = _ten_company_anti_correlated_pair(e1, e2)

    result = a2.compute(_inputs(raw_runs, mentions, entities))
    pair_key = f"{e1}|{e2}"
    row = next(r for r in result.notes["summary_pairs"] if r["pair"] == pair_key)
    ci = result.notes["ci_by_pair"][pair_key]
    assert row["r"] == round(result.value[pair_key], 3)
    assert row["ci_low"] == round(ci["ci_low"], 3)
    assert row["ci_high"] == round(ci["ci_high"], 3)
    assert row["label"] == "exploratory, unexplained"
    assert pair_key in result.notes["ci_excludes_zero_pairs"]
    assert "0.3 false exclusions" in result.notes["multiple_pairs_note"]
