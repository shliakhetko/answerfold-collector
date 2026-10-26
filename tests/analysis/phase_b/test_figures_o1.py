import pytest

from analysis.phase_b.figures import o1
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


def _day_rate_runs(entity_id, engine, day_index, n_mentioned, *, reps=10):
    raw_runs, mentions = [], []
    for rep_index in range(1, reps + 1):
        run_id = f"{entity_id}-{engine}-d{day_index}-r{rep_index}"
        raw_runs.append(_raw_run(run_id, engine=engine, day_index=day_index, rep_index=rep_index))
        if rep_index <= n_mentioned:
            mentions.append(_mention(run_id, entity_id))
    return raw_runs, mentions


def test_zero_cells_is_untestable():
    result = o1.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_engine_below_head_size_is_untestable():
    engine = "openai_chatgpt"
    raw_runs, mentions = _day_rate_runs("acme", engine, 1, 10)
    entities = [_entity("acme", "https://acme.com", "acme")]

    result = o1.compute(_inputs(raw_runs, mentions, entities))
    assert result.value[engine] is None
    assert engine in result.notes["untestable_engines"]
    assert "13" in result.notes["untestable_engines"][engine]


def test_fixed_head_set_gives_perfect_head_jaccard(monkeypatch):
    monkeypatch.setattr(o1, "HEAD_SIZE", 2)
    engine = "openai_chatgpt"
    # h1/h2 always rank highest, every day, out of 6 qualifying companies.
    entity_ids = ["h1", "h2", "t1", "t2", "t3", "t4"]
    raw_runs, mentions, entities = [], [], []
    for day_index in range(1, 8):
        for entity_id, n_mentioned in [("h1", 10), ("h2", 9), ("t1", 4), ("t2", 3), ("t3", 2), ("t4", 1)]:
            rr, mm = _day_rate_runs(entity_id, engine, day_index, n_mentioned)
            raw_runs += rr
            mentions += mm
    for entity_id in entity_ids:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = o1.compute(_inputs(raw_runs, mentions, entities))
    assert result.value[engine]["head_mean_jaccard"] == pytest.approx(1.0)
    for day_index in range(1, 8):
        assert set(result.notes["head_sets_by_day"][engine][day_index]) == {"h1", "h2"}


def test_tail_fully_rotates_to_zero_jaccard(monkeypatch):
    monkeypatch.setattr(o1, "HEAD_SIZE", 2)
    engine = "openai_chatgpt"
    entity_ids = ["c1", "c2", "c3", "c4", "c5", "c6"]
    raw_runs, mentions, entities = [], [], []
    # Day 1: c1..c3 rank highest, c4..c6 rank lowest (tail).
    day1_counts = {"c1": 10, "c2": 9, "c3": 8, "c4": 3, "c5": 2, "c6": 1}
    # Day 2: fully reversed -- c4..c6 now rank highest, c1..c3 now the tail.
    day2_counts = {"c4": 10, "c5": 9, "c6": 8, "c1": 3, "c2": 2, "c3": 1}
    for day_index, counts in [(1, day1_counts), (2, day2_counts)]:
        for entity_id, n_mentioned in counts.items():
            rr, mm = _day_rate_runs(entity_id, engine, day_index, n_mentioned)
            raw_runs += rr
            mentions += mm
    for entity_id in entity_ids:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = o1.compute(_inputs(raw_runs, mentions, entities))
    # n_qualifying=6 -> tail_size=3. Day1 tail={c4,c5,c6}, day2 tail={c1,c2,c3} -- disjoint.
    assert result.notes["n_tail_by_engine"][engine] == 3
    assert set(result.notes["tail_sets_by_day"][engine][1]) == {"c4", "c5", "c6"}
    assert set(result.notes["tail_sets_by_day"][engine][2]) == {"c1", "c2", "c3"}
    assert result.value[engine]["tail_mean_jaccard"] == pytest.approx(0.0)


def _big_qualifying_pool(engine, n_entities=20, days=range(1, 8)):
    entity_ids = [f"co{i}" for i in range(n_entities)]
    raw_runs, mentions, entities = [], [], []
    for day_index in days:
        for i, entity_id in enumerate(entity_ids):
            n_mentioned = (i * 3 + day_index) % 11  # varied, non-degenerate ranking
            rr, mm = _day_rate_runs(entity_id, engine, day_index, n_mentioned)
            raw_runs += rr
            mentions += mm
    for entity_id in entity_ids:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    return raw_runs, mentions, entities


def test_permutation_null_deterministic_given_fixed_seed():
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = _big_qualifying_pool(engine)
    inputs = _inputs(raw_runs, mentions, entities)

    result1 = o1.compute(inputs)
    result2 = o1.compute(inputs)
    null1 = result1.notes["permutation_null"][engine]["head"]
    null2 = result2.notes["permutation_null"][engine]["head"]
    assert null1["null_mean_jaccard"] == null2["null_mean_jaccard"]
    assert null1["n_iterations"] == o1.N_PERMUTATION_ITERATIONS


def test_permutation_null_head_and_tail_differ_when_k_differs():
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = _big_qualifying_pool(engine)  # n_qual=20 -> head=13, tail=10
    result = o1.compute(_inputs(raw_runs, mentions, entities))
    null = result.notes["permutation_null"][engine]
    assert null["head"]["n_iterations"] == o1.N_PERMUTATION_ITERATIONS
    assert null["tail"]["n_iterations"] == o1.N_PERMUTATION_ITERATIONS
    assert null["head"]["null_mean_jaccard"] != null["tail"]["null_mean_jaccard"]


def test_tie_diagnostic_detects_boundary_tie_block(monkeypatch):
    monkeypatch.setattr(o1, "HEAD_SIZE", 2)
    engine = "openai_chatgpt"
    counts = {"h1": 10, "h2": 5, "h3": 5, "t1": 3, "t2": 2, "t3": 1}  # h2/h3 tied at the rank-2 boundary
    raw_runs, mentions, entities = [], [], []
    for day_index in (1, 2):
        for entity_id, n_mentioned in counts.items():
            rr, mm = _day_rate_runs(entity_id, engine, day_index, n_mentioned)
            raw_runs += rr
            mentions += mm
    for entity_id in counts:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = o1.compute(_inputs(raw_runs, mentions, entities))
    tie = result.notes["tie_diagnostics"][engine][1]["head"]
    assert tie["tie_block_size"] == 2
    # rate denominator includes the run x qualifying-entity cross product
    # (6 entities x 10 reps sharing the same intent/paraphrase/day bucket),
    # so h2/h3's rate is 5/60, not 5/10 -- the tie itself is what's asserted.
    assert tie["boundary_rate"] == pytest.approx(5 / 60)
    assert result.notes["tie_diagnostics_summary"][engine]["days_with_head_tie_at_boundary"] == 2


def test_raw_head_vs_tail_is_descriptive_not_a_label(monkeypatch):
    monkeypatch.setattr(o1, "HEAD_SIZE", 2)
    engine = "openai_chatgpt"
    # tail ({t1,t2,t3}) is fixed every day (jaccard 1.0); head alternates
    # between {h1,h2} and {h1,h3} every day (jaccard 1/3) -- the literal
    # reverse of "stable head, churning tail".
    odd_counts = {"h1": 10, "h2": 8, "h3": 6, "t1": 3, "t2": 2, "t3": 1}
    even_counts = {"h1": 10, "h3": 8, "h2": 6, "t1": 3, "t2": 2, "t3": 1}
    raw_runs, mentions, entities = [], [], []
    for day_index in range(1, 8):
        counts = odd_counts if day_index % 2 == 1 else even_counts
        for entity_id, n_mentioned in counts.items():
            rr, mm = _day_rate_runs(entity_id, engine, day_index, n_mentioned)
            raw_runs += rr
            mentions += mm
    for entity_id in odd_counts:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = o1.compute(_inputs(raw_runs, mentions, entities))
    assert result.value[engine]["tail_mean_jaccard"] == pytest.approx(1.0)
    assert result.value[engine]["head_mean_jaccard"] == pytest.approx(1 / 3)
    # raw head-vs-tail is descriptive only; the label comes from the noise null
    assert result.notes["raw_head_vs_tail_descriptive"][engine]["direction"] == "tail_higher"
    assert result.notes["interpretation_band"][engine] != "tail_more_stable_than_head"


def test_tie_break_average_deterministic_given_fixed_seed():
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = _big_qualifying_pool(engine)
    inputs = _inputs(raw_runs, mentions, entities)

    result1 = o1.compute(inputs)
    result2 = o1.compute(inputs)
    assert result1.value[engine]["head_mean_jaccard"] == result2.value[engine]["head_mean_jaccard"]
    assert result1.value[engine]["tail_mean_jaccard"] == result2.value[engine]["tail_mean_jaccard"]


def test_random_tie_break_lowers_jaccard_vs_alphabetical(monkeypatch):
    monkeypatch.setattr(o1, "HEAD_SIZE", 2)
    engine = "openai_chatgpt"
    # h1 always rank 1. a/b/c are tied with each other (identical rate,
    # unchanged across both days) for the remaining head slot. The OLD
    # alphabetical tie-break would pick "a" every day -> Jaccard 1.0,
    # falsely reporting "perfectly stable". Random tie-break should average
    # well below 1.0: E[Jaccard] = (1/3)*1.0 + (2/3)*(1/3) = 5/9 when the
    # two days' picks are independent uniform draws over {a,b,c}.
    counts = {"h1": 10, "a": 5, "b": 5, "c": 5}
    raw_runs, mentions, entities = [], [], []
    for day_index in (1, 2):
        for entity_id, n_mentioned in counts.items():
            rr, mm = _day_rate_runs(entity_id, engine, day_index, n_mentioned)
            raw_runs += rr
            mentions += mm
    for entity_id in counts:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = o1.compute(_inputs(raw_runs, mentions, entities))
    assert result.value[engine]["head_mean_jaccard"] == pytest.approx(5 / 9, abs=0.05)
    assert result.value[engine]["head_mean_jaccard"] < 0.9


def test_noise_only_null_deterministic_given_fixed_seed():
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = _big_qualifying_pool(engine)
    inputs = _inputs(raw_runs, mentions, entities)

    result1 = o1.compute(inputs)
    result2 = o1.compute(inputs)
    null1 = result1.notes["noise_only_null"][engine]["head"]
    null2 = result2.notes["noise_only_null"][engine]["head"]
    assert null1["null_mean_jaccard"] == null2["null_mean_jaccard"]
    assert null1["n_iterations"] == o1.N_NOISE_NULL_REPLICATES


def test_noise_only_null_flags_real_drift_as_a_low_percentile(monkeypatch):
    # A fully rotating tail (round 2's own fixture) is far more churn than
    # sampling noise around a fixed pooled rate would ever produce -- the
    # observed tail Jaccard should sit at the LOW end of the noise-only
    # null (real drift beyond noise), not the middle.
    monkeypatch.setattr(o1, "HEAD_SIZE", 2)
    engine = "openai_chatgpt"
    entity_ids = ["c1", "c2", "c3", "c4", "c5", "c6"]
    day1_counts = {"c1": 10, "c2": 9, "c3": 8, "c4": 3, "c5": 2, "c6": 1}
    day2_counts = {"c4": 10, "c5": 9, "c6": 8, "c1": 3, "c2": 2, "c3": 1}
    raw_runs, mentions, entities = [], [], []
    for day_index, counts in [(1, day1_counts), (2, day2_counts)]:
        for entity_id, n_mentioned in counts.items():
            rr, mm = _day_rate_runs(entity_id, engine, day_index, n_mentioned)
            raw_runs += rr
            mentions += mm
    for entity_id in entity_ids:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = o1.compute(_inputs(raw_runs, mentions, entities))
    tail_null = result.notes["noise_only_null"][engine]["tail"]
    assert tail_null["observed_percentile_in_null"] < 0.1


def test_label_rule_uses_noise_null_5th_95th_band():
    null = {"null_p05_jaccard": 0.30, "null_p95_jaccard": 0.50}
    assert o1._noise_label(0.40, null) == "consistent_with_noise"
    assert o1._noise_label(0.30, null) == "consistent_with_noise"
    assert o1._noise_label(0.50, null) == "consistent_with_noise"
    assert o1._noise_label(0.51, null) == "above_noise"
    assert o1._noise_label(0.29, null) == "below_noise"
    assert o1._noise_label(None, null) is None
    assert o1._engine_label("consistent_with_noise", "consistent_with_noise") == "consistent_with_noise"
    assert o1._engine_label("above_noise", "consistent_with_noise") == "head_above_noise__tail_consistent_with_noise"


def test_engine_label_matches_per_side_noise_labels():
    engine = "openai_chatgpt"
    raw_runs, mentions, entities = _big_qualifying_pool(engine)
    result = o1.compute(_inputs(raw_runs, mentions, entities))
    sides = result.notes["noise_labels_by_engine"][engine]
    assert result.notes["interpretation_band"][engine] == o1._engine_label(sides["head"], sides["tail"])
    assert "null_p05_jaccard" in result.notes["noise_only_null"][engine]["head"]
