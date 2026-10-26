import pytest

from analysis.phase_b.figures import a3
from analysis.phase_b.reading import FrozenInputs


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _cell_runs(entity_id, engine, intent_id, n_total, n_mentioned, n_recommended):
    """n_total runs; the first n_mentioned mention entity_id, and the
    first n_recommended of those mentions are also flagged recommended."""
    assert n_recommended <= n_mentioned <= n_total
    raw_runs, mentions = [], []
    for i in range(n_total):
        run_id = f"{entity_id}-{engine}-i{intent_id}-{i}"
        raw_runs.append(
            {
                "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": 1,
                "day_index": 1, "error": None, "raw_response_text": "x", "is_manual": None,
            }
        )
        if i < n_mentioned:
            mentions.append(
                {
                    "run_id": run_id, "surface_form": entity_id, "evidence_span": "x",
                    "is_recommended": i < n_recommended, "votes": [],
                }
            )
    return raw_runs, mentions


def _inputs(raw_runs, mentions_raw, entities):
    return FrozenInputs(
        entities=entities,
        alt_domains_rows=[],
        entity_verification_rows=[],
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run={},
    )


def test_zero_cells_is_untestable():
    result = a3.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_fewer_than_min_qualifying_companies_marks_engine_untestable():
    # only 1 qualifying company in this engine
    raw_runs, mentions = _cell_runs("acme", "openai_chatgpt", 1, 10, 5, 5)
    entities = [_entity("acme", "https://acme.com", "acme")]
    result = a3.compute(_inputs(raw_runs, mentions, entities))
    assert result.value["openai_chatgpt"] is None
    assert "openai_chatgpt" in result.notes["untestable_engines"]


def test_perfect_correlation_when_mention_and_recommendation_rank_match():
    raw_runs, mentions, entities = [], [], []
    for i in range(1, 6):
        entity_id = f"e{i}"
        n_mentioned = i * 10  # 10, 20, 30, 40, 50 -- strictly increasing, no ties
        rr, mm = _cell_runs(entity_id, "openai_chatgpt", 1, 100, n_mentioned, n_mentioned)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a3.compute(_inputs(raw_runs, mentions, entities))
    assert result.value["openai_chatgpt"] == pytest.approx(1.0)
    assert result.notes["teardown_by_engine"]["openai_chatgpt"] == []


def test_teardown_flags_company_with_large_rank_gap():
    raw_runs, mentions, entities = [], [], []
    # 11 companies with strictly decreasing mention counts (clean, distinct ranks)
    mentioned_counts = [92, 84, 76, 68, 60, 52, 44, 36, 28, 20, 12]
    for i, n_mentioned in enumerate(mentioned_counts, start=1):
        entity_id = f"e{i}"
        # e1 (highest mention count, rank 1) is never recommended -> recommendation rank last
        n_recommended = 0 if i == 1 else n_mentioned
        rr, mm = _cell_runs(entity_id, "openai_chatgpt", 1, 100, n_mentioned, n_recommended)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a3.compute(_inputs(raw_runs, mentions, entities))
    teardown_ids = {t["entity_id"] for t in result.notes["teardown_by_engine"]["openai_chatgpt"]}
    assert "e1" in teardown_ids
    e1 = next(t for t in result.notes["teardown_by_engine"]["openai_chatgpt"] if t["entity_id"] == "e1")
    assert e1["mention_rank"] == 1
    assert e1["gap"] >= 10


def test_teardown_is_one_directional_recommended_more_than_mentioned_not_flagged():
    """2026-09-25 fix: a company recommended MORE than it's mentioned, in
    rank terms (recommendation rank better than mention rank), is the
    opposite failure mode and must never be flagged, even with a large
    gap -- only the reverse (mentioned much more than recommended) is."""
    raw_runs, mentions, entities = [], [], []
    mentioned_counts = [92, 84, 76, 68, 60, 52, 44, 36, 28, 20]  # e1-e10 only, all > e11's 12
    for i, n_mentioned in enumerate(mentioned_counts, start=1):
        entity_id = f"e{i}"
        # e1-e10 each recommended only once (a 10-way tie on the
        # recommendation axis, all much lower than e11's).
        rr, mm = _cell_runs(entity_id, "openai_chatgpt", 1, 100, n_mentioned, 1)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    # e11: fewest mentions (worst mention rank, 11th) but recommended every
    # time it's mentioned -- the highest recommendation rate of the group,
    # so its recommendation rank is the best (1st) despite its mention
    # rank being the worst.
    rr, mm = _cell_runs("e11", "openai_chatgpt", 1, 100, 12, 12)
    raw_runs += rr
    mentions += mm
    entities.append(_entity("e11", "https://e11.com", "e11"))

    result = a3.compute(_inputs(raw_runs, mentions, entities))
    e11 = next(t for t in result.notes["scatter_points_by_engine"]["openai_chatgpt"] if t["entity_id"] == "e11")
    assert e11["mention_rank"] == 11
    assert e11["recommendation_rank"] == 1
    teardown_ids = {t["entity_id"] for t in result.notes["teardown_by_engine"]["openai_chatgpt"]}
    assert "e11" not in teardown_ids


def test_bottom_rank_tie_check_present_and_correct():
    raw_runs, mentions, entities = [], [], []
    # e1/e2/e3 tied at the worst mention rank (same mention count each);
    # e4 has a unique best mention rank. Recommendation counts (0/1/2/3)
    # are all distinct, so there are zero ties on that axis.
    for entity_id, n_mentioned, n_recommended in [
        ("e1", 5, 0), ("e2", 5, 1), ("e3", 5, 2), ("e4", 40, 3),
    ]:
        rr, mm = _cell_runs(entity_id, "openai_chatgpt", 1, 100, n_mentioned, n_recommended)
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))

    result = a3.compute(_inputs(raw_runs, mentions, entities))
    tie_check = result.notes["bottom_rank_tie_check"]["openai_chatgpt"]
    assert tie_check["n_tied_at_bottom_mention"] == 3
    assert tie_check["n_tied_at_bottom_recommendation"] == 1
    assert tie_check["max_possible_tie_only_distortion"] == pytest.approx(1.0)  # (3-1)/2
    assert tie_check["ties_alone_can_reach_threshold"] is False
