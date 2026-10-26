import pytest

from analysis.phase_b.figures import a1
from analysis.phase_b.reading import FrozenInputs


def _mention(run_id, surface_form):
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": False, "votes": []}


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _raw_run(run_id, *, engine="openai_chatgpt", intent_id=1, error=None, raw_response_text="some text"):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": 1,
        "day_index": 1, "error": error, "raw_response_text": raw_response_text, "is_manual": None,
    }


def _inputs(raw_runs, mentions_raw, entities, serp_raw_rows=()):
    return FrozenInputs(
        entities=entities,
        alt_domains_rows=[],
        entity_verification_rows=[],
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run={},
        serp_raw_rows=list(serp_raw_rows),
    )


def _correlated_fixture(engine="openai_chatgpt"):
    """5 qualifying companies (both groups clear the >=5-mention
    eligibility floor), all scoped to intent 1 only. 2 are top-10 for
    intent 1 (2/5 = 40%, inside the 20-80% band) and mentioned in 20 of 25
    intent-1 runs (rate 0.8). The other 3 are not top-10 and mentioned in
    5 of 25 intent-1 runs (rate 0.2) -- a clean, strongly correlated
    separation with a non-zero mean for both groups, so the top10-vs-not
    ratio is well defined."""
    raw_runs, mentions, entities, serp_rows = [], [], [], []
    top10_ids = ["a", "b"]
    not_top10_ids = ["c", "d", "e"]
    for entity_id in top10_ids + not_top10_ids:
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
        n_mentioned = 20 if entity_id in top10_ids else 5
        for i in range(n_mentioned):
            run_id = f"{entity_id}-mention-{i}"
            raw_runs.append(_raw_run(run_id, engine=engine, intent_id=1))
            mentions.append(_mention(run_id, entity_id))
        for i in range(25 - n_mentioned):
            raw_runs.append(_raw_run(f"{entity_id}-filler-{i}", engine=engine, intent_id=1))
        if entity_id in top10_ids:
            serp_rows.append({"intent_id": 1, "position": 1, "url": f"https://{entity_id}.com"})
    return raw_runs, mentions, entities, serp_rows


def test_intent_excluded_outside_20_80_band():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    # add a third intent where every company is top-10 (100% -> excluded)
    for entity_id in ("a", "b", "c", "d", "e"):
        raw_runs.append(_raw_run(f"{entity_id}-filler-3", intent_id=3))
        serp_rows.append({"intent_id": 3, "position": 1, "url": f"https://{entity_id}.com"})

    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    excluded_ids = {e["intent_id"] for e in result.notes["excluded_intents_by_engine"]["openai_chatgpt"]}
    included_ids = set(result.notes["included_intents_by_engine"]["openai_chatgpt"])
    assert 3 in excluded_ids  # 100% top-10 -> outside the band
    assert 1 in included_ids  # 40% top-10 -> inside the band


def test_serp_collection_limitation_present_on_ok_result():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    assert result.status == "ok"
    assert "SERP data was collected 2026-09-06" in result.notes["serp_collection_limitation"]


def test_no_serp_data_at_all_is_untestable_not_a_crash():
    result = a1.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_point_biserial_r_reflects_the_fixtures_correlation():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    r = result.value["openai_chatgpt"]
    assert r is not None
    assert r > 0.8  # top-10 companies are mentioned in intent 1, never intent 2, and vice versa
    assert result.notes["verdict_by_engine"]["openai_chatgpt"] == "falsified"  # r >= 0.5


def test_top10_vs_not_mean_and_ratio_reported():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    stats = result.notes["top10_vs_not_by_engine"]["openai_chatgpt"]
    assert stats["mean_top10"] is not None
    assert stats["mean_not"] is not None
    assert stats["ratio"] is not None


def test_structural_ceiling_includes_highest_observed_fraction():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    ceiling = result.notes["structural_ceiling_by_engine"]["openai_chatgpt"]
    # only intent 1 has data in this fixture, top10 fraction there is 2/5 = 40%
    assert ceiling["highest_observed_fraction"] == pytest.approx(0.4)


def test_gap_beyond_slot_limit_field_removed():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    assert "gap_beyond_slot_limit_by_engine" not in result.notes


def test_intent_slot_detail_includes_name_and_non_vendor_count():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    serp_rows.append({"intent_id": 1, "position": 2, "url": "https://unrelated-blog.com"})
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    detail = result.notes["intent_top10_slot_detail"][1]
    assert detail["intent_name"] == "general"
    # 2 raw slots seeded for intent 1 in this fixture (one per top10 company)
    # plus the unrelated-blog.com row just added -> 3 raw slots, 2 matched,
    # 1 non-vendor
    assert detail["n_raw_slots"] == 3
    assert detail["n_slots_matched_any_in_frame_entity"] == 2
    assert detail["n_slots_non_vendor"] == 1


def test_exploratory_entity_ids_listed():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    entity_ids = result.notes["exploratory_top10_vs_not"]["openai_chatgpt"]["entity_ids"]
    assert entity_ids == ["a", "b"]


def test_exploratory_top10_rates_are_each_pairs_own_rate_not_the_mean_repeated():
    """Round 5 fix: the raw per-pair rates used to be dropped before
    reaching the seam file, and the render script plotted the group mean
    cloned once per pair as fake "dots". mention_rate's denominator is the
    shared run pool for the intent/engine (build_run_company_panel crosses
    every run with every entity), not a per-entity run count, so a and b
    share one 25-run pool here: a mentioned in 20 of them (rate 0.8), b in
    10 (rate 0.4), mean 0.6. A fixed pair of identical 0.6 values would mean
    the real per-pair rates were lost again."""
    raw_runs, mentions, entities, serp_rows = [], [], [], []
    for i in range(25):
        raw_runs.append(_raw_run(f"run-{i}", intent_id=1))
    for i in range(20):
        mentions.append(_mention(f"run-{i}", "a"))
    for i in range(10):
        mentions.append(_mention(f"run-{i}", "b"))
    for i in range(5):
        mentions.append(_mention(f"run-{i}", "c"))  # not top-10, just clears the eligibility floor
    for entity_id in ("a", "b", "c"):
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    serp_rows.append({"intent_id": 1, "position": 1, "url": "https://a.com"})
    serp_rows.append({"intent_id": 1, "position": 2, "url": "https://b.com"})

    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    expl = result.notes["exploratory_top10_vs_not"]["openai_chatgpt"]
    assert sorted(expl["top10_rates"]) == pytest.approx([0.4, 0.8])
    assert expl["mean_top10"] == pytest.approx(0.6)
    # the bug this test guards against: every dot equal to the mean
    assert len(set(expl["top10_rates"])) > 1


def test_confirmatory_summary_sentence_reflects_testability():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    assert "testable in 1 of 1 engines" in result.notes["confirmatory_summary_sentence"]

    # zero qualifying companies -> untestable, wording differs
    result2 = a1.compute(_inputs([_raw_run("run1")], [_mention("run1", "acme")], [_entity("acme", "https://acme.com", "acme")]))
    assert "untestable in all 1 engines" in result2.notes["confirmatory_summary_sentence"]


def test_engine_with_zero_qualifying_companies_marked_untestable_others_still_computed():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture(engine="openai_chatgpt")
    # a second engine with a company that never clears the eligibility floor
    raw_runs.append(_raw_run("gemini-run", engine="google_gemini", intent_id=1))
    mentions.append(_mention("gemini-run", "a"))

    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    assert result.status == "ok"
    assert result.value["openai_chatgpt"] is not None
    assert result.value["google_gemini"] is None
    assert "google_gemini" in result.notes["untestable_engines"]


def test_theoretical_ceiling_matches_slots_over_n_qualifying():
    # 5 qualifying companies in the fixture -> min(10, 5) / 5 == 1.0
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    ceiling = result.notes["structural_ceiling_by_engine"]["openai_chatgpt"]
    assert ceiling["n_qualifying"] == 5
    assert ceiling["max_slots"] == 10
    assert ceiling["theoretical_max_fraction"] == 1.0
    assert "maximum possible top-10 fraction is 100" in ceiling["sentence"]


def test_duplicate_company_slots_detected():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    # two of Google's top-10 slots for intent 1 both belong to company "a"
    serp_rows += [
        {"intent_id": 1, "position": 2, "url": "https://a.com/page1"},
        {"intent_id": 1, "position": 3, "url": "https://a.com/page2"},
    ]
    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    detail = result.notes["intent_top10_slot_detail"][1]
    # _correlated_fixture already seeds one slot each for "a" and "b" (both
    # top-10 for intent 1); the two new slots are additional "a" pages, so
    # 4 total matched slots but only 2 distinct companies (a, b)
    assert detail["n_slots_matched_any_in_frame_entity"] == 4
    assert detail["n_distinct_companies_matched"] == 2
    assert detail["n_duplicate_company_slots"] == 2


def test_exploratory_stats_pool_intents_the_confirmatory_calc_excludes():
    raw_runs, mentions, entities, serp_rows = _correlated_fixture()
    # intent 3: every company is top-10 (100% -> excluded from the confirmatory calc)
    for entity_id in ("a", "b", "c", "d", "e"):
        raw_runs.append(_raw_run(f"{entity_id}-filler-3", intent_id=3))
        serp_rows.append({"intent_id": 3, "position": 1, "url": f"https://{entity_id}.com"})

    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    assert 3 not in result.notes["included_intents_by_engine"]["openai_chatgpt"]

    expl = result.notes["exploratory_top10_vs_not"]["openai_chatgpt"]
    n_expl_total = expl["n_top10"] + expl["n_not_top10"]
    # confirmatory only pools included intents (just intent 1); exploratory
    # pools every unbranded intent with data (intents 1 and 3), so it must
    # cover strictly more (entity, intent) pairs
    assert n_expl_total > result.n_included["openai_chatgpt"]


def test_untestable_message_names_structural_ceiling_when_zero_intents_included():
    # 1 qualifying company only -> every intent's top-10 fraction is 0% or
    # 100%, never inside 20-80%, so included_intents is always empty
    raw_runs, mentions = _cell_runs_helper("solo", "openai_chatgpt", 1, 10, 5)
    entities = [_entity("solo", "https://solo.com", "solo")]
    serp_rows = [{"intent_id": 1, "position": 1, "url": "https://solo.com"}]

    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    assert result.notes["included_intents_by_engine"]["openai_chatgpt"] == []
    msg = result.notes["untestable_engines"]["openai_chatgpt"]
    assert "structural ceiling" in msg
    assert "untestable by design" in msg


def test_untestable_message_stays_generic_for_degenerate_variance_with_included_intents():
    # 5 companies, 2 top-10 for intent 1 (40% -> included), but every
    # company has the IDENTICAL mention_rate -> zero variance in y, a
    # different failure mode from the structural-ceiling case above
    raw_runs, mentions, entities, serp_rows = [], [], [], []
    for entity_id in ("a", "b", "c", "d", "e"):
        rr, mm = _cell_runs_helper(entity_id, "openai_chatgpt", 1, 20, 10)  # every company: rate 0.5
        raw_runs += rr
        mentions += mm
        entities.append(_entity(entity_id, f"https://{entity_id}.com", entity_id))
    for entity_id in ("a", "b"):
        serp_rows.append({"intent_id": 1, "position": 1, "url": f"https://{entity_id}.com"})

    result = a1.compute(_inputs(raw_runs, mentions, entities, serp_rows))
    assert 1 in result.notes["included_intents_by_engine"]["openai_chatgpt"]
    msg = result.notes["untestable_engines"]["openai_chatgpt"]
    assert "insufficient variation to compute a correlation" in msg
    assert "structural ceiling" not in msg


def _cell_runs_helper(entity_id, engine, intent_id, n_total, n_mentioned):
    raw_runs, mentions = [], []
    for i in range(n_total):
        run_id = f"{entity_id}-{engine}-i{intent_id}-{i}"
        raw_runs.append(_raw_run(run_id, engine=engine, intent_id=intent_id))
        if i < n_mentioned:
            mentions.append(_mention(run_id, entity_id))
    return raw_runs, mentions
