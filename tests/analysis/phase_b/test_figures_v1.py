import pytest

from analysis.phase_b.figures import v1
from analysis.phase_b.reading import FrozenInputs


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _cell_runs(entity_id, engine, intent_id, n_total, n_mentioned):
    """n_total runs for this (entity, engine, intent) -- the first
    n_mentioned of them mention entity_id, the rest don't -- giving an
    exact mention_rate of n_mentioned / n_total."""
    raw_runs = []
    mentions = []
    for i in range(n_total):
        run_id = f"{entity_id}-{engine}-i{intent_id}-{i}"
        raw_runs.append(
            {
                "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": 1,
                "day_index": 1, "error": None, "raw_response_text": "x", "is_manual": None,
            }
        )
        if i < n_mentioned:
            mentions.append({"run_id": run_id, "surface_form": entity_id, "evidence_span": "x", "is_recommended": False, "votes": []})
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
    result = v1.compute(_inputs([], [], []))
    assert result.status == "untestable"


def test_engine_spread_zero_excluded_and_counted():
    entity_id = "flat"
    raw_runs, mentions = [], []
    # identical intent-1 mention_rate (1.0) in both engines -> engine_spread == 0
    for engine in ("engineA", "engineB"):
        rr, mm = _cell_runs(entity_id, engine, 1, 10, 10)
        raw_runs += rr
        mentions += mm
        # vertical cells present (intents 2/3/4) so the vertical-spread step
        # doesn't itself short-circuit before the engine-spread check runs
        for intent_id, n_mentioned in ((2, 0), (3, 5), (4, 0)):
            rr2, mm2 = _cell_runs(entity_id, engine, intent_id, 10, n_mentioned)
            raw_runs += rr2
            mentions += mm2
    entities = [_entity(entity_id, f"https://{entity_id}.com", entity_id)]

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    entry = next(e for e in result.notes["per_entity"] if e["entity_id"] == entity_id)
    assert entry["ratio"] is None
    assert entry["excluded_reason"] == "engine_spread_zero"
    assert result.notes["n_excluded_engine_spread_zero"] >= 1


def test_ratio_computed_from_vertical_and_engine_spread():
    entity_id = "spread"
    raw_runs, mentions = [], []
    for engine in ("engineA", "engineB"):
        # intent 1: 1.0 in engineA, 0.0 in engineB -> engine_spread = pstdev([1.0, 0.0]) = 0.5
        rate = 1.0 if engine == "engineA" else 0.0
        rr, mm = _cell_runs(entity_id, engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        # intents 2/3/4: 0.0 / 0.5 / 1.0 in both engines -> per-engine vertical SD = pstdev([0, 0.5, 1])
        for intent_id, r in ((2, 0.0), (3, 0.5), (4, 1.0)):
            rr2, mm2 = _cell_runs(entity_id, engine, intent_id, 10, int(r * 10))
            raw_runs += rr2
            mentions += mm2
    entities = [_entity(entity_id, f"https://{entity_id}.com", entity_id)]

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    entry = next(e for e in result.notes["per_entity"] if e["entity_id"] == entity_id)
    assert entry["ratio"] is not None
    # vertical_spread = pstdev([0, 0.5, 1]) ~= 0.4082, engine_spread = 0.5
    assert entry["ratio"] == pytest.approx(0.4082 / 0.5, abs=0.01)
    assert result.value == pytest.approx(entry["ratio"], abs=1e-9)  # single-company median == that company's ratio


def test_raw_spread_components_exposed_per_entity():
    entity_id = "spread"
    raw_runs, mentions = [], []
    for engine in ("engineA", "engineB"):
        rate = 1.0 if engine == "engineA" else 0.0
        rr, mm = _cell_runs(entity_id, engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        for intent_id, r in ((2, 0.0), (3, 0.5), (4, 1.0)):
            rr2, mm2 = _cell_runs(entity_id, engine, intent_id, 10, int(r * 10))
            raw_runs += rr2
            mentions += mm2
    entities = [_entity(entity_id, f"https://{entity_id}.com", entity_id)]

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    entry = next(e for e in result.notes["per_entity"] if e["entity_id"] == entity_id)
    assert entry["vertical_spread"] == pytest.approx(0.4082, abs=0.01)
    assert entry["engine_spread"] == pytest.approx(0.5, abs=1e-9)


def test_exclusion_sentence_reports_zero_mention_cause():
    entity_id = "neverNamed"
    raw_runs, mentions = [], []
    # qualifies via intent 3 (5 mentions in engineA, clears the >=5 floor)
    # but is never once mentioned in intent 1, in either engine -> engine
    # spread on intent 1 is 0 (both engines read exactly 0.0)
    for engine in ("engineA", "engineB"):
        rr, mm = _cell_runs(entity_id, engine, 1, 10, 0)
        raw_runs += rr
        mentions += mm
        for intent_id, n_mentioned in ((2, 0), (3, 5 if engine == "engineA" else 0), (4, 0)):
            rr2, mm2 = _cell_runs(entity_id, engine, intent_id, 10, n_mentioned)
            raw_runs += rr2
            mentions += mm2
    entities = [_entity(entity_id, f"https://{entity_id}.com", entity_id)]

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    entry = next(e for e in result.notes["per_entity"] if e["entity_id"] == entity_id)
    assert entry["excluded_reason"] == "engine_spread_zero"

    exclusion = result.notes["excluded_because_zero_mentions_intent1_all_engines"]
    assert exclusion["n_excluded_total"] == 1
    assert exclusion["n_all_zero"] == 1
    assert exclusion["n_nonzero_equal_rate"] == 0
    assert "never named in the general 'best' question on any engine" in exclusion["sentence"]


def test_zero_ratio_entities_always_counted_in_median():
    entity_id = "zeroratio"
    raw_runs, mentions = [], []
    # vertical spread 0 (identical 0.0 rate across intents 2/3/4 in both
    # engines) but a real, nonzero engine spread on intent 1 -> ratio == 0.0
    for engine, rate in (("engineA", 0.5), ("engineB", 0.1)):
        rr, mm = _cell_runs(entity_id, engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        for intent_id in (2, 3, 4):
            rr2, mm2 = _cell_runs(entity_id, engine, intent_id, 10, 0)
            raw_runs += rr2
            mentions += mm2
    entities = [_entity(entity_id, f"https://{entity_id}.com", entity_id)]

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    entry = next(e for e in result.notes["per_entity"] if e["entity_id"] == entity_id)
    assert entry["ratio"] == 0.0
    assert entity_id in result.notes["zero_ratio_median_check"]["entity_ids"]
    assert result.value == 0.0  # single-company median == that company's (zero) ratio


def test_zero_ratio_vertical_rates_and_zero_industry_detection():
    """Round 5 fix: for each zero-ratio company, print its actual
    dental/legal/home-services rates, and flag the ones where all three
    are 0.0 -- industry was absent from the data, not tested at a real
    (if identical) rate. Reuses the "zeroratio" fixture above, which has
    zero mentions in intents 2/3/4 on both engines."""
    entity_id = "zeroratio"
    raw_runs, mentions = [], []
    for engine, rate in (("engineA", 0.5), ("engineB", 0.1)):
        rr, mm = _cell_runs(entity_id, engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        for intent_id in (2, 3, 4):
            rr2, mm2 = _cell_runs(entity_id, engine, intent_id, 10, 0)
            raw_runs += rr2
            mentions += mm2
    entities = [_entity(entity_id, f"https://{entity_id}.com", entity_id)]

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    check = result.notes["zero_ratio_median_check"]
    assert check["vertical_rates_by_entity"][entity_id] == {"dental": 0.0, "legal": 0.0, "home services": 0.0}
    assert entity_id in check["all_zero_industry_entity_ids"]
    assert check["n_all_zero_industry"] == 1
    assert "industry was absent" in check["zero_industry_absent_sentence"]
    assert "asymmetrically" in result.notes["asymmetry_sentence"]

    [group] = check["rate_groups"]  # single company -> single group
    assert group == {
        "dental": 0.0, "legal": 0.0, "home_services": 0.0,
        "all_zero_industry": True, "entity_ids": [entity_id],
    }


def test_exploratory_median_excludes_only_zero_industry_group():
    """The EXPLORATORY median drops the 0/0/0 (industry-absent) group but
    keeps every other zero-ratio company, and never touches the
    pre-registered value/verdict."""
    raw_runs, mentions, entities = [], [], []
    # "zeroratio": 0/0/0 across verticals -> industry absent, dropped from
    # the exploratory median.
    for engine, rate in (("engineA", 0.5), ("engineB", 0.1)):
        rr, mm = _cell_runs("zeroratio", engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        for intent_id in (2, 3, 4):
            rr2, mm2 = _cell_runs("zeroratio", engine, intent_id, 10, 0)
            raw_runs += rr2
            mentions += mm2
    entities.append(_entity("zeroratio", "https://zeroratio.com", "zeroratio"))

    # "spread": a real, nonzero ratio -- always kept, in both medians.
    for engine in ("engineA", "engineB"):
        rate = 1.0 if engine == "engineA" else 0.0
        rr, mm = _cell_runs("spread", engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        for intent_id, r in ((2, 0.0), (3, 0.5), (4, 1.0)):
            rr2, mm2 = _cell_runs("spread", engine, intent_id, 10, int(r * 10))
            raw_runs += rr2
            mentions += mm2
    entities.append(_entity("spread", "https://spread.com", "spread"))

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    expl = result.notes["exploratory_median_excluding_zero_industry"]
    assert expl["is_exploratory"] is True
    assert expl["computed_post_data"] is True
    assert expl["n_excluded_zero_industry"] == 1
    assert expl["n_included"] == 1  # only "spread" survives the exclusion
    assert expl["median"] == pytest.approx(0.4082 / 0.5, abs=0.01)
    # pre-registered median is untouched: both companies' ratios (0.0 and
    # ~0.816) still contribute, so it differs from the exploratory one.
    assert result.value != pytest.approx(expl["median"])


def test_title_sentence_and_line_counts():
    raw_runs, mentions, entities = [], [], []
    # "above": engine spread 0.5 > vertical spread 0 -> ratio 0 < 1
    for engine, rate in (("engineA", 0.5), ("engineB", 0.1)):
        rr, mm = _cell_runs("above", engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        for intent_id in (2, 3, 4):
            rr2, mm2 = _cell_runs("above", engine, intent_id, 10, 0)
            raw_runs += rr2
            mentions += mm2
    entities.append(_entity("above", "https://above.com", "above"))

    # "below": vertical spread large, engine spread small but nonzero -> ratio > 1
    for engine in ("engineA", "engineB"):
        rate = 0.5 if engine == "engineA" else 0.4  # small, nonzero engine spread
        rr, mm = _cell_runs("below", engine, 1, 10, int(rate * 10))
        raw_runs += rr
        mentions += mm
        for intent_id, r in ((2, 0.0), (3, 0.5), (4, 1.0)):
            rr2, mm2 = _cell_runs("below", engine, intent_id, 10, int(r * 10))
            raw_runs += rr2
            mentions += mm2
    entities.append(_entity("below", "https://below.com", "below"))

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    assert result.notes["n_above_line_engine_matters_more"] == 1
    assert result.notes["n_below_line_industry_matters_more"] == 1
    assert "For 1 of 2 companies" in result.notes["title_sentence"]


def test_primary_includes_partial_qualifier_sensitivity_excludes_it():
    entity_id = "partial"
    raw_runs, mentions = [], []
    # engineA: 5 mentions total (intent 1 alone) -> clears the >=5 floor.
    # engineB: 1 mention total (intent 1 alone) -> below the floor.
    # intents 2/3/4 contribute zero mentions in both engines (vertical
    # spread ends up 0, which is a perfectly valid, non-excluded ratio),
    # so the eligibility gap comes only from intent 1 as intended.
    for engine, n_mentioned in (("engineA", 5), ("engineB", 1)):
        rr, mm = _cell_runs(entity_id, engine, 1, 10, n_mentioned)
        raw_runs += rr
        mentions += mm
        for intent_id in (2, 3, 4):
            rr2, mm2 = _cell_runs(entity_id, engine, intent_id, 10, 0)
            raw_runs += rr2
            mentions += mm2
    entities = [_entity(entity_id, f"https://{entity_id}.com", entity_id)]

    result = v1.compute(_inputs(raw_runs, mentions, entities))
    primary_ids = {e["entity_id"] for e in result.notes["per_entity"]}
    sensitivity_ids = {e["entity_id"] for e in result.notes["sensitivity_all_4_engines"]["per_entity"]}
    assert entity_id in primary_ids
    assert entity_id not in sensitivity_ids
