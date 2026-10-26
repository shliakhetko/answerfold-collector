from analysis.phase_b.daily_panel import (
    KNOWN_INCOMPLETE_CELLS,
    build_daily_panel,
    check_daily_completeness,
    company_day_rates,
)
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


def _qualifying_floor_runs(entity_id, engine="openai_chatgpt"):
    """5 usable runs across 5 distinct unbranded intents, day 1, each
    mentioning entity_id once -- the minimum that clears the >=5-mention
    eligibility floor (analysis.coding.b3_aggregates.ELIGIBILITY_FLOOR)."""
    raw_runs, mentions = [], []
    for intent_id in range(1, 6):
        run_id = f"{entity_id}-{engine}-floor-i{intent_id}"
        raw_runs.append(_raw_run(run_id, engine=engine, intent_id=intent_id, paraphrase_id=1, day_index=1, rep_index=1))
        mentions.append(_mention(run_id, entity_id))
    return raw_runs, mentions


def _grid_run(engine, intent_id, paraphrase_id, day_index, rep_index, *, error=None):
    run_id = f"{engine}-i{intent_id}-p{paraphrase_id}-d{day_index}-r{rep_index}"
    return _raw_run(
        run_id, engine=engine, intent_id=intent_id, paraphrase_id=paraphrase_id,
        day_index=day_index, rep_index=rep_index, error=error,
    )


# --- build_daily_panel ---

def test_qualifying_entity_produces_cells_with_full_grid_fields():
    raw_runs, mentions = _qualifying_floor_runs("acme")
    raw_runs.append(_raw_run("acme-d2", intent_id=2, paraphrase_id=3, day_index=2, rep_index=2))
    mentions.append(_mention("acme-d2", "acme"))
    entities = [_entity("acme", "https://acme.com", "acme")]

    cells, report = build_daily_panel(_inputs(raw_runs, mentions, entities))

    cell = next(c for c in cells if c.run_id == "acme-d2")
    assert cell.entity_id == "acme"
    assert cell.engine == "openai_chatgpt"
    assert cell.intent_id == 2
    assert cell.paraphrase_id == 3
    assert cell.day_index == 2
    assert cell.rep_index == 2
    assert cell.mentioned is True
    assert report.qualifying_by_engine["openai_chatgpt"] == 1


def test_non_qualifying_entity_produces_no_cells():
    # only 2 mentions -- below the 5-mention floor
    raw_runs = [_raw_run(f"r{i}", intent_id=i) for i in range(1, 3)]
    mentions = [_mention(f"r{i}", "acme") for i in range(1, 3)]
    entities = [_entity("acme", "https://acme.com", "acme")]

    cells, report = build_daily_panel(_inputs(raw_runs, mentions, entities))
    assert cells == []
    assert report.qualifying_by_engine == {}


def test_branded_intent_excluded_even_for_qualifying_entity():
    raw_runs, mentions = _qualifying_floor_runs("acme")
    raw_runs.append(_raw_run("acme-branded", intent_id=7, day_index=3))
    mentions.append(_mention("acme-branded", "acme"))
    entities = [_entity("acme", "https://acme.com", "acme")]

    cells, _report = build_daily_panel(_inputs(raw_runs, mentions, entities))
    assert all(c.intent_id != 7 for c in cells)
    assert not any(c.run_id == "acme-branded" for c in cells)


def test_errored_run_absent_never_imputed():
    raw_runs, mentions = _qualifying_floor_runs("acme")
    raw_runs.append(_raw_run("acme-errored", intent_id=2, day_index=4, error="collection_error"))
    entities = [_entity("acme", "https://acme.com", "acme")]

    cells, _report = build_daily_panel(_inputs(raw_runs, mentions, entities))
    assert not any(c.run_id == "acme-errored" for c in cells)
    assert not any(c.day_index == 4 for c in cells)  # day 4 is simply absent, never backfilled


# --- check_daily_completeness ---

def test_complete_grid_has_no_gaps():
    raw_runs = [
        _grid_run("openai_chatgpt", 1, 1, day, rep)
        for day in range(1, 3)
        for rep in range(1, 4)
    ]
    report = check_daily_completeness(raw_runs)
    assert report.is_complete
    assert report.known_incomplete == []
    assert report.unwhitelisted_missing == []
    assert report.total_cells_expected == 2


def test_unwhitelisted_gap_fails_completeness():
    raw_runs = [_grid_run("openai_chatgpt", 1, 1, 1, rep) for rep in range(1, 4)]
    raw_runs += [_grid_run("openai_chatgpt", 1, 1, 2, rep) for rep in (1, 2)]  # only 2 reps, day 2

    report = check_daily_completeness(raw_runs)
    assert not report.is_complete
    assert len(report.unwhitelisted_missing) == 1
    gap = report.unwhitelisted_missing[0]
    assert gap["day_index"] == 2
    assert gap["usable_reps"] == 2


def test_known_gap_is_whitelisted_not_a_gate_failure():
    engine, intent_id, paraphrase_id, day_index = "google_gemini", 6, 4, 7
    known = KNOWN_INCOMPLETE_CELLS[(engine, intent_id, paraphrase_id, day_index)]
    assert known["usable_reps"] == 2
    raw_runs = [_grid_run(engine, intent_id, paraphrase_id, day_index, rep) for rep in (1, 2)]
    raw_runs.append(_grid_run(engine, intent_id, paraphrase_id, day_index, 3, error="500 INTERNAL"))

    report = check_daily_completeness(raw_runs)
    assert report.is_complete  # a known gap does not fail the gate
    assert report.unwhitelisted_missing == []
    assert len(report.known_incomplete) == 1
    assert report.known_incomplete[0]["run_ids"] == known["run_ids"]


def test_mismatched_rep_count_at_a_known_cell_is_not_whitelisted():
    # the known cell is documented at 2 usable reps -- a DIFFERENT count at
    # that same (engine, intent, paraphrase, day) is a new, unexplained fact
    # and must not be silently absorbed by the existing whitelist entry.
    engine, intent_id, paraphrase_id, day_index = "google_gemini", 6, 4, 7
    raw_runs = [_grid_run(engine, intent_id, paraphrase_id, day_index, 1)]  # only 1 usable rep

    report = check_daily_completeness(raw_runs)
    assert not report.is_complete
    assert len(report.unwhitelisted_missing) == 1


# --- company_day_rates ---

def test_company_day_rates_pools_across_intent_and_paraphrase():
    raw_runs, mentions = _qualifying_floor_runs("acme")  # 5 mentioned runs, day 1
    raw_runs.append(_raw_run("acme-not-mentioned", intent_id=6, paraphrase_id=2, day_index=1, rep_index=1))
    entities = [_entity("acme", "https://acme.com", "acme")]

    cells, _report = build_daily_panel(_inputs(raw_runs, mentions, entities))
    rates = company_day_rates(cells)
    n_mentioned, n_total = rates[("acme", "openai_chatgpt", 1)]
    assert n_mentioned == 5
    assert n_total == 6
