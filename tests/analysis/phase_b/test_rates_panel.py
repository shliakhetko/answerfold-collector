from analysis.phase_b.rates_panel import build_rates_panel
from analysis.phase_b.reading import FrozenInputs


def _mention(run_id, surface_form, is_recommended=False):
    return {
        "run_id": run_id, "surface_form": surface_form, "evidence_span": "x",
        "is_recommended": is_recommended, "votes": [],
    }


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _raw_run(run_id, *, engine="openai_chatgpt", intent_id=1, error=None, raw_response_text="some text"):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": 1,
        "day_index": 1, "error": error, "raw_response_text": raw_response_text, "is_manual": None,
    }


def _inputs(raw_runs, mentions_raw, entities, sources_by_run=None):
    return FrozenInputs(
        entities=entities,
        alt_domains_rows=[],
        entity_verification_rows=[],
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run=sources_by_run or {},
    )


def test_errored_run_excluded_from_n_runs():
    raw_runs = [_raw_run("run1"), _raw_run("run2", error="collection_error")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    cells, report = build_rates_panel(_inputs(raw_runs, mentions, entities))

    cell = next(c for c in cells if c.entity_id == "acme")
    assert cell.n_runs == 1  # run2 excluded (error), never counted in the denominator
    assert cell.n_mentioned == 1
    assert cell.mention_rate == 1.0


def test_manual_run_excluded_from_n_runs():
    raw_runs = [_raw_run("run1"), _raw_run("run2")]
    raw_runs[1]["is_manual"] = True
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    cells, _report = build_rates_panel(_inputs(raw_runs, mentions, entities))

    cell = next(c for c in cells if c.entity_id == "acme")
    assert cell.n_runs == 1


def test_branded_intent_7_excluded():
    raw_runs = [_raw_run("run1", intent_id=7)]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    cells, report = build_rates_panel(_inputs(raw_runs, mentions, entities))

    assert cells == []
    assert report.cells_produced == 0


def test_mention_and_recommendation_rate_computed_per_cell():
    raw_runs = [_raw_run(f"run{i}", engine="openai_chatgpt", intent_id=1) for i in range(1, 5)]
    mentions = [
        _mention("run1", "Acme", is_recommended=True),
        _mention("run2", "Acme", is_recommended=False),
    ]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    cells, _report = build_rates_panel(_inputs(raw_runs, mentions, entities))

    cell = next(c for c in cells if c.entity_id == "acme" and c.intent_id == 1)
    assert cell.n_runs == 4
    assert cell.n_mentioned == 2
    assert cell.n_recommended == 1
    assert cell.mention_rate == 0.5
    assert cell.recommendation_rate == 0.25


def test_qualifies_true_at_five_mentions_across_unbranded_intents():
    raw_runs = [_raw_run(f"run{i}", engine="openai_chatgpt", intent_id=i) for i in range(1, 6)]
    mentions = [_mention(f"run{i}", "Acme") for i in range(1, 6)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    cells, report = build_rates_panel(_inputs(raw_runs, mentions, entities))

    assert all(c.qualifies for c in cells if c.entity_id == "acme")
    assert report.qualifying_by_engine["openai_chatgpt"] == 1


def test_qualifies_false_below_floor():
    raw_runs = [_raw_run(f"run{i}", engine="openai_chatgpt", intent_id=i) for i in range(1, 4)]
    mentions = [_mention(f"run{i}", "Acme") for i in range(1, 4)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    cells, report = build_rates_panel(_inputs(raw_runs, mentions, entities))

    assert all(not c.qualifies for c in cells if c.entity_id == "acme")
    assert report.qualifying_by_engine["openai_chatgpt"] == 0


def test_qualifying_by_engine_reports_zero_for_engine_with_cells_but_no_qualifiers():
    raw_runs = [_raw_run("run1", engine="google_gemini", intent_id=1)]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    _cells, report = build_rates_panel(_inputs(raw_runs, mentions, entities))

    assert report.qualifying_by_engine == {"google_gemini": 0}
    assert report.engines_seen == ["google_gemini"]


def test_out_of_frame_entity_never_produces_a_cell():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "OutOfFrame")]
    entities = [_entity("oof", "https://oof.com", "OutOfFrame", meets_frame="out")]
    cells, _report = build_rates_panel(_inputs(raw_runs, mentions, entities))
    assert cells == []
