from analysis.phase_b.figures import s2
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


def _source(url):
    return {"url_raw": url, "domain": None}


def _inputs(raw_runs, mentions_raw, entities, sources_by_run, alt_domains_rows=()):
    return FrozenInputs(
        entities=entities,
        alt_domains_rows=list(alt_domains_rows),
        entity_verification_rows=[],
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
    )


def test_restricted_share_excludes_zero_source_runs():
    raw_runs = [_raw_run("run1"), _raw_run("run2")]
    mentions = [_mention("run1", "Acme"), _mention("run2", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {
        "run1": [_source("https://someoneelse.com")],  # has sources, own domain not cited
        "run2": [],  # zero sources -- excluded from restricted, kept in unrestricted
    }
    result = s2.compute(_inputs(raw_runs, mentions, entities, sources_by_run))

    assert result.value["openai_chatgpt"]["restricted"] == 1.0  # 1/1 mentioned-without-own-citation
    assert result.value["openai_chatgpt"]["unrestricted"] == 1.0  # both count as own_domain_cited=False
    assert result.n_excluded["openai_chatgpt"]["restricted"]["no_sources"] == 1
    assert result.denominator_before_exclusion["openai_chatgpt"]["unrestricted"] == 2
    assert result.denominator_before_exclusion["openai_chatgpt"]["restricted"] == 1


def test_own_domain_cited_none_excluded_from_both_sides():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", None, "Acme")]  # no known domain -> own_domain_cited is None
    sources_by_run = {"run1": [_source("https://someoneelse.com")]}
    result = s2.compute(_inputs(raw_runs, mentions, entities, sources_by_run))

    assert result.value["openai_chatgpt"]["restricted"] is None  # no measurable pairs
    assert result.n_excluded["openai_chatgpt"]["restricted"]["own_domain_unmeasurable"] == 1
    assert result.denominator_after_exclusion["openai_chatgpt"]["restricted"] == 0


def test_own_domain_cited_true_lowers_the_share():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [_source("https://acme.com/pricing")]}
    result = s2.compute(_inputs(raw_runs, mentions, entities, sources_by_run))
    assert result.value["openai_chatgpt"]["restricted"] == 0.0  # own domain WAS cited


def test_unusable_run_excluded():
    raw_runs = [_raw_run("run1", error="collection_error")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    result = s2.compute(_inputs(raw_runs, mentions, entities, {"run1": []}))
    assert result.status == "untestable"


def test_branded_intent_7_excluded():
    raw_runs = [_raw_run("run1", intent_id=7)]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    result = s2.compute(_inputs(raw_runs, mentions, entities, {"run1": [_source("https://x.com")]}))
    assert result.status == "untestable"


def test_teardown_counts_qualifying_entity_under_threshold():
    # 5 mentions (clears the floor), own domain cited in 0 of them -> 0% < 10%
    raw_runs = [_raw_run(f"run{i}", intent_id=i) for i in range(1, 6)]
    mentions = [_mention(f"run{i}", "Acme") for i in range(1, 6)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {f"run{i}": [_source("https://someoneelse.com")] for i in range(1, 6)}
    result = s2.compute(_inputs(raw_runs, mentions, entities, sources_by_run))

    teardown = result.notes["teardown_by_engine"]["openai_chatgpt"]
    assert teardown["qualifying_entities"] == 1
    assert teardown["entities_under_10pct_own_citation"] == 1
    assert teardown["entity_ids_under_10pct"] == ["acme"]


def test_teardown_excludes_entity_below_eligibility_floor():
    # only 3 mentions -- below ELIGIBILITY_FLOOR (5), never qualifies
    raw_runs = [_raw_run(f"run{i}", intent_id=i) for i in range(1, 4)]
    mentions = [_mention(f"run{i}", "Acme") for i in range(1, 4)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {f"run{i}": [_source("https://someoneelse.com")] for i in range(1, 4)}
    result = s2.compute(_inputs(raw_runs, mentions, entities, sources_by_run))

    teardown = result.notes["teardown_by_engine"]["openai_chatgpt"]
    assert teardown["qualifying_entities"] == 0
    assert teardown["entities_under_10pct_own_citation"] == 0


def test_alternates_limitation_line_present_when_zero():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    result = s2.compute(_inputs(raw_runs, mentions, entities, {"run1": []}, alt_domains_rows=[]))
    assert "limitation" in result.notes


def test_alternates_limitation_line_absent_when_nonzero():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", None, "Acme")]  # only reachable via alt_domains
    alt_domains_rows = [{"entity_id": "acme", "alt_domain": "acme-alt.com", "source_note": "test"}]
    sources_by_run = {"run1": [_source("https://acme-alt.com")]}
    result = s2.compute(_inputs(raw_runs, mentions, entities, sources_by_run, alt_domains_rows))
    assert "limitation" not in result.notes


def test_source_definition_by_engine_covers_all_four_engines():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    result = s2.compute(_inputs(raw_runs, mentions, entities, {"run1": []}))
    defs = result.notes["source_definition_by_engine"]
    for engine in ("anthropic_claude", "openai_chatgpt", "google_gemini", "google_ai_mode"):
        assert engine in defs
        assert defs[engine]  # non-empty


def test_s2_entity_set_is_subset_of_population():
    """2026-09-23, owner review Fix 8: S2's entity set must be a subset of
    the study population -- confirmed to already be the 415 discovered/
    verified in-frame entities (owner decision, not a fixed ~40-50 roster
    that doesn't exist anywhere in this repo). An out-of-frame entity
    (meets_frame != "in") must never appear in S2's output at all."""
    from analysis.coding.b1_per_run_company import in_frame_entities

    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme"), _mention("run1", "OutOfFrame")]
    entities = [
        _entity("acme", "https://acme.com", "Acme"),
        _entity("out-of-frame-co", "https://oof.com", "OutOfFrame", meets_frame="borderline"),
    ]
    result = s2.compute(_inputs(raw_runs, mentions, entities, {"run1": [_source("https://acme.com")]}))

    population_ids = {e["entity_id"] for e in in_frame_entities(entities)}
    assert "out-of-frame-co" not in population_ids  # fixture sanity check

    for engine, teardown in result.notes["teardown_by_engine"].items():
        entity_ids = {d["entity_id"] for d in teardown["qualifying_entity_details"]} | set(
            teardown["entity_ids_under_10pct"]
        )
        assert entity_ids <= population_ids


def test_qualifying_entity_details_reports_mentions_and_rate():
    # 5 mentions (clears the floor), own domain cited in 2 of them -> 40% rate
    raw_runs = [_raw_run(f"run{i}", intent_id=i) for i in range(1, 6)]
    mentions = [_mention(f"run{i}", "Acme") for i in range(1, 6)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {
        "run1": [_source("https://acme.com/x")],
        "run2": [_source("https://acme.com/y")],
        "run3": [_source("https://someoneelse.com")],
        "run4": [_source("https://someoneelse.com")],
        "run5": [_source("https://someoneelse.com")],
    }
    result = s2.compute(_inputs(raw_runs, mentions, entities, sources_by_run))
    details = result.notes["teardown_by_engine"]["openai_chatgpt"]["qualifying_entity_details"]
    assert details == [{"entity_id": "acme", "mentions": 5, "own_citation_rate": 0.4}]
