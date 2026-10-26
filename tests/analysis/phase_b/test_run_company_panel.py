from analysis.phase_b.reading import FrozenInputs
from analysis.phase_b.run_company_panel import build_run_company_panel


def _mention(run_id, surface_form, *, is_recommended=False):
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": is_recommended, "votes": []}


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _raw_run(run_id, *, engine="openai_chatgpt", intent_id=1, paraphrase_id=1, day_index=1, error=None, raw_response_text="some text"):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": paraphrase_id,
        "day_index": day_index, "error": error, "raw_response_text": raw_response_text, "is_manual": None,
    }


def _inputs(raw_runs, mentions_raw, entities, sources_by_run, alt_domains_rows=()):
    return FrozenInputs(
        entities=entities,
        alt_domains_rows=list(alt_domains_rows),
        entity_verification_rows=[],
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
    )


def test_own_domain_cited_true_when_domain_matched():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://acme.com/pricing", "domain": None}]}

    rows, eligibility, report = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run))

    row = next(r for r in rows if r.run_id == "run1" and r.entity_id == "acme")
    assert row.mentioned is True
    assert row.own_domain_cited is True
    assert row.linked is True
    assert report.own_domain_cited_true_false_null == {"true": 1, "false": 0, "null": 0}


def test_own_domain_cited_false_when_domain_not_cited():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://someoneelse.com/x", "domain": None}]}

    rows, _, report = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run))

    row = next(r for r in rows if r.run_id == "run1" and r.entity_id == "acme")
    assert row.own_domain_cited is False
    assert row.linked is False
    assert report.own_domain_cited_true_false_null == {"true": 0, "false": 1, "null": 0}


def test_own_domain_cited_none_when_entity_has_no_known_domain():
    """The real delta this module exists to fix: B1's own own_domain_cited
    would silently read False here (no homepage_url, no fallback) -- the
    panel keeps this an honest None (unmeasurable) instead."""
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", None, "Acme")]  # no homepage_url at all
    sources_by_run = {"run1": []}

    rows, _, report = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run))

    row = next(r for r in rows if r.run_id == "run1" and r.entity_id == "acme")
    assert row.own_domain_cited is None
    assert row.linked is None
    assert report.own_domain_cited_true_false_null == {"true": 0, "false": 0, "null": 1}
    assert report.rows_flipped_false_to_none == 1  # B1 would have said False; panel correctly says None


def test_non_mentioned_rows_excluded_from_own_domain_cited_report():
    """B1 produces the full run x in-frame-entity cross product; a
    non-mentioned pair's own_domain_cited lookup is always a miss (never
    considered by domain_ownership), which must not be counted alongside
    genuinely-measured mentioned rows."""
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme"), _entity("beta", None, "Beta")]
    sources_by_run = {"run1": [{"url_raw": "https://acme.com/x", "domain": None}]}

    rows, _, report = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run))

    beta_row = next(r for r in rows if r.entity_id == "beta")
    assert beta_row.mentioned is False
    assert beta_row.own_domain_cited is None
    # only acme's mentioned row enters the report -- beta (not mentioned,
    # never a domain_ownership candidate) does not inflate the null count
    assert report.own_domain_cited_true_false_null == {"true": 1, "false": 0, "null": 0}
    assert report.rows_flipped_false_to_none == 0
    assert report.rows_produced == 2  # full cross product: 1 run x 2 entities
    assert report.mentioned_rows == 1


def test_alternates_stats_pass_through_from_domain_ownership():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", None, "Acme")]  # no homepage_url -- only reachable via alt_domains
    sources_by_run = {"run1": [{"url_raw": "https://alt-acme.example", "domain": None}]}
    alt_domains_rows = [{"entity_id": "acme", "alt_domain": "alt-acme.example", "source_note": "test"}]

    _, _, report = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run, alt_domains_rows))

    assert report.entities_with_alternates == 1
    assert report.matches_via_alternate == 1


def test_recommended_passes_through_from_b1():
    raw_runs = [_raw_run("run1")]
    mentions = [_mention("run1", "Acme", is_recommended=True)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": []}

    rows, _, _ = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run))

    row = next(r for r in rows if r.run_id == "run1" and r.entity_id == "acme")
    assert row.recommended is True


def test_eligibility_floor_reuses_apply_eligibility_floor():
    # 5 mentions across 5 distinct unbranded-intent runs, one engine ->
    # exactly clears ELIGIBILITY_FLOOR (5)
    raw_runs = [_raw_run(f"run{i}", intent_id=i) for i in range(1, 6)]
    mentions = [_mention(f"run{i}", "Acme") for i in range(1, 6)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {f"run{i}": [] for i in range(1, 6)}

    _, eligibility, report = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run))

    assert "acme" in eligibility.qualifying_by_engine.get("openai_chatgpt", set())
    assert report.qualifying_entities_by_engine == {"openai_chatgpt": 1}


def test_eligibility_floor_excludes_below_threshold():
    raw_runs = [_raw_run(f"run{i}", intent_id=i) for i in range(1, 4)]  # only 3 mentions
    mentions = [_mention(f"run{i}", "Acme") for i in range(1, 4)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {f"run{i}": [] for i in range(1, 4)}

    _, eligibility, report = build_run_company_panel(_inputs(raw_runs, mentions, entities, sources_by_run))

    assert "acme" not in eligibility.qualifying_by_engine.get("openai_chatgpt", set())
    assert report.qualifying_entities_by_engine == {}
