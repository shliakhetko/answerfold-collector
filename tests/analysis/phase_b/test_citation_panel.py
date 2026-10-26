from analysis.phase_b.citation_panel import (
    all_in_frame_owned_domains,
    build_citation_panel,
    mentioned_own_domains_by_run,
    url_class_publisher_crosstab,
)
from analysis.phase_b.reading import FrozenInputs


def _run(run_id, engine, *, intent_id, error=None, raw_response_text="some text", paraphrase_id=1):
    return {
        "run_id": run_id,
        "engine": engine,
        "intent_id": intent_id,
        "paraphrase_id": paraphrase_id,
        "error": error,
        "raw_response_text": raw_response_text,
        "is_manual": None,
    }


def _registry_row(url_normalized, url_class, publisher_is_in_category, *, settled_class=None, settled_pub=None):
    return {
        "url_normalized": url_normalized,
        "url_class": url_class,
        "publisher_is_in_category": publisher_is_in_category,
        "settled_class": settled_class if settled_class is not None else url_class,
        "settled_publisher_is_in_category": settled_pub if settled_pub is not None else publisher_is_in_category,
    }


def _mention(run_id, surface_form):
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": False, "votes": []}


def _entity(entity_id, homepage_url, surface_forms, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _inputs(raw_runs, sources_by_run, url_registry_rows=(), entities=(), mentions_raw=(), alt_domains_rows=()):
    return FrozenInputs(
        entities=list(entities),
        alt_domains_rows=list(alt_domains_rows),
        entity_verification_rows=[],
        mentions_raw=list(mentions_raw),
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
        url_registry_rows=list(url_registry_rows),
    )


def test_basic_row_with_registry_match():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://acme.com/page", "domain": "acme.com"}]}
    registry = [_registry_row("https://acme.com/page", "vendor page", True)]

    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run, registry))

    assert len(rows) == 1
    row = rows[0]
    assert row.url_class == "vendor page"
    assert row.publisher_is_in_category is True
    assert row.url_normalized == "https://acme.com/page"
    assert report.rows_produced == 1
    assert report.runs_in_scope == 1
    assert report.url_class_counts == {"vendor page": 1}
    assert report.citations_unclassified == 0


def test_branded_intent_7_excluded():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=7)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://acme.com/page"}]}
    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run))
    assert rows == []
    assert report.runs_in_scope == 0


def test_excluded_run_never_enters_panel():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1, error="timeout")]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://acme.com/page"}]}
    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run))
    assert rows == []
    assert report.usable_runs_total == 0


def test_settled_fields_override_raw_model_output():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=2)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://reviewsite.example/x"}]}
    registry = [
        _registry_row(
            "https://reviewsite.example/x", "review site", False, settled_class="vendor page", settled_pub=True
        )
    ]
    rows, _ = build_citation_panel(_inputs(raw_runs, sources_by_run, registry))
    assert rows[0].url_class == "vendor page"
    assert rows[0].publisher_is_in_category is True


def test_unclassified_citation_split_by_reason():
    raw_runs = [_run("r1", "google_gemini", intent_id=3)]
    sources_by_run = {
        "r1": [
            {"position": 1, "url_raw": "https://never-classified.example/x", "resolution_method": None},
            {
                "position": 2,
                "url_raw": "https://acme.com/",
                "resolution_method": "title_fallback",
            },
        ]
    }
    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run, []))

    assert all(r.url_class is None for r in rows)
    assert report.citations_unclassified == 2
    assert report.citations_unclassified_other == 1
    assert report.citations_unclassified_title_fallback == 1
    assert report.url_class_counts == {"unclassified": 2}


def test_engines_in_scope_counts_citations_per_engine():
    raw_runs = [
        _run("r1", "openai_chatgpt", intent_id=1),
        _run("r2", "google_gemini", intent_id=1),
    ]
    sources_by_run = {
        "r1": [{"position": 1, "url_raw": "https://a.example"}],
        "r2": [{"position": 1, "url_raw": "https://b.example"}, {"position": 2, "url_raw": "https://c.example"}],
    }
    _, report = build_citation_panel(_inputs(raw_runs, sources_by_run, []))
    assert report.engines_in_scope == {"openai_chatgpt": 1, "google_gemini": 2}


def test_unclassified_domains_grouped_by_reason():
    raw_runs = [_run("r1", "google_gemini", intent_id=1)]
    sources_by_run = {
        "r1": [
            {"position": 1, "url_raw": "https://a-domain.com/x", "resolution_method": None},
            {"position": 2, "url_raw": "https://a-domain.com/y", "resolution_method": None},
            {"position": 3, "url_raw": "https://b-domain.com/", "resolution_method": "title_fallback"},
        ]
    }
    _, report = build_citation_panel(_inputs(raw_runs, sources_by_run, []))
    assert report.unclassified_domains_by_reason == {
        "a-domain.com": {"title_fallback": 0, "other": 2},
        "b-domain.com": {"title_fallback": 1, "other": 0},
    }


def test_url_class_publisher_crosstab():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    sources_by_run = {
        "r1": [
            {"position": 1, "url_raw": "https://a-domain.com/x"},
            {"position": 2, "url_raw": "https://b-domain.com/y"},
            {"position": 3, "url_raw": "https://c-domain.com/z"},
        ]
    }
    registry = [
        _registry_row("https://a-domain.com/x", "vendor page", True),
        _registry_row("https://b-domain.com/y", "vendor listicle", False),
    ]
    rows, _ = build_citation_panel(_inputs(raw_runs, sources_by_run, registry))
    tab = url_class_publisher_crosstab(rows)
    assert tab["openai_chatgpt"]["vendor page"] == {"true": 1, "false": 0, "null": 0}
    assert tab["openai_chatgpt"]["vendor listicle"] == {"true": 0, "false": 1, "null": 0}
    assert tab["openai_chatgpt"]["unclassified"] == {"true": 0, "false": 0, "null": 1}


# --------------------------------------------------------------------------
# 2026-09-23, owner review Fix D: publisher_is_in_category domain-ownership
# override -- A5's raw per-URL judgment is inconsistent on a known vendor's
# own content-marketing pages (goodcall.com/pricing=True,
# goodcall.com/answering-services/...=False, both Goodcall's own domain).
# --------------------------------------------------------------------------

def test_override_flips_false_to_true_for_known_vendor_domain():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://acme.com/answering-services/dental"}]}
    registry = [_registry_row("https://acme.com/answering-services/dental", "vendor listicle", False)]
    entities = [_entity("acme", "https://acme.com", "Acme")]

    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run, registry, entities=entities))

    row = rows[0]
    assert row.publisher_is_in_category is True  # forced from the registry's False
    assert row.publisher_is_in_category_overridden is True
    assert row.url_class == "vendor listicle"  # content-type label untouched
    assert report.citations_publisher_override_count == 1


def test_override_does_not_fire_for_unknown_domain():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://thirdparty.example/x"}]}
    registry = [_registry_row("https://thirdparty.example/x", "review site", False)]
    entities = [_entity("acme", "https://acme.com", "Acme")]  # a different domain

    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run, registry, entities=entities))

    row = rows[0]
    assert row.publisher_is_in_category is False
    assert row.publisher_is_in_category_overridden is False
    assert report.citations_publisher_override_count == 0


def test_override_does_not_fire_when_already_true():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://acme.com/pricing"}]}
    registry = [_registry_row("https://acme.com/pricing", "vendor page", True)]
    entities = [_entity("acme", "https://acme.com", "Acme")]

    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run, registry, entities=entities))

    row = rows[0]
    assert row.publisher_is_in_category is True
    assert row.publisher_is_in_category_overridden is False  # already True, nothing to override
    assert report.citations_publisher_override_count == 0


def test_override_does_not_fire_for_unclassified_citations():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://acme.com/never-classified"}]}
    entities = [_entity("acme", "https://acme.com", "Acme")]

    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run, [], entities=entities))

    row = rows[0]
    assert row.url_class is None
    assert row.publisher_is_in_category is None  # never touched -- no registry match to override
    assert row.publisher_is_in_category_overridden is False
    assert report.citations_publisher_override_count == 0


def test_override_ignores_out_of_frame_entities():
    """A domain belonging to an out-of-frame entity must not trigger the
    override -- matches K2's own-domain rule, both read the same
    all_in_frame_owned_domains set."""
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    sources_by_run = {"r1": [{"position": 1, "url_raw": "https://outframe.com/x"}]}
    registry = [_registry_row("https://outframe.com/x", "vendor listicle", False)]
    entities = [_entity("out", "https://outframe.com", "OutFrame", meets_frame="out")]

    rows, report = build_citation_panel(_inputs(raw_runs, sources_by_run, registry, entities=entities))

    row = rows[0]
    assert row.publisher_is_in_category is False
    assert report.citations_publisher_override_count == 0


def test_all_in_frame_owned_domains():
    entities = [
        _entity("acme", "https://acme.com", "Acme"),
        _entity("out", "https://outframe.com", "OutFrame", meets_frame="out"),
    ]
    inputs = _inputs([], {}, entities=entities)
    assert all_in_frame_owned_domains(inputs) == {"acme.com"}


def test_mentioned_own_domains_by_run():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=1)]
    entities = [_entity("acme", "https://acme.com", "Acme"), _entity("beta", "https://beta.com", "Beta")]
    mentions = [_mention("r1", "Acme")]  # only Acme mentioned in r1
    inputs = _inputs(raw_runs, {}, entities=entities, mentions_raw=mentions)
    result = mentioned_own_domains_by_run(inputs)
    assert result == {"r1": {"acme.com"}}
