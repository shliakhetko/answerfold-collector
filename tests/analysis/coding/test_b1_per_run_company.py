from analysis.coding.b1_per_run_company import (
    build_per_run_company,
    entity_domains,
    in_frame_entities,
    surface_form_to_entity_id,
)

ENTITY_REGISTRY_ROWS = [
    {
        "entity_id": "goodcall", "canonical_name": "Goodcall", "surface_forms": "Goodcall",
        "homepage_url": "https://goodcall.com/", "meets_frame": "in",
    },
    {
        "entity_id": "vapi", "canonical_name": "Vapi", "surface_forms": "Vapi",
        "homepage_url": "https://vapi.ai/", "meets_frame": "in",
    },
    {
        "entity_id": "metro-answering", "canonical_name": "Metro Answering Service",
        "surface_forms": "Metro Answering Service", "homepage_url": "https://metroanswering.example.com/",
        "meets_frame": "out",  # out-of-frame -- must never enter a denominator
    },
]

RUN_ROWS = [
    {
        "run_id": "run-1",
        "raw_response_text": "Unlike Vapi, which needs setup, Goodcall works out of the box. See https://goodcall.com/.",
    },
    {"run_id": "run-2", "raw_response_text": "Metro Answering Service is a live human option."},
]

SOURCES_BY_RUN = {
    "run-1": [{"url_raw": "https://goodcall.com/pricing", "domain": "goodcall.com", "title": "Goodcall Pricing"}],
    "run-2": [{"url_raw": "https://metroanswering.example.com/", "domain": "metroanswering.example.com", "title": "Metro"}],
}

MENTIONS_RAW_ROWS = [
    {"run_id": "run-1", "surface_form": "Goodcall", "evidence_span": "Goodcall works out of the box.", "is_recommended": True, "votes": ["a", "b"]},
    {"run_id": "run-1", "surface_form": "Vapi", "evidence_span": "Unlike Vapi, which needs setup,", "is_recommended": False, "votes": ["a", "b"]},
    # Out-of-frame entity's mention -- must be dropped, never joined.
    {"run_id": "run-2", "surface_form": "Metro Answering Service", "evidence_span": "Metro Answering Service is a live human option.", "is_recommended": True, "votes": ["a", "b"]},
]


def test_in_frame_entities_excludes_meets_frame_false():
    entities = in_frame_entities(ENTITY_REGISTRY_ROWS)
    assert {e["entity_id"] for e in entities} == {"goodcall", "vapi"}


def test_surface_form_to_entity_id_only_covers_in_frame_entities():
    mapping = surface_form_to_entity_id(in_frame_entities(ENTITY_REGISTRY_ROWS))
    assert mapping == {"Goodcall": "goodcall", "Vapi": "vapi"}


def test_entity_domains_includes_alt_domains():
    entities = in_frame_entities(ENTITY_REGISTRY_ROWS)
    domains = entity_domains(entities, alt_domains_rows=[{"entity_id": "goodcall", "alt_domain": "goodcall.io"}])
    assert domains["goodcall"] == {"goodcall.com", "goodcall.io"}
    assert domains["vapi"] == {"vapi.ai"}


def test_build_per_run_company_is_a_full_cross_product_of_runs_and_in_frame_entities():
    rows = build_per_run_company(RUN_ROWS, MENTIONS_RAW_ROWS, ENTITY_REGISTRY_ROWS, SOURCES_BY_RUN, alt_domains_rows=[])
    # 2 runs x 2 in-frame entities = 4 rows. metro-answering is out-of-frame
    # and never appears, even though it was mentioned in run-2.
    assert len(rows) == 4
    assert {r.entity_id for r in rows} == {"goodcall", "vapi"}


def test_build_per_run_company_goodcall_mentioned_recommended_and_own_domain_cited():
    rows = build_per_run_company(RUN_ROWS, MENTIONS_RAW_ROWS, ENTITY_REGISTRY_ROWS, SOURCES_BY_RUN, alt_domains_rows=[])
    row = next(r for r in rows if r.run_id == "run-1" and r.entity_id == "goodcall")
    assert row.mentioned is True
    assert row.recommended is True
    assert row.own_domain_cited is True
    assert row.linked is True
    assert row.mention_without_own_citation is False
    assert row.first_mention_position == RUN_ROWS[0]["raw_response_text"].find("Goodcall works out of the box.")


def test_build_per_run_company_vapi_mentioned_but_not_recommended_caveat():
    rows = build_per_run_company(RUN_ROWS, MENTIONS_RAW_ROWS, ENTITY_REGISTRY_ROWS, SOURCES_BY_RUN, alt_domains_rows=[])
    row = next(r for r in rows if r.run_id == "run-1" and r.entity_id == "vapi")
    assert row.mentioned is True
    assert row.recommended is False  # named in a caveat, not recommended
    assert row.own_domain_cited is False  # no vapi.ai source cited in run-1
    assert row.mention_without_own_citation is True


def test_build_per_run_company_not_mentioned_means_not_recommended():
    rows = build_per_run_company(RUN_ROWS, MENTIONS_RAW_ROWS, ENTITY_REGISTRY_ROWS, SOURCES_BY_RUN, alt_domains_rows=[])
    row = next(r for r in rows if r.run_id == "run-2" and r.entity_id == "goodcall")
    assert row.mentioned is False
    assert row.recommended is False
    assert row.own_domain_cited is False
    assert row.mention_without_own_citation is False  # not mentioned, so this is False too
    assert row.first_mention_position is None


def test_first_mention_position_finds_evidence_span_across_a_stripped_markdown_link():
    # A1 (a1_answer_coding.py) extracts evidence_span from markdown-link-
    # stripped text -- if this span straddled where a link used to be (e.g.
    # "[Fathom](url) is the overall best option." -> evidence_span "Fathom
    # is the overall best option."), searching the UNTOUCHED raw text for it
    # would fail (the raw text still has the link syntax in the way). This
    # is the regression test for that: raw_response_text still carries the
    # original markdown link, but first_mention_position must still resolve.
    run_rows = [{
        "run_id": "run-3",
        "raw_response_text": "[Fathom](https://fathom.video/) is the overall best option.",
    }]
    entity_registry_rows = [{
        "entity_id": "fathom", "canonical_name": "Fathom", "surface_forms": "Fathom",
        "homepage_url": "https://fathom.video/", "meets_frame": "in",
    }]
    mentions_raw_rows = [{
        "run_id": "run-3", "surface_form": "Fathom",
        "evidence_span": "Fathom is the overall best option.", "is_recommended": True, "votes": ["a", "b"],
    }]

    rows = build_per_run_company(run_rows, mentions_raw_rows, entity_registry_rows, sources_by_run={}, alt_domains_rows=[])

    row = next(r for r in rows if r.run_id == "run-3" and r.entity_id == "fathom")
    assert row.first_mention_position is not None


def test_out_of_frame_entity_never_produces_a_row():
    rows = build_per_run_company(RUN_ROWS, MENTIONS_RAW_ROWS, ENTITY_REGISTRY_ROWS, SOURCES_BY_RUN, alt_domains_rows=[])
    assert all(r.entity_id != "metro-answering" for r in rows)
