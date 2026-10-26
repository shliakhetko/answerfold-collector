from analysis.phase_b.domain_ownership import (
    OwnershipRow,
    _entities_with_no_domain_breakdown,
    _gemini_citation_resolution_stats,
    _gemini_owned_share_bound,
    _gemini_title_recovery,
    build_source_ownership,
    entity_owned_domains,
    registrable_domain,
)


def test_registrable_domain_plain():
    assert registrable_domain("acme.com") == "acme.com"


def test_registrable_domain_strips_www_and_scheme():
    assert registrable_domain("https://www.acme.com") == "acme.com"


def test_registrable_domain_collapses_subdomain():
    assert registrable_domain("https://blog.acme.com") == "acme.com"


def test_registrable_domain_strips_port():
    assert registrable_domain("https://acme.com:8080/path") == "acme.com"


def test_registrable_domain_drops_path_and_query():
    assert registrable_domain("https://acme.com/pricing?utm_source=x") == "acme.com"


def test_registrable_domain_multilabel_suffix():
    assert registrable_domain("https://sub.acme.co.uk/x") == "acme.co.uk"


def test_registrable_domain_lowercases():
    assert registrable_domain("HTTPS://WWW.ACME.COM") == "acme.com"


def test_registrable_domain_bare_alt_domain_convention():
    # alt_domains.csv rows are bare domains, no scheme
    assert registrable_domain("acme.com") == registrable_domain("https://acme.com")


def test_registrable_domain_none_for_empty():
    assert registrable_domain("") is None
    assert registrable_domain(None) is None


def test_registrable_domain_none_for_malformed():
    assert registrable_domain("not a url with spaces") is None


def test_registrable_domain_none_for_unbalanced_brackets():
    # urlparse raises ValueError (attempted IPv6 literal) on this shape
    # rather than returning an empty netloc -- surfaced 2026-09-22 when a
    # free-text page title (not domain-shaped) reached this function for
    # the first time; malformed input -> None is the documented contract
    # regardless of which caller triggers it.
    assert registrable_domain("Best AI Tools [2026]") is None


def test_entity_owned_domains_empty_when_no_homepage_and_no_alt():
    malformed: list[str] = []
    entity = {"entity_id": "e1", "homepage_url": None}
    assert entity_owned_domains(entity, [], malformed) == {}
    assert malformed == []


def test_entity_owned_domains_homepage_and_alt_are_equal_citizens():
    malformed: list[str] = []
    entity = {"entity_id": "e1", "homepage_url": "https://acme.com"}
    alt_rows = [{"entity_id": "e1", "alt_domain": "acme.io"}]
    owned = entity_owned_domains(entity, alt_rows, malformed)
    assert owned == {"acme.com": "homepage", "acme.io": "alt_domain:acme.io"}


def test_entity_owned_domains_flags_malformed_homepage():
    malformed: list[str] = []
    entity = {"entity_id": "e1", "homepage_url": "not a url"}
    owned = entity_owned_domains(entity, [], malformed)
    assert owned == {}
    assert len(malformed) == 1


def _mention(run_id: str, surface_form: str) -> dict:
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": False, "votes": []}


def _entity(entity_id: str, homepage_url: str | None, surface_forms: str, meets_frame: str = "in") -> dict:
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": surface_forms, "meets_frame": meets_frame}


def _raw_run(
    run_id: str, *, engine: str = "openai_chatgpt", intent_id: int = 1, paraphrase_id: int = 1,
    error: str | None = None, raw_response_text: str | None = "some text", is_manual: int | None = None,
    day_index: int | None = 1,
) -> dict:
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent_id, "paraphrase_id": paraphrase_id,
        "error": error, "raw_response_text": raw_response_text, "is_manual": is_manual, "day_index": day_index,
    }


def test_build_source_ownership_null_when_no_domain():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", None, "Acme")]
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run={}, raw_runs=[_raw_run("run1")])
    assert len(rows) == 1
    assert rows[0].owned is None
    assert rows[0].matched_domain is None
    assert report.entities_with_no_domain == 1
    assert report.distinct_entities_covered == 1
    assert report.rows_for_entities_with_no_domain_count == 1
    assert report.owned_true_false_null == {"true": 0, "false": 0, "null": 1}


def test_build_source_ownership_true_when_homepage_domain_cited():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://acme.com/pricing"}]}
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run, raw_runs=[_raw_run("run1")])
    assert rows[0].owned is True
    assert rows[0].matched_domain == "acme.com"
    assert rows[0].n_sources == 1
    assert report.matches_via_subdomain == 0
    assert report.matches_via_alternate == 0


def test_build_source_ownership_true_when_subdomain_cited_and_counted_as_such():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://blog.acme.com/post"}]}
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run, raw_runs=[_raw_run("run1")])
    assert rows[0].owned is True
    assert rows[0].matched_domain == "acme.com"
    assert report.matches_via_subdomain == 1


def test_build_source_ownership_false_when_domain_known_but_not_cited():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://otherco.com"}]}
    rows, _ = build_source_ownership(mentions, entities, [], sources_by_run, raw_runs=[_raw_run("run1")])
    assert rows[0].owned is False
    assert rows[0].matched_domain is None
    assert rows[0].n_sources == 1


def test_build_source_ownership_false_when_no_sources_but_domain_known():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run={"run1": []}, raw_runs=[_raw_run("run1")])
    assert rows[0].owned is False
    assert rows[0].n_sources == 0
    assert report.responses_with_zero_sources == 1


def test_build_source_ownership_alt_domain_matches_and_counted_as_such():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", None, "Acme")]
    alt_rows = [{"entity_id": "acme", "alt_domain": "acme.io"}]
    sources_by_run = {"run1": [{"url_raw": "https://acme.io/x"}]}
    rows, report = build_source_ownership(mentions, entities, alt_rows, sources_by_run, raw_runs=[_raw_run("run1")])
    assert rows[0].owned is True
    assert rows[0].matched_domain == "acme.io"
    assert report.entities_with_alternates == 1
    assert report.matches_via_alternate == 1


def test_build_source_ownership_flags_unresolvable_surface_form():
    mentions = [_mention("run1", "Nonexistent Co")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run={}, raw_runs=[])
    assert rows == []
    assert report.malformed_input_count == 1


def test_build_source_ownership_flags_alt_domain_row_with_unknown_entity_id():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    alt_rows = [{"entity_id": "STUB-ghost", "alt_domain": "ghost.io", "source_note": "leftover"}]
    _rows, report = build_source_ownership(mentions, entities, alt_rows, sources_by_run={}, raw_runs=[_raw_run("run1")])
    assert report.malformed_input_count == 1
    assert any("STUB-ghost" in m for m in report.malformed_input_samples)


def test_build_source_ownership_deduplicates_repeated_mentions():
    mentions = [_mention("run1", "Acme"), _mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run={"run1": []}, raw_runs=[_raw_run("run1")])
    assert len(rows) == 1
    assert report.rows_produced == 1


def test_build_source_ownership_row_carries_engine_intent_paraphrase():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    raw_runs = [_raw_run("run1", engine="google_gemini", intent_id=3, paraphrase_id=5)]
    rows, _ = build_source_ownership(mentions, entities, [], sources_by_run={"run1": []}, raw_runs=raw_runs)
    row = rows[0]
    assert row.engine == "google_gemini"
    assert row.intent_id == 3
    assert row.paraphrase_id == 5


def test_build_source_ownership_report_counts_are_internally_consistent():
    mentions = [_mention("run1", "Acme"), _mention("run2", "Beta")]
    entities = [_entity("acme", "https://acme.com", "Acme"), _entity("beta", None, "Beta")]
    sources_by_run = {"run1": [{"url_raw": "https://acme.com"}], "run2": []}
    raw_runs = [_raw_run("run1"), _raw_run("run2")]
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run, raw_runs)

    assert report.rows_produced == len(rows) == 2
    triple = report.owned_true_false_null
    assert triple["true"] + triple["false"] + triple["null"] == report.rows_produced
    assert report.true_rows_without_matched_domain_count == 0
    assert report.false_or_null_rows_with_matched_domain_count == 0
    assert report.rows_for_entities_with_no_domain_count == triple["null"]
    assert report.total_responses_available == 2
    assert report.distinct_responses_covered == 2


def test_build_source_ownership_samples_are_capped_and_shaped():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://acme.com"}]}
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run, raw_runs=[_raw_run("run1")])
    assert len(report.sample_matched_rows) == 1
    assert report.sample_matched_rows[0]["matched_domain"] == "acme.com"
    assert report.sample_unmatched_with_sources_rows == []


def test_build_source_ownership_sample_unmatched_with_sources():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://otherco.com"}]}
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run, raw_runs=[_raw_run("run1")])
    assert len(report.sample_unmatched_with_sources_rows) == 1
    assert report.sample_unmatched_with_sources_rows[0]["n_sources"] == 1
    assert report.sample_unmatched_with_sources_rows[0]["owned"] is False


def test_ownership_row_to_json_shape():
    row = OwnershipRow(
        run_id="r1", entity_id="e1", engine="openai_chatgpt", intent_id=1, paraphrase_id=1,
        owned=True, matched_domain="acme.com", n_sources=2,
    )
    assert row.to_json() == {
        "run_id": "r1", "entity_id": "e1", "engine": "openai_chatgpt", "intent_id": 1, "paraphrase_id": 1,
        "owned": True, "matched_domain": "acme.com", "n_sources": 2,
    }


def test_build_source_ownership_total_responses_available_excludes_unusable_runs():
    """total_responses_available must mean the same "usable" as S3's own
    denominator -- an errored run doesn't count, matching
    analysis.phase_b.exclusions."""
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    raw_runs = [_raw_run("run1"), _raw_run("run2", error="500 INTERNAL"), _raw_run("run3", is_manual=1)]
    _rows, report = build_source_ownership(mentions, entities, [], sources_by_run={"run1": []}, raw_runs=raw_runs)
    assert report.total_responses_available == 1  # only run1 is usable


def test_build_source_ownership_reports_alt_domains_rows_loaded_plainly():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    alt_rows = [{"entity_id": "acme", "alt_domain": "acme.io"}]
    _rows, report = build_source_ownership(mentions, entities, alt_rows, sources_by_run={"run1": []}, raw_runs=[_raw_run("run1")])
    assert report.alt_domains_rows_loaded == 1


def test_build_source_ownership_alt_domains_rows_loaded_zero_when_empty():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    _rows, report = build_source_ownership(mentions, entities, [], sources_by_run={"run1": []}, raw_runs=[_raw_run("run1")])
    assert report.alt_domains_rows_loaded == 0


def test_build_source_ownership_excludes_out_of_frame_entity_silently():
    """A mention resolving to a real, known entity that fails the frame is
    dropped from the denominator, not flagged as malformed -- the frame
    exclusion and a data-quality problem must never be counted together."""
    mentions = [_mention("run1", "Acme"), _mention("run1", "Outsider")]
    entities = [
        _entity("acme", "https://acme.com", "Acme", meets_frame="in"),
        _entity("outsider", "https://outsider.com", "Outsider", meets_frame="out"),
    ]
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run={"run1": []}, raw_runs=[_raw_run("run1")])
    assert len(rows) == 1
    assert rows[0].entity_id == "acme"
    assert report.distinct_entities_covered == 1
    assert report.malformed_input_count == 0


def test_build_source_ownership_unresolvable_surface_form_still_malformed_when_out_of_frame_entities_exist():
    mentions = [_mention("run1", "Nonexistent Co")]
    entities = [_entity("outsider", "https://outsider.com", "Outsider", meets_frame="out")]
    rows, report = build_source_ownership(mentions, entities, [], sources_by_run={}, raw_runs=[])
    assert rows == []
    assert report.malformed_input_count == 1


def test_entity_owned_domains_falls_back_to_verified_homepage_when_registry_blank():
    malformed: list[str] = []
    entity = {"entity_id": "e1", "homepage_url": None}
    verified = {"e1": "https://acme-verified.com"}
    owned = entity_owned_domains(entity, [], malformed, verified)
    assert owned == {"acme-verified.com": "verified_homepage"}
    assert malformed == []


def test_entity_owned_domains_registry_homepage_wins_over_verified():
    malformed: list[str] = []
    entity = {"entity_id": "e1", "homepage_url": "https://acme.com"}
    verified = {"e1": "https://acme-verified.com"}
    owned = entity_owned_domains(entity, [], malformed, verified)
    assert owned == {"acme.com": "homepage"}


def test_build_source_ownership_verified_homepage_fallback_end_to_end():
    mentions = [_mention("run1", "Acme")]
    entities = [_entity("acme", None, "Acme")]
    sources_by_run = {"run1": [{"url_raw": "https://acme-verified.com/pricing"}]}
    verification = [{"entity_id": "acme", "verified_homepage_url": "https://acme-verified.com"}]
    rows, report = build_source_ownership(
        mentions, entities, [], sources_by_run, raw_runs=[_raw_run("run1")], entity_verification_rows=verification
    )
    assert rows[0].owned is True
    assert rows[0].matched_domain == "acme-verified.com"
    assert report.entities_with_no_domain == 0


# --- Gemini citation resolution reporting (2026-09-22) ---


def _gemini_source(url_raw: str, *, resolution_status: str | None = None) -> dict:
    row = {"url_raw": url_raw}
    if resolution_status is not None:
        row["resolution_status"] = resolution_status
    return row


def test_gemini_citation_stats_counts_resolved_and_unresolved():
    sources_by_run = {
        "run1": [
            _gemini_source("https://realpublisher.com/post"),  # resolved (no wrapper marker)
            _gemini_source("https://vertexaisearch.cloud.google.com/x", resolution_status="http_404"),
        ]
    }
    raw_runs = [_raw_run("run1", engine="google_gemini", day_index=3)]
    stats = _gemini_citation_resolution_stats(sources_by_run, raw_runs)
    assert stats["gemini_citations_total"] == 2
    assert stats["gemini_citations_resolved"] == 1
    assert stats["gemini_citations_unresolved"] == 1
    assert stats["gemini_unresolved_by_status"] == {"http_404": 1}
    assert stats["gemini_unresolved_by_day_index"] == {"3": 1}
    assert "genuinely unreachable" in stats["gemini_unresolved_note"]


def test_gemini_citation_stats_flags_never_matched_as_a_real_gap():
    sources_by_run = {"run1": [_gemini_source("https://vertexaisearch.cloud.google.com/x")]}  # no resolution_status
    raw_runs = [_raw_run("run1", engine="google_gemini", day_index=2)]
    stats = _gemini_citation_resolution_stats(sources_by_run, raw_runs)
    assert stats["gemini_unresolved_by_status"] == {"never_matched_in_aux_capture": 1}
    assert "never captured by the Sep 6/7 aux capture" in stats["gemini_unresolved_note"]
    assert "not fixed yet" in stats["gemini_unresolved_note"].lower()


def test_gemini_citation_stats_ignores_other_engines():
    sources_by_run = {"run1": [_gemini_source("https://vertexaisearch.cloud.google.com/x", resolution_status="http_404")]}
    raw_runs = [_raw_run("run1", engine="openai_chatgpt")]
    stats = _gemini_citation_resolution_stats(sources_by_run, raw_runs)
    assert stats["gemini_citations_total"] == 0
    assert stats["gemini_unresolved_note"].startswith("0 unresolved")


def test_gemini_citation_stats_zero_unresolved_note():
    sources_by_run = {"run1": [_gemini_source("https://realpublisher.com/post")]}
    raw_runs = [_raw_run("run1", engine="google_gemini")]
    stats = _gemini_citation_resolution_stats(sources_by_run, raw_runs)
    assert stats["gemini_citations_unresolved"] == 0
    assert "fully covers" in stats["gemini_unresolved_note"]


# --- entities_with_no_domain breakdown (2026-09-22) ---


def test_entities_with_no_domain_breakdown_partitions_correctly():
    mentions = [
        _mention("run1", "OutOfFrame"),
        _mention("run1", "ViaFallback"),
        _mention("run1", "ViaAlt"),
        _mention("run1", "StillNone"),
        _mention("run1", "AlreadyHadHomepage"),
    ]
    entities = [
        _entity("out-of-frame", None, "OutOfFrame", meets_frame="out"),
        _entity("via-fallback", None, "ViaFallback", meets_frame="in"),
        _entity("via-alt", None, "ViaAlt", meets_frame="in"),
        _entity("still-none", None, "StillNone", meets_frame="in"),
        _entity("already-had", "https://already-had.com", "AlreadyHadHomepage", meets_frame="in"),
    ]
    entities_by_id = {e["entity_id"]: e for e in entities}
    in_frame_ids = {e["entity_id"] for e in entities if e["meets_frame"] == "in"}
    verified_homepage_by_entity = {"via-fallback": "https://via-fallback.com"}
    alt_domains_rows = [{"entity_id": "via-alt", "alt_domain": "via-alt-alt.com"}]

    breakdown = _entities_with_no_domain_breakdown(
        mentions, entities, entities_by_id, in_frame_ids, verified_homepage_by_entity, alt_domains_rows
    )
    assert breakdown == {
        "entities_with_no_domain_before_fix": 4,  # everyone except already-had
        "entities_dropped_by_in_frame_fix": 1,
        "entities_gained_domain_via_homepage_fallback": 1,
        "entities_gained_domain_via_alt_domains_only": 1,
        "entities_still_with_no_domain_after_fix": 1,
    }


# --- zero-source reconcile (2026-09-22) ---


def test_build_source_ownership_zero_source_reconcile_sums_correctly():
    # run1: covered, zero sources -> counts toward responses_with_zero_sources
    # run2: usable but zero extracted mentions, also zero sources -> excluded bucket
    # run3: covered, has a source -> not zero-source at all
    mentions = [_mention("run1", "Acme"), _mention("run3", "Acme")]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    sources_by_run = {"run1": [], "run3": [{"url_raw": "https://acme.com"}]}
    raw_runs = [_raw_run("run1"), _raw_run("run2"), _raw_run("run3")]
    _rows, report = build_source_ownership(mentions, entities, [], sources_by_run, raw_runs)

    assert report.responses_with_zero_sources == 1  # run1
    assert report.total_zero_source_usable_responses == 2  # run1 + run2
    assert report.zero_source_responses_excluded_zero_mentions == 1  # run2
    assert report.zero_source_responses_excluded_out_of_frame_only == 0
    assert (
        report.responses_with_zero_sources
        + report.zero_source_responses_excluded_zero_mentions
        + report.zero_source_responses_excluded_out_of_frame_only
        == report.total_zero_source_usable_responses
    )


# --- Gemini owned-share bound (2026-09-22) ---


def _ownership_row(run_id, entity_id, *, engine="google_gemini", owned) -> OwnershipRow:
    return OwnershipRow(
        run_id=run_id, entity_id=entity_id, engine=engine, intent_id=1, paraphrase_id=1,
        owned=owned, matched_domain=None, n_sources=1,
    )


def test_gemini_owned_share_bound_current_share():
    rows = [
        _ownership_row("run1", "acme", owned=True),
        _ownership_row("run2", "beta", owned=False),
    ]
    sources_by_run = {"run1": [], "run2": []}
    bound = _gemini_owned_share_bound(rows, sources_by_run)
    assert bound["gemini_owned_share_current"] == 0.5
    assert bound["gemini_owned_share_naive_upper_bound"] == 0.5  # no unresolved citations -> nothing flippable
    assert bound["gemini_owned_share_tight_upper_bound"] == 0.5


def test_gemini_owned_share_bound_naive_overcounts_shared_run():
    # two entities, same run, one unresolved citation between them -- at
    # most one of the two could really own it, so tight must be lower
    # than naive
    rows = [
        _ownership_row("run1", "acme", owned=False),
        _ownership_row("run1", "beta", owned=False),
    ]
    sources_by_run = {"run1": [{"url_raw": "https://vertexaisearch.cloud.google.com/x"}]}
    bound = _gemini_owned_share_bound(rows, sources_by_run)
    assert bound["gemini_owned_share_naive_upper_bound"] == 1.0  # both counted flippable
    assert bound["gemini_owned_share_tight_upper_bound"] == 0.5  # capped at 1 unresolved citation
    assert bound["gemini_owned_share_tight_upper_bound"] < bound["gemini_owned_share_naive_upper_bound"]


def test_gemini_owned_share_bound_no_measurable_rows():
    bound = _gemini_owned_share_bound([], {})
    assert bound["gemini_owned_share_current"] is None
    assert "no measurable" in bound["gemini_owned_share_bound_note"]


# --- Gemini title recovery (2026-09-22) ---


def test_gemini_title_recovery_validates_resolved_citations():
    rows = []
    sources_by_run = {"run1": [{"url_raw": "https://acme.com/post", "title": "acme.com"}]}
    raw_runs = [_raw_run("run1", engine="google_gemini")]
    result = _gemini_title_recovery(rows, sources_by_run, raw_runs, {})
    assert result["gemini_title_validation_checked"] == 1
    assert result["gemini_title_validation_matches"] == 1


def test_gemini_title_recovery_flips_row_when_title_matches_owned_domain():
    rows = [_ownership_row("run1", "acme", owned=False)]
    sources_by_run = {
        "run1": [{"url_raw": "https://vertexaisearch.cloud.google.com/x", "title": "acme.com"}]
    }
    raw_runs = [_raw_run("run1", engine="google_gemini")]
    owned_domains_by_entity = {"acme": {"acme.com": "homepage"}}
    result = _gemini_title_recovery(rows, sources_by_run, raw_runs, owned_domains_by_entity)
    assert result["gemini_title_recovery_residual_recoverable_rows"] == 1
    assert result["gemini_title_recovery_residual_owned_share"] == 1.0


def test_gemini_title_recovery_ignores_other_engines_and_never_crashes():
    # a real page title (not domain-shaped, with brackets) on a
    # non-Gemini source must be skipped entirely, not passed through
    # registrable_domain -- regression test for the ValueError this
    # tripped before the engine filter was added
    rows = [_ownership_row("run1", "acme", engine="openai_chatgpt", owned=False)]
    sources_by_run = {"run1": [{"url_raw": "https://example.com", "title": "Best AI Tools [2026]"}]}
    raw_runs = [_raw_run("run1", engine="openai_chatgpt")]
    result = _gemini_title_recovery(rows, sources_by_run, raw_runs, {"acme": {"acme.com": "homepage"}})
    assert result["gemini_title_validation_checked"] == 0
    assert result["gemini_title_recovery_residual_recoverable_rows"] == 0


def test_gemini_title_recovery_reports_near_zero_residual_on_post_fix_data():
    """As of 2026-09-23 the real fix runs upstream (in
    analysis.coding.cited_url_resolution.resolve_sources_by_run), so on
    real production data sources_by_run handed to this function already
    has title-recovered rows with url_raw rewritten (no wrapper marker,
    resolution_method="title_fallback") -- this function's own recovery
    scan should then find nothing left to recover, and its validation
    loop must not count a title-fallback row as independent ground truth
    (that would be circular: the url_raw it's "validating" was itself
    derived from the title)."""
    # owned=False here (despite the citation carrying the entity's own
    # domain) is the realistic post-fix shape: the real join already
    # matched this via registrable_domain(url_raw) and would have set
    # owned=True, so a row reaching this function still owned=False with
    # only a title-fallback source attached means some OTHER, unrelated
    # entity's row -- the wrapper-marker check (not the owned filter) is
    # what must correctly skip a title-fallback row here.
    rows = [_ownership_row("run1", "acme", owned=False)]
    sources_by_run = {
        "run1": [
            {
                "url_raw": "https://acme.com/",
                "domain": "acme.com",
                "title": "acme.com",
                "resolution_method": "title_fallback",
            }
        ]
    }
    raw_runs = [_raw_run("run1", engine="google_gemini")]
    result = _gemini_title_recovery(rows, sources_by_run, raw_runs, {"acme": {"acme.com": "homepage"}})
    assert result["gemini_title_validation_checked"] == 0  # excluded: not independent ground truth
    assert result["gemini_title_recovery_residual_recoverable_rows"] == 0
