import json

from analysis.coding.cited_url_resolution import (
    GEMINI_ENGINE,
    GEMINI_WRAPPER_MARKER,
    ResolvedCitation,
    load_resolution_map,
    resolve_sources_by_run,
    title_domain_match_rate_by_engine,
)

WRAP = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA"


# --------------------------------------------------------------------------
# load_resolution_map
# --------------------------------------------------------------------------

def test_load_missing_or_none_path_returns_empty():
    assert load_resolution_map(None) == {}
    assert load_resolution_map("/no/such/file.jsonl") == {}


def test_load_parses_and_strips_www(tmp_path):
    path = tmp_path / "resolution.jsonl"
    path.write_text(
        json.dumps({
            "url": WRAP, "status_bucket": "ok",
            "final_url": "https://www.cloudtalk.io/features/",
            "final_registrable_domain": "www.cloudtalk.io",
        }) + "\n"
        + "\n"  # blank line tolerated
        + json.dumps({
            "url": "https://dead.example/x", "status_bucket": "http_404",
            "final_url": "https://dead.example/x", "final_registrable_domain": "dead.example",
        }) + "\n",
        encoding="utf-8",
    )
    m = load_resolution_map(path)

    assert set(m) == {WRAP, "https://dead.example/x"}
    assert m[WRAP].final_domain == "cloudtalk.io"
    assert m[WRAP].is_reachable is True
    assert m[WRAP].is_dead is False
    assert m["https://dead.example/x"].is_dead is True
    assert m["https://dead.example/x"].is_reachable is False


def test_load_last_line_wins_on_duplicate_url(tmp_path):
    path = tmp_path / "resolution.jsonl"
    path.write_text(
        json.dumps({"url": WRAP, "status_bucket": "timeout", "final_url": None}) + "\n"
        + json.dumps({"url": WRAP, "status_bucket": "ok", "final_url": "https://x.io/"}) + "\n",
        encoding="utf-8",
    )
    assert load_resolution_map(path)[WRAP].status_bucket == "ok"


def test_reachable_needs_a_final_url():
    assert ResolvedCitation(WRAP, None, "x.io", "ok").is_reachable is False


# --------------------------------------------------------------------------
# resolve_sources_by_run
# --------------------------------------------------------------------------

def _sources(*urls):
    return {"run-1": [{"position": i, "url_raw": u, "domain": None, "title": "t"} for i, u in enumerate(urls)]}


def test_reachable_citation_is_swapped_and_original_kept():
    m = {WRAP: ResolvedCitation(WRAP, "https://cloudtalk.io/x", "cloudtalk.io", "ok")}
    resolved, stats = resolve_sources_by_run(_sources(WRAP), m)

    row = resolved["run-1"][0]
    assert row["url_raw"] == "https://cloudtalk.io/x"
    assert row["domain"] == "cloudtalk.io"
    assert row["url_cited"] == WRAP
    assert row["resolution_status"] == "ok"
    assert row["resolution_dead"] is False
    assert stats.citations_swapped == 1
    assert stats.citations_dead == 0


def test_dead_citation_is_flagged_not_swapped():
    dead = "https://otter.ai/gone"
    m = {dead: ResolvedCitation(dead, dead, "otter.ai", "http_404")}
    resolved, stats = resolve_sources_by_run(_sources(dead), m)

    row = resolved["run-1"][0]
    assert row["url_raw"] == dead  # unchanged
    assert row["resolution_dead"] is True
    assert stats.citations_dead == 1
    assert stats.citations_swapped == 0


def test_unmatched_citation_passes_through():
    resolved, stats = resolve_sources_by_run(_sources("https://x.io/a"), {})
    row = resolved["run-1"][0]
    assert row["url_raw"] == "https://x.io/a"
    assert row["resolution_status"] is None
    assert row["resolution_dead"] is False
    assert stats.citations_matched == 0


def test_unique_url_counts_collapse_wrappers():
    wrap_b = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/BBB"
    m = {
        WRAP: ResolvedCitation(WRAP, "https://otter.ai/pricing", "otter.ai", "ok"),
        wrap_b: ResolvedCitation(wrap_b, "https://otter.ai/pricing", "otter.ai", "ok"),
    }
    _, stats = resolve_sources_by_run(_sources(WRAP, wrap_b), m)
    assert stats.unique_urls_before == 2
    assert stats.unique_urls_after == 1


def test_inputs_not_mutated():
    src = _sources(WRAP)
    m = {WRAP: ResolvedCitation(WRAP, "https://x.io/", "x.io", "ok")}
    resolve_sources_by_run(src, m)
    assert src["run-1"][0]["url_raw"] == WRAP


def test_resolution_method_is_none_when_no_fallback_fires():
    resolved, _ = resolve_sources_by_run(_sources("https://x.io/a"), {})
    assert resolved["run-1"][0]["resolution_method"] is None


# --------------------------------------------------------------------------
# Gemini title fallback
# --------------------------------------------------------------------------

def _gemini_source(url_raw, title, *, run_id="run-1"):
    return {run_id: [{"position": 0, "url_raw": url_raw, "domain": None, "title": title}]}


def test_title_fallback_recovers_citation_never_seen_by_aux_capture():
    """No row in resolution_map at all (the aux capture never matched this
    exact URL) -- the old is_dead-based eligibility check would have
    missed this case entirely, since it only fires on a resolution that
    exists and is flagged dead."""
    src = _gemini_source(WRAP, "acme.com")
    resolved, stats = resolve_sources_by_run(src, {}, run_engine_by_id={"run-1": GEMINI_ENGINE})
    row = resolved["run-1"][0]

    assert row["resolution_method"] == "title_fallback"
    assert row["url_raw"] == "https://acme.com/"
    assert row["domain"] == "acme.com"
    assert row["url_cited"] == WRAP
    assert stats.citations_title_fallback_attempted == 1
    assert stats.citations_title_fallback_recovered == 1
    assert stats.citations_title_fallback_distinct_domains == 1


def test_title_fallback_recovers_citation_flagged_dead_by_a_non_dead_bucket():
    """http_403 / paywalled / server_error / other keep url_raw unswapped
    without being flagged resolution_dead -- 86% of the real unresolved
    population falls here, not in DEAD_BUCKETS. Gating fallback eligibility
    on is_dead (the bug this fix corrects) would silently skip this case."""
    src = _gemini_source(WRAP, "acme.com")
    m = {WRAP: ResolvedCitation(WRAP, WRAP, None, "http_403")}
    resolved, stats = resolve_sources_by_run(src, m, run_engine_by_id={"run-1": GEMINI_ENGINE})
    row = resolved["run-1"][0]

    assert row["resolution_dead"] is False  # http_403 is not in DEAD_BUCKETS
    assert row["resolution_method"] == "title_fallback"
    assert row["domain"] == "acme.com"
    assert stats.citations_title_fallback_recovered == 1


def test_title_fallback_skipped_without_engine_info():
    src = _gemini_source(WRAP, "acme.com")
    resolved, stats = resolve_sources_by_run(src, {})  # no run_engine_by_id
    row = resolved["run-1"][0]

    assert row["url_raw"] == WRAP
    assert row["resolution_method"] is None
    assert stats.citations_title_fallback_attempted == 0


def test_title_fallback_never_fires_for_a_non_gemini_engine():
    src = _gemini_source(WRAP, "acme.com")
    resolved, stats = resolve_sources_by_run(src, {}, run_engine_by_id={"run-1": "openai_chatgpt"})
    row = resolved["run-1"][0]

    assert row["url_raw"] == WRAP
    assert row["resolution_method"] is None
    assert stats.citations_title_fallback_attempted == 0


def test_title_fallback_skipped_when_title_has_no_usable_domain():
    src = _gemini_source(WRAP, "10 Best AI Receptionists for Dental Offices in 2026")
    resolved, stats = resolve_sources_by_run(src, {}, run_engine_by_id={"run-1": GEMINI_ENGINE})
    row = resolved["run-1"][0]

    assert row["url_raw"] == WRAP  # left as the unresolved wrapper
    assert row["resolution_method"] is None
    assert stats.citations_title_fallback_attempted == 1
    assert stats.citations_title_fallback_recovered == 0


def test_title_fallback_does_not_refire_on_an_already_reachable_citation():
    m = {WRAP: ResolvedCitation(WRAP, "https://real.example/page", "real.example", "ok")}
    resolved, stats = resolve_sources_by_run(
        _gemini_source(WRAP, "other.example"), m, run_engine_by_id={"run-1": GEMINI_ENGINE}
    )
    row = resolved["run-1"][0]

    assert row["url_raw"] == "https://real.example/page"  # the real swap wins, title never consulted
    assert row["resolution_method"] is None
    assert stats.citations_title_fallback_attempted == 0


def test_title_fallback_domains_collapse_to_shared_synthetic_root_url():
    """Two different unresolved citations recovered to the same domain via
    title collapse onto the identical synthetic root URL -- a real,
    reported side effect (no real path is known), not silently absorbed."""
    wrap_b = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/BBB"
    src = {
        "run-1": [
            {"position": 0, "url_raw": WRAP, "domain": None, "title": "acme.com"},
            {"position": 1, "url_raw": wrap_b, "domain": None, "title": "acme.com"},
        ]
    }
    _, stats = resolve_sources_by_run(src, {}, run_engine_by_id={"run-1": GEMINI_ENGINE})

    assert stats.citations_title_fallback_recovered == 2
    assert stats.citations_title_fallback_distinct_domains == 1
    assert stats.unique_urls_after == 1


def test_title_domain_match_rate_by_engine():
    sources_by_run = {
        "g1": [{"url_raw": "https://acme.com/x", "title": "acme.com"}],
        "g2": [{"url_raw": WRAP, "title": "acme.com"}],  # excluded: no ground truth
        "o1": [{"url_raw": "https://acme.com/y", "title": "Why We Switched to Acme"}],  # real title, no domain match
    }
    run_engine_by_id = {"g1": GEMINI_ENGINE, "g2": GEMINI_ENGINE, "o1": "openai_chatgpt"}
    result = title_domain_match_rate_by_engine(sources_by_run, run_engine_by_id)

    assert result[GEMINI_ENGINE] == {"checked": 1, "matched": 1, "match_rate": 1.0}
    assert result["openai_chatgpt"] == {"checked": 1, "matched": 0, "match_rate": 0.0}


def test_title_domain_match_rate_by_engine_excludes_title_fallback_rows():
    """A title-fallback-recovered row's url_raw is itself derived from its
    title -- counting it as independent ground truth would trivially
    inflate Gemini's match rate to 100% by construction, not by evidence."""
    sources_by_run = {
        "g1": [
            {
                "url_raw": "https://acme.com/",
                "title": "acme.com",
                "resolution_method": "title_fallback",
            }
        ],
    }
    result = title_domain_match_rate_by_engine(sources_by_run, {"g1": GEMINI_ENGINE})
    assert result == {}
