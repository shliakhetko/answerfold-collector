import json

import pytest

from analysis.phase_b.figures import k2
from analysis.phase_b.reading import FrozenInputs


def _run(run_id, engine, *, intent_id=1, error=None, raw_response_text="some text"):
    return {
        "run_id": run_id,
        "engine": engine,
        "intent_id": intent_id,
        "paraphrase_id": 1,
        "error": error,
        "raw_response_text": raw_response_text,
        "is_manual": None,
    }


def _source(url, *, domain=None):
    return {"position": 1, "url_raw": url, "domain": domain}


def _registry(url, url_class="other", publisher_is_in_category=False):
    return {
        "url_normalized": url,
        "url_class": url_class,
        "publisher_is_in_category": publisher_is_in_category,
        "settled_class": url_class,
        "settled_publisher_is_in_category": publisher_is_in_category,
    }


def _mention(run_id, surface_form):
    return {"run_id": run_id, "surface_form": surface_form, "evidence_span": "x", "is_recommended": False, "votes": []}


def _entity(entity_id, homepage_url, meets_frame="in"):
    return {"entity_id": entity_id, "homepage_url": homepage_url, "surface_forms": entity_id, "meets_frame": meets_frame}


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


def test_basic_concentration_pooled_and_per_engine():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a-domain.com/x"), _source("https://a-domain.com/y"), _source("https://b-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))

    assert result.value["pooled"]["included"] == 1.0  # only 2 distinct domains, both in top 10
    assert result.notes["unique_domains"]["pooled"]["included"] == 2
    assert result.notes["count_to_50pct"]["pooled"]["included"] == 1  # a-domain.com alone has 2/3 >= 50%
    assert result.value["openai_chatgpt"]["included"] == result.value["pooled"]["included"]


def test_own_domain_included_vs_excluded_population():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://acme.com/x"), _source("https://thirdparty.com")]}
    entities = [_entity("acme", "https://acme.com")]
    result = k2.compute(_inputs(raw_runs, sources_by_run, entities=entities))

    assert result.notes["unique_domains"]["pooled"]["included"] == 2
    assert result.notes["unique_domains"]["pooled"]["excluded_population"] == 1  # acme.com dropped
    assert result.denominator_after_exclusion["pooled"]["included"] == 2
    assert result.denominator_after_exclusion["pooled"]["excluded_population"] == 1


def test_excluded_self_citation_keeps_competitor_listicle():
    """2026-09-23, owner review Fix B: a competitor's own vendor listicle
    must NOT be dropped under Rule 1 just because the listicle's
    publisher is also a population company -- only a genuine
    self-citation (the domain of an entity actually mentioned in that
    same run) is excluded. Rule 2 (excluded_population) still drops it,
    since cloudtalk.io is itself an in-frame entity regardless of what
    that specific run was about."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    # r1 mentions Acme (matches _entity's surface_forms=entity_id below);
    # the citation is to CloudTalk's own listicle (a different population
    # company, not mentioned in r1)
    mentions = [_mention("r1", "acme")]
    sources_by_run = {"r1": [_source("https://cloudtalk.io/blog/best-x")]}
    entities = [_entity("acme", "https://acme.com"), _entity("cloudtalk", "https://cloudtalk.io")]
    result = k2.compute(_inputs(raw_runs, sources_by_run, entities=entities, mentions_raw=mentions))

    assert result.notes["unique_domains"]["pooled"]["excluded_self_citation"] == 1  # kept -- not a self-citation
    assert result.notes["unique_domains"]["pooled"]["excluded_population"] == 0  # dropped -- CloudTalk is in-frame


def test_excluded_self_citation_drops_genuine_self_citation():
    raw_runs = [_run("r1", "openai_chatgpt")]
    mentions = [_mention("r1", "acme")]  # matches _entity's surface_forms=entity_id below
    sources_by_run = {"r1": [_source("https://acme.com/pricing")]}
    entities = [_entity("acme", "https://acme.com")]
    result = k2.compute(_inputs(raw_runs, sources_by_run, entities=entities, mentions_raw=mentions))

    assert result.notes["unique_domains"]["pooled"]["excluded_self_citation"] == 0  # dropped -- genuine self-citation


def test_top_n_share_computed_correctly():
    # top-10 share = 100% (only one domain)
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://only-domain.com") for _ in range(5)]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert result.value["pooled"]["included"] == 1.0

    # spread across 30 distinct domains, one citation each -> top 10 share 10/30
    raw_runs2 = [_run("r2", "openai_chatgpt")]
    sources_by_run2 = {"r2": [_source(f"https://d{i}.com") for i in range(30)]}
    result2 = k2.compute(_inputs(raw_runs2, sources_by_run2))
    assert result2.value["pooled"]["included"] == pytest.approx(10 / 30)


def test_no_verdict_or_threshold_language_anywhere():
    """2026-09-24 review feedback: K2 carries no pass/fail verdict bands
    or 40%/60% threshold language -- it is a pre-specified table, not a
    claim with a decision rule."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source(f"https://d{i}.com") for i in range(20)]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert result.threshold is None
    assert "verdict" not in result.notes
    assert result.value["pooled"]["included"] == pytest.approx(10 / 20)


def test_capterra_and_g2_presence_per_treatment():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {
        "r1": [
            _source("https://capterra.com/p/x"),
            _source("https://capterra.com/p/y"),
            _source("https://g2.com/p/z"),
            _source("https://other-domain.com"),
        ]
    }
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert result.notes["capterra_present"]["pooled"]["included"] is True
    assert result.notes["capterra_count"]["pooled"]["included"] == 2
    assert result.notes["g2_present"]["pooled"]["included"] is True
    assert result.notes["g2_count"]["pooled"]["included"] == 1
    # neither is anyone's own domain -- present under every treatment
    assert result.notes["capterra_present"]["pooled"]["excluded_population"] is True
    assert result.notes["g2_present"]["pooled"]["excluded_self_citation"] is True


def test_capterra_absent():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://other-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert result.notes["capterra_present"]["pooled"]["included"] is False
    assert result.notes["capterra_count"]["pooled"]["included"] == 0
    assert result.notes["g2_present"]["pooled"]["included"] is False


def test_malformed_domain_excluded_and_counted():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("not a url with spaces"), _source("https://real-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert result.n_excluded["pooled"]["malformed_domain"] == 1
    assert result.denominator_before_exclusion["pooled"] == 2
    assert result.denominator_after_exclusion["pooled"]["included"] == 1


def test_modal_url_class_in_top_10_table():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a-domain.com/1"), _source("https://a-domain.com/2")]}
    registry = [_registry("https://a-domain.com/1", "vendor page"), _registry("https://a-domain.com/2", "vendor page")]
    result = k2.compute(_inputs(raw_runs, sources_by_run, registry))
    top = result.notes["top_10_ranked"]["pooled"]["included"]
    assert top[0]["domain"] == "a-domain.com"
    assert top[0]["modal_url_class"] == "vendor page"
    assert top[0]["count"] == 2


def test_cumulative_share_curve_matches_rank_shares():
    raw_runs = [_run("r1", "openai_chatgpt")]
    # 2/3 to domain a, 1/3 to domain b -- cumulative after rank 1 is 2/3,
    # after rank 2 is 1.0
    sources_by_run = {
        "r1": [_source("https://a-domain.com/x"), _source("https://a-domain.com/y"), _source("https://b-domain.com")]
    }
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    curve = result.notes["cumulative_share_curve"]["pooled"]["included"]
    assert curve == [2 / 3, 1.0]


def test_branded_intent_7_excluded_via_citation_panel():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=7)]
    sources_by_run = {"r1": [_source("https://a-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert result.status == "untestable"


def test_share_present_per_engine_and_pooled_no_blended_overall():
    """2026-09-23, owner review Fix 2 (still true after the 2026-09-24
    verdict removal): K2 is a table, not a confirmatory claim -- there is
    no single blended overall value field, only a share per scope (pooled
    + each engine) plus an explanatory note."""
    engines = ["openai_chatgpt", "anthropic_claude"]
    raw_runs = [_run(f"r{i}", e) for i, e in enumerate(engines)]
    sources_by_run = {f"r{i}": [_source("https://only-domain.com")] for i in range(len(engines))}
    result = k2.compute(_inputs(raw_runs, sources_by_run))

    for scope in ["pooled", "openai_chatgpt", "anthropic_claude"]:
        assert result.value[scope]["included"] == 1.0
    assert "overall_value" not in result.notes
    assert "table" in result.notes["scope_reading_note"].lower()
    assert "pooled" in result.notes["scope_reading_note"].lower()


def test_treatment_rule_note_and_pre_reg_bands_note_present():
    """2026-09-24 review feedback (second pass): the footer must state
    three things only -- what the pre-reg set (quoted, with line number),
    that it was a decision rule dropped after seeing the data (with the
    ledger entry id), and the observed numbers -- never a forecast."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert "excluded_self_citation" in result.notes["treatment_rule_note"]
    assert "excluded_population" in result.notes["treatment_rule_note"]

    note = result.notes["pre_reg_bands_note"]
    assert "EXPERIMENT.md line 315" in note  # what the pre-reg set, quoted
    assert "40" in note and "60" in note
    assert "decision rule" in note and "dropped after seeing the data" in note
    assert "k2-verdict-threshold-removed-2026-09-24" in note  # the ledger entry id
    assert "openai_chatgpt" in note  # observed numbers, per engine


def test_no_expected_language_anywhere_in_k2_output():
    """The word 'expected' in any form (case-insensitive) may never appear
    anywhere in a K2 FigureResult -- the 40%/60% band was a decision rule,
    never a forecast, and framing it as one invents a prediction the
    owner never made."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    serialized = json.dumps(result.to_json()).lower()
    assert "expected" not in serialized


def test_headline_choice_stated_as_decided():
    """2026-09-25 owner decision (k2-headline-raw-self-citation-2026-09-25):
    the headline exclusion rule is no longer open -- K2's output states
    which treatment is the headline (raw/included) and which is the
    second view (excluded_self_citation), not that the choice is pending."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    note = result.notes["pre_reg_bands_note"].lower()
    assert "headline = raw" in note
    assert "still open" not in note and "pending owner decision" not in note


def test_retell_dapta_footnote_present():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a-domain.com")]}
    result = k2.compute(_inputs(raw_runs, sources_by_run))
    assert "retellai.com" in result.notes["retell_dapta_footnote"]
    assert "dapta.ai" in result.notes["retell_dapta_footnote"]
