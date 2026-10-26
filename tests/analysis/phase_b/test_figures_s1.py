from analysis.phase_b.figures import s1
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


def _source(url, *, title=None):
    return {"position": 1, "url_raw": url, "domain": None, "title": title}


def _registry(url, url_class, publisher_is_in_category):
    return {
        "url_normalized": url,
        "url_class": url_class,
        "publisher_is_in_category": publisher_is_in_category,
        "settled_class": url_class,
        "settled_publisher_is_in_category": publisher_is_in_category,
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


def test_vendor_bucket_defined_by_publisher_is_in_category_not_label():
    """A citation labeled 'review site' by the classifier but with
    publisher_is_in_category=True must still count toward vendor-published
    -- the spec's explicit instruction ("decided by publisher_is_in_category,
    not by the label text")."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example"), _source("https://b.example")]}
    registry = [
        _registry("https://a.example", "review site", True),  # vendor-published despite the label
        _registry("https://b.example", "review site", False),
    ]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.value["openai_chatgpt"] == 0.5
    assert result.notes["primary_test_pass_by_engine"]["openai_chatgpt"] is True  # 1 vs 1, tie goes to vendor


def test_primary_test_falsified_when_another_class_is_strictly_larger():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source(f"https://x{i}.example") for i in range(3)]}
    registry = [
        _registry("https://x0.example", "vendor page", True),
        _registry("https://x1.example", "review site", False),
        _registry("https://x2.example", "review site", False),
    ]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["primary_test_pass_by_engine"]["openai_chatgpt"] is False
    assert result.value["openai_chatgpt"] == 1 / 3


def test_overall_pass_requires_at_least_three_of_four_engines():
    engines = ["openai_chatgpt", "anthropic_claude", "google_gemini", "google_ai_mode"]
    raw_runs = [_run(f"r{i}", e) for i, e in enumerate(engines)]
    # first 3 engines: vendor-published wins; 4th: review site wins
    sources_by_run = {}
    registry = []
    for i, e in enumerate(engines):
        url = f"https://{e}.example"
        sources_by_run[f"r{i}"] = [_source(url)]
        registry.append(_registry(url, "vendor page" if i < 3 else "review site", i < 3))
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["overall_primary_pass"] is True
    assert len(result.notes["engines_passing_primary"]) == 3


def test_secondary_verdict_thresholds():
    raw_runs = [_run("r1", "openai_chatgpt")]
    # 6/10 = 60% vendor -> supported
    sources_by_run = {"r1": [_source(f"https://v{i}.example") for i in range(6)] + [_source(f"https://n{i}.example") for i in range(4)]}
    registry = [_registry(f"https://v{i}.example", "vendor page", True) for i in range(6)]
    registry += [_registry(f"https://n{i}.example", "review site", False) for i in range(4)]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["secondary_verdict_by_engine"]["openai_chatgpt"] == "supported"


def test_unclassified_citations_excluded_from_denominator_and_counted():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://known.example"), _source("https://never-classified.example")]}
    registry = [_registry("https://known.example", "vendor page", True)]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.denominator_before_exclusion["openai_chatgpt"] == 2
    assert result.denominator_after_exclusion["openai_chatgpt"] == 1
    assert result.n_excluded["openai_chatgpt"] == {"unclassified": 1}
    assert result.value["openai_chatgpt"] == 1.0


def test_branded_intent_7_excluded_via_citation_panel():
    raw_runs = [_run("r1", "openai_chatgpt", intent_id=7)]
    sources_by_run = {"r1": [_source("https://a.example")]}
    registry = [_registry("https://a.example", "vendor page", True)]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.status == "untestable"


def test_full_class_breakdown_uses_raw_labels_not_vendor_bucket():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example"), _source("https://b.example")]}
    registry = [
        _registry("https://a.example", "vendor page", True),
        _registry("https://b.example", "forum", False),
    ]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["full_class_breakdown_by_engine"]["openai_chatgpt"] == {"vendor page": 1, "forum": 1}


def test_unique_url_share_dedups_repeated_citations():
    raw_runs = [_run("r1", "openai_chatgpt"), _run("r2", "openai_chatgpt", intent_id=2)]
    # same vendor URL cited twice, one distinct non-vendor URL once
    sources_by_run = {
        "r1": [_source("https://vendor.example")],
        "r2": [_source("https://vendor.example"), _source("https://other.example")],
    }
    registry = [
        _registry("https://vendor.example", "vendor page", True),
        _registry("https://other.example", "review site", False),
    ]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    # instance-level: 2 vendor / 3 total = 0.667; URL-level: 1 vendor / 2 distinct = 0.5
    assert result.value["openai_chatgpt"] == 2 / 3
    assert result.notes["unique_url_vendor_share_by_engine"]["openai_chatgpt"] == 0.5


def test_non_vendor_breakdown_excludes_vendor_labeled_but_vendor_published_rows():
    """A citation raw-labeled 'vendor page' with publisher_is_in_category=True
    belongs only in the vendor-published bucket -- it must not also appear
    under 'vendor page' in non_vendor_class_breakdown_by_engine, or a
    stacked chart built from vendor share + this breakdown would double-
    count past 100%."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example"), _source("https://b.example")]}
    registry = [
        _registry("https://a.example", "vendor page", True),  # vendor-published
        _registry("https://b.example", "vendor page", False),  # labeled the same, NOT vendor-published
    ]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    non_vendor = result.notes["non_vendor_class_breakdown_by_engine"]["openai_chatgpt"]
    assert non_vendor == {"vendor page": 1}  # only the non-vendor-published one
    vendor_count = round(result.value["openai_chatgpt"] * result.denominator_after_exclusion["openai_chatgpt"])
    assert vendor_count + sum(non_vendor.values()) == result.denominator_after_exclusion["openai_chatgpt"]


def test_zero_classified_citations_untestable_for_engine_only():
    raw_runs = [_run("r1", "openai_chatgpt"), _run("r2", "google_gemini")]
    sources_by_run = {"r1": [_source("https://never-classified.example")], "r2": [_source("https://known.example")]}
    registry = [_registry("https://known.example", "vendor page", True)]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.status == "ok"
    assert result.value["openai_chatgpt"] is None
    assert result.value["google_gemini"] == 1.0


def test_raw_vendor_page_plus_listicle_share_differs_from_primary_value():
    """The raw-label comparison number is NOT the same as the primary
    vendor-published share whenever a page/listicle-labeled citation has
    publisher_is_in_category=False -- the exact gap Fix 1 exists to make
    visible rather than silently conflate."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example"), _source("https://b.example")]}
    registry = [
        _registry("https://a.example", "vendor page", True),  # vendor-published
        _registry("https://b.example", "vendor listicle", False),  # page/listicle label, NOT vendor-published
    ]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.value["openai_chatgpt"] == 0.5  # primary: publisher_is_in_category-based
    assert result.notes["raw_vendor_page_plus_listicle_share_by_engine"]["openai_chatgpt"] == 1.0  # raw label only
    assert result.value["openai_chatgpt"] != result.notes["raw_vendor_page_plus_listicle_share_by_engine"]["openai_chatgpt"]


def test_vendor_share_including_unclassified_is_lower_than_primary():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://known.example"), _source("https://never-classified.example")]}
    registry = [_registry("https://known.example", "vendor page", True)]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.value["openai_chatgpt"] == 1.0  # classified-only (primary)
    assert result.notes["vendor_share_including_unclassified_by_engine"]["openai_chatgpt"] == 0.5  # 1 of 2 total


def test_vendor_share_excluding_mentioned_own_domain():
    """A citation to the mentioned entity's own domain is excluded from
    this secondary metric on both sides -- self-referential, not evidence
    about who else gets published."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://acme.com/pricing"), _source("https://thirdparty.example")]}
    registry = [
        _registry("https://acme.com/pricing", "vendor page", True),
        _registry("https://thirdparty.example", "review site", False),
    ]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    mentions = [_mention("r1", "Acme")]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry, entities=entities, mentions_raw=mentions))

    # primary (unrestricted): 1 vendor-published of 2 -> 0.5
    assert result.value["openai_chatgpt"] == 0.5
    # secondary: acme.com dropped (it's Acme's own domain, and Acme was
    # mentioned in r1) -> only thirdparty.example remains, 0 vendor-published of 1
    secondary = result.notes["vendor_share_excluding_mentioned_own_domain_by_engine"]["openai_chatgpt"]
    assert secondary == 0.0


def test_classifier_only_reconstructs_pre_override_bucket_and_flags_the_flip():
    """Round 5 fix: acme.com is an in-frame entity's own domain, so
    citation_panel's domain-ownership override forces this citation's
    publisher_is_in_category from False (A5's raw call) to True -- the
    headline (override-inclusive) number treats it as vendor-published,
    the classifier-only number reconstructs the pre-override False and
    buckets it under its raw 'review site' label instead. With only this
    one citation, that flips both the primary pass/fail and the secondary
    verdict, and verdict_flip_by_engine must say so."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://acme.com/blog")]}
    registry = [_registry("https://acme.com/blog", "review site", False)]
    entities = [_entity("acme", "https://acme.com", "Acme")]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry, entities=entities))

    # headline: override applied -> vendor-published, primary pass, secondary supported
    assert result.value["openai_chatgpt"] == 1.0
    assert result.notes["primary_test_pass_by_engine"]["openai_chatgpt"] is True
    assert result.notes["secondary_verdict_by_engine"]["openai_chatgpt"] == "supported"

    # classifier-only: pre-override False -> non-vendor ("review site"), primary fail, secondary falsified
    assert result.notes["classifier_only_value"]["openai_chatgpt"] == 0.0
    assert result.notes["classifier_only_primary_pass_by_engine"]["openai_chatgpt"] is False
    assert result.notes["classifier_only_secondary_verdict_by_engine"]["openai_chatgpt"] == "falsified"

    flip = result.notes["verdict_flip_by_engine"]["openai_chatgpt"]
    assert flip == {"primary_pass_flipped": True, "secondary_verdict_flipped": True}


def test_classifier_only_matches_headline_when_no_override_fired():
    """A citation with no domain-ownership override in play (independent
    third-party domain) must show identical headline and classifier-only
    numbers, with no flip flagged."""
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example")]}
    registry = [_registry("https://a.example", "vendor page", True)]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))

    assert result.notes["classifier_only_value"]["openai_chatgpt"] == result.value["openai_chatgpt"]
    flip = result.notes["verdict_flip_by_engine"]["openai_chatgpt"]
    assert flip == {"primary_pass_flipped": False, "secondary_verdict_flipped": False}


def test_url_class_publisher_crosstab_and_source_of_truth_note_present():
    raw_runs = [_run("r1", "openai_chatgpt")]
    sources_by_run = {"r1": [_source("https://a.example")]}
    registry = [_registry("https://a.example", "vendor page", True)]
    result = s1.compute(_inputs(raw_runs, sources_by_run, registry))
    assert result.notes["url_class_publisher_crosstab_by_engine"]["openai_chatgpt"]["vendor page"] == {
        "true": 1,
        "false": 0,
        "null": 0,
    }
    assert "a5_url_classification.py" in result.notes["source_of_truth_note"]
