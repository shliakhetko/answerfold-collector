"""Owner review (2026-09-23), Fix 1: S1 and X1 must read consistent counts
off the same shared citation_panel, even though their headline SHARES are
not literally equal by design (S1's vendor-published share is decided by
publisher_is_in_category; X1's per-class shares are raw url_class labels
only -- see each figure's own module docstring and
S1.notes["source_of_truth_note"]).

The real, always-true-by-construction invariant: S1's raw per-class
citation counts (full_class_breakdown_by_engine, vendor-published or not)
must equal X1's raw per-class counts (reconstructed from its per-class
shares and denominator) for every engine and every url_class. A mismatch
here would mean the two figures diverged on citation_panel scoping or
exclusion somehow -- exactly the class of bug the owner's review was
worried about, tested precisely rather than via a false equality.
"""
from analysis.phase_b.figures import s1, x1
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


def _source(url):
    return {"position": 1, "url_raw": url, "domain": None}


def _registry(url, url_class, publisher_is_in_category=False):
    return {
        "url_normalized": url,
        "url_class": url_class,
        "publisher_is_in_category": publisher_is_in_category,
        "settled_class": url_class,
        "settled_publisher_is_in_category": publisher_is_in_category,
    }


def _inputs(raw_runs, sources_by_run, url_registry_rows):
    return FrozenInputs(
        entities=[],
        alt_domains_rows=[],
        entity_verification_rows=[],
        mentions_raw=[],
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
        url_registry_rows=list(url_registry_rows),
    )


def _realistic_mixed_fixture():
    """Multiple engines, a genuine mix of vendor-published and
    non-vendor-published page/listicle citations (the exact shape that
    made S1's 89% and X1's 98% look like a mismatch during the owner's
    review), plus other classes and one unclassified citation."""
    raw_runs = [
        _run("r1", "anthropic_claude"),
        _run("r2", "anthropic_claude"),
        _run("r3", "openai_chatgpt"),
    ]
    sources_by_run = {
        "r1": [
            _source("https://vendor-a.example"),  # vendor page, vendor-published
            _source("https://vendor-b.example"),  # vendor listicle, vendor-published
            _source("https://third-party-listicle.example"),  # vendor listicle, NOT vendor-published
            _source("https://reviews.example"),  # review site
        ],
        "r2": [
            _source("https://never-classified.example"),  # unclassified
        ],
        "r3": [
            _source("https://vendor-c.example"),  # vendor page, vendor-published
            _source("https://forum.example"),  # forum
        ],
    }
    registry = [
        _registry("https://vendor-a.example", "vendor page", True),
        _registry("https://vendor-b.example", "vendor listicle", True),
        _registry("https://third-party-listicle.example", "vendor listicle", False),
        _registry("https://reviews.example", "review site", False),
        _registry("https://vendor-c.example", "vendor page", True),
        _registry("https://forum.example", "forum", False),
    ]
    return _inputs(raw_runs, sources_by_run, registry)


def test_s1_headline_share_genuinely_differs_from_x1_raw_page_plus_listicle():
    """Confirms the premise: without Fix 1's presentation change, reading
    S1's value and X1's page+listicle shares side by side looks like a
    mismatch -- because they measure different things, not because either
    is wrong."""
    inputs = _realistic_mixed_fixture()
    s1_result = s1.compute(inputs)
    x1_result = x1.compute(inputs)

    engine = "anthropic_claude"
    s1_share = s1_result.value[engine]
    x1_raw_share = x1_result.value[engine].get("vendor page", 0) + x1_result.value[engine].get("vendor listicle", 0)
    assert s1_share != x1_raw_share  # 2/3 vendor-published vs 3/3 raw page+listicle label


def test_s1_and_x1_raw_class_counts_reconcile_exactly():
    inputs = _realistic_mixed_fixture()
    s1_result = s1.compute(inputs)
    x1_result = x1.compute(inputs)

    for engine in s1_result.value:
        s1_breakdown = s1_result.notes["full_class_breakdown_by_engine"][engine]
        x1_shares = x1_result.value[engine]
        x1_denom = x1_result.denominator_after_exclusion[engine]
        assert x1_denom == s1_result.denominator_after_exclusion[engine]

        x1_counts = {cls: round(share * x1_denom) for cls, share in x1_shares.items()}
        assert s1_breakdown == x1_counts


def test_s1_non_vendor_breakdown_plus_vendor_count_equals_x1_raw_page_and_listicle():
    """The precise decomposition identity Fix 1's test replaces the
    owner's literal (false) ask with: vendor-published count + the
    non-vendor-published residual of page/listicle citations equals X1's
    raw page+listicle count -- proving S1's math correctly accounts for
    every page/listicle-labeled citation, whichever bucket it landed in."""
    inputs = _realistic_mixed_fixture()
    s1_result = s1.compute(inputs)
    x1_result = x1.compute(inputs)

    for engine in s1_result.value:
        non_vendor = s1_result.notes["non_vendor_class_breakdown_by_engine"][engine]
        non_vendor_page_and_listicle = non_vendor.get("vendor page", 0) + non_vendor.get("vendor listicle", 0)

        x1_shares = x1_result.value[engine]
        x1_denom = x1_result.denominator_after_exclusion[engine]
        x1_raw_page_and_listicle = round(
            (x1_shares.get("vendor page", 0) + x1_shares.get("vendor listicle", 0)) * x1_denom
        )

        # every page/listicle-labeled citation is accounted for exactly
        # once, either as vendor-published or in the non-vendor residual --
        # read directly off the crosstab rather than S1's overall
        # vendor_count, which also includes non-page/listicle
        # vendor-published citations (rare, e.g. a mislabeled "review
        # site" with publisher_is_in_category=True).
        crosstab = s1_result.notes["url_class_publisher_crosstab_by_engine"][engine]
        vendor_published_page_and_listicle = crosstab.get("vendor page", {}).get("true", 0) + crosstab.get(
            "vendor listicle", {}
        ).get("true", 0)
        assert vendor_published_page_and_listicle + non_vendor_page_and_listicle == x1_raw_page_and_listicle
