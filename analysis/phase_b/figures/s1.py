"""S1 -- Vendor share of citations (CONFIRMATORY, pinned EXPERIMENT.md /
Notion Part 3.1).

Claim: vendor-published pages, not neutral sources, are the modal citation
in this category. Unit: the citation instance. Denominator: all citation
instances in the six unbranded intents (1-6), per engine -- read from
analysis.phase_b.citation_panel, the one url_class join every figure that
needs url_class reuses (this module recomputes no class of its own).

"Vendor-published is decided by publisher_is_in_category, not by the
label text" (spec, verbatim): the primary-test comparison bucket is
publisher_is_in_category is True, standing alongside every other
citation's raw url_class label -- not the "vendor page"/"vendor listicle"
labels themselves, which a classifier can attach to a listicle that isn't
actually published by a category vendor (publisher_is_in_category=False),
or withhold from a vendor's own page mislabeled "review site". The full
6-class raw breakdown is reported separately, beside it, exactly as the
spec asks for -- never substituted for the primary test.

Citations with no url_registry match (url_class is None -- including
every Gemini title-fallback synthetic URL, which the registry predates by
11 days) are excluded from the denominator and counted explicitly, never
silently dropped and never folded into "other".

publisher_is_in_category itself already reflects the domain-ownership
override applied once, upstream, in citation_panel.py's
build_citation_panel (2026-09-23, round 2 review) -- a known in-frame
vendor's own content-marketing pages no longer read False just because
A5's per-URL judgment treated a listicle-style title as third-party
content. This module does not need to know that correction happened; it
just reads the corrected field.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.coding.registrable_domain import registrable_domain
from analysis.phase_b.citation_panel import (
    CitationRow,
    build_citation_panel,
    mentioned_own_domains_by_run,
    url_class_publisher_crosstab,
)
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "S1"
VENDOR_BUCKET = "vendor-published"
# Notion Part 3.1 (S1, secondary figure), verbatim: "vendor share at or
# above 55% supports. Below 25% falsifies. 25 to 55 inconclusive, observed
# number stated plainly."
SECONDARY_SUPPORTED_THRESHOLD = 0.55
SECONDARY_FALSIFIED_THRESHOLD = 0.25
# Notion Part 3.1 (S1, primary test), verbatim: "Passes if: it holds in at
# least 3 of the 4 engines."
OVERALL_PASS_MIN_ENGINES = 3
SOURCE_OF_TRUTH_NOTE = (
    "url_class and publisher_is_in_category are two fields from ONE A5 classification call, not two "
    "classifier schemes (analysis/coding/a5_url_classification.py's _PROMPT_TEMPLATE asks for both keys "
    "in a single JSON object per URL). Empirically (2026-09-23 round 2 review): publisher_is_in_category=True "
    "is concentrated almost entirely in the 'vendor page'/'vendor listicle' labels, but not 100% contained "
    "there -- some page/listicle-labeled citations are genuine third-party content (a listicle published by "
    "an independent review site, not a vendor doing content marketing), which is exactly why 'vendor share = "
    "page + listicle' (raw label only) is not the same number as this figure's vendor-published share, by "
    "design. A separate, real gap in A5's raw judgment (a known vendor's own content-marketing pages reading "
    "False despite an unambiguous domain) is corrected upstream via a domain-ownership override -- see "
    "citation_panel.py::all_in_frame_owned_domains's docstring and the 2026-09-23 publisher-override ledger "
    "entry. See url_class_publisher_crosstab_by_engine for the full breakdown and "
    "raw_vendor_page_plus_listicle_share_by_engine for the raw-label comparison number."
)
# Round 5 fix (k2-headline-raw-self-citation-2026-09-25's sibling decision
# for S1, s1-domain-ownership-override-approved-2026-09-25 in the ambiguity
# ledger): headline uses the domain-ownership override (real ground truth
# for a domain S1 already has it for wins over a classifier miss); the
# classifier-only number shows what publisher_is_in_category alone (before
# that override) would have said, per engine, so the override's effect on
# both the primary pass/fail and the secondary verdict is visible, not
# just asserted.
# Pre-freeze check (2026-09-25, gemini-citation-count-scope-note ledger):
# citation_panel's per-engine denominator (this figure's scope) is deliberately
# narrower than a whole-dataset citation count -- stated here so the gap
# doesn't read as an error when the two are compared side by side.
DENOMINATOR_SCOPE_NOTE = (
    "This figure's denominator counts only unbranded-intent (1-6), usable-run citations "
    "(citation_panel.build_citation_panel's scope, analysis/phase_b/exclusions.py::usable_run_ids) and will not "
    "match a whole-dataset citation count such as domain_ownership.BuildReport.gemini_citations_total, which "
    "additionally includes branded intent 7 -- for google_gemini specifically, 7201 (all intents, all runs) = "
    "1064 branded-intent-7 citations + 6137 unbranded-usable citations (this figure's scope), with none lost to "
    "the usable-run filter (2026-09-25 reconciliation)."
)
CLASSIFIER_ONLY_NOTE = (
    "classifier_only_value/classifier_only_primary_pass_by_engine/classifier_only_secondary_verdict_by_engine "
    "recompute this figure's tests with the domain-ownership override backed out (CitationRow."
    "publisher_is_in_category_overridden reconstructs the pre-override value: it is only ever True when the "
    "pre-override value was not True and this override forced it to True, so a classifier-only bucket treats "
    "any overridden row as False). verdict_flip_by_engine states, per engine, whether either the primary "
    "pass/fail or the secondary verdict differs between the override headline and the classifier-only number."
)


def _bucket(row: CitationRow) -> str:
    """The primary-test bucket a citation falls into: vendor-published
    (publisher_is_in_category is True) regardless of its raw url_class
    label, else that raw label."""
    if row.publisher_is_in_category is True:
        return VENDOR_BUCKET
    return row.url_class  # never None here -- caller already filtered classified rows


def _classifier_only_bucket(row: CitationRow) -> str:
    """Round 5 fix: the bucket this citation would fall into WITHOUT the
    domain-ownership override applied in citation_panel.build_citation_panel
    -- i.e. A5's classifier call alone (or the human settled-spot-check
    correction, whichever `publisher_is_in_category` reflects pre-override).
    `publisher_is_in_category_overridden` is true only when the pre-override
    value was not True and this override forced it to True, so the
    pre-override value for an overridden row is reconstructable as False
    without needing a second upstream field."""
    classifier_only_poc = False if row.publisher_is_in_category_overridden else row.publisher_is_in_category
    if classifier_only_poc is True:
        return VENDOR_BUCKET
    return row.url_class


def _secondary_verdict(share: float | None) -> str | None:
    if share is None:
        return None
    if share >= SECONDARY_SUPPORTED_THRESHOLD:
        return "supported"
    if share < SECONDARY_FALSIFIED_THRESHOLD:
        return "falsified"
    return "inconclusive"


def compute(inputs: FrozenInputs) -> FigureResult:
    rows, _panel_report = build_citation_panel(inputs)

    rows_by_engine: dict[str, list[CitationRow]] = defaultdict(list)
    for row in rows:
        rows_by_engine[row.engine].append(row)

    if not rows_by_engine:
        return FigureResult(
            figure_id=FIGURE_ID,
            status="untestable",
            value=None,
            denominator_before_exclusion=None,
            denominator_after_exclusion=None,
            n_included=None,
            n_excluded=None,
            threshold=None,
            reason_untestable="zero in-scope citations (no usable run in intents 1-6 with any source)",
        )

    mentioned_own_domains = mentioned_own_domains_by_run(inputs)

    value: dict[str, float | None] = {}
    denominator_before_exclusion: dict[str, int] = {}
    denominator_after_exclusion: dict[str, int] = {}
    n_included: dict[str, int] = {}
    n_excluded: dict[str, dict[str, int]] = {}
    primary_pass_by_engine: dict[str, bool | None] = {}
    secondary_verdict_by_engine: dict[str, str | None] = {}
    full_class_breakdown_by_engine: dict[str, dict[str, int]] = {}
    non_vendor_class_breakdown_by_engine: dict[str, dict[str, int]] = {}
    unique_url_vendor_share_by_engine: dict[str, float | None] = {}
    raw_vendor_page_plus_listicle_share_by_engine: dict[str, float | None] = {}
    vendor_share_including_unclassified_by_engine: dict[str, float | None] = {}
    vendor_share_excluding_mentioned_own_domain_by_engine: dict[str, float | None] = {}
    classifier_only_value: dict[str, float | None] = {}
    classifier_only_primary_pass_by_engine: dict[str, bool | None] = {}
    classifier_only_secondary_verdict_by_engine: dict[str, str | None] = {}
    verdict_flip_by_engine: dict[str, dict[str, bool] | None] = {}
    supporting_table: list[dict[str, Any]] = []
    untestable_engines: dict[str, str] = {}

    for engine in sorted(rows_by_engine):
        engine_rows = rows_by_engine[engine]
        unclassified = sum(1 for r in engine_rows if r.url_class is None)
        classified = [r for r in engine_rows if r.url_class is not None]

        denominator_before_exclusion[engine] = len(engine_rows)
        denominator_after_exclusion[engine] = len(classified)
        n_included[engine] = len(classified)
        n_excluded[engine] = {"unclassified": unclassified} if unclassified else {}

        full_class_breakdown_by_engine[engine] = {}
        for r in classified:
            full_class_breakdown_by_engine[engine][r.url_class] = (
                full_class_breakdown_by_engine[engine].get(r.url_class, 0) + 1
            )
        raw_page_plus_listicle = full_class_breakdown_by_engine[engine].get(
            "vendor page", 0
        ) + full_class_breakdown_by_engine[engine].get("vendor listicle", 0)
        raw_vendor_page_plus_listicle_share_by_engine[engine] = (
            raw_page_plus_listicle / len(classified) if classified else None
        )

        if not classified:
            value[engine] = None
            primary_pass_by_engine[engine] = None
            secondary_verdict_by_engine[engine] = None
            unique_url_vendor_share_by_engine[engine] = None
            vendor_share_including_unclassified_by_engine[engine] = None
            vendor_share_excluding_mentioned_own_domain_by_engine[engine] = None
            classifier_only_value[engine] = None
            classifier_only_primary_pass_by_engine[engine] = None
            classifier_only_secondary_verdict_by_engine[engine] = None
            verdict_flip_by_engine[engine] = None
            untestable_engines[engine] = "zero classified citations for this engine"
            continue

        bucket_counts: dict[str, int] = {}
        for r in classified:
            b = _bucket(r)
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

        vendor_count = bucket_counts.get(VENDOR_BUCKET, 0)
        other_counts = {k: v for k, v in bucket_counts.items() if k != VENDOR_BUCKET}
        max_other = max(other_counts.values()) if other_counts else 0
        primary_pass_by_engine[engine] = vendor_count >= max_other
        # Non-vendor-published citations only, bucketed by raw url_class --
        # unlike full_class_breakdown_by_engine (every classified citation's
        # raw label, vendor-published or not), this partitions cleanly
        # alongside the vendor-published bucket: the two sum to
        # denominator_after_exclusion with no double-count, which is what a
        # stacked-bar chart of this figure needs (a "vendor page"-labeled
        # citation that IS vendor-published belongs only in the vendor
        # bucket, not counted again under its raw label).
        non_vendor_class_breakdown_by_engine[engine] = other_counts

        share = vendor_count / len(classified)
        value[engine] = share
        secondary_verdict_by_engine[engine] = _secondary_verdict(share)

        # Round 5 fix: the same primary/secondary tests, recomputed with
        # the domain-ownership override backed out -- shows what the
        # numbers and verdicts would be from A5's classifier call alone,
        # alongside the override-inclusive headline above, per engine.
        classifier_only_bucket_counts: dict[str, int] = {}
        for r in classified:
            b = _classifier_only_bucket(r)
            classifier_only_bucket_counts[b] = classifier_only_bucket_counts.get(b, 0) + 1
        classifier_only_vendor_count = classifier_only_bucket_counts.get(VENDOR_BUCKET, 0)
        classifier_only_other = {k: v for k, v in classifier_only_bucket_counts.items() if k != VENDOR_BUCKET}
        classifier_only_max_other = max(classifier_only_other.values()) if classifier_only_other else 0
        classifier_only_primary_pass = classifier_only_vendor_count >= classifier_only_max_other
        classifier_only_share = classifier_only_vendor_count / len(classified)
        classifier_only_verdict = _secondary_verdict(classifier_only_share)

        classifier_only_value[engine] = classifier_only_share
        classifier_only_primary_pass_by_engine[engine] = classifier_only_primary_pass
        classifier_only_secondary_verdict_by_engine[engine] = classifier_only_verdict
        verdict_flip_by_engine[engine] = {
            "primary_pass_flipped": primary_pass_by_engine[engine] != classifier_only_primary_pass,
            "secondary_verdict_flipped": secondary_verdict_by_engine[engine] != classifier_only_verdict,
        }

        # Secondary comparison number (2026-09-23, round 2 review):
        # unclassified citations counted in the denominator as non-vendor --
        # never replaces the primary classified-only denominator above,
        # reported alongside it so a reader can see how much the exclusion
        # itself moves the number.
        vendor_share_including_unclassified_by_engine[engine] = vendor_count / denominator_before_exclusion[engine]

        # Secondary column (2026-09-23, round 2 review): vendor share
        # recomputed with every citation to the domain of an entity actually
        # mentioned in that same run excluded from both sides -- a citation
        # to the subject's own site is expected/self-referential, not
        # evidence about who else gets published.
        remaining = [
            r
            for r in classified
            if (registrable_domain(r.url_normalized) or r.url_normalized)
            not in mentioned_own_domains.get(r.run_id, set())
        ]
        if remaining:
            remaining_vendor_count = sum(1 for r in remaining if _bucket(r) == VENDOR_BUCKET)
            vendor_share_excluding_mentioned_own_domain_by_engine[engine] = remaining_vendor_count / len(remaining)
        else:
            vendor_share_excluding_mentioned_own_domain_by_engine[engine] = None

        unique_urls: dict[str, bool] = {}  # url_normalized -> is vendor-published
        for r in classified:
            unique_urls.setdefault(r.url_normalized, r.publisher_is_in_category is True)
        unique_url_vendor_share_by_engine[engine] = (
            sum(1 for is_vendor in unique_urls.values() if is_vendor) / len(unique_urls) if unique_urls else None
        )

        for bucket_name, count in sorted(bucket_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            supporting_table.append(
                {
                    "engine": engine,
                    "bucket": bucket_name,
                    "count": count,
                    "share_of_classified": count / len(classified),
                }
            )

    if untestable_engines and len(untestable_engines) == len(rows_by_engine):
        return FigureResult(
            figure_id=FIGURE_ID,
            status="untestable",
            value=None,
            denominator_before_exclusion=None,
            denominator_after_exclusion=None,
            n_included=None,
            n_excluded=None,
            threshold=None,
            reason_untestable="zero classified citations across every engine",
            supporting_table=supporting_table,
        )

    engines_passing_primary = [e for e, passed in primary_pass_by_engine.items() if passed]
    overall_primary_pass = len(engines_passing_primary) >= OVERALL_PASS_MIN_ENGINES

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold={
            "secondary_supported_at_or_above": SECONDARY_SUPPORTED_THRESHOLD,
            "secondary_falsified_below": SECONDARY_FALSIFIED_THRESHOLD,
            "overall_pass_min_engines": OVERALL_PASS_MIN_ENGINES,
        },
        reason_untestable=None,
        supporting_table=supporting_table,
        notes={
            "primary_test_pass_by_engine": primary_pass_by_engine,
            "overall_primary_pass": overall_primary_pass,
            "engines_passing_primary": engines_passing_primary,
            "secondary_verdict_by_engine": secondary_verdict_by_engine,
            "full_class_breakdown_by_engine": full_class_breakdown_by_engine,
            "non_vendor_class_breakdown_by_engine": non_vendor_class_breakdown_by_engine,
            "unique_url_vendor_share_by_engine": unique_url_vendor_share_by_engine,
            "raw_vendor_page_plus_listicle_share_by_engine": raw_vendor_page_plus_listicle_share_by_engine,
            "vendor_share_including_unclassified_by_engine": vendor_share_including_unclassified_by_engine,
            "vendor_share_excluding_mentioned_own_domain_by_engine": vendor_share_excluding_mentioned_own_domain_by_engine,
            "url_class_publisher_crosstab_by_engine": url_class_publisher_crosstab(rows),
            "source_of_truth_note": SOURCE_OF_TRUTH_NOTE,
            "classifier_only_value": classifier_only_value,
            "classifier_only_primary_pass_by_engine": classifier_only_primary_pass_by_engine,
            "classifier_only_secondary_verdict_by_engine": classifier_only_secondary_verdict_by_engine,
            "verdict_flip_by_engine": verdict_flip_by_engine,
            "classifier_only_note": CLASSIFIER_ONLY_NOTE,
            "denominator_scope_note": DENOMINATOR_SCOPE_NOTE,
        },
    )
