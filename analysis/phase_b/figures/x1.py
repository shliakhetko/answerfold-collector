"""X1 -- Engine against modal url_class (EXPLORATORY, pinned EXPERIMENT.md
/ Notion Part 3.3).

Table: 4 engines x 6 url_classes, cell = share of that engine's
citations. Flag: is the modal class identical across all four engines --
if not, directory advice becomes engine-specific rather than universal
(spec). Unit: the citation instance, per engine, intents 1-6, read from
citation_panel -- the shared url_class join; this module recomputes no
class of its own.

Labelled exploratory: no pre-registered threshold, generated after
seeing the data (Notion Part 3.3's own framing).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.phase_b.citation_panel import CitationRow, build_citation_panel
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "X1"
# A modal class decided by less than this margin over the runner-up is
# flagged "weak" (2026-09-23, round 2 review) -- a near-tie shouldn't
# be reported with the same confidence as a decisive gap.
WEAK_MODAL_MARGIN_THRESHOLD = 0.10
S1_CROSSTAB_NOTE = (
    "vendor page + vendor listicle here are RAW url_class label shares (this figure applies no "
    "publisher_is_in_category filter) -- they will not match S1's vendor-published share, which is decided "
    "by publisher_is_in_category, not the label text. See S1's url_class_publisher_crosstab_by_engine and "
    "raw_vendor_page_plus_listicle_share_by_engine notes for the reconciling numbers."
)


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

    value: dict[str, dict[str, float] | None] = {}
    denominator_before_exclusion: dict[str, int] = {}
    denominator_after_exclusion: dict[str, int] = {}
    n_included: dict[str, int] = {}
    n_excluded: dict[str, dict[str, int]] = {}
    modal_class_by_engine: dict[str, str | None] = {}
    modal_margin_by_engine: dict[str, float | None] = {}
    weak_modal_by_engine: dict[str, bool | None] = {}
    raw_vendor_share_by_engine: dict[str, float | None] = {}
    review_site_share_by_engine: dict[str, float | None] = {}
    untestable_engines: dict[str, str] = {}

    for engine in sorted(rows_by_engine):
        engine_rows = rows_by_engine[engine]
        unclassified = sum(1 for r in engine_rows if r.url_class is None)
        classified = [r for r in engine_rows if r.url_class is not None]

        denominator_before_exclusion[engine] = len(engine_rows)
        denominator_after_exclusion[engine] = len(classified)
        n_included[engine] = len(classified)
        n_excluded[engine] = {"unclassified": unclassified} if unclassified else {}

        if not classified:
            value[engine] = None
            modal_class_by_engine[engine] = None
            modal_margin_by_engine[engine] = None
            weak_modal_by_engine[engine] = None
            raw_vendor_share_by_engine[engine] = None
            review_site_share_by_engine[engine] = None
            untestable_engines[engine] = "zero classified citations for this engine"
            continue

        class_counts: dict[str, int] = {}
        for r in classified:
            class_counts[r.url_class] = class_counts.get(r.url_class, 0) + 1
        total = len(classified)
        value[engine] = {cls: count / total for cls, count in class_counts.items()}
        modal_class_by_engine[engine] = max(class_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        ranked = sorted(class_counts.values(), reverse=True)
        top_share = ranked[0] / total
        runner_up_share = (ranked[1] / total) if len(ranked) > 1 else 0.0
        modal_margin_by_engine[engine] = top_share - runner_up_share
        weak_modal_by_engine[engine] = modal_margin_by_engine[engine] < WEAK_MODAL_MARGIN_THRESHOLD
        raw_vendor_share_by_engine[engine] = value[engine].get("vendor page", 0.0) + value[engine].get(
            "vendor listicle", 0.0
        )
        review_site_share_by_engine[engine] = value[engine].get("review site", 0.0)

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
        )

    testable_modal_classes = {v for e, v in modal_class_by_engine.items() if v is not None}
    modal_class_identical = len(testable_modal_classes) == 1 and len(modal_class_by_engine) == len(rows_by_engine)

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold=None,  # exploratory -- no pre-registered threshold
        reason_untestable=None,
        notes={
            "modal_class_by_engine": modal_class_by_engine,
            "modal_class_identical_across_engines": modal_class_identical,
            "modal_margin_by_engine": modal_margin_by_engine,
            "weak_modal_by_engine": weak_modal_by_engine,
            "raw_vendor_share_by_engine": raw_vendor_share_by_engine,
            "review_site_share_by_engine": review_site_share_by_engine,
            "s1_crosstab_note": S1_CROSSTAB_NOTE,
        },
    )
