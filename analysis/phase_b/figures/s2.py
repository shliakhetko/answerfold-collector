"""S2 -- Mention without own citation (PRE-SPECIFIED TABLE, "the pitch",
pinned EXPERIMENT.md / Notion Part 3.2).

Unit: the run x company pair where mentioned = 1, per engine, intents
1-6, usable runs -- read from analysis.phase_b.run_company_panel.

Primary denominator: those pairs, restricted to answers that cite at
least one source (spec: "If a third of answers cite nobody at all, the
unrestricted number is inflated by answers that could not have cited
anyone"). "Restricted" is a subset of "unrestricted" (all mentioned
pairs) -- the restriction itself (dropped for zero sources) is reported
as its own n_excluded reason, separate from own_domain_cited being
unmeasurable (entity has no known domain), so a reader can tell which
exclusion did what.

own_domain_cited is three-valued (see run_company_panel.py); None is
excluded from both numerator and denominator here, matching the
established convention already used by
analysis.phase_b.domain_ownership.BuildReport.owned_true_share_by_engine
(measurable = true + false, null excluded entirely) -- a precedent
citation, not a fresh judgment call.

S3 (share of runs citing nothing) is a stated comparison number, not
computed here -- figures never call each other (see
analysis/phase_b/figures/__init__.py's docstring); the display layer
(scripts/results/render_s2_image.py) reads S3's own record from the seam file
directly.

Teardown: companies whose own domain is cited in under 10% of their own
qualifying mention events, gated by the pre-registered eligibility floor
(>=5 mentions across intents 1-6, within an engine -- reused from
run_company_panel's EligibilityReport, not reimplemented). A company with
too few mentions to qualify never enters this count, exactly as the floor
requires.

Depends on the alt_domains fix (2026-09-22, eb5d783/71f6c8f): if
entities_with_alternates is still zero on the inputs this runs against,
S2 carries the limitation line the spec calls for rather than silently
proceeding as if the fix were irrelevant.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS
from analysis.phase_b.exclusions import usable_run_ids
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs
from analysis.phase_b.run_company_panel import RunCompanyRow, build_run_company_panel

FIGURE_ID = "S2"
TEARDOWN_THRESHOLD = 0.10
ALTERNATES_LIMITATION_TEXT = (
    "S2 depends on the alt_domains fix (2026-09-22 own-domain-join-additional-gaps entry): "
    "entities_with_alternates is 0 on these inputs, meaning no entity's own_domain_cited match came via "
    "an alternate domain -- own_domain_cited here reflects only each entity's primary homepage domain. "
    "This limitation line is stated automatically whenever that count is 0, not asserted once and left "
    "to go stale."
)
# 2026-09-23, round 2 review: is "cited" the same thing across all
# four engines? Checked both the live collector adapters and the offline
# replay path (analysis/aggregate/response_fields.py) -- each engine's
# sources table is built from that engine's own inline citation/grounding
# mechanism, never a broader "everything consulted" list. Not a design
# decision here; a stated, checked fact.
# Pre-freeze check (2026-09-25): domain_ownership.BuildReport.
# owned_true_share_by_engine ("owned=True share by engine" printed by
# scripts/results/run_phase_b.py) pools all 7 intents including branded intent 7;
# S2 restricts to unbranded intents 1-6 per the pre-registration. Mechanically
# confirmed: recomputing that build report restricted to intents 1-6 only
# makes it match S2's unrestricted share exactly. Stated here so the gap
# doesn't read as an error, especially for google_ai_mode.
NOT_CITED_VS_NOT_OWNED_NOTE = (
    "S2's own_domain_cited share and domain_ownership.BuildReport.owned_true_share_by_engine (the \"owned=True "
    "share by engine\" line scripts/results/run_phase_b.py prints) are not expected to match: the build report pools "
    "all 7 intents including branded intent 7, while S2 restricts to the six unbranded intents 1-6 per the "
    "pre-registration -- this matters far more for google_ai_mode than the other three engines because its "
    "own-domain citation rate collapses from ~49% on unbranded intents to ~29% on branded intent 7 (~20 points, "
    "roughly double google_gemini's and far larger than anthropic_claude's or openai_chatgpt's), which is why "
    "its all-intents 'not owned' figure (55.5%) sits visibly further from S2's unbranded-only figure (~51%) "
    "than for any other engine."
)
SOURCE_DEFINITION_BY_ENGINE = {
    "anthropic_claude": (
        "citations of type web_search_result_location attached to text blocks "
        "(collector/engines/anthropic_claude.py, analysis/aggregate/response_fields.py::_sources_anthropic)"
    ),
    "openai_chatgpt": (
        "annotations of type url_citation attached to output text "
        "(collector/engines/openai_chatgpt.py::_extract_sources, "
        "analysis/aggregate/response_fields.py::_sources_openai) -- inline citations only, "
        "not a broader consulted-sources list; verified identical in both the live adapter and the "
        "offline replay path"
    ),
    "google_gemini": (
        "grounding_chunks from grounding_metadata (analysis/aggregate/response_fields.py::_sources_gemini)"
    ),
    "google_ai_mode": "the references list SerpApi returns for the google_ai_mode engine (scraper-mediated)",
}


def _measurable_share(rows: list[RunCompanyRow]) -> tuple[float | None, int, int, int]:
    """(share_false, denominator_before, denominator_after, n_unmeasurable).
    share_false = count(own_domain_cited is False) / count(own_domain_cited
    is not None) -- None excluded from both sides."""
    measurable = [r for r in rows if r.own_domain_cited is not None]
    unmeasurable = len(rows) - len(measurable)
    if not measurable:
        return None, len(rows), 0, unmeasurable
    false_count = sum(1 for r in measurable if r.own_domain_cited is False)
    return false_count / len(measurable), len(rows), len(measurable), unmeasurable


def compute(inputs: FrozenInputs) -> FigureResult:
    rows, eligibility, panel_report = build_run_company_panel(inputs)
    usable_ids = usable_run_ids(inputs.raw_runs)
    run_has_sources = {run_id: len(sources) > 0 for run_id, sources in inputs.sources_by_run.items()}

    mentioned_rows = [
        r for r in rows if r.mentioned and r.run_id in usable_ids and r.intent_id in UNBRANDED_INTENT_IDS
    ]

    rows_by_engine: dict[str, list[RunCompanyRow]] = defaultdict(list)
    for r in mentioned_rows:
        rows_by_engine[r.engine].append(r)

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
            reason_untestable="zero mentioned pairs in scope (no usable run in intents 1-6 mentioning any in-frame entity)",
        )

    value: dict[str, dict[str, float | None]] = {}
    denominator_before_exclusion: dict[str, dict[str, int]] = {}
    denominator_after_exclusion: dict[str, dict[str, int]] = {}
    n_included: dict[str, dict[str, int]] = {}
    n_excluded: dict[str, dict[str, dict[str, int]]] = {}
    teardown_by_engine: dict[str, dict[str, Any]] = {}

    for engine in sorted(rows_by_engine):
        engine_rows = rows_by_engine[engine]
        sourced_rows = [r for r in engine_rows if run_has_sources.get(r.run_id, False)]
        n_dropped_no_sources = len(engine_rows) - len(sourced_rows)

        unrestricted_share, unr_before, unr_after, unr_unmeasurable = _measurable_share(engine_rows)
        restricted_share, _res_before_raw, res_after, res_unmeasurable = _measurable_share(sourced_rows)

        value[engine] = {"restricted": restricted_share, "unrestricted": unrestricted_share}
        denominator_before_exclusion[engine] = {"restricted": len(sourced_rows), "unrestricted": unr_before}
        denominator_after_exclusion[engine] = {"restricted": res_after, "unrestricted": unr_after}
        n_included[engine] = denominator_after_exclusion[engine]
        n_excluded[engine] = {
            "restricted": {"no_sources": n_dropped_no_sources, "own_domain_unmeasurable": res_unmeasurable},
            "unrestricted": {"own_domain_unmeasurable": unr_unmeasurable},
        }

        qualifying = eligibility.qualifying_by_engine.get(engine, set())
        by_entity: dict[str, list[RunCompanyRow]] = defaultdict(list)
        for r in engine_rows:
            if r.entity_id in qualifying:
                by_entity[r.entity_id].append(r)

        under_threshold: list[str] = []
        unmeasurable_entities: list[str] = []
        # Full per-entity breakdown (2026-09-23, round 2 review) --
        # every qualifying entity, not just the ones under the teardown
        # threshold, so scripts/results/export_s2_teardown.py has real
        # mentions/own_citation_rate to write per company per engine,
        # rather than just a boolean under/over-10% flag.
        qualifying_entity_details: list[dict[str, Any]] = []
        for entity_id, entity_rows in sorted(by_entity.items()):
            measurable = [r for r in entity_rows if r.own_domain_cited is not None]
            if not measurable:
                unmeasurable_entities.append(entity_id)
                continue
            own_share = sum(1 for r in measurable if r.own_domain_cited is True) / len(measurable)
            qualifying_entity_details.append(
                {"entity_id": entity_id, "mentions": len(measurable), "own_citation_rate": own_share}
            )
            if own_share < TEARDOWN_THRESHOLD:
                under_threshold.append(entity_id)

        teardown_by_engine[engine] = {
            "qualifying_entities": len(qualifying),
            "entities_under_10pct_own_citation": len(under_threshold),
            "entity_ids_under_10pct": sorted(under_threshold),
            "entities_unmeasurable_excluded_from_teardown": len(unmeasurable_entities),
            "qualifying_entity_details": qualifying_entity_details,
        }

    notes: dict[str, Any] = {
        "teardown_by_engine": teardown_by_engine,
        "source_definition_by_engine": SOURCE_DEFINITION_BY_ENGINE,
        "not_cited_vs_not_owned_note": NOT_CITED_VS_NOT_OWNED_NOTE,
    }
    if panel_report.entities_with_alternates == 0:
        notes["limitation"] = ALTERNATES_LIMITATION_TEXT

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold={"teardown_own_citation_below": TEARDOWN_THRESHOLD},
        reason_untestable=None,
        notes=notes,
    )
