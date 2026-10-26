"""Phase B, first shared intermediate: source-domain ownership.

One row per response (run) and per candidate entity mentioned in that
response, recording whether any source attached to that response resolves
to a domain the entity owns. The matching rule lives in
analysis.coding.registrable_domain (moved there 2026-09-23 so
analysis/coding/cited_url_resolution.py could reuse it for the Gemini
title fallback -- analysis/coding never imports from analysis/phase_b) and
nowhere else -- both sides (an entity's own domains, and a cited source's
domain) go through the exact same `registrable_domain` reduction, so the
join cannot silently diverge between the two sides or between a future
second caller. Several downstream figures read this table; none of them
may reimplement any part of the matching rule.

Grain is deliberately narrow: only entities actually mentioned (per
mentions_raw), not the full entity x run cross product -- that's a
different, not-yet-built table.

Registers no study figure and computes no share, rate or ratio -- this
module only builds the join and reports on how it was built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analysis.coding.b1_per_run_company import in_frame_entities, surface_form_to_entity_id
from analysis.coding.cited_url_resolution import GEMINI_ENGINE, GEMINI_WRAPPER_MARKER
from analysis.coding.registrable_domain import _normalize_host, psl_info, registrable_domain
from analysis.phase_b.exclusions import usable_run_ids

__all__ = [
    "GEMINI_ENGINE",
    "GEMINI_WRAPPER_MARKER",
    "_normalize_host",
    "psl_info",
    "registrable_domain",
    "BuildReport",
    "OwnershipRow",
    "build_source_ownership",
    "entity_owned_domains",
]

SAMPLE_SIZE = 10


@dataclass(frozen=True)
class OwnershipRow:
    """Enough to join, to check, and to explain. paraphrase_id is a
    first-class column here (not something a later reader has to recover
    by parsing run_id), same reasoning as S3's supporting table."""

    run_id: str
    entity_id: str
    engine: str
    intent_id: int | None
    paraphrase_id: int | None
    owned: bool | None  # None = entity has no known domain, cannot be measured
    matched_domain: str | None  # which owned domain matched, if any
    n_sources: int  # sources attached to this response, regardless of match

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "entity_id": self.entity_id,
            "engine": self.engine,
            "intent_id": self.intent_id,
            "paraphrase_id": self.paraphrase_id,
            "owned": self.owned,
            "matched_domain": self.matched_domain,
            "n_sources": self.n_sources,
        }


@dataclass(frozen=True)
class BuildReport:
    """No headline number lives here -- this is a build report, not a
    figure record. Kept under its own key in the seam file so no display
    surface can mistake it for one."""

    rows_produced: int
    distinct_responses_covered: int
    distinct_entities_covered: int
    owned_true_false_null: dict[str, int]  # sums to rows_produced
    entities_with_no_domain: int
    entities_with_alternates: int
    alt_domains_rows_loaded: int  # raw row count of the alt_domains input itself, before any
    # validation -- reported plainly so a near-empty alternates input is
    # impossible to miss and never mistaken for "checked, found none".
    matches_via_alternate: int
    matches_via_subdomain: int
    responses_with_zero_sources: int
    psl_version: dict[str, str]
    malformed_input_count: int
    malformed_input_samples: list[str]
    # Integrity-support counts: computed here (where the actual rows are),
    # so the author display surface can verify these invariants without
    # ever reading a raw row itself.
    true_rows_without_matched_domain_count: int  # should be 0
    false_or_null_rows_with_matched_domain_count: int  # should be 0
    rows_for_entities_with_no_domain_count: int  # should equal owned_true_false_null["null"]
    total_responses_available: int  # for the "covered <= usable" check
    before_after_resolution: dict[str, int]
    # Same before/after-exclusion split as figure_shape.FigureResult, so a
    # reader comparing this build report to S3 sees the same two-field
    # convention rather than one field doing double duty (see the
    # 2026-09-22 ambiguity ledger entry). before = every usable window
    # response (== total_responses_available); after = the narrower pool
    # this build actually covers -- usable responses that also had at
    # least one candidate (run, in-frame entity) pair to join, i.e.
    # distinct_responses_covered.
    denominator_before_exclusion: int
    denominator_after_exclusion: int
    # The denominator_before_exclusion-vs-after gap, broken down instead
    # of left as an unexplained delta: usable responses that produced
    # zero extractable company mentions at all (a legitimate empty-
    # mention response, not an error -- errored/manual/no-text runs are
    # excluded before this point) plus usable responses whose only
    # extracted mentions all resolved to out-of-frame entities (excluded
    # by the pre-registered frame rule, not a data-quality problem).
    # These two plus denominator_after_exclusion sum exactly to
    # denominator_before_exclusion. Computed directly against
    # mentions_raw and entity_registry, not inferred from the gap.
    usable_runs_with_zero_extracted_mentions: int
    usable_runs_with_mentions_all_out_of_frame: int
    # The S3-vs-this-report zero-source gap (141 total usable zero-source
    # responses across all engines, vs responses_with_zero_sources here,
    # a narrower count restricted to the covered pool): the difference
    # sits entirely inside the excluded set above, split by which
    # exclusion reason it falls under. total_zero_source_usable_responses
    # is the S3-style count (every usable response, any engine, zero
    # sources) computed independently here as a cross-check; it must
    # equal responses_with_zero_sources + the two excluded-zero-source
    # counts below.
    total_zero_source_usable_responses: int
    zero_source_responses_excluded_zero_mentions: int
    zero_source_responses_excluded_out_of_frame_only: int
    coverage_reconcile_note: str
    # Per-engine share of owned=True among measurable rows (true+false;
    # null is always 0 post-fix, since every in-frame entity now has a
    # known domain) -- printed so a gap like Gemini's ~25% against the
    # other three engines' ~51% average is visible directly in the
    # report, not just inferable from raw counts.
    owned_true_share_by_engine: dict[str, float | None]
    # Gemini-specific: every one of Gemini's raw citations is a
    # vertexaisearch redirect wrapper; these fields state how many are
    # still a wrapper after both the Task-5 aux-capture join AND (as of
    # 2026-09-23) the title fallback in
    # analysis.coding.cited_url_resolution.resolve_sources_by_run -- the
    # genuine residual, not the pre-fix 569.
    gemini_citations_total: int
    gemini_citations_resolved: int
    gemini_citations_unresolved: int
    gemini_unresolved_by_status: dict[str, int]
    gemini_unresolved_by_day_index: dict[str, int]
    gemini_unresolved_samples: list[dict[str, Any]]
    gemini_unresolved_note: str
    # How much the genuine residual-unresolved citations could still be
    # moving Gemini's owned=True share -- a structural bound (no title
    # data used) plus a post-fix sanity check (see _gemini_title_recovery's
    # docstring: this should be ~0 now that the real fix runs upstream;
    # non-zero means the upstream fix missed a case). matched_domain/owned
    # above already reflect the real fix -- these fields describe what's
    # left over, not a pending recovery.
    gemini_owned_share_current: float | None
    gemini_owned_share_naive_upper_bound: float | None
    gemini_owned_share_tight_upper_bound: float | None
    gemini_owned_share_bound_note: str
    gemini_title_validation_checked: int
    gemini_title_validation_matches: int
    gemini_title_recovery_residual_recoverable_rows: int
    gemini_title_recovery_residual_owned_share: float | None
    gemini_title_recovery_note: str
    # entities_with_no_domain's 560 -> (current value) change, split into
    # the four buckets it partitions into (2026-09-22, see the ambiguity
    # ledger) -- sums to entities_with_no_domain_before_fix by
    # construction.
    entities_with_no_domain_before_fix: int
    entities_dropped_by_in_frame_fix: int
    entities_gained_domain_via_homepage_fallback: int
    entities_gained_domain_via_alt_domains_only: int
    entities_still_with_no_domain_after_fix: int
    sample_matched_rows: list[dict[str, Any]]
    sample_unmatched_with_sources_rows: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "rows_produced": self.rows_produced,
            "distinct_responses_covered": self.distinct_responses_covered,
            "distinct_entities_covered": self.distinct_entities_covered,
            "owned_true_false_null": self.owned_true_false_null,
            "entities_with_no_domain": self.entities_with_no_domain,
            "entities_with_alternates": self.entities_with_alternates,
            "alt_domains_rows_loaded": self.alt_domains_rows_loaded,
            "matches_via_alternate": self.matches_via_alternate,
            "matches_via_subdomain": self.matches_via_subdomain,
            "responses_with_zero_sources": self.responses_with_zero_sources,
            "psl_version": self.psl_version,
            "malformed_input_count": self.malformed_input_count,
            "malformed_input_samples": self.malformed_input_samples,
            "true_rows_without_matched_domain_count": self.true_rows_without_matched_domain_count,
            "false_or_null_rows_with_matched_domain_count": self.false_or_null_rows_with_matched_domain_count,
            "rows_for_entities_with_no_domain_count": self.rows_for_entities_with_no_domain_count,
            "total_responses_available": self.total_responses_available,
            "before_after_resolution": self.before_after_resolution,
            "denominator_before_exclusion": self.denominator_before_exclusion,
            "denominator_after_exclusion": self.denominator_after_exclusion,
            "usable_runs_with_zero_extracted_mentions": self.usable_runs_with_zero_extracted_mentions,
            "usable_runs_with_mentions_all_out_of_frame": self.usable_runs_with_mentions_all_out_of_frame,
            "total_zero_source_usable_responses": self.total_zero_source_usable_responses,
            "zero_source_responses_excluded_zero_mentions": self.zero_source_responses_excluded_zero_mentions,
            "zero_source_responses_excluded_out_of_frame_only": self.zero_source_responses_excluded_out_of_frame_only,
            "coverage_reconcile_note": self.coverage_reconcile_note,
            "owned_true_share_by_engine": self.owned_true_share_by_engine,
            "gemini_citations_total": self.gemini_citations_total,
            "gemini_citations_resolved": self.gemini_citations_resolved,
            "gemini_citations_unresolved": self.gemini_citations_unresolved,
            "gemini_unresolved_by_status": self.gemini_unresolved_by_status,
            "gemini_unresolved_by_day_index": self.gemini_unresolved_by_day_index,
            "gemini_unresolved_samples": self.gemini_unresolved_samples,
            "gemini_unresolved_note": self.gemini_unresolved_note,
            "gemini_owned_share_current": self.gemini_owned_share_current,
            "gemini_owned_share_naive_upper_bound": self.gemini_owned_share_naive_upper_bound,
            "gemini_owned_share_tight_upper_bound": self.gemini_owned_share_tight_upper_bound,
            "gemini_owned_share_bound_note": self.gemini_owned_share_bound_note,
            "gemini_title_validation_checked": self.gemini_title_validation_checked,
            "gemini_title_validation_matches": self.gemini_title_validation_matches,
            "gemini_title_recovery_residual_recoverable_rows": self.gemini_title_recovery_residual_recoverable_rows,
            "gemini_title_recovery_residual_owned_share": self.gemini_title_recovery_residual_owned_share,
            "gemini_title_recovery_note": self.gemini_title_recovery_note,
            "entities_with_no_domain_before_fix": self.entities_with_no_domain_before_fix,
            "entities_dropped_by_in_frame_fix": self.entities_dropped_by_in_frame_fix,
            "entities_gained_domain_via_homepage_fallback": self.entities_gained_domain_via_homepage_fallback,
            "entities_gained_domain_via_alt_domains_only": self.entities_gained_domain_via_alt_domains_only,
            "entities_still_with_no_domain_after_fix": self.entities_still_with_no_domain_after_fix,
            "sample_matched_rows": self.sample_matched_rows,
            "sample_unmatched_with_sources_rows": self.sample_unmatched_with_sources_rows,
        }


GEMINI_UNRESOLVED_SAMPLE_SIZE = 15


def _gemini_citation_resolution_stats(
    sources_by_run: dict[str, list[dict[str, Any]]], raw_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reports, does not fix (2026-09-22 -- see the ambiguity ledger).
    sources_by_run here is post-swap (analysis.phase_b.reading already
    applied resolve_sources_by_run): a Gemini citation is "resolved" if
    its url_raw no longer contains the vertexaisearch wrapper marker,
    "unresolved" if it still does -- either the Task-5 capture never saw
    this exact URL at all (resolution_status is None) or it did and the
    target itself is genuinely unreachable (resolution_status is a dead
    bucket: paywalled/http_403/http_404/server_error/timeout/etc, in
    which case resolve_sources_by_run leaves url_raw untouched by
    design). Only google_gemini is checked -- it is the only engine
    whose raw citations are redirect wrappers in the first place."""
    run_by_id = {r["run_id"]: r for r in raw_runs}
    total = 0
    resolved = 0
    unresolved = 0
    by_status: dict[str, int] = {}
    by_day_index: dict[str, int] = {}
    samples: list[dict[str, Any]] = []

    for run_id, sources in sources_by_run.items():
        run = run_by_id.get(run_id)
        if run is None or run.get("engine") != GEMINI_ENGINE:
            continue
        for source in sources:
            total += 1
            url_raw = source.get("url_raw") or ""
            if GEMINI_WRAPPER_MARKER in url_raw:
                unresolved += 1
                status = source.get("resolution_status") or "never_matched_in_aux_capture"
                by_status[status] = by_status.get(status, 0) + 1
                day_key = str(run.get("day_index"))
                by_day_index[day_key] = by_day_index.get(day_key, 0) + 1
                if len(samples) < GEMINI_UNRESOLVED_SAMPLE_SIZE:
                    samples.append(
                        {
                            "run_id": run_id,
                            "day_index": run.get("day_index"),
                            "resolution_status": status,
                            "url_cited": source.get("url_cited") or url_raw,
                        }
                    )
            else:
                resolved += 1

    never_matched = by_status.get("never_matched_in_aux_capture", 0)
    if unresolved == 0:
        note = "0 unresolved Gemini citations -- the Task-5 aux capture fully covers this engine's sources."
    elif never_matched == 0:
        note = (
            f"All {unresolved} unresolved Gemini citations were present in the Sep 6/7 aux capture "
            f"(data/02_aux/cited_urls/resolution.jsonl) -- the capture attempted every one, spread across "
            f"all 7 window days ({by_day_index}); the target itself is genuinely unreachable now "
            f"(status breakdown: {by_status}), not a gap in what the capture covers."
        )
    else:
        note = (
            f"{never_matched} of {unresolved} unresolved Gemini citations were never captured by the "
            f"Sep 6/7 aux capture at all (no row in resolution.jsonl for that exact URL) -- a real "
            f"coverage gap, not a dead-target result. By day_index: {by_day_index}. Not fixed yet, "
            f"per instruction -- report only."
        )

    return {
        "gemini_citations_total": total,
        "gemini_citations_resolved": resolved,
        "gemini_citations_unresolved": unresolved,
        "gemini_unresolved_by_status": by_status,
        "gemini_unresolved_by_day_index": by_day_index,
        "gemini_unresolved_samples": samples,
        "gemini_unresolved_note": note,
    }


def _gemini_owned_share_bound(
    rows: list["OwnershipRow"], sources_by_run: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Reports, does not fix (2026-09-22 -- see the ambiguity ledger). The
    569 unresolved Gemini citations are counted as non-matches today
    (never own-domain, never someone else's -- registrable_domain of a
    vertexaisearch wrapper URL can't equal any entity's owned domain), so
    every Gemini owned=False row with an unresolved citation attached is
    a row that *could* have been True in the most favorable case. Bounds
    the resulting uncertainty on Gemini's owned=True share two ways:
    naive (every such row counted as flippable independently -- loose,
    since two different entities in the same response can't both really
    own the same one mystery URL) and tight (capped per run by
    min(unresolved citations in that run, flippable rows in that run) --
    the correct upper bound if the run's unresolved citations were
    distributed one-to-one across its flippable entities in the best
    possible way)."""
    gemini_rows = [r for r in rows if r.engine == GEMINI_ENGINE]
    true_count = sum(1 for r in gemini_rows if r.owned is True)
    false_rows = [r for r in gemini_rows if r.owned is False]
    measurable = true_count + len(false_rows)
    current_share = (true_count / measurable) if measurable else None

    false_rows_by_run: dict[str, list["OwnershipRow"]] = {}
    for r in false_rows:
        false_rows_by_run.setdefault(r.run_id, []).append(r)

    naive_flippable = 0
    tight_flippable = 0
    for run_id, run_false_rows in false_rows_by_run.items():
        sources = sources_by_run.get(run_id, [])
        n_unresolved = sum(1 for s in sources if GEMINI_WRAPPER_MARKER in (s.get("url_raw") or ""))
        if n_unresolved == 0:
            continue
        flippable_in_run = sum(
            1 for r in run_false_rows if any(GEMINI_WRAPPER_MARKER in (s.get("url_raw") or "") for s in sources)
        )
        naive_flippable += flippable_in_run
        tight_flippable += min(n_unresolved, flippable_in_run)

    naive_upper_share = ((true_count + naive_flippable) / measurable) if measurable else None
    tight_upper_share = ((true_count + tight_flippable) / measurable) if measurable else None

    return {
        "gemini_owned_share_current": current_share,
        "gemini_owned_share_naive_upper_bound": naive_upper_share,
        "gemini_owned_share_tight_upper_bound": tight_upper_share,
        "gemini_owned_share_bound_note": (
            f"569 unresolved Gemini citations are counted as non-matches today. Current share "
            f"{current_share:.1%} vs a tight upper bound of {tight_upper_share:.1%} (capped per run so "
            f"one mystery citation can flip at most one entity's row, not every candidate simultaneously) "
            f"-- not tiny, so not just logged; see gemini_title_recovery for the actual figure rather than "
            f"a bound (naive, uncapped bound is {naive_upper_share:.1%}, included for reference only -- it "
            f"overcounts whenever a run has more flippable rows than unresolved citations to explain them)."
            if measurable
            else "no measurable Gemini rows"
        ),
    }


def _gemini_title_recovery(
    rows: list["OwnershipRow"],
    sources_by_run: dict[str, list[dict[str, Any]]],
    raw_runs: list[dict[str, Any]],
    owned_domains_by_entity: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """As of 2026-09-23 the title fallback this function used to only
    simulate is applied for real, upstream, in
    analysis.coding.cited_url_resolution.resolve_sources_by_run -- so
    sources_by_run here already reflects it (a recovered Gemini source's
    url_raw is rewritten to a synthetic https://<title-domain>/, no longer
    carrying GEMINI_WRAPPER_MARKER, tagged resolution_method="title_fallback").
    This function now does two things, both **checks against that real
    fix**, not a second implementation of it:

    1. Validates the title field still reduces to the true domain on every
       citation that was resolved *for real* (a genuine fetched
       destination, not itself a title-fallback row -- checking a
       title-fallback row's title against its own title-derived url_raw
       would be circular and prove nothing).
    2. Re-runs the same "would this recover a row" scan the pre-fix
       simulation used, restricted now to whatever citations still carry
       the wrapper marker (the genuine residual -- titles that didn't
       reduce to a usable domain at all). On correct production data this
       should report ~0 additional recoverable rows, since anything it
       could recover the real fix already would have. A non-zero result
       here is a signal the upstream fix missed a case, not a pending
       decision -- see the ambiguity ledger.

    Restricted to google_gemini: other engines' `title` is a real page
    title (free text, not domain-shaped) -- checked empirically per engine
    via analysis.coding.cited_url_resolution.title_domain_match_rate_by_engine,
    not assumed; see that function's docstring."""
    run_by_id = {r["run_id"]: r for r in raw_runs}
    validated = 0
    validated_total = 0
    unresolved_total = 0
    unresolved_with_usable_title = 0
    for run_id, sources in sources_by_run.items():
        if run_by_id.get(run_id, {}).get("engine") != GEMINI_ENGINE:
            continue
        for s in sources:
            url_raw = s.get("url_raw") or ""
            if GEMINI_WRAPPER_MARKER in url_raw:
                unresolved_total += 1
                if registrable_domain(s.get("title")) is not None:
                    unresolved_with_usable_title += 1
                continue  # unresolved -- no ground truth to check title against
            if s.get("resolution_method") == "title_fallback":
                continue  # url_raw was derived from title itself -- not independent ground truth
            title = s.get("title")
            if not title:
                continue
            actual_domain = registrable_domain(url_raw)
            if actual_domain is None:
                continue
            validated_total += 1
            if registrable_domain(title) == actual_domain:
                validated += 1

    residual_recoverable_rows = 0
    for row in rows:
        if row.engine != GEMINI_ENGINE or row.owned is not False:
            continue
        owned_domains = owned_domains_by_entity.get(row.entity_id) or {}
        if not owned_domains:
            continue
        for source in sources_by_run.get(row.run_id, []):
            url_raw = source.get("url_raw") or ""
            if GEMINI_WRAPPER_MARKER not in url_raw:
                continue
            title_domain = registrable_domain(source.get("title"))
            if title_domain in owned_domains:
                residual_recoverable_rows += 1
                break

    gemini_rows = [r for r in rows if r.engine == GEMINI_ENGINE]
    true_count = sum(1 for r in gemini_rows if r.owned is True)
    measurable = len(gemini_rows)
    residual_share = ((true_count + residual_recoverable_rows) / measurable) if measurable else None

    if validated_total == 0 and unresolved_total == 0:
        note = "no Gemini citations present"
    else:
        note = (
            f"title field validated against {validated_total} for-real-resolved Gemini citations "
            f"(title-fallback rows excluded -- checking a title-derived url_raw against its own title "
            f"would be circular): {validated}/{validated_total} exact match on registrable domain"
            + (" (0 mismatches)" if validated == validated_total else f" ({validated_total - validated} mismatches)")
            + f". Of {unresolved_total} citations still unresolved after the real upstream fix "
            f"(analysis.coding.cited_url_resolution.resolve_sources_by_run, applied 2026-09-23), "
            f"{unresolved_with_usable_title} carry a title that reduces to a usable domain -- expected "
            f"near 0 since the real fix already recovers anything recoverable this way; a non-zero count "
            f"here means the upstream fix missed a case, not a pending decision. This residual would flip "
            f"{residual_recoverable_rows} more Gemini rows from False to True"
            + (f", moving owned=True share to {residual_share:.1%}" if residual_share is not None else "")
            + "."
        )

    return {
        "gemini_title_validation_checked": validated_total,
        "gemini_title_validation_matches": validated,
        "gemini_title_recovery_residual_recoverable_rows": residual_recoverable_rows,
        "gemini_title_recovery_residual_owned_share": residual_share,
        "gemini_title_recovery_note": note,
    }


def _entities_with_no_domain_breakdown(
    mentions_raw: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    entities_by_id: dict[str, dict[str, Any]],
    in_frame_ids: set[str],
    verified_homepage_by_entity: dict[str, str],
    alt_domains_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Diagnostic only (2026-09-22 -- see the ambiguity ledger): explains
    entities_with_no_domain's 560 -> current-value change by reproducing
    the pre-fix computation (every resolved entity, in-frame or not, no
    verified_homepage_url fallback, alt_domains as it stood before this
    fix) and classifying each of those 560 into exactly one bucket. The
    four bucket counts sum to the pre-fix total by construction -- not
    asserted separately, since they partition it directly."""
    sf_to_entity_all = surface_form_to_entity_id(entities)
    all_candidate_ids = {
        sf_to_entity_all[m["surface_form"]]
        for m in mentions_raw
        if m.get("surface_form") in sf_to_entity_all
    }
    malformed_scratch: list[str] = []
    pre_fix_no_domain: set[str] = set()
    for entity_id in all_candidate_ids:
        entity = entities_by_id.get(entity_id)
        if entity is None:
            continue
        owned = entity_owned_domains(entity, [], malformed_scratch, None)
        if not owned:
            pre_fix_no_domain.add(entity_id)

    alt_domain_entity_ids = {row["entity_id"] for row in alt_domains_rows}
    dropped_out_of_frame = 0
    gained_via_homepage_fallback = 0
    gained_via_alt_domains_only = 0
    still_no_domain = 0
    for entity_id in pre_fix_no_domain:
        if entity_id not in in_frame_ids:
            dropped_out_of_frame += 1
        elif verified_homepage_by_entity.get(entity_id):
            gained_via_homepage_fallback += 1
        elif entity_id in alt_domain_entity_ids:
            gained_via_alt_domains_only += 1
        else:
            still_no_domain += 1

    return {
        "entities_with_no_domain_before_fix": len(pre_fix_no_domain),
        "entities_dropped_by_in_frame_fix": dropped_out_of_frame,
        "entities_gained_domain_via_homepage_fallback": gained_via_homepage_fallback,
        "entities_gained_domain_via_alt_domains_only": gained_via_alt_domains_only,
        "entities_still_with_no_domain_after_fix": still_no_domain,
    }


def entity_owned_domains(
    entity: dict[str, Any],
    alt_domains_rows: list[dict[str, Any]],
    malformed: list[str],
    verified_homepage_by_entity: dict[str, str] | None = None,
) -> dict[str, str]:
    """registrable domain -> where it came from ("homepage",
    "verified_homepage", or "alt_domain:<raw value>"), for one entity.
    Empty dict = no known domain (yields None downstream, never False).
    Alternate owned domains are equal citizens to the primary one -- same
    dict, same lookup.

    entity_registry.csv's homepage_url is blank for 141/415 in-frame
    entities (only entity_verification.verified_homepage_url, the
    A4-prime search-backed check, has it -- see
    analysis/aux/roster.py for the same fallback already used to build
    the aux-collection roster). Falling back here closes the largest
    single hole in this join: without it, those 141 entities have no
    owned domain at all and every one of their mentions reads `owned=None`
    (unmeasurable), regardless of alt_domains."""
    owned: dict[str, str] = {}
    homepage = entity.get("homepage_url")
    source = "homepage"
    if not homepage and verified_homepage_by_entity:
        homepage = verified_homepage_by_entity.get(entity["entity_id"])
        source = "verified_homepage"
    if homepage:
        domain = registrable_domain(homepage)
        if domain:
            owned[domain] = source
        else:
            malformed.append(f"entity {entity['entity_id']} homepage_url={homepage!r}")
    for row in alt_domains_rows:
        if row.get("entity_id") != entity["entity_id"]:
            continue
        alt = row.get("alt_domain")
        if not alt:
            continue
        domain = registrable_domain(alt)
        if domain:
            owned.setdefault(domain, f"alt_domain:{alt}")
        else:
            malformed.append(f"entity {entity['entity_id']} alt_domain={alt!r}")
    return owned


def build_source_ownership(
    mentions_raw: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    alt_domains_rows: list[dict[str, Any]],
    sources_by_run: dict[str, list[dict[str, Any]]],
    raw_runs: list[dict[str, Any]],
    entity_verification_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[OwnershipRow], BuildReport]:
    """The shared intermediate, built exactly once. sources_by_run and
    raw_runs are exactly what the reading layer handed over -- this
    function reads nothing it wasn't given, and computes no share, rate
    or ratio; that is a figure's job, not this one's.

    Restricted to meets_frame == 'in' entities (analysis.coding.
    b1_per_run_company.in_frame_entities), mirroring B1's own precedent
    and the pre-registered rule "do not let entities failing the frame
    enter any denominator" (Notion Part 3.0 / EXPERIMENT.md). A mention
    resolving to a real but out-of-frame entity_id is silently excluded
    here (expected, not malformed); a surface_form that resolves to no
    entity_id at all, in the full registry, is still recorded as
    malformed -- the two are checked against separate mappings so
    out-of-frame volume never gets miscounted as a data-quality problem.
    """
    sf_to_entity_all = surface_form_to_entity_id(entities)
    in_frame_ids = {e["entity_id"] for e in in_frame_entities(entities)}
    entities_by_id = {e["entity_id"]: e for e in entities}
    run_by_id = {r["run_id"]: r for r in raw_runs}
    verified_homepage_by_entity = {
        r["entity_id"]: r["verified_homepage_url"]
        for r in (entity_verification_rows or [])
        if r.get("verified_homepage_url")
    }

    malformed: list[str] = []
    for row in alt_domains_rows:
        if row.get("entity_id") not in entities_by_id:
            malformed.append(
                f"alt_domains row entity_id={row.get('entity_id')!r} not found in entity_registry "
                f"(alt_domain={row.get('alt_domain')!r}, source_note={row.get('source_note')!r})"
            )

    candidate_pairs: set[tuple[str, str]] = set()
    for m in mentions_raw:
        surface_form = m.get("surface_form")
        entity_id = sf_to_entity_all.get(surface_form) if surface_form else None
        if entity_id is None:
            malformed.append(f"run {m.get('run_id')} surface_form={surface_form!r}")
            continue
        if entity_id not in in_frame_ids:
            continue  # resolved, but out-of-frame -- excluded, never a denominator, not malformed
        candidate_pairs.add((m["run_id"], entity_id))

    candidate_entity_ids = sorted({entity_id for _, entity_id in candidate_pairs})
    owned_domains_by_entity: dict[str, dict[str, str]] = {}
    entities_with_alternates = 0
    for entity_id in candidate_entity_ids:
        entity = entities_by_id.get(entity_id)
        if entity is None:  # candidate resolved to an entity_id not in the registry -- malformed
            malformed.append(f"entity_id {entity_id!r} not found in entity_registry")
            owned_domains_by_entity[entity_id] = {}
            continue
        owned = entity_owned_domains(entity, alt_domains_rows, malformed, verified_homepage_by_entity)
        owned_domains_by_entity[entity_id] = owned
        if any(source.startswith("alt_domain:") for source in owned.values()):
            entities_with_alternates += 1

    entities_with_no_domain = sum(1 for d in owned_domains_by_entity.values() if not d)

    rows: list[OwnershipRow] = []
    matches_via_alternate = 0
    matches_via_subdomain = 0
    for run_id, entity_id in sorted(candidate_pairs):
        run = run_by_id.get(run_id, {})
        sources = sources_by_run.get(run_id, [])
        n_sources = len(sources)
        owned_domains = owned_domains_by_entity[entity_id]

        matched_domain: str | None = None
        if owned_domains:
            for source in sources:
                url_raw = source.get("url_raw")
                source_domain = registrable_domain(url_raw)
                if source_domain is None and url_raw:
                    malformed.append(f"run {run_id} source url_raw={url_raw!r}")
                    continue
                if source_domain in owned_domains:
                    matched_domain = source_domain
                    provenance = owned_domains[source_domain]
                    if provenance.startswith("alt_domain:"):
                        matches_via_alternate += 1
                    if _normalize_host(url_raw) != source_domain:
                        matches_via_subdomain += 1
                    break

        owned = None if not owned_domains else matched_domain is not None
        rows.append(
            OwnershipRow(
                run_id=run_id,
                entity_id=entity_id,
                engine=run.get("engine"),
                intent_id=run.get("intent_id"),
                paraphrase_id=run.get("paraphrase_id"),
                owned=owned,
                matched_domain=matched_domain,
                n_sources=n_sources,
            )
        )

    distinct_responses = {run_id for run_id, _ in candidate_pairs}
    responses_with_zero_sources = sum(1 for run_id in distinct_responses if not sources_by_run.get(run_id))

    triple = {"true": 0, "false": 0, "null": 0}
    true_without_domain = 0
    false_or_null_with_domain = 0
    rows_for_no_domain_entities = 0
    true_false_by_engine: dict[str, dict[str, int]] = {}
    sample_matched: list[dict[str, Any]] = []
    sample_unmatched_with_sources: list[dict[str, Any]] = []
    for row in rows:
        engine_counts = true_false_by_engine.setdefault(row.engine, {"true": 0, "false": 0})
        if row.owned is True:
            triple["true"] += 1
            engine_counts["true"] += 1
            if row.matched_domain is None:
                true_without_domain += 1
            elif len(sample_matched) < SAMPLE_SIZE:
                sample_matched.append(row.to_json())
        elif row.owned is False:
            triple["false"] += 1
            engine_counts["false"] += 1
            if row.matched_domain is not None:
                false_or_null_with_domain += 1
            if row.n_sources > 0 and len(sample_unmatched_with_sources) < SAMPLE_SIZE:
                sample_unmatched_with_sources.append(row.to_json())
        else:
            triple["null"] += 1
            if row.matched_domain is not None:
                false_or_null_with_domain += 1
        if not owned_domains_by_entity.get(row.entity_id):
            rows_for_no_domain_entities += 1

    owned_true_share_by_engine: dict[str, float | None] = {}
    for engine, counts in true_false_by_engine.items():
        measurable = counts["true"] + counts["false"]
        owned_true_share_by_engine[engine] = (counts["true"] / measurable) if measurable else None

    usable_ids = usable_run_ids(raw_runs)
    mention_run_ids_any = {m["run_id"] for m in mentions_raw}
    zero_mention_runs = usable_ids - mention_run_ids_any
    out_of_frame_only_runs = (usable_ids & mention_run_ids_any) - distinct_responses

    # S3-style zero-source count (every usable response, any engine),
    # computed independently here as a cross-check against S3's own
    # figure -- and split by exactly which exclusion bucket a zero-source
    # response not covered here falls under, since S3's denominator is
    # all usable responses while this report's is the narrower covered
    # pool (2026-09-22, see the ambiguity ledger).
    zero_source_usable_ids = {run_id for run_id in usable_ids if not sources_by_run.get(run_id)}
    zero_source_excluded_zero_mentions = len(zero_source_usable_ids & zero_mention_runs)
    zero_source_excluded_out_of_frame_only = len(zero_source_usable_ids & out_of_frame_only_runs)

    coverage_reconcile_note = (
        f"{len(usable_ids)} usable responses split into {len(distinct_responses)} covered here "
        f"(>=1 in-frame resolvable mention) + {len(zero_mention_runs)} with zero extracted mentions "
        f"(a legitimate empty-mention response, not an error) + {len(out_of_frame_only_runs)} whose "
        f"only extracted mentions were all out-of-frame entities (excluded by the pre-registered frame "
        f"rule, not a data-quality problem). Zero-source reconcile: S3 reports "
        f"{len(zero_source_usable_ids)} zero-source usable responses across all engines; this report's "
        f"responses_with_zero_sources ({responses_with_zero_sources}) is narrower (covered pool only) -- "
        f"the {len(zero_source_usable_ids) - responses_with_zero_sources}-response gap sits entirely "
        f"inside the excluded set above: {zero_source_excluded_zero_mentions} of the zero-extracted-"
        f"mentions exclusions and {zero_source_excluded_out_of_frame_only} of the all-out-of-frame "
        f"exclusions were themselves zero-source responses."
    )

    gemini_stats = _gemini_citation_resolution_stats(sources_by_run, raw_runs)
    gemini_bound = _gemini_owned_share_bound(rows, sources_by_run)
    gemini_title_recovery = _gemini_title_recovery(rows, sources_by_run, raw_runs, owned_domains_by_entity)
    no_domain_breakdown = _entities_with_no_domain_breakdown(
        mentions_raw, entities, entities_by_id, in_frame_ids, verified_homepage_by_entity, alt_domains_rows
    )

    report = BuildReport(
        rows_produced=len(rows),
        distinct_responses_covered=len(distinct_responses),
        distinct_entities_covered=len(candidate_entity_ids),
        owned_true_false_null=triple,
        entities_with_no_domain=entities_with_no_domain,
        entities_with_alternates=entities_with_alternates,
        alt_domains_rows_loaded=len(alt_domains_rows),
        matches_via_alternate=matches_via_alternate,
        matches_via_subdomain=matches_via_subdomain,
        responses_with_zero_sources=responses_with_zero_sources,
        psl_version=psl_info(),
        malformed_input_count=len(malformed),
        malformed_input_samples=malformed[:SAMPLE_SIZE],
        true_rows_without_matched_domain_count=true_without_domain,
        false_or_null_rows_with_matched_domain_count=false_or_null_with_domain,
        rows_for_entities_with_no_domain_count=rows_for_no_domain_entities,
        total_responses_available=len(usable_ids),
        before_after_resolution={"before": len(mentions_raw), "after": len(rows)},
        denominator_before_exclusion=len(usable_ids),
        denominator_after_exclusion=len(distinct_responses),
        usable_runs_with_zero_extracted_mentions=len(zero_mention_runs),
        usable_runs_with_mentions_all_out_of_frame=len(out_of_frame_only_runs),
        total_zero_source_usable_responses=len(zero_source_usable_ids),
        zero_source_responses_excluded_zero_mentions=zero_source_excluded_zero_mentions,
        zero_source_responses_excluded_out_of_frame_only=zero_source_excluded_out_of_frame_only,
        coverage_reconcile_note=coverage_reconcile_note,
        owned_true_share_by_engine=owned_true_share_by_engine,
        gemini_citations_total=gemini_stats["gemini_citations_total"],
        gemini_citations_resolved=gemini_stats["gemini_citations_resolved"],
        gemini_citations_unresolved=gemini_stats["gemini_citations_unresolved"],
        gemini_unresolved_by_status=gemini_stats["gemini_unresolved_by_status"],
        gemini_unresolved_by_day_index=gemini_stats["gemini_unresolved_by_day_index"],
        gemini_unresolved_samples=gemini_stats["gemini_unresolved_samples"],
        gemini_unresolved_note=gemini_stats["gemini_unresolved_note"],
        gemini_owned_share_current=gemini_bound["gemini_owned_share_current"],
        gemini_owned_share_naive_upper_bound=gemini_bound["gemini_owned_share_naive_upper_bound"],
        gemini_owned_share_tight_upper_bound=gemini_bound["gemini_owned_share_tight_upper_bound"],
        gemini_owned_share_bound_note=gemini_bound["gemini_owned_share_bound_note"],
        gemini_title_validation_checked=gemini_title_recovery["gemini_title_validation_checked"],
        gemini_title_validation_matches=gemini_title_recovery["gemini_title_validation_matches"],
        gemini_title_recovery_residual_recoverable_rows=gemini_title_recovery[
            "gemini_title_recovery_residual_recoverable_rows"
        ],
        gemini_title_recovery_residual_owned_share=gemini_title_recovery["gemini_title_recovery_residual_owned_share"],
        gemini_title_recovery_note=gemini_title_recovery["gemini_title_recovery_note"],
        entities_with_no_domain_before_fix=no_domain_breakdown["entities_with_no_domain_before_fix"],
        entities_dropped_by_in_frame_fix=no_domain_breakdown["entities_dropped_by_in_frame_fix"],
        entities_gained_domain_via_homepage_fallback=no_domain_breakdown["entities_gained_domain_via_homepage_fallback"],
        entities_gained_domain_via_alt_domains_only=no_domain_breakdown["entities_gained_domain_via_alt_domains_only"],
        entities_still_with_no_domain_after_fix=no_domain_breakdown["entities_still_with_no_domain_after_fix"],
        sample_matched_rows=sample_matched,
        sample_unmatched_with_sources_rows=sample_unmatched_with_sources,
    )
    return rows, report
