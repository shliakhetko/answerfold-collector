"""K2 -- Citation domain concentration (PRE-SPECIFIED TABLE, pinned
EXPERIMENT.md / Notion Part 3.2).

Unit: the citation instance. Denominator: all citation instances in the
six unbranded intents (1-6), per engine and pooled -- read from
analysis.phase_b.citation_panel, the shared url_class join (K2 also uses
its url_class column for the top-10 table, though the core concentration
figures need no url_class filter at all -- unlike S1, K2's denominator is
every citation with a resolvable domain, classified or not).

Domain grouping uses registrable_domain(url_normalized) directly --
citation_panel.domain is not used here, since its granularity varies by
resolution path (aux-capture final_registrable_domain vs. a raw,
un-reduced host for a never-matched citation) and K2's whole point is a
domain-level concentration ranking that must use one consistent
reduction, the same one the rest of Phase B treats as canonical.

**Headline exclusion rule: decided 2026-09-25** (see
k2-headline-raw-self-citation-2026-09-25 in the ambiguity ledger, which
supersedes the still-open framing this docstring carried through
2026-09-24). Headline = `included` (raw, no own-domain exclusion) --
matches the planned E3 definition. Second view = `excluded_self_citation`,
labeled "pages you can act on" in the render script, since it keeps
competitor vendor listicles (pages a client can actually be added to).
`excluded_population` stays computed here (all three treatments are always
computed, this module makes no display decisions) but is not shown on the
rendered figure at all -- see scripts/results/render_k2_image.py.

**Two own-domain exclusion rules, both computed, neither the default**
(2026-09-23, round 2 review -- both are shown, neither picked; see the
`ledger-status-review-2026-09-24` entry for why this is attributed to
round 2 review rather than a confirmed owner decision):

- **excluded_population**: a citation's domain is dropped whenever it
  belongs to ANY in-frame entity's owned-domain set
  (analysis.phase_b.citation_panel.all_in_frame_owned_domains),
  regardless of which company that specific answer was actually about.
  This is the rule K2 had before this fix, unchanged in behavior, renamed
  for clarity now that a second rule exists.
- **excluded_self_citation**: a citation's domain is dropped only when it
  belongs to an entity actually mentioned in that same run
  (analysis.phase_b.citation_panel.mentioned_own_domains_by_run, the same
  per-run join S1 Fix 6 already uses). A competitor's vendor listicle is
  never dropped under this rule just because the listicle's publisher is
  also a population company -- only a genuine self-citation is.

Own-domain rules only ever *remove* domains from `included`; they never
add or reclassify one.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.coding.registrable_domain import registrable_domain
from analysis.phase_b.citation_panel import (
    CitationRow,
    all_in_frame_owned_domains,
    build_citation_panel,
    mentioned_own_domains_by_run,
)
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "K2"
TOP_N = 10
POOLED = "pooled"
# Notion Part 3.2's own decision-threshold sentence, verbatim (EXPERIMENT.md
# line 315): "At or above 60% the ranked top-10 list becomes a client
# deliverable rather than a finding. Below 40% the ranked-page-list
# deliverable comes out of the remediation package." No verdict bands or
# pass/fail labels are computed from this any more (2026-09-24 review
# feedback: K2 is a pre-specified table, not a claim with a decision rule
# -- see the 2026-09-24 k2-verdict-threshold-removed ledger entry). The
# band itself is kept only as the quoted pre-registration text in
# PRE_REG_BANDS_NOTE_PREFIX below, never derived from observed data.
#
# 2026-09-24 review feedback (second pass): the first version of this note
# stated the 40-60% band as if it were a forecast of what the data would
# show -- wrong. The owner never had a forecast for K2; 40%/60% were a
# decision RULE (when does a ranked list become a client deliverable),
# not a prediction. Framing the band that way after seeing the data
# invents a prediction that was never made -- exactly the kind of
# verdict-in-disguise the whole exercise of dropping the bands was meant
# to remove. The note below states three things only: what the pre-reg
# set (quoted), that it was a decision rule dropped after seeing the data
# (with the ledger entry id), and the observed numbers -- nothing that
# reads as a forecast (see k2-footer-invented-prediction-removed-2026-09-24).
DECISION_THRESHOLD_QUOTE = (
    "EXPERIMENT.md line 315, verbatim: \"At or above 60% the ranked top-10 list becomes a client deliverable "
    "rather than a finding. Below 40% the ranked-page-list deliverable comes out of the remediation package.\""
)
PRE_REG_BANDS_NOTE_PREFIX = (
    f"Pre-registered decision bands ({DECISION_THRESHOLD_QUOTE}). These were a decision rule, not a "
    "prediction, and were dropped after seeing the data (see k2-verdict-threshold-removed-2026-09-24 in the "
    "ambiguity ledger). Observed per engine, all three treatments (headline = raw/included, second view = "
    "excluded_self_citation \"pages you can act on\", per k2-headline-raw-self-citation-2026-09-25 -- see "
    "SCOPE_READING_NOTE): "
)
CAPTERRA_DOMAIN = "capterra.com"
G2_DOMAIN = "g2.com"
TREATMENTS = ("included", "excluded_self_citation", "excluded_population")
TREATMENT_RULE_NOTE = (
    "included: every citation, own domain or not. excluded_self_citation: dropped only when the citation's "
    "domain belongs to an entity actually mentioned in that same run (a genuine self-citation). "
    "excluded_population: dropped whenever the domain belongs to ANY in-frame population entity, regardless "
    "of which company that answer was actually about -- a competitor's vendor listicle is excluded under "
    "this rule too, since the listicle's publisher is itself a population company (2026-09-23, round 2 "
    "review)."
)
# 2026-09-24 review feedback: renamed from VERDICT_NOTE now that K2 carries
# no verdict language anywhere -- the underlying fact (no pre-registered
# combination rule across engines, pooled is context not the answer)
# stands regardless.
SCOPE_READING_NOTE = (
    "K2 is a pre-specified TABLE, not a confirmatory claim (Notion Part 3.0: tables \"get a specification, "
    "never a prediction... nothing here can be wrong, only uninteresting\"). No combination rule for turning "
    "4 per-engine numbers into one blended answer is pre-registered anywhere -- unlike S1, which has an "
    "explicit \"3 of 4 engines\" rule. Read each engine on its own; the pooled row is context, weighted "
    "toward whichever engine has the most citations (see denominator_before_exclusion per scope), never the "
    "answer on its own (2026-09-23, round 2 review)."
)
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
# 2026-09-23, round 2 review: retellai.com/dapta.ai are never excluded under
# EITHER own-domain rule, because both all_in_frame_owned_domains() and
# mentioned_own_domains_by_run() only draw from meets_frame=="in" entities,
# and both are meets_frame="out"/"borderline" -- real, A4-prime-verified
# vendors that Phase A's frame check nonetheless excluded from in-frame
# (see round2-fix-b-k2-self-citation-rule-2026-09-23 in the ambiguity
# ledger for the exact numbers: retellai.com 959/959/959, dapta.ai
# 420/420/420, unchanged across all three treatments). Their own sites
# count as third-party pages under every rule this figure computes.
RETELL_DAPTA_FOOTNOTE = (
    "retellai.com (entity retell-ai, meets_frame=out) and dapta.ai (entity dapta, meets_frame=borderline) are "
    "never excluded under either own-domain rule -- both are out-of-frame, so their own sites count as "
    "third-party pages under every treatment here, same as any other unaffiliated domain."
)


def _concentration(domains: list[str]) -> dict[str, Any]:
    """domains: one entry per citation instance (already own-domain-
    filtered by the caller), each a registrable domain."""
    total = len(domains)
    counts: dict[str, int] = {}
    for d in domains:
        counts[d] = counts.get(d, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    top_n_count = sum(count for _, count in ranked[:TOP_N])
    top_n_share = (top_n_count / total) if total else None

    count_to_50pct = None
    cumulative_share_curve: list[float] = []
    if total:
        cumulative = 0
        for i, (_, count) in enumerate(ranked, start=1):
            cumulative += count
            cumulative_share_curve.append(cumulative / total)
            if count_to_50pct is None and cumulative / total >= 0.5:
                count_to_50pct = i

    return {
        "total_citations": total,
        "top_n_share": top_n_share,
        "unique_domains": len(ranked),
        "count_to_50pct": count_to_50pct,
        "top_n_ranked": [{"domain": d, "count": c, "share": c / total} for d, c in ranked[:TOP_N]],
        # one entry per domain rank -- the chart's cumulative-share-by-rank
        # curve is drawn from this directly, never recomputed by the
        # render script (which reads only the seam file and computes
        # nothing, same discipline as every other render_*_image.py).
        "cumulative_share_curve": cumulative_share_curve,
    }


def compute(inputs: FrozenInputs) -> FigureResult:
    rows, _panel_report = build_citation_panel(inputs)
    owned_domains = all_in_frame_owned_domains(inputs)
    mentioned_own_domains = mentioned_own_domains_by_run(inputs)

    rows_by_scope: dict[str, list[CitationRow]] = defaultdict(list)
    for row in rows:
        rows_by_scope[POOLED].append(row)
        rows_by_scope[row.engine].append(row)

    if not rows_by_scope:
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

    value: dict[str, dict[str, float | None]] = {}
    denominator_before_exclusion: dict[str, int] = {}
    denominator_after_exclusion: dict[str, dict[str, int]] = {}
    n_included: dict[str, dict[str, int]] = {}
    n_excluded: dict[str, dict[str, int]] = {}
    unique_domains: dict[str, dict[str, int | None]] = {}
    count_to_50pct: dict[str, dict[str, int | None]] = {}
    top_n_ranked: dict[str, dict[str, list[dict[str, Any]]]] = {}
    capterra_present: dict[str, dict[str, bool]] = {}
    capterra_count: dict[str, dict[str, int]] = {}
    g2_present: dict[str, dict[str, bool]] = {}
    g2_count: dict[str, dict[str, int]] = {}
    cumulative_share_curve: dict[str, dict[str, list[float]]] = {}
    domain_class_modes: dict[str, str] = {}  # domain -> modal url_class, computed once over all rows

    class_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        domain = registrable_domain(row.url_normalized)
        if domain and row.url_class:
            class_votes[domain][row.url_class] += 1
    for domain, votes in class_votes.items():
        domain_class_modes[domain] = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]

    for scope in sorted(rows_by_scope, key=lambda s: (s != POOLED, s)):
        scope_rows = rows_by_scope[scope]
        malformed = 0
        domains_by_treatment: dict[str, list[str]] = {t: [] for t in TREATMENTS}
        capterra_hits: dict[str, int] = {t: 0 for t in TREATMENTS}
        g2_hits: dict[str, int] = {t: 0 for t in TREATMENTS}

        for row in scope_rows:
            domain = registrable_domain(row.url_normalized)
            if domain is None:
                malformed += 1
                continue
            keep_by_treatment = {
                "included": True,
                "excluded_self_citation": domain not in mentioned_own_domains.get(row.run_id, set()),
                "excluded_population": domain not in owned_domains,
            }
            for t, keep in keep_by_treatment.items():
                if not keep:
                    continue
                domains_by_treatment[t].append(domain)
                if domain == CAPTERRA_DOMAIN:
                    capterra_hits[t] += 1
                if domain == G2_DOMAIN:
                    g2_hits[t] += 1

        stats_by_treatment = {t: _concentration(domains_by_treatment[t]) for t in TREATMENTS}

        denominator_before_exclusion[scope] = len(scope_rows)
        denominator_after_exclusion[scope] = {t: stats_by_treatment[t]["total_citations"] for t in TREATMENTS}
        n_included[scope] = denominator_after_exclusion[scope]
        n_excluded[scope] = {"malformed_domain": malformed}
        value[scope] = {t: stats_by_treatment[t]["top_n_share"] for t in TREATMENTS}
        unique_domains[scope] = {t: stats_by_treatment[t]["unique_domains"] for t in TREATMENTS}
        count_to_50pct[scope] = {t: stats_by_treatment[t]["count_to_50pct"] for t in TREATMENTS}
        top_n_ranked[scope] = {
            t: [
                {**entry, "modal_url_class": domain_class_modes.get(entry["domain"])}
                for entry in stats_by_treatment[t]["top_n_ranked"]
            ]
            for t in TREATMENTS
        }
        capterra_present[scope] = {t: capterra_hits[t] > 0 for t in TREATMENTS}
        capterra_count[scope] = capterra_hits
        g2_present[scope] = {t: g2_hits[t] > 0 for t in TREATMENTS}
        g2_count[scope] = g2_hits
        cumulative_share_curve[scope] = {t: stats_by_treatment[t]["cumulative_share_curve"] for t in TREATMENTS}

    # Plain observational statement, no verdict bands and no invented
    # prediction (2026-09-24 review feedback, second pass). The
    # pre-registered band is quoted verbatim in PRE_REG_BANDS_NOTE_PREFIX,
    # never derived from the observed numbers below -- only the
    # "observed per engine" half is computed here.
    engine_scopes = [s for s in sorted(value) if s != POOLED]
    per_engine_text = "; ".join(
        f"{engine}: " + ", ".join(f"{t}={value[engine][t]:.1%}" if value[engine][t] is not None else f"{t}=n/a" for t in TREATMENTS)
        for engine in engine_scopes
    )
    pre_reg_bands_note = PRE_REG_BANDS_NOTE_PREFIX + per_engine_text

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold=None,  # pre-specified table, no verdict bands (2026-09-24 review feedback)
        reason_untestable=None,
        notes={
            "unique_domains": unique_domains,
            "count_to_50pct": count_to_50pct,
            "top_10_ranked": top_n_ranked,
            "capterra_present": capterra_present,
            "capterra_count": capterra_count,
            "g2_present": g2_present,
            "g2_count": g2_count,
            "cumulative_share_curve": cumulative_share_curve,
            "scope_reading_note": SCOPE_READING_NOTE,
            "pre_reg_bands_note": pre_reg_bands_note,
            "treatment_rule_note": TREATMENT_RULE_NOTE,
            "retell_dapta_footnote": RETELL_DAPTA_FOOTNOTE,
            "denominator_scope_note": DENOMINATOR_SCOPE_NOTE,
        },
    )
