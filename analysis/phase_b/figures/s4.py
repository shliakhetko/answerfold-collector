"""S4 -- share of usable runs with search_fired=False, per engine x
intent x paraphrase (exploratory, not pre-registered).

Follows up the finding ledgered in
`s3-search-fired-zero-sources-identity-noticed-2026-09-12` (stubbed
2026-09-22): for three engines the zero-source runs and the
search_fired=False runs are the same set, concentrated in one Gemini
paraphrase. S3 is untouched: this figure has its own unit (a prompt
string per engine) and does not feed S3.

All seven intents are included (S4 is a description of retrieval
behaviour, not a per-company statistic, so the unbranded-only rule that
governs pooled company metrics does not apply); each cell carries the
intent's branded flag through the supporting table so a reader can
separate them.

value[engine]["i{intent}-p{paraphrase}"] = share of usable runs in that
cell with search_fired is False. Runs whose search_fired is None (not
reported) are excluded with a named reason, never counted as False.
Usable/excluded comes from analysis.phase_b.exclusions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.phase_b.exclusions import exclusion_reason
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "S4"
SEARCH_FIRED_UNKNOWN = "search_fired_not_reported"
LEDGER_REFS = (
    "s3-search-fired-zero-sources-identity-noticed-2026-09-12",
    "2026-09-22 stub entry (S4 stubbed)",
    "s4-search-fired-build-2026-09-29",
)


def cell_key(intent_id: Any, paraphrase_id: Any) -> str:
    return f"i{intent_id}-p{paraphrase_id}"


def compute(inputs: FrozenInputs) -> FigureResult:
    runs_by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in inputs.raw_runs:
        runs_by_engine[run["engine"]].append(run)
    if not runs_by_engine:
        return FigureResult(
            figure_id=FIGURE_ID, status="untestable", value=None, denominator_before_exclusion=None,
            denominator_after_exclusion=None, n_included=None, n_excluded=None, threshold=None,
            reason_untestable="no runs",
        )

    value: dict[str, dict[str, float]] = {}
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    excluded: dict[str, dict[str, int]] = {}
    supporting: list[dict[str, Any]] = []
    overall_no_search: dict[str, int] = {}

    for engine in sorted(runs_by_engine):
        cells: dict[tuple[Any, Any], list[int]] = defaultdict(lambda: [0, 0])  # [n_usable, n_no_search]
        reasons: dict[str, int] = defaultdict(int)
        for run in runs_by_engine[engine]:
            reason = exclusion_reason(run)
            if reason is None and run.get("search_fired") is None:
                reason = SEARCH_FIRED_UNKNOWN
            if reason is not None:
                reasons[reason] += 1
                continue
            c = cells[(run["intent_id"], run["paraphrase_id"])]
            c[0] += 1
            if run["search_fired"] is False:
                c[1] += 1
        value[engine] = {}
        for (intent_id, paraphrase_id) in sorted(cells):
            n_usable, n_no_search = cells[(intent_id, paraphrase_id)]
            share = n_no_search / n_usable
            value[engine][cell_key(intent_id, paraphrase_id)] = share
            supporting.append(
                {"engine": engine, "intent_id": intent_id, "paraphrase_id": paraphrase_id,
                 "n_usable_runs": n_usable, "n_search_not_fired": n_no_search, "share": share}
            )
        before[engine] = len(runs_by_engine[engine])
        after[engine] = sum(c[0] for c in cells.values())
        overall_no_search[engine] = sum(c[1] for c in cells.values())
        excluded[engine] = dict(reasons)

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=before,
        denominator_after_exclusion=after,
        n_included=dict(after),
        n_excluded=excluded,
        threshold=None,  # exploratory -- no pre-registered threshold, same as S3
        reason_untestable=None,
        supporting_table=supporting,
        notes={
            "exploratory": True,
            "n_search_not_fired_by_engine": overall_no_search,
            "ledger_refs": list(LEDGER_REFS),
            "unit": "share of usable runs with search_fired=False per engine x intent x paraphrase cell",
        },
    )
