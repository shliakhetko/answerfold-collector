"""S3 -- Answers citing nothing (pinned EXPERIMENT.md, Notion Part 3.2).

Per engine, the share of usable responses that attached no sources at
all. First registered figure: needs only raw rows (no shared
intermediate), so it exercises the reading -> computing -> serializing
path end to end cheaply, and a later headline figure (S2) is
uninterpretable without knowing this baseline first.

denominator_after_exclusion is exactly what the share was computed over:
the usable (post-exclusion) pool, same number as n_included. They are not
independent facts here -- n_included exists as its own field because a
future figure may legitimately have denominator_after_exclusion !=
n_included (a pooled statistic whose divisor isn't literally "how many
units entered"), and the contract is that denominator_after_exclusion is
always the true divisor. denominator_before_exclusion is the full
per-engine run count prior to any exclusion (usable + excluded); "how
many were excluded, and why" is recorded in n_excluded for the record.

Usable/excluded is decided once, in analysis.phase_b.exclusions, and
reused verbatim -- not reimplemented here.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.phase_b.exclusions import exclusion_reason
from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.reading import FrozenInputs

FIGURE_ID = "S3"


def _search_fired_zero_sources_identity(
    zero_sources_run_ids: set[str], no_search_run_ids: set[str]
) -> dict[str, Any]:
    """Whether "attached no sources" and "never fired search" are the
    identical set of runs, in both directions -- stated as a computed
    fact here, not left for a reader to infer by eyeballing two supporting-
    table columns. Not assumed to hold for every engine: google_ai_mode has
    no real search-tool concept and its zero-source cases turn out to be a
    different phenomenon entirely (search_fired is always True there)."""
    both = zero_sources_run_ids & no_search_run_ids
    return {
        "n_zero_sources": len(zero_sources_run_ids),
        "n_no_search": len(no_search_run_ids),
        "n_both": len(both),
        "identical_sets": zero_sources_run_ids == no_search_run_ids,
    }


def compute(inputs: FrozenInputs) -> FigureResult:
    runs_by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in inputs.raw_runs:
        runs_by_engine[run["engine"]].append(run)

    value: dict[str, float | None] = {}
    denominator_before_exclusion: dict[str, int] = {}
    denominator_after_exclusion: dict[str, int] = {}
    n_included: dict[str, int] = {}
    n_excluded: dict[str, dict[str, int]] = {}
    supporting_table: list[dict[str, Any]] = []
    untestable_engines: dict[str, str] = {}
    search_fired_identity: dict[str, dict[str, Any]] = {}

    for engine in sorted(runs_by_engine):
        excluded_by_reason: dict[str, int] = defaultdict(int)
        no_sources = 0
        usable = 0
        zero_sources_run_ids: set[str] = set()
        no_search_run_ids: set[str] = set()

        for run in runs_by_engine[engine]:
            reason = exclusion_reason(run)
            included = reason is None
            has_sources = None
            if included:
                usable += 1
                sources = inputs.sources_by_run.get(run["run_id"], [])
                has_sources = len(sources) > 0
                if not has_sources:
                    no_sources += 1
                    zero_sources_run_ids.add(run["run_id"])
                if run.get("search_fired") is False:
                    no_search_run_ids.add(run["run_id"])
            else:
                excluded_by_reason[reason] += 1

            supporting_table.append(
                {
                    "run_id": run["run_id"],
                    "engine": engine,
                    "paraphrase_id": run.get("paraphrase_id"),
                    "has_sources": has_sources,
                    "search_fired": run.get("search_fired"),
                    "included": included,
                    "excluded_reason": reason,
                }
            )

        search_fired_identity[engine] = _search_fired_zero_sources_identity(zero_sources_run_ids, no_search_run_ids)
        # denominator_after_exclusion == n_included: the usable pool is
        # what the share was actually computed over (see module
        # docstring). denominator_before_exclusion is every run for this
        # engine prior to exclusion.
        denominator_before_exclusion[engine] = len(runs_by_engine[engine])
        denominator_after_exclusion[engine] = usable
        n_included[engine] = usable
        n_excluded[engine] = dict(excluded_by_reason)
        if usable == 0:
            value[engine] = None
            untestable_engines[engine] = "zero usable responses for this engine"
        else:
            value[engine] = no_sources / usable

    if untestable_engines and len(untestable_engines) == len(runs_by_engine):
        return FigureResult(
            figure_id=FIGURE_ID,
            status="untestable",
            value=None,
            denominator_before_exclusion=None,
            denominator_after_exclusion=None,
            n_included=None,
            n_excluded=None,
            threshold=None,
            reason_untestable="zero usable responses across every engine",
            supporting_table=supporting_table,
        )

    return FigureResult(
        figure_id=FIGURE_ID,
        status="ok",
        value=value,
        denominator_before_exclusion=denominator_before_exclusion,
        denominator_after_exclusion=denominator_after_exclusion,
        n_included=n_included,
        n_excluded=n_excluded,
        threshold=None,  # no pre-registered threshold for S3
        reason_untestable=None,
        supporting_table=supporting_table,
        notes={"search_fired_zero_sources_identity": search_fired_identity},
    )
