"""Phase B shared intermediate: the run x company panel S2 reads.

Composes two existing joins rather than reimplementing either:

- analysis.coding.b1_per_run_company.build_per_run_company for
  `mentioned` / `recommended` / `first_mention_position` -- reused
  verbatim, since this is already-correct LLM-derived mention/
  recommendation logic with its own edge rule (named as a competitor,
  in a caveat, or in an aside counts as mentioned but not recommended).

- analysis.phase_b.domain_ownership.build_source_ownership for
  `own_domain_cited` / `linked`, via OwnershipRow.owned keyed by
  (run_id, entity_id) -- deliberately OVERRIDING B1's own weaker
  own_domain_cited, which is a plain bool (never None) because its
  entity_domains() reads homepage_url only, with no fallback to
  entity_verification.verified_homepage_url -- a gap Phase B's own join
  already fixed 2026-09-22 for 141 of 415 in-frame entities (see the
  ambiguity ledger, own-domain-join-additional-gaps-2026-09-22). Using
  B1's own_domain_cited directly for a headline "pitch" figure like S2
  would silently inherit that known bug.

own_domain_cited/linked are three-valued: None when the entity has no
known domain at all (unmeasurable), never coerced to False. This is a
real, consequential difference from B1's own own_domain_cited, which has
no unmeasurable state at all -- reported explicitly in the build report,
not left for a reader to infer.

The eligibility floor (>=5 mentions across intents 1-6, within an engine)
is reused from analysis.coding.b3_aggregates.apply_eligibility_floor,
computed from the B1 rows (mention volume only) before the ownership
override -- not reimplemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analysis.coding.b1_per_run_company import build_per_run_company
from analysis.coding.b3_aggregates import EligibilityReport, apply_eligibility_floor
from analysis.phase_b import domain_ownership
from analysis.phase_b.citation_panel import UNBRANDED_INTENT_IDS
from analysis.phase_b.reading import FrozenInputs


@dataclass(frozen=True)
class RunCompanyRow:
    run_id: str
    entity_id: str
    engine: str | None
    intent_id: int | None
    mentioned: bool
    recommended: bool
    own_domain_cited: bool | None  # None = entity has no known domain, unmeasurable
    linked: bool | None
    first_mention_position: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "entity_id": self.entity_id,
            "engine": self.engine,
            "intent_id": self.intent_id,
            "mentioned": self.mentioned,
            "recommended": self.recommended,
            "own_domain_cited": self.own_domain_cited,
            "linked": self.linked,
            "first_mention_position": self.first_mention_position,
        }


@dataclass(frozen=True)
class RunCompanyPanelBuildReport:
    """No headline number lives here -- a build report, not a figure
    record, same convention as domain_ownership.BuildReport.

    own_domain_cited_true_false_null and rows_flipped_false_to_none are
    restricted to mentioned==True rows -- the only population S2 (or any
    other own_domain_cited consumer) actually reads. Counting non-mentioned
    rows too would swamp the real delta this module exists to fix with a
    trivial majority: domain_ownership.build_source_ownership only ever
    considers mentioned, in-frame candidate pairs, so every non-mentioned
    row's owned lookup is always a miss (never computed, not "found no
    domain") -- a different fact from a mentioned entity genuinely having
    no known domain."""

    rows_produced: int  # B1's full run x in-frame-entity cross product
    mentioned_rows: int
    own_domain_cited_true_false_null: dict[str, int]  # mentioned rows only
    # How many mentioned rows flip from B1's own_domain_cited=False to
    # this panel's own_domain_cited=None -- the real, consequential
    # B1-vs-Phase-B delta this module exists to fix, stated as a fact
    # rather than left implicit (module docstring above).
    rows_flipped_false_to_none: int
    qualifying_entities_by_engine: dict[str, int]  # eligibility floor, per engine
    # Passed through from domain_ownership.BuildReport (computed here
    # anyway, previously discarded) so S2 can state whether it depends on
    # a real alternates fix or a zero one, per its own spec text ("If
    # alternates are still zero, S2 carries the limitation line") without
    # a figure needing to build_source_ownership a second time itself.
    entities_with_alternates: int
    matches_via_alternate: int

    def to_json(self) -> dict[str, Any]:
        return {
            "rows_produced": self.rows_produced,
            "mentioned_rows": self.mentioned_rows,
            "own_domain_cited_true_false_null": self.own_domain_cited_true_false_null,
            "rows_flipped_false_to_none": self.rows_flipped_false_to_none,
            "qualifying_entities_by_engine": self.qualifying_entities_by_engine,
            "entities_with_alternates": self.entities_with_alternates,
            "matches_via_alternate": self.matches_via_alternate,
        }


def build_run_company_panel(
    inputs: FrozenInputs,
) -> tuple[list[RunCompanyRow], EligibilityReport, RunCompanyPanelBuildReport]:
    b1_rows = build_per_run_company(
        inputs.raw_runs,
        inputs.mentions_raw,
        inputs.entities,
        inputs.sources_by_run,
        inputs.alt_domains_rows,
    )

    run_meta_by_id = {
        r["run_id"]: {"engine": r.get("engine"), "intent_id": r.get("intent_id"), "day_index": r.get("day_index")}
        for r in inputs.raw_runs
    }

    eligibility = apply_eligibility_floor(b1_rows, run_meta_by_id, UNBRANDED_INTENT_IDS)

    ownership_rows, ownership_report = domain_ownership.build_source_ownership(
        inputs.mentions_raw,
        inputs.entities,
        inputs.alt_domains_rows,
        inputs.sources_by_run,
        inputs.raw_runs,
        inputs.entity_verification_rows,
    )
    owned_by_pair: dict[tuple[str, str], bool | None] = {
        (row.run_id, row.entity_id): row.owned for row in ownership_rows
    }

    rows: list[RunCompanyRow] = []
    own_domain_counts = {"true": 0, "false": 0, "null": 0}
    rows_flipped_false_to_none = 0
    for b1_row in b1_rows:
        meta = run_meta_by_id.get(b1_row.run_id, {})
        # domain_ownership.build_source_ownership only ever builds a
        # candidate pair for a MENTIONED, in-frame entity -- for a
        # non-mentioned pair, owned_by_pair has no entry at all (never
        # considered), which is a different fact from "considered and
        # found no domain." Reported as None here either way (this panel
        # makes no ownership claim for a non-mention), but only mentioned
        # rows enter the build report's own_domain_cited counts below, so
        # that trivial majority never swamps the real delta.
        owned = owned_by_pair.get((b1_row.run_id, b1_row.entity_id))
        if b1_row.mentioned:
            if owned is None:
                own_domain_counts["null"] += 1
                if b1_row.own_domain_cited is False:
                    rows_flipped_false_to_none += 1
            elif owned is True:
                own_domain_counts["true"] += 1
            else:
                own_domain_counts["false"] += 1

        rows.append(
            RunCompanyRow(
                run_id=b1_row.run_id,
                entity_id=b1_row.entity_id,
                engine=meta.get("engine"),
                intent_id=meta.get("intent_id"),
                mentioned=b1_row.mentioned,
                recommended=b1_row.recommended,
                own_domain_cited=owned,
                linked=owned,
                first_mention_position=b1_row.first_mention_position,
            )
        )

    report = RunCompanyPanelBuildReport(
        rows_produced=len(rows),
        mentioned_rows=sum(1 for r in rows if r.mentioned),
        own_domain_cited_true_false_null=own_domain_counts,
        rows_flipped_false_to_none=rows_flipped_false_to_none,
        qualifying_entities_by_engine={
            engine: eligibility.qualifying_count(engine) for engine in eligibility.qualifying_by_engine
        },
        entities_with_alternates=ownership_report.entities_with_alternates,
        matches_via_alternate=ownership_report.matches_via_alternate,
    )
    return rows, eligibility, report
