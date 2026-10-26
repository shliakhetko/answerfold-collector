"""B2: per run rollups (Post-Collection Coding Pipeline Spec). No model.

n_out_of_frame_brands is a COUNT only (spec: "Do not let entities failing
the frame enter any denominator. They are a count.") -- computed from
mentions_raw + the full entity_registry (including out-of-frame rows), not
from per_run_company (which only ever contains in-frame entities).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analysis.coding.b1_per_run_company import (
    PerRunCompanyRow,
    _domain_of,
    in_frame_entities,
    surface_form_to_entity_id,
)


@dataclass(frozen=True)
class PerRunRollupRow:
    run_id: str
    n_companies_mentioned: int
    n_sources: int
    vendor_source_share: float | None  # None when n_sources == 0 -- no share of nothing
    n_out_of_frame_brands: int


def build_per_run_rollups(
    per_run_company_rows: list[PerRunCompanyRow],
    run_rows: list[dict[str, Any]],
    mentions_raw_rows: list[dict[str, Any]],
    sources_by_run: dict[str, list[dict[str, Any]]],
    entity_registry_rows: list[dict[str, Any]],
    url_class_by_domain: dict[str, str] | None = None,
) -> list[PerRunRollupRow]:
    """url_class_by_domain: normalized domain -> url_class from
    url_registry.csv (A5), used to compute vendor_source_share ("vendor
    page"/"vendor listicle" sources over all sources for that run). None or
    an empty mapping means every share comes back as 0.0 for a run with
    sources, rather than raising -- A5 may not have run yet."""
    url_class_by_domain = url_class_by_domain or {}

    mentioned_by_run: dict[str, int] = {}
    for row in per_run_company_rows:
        if row.mentioned:
            mentioned_by_run[row.run_id] = mentioned_by_run.get(row.run_id, 0) + 1

    in_frame_ids = {e["entity_id"] for e in in_frame_entities(entity_registry_rows)}
    out_of_frame_entities = [e for e in entity_registry_rows if e["entity_id"] not in in_frame_ids]
    out_of_frame_sf_to_id = surface_form_to_entity_id(out_of_frame_entities)

    out_of_frame_by_run: dict[str, set[str]] = {}
    for mention in mentions_raw_rows:
        entity_id = out_of_frame_sf_to_id.get(mention["surface_form"])
        if entity_id is not None:
            out_of_frame_by_run.setdefault(mention["run_id"], set()).add(entity_id)

    rows: list[PerRunRollupRow] = []
    for run in run_rows:
        run_id = run["run_id"]
        sources = sources_by_run.get(run_id, [])
        n_sources = len(sources)

        vendor_source_share: float | None = None
        if n_sources > 0:
            vendor_sources = sum(
                1
                for s in sources
                if url_class_by_domain.get(_domain_of(s.get("domain") or s.get("url_raw", ""))) in
                ("vendor page", "vendor listicle")
            )
            vendor_source_share = vendor_sources / n_sources

        rows.append(
            PerRunRollupRow(
                run_id=run_id,
                n_companies_mentioned=mentioned_by_run.get(run_id, 0),
                n_sources=n_sources,
                vendor_source_share=vendor_source_share,
                n_out_of_frame_brands=len(out_of_frame_by_run.get(run_id, set())),
            )
        )
    return rows
