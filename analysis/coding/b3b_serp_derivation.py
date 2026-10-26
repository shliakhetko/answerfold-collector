"""B3b: A1-derivation (Post-Collection Coding Pipeline Spec). No model.

google_top10_serp is captured during the window as raw top 10 organic URLs
per intent head phrase -- pre-entity-resolution: at capture time nothing
knows which company owns which URL yet. The per-company binary is derived
here by joining those URLs to entity_registry.homepage_url plus
alt_domains.csv. This is why figure "A1" survives the population being
built after collection (note: figure ID "A1" is unrelated to pipeline
stage A1, answer coding -- same label, different thing, spelled out here to
avoid confusion between the two, same note as in b4_figures.py).

schema.sql's `ranking_snapshot` table requires entity_id NOT NULL, so it
fits storing this function's *output* (once real content exists), not the
*input* raw URL capture the join needs -- there is no per-intent raw SERP
capture mechanism in collector/ today (out of scope for this repo's
current instrument), so raw_serp_rows is accepted as a plain in-memory
parameter rather than read from any table.
"""
from __future__ import annotations

from typing import Any

from analysis.coding.b1_per_run_company import _domain_of, entity_domains, in_frame_entities


def derive_serp_binary(
    raw_serp_rows: list[dict[str, Any]],  # each: {"intent_id": int, "position": int, "url": str}
    entity_registry_rows: list[dict[str, Any]],
    alt_domains_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per (entity_id, intent_id) pair that appeared in the top 10 --
    entities never in any raw_serp_rows URL for an intent simply have no
    row for that intent (absence means False, same discipline as B1's
    mentioned field -- a caller wanting the full cross product can build it
    from entity_registry_rows x the set of intent_ids seen in raw_serp_rows)."""
    entities = in_frame_entities(entity_registry_rows)
    domains_by_entity = entity_domains(entities, alt_domains_rows)

    hits: set[tuple[str, int]] = set()
    for row in raw_serp_rows:
        url_domain = _domain_of(row["url"])
        for entity_id, domains in domains_by_entity.items():
            if url_domain in domains:
                hits.add((entity_id, row["intent_id"]))

    return [{"entity_id": entity_id, "intent_id": intent_id, "in_google_top10": True} for entity_id, intent_id in sorted(hits)]
