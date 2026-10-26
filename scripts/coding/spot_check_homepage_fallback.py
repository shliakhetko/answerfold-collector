#!/usr/bin/env python3
"""Ad hoc spot check (2026-09-22) -- NOT part of the Phase B compute/
display contract, marks nothing as verified. Picks 10 of the 141
in-frame entities whose owned domain comes from
entity_verification.verified_homepage_url (entity_registry.homepage_url
blank), restricted to the ones with at least one actual owned=True
citation match in this window (so there is always a real cited URL to
show), sampled at random with a fixed seed for reproducibility.

Writes a Markdown file for hand review. This script marks nothing as
verified -- that judgment belongs to the person reading the file.

Usage:
    uv run python -m scripts.coding.spot_check_homepage_fallback
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from analysis.phase_b.domain_ownership import build_source_ownership, registrable_domain
from analysis.phase_b.reading import PhaseBInputError, read_frozen_inputs

from collector import paths
REPO_ROOT = paths.REPO_ROOT
OUTPUT_PATH = paths.HOMEPAGE_FALLBACK_SPOT_CHECK_MD
SAMPLE_SIZE = 10
RANDOM_SEED = 20260922  # date-based, matches the A5 spot-check convention (see ambiguity ledger, 2026-09-12)


def main() -> int:
    try:
        inputs = read_frozen_inputs(REPO_ROOT)
    except PhaseBInputError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1

    rows, _report = build_source_ownership(
        inputs.mentions_raw,
        inputs.entities,
        inputs.alt_domains_rows,
        inputs.sources_by_run,
        inputs.raw_runs,
        inputs.entity_verification_rows,
    )

    verified_by_entity = {
        r["entity_id"]: r["verified_homepage_url"] for r in inputs.entity_verification_rows if r.get("verified_homepage_url")
    }
    entities_by_id = {e["entity_id"]: e for e in inputs.entities}
    disambiguation_by_entity = {r["entity_id"]: r.get("disambiguation_note") for r in inputs.entity_verification_rows}

    fallback_entity_ids = sorted(
        e["entity_id"]
        for e in inputs.entities
        if e.get("meets_frame") == "in" and not e.get("homepage_url") and verified_by_entity.get(e["entity_id"])
    )

    matched_row_by_entity: dict[str, Any] = {}
    for row in rows:
        if row.owned is True and row.entity_id in fallback_entity_ids and row.entity_id not in matched_row_by_entity:
            matched_row_by_entity[row.entity_id] = row

    with_match = sorted(matched_row_by_entity)
    rng = random.Random(RANDOM_SEED)
    sample_ids = rng.sample(with_match, min(SAMPLE_SIZE, len(with_match)))

    lines = [
        "# Homepage fallback spot check",
        "",
        f"Generated 2026-09-22 by `scripts.coding.spot_check_homepage_fallback`, seed {RANDOM_SEED}. "
        "**Not verified** -- this is a hand-review input, no row here has been marked confirmed by anyone.",
        "",
        f"Population: {len(fallback_entity_ids)} in-frame entities whose owned domain comes from "
        f"`entity_verification.verified_homepage_url` (blank `entity_registry.homepage_url`). Of those, "
        f"{len(with_match)} have at least one actual `owned=True` citation match in this window's data "
        f"-- the sample below is drawn from that {len(with_match)}, not all {len(fallback_entity_ids)} "
        f"(so a real cited URL exists to show for every row; the other "
        f"{len(fallback_entity_ids) - len(with_match)} entities never had their fallback domain cited at "
        "all in this window, which is a separate, expected fact, not a defect in the fallback).",
        "",
        "| entity_id | canonical_name | fallback domain (verified_homepage_url) | matched cited URL | run_id |",
        "| --- | --- | --- | --- | --- |",
    ]

    for entity_id in sample_ids:
        entity = entities_by_id[entity_id]
        row = matched_row_by_entity[entity_id]
        raw_homepage = verified_by_entity[entity_id]
        fallback_domain = registrable_domain(raw_homepage)
        sources = inputs.sources_by_run.get(row.run_id, [])
        cited_url = next(
            (s.get("url_raw") for s in sources if registrable_domain(s.get("url_raw")) == fallback_domain),
            "(not found -- see matched_domain instead)",
        )
        lines.append(
            f"| `{entity_id}` | {entity['canonical_name']} | {raw_homepage} (`{fallback_domain}`) | {cited_url} | `{row.run_id}` |"
        )

    lines.append("")
    lines.append("## Disambiguation notes on record (A4-prime), for context")
    lines.append("")
    for entity_id in sample_ids:
        note = disambiguation_by_entity.get(entity_id)
        lines.append(f"- `{entity_id}`: {note or '(none)'}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(sample_ids)} entities sampled from {len(with_match)} with a citation match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
