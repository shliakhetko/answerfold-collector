#!/usr/bin/env python3
"""Ad hoc scan (2026-09-22) -- NOT part of the Phase B compute/display
contract, fixes nothing. Prompted by the homepage-fallback spot check
(reports/homepage_fallback_spot_check.md) flagging one likely wrong
domain (agentshub -> bibha.ai) among the 48 fallback entities with a
cited-URL match. This scan covers the other case a spot check can't
reach: all 141 fallback entities, most of which never got their own
domain cited in this window (see the spot-check file's note), so a
wrong domain among them shows up nowhere as a bad match -- only as a
name/domain mismatch, checkable without a citation at all.

For every in-frame entity whose owned domain comes from
entity_verification.verified_homepage_url (entity_registry.homepage_url
blank), checks whether the fallback domain contains the entity's brand
name after normalizing both (lowercase; strip spaces, dots, hyphens,
and the substring "ai"). Lists every one that does NOT contain it.

Usage:
    uv run python -m scripts.coding.scan_homepage_fallback_name_mismatch
"""
from __future__ import annotations

import sys
from pathlib import Path

from analysis.phase_b.domain_ownership import registrable_domain
from analysis.phase_b.reading import PhaseBInputError, read_frozen_inputs

from collector import paths
REPO_ROOT = paths.REPO_ROOT
OUTPUT_PATH = paths.HOMEPAGE_FALLBACK_NAME_MISMATCH_MD


def _normalize(s: str) -> str:
    s = s.lower()
    for ch in (" ", ".", "-"):
        s = s.replace(ch, "")
    return s.replace("ai", "")


def main() -> int:
    try:
        inputs = read_frozen_inputs(REPO_ROOT)
    except PhaseBInputError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1

    verified_by_entity = {
        r["entity_id"]: r["verified_homepage_url"] for r in inputs.entity_verification_rows if r.get("verified_homepage_url")
    }
    disambiguation_by_entity = {r["entity_id"]: r.get("disambiguation_note") for r in inputs.entity_verification_rows}

    fallback_entities = [
        e
        for e in inputs.entities
        if e.get("meets_frame") == "in" and not e.get("homepage_url") and verified_by_entity.get(e["entity_id"])
    ]

    mismatches = []
    for e in fallback_entities:
        raw_homepage = verified_by_entity[e["entity_id"]]
        fallback_domain = registrable_domain(raw_homepage)
        if fallback_domain is None:
            continue
        brand = _normalize(e["canonical_name"])
        domain_norm = _normalize(fallback_domain)
        if brand and brand not in domain_norm:
            mismatches.append((e["entity_id"], e["canonical_name"], raw_homepage, fallback_domain))

    lines = [
        "# Homepage fallback: name/domain mismatch scan",
        "",
        f"Generated 2026-09-22 by `scripts.coding.scan_homepage_fallback_name_mismatch`. Not fixed -- for hand review.",
        "",
        f"Population: all {len(fallback_entities)} in-frame entities whose owned domain comes from "
        "`entity_verification.verified_homepage_url` (blank `entity_registry.homepage_url`) -- not just the "
        "48 with a cited-URL match the spot check covered. A wrong domain among the other "
        f"{len(fallback_entities) - 48} would show up nowhere as a bad match (no citation to check against), "
        "only as a name/domain mismatch, which is what this scan checks instead.",
        "",
        f"Match rule: lowercase both `canonical_name` and the fallback domain, strip spaces/dots/hyphens/"
        f'"ai", then check the brand name is a substring of the domain. **{len(mismatches)} of '
        f"{len(fallback_entities)} fail this check** -- flagged below, not fixed.",
        "",
        "| entity_id | canonical_name | fallback domain | A4-prime disambiguation_note |",
        "| --- | --- | --- | --- |",
    ]
    for entity_id, canonical_name, raw_homepage, fallback_domain in mismatches:
        note = disambiguation_by_entity.get(entity_id) or "(none)"
        lines.append(f"| `{entity_id}` | {canonical_name} | {raw_homepage} (`{fallback_domain}`) | {note} |")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(mismatches)} of {len(fallback_entities)} flagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
