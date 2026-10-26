#!/usr/bin/env python3
"""Task 2 -- provisional entity roster.

Writes data/02_aux/roster_provisional.csv from two sources:
  (a) the coding pipeline's resolved `entities` (from --coding-db), plus the
      seed vendor list (config/seed_vendor_list.csv);
  (b) candidate brand strings from raw answer text -- the coding DB's
      `surface_forms` view rows that never resolved to an entity.

Provenance per row: seed | discovered | both. Domain comes from
entities.homepage_url, else the A4-prime entity_verification.verified_homepage_url,
else blank + needs_manual=true (no guessing). The file is marked PROVISIONAL.
Rerunnable: existing manual domain fixes and needs_manual flags are preserved
when a row is rebuilt.

The default --coding-db is data/03_coded/coding_a1.sqlite3 (the real-window coding run).

Usage:
    uv run python -m scripts.aux.build_roster
    uv run python -m scripts.aux.build_roster --coding-db data/03_coded/coding_a1.sqlite3 --in-frame-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aux.roster import DEFAULT_CODING_DB, DEFAULT_ROSTER_CSV, RosterEntity, load_from_csv, write_csv
from analysis.coding import storage as coding_storage
from analysis.coding.a4_entity_classification import DEFAULT_SEED_LIST_PATH, is_in_seed_list, load_seed_list
from analysis.coding.b1_per_run_company import _domain_of
from collector.storage.db import get_connection
from scripts.aux._run import rel

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_CANDIDATE_MENTIONS = 2  # unresolved surface forms below this are noise


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "unknown"


def build(coding_db: Path, seed_list_path: Path, in_frame_only: bool) -> tuple[list[RosterEntity], dict]:
    conn = get_connection(coding_db)
    try:
        entities = coding_storage.list_entities(conn=conn)
        surface_forms = coding_storage.list_surface_forms(conn=conn)
        alt_rows = coding_storage.list_alt_domains(conn=conn)
        verified_homepage = {
            v["entity_id"]: v.get("verified_homepage_url")
            for v in coding_storage.list_entity_verification(conn=conn)
        }
    finally:
        conn.close()

    seed_list = load_seed_list(seed_list_path) if Path(seed_list_path).exists() else set()
    alt_by_entity: dict[str, list[str]] = {}
    for r in alt_rows:
        alt_by_entity.setdefault(r["entity_id"], []).append(_domain_of(r["alt_domain"]))

    resolved_surface_forms = {
        sf for e in entities for sf in (e.get("surface_forms") or "").split(";") if sf
    }

    rows: list[RosterEntity] = []
    for e in entities:
        if in_frame_only and e.get("meets_frame") != "in":
            continue
        name = e.get("canonical_name") or e["entity_id"]
        homepage = e.get("homepage_url")
        from_verification = False
        if not homepage:
            homepage = verified_homepage.get(e["entity_id"])
            from_verification = bool(homepage)
        primary = _domain_of(homepage) if homepage else None
        in_seed = bool(e.get("in_seed_list")) or is_in_seed_list(
            canonical_name=name, homepage_url=homepage, seed_list=seed_list
        )
        if primary and from_verification:
            note = "homepage from A4' verification"
        elif primary:
            note = ""
        else:
            note = "no homepage_url in coding DB"
        rows.append(
            RosterEntity(
                entity_id=e["entity_id"],
                canonical_name=name,
                primary_domain=primary,
                alt_domains=sorted({d for d in alt_by_entity.get(e["entity_id"], []) if d and d != primary}),
                provenance="both" if in_seed else "discovered",
                needs_manual=primary is None,
                notes=note,
            )
        )

    # seed-list entries the pipeline never surfaced as an entity
    known_names = {r.canonical_name.strip().lower() for r in rows}
    seed_only = 0
    if Path(seed_list_path).exists():
        import csv as _csv

        with Path(seed_list_path).open(encoding="utf-8") as f:
            for r in _csv.DictReader(line for line in f if not line.startswith("#")):
                nm = (r.get("canonical_name") or "").strip()
                dom = (r.get("homepage_domain") or "").strip() or None
                if nm and nm.lower() not in known_names:
                    rows.append(
                        RosterEntity(
                            entity_id=_slug(nm),
                            canonical_name=nm,
                            primary_domain=_domain_of(dom) if dom else None,
                            provenance="seed",
                            needs_manual=dom is None,
                            notes="from seed list; not surfaced by the coding pipeline",
                        )
                    )
                    seed_only += 1

    # unresolved candidate brand strings from answer text
    candidates = 0
    for sf in surface_forms:
        form = sf["surface_form"]
        if form in resolved_surface_forms:
            continue
        if (sf.get("mention_count") or 0) < MIN_CANDIDATE_MENTIONS:
            continue
        rows.append(
            RosterEntity(
                entity_id=f"cand-{_slug(form)}",
                canonical_name=form,
                primary_domain=None,
                provenance="discovered",
                needs_manual=True,
                notes=f"unresolved surface form, {sf['mention_count']} mentions; resolve or discard",
            )
        )
        candidates += 1

    stats = {
        "in_frame_entities": sum(1 for r in rows if r.provenance in {"seed", "both", "discovered"} and not r.entity_id.startswith("cand-")),
        "seed_only_rows": seed_only,
        "unresolved_candidates": candidates,
        "needs_manual": sum(1 for r in rows if r.needs_manual),
        "total_rows": len(rows),
    }
    return rows, stats


def merge_preserving_manual(new_rows: list[RosterEntity], existing_path: Path) -> list[RosterEntity]:
    if not existing_path.exists():
        return new_rows
    try:
        existing = {r.entity_id: r for r in load_from_csv(existing_path)}
    except Exception:
        return new_rows
    merged: list[RosterEntity] = []
    for r in new_rows:
        prev = existing.get(r.entity_id)
        if prev and prev.primary_domain and not r.primary_domain:
            r.primary_domain = prev.primary_domain
            r.needs_manual = prev.needs_manual
            r.notes = (prev.notes or r.notes) + " [domain kept from prior manual edit]"
        if prev and not prev.needs_manual and r.needs_manual and prev.primary_domain:
            r.needs_manual = False
        merged.append(r)
    # keep purely-manual rows that the rebuild didn't reproduce
    new_ids = {r.entity_id for r in new_rows}
    for eid, prev in existing.items():
        if eid not in new_ids and prev.notes and "manual" in prev.notes.lower():
            merged.append(prev)
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coding-db", type=Path, default=DEFAULT_CODING_DB)
    p.add_argument("--seed-list", type=Path, default=DEFAULT_SEED_LIST_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_ROSTER_CSV)
    p.add_argument("--in-frame-only", action="store_true", default=True)
    p.add_argument("--all-entities", dest="in_frame_only", action="store_false")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.coding_db.exists():
        raise SystemExit(f"coding DB not found: {args.coding_db}")
    rows, stats = build(args.coding_db, args.seed_list, args.in_frame_only)
    rows = merge_preserving_manual(rows, args.out)
    write_csv(rows, args.out)
    print(f"[roster] wrote {rel(args.out)}")
    for k, v in stats.items():
        print(f"[roster]   {k}: {v}")
    print(f"[roster]   source: {args.coding_db.name} (in_frame_only={args.in_frame_only})")


if __name__ == "__main__":
    main()
