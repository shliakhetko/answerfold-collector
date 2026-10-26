"""The roster the auxiliary collectors target.

Per the user's decision, the roster comes from the coding pipeline's resolved
`entities` table (rows with meets_frame == "in") -- the verified in-vertical
company list produced after the A1-A5 AI aggregation. `--coding-db` defaults to
data/03_coded/coding_a1.sqlite3, the real-window coding run (its `final_entity_list`
view is the same 415-company set).

A homepage comes from `entities.homepage_url` when present, otherwise from the
A4-prime `entity_verification.verified_homepage_url` (populated for every
in-frame entity) -- only URLs the pipeline actually verified, never a guess.

Collectors are idempotent, so re-pointing at a newer coding DB later fetches
only the new rows.

This module also reads/writes data/02_aux/roster_provisional.csv, the flat file
Tasks 4/6/7 consume (so a collector run doesn't need a live DB connection and
manual domain fixes survive a rebuild).

Reuses analysis.coding.storage + analysis.coding.b1_per_run_company rather
than re-querying -- same join b3b_serp_derivation already performs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.coding import storage as coding_storage
from analysis.coding.b1_per_run_company import _domain_of, entity_domains, in_frame_entities
from collector.storage.db import get_connection

from collector import paths
DEFAULT_CODING_DB = paths.CODING_DB
DEFAULT_ROSTER_CSV = paths.AUX_ROSTER_CSV

ROSTER_CSV_HEADER = "# PROVISIONAL roster -- rebuilt by scripts.aux.build_roster; manual domain fixes are preserved on rebuild"
ROSTER_CSV_COLUMNS = [
    "entity_id",
    "canonical_name",
    "primary_domain",
    "alt_domains",
    "provenance",
    "needs_manual",
    "notes",
]


@dataclass
class RosterEntity:
    entity_id: str
    canonical_name: str
    primary_domain: str | None
    alt_domains: list[str] = field(default_factory=list)
    provenance: str = "seed"  # seed | discovered | both
    needs_manual: bool = False
    notes: str = ""

    @property
    def domains(self) -> list[str]:
        seen: list[str] = []
        for d in ([self.primary_domain] if self.primary_domain else []) + self.alt_domains:
            nd = _domain_of(d)
            if nd and nd not in seen:
                seen.append(nd)
        return seen


def load_from_coding_db(db_path: Path | str = DEFAULT_CODING_DB, *, in_frame_only: bool = True) -> list[RosterEntity]:
    conn = get_connection(db_path)
    try:
        entities = coding_storage.list_entities(conn=conn)
        alt_rows = coding_storage.list_alt_domains(conn=conn)
        verified_homepage = {
            v["entity_id"]: v.get("verified_homepage_url")
            for v in coding_storage.list_entity_verification(conn=conn)
        }
    finally:
        conn.close()

    selected = in_frame_entities(entities) if in_frame_only else entities
    domains_by_entity = entity_domains(selected, alt_rows)

    roster: list[RosterEntity] = []
    for e in selected:
        entity_id = e["entity_id"]
        homepage = e.get("homepage_url") or verified_homepage.get(entity_id)
        primary = _domain_of(homepage) if homepage else None
        alts = sorted(d for d in domains_by_entity.get(entity_id, set()) if d and d != primary)
        roster.append(
            RosterEntity(
                entity_id=entity_id,
                canonical_name=e.get("canonical_name") or entity_id,
                primary_domain=primary or None,
                alt_domains=alts,
                provenance="seed",
                needs_manual=primary is None,
                notes="" if primary else "no homepage_url in coding DB; resolve manually",
            )
        )
    return roster


def load_from_csv(path: Path | str = DEFAULT_ROSTER_CSV) -> list[RosterEntity]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python -m scripts.aux.build_roster` first (Task 2)."
        )
    roster: list[RosterEntity] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_line for _line in f if not _line.startswith("#"))
        for row in reader:
            roster.append(
                RosterEntity(
                    entity_id=row["entity_id"],
                    canonical_name=row.get("canonical_name") or row["entity_id"],
                    primary_domain=(row.get("primary_domain") or "").strip() or None,
                    alt_domains=[d for d in (row.get("alt_domains") or "").split(";") if d],
                    provenance=row.get("provenance") or "seed",
                    needs_manual=(row.get("needs_manual") or "").strip().lower() in {"1", "true", "yes"},
                    notes=row.get("notes") or "",
                )
            )
    return roster


def write_csv(rows: list[RosterEntity], path: Path | str = DEFAULT_ROSTER_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(ROSTER_CSV_HEADER + "\n")
        writer = csv.DictWriter(f, fieldnames=ROSTER_CSV_COLUMNS)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x.entity_id):
            writer.writerow(
                {
                    "entity_id": r.entity_id,
                    "canonical_name": r.canonical_name,
                    "primary_domain": r.primary_domain or "",
                    "alt_domains": ";".join(r.alt_domains),
                    "provenance": r.provenance,
                    "needs_manual": "true" if r.needs_manual else "false",
                    "notes": r.notes,
                }
            )
    return path


def resolvable_domains(roster: list[RosterEntity]) -> list[tuple[RosterEntity, str]]:
    """(entity, domain) pairs for every roster entity that has at least one
    resolvable domain -- the unit Tasks 4 and 6 iterate. Entities flagged
    needs_manual with no domain are skipped (and reported)."""
    out: list[tuple[RosterEntity, str]] = []
    for entity in roster:
        for domain in entity.domains:
            out.append((entity, domain))
    return out
