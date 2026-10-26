#!/usr/bin/env python3
"""Phase A close (2026-09-12; alt_domains/entity_verification refreeze 2026-09-22).

Phase A -- every model call in the project -- is finished. This script
computes nothing and changes no value except re-reading the human-curated
config/alt_domains_seed.csv into the alt_domains table (build_alt_domains,
idempotent, no model call): otherwise it is a straight table dump of
Phase-A outputs that were never exported to a standalone file
(mentions_raw, entity_registry, alt_domains, entity_verification all
otherwise live only inside data/03_coded/coding_a1.sqlite3), matching the
naming/format precedent already set by
data/_trial/coded/{mentions_raw.jsonl,entity_registry.csv,alt_domains.csv}
for the unrelated trial vertical. url_registry was already frozen in the
prior session (data/03_coded/url_registry_frozen.csv) and
data/03_coded/coding_a1_report.json is the validation_report artifact.
entity_verification was added 2026-09-22 so Phase B can read
verified_homepage_url as a fallback (see analysis/phase_b/domain_ownership.py).

Then (re)writes data/03_coded/MANIFEST.json listing all 6 canonical
Phase-A artifacts plus the A5 spot-check evidence file, each with a
sha256 checksum, so the phase-boundary tag has something concrete to
point at.

Usage:
    uv run python -m scripts.coding.freeze_phase_a
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.coding import storage
from analysis.coding.alt_domains import build_alt_domains
from collector.storage.db import get_connection
from scripts.aux._run import rel

from collector import paths
DB_PATH = paths.CODING_DB
CODING_DIR = paths.CODED_DIR
MANIFEST_PATH = CODING_DIR / "MANIFEST.json"
PHASE_A_TAG = "phase-a-close-2026-09-12"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_entry(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    return {"path": rel(p), "sha256": _sha256_file(p), "bytes": p.stat().st_size, "lines": lines}


def export_mentions_raw(conn) -> Path:
    path = CODING_DIR / "mentions_raw.jsonl"
    rows = storage.list_mentions_raw(conn=conn)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return path


ENTITY_REGISTRY_COLUMNS = [
    "entity_id", "canonical_name", "homepage_url", "meets_frame",
    "in_seed_list", "segment", "evidence_url", "frame_reason", "surface_forms",
]


def export_entity_registry(conn) -> Path:
    path = CODING_DIR / "entity_registry.csv"
    rows = sorted(storage.list_entities(conn=conn), key=lambda r: r["entity_id"])
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# entity_registry -- Phase A frozen artifact, scripts/coding/freeze_phase_a.py\n")
        writer = csv.DictWriter(f, fieldnames=ENTITY_REGISTRY_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: ("" if r.get(col) is None else r.get(col)) for col in ENTITY_REGISTRY_COLUMNS})
    return path


ALT_DOMAINS_COLUMNS = ["entity_id", "alt_domain", "source_note"]


def export_alt_domains(conn) -> Path:
    path = CODING_DIR / "alt_domains.csv"
    rows = sorted(storage.list_alt_domains(conn=conn), key=lambda r: (r["entity_id"], r["alt_domain"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# alt_domains -- Phase A frozen artifact, scripts/coding/freeze_phase_a.py\n")
        writer = csv.DictWriter(f, fieldnames=ALT_DOMAINS_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: ("" if r.get(col) is None else r.get(col)) for col in ALT_DOMAINS_COLUMNS})
    return path


ENTITY_VERIFICATION_COLUMNS = [
    "entity_id", "is_real", "entity_type", "verified_homepage_url", "disambiguation_note", "confidence",
]


def export_entity_verification(conn) -> Path:
    """A4-prime's search-backed verification (analysis/coding/a3b_entity_verification.py)
    -- added to the frozen set 2026-09-22 so Phase B can read
    verified_homepage_url as a fallback where entity_registry.homepage_url
    is blank (141/415 in-frame entities), instead of only ever seeing the
    141 as "no known domain". Was previously live only in
    data/03_coded/coding_a1.sqlite3, already consumed the same way by
    analysis/aux/roster.py."""
    path = CODING_DIR / "entity_verification.csv"
    rows = sorted(storage.list_entity_verification(conn=conn), key=lambda r: r["entity_id"])
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# entity_verification -- Phase A frozen artifact, scripts/coding/freeze_phase_a.py\n")
        writer = csv.DictWriter(f, fieldnames=ENTITY_VERIFICATION_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: ("" if r.get(col) is None else r.get(col)) for col in ENTITY_VERIFICATION_COLUMNS})
    return path


def main() -> None:
    CODING_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(DB_PATH)
    try:
        build_alt_domains(conn=conn)  # re-read config/alt_domains_seed.csv into the alt_domains table
        mentions_raw_path = export_mentions_raw(conn)
        entity_registry_path = export_entity_registry(conn)
        alt_domains_path = export_alt_domains(conn)
        entity_verification_path = export_entity_verification(conn)
    finally:
        conn.close()

    frozen_csv = CODING_DIR / "url_registry_frozen.csv"
    report_json = paths.CODING_REPORT_JSON
    spot_check_csv = CODING_DIR / "url_classification_spot_check.csv"

    artifacts = [
        _manifest_entry(mentions_raw_path),
        _manifest_entry(entity_registry_path),
        _manifest_entry(alt_domains_path),
        _manifest_entry(entity_verification_path),
        _manifest_entry(frozen_csv),
        _manifest_entry(report_json),
        _manifest_entry(spot_check_csv),
    ]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/coding/freeze_phase_a.py",
        "phase": "A",
        "tag": PHASE_A_TAG,
        "artifacts": artifacts,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    for entry in artifacts:
        print(f"{entry['path']}  sha256={entry['sha256']}  bytes={entry['bytes']}  lines={entry['lines']}")
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
