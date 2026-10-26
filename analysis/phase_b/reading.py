"""Phase B reading layer.

Loads frozen Phase-A artifacts (checksum-verified against
data/03_coded/MANIFEST.json) and raw collection rows (checksum-verified
against data/01_raw/MANIFEST.json, parsed through the same engine-specific
extraction the original sqlite load used). Has no opinion about the
study: it hands back data, or it stops loudly. A partial read never
propagates -- any missing file or checksum mismatch raises
PhaseBInputError before anything is returned.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis.aggregate.load_runs import parse_run_record, sources_rows_for_record
from analysis.coding.calibration import _truthy
from analysis.coding.cited_url_resolution import load_resolution_map, resolve_sources_by_run
from analysis.coding.registrable_domain import psl_info
from analysis.phase_b.serp_loader import read_head_phrase_serp_rows
from analysis.phase_b.ui_calibration import read_eyeball, read_ui_runs
from collector import paths

PHASE_A_MANIFEST = paths.rel(paths.PHASE_A_MANIFEST)
RAW_MANIFEST = paths.rel(paths.RAW_MANIFEST)
# 2026-09-29 freeze: the aux files Phase B reads and the 30 UI-calibration
# captures are hash-verified too (ledger: freeze-data-root-and-manifest-
# scope-2026-09-29). Manifests live in the repo; the files live under the
# data root.
AUX_MANIFEST = paths.rel(paths.AUX_MANIFEST)
UI_MANIFEST = paths.rel(paths.UI_MANIFEST)
DATA_ROOT_ENV = "GEO_DATA_ROOT"
ROSTER_PATH = paths.rel(paths.AUX_ROSTER_CSV)
ROBOTS_DIR = paths.rel(paths.AUX_ROBOTS_DIR)
# Post-window aux snapshot (Task 5, collected 2026-09-06/07), not a Phase-A
# frozen/checksummed artifact -- read directly if present, same as
# scripts/coding/run_coding_pipeline.py does. Missing file -> {} -> no swap, never
# an error (matches load_resolution_map's own missing-path contract).
CITED_URL_RESOLUTION_PATH = paths.rel(paths.AUX_RESOLUTION_JSONL)


class PhaseBInputError(Exception):
    """A frozen input is missing, unreadable, or fails its recorded
    checksum. Raised immediately -- never caught internally to patch
    together a partial result."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """The frozen-data root: an explicit value (CLI flag) or the
    GEO_DATA_ROOT environment variable. NO default -- unset or a missing
    directory stops the build loudly, so a fresh clone can never quietly
    run against whatever happens to sit next to it."""
    value = explicit if explicit else os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise PhaseBInputError(
            f"no data path: set {DATA_ROOT_ENV} (or pass --data-root) to the directory holding the frozen "
            "artifacts (laid out like the repo's data/ directory). There is no default."
        )
    root = Path(value)
    if not root.is_dir():
        raise PhaseBInputError(f"data path {root} does not exist or is not a directory")
    return root


def resolve_data_path(data_root: Path, repo_relative: str | Path) -> Path:
    """Manifest paths are repo-relative. 'data/<rest>' lives at
    <data_root>/<rest>; 'data/01_raw/runs/<file>' (the raw window) lives at
    <data_root>/raw_frozen/<file>, the frozen copy whose sha256s equal
    the data/01_raw/runs/ originals."""
    rel = Path(repo_relative)
    parts = rel.parts
    runs_rel = paths.rel(paths.RUNS_DIR)
    if rel.parts[: len(runs_rel.parts)] == runs_rel.parts:
        return Path(data_root) / paths.data_rel(paths.RAW_FROZEN_DIR) / Path(*parts[len(runs_rel.parts):])
    data_rel_root = paths.rel(paths.DATA_DIR)
    if rel.parts[: len(data_rel_root.parts)] == data_rel_root.parts:
        return Path(data_root).joinpath(*parts[len(data_rel_root.parts):])
    raise PhaseBInputError(f"cannot map manifest path {repo_relative!r} under the data root")


def _verify(data_root: Path, manifest_path: Path, entries: list[dict[str, Any]], manifest_label: str) -> None:
    for entry in entries:
        path = resolve_data_path(data_root, entry["path"])
        if not path.exists():
            raise PhaseBInputError(f"{manifest_label}: {entry['path']} listed in {manifest_path} but missing on disk")
        actual = _sha256_file(path)
        if actual != entry["sha256"]:
            raise PhaseBInputError(
                f"{manifest_label}: {entry['path']} checksum mismatch -- "
                f"expected {entry['sha256']}, got {actual} (recorded in {manifest_path})"
            )


def _load_manifest(repo_root: Path, manifest_path: Path, label: str) -> dict[str, Any]:
    full_path = repo_root / manifest_path
    if not full_path.exists():
        raise PhaseBInputError(f"{label} manifest not found at {manifest_path}")
    with full_path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Skips a leading '#'-comment header line (the frozen-artifact
    convention set by scripts/coding/freeze_phase_a.py) and maps '' back to
    None, the inverse of that script's None -> '' write."""
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        return [{k: (None if v == "" else v) for k, v in row.items()} for row in reader]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class FrozenInputs:
    entities: list[dict[str, Any]]
    alt_domains_rows: list[dict[str, Any]]
    entity_verification_rows: list[dict[str, Any]]
    mentions_raw: list[dict[str, Any]]
    raw_runs: list[dict[str, Any]]  # parsed via parse_run_record -- same shape as the runs table
    # url_raw/domain already resolved through the Sep 6/7 aux redirect
    # capture where a citation was reachable (Gemini's vertexaisearch
    # wrappers unwrapped to their real publisher, etc.), plus (2026-09-23)
    # the Gemini title fallback for whatever the aux capture still left
    # wrapped -- every consumer (S3, domain_ownership, citation_panel)
    # sees the same resolved sources, resolved once here rather than each
    # re-implementing the swap.
    sources_by_run: dict[str, list[dict[str, Any]]]
    # A5's url_class/publisher_is_in_category classification, one row per
    # unique url_normalized, frozen 2026-09-12 -- already a checksummed
    # Phase-A artifact (data/03_coded/MANIFEST.json), just not previously
    # read by this dataclass. publisher_is_in_category/
    # settled_publisher_is_in_category are coerced to real bool|None here
    # (the raw CSV stores "1"/"0" strings, which _read_csv_rows does not
    # itself coerce -- reading them as strings downstream would make every
    # citation silently read as publisher_is_in_category=True, since a
    # non-empty string is truthy).
    url_registry_rows: list[dict[str, Any]] = field(default_factory=list)
    inputs_manifest: dict[str, Any] = field(default_factory=dict)
    # UI calibration (30 manual web-UI captures, answer text already cut
    # out by the fixed container selector) and the by-eye check records.
    ui_test_runs: list[dict[str, Any]] = field(default_factory=list)
    ui_eyeball: dict[str, Any] = field(default_factory=dict)
    # Crawler pre-check inputs: roster rows and each domain's robots.txt
    # text (None when the file is absent).
    roster_rows: list[dict[str, Any]] = field(default_factory=list)
    robots_by_domain: dict[str, str | None] = field(default_factory=dict)
    # A1's raw Google top-10 SERP rows (analysis/phase_b/serp_loader.py) --
    # a loose post-window aux read, not a checksummed Phase-A artifact.
    # Empty list (never an error) when data/02_aux/google_serp/ is absent, same
    # missing-file contract as CITED_URL_RESOLUTION_PATH above.
    serp_raw_rows: list[dict[str, Any]] = field(default_factory=list)


def _read_roster_and_robots(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    roster_path = resolve_data_path(data_root, ROSTER_PATH)
    roster_rows = _read_csv_rows(roster_path) if roster_path.exists() else []
    robots_by_domain: dict[str, str | None] = {}
    robots_dir = resolve_data_path(data_root, ROBOTS_DIR)
    if robots_dir.is_dir():
        for domain_dir in sorted(p for p in robots_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            robots_path = domain_dir / "robots.txt"
            robots_by_domain[domain_dir.name] = (
                robots_path.read_text(encoding="utf-8", errors="replace") if robots_path.exists() else None
            )
    return roster_rows, robots_by_domain


def read_frozen_inputs(repo_root: Path, data_root: Path) -> FrozenInputs:
    """repo_root holds the tracked manifests; data_root holds the frozen
    data files (required -- see resolve_data_root). Every manifest (Phase
    A, raw window, aux, ui_test) is verified BEFORE any file is parsed."""
    repo_root = Path(repo_root)
    data_root = Path(data_root)

    phase_a_manifest = _load_manifest(repo_root, PHASE_A_MANIFEST, "Phase A")
    raw_manifest = _load_manifest(repo_root, RAW_MANIFEST, "raw window")
    aux_manifest = _load_manifest(repo_root, AUX_MANIFEST, "aux")
    ui_manifest = _load_manifest(repo_root, UI_MANIFEST, "ui_test")
    _verify(data_root, PHASE_A_MANIFEST, phase_a_manifest["artifacts"], "Phase A")
    _verify(data_root, RAW_MANIFEST, raw_manifest["raw_files"], "raw window")
    _verify(data_root, AUX_MANIFEST, aux_manifest["files"], "aux")
    _verify(data_root, UI_MANIFEST, ui_manifest["files"], "ui_test")

    artifact_paths = {Path(a["path"]).name: resolve_data_path(data_root, a["path"]) for a in phase_a_manifest["artifacts"]}
    entities = _read_csv_rows(artifact_paths["entity_registry.csv"])
    alt_domains_rows = _read_csv_rows(artifact_paths["alt_domains.csv"])
    entity_verification_rows = _read_csv_rows(artifact_paths["entity_verification.csv"])
    mentions_raw = _read_jsonl(artifact_paths["mentions_raw.jsonl"])
    url_registry_rows = _read_csv_rows(artifact_paths["url_registry_frozen.csv"])
    for row in url_registry_rows:
        row["publisher_is_in_category"] = _truthy(row.get("publisher_is_in_category"))
        row["settled_publisher_is_in_category"] = _truthy(row.get("settled_publisher_is_in_category"))

    instrument_commit = raw_manifest.get("collector_version", "unknown")
    raw_runs: list[dict[str, Any]] = []
    sources_by_run: dict[str, list[dict[str, Any]]] = {}
    run_engine_by_id: dict[str, str] = {}
    for raw_file in raw_manifest["raw_files"]:
        path = resolve_data_path(data_root, raw_file["path"])
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                run = parse_run_record(record, instrument_commit=instrument_commit)
                raw_runs.append(run)
                sources_by_run[run["run_id"]] = sources_rows_for_record(record)
                run_engine_by_id[run["run_id"]] = run["engine"]

    resolution_map = load_resolution_map(resolve_data_path(data_root, CITED_URL_RESOLUTION_PATH))
    sources_by_run, resolution_stats = resolve_sources_by_run(
        sources_by_run, resolution_map, run_engine_by_id=run_engine_by_id
    )

    inputs_manifest = {
        "phase_a_manifest_sha256": _sha256_file(repo_root / PHASE_A_MANIFEST),
        "raw_manifest_sha256": _sha256_file(repo_root / RAW_MANIFEST),
        "aux_manifest_sha256": _sha256_file(repo_root / AUX_MANIFEST),
        "ui_manifest_sha256": _sha256_file(repo_root / UI_MANIFEST),
        "instrument_commit": instrument_commit,
        "public_suffix_list": psl_info(),
        "cited_url_resolution_rows_loaded": len(resolution_map),
        "cited_url_resolution": resolution_stats.to_dict(),
    }

    serp_raw_rows = read_head_phrase_serp_rows(data_root)
    ui_test_runs = read_ui_runs(data_root)
    ui_eyeball = read_eyeball(data_root)
    roster_rows, robots_by_domain = _read_roster_and_robots(data_root)

    return FrozenInputs(
        entities=entities,
        alt_domains_rows=alt_domains_rows,
        entity_verification_rows=entity_verification_rows,
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
        url_registry_rows=url_registry_rows,
        inputs_manifest=inputs_manifest,
        serp_raw_rows=serp_raw_rows,
        ui_test_runs=ui_test_runs,
        ui_eyeball=ui_eyeball,
        roster_rows=roster_rows,
        robots_by_domain=robots_by_domain,
    )
