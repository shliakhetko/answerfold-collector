"""PUBLIC side of the publish package: rebuild every figure from the shipped
tables and compare the result with the shipped frozen copy.

It reads only files that ship under publish/, with real company names,
slugs and domains, exactly as the study coded them. Two inputs differ from
the private originals, and this module is where the reader handles them:

* Google Gemini and Google AI Mode answers do not ship (Google's terms);
  their rows are stripped in place (analysis.publication.withheld). The
  reader restores a marker text and the per-run search count, so the shared
  extractors count exactly what they counted on the originals, and the
  companies named in each withheld text come from data/coded/name_matches.jsonl.
* The raw Google results pages (SerpApi copies) do not ship; A1 reads the
  shipped rank / url / domain table instead.

Layout it expects (relative to the package root, i.e. publish/):

    config/prompts/prompts.csv
    manifests/{phase_a,raw,aux,ui_test}_MANIFEST.json     copies of the private manifests
    data/coding/company_table.csv
    data/coding/company_alt_domains.csv
    data/coding/company_verification.csv
    data/coding/mentions_coded.jsonl
    data/coding/url_classes.csv
    data/raw/<date>.jsonl                                 window rows (Gemini / AI Mode stripped)
    data/aux/cited_urls/resolution.jsonl
    data/aux/google_serp_top10.csv
    data/aux/roster_provisional.csv
    data/aux/robots/<domain>/robots.txt
    data/ui/ui_runs.jsonl, data/ui/eyeball_check.json
    data/coded/name_matches.jsonl                         companies named in each withheld text
    figures/figures.json                                  the frozen copy
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from analysis.aggregate.load_runs import parse_run_record, sources_rows_for_record
from analysis.coding.calibration import _truthy
from analysis.coding.cited_url_resolution import load_resolution_map, resolve_sources_by_run
from analysis.coding.registrable_domain import psl_info, registrable_domain
from analysis.phase_b import domain_ownership, intent_map
from analysis.phase_b.citation_panel import build_citation_panel
from analysis.phase_b.crawler_precheck import crawler_precheck
from analysis.phase_b.daily_panel import build_daily_panel
from analysis.phase_b.figures import FIGURES
from analysis.phase_b.figures import ui_calibration as ui_calibration_figure
from analysis.phase_b.rates_panel import build_rates_panel
from analysis.phase_b.reading import FrozenInputs, _read_csv_rows, _read_jsonl
from analysis.phase_b.run_company_panel import build_run_company_panel
from analysis.phase_b.serialize import write_seam_file
from analysis.publication import withheld
from collector import paths

COMPANY_TABLE = paths.PUBLIC_COMPANY_TABLE
ALT_DOMAINS = paths.PUBLIC_ALT_DOMAINS
VERIFICATION = paths.PUBLIC_VERIFICATION
MENTIONS = paths.PUBLIC_MENTIONS
URL_CLASSES = paths.PUBLIC_URL_CLASSES
RAW_DIR = paths.PUBLIC_RAW_DIR
RESOLUTION = paths.PUBLIC_RESOLUTION
SERP_TABLE = paths.PUBLIC_SERP_TABLE
UI_RUNS = paths.PUBLIC_UI_RUNS
UI_EYEBALL = paths.PUBLIC_UI_EYEBALL
NAME_MATCHES = paths.PUBLIC_NAME_MATCHES
ROSTER = paths.PUBLIC_ROSTER
ROBOTS_DIR = paths.PUBLIC_ROBOTS_DIR
FIGURES_COPY = Path("figures/figures.json")
MANIFEST_FILES = {
    "phase_a": Path("manifests/phase_a_MANIFEST.json"),
    "raw": Path("manifests/raw_MANIFEST.json"),
    "aux": Path("manifests/aux_MANIFEST.json"),
    "ui_test": Path("manifests/ui_test_MANIFEST.json"),
}
SERP_COLUMNS = ("question_id", "rank", "url", "domain")


class BundleError(Exception):
    """The package is incomplete or inconsistent. Stops the build."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _need(root: Path, rel: Path) -> Path:
    path = Path(root) / rel
    if not path.exists():
        raise BundleError(f"missing {rel} under {root}")
    return path


def read_window_records(root: Path) -> list[dict[str, Any]]:
    """Every shipped window row, in file then line order, as written."""
    raw_files = sorted((Path(root) / RAW_DIR).glob("*.jsonl"))
    if not raw_files:
        raise BundleError(f"no window files under {Path(root) / RAW_DIR}")
    records: list[dict[str, Any]] = []
    for path in raw_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def read_serp_table(root: Path) -> list[dict[str, Any]]:
    """A1's Google top-10 rows from the shipped table, in the shape
    analysis.phase_b.serp_loader gives on the raw SerpApi files."""
    rows: list[dict[str, Any]] = []
    with _need(root, SERP_TABLE).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            rows.append({"intent_id": int(row["question_id"]), "position": int(row["rank"]), "url": row["url"]})
    return rows


def serp_table_rows(serp_raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The shipped table built from the loader's rows (question id = intent id
    of the head phrase, paraphrase 1; rank = organic position)."""
    return [
        {"question_id": r["intent_id"], "rank": r["position"], "url": r["url"], "domain": registrable_domain(r["url"]) or ""}
        for r in serp_raw_rows
    ]


def load_inputs(root: Path) -> FrozenInputs:
    root = Path(root)
    entities = _read_csv_rows(_need(root, COMPANY_TABLE))
    alt_domains_rows = _read_csv_rows(_need(root, ALT_DOMAINS))
    verification_rows = _read_csv_rows(_need(root, VERIFICATION))
    mentions_raw = _read_jsonl(_need(root, MENTIONS))
    url_registry_rows = _read_csv_rows(_need(root, URL_CLASSES))
    for row in url_registry_rows:
        row["publisher_is_in_category"] = _truthy(row.get("publisher_is_in_category"))
        row["settled_publisher_is_in_category"] = _truthy(row.get("settled_publisher_is_in_category"))

    manifests = {name: json.loads(_need(root, rel).read_text(encoding="utf-8")) for name, rel in MANIFEST_FILES.items()}
    instrument_commit = manifests["raw"].get("collector_version", "unknown")

    raw_runs: list[dict[str, Any]] = []
    sources_by_run: dict[str, list[dict[str, Any]]] = {}
    run_engine_by_id: dict[str, str] = {}
    for shipped in read_window_records(root):
        record = withheld.restore_for_parse(shipped)
        run = parse_run_record(record, instrument_commit=instrument_commit)
        raw_runs.append(run)
        sources_by_run[run["run_id"]] = sources_rows_for_record(record)
        run_engine_by_id[run["run_id"]] = run["engine"]

    resolution_map = load_resolution_map(_need(root, RESOLUTION))
    sources_by_run, resolution_stats = resolve_sources_by_run(sources_by_run, resolution_map, run_engine_by_id=run_engine_by_id)

    inputs_manifest = {
        "phase_a_manifest_sha256": sha256_file(root / MANIFEST_FILES["phase_a"]),
        "raw_manifest_sha256": sha256_file(root / MANIFEST_FILES["raw"]),
        "aux_manifest_sha256": sha256_file(root / MANIFEST_FILES["aux"]),
        "ui_manifest_sha256": sha256_file(root / MANIFEST_FILES["ui_test"]),
        "instrument_commit": instrument_commit,
        "public_suffix_list": psl_info(),
        "cited_url_resolution_rows_loaded": len(resolution_map),
        "cited_url_resolution": resolution_stats.to_dict(),
    }

    ui_test_runs = [withheld.restore_ui_run(r) for r in _read_jsonl(_need(root, UI_RUNS))]
    ui_eyeball = json.loads(_need(root, UI_EYEBALL).read_text(encoding="utf-8"))
    roster_rows, robots_by_domain = read_roster_and_robots(root)

    return FrozenInputs(
        entities=entities,
        alt_domains_rows=alt_domains_rows,
        entity_verification_rows=verification_rows,
        mentions_raw=mentions_raw,
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
        url_registry_rows=url_registry_rows,
        inputs_manifest=inputs_manifest,
        serp_raw_rows=read_serp_table(root),
        ui_test_runs=ui_test_runs,
        ui_eyeball=ui_eyeball,
        roster_rows=roster_rows,
        robots_by_domain=robots_by_domain,
    )


def read_roster_and_robots(root: Path) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    """Same result as the private reader: one entry per domain folder, the
    robots.txt text or None. A folder that had no robots.txt ships as an
    empty marker file, because empty folders do not survive a copy."""
    roster_path = Path(root) / ROSTER
    roster_rows = _read_csv_rows(roster_path) if roster_path.exists() else []
    robots_by_domain: dict[str, str | None] = {}
    robots_dir = Path(root) / ROBOTS_DIR
    if robots_dir.is_dir():
        for domain_dir in sorted(p for p in robots_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            robots_path = domain_dir / "robots.txt"
            robots_by_domain[domain_dir.name] = (
                robots_path.read_text(encoding="utf-8", errors="replace") if robots_path.exists() else None
            )
    return roster_rows, robots_by_domain


ROBOTS_ABSENT_MARKER = "robots.txt.absent"  # stands in for a domain folder with no robots.txt


def load_name_matches(root: Path) -> dict[str, list[str]]:
    """withheld text id (run id or app-check run key) -> company ids the
    study's name matcher found in the original text."""
    matches: dict[str, list[str]] = {}
    for row in _read_jsonl(_need(Path(root), NAME_MATCHES)):
        matches[row["text_id"]] = list(row["entity_ids"])
    return matches


@contextlib.contextmanager
def withheld_text_matcher(matches: dict[str, list[str]]) -> Iterator[None]:
    """UI_CALIBRATION counts company names in answer text with the study's
    deterministic name matcher. ChatGPT and Claude texts ship, so the matcher
    runs on them here. A withheld Gemini or AI Mode text is a marker; its
    companies come from the shipped table, computed privately on the original
    text with the same matcher. A marker with no row is a hard error, never a
    silent empty set."""
    original = ui_calibration_figure.match_entities

    def lookup(text: str, matcher: Any) -> set[str]:
        text_id = withheld.marker_id(text)
        if text_id is None:
            return original(text, matcher)
        if text_id not in matches:
            raise BundleError(f"withheld text {text_id} has no row in {NAME_MATCHES.as_posix()}")
        return set(matches[text_id])

    ui_calibration_figure.match_entities = lookup
    try:
        yield
    finally:
        ui_calibration_figure.match_entities = original


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_seam(root: Path, output_path: Path, *, log=print) -> Path:
    """Compute every registered figure and the shared-intermediate summaries
    from the shipped tables and write one seam file. Mirrors
    scripts/results/run_phase_b.py: same figure registry, same intermediates."""
    root = Path(root)
    inputs = load_inputs(root)
    log(f"read {len(inputs.raw_runs)} runs, {len(inputs.entities)} company-table rows, {len(inputs.mentions_raw)} coded mentions")

    _cells, daily_report = build_daily_panel(inputs)
    if daily_report.completeness.unwhitelisted_missing:
        raise BundleError(f"unexpected missing day/rep cells: {daily_report.completeness.unwhitelisted_missing}")
    resolved_intent_map = intent_map.read_intent_map(root)
    intent_map.assert_intent_map(resolved_intent_map)
    _rate_cells, rates_report = build_rates_panel(inputs)
    log(f"qualifying companies by engine: {rates_report.qualifying_by_engine}")
    _ownership_rows, ownership_report = domain_ownership.build_source_ownership(
        inputs.mentions_raw,
        inputs.entities,
        inputs.alt_domains_rows,
        inputs.sources_by_run,
        inputs.raw_runs,
        inputs.entity_verification_rows,
    )
    _citation_rows, citation_report = build_citation_panel(inputs)
    _rc_rows, _eligibility, run_company_report = build_run_company_panel(inputs)

    with withheld_text_matcher(load_name_matches(root)):
        figure_results = {figure_id: compute(inputs) for figure_id, compute in FIGURES.items()}

    write_seam_file(
        Path(output_path),
        inputs_manifest=inputs.inputs_manifest,
        figure_results=figure_results,
        intermediates={
            "intent_map": resolved_intent_map,
            "source_ownership": ownership_report.to_json(),
            "citation_panel": citation_report.to_json(),
            "run_company_panel": run_company_report.to_json(),
            "rates_panel": rates_report.to_json(),
            "daily_panel": daily_report.to_json(),
            "crawler_precheck": crawler_precheck(inputs.roster_rows, inputs.robots_by_domain),
        },
    )
    return Path(output_path)


def compare_to_shipped(rebuilt: Path, shipped: Path) -> tuple[bool, str]:
    """Byte-for-byte equality of the rebuilt seam and the shipped frozen
    copy, with a short pointer to the first difference if not."""
    a = Path(rebuilt).read_bytes()
    b = Path(shipped).read_bytes()
    if a == b:
        return True, f"identical, sha256 {hashlib.sha256(a).hexdigest()}"
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    line = a[:first].count(b"\n") + 1
    return False, f"MISMATCH at byte {first} (line {line}); rebuilt {len(a)} bytes, shipped {len(b)} bytes"


__all__ = [
    "BundleError", "build_seam", "compare_to_shipped", "load_inputs", "load_name_matches",
    "read_serp_table", "serp_table_rows", "withheld_text_matcher",
]
