"""Every directory and well-known file of the repo layout, in one place.

Code never spells a `data/...`, `runs/...` or `config/...` path itself: it
imports the constant from here, so a layout change is an edit of this file.

Three families:

* repo layout (absolute `Path`s under REPO_ROOT) -- what the code reads and
  writes in a working checkout;
* `rel(path)` / `data_rel(path)` -- the same paths relative to the repo root /
  to the data directory (manifests record repo-relative paths; a frozen "data
  root" mirrors the data directory);
* PUBLIC_* -- the layout of the public deposit under publish/. It is a
  separate product with its own fixed layout and does not follow the repo
  layout.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- config / inputs ------------------------------------------------------
CONFIG_DIR = REPO_ROOT / "config"
EXPERIMENT_YAML = CONFIG_DIR / "experiment.yaml"
EXPERIMENT_PILOT_YAML = CONFIG_DIR / "experiment.pilot.yaml"
EXPERIMENT_RECHECK_YAML = CONFIG_DIR / "experiment.recheck.yaml"
EXPERIMENT_EXAMPLE_YAML = CONFIG_DIR / "experiment.example.yaml"
PROMPTS_DIR = CONFIG_DIR / "prompts"
PROMPTS_CSV = PROMPTS_DIR / "prompts.csv"
INCLUSION_FRAME_YAML = CONFIG_DIR / "inclusion_frame.yaml"
SEED_VENDOR_LIST_CSV = CONFIG_DIR / "seed_vendor_list.csv"
ALIAS_LIST_CSV = CONFIG_DIR / "alias_list.csv"
ALT_DOMAINS_SEED_CSV = CONFIG_DIR / "alt_domains_seed.csv"
FIGURE_FORMULAS_MD = CONFIG_DIR / "figure_formulas.md"
PAPER_DISPLAY_JSON = CONFIG_DIR / "paper_display.json"
RESULTS_DRAFT_DEV_TEMPLATE = CONFIG_DIR / "results_draft_dev.md"
RESULTS_DRAFT_USER_TEMPLATE = CONFIG_DIR / "results_draft_user.md"  # the user view: S1, A1, V1 business blocks
USER_CHARTS_MD = CONFIG_DIR / "user_charts.md"  # all text and labels of the six user charts
RESULTS_PUBLIC_TEMPLATE = CONFIG_DIR / "results_public.md"  # the public page (about 1,500 words, seven charts)
PUBLIC_CHARTS_MD = CONFIG_DIR / "public_charts.md"  # overlay of user_charts.md: short engine names, pub_ chart text
SOURCE_DOMAIN_CLASSES_CSV = CONFIG_DIR / "source_domain_classes.csv"
PUBLISH_DOCS_TEMPLATE_DIR = CONFIG_DIR / "publish_docs"
SCHEMA_SQL = REPO_ROOT / "collector" / "storage" / "schema.sql"

# --- reports (human-readable ledgers and audits) ---------------------------
REPORTS_DIR = REPO_ROOT / "reports"
AMBIGUITY_LEDGER_MD = REPORTS_DIR / "ambiguity_ledger.md"
COLLECTION_AUDIT_MD = REPORTS_DIR / "collection_audit.md"
AUX_COLLECTION_SUMMARY_MD = REPORTS_DIR / "aux_collection_summary.md"
HOMEPAGE_FALLBACK_NAME_MISMATCH_MD = REPORTS_DIR / "homepage_fallback_name_mismatch.md"
HOMEPAGE_FALLBACK_SPOT_CHECK_MD = REPORTS_DIR / "homepage_fallback_spot_check.md"

# --- public build ----------------------------------------------------------
PUBLISH_DIR = REPO_ROOT / "publish"

# --- data ------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"

# 01_raw: collector output
RAW_DIR = DATA_DIR / "01_raw"
RUNS_DIR = RAW_DIR / "runs"
COLLECTOR_DB = RAW_DIR / "collector.sqlite3"
RUNS_EXPORT_CSV = RAW_DIR / "runs_export.csv"
RAW_MANIFEST = RAW_DIR / "MANIFEST.json"
RAW_FROZEN_DIR = RAW_DIR / "raw_frozen"
UI_TEST_DIR = RAW_DIR / "ui_test"
UI_MANIFEST = UI_TEST_DIR / "MANIFEST.json"
UI_EYEBALL_FILE = UI_TEST_DIR / "eyeball_check.json"

# 02_aux: auxiliary collection
AUX_DIR = DATA_DIR / "02_aux"
AUX_MANIFEST = AUX_DIR / "MANIFEST.json"
AUX_RESOLUTION_JSONL = AUX_DIR / "cited_urls" / "resolution.jsonl"
AUX_ROSTER_CSV = AUX_DIR / "roster_provisional.csv"
AUX_ROBOTS_DIR = AUX_DIR / "robots"
AUX_GOOGLE_SERP_DIR = AUX_DIR / "google_serp"
AUX_DIRECTORIES_INDEX = AUX_DIR / "directories" / "index.jsonl"

# 03_coded: coding pipeline
CODED_DIR = DATA_DIR / "03_coded"
PHASE_A_MANIFEST = CODED_DIR / "MANIFEST.json"
LLM_CACHE_DIR = CODED_DIR / "llm_cache"
CODING_DB = CODED_DIR / "coding_a1.sqlite3"
CODING_REPORT_JSON = CODED_DIR / "coding_a1_report.json"
VALIDATION_REPORT_MD = CODED_DIR / "validation_report.md"

# 04_results: Phase B (figures, charts, rendered drafts)
RESULTS_DIR = DATA_DIR / "04_results"
FIGURES_JSON = RESULTS_DIR / "figures.json"
BUSINESS_FIGURES_JSON = RESULTS_DIR / "business_figures.json"
BUSINESS_FIGURES_A1_V1_JSON = RESULTS_DIR / "business_figures_a1_v1.json"
BUSINESS_FIGURES_REST_JSON = RESULTS_DIR / "business_figures_rest.json"
BUSINESS_FIGURES_PUBLIC_JSON = RESULTS_DIR / "business_figures_public.json"
USER_CHARTS_DIR = RESULTS_DIR / "user_charts"

# _trial: disposable trial batch (its config lives in config/trial/)
TRIAL_DIR = DATA_DIR / "_trial"
TRIAL_CONFIG_DIR = CONFIG_DIR / "trial"
TRIAL_RUNS_DIR = TRIAL_DIR / "runs"
TRIAL_CODED_DIR = TRIAL_DIR / "coded"
TRIAL_CODING_DB = TRIAL_DIR / "coding_trial.sqlite3"

# _review: hand-review exports (unredacted)
REVIEW_DIR = DATA_DIR / "_review"
ENTITY_REVIEW_CSV = REVIEW_DIR / "final_entity_list_review.csv"


# --- relative views ----------------------------------------------------------
def rel(path: Path | str) -> Path:
    """`path` relative to the repo root (the form manifests record)."""
    return Path(path).relative_to(REPO_ROOT)


def data_rel(path: Path | str) -> Path:
    """`path` relative to the data directory (the form inside a frozen data root)."""
    return Path(path).relative_to(DATA_DIR)


def manifest_path(path_in_data: Path | str) -> str:
    """The repo-relative posix string a manifest records for a file given relative to the data dir."""
    return (rel(DATA_DIR) / path_in_data).as_posix()


# --- public deposit layout (publish/), relative to the deposit root -----------
PUBLIC_DATA = Path("data")
PUBLIC_CODING = PUBLIC_DATA / "coding"
PUBLIC_COMPANY_TABLE = PUBLIC_CODING / "company_table.csv"
PUBLIC_ALT_DOMAINS = PUBLIC_CODING / "company_alt_domains.csv"
PUBLIC_VERIFICATION = PUBLIC_CODING / "company_verification.csv"
PUBLIC_MENTIONS = PUBLIC_CODING / "mentions_coded.jsonl"
PUBLIC_URL_CLASSES = PUBLIC_CODING / "url_classes.csv"
PUBLIC_RAW_DIR = PUBLIC_DATA / "raw"
PUBLIC_AUX_DIR = PUBLIC_DATA / "aux"
PUBLIC_RESOLUTION = PUBLIC_AUX_DIR / "cited_urls" / "resolution.jsonl"
PUBLIC_ROSTER = PUBLIC_AUX_DIR / "roster_provisional.csv"
PUBLIC_ROBOTS_DIR = PUBLIC_AUX_DIR / "robots"
PUBLIC_SERP_TABLE = PUBLIC_AUX_DIR / "google_serp_top10.csv"
PUBLIC_UI_RUNS = PUBLIC_DATA / "ui" / "ui_runs.jsonl"
PUBLIC_UI_EYEBALL = PUBLIC_DATA / "ui" / "eyeball_check.json"
PUBLIC_NAME_MATCHES = PUBLIC_DATA / "coded" / "name_matches.jsonl"
# The package mirrors the repo layout for the two config files it ships, so the
# collection script and the figure code find them at their usual place.
PUBLIC_PROMPTS_CSV = Path("config") / "prompts" / "prompts.csv"
PUBLIC_EXPERIMENT_EXAMPLE_YAML = Path("config") / "experiment.example.yaml"
