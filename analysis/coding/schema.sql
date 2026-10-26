-- Post-Collection Coding Pipeline schema (A1-B3 outputs). Applied
-- additively to the same database collector/storage/schema.sql initializes (see
-- analysis/coding/db.py::init_coding_schema) -- never redefines runs/
-- sources, and collector/ never reads this file (INSTRUMENT.md section 12's
-- one-way collector -> analysis rule stays intact).
--
-- Config inputs (config/inclusion_frame.yaml, config/alias_list.csv,
-- config/seed_vendor_list.csv, config/alt_domains_seed.csv,
-- config/figure_formulas.md) are human-maintained and stay as files --
-- this schema only covers pipeline-computed outputs. The LLM cache
-- (data/03_coded/llm_cache/*.json) also stays as-is; it is memoization, not
-- a result artifact.
--
-- Every statement is idempotent (CREATE ... IF NOT EXISTS), same
-- convention as collector/storage/schema.sql.

PRAGMA foreign_keys = ON;

-- A1: one row per consensus mention (2-of-3 extractor agreement).
CREATE TABLE IF NOT EXISTS mentions_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    surface_form    TEXT NOT NULL,
    -- A literal substring of runs.raw_response_text with inline citation
    -- markdown links stripped to their anchor text (analysis/coding/
    -- text_clean.py::strip_markdown_links), NOT necessarily of the raw
    -- column itself -- b1_per_run_company.py applies the same stripping
    -- before re-searching for this span.
    evidence_span   TEXT NOT NULL,
    is_recommended  INTEGER NOT NULL,
    -- JSON array of contributing extractor names, e.g. ["anthropic","openai"].
    votes           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mentions_raw_run_id ON mentions_raw(run_id);
CREATE INDEX IF NOT EXISTS idx_mentions_raw_surface_form ON mentions_raw(surface_form);

-- A2 (surface form collapse) is this view, not a write stage -- exact
-- surface_form grouping, never normalized (that's voting-only, A1's job).
-- run_ids semicolon-separated to match a3_entity_resolution.py's
-- `row["run_ids"].split(";")` (the same separator the old CSV used).
CREATE VIEW IF NOT EXISTS surface_forms AS
    SELECT surface_form, COUNT(*) AS mention_count, GROUP_CONCAT(run_id, ';') AS run_ids
    FROM mentions_raw
    GROUP BY surface_form;

-- A3 base columns + A4-appended classification columns on the same row
-- (meets_frame/in_seed_list/segment/evidence_url/frame_reason are NULL
-- until A4 runs).
CREATE TABLE IF NOT EXISTS entities (
    entity_id       TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    homepage_url    TEXT,
    -- 'in' | 'out' | 'borderline' | 'unresolved' (config/inclusion_frame.yaml's
    -- 4-way outcome). Only 'in' counts toward any denominator (B1).
    meets_frame     TEXT,
    in_seed_list    INTEGER,
    segment         TEXT,
    evidence_url    TEXT,
    -- an OUT reason code (not_a_company | not_a_vendor | end_user |
    -- unrelated | adjacent_product), or a one-line explanation when
    -- meets_frame = 'borderline'. NULL when meets_frame = 'in'.
    frame_reason    TEXT
);

CREATE TABLE IF NOT EXISTS entity_surface_forms (
    entity_id       TEXT NOT NULL REFERENCES entities(entity_id),
    surface_form    TEXT NOT NULL,
    PRIMARY KEY (entity_id, surface_form)
);

-- A4-prime / a3b: search-backed entity verification (analysis/coding/
-- a3b_entity_verification.py). Not in the frozen coding spec -- see the
-- ambiguity ledger. One row per entity; is_real/entity_type gate whether
-- A4 runs its frozen-frame check on that entity at all.
CREATE TABLE IF NOT EXISTS entity_verification (
    entity_id             TEXT PRIMARY KEY REFERENCES entities(entity_id),
    is_real               INTEGER,
    entity_type           TEXT,
    verified_homepage_url TEXT,
    disambiguation_note   TEXT,
    confidence            REAL
);

-- A5. settled_class/settled_publisher_is_in_category are a one-time human
-- (AI-assisted) review overlay added during the 2026-09-12 close-out --
-- never written by A5 itself, see analysis/coding/db.py's _ensure_column
-- retrofit for the already-populated database and
-- scripts/coding/close_out_a5_url_classification.py.
CREATE TABLE IF NOT EXISTS url_registry (
    url_normalized                       TEXT PRIMARY KEY,
    url_class                            TEXT,
    publisher_is_in_category             INTEGER,
    classifier_confidence                REAL,
    settled_class                        TEXT,
    settled_publisher_is_in_category     INTEGER
);

-- Task 5 (post-window auxiliary batch) cited-URL resolution snapshot,
-- loaded from data/02_aux/cited_urls/resolution.jsonl by
-- scripts/coding/run_coding_pipeline.py. Read-only audit copy: the pipeline swaps
-- each reachable citation for its final_url in memory before A3/A5/B1/B2 --
-- this table is here so that swap stays inspectable / joinable. Not
-- produced by an LLM stage; absent (empty) when the aux batch never ran.
CREATE TABLE IF NOT EXISTS cited_url_resolution (
    cited_url       TEXT PRIMARY KEY,
    final_url       TEXT,
    final_domain    TEXT,
    status_bucket   TEXT
);

-- Gap-fill frozen output the spec names but defines no producing stage for
-- (see analysis/coding/alt_domains.py) -- copies config/alt_domains_seed.csv
-- through, appendable later from domain_redirect_snapshot (collector/storage/schema.sql).
CREATE TABLE IF NOT EXISTS alt_domains (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       TEXT NOT NULL,
    alt_domain      TEXT NOT NULL,
    source_note     TEXT
);

-- B1: full cross product of runs x in-frame entities.
CREATE TABLE IF NOT EXISTS per_run_company (
    run_id                          TEXT NOT NULL REFERENCES runs(run_id),
    entity_id                       TEXT NOT NULL REFERENCES entities(entity_id),
    mentioned                       INTEGER NOT NULL,
    recommended                     INTEGER NOT NULL,
    own_domain_cited                INTEGER NOT NULL,
    linked                          INTEGER NOT NULL,
    mention_without_own_citation    INTEGER NOT NULL,
    first_mention_position          INTEGER,
    PRIMARY KEY (run_id, entity_id)
);

-- B2
CREATE TABLE IF NOT EXISTS per_run_rollups (
    run_id                  TEXT PRIMARY KEY REFERENCES runs(run_id),
    n_companies_mentioned   INTEGER NOT NULL,
    n_sources               INTEGER NOT NULL,
    -- NULL, not 0.0, when n_sources = 0 -- no share of nothing.
    vendor_source_share     REAL,
    n_out_of_frame_brands   INTEGER NOT NULL
);

-- B3: apply_eligibility_floor's qualifying set, per engine.
CREATE TABLE IF NOT EXISTS eligibility_qualifying (
    engine      TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    PRIMARY KEY (engine, entity_id)
);

-- B3: build_cell_rates, restricted to eligibility_qualifying entities.
CREATE TABLE IF NOT EXISTS cell_rates (
    entity_id             TEXT NOT NULL,
    intent_id             INTEGER NOT NULL,
    engine                TEXT NOT NULL,
    n_runs                INTEGER NOT NULL,
    mention_rate          REAL NOT NULL,
    recommendation_rate   REAL NOT NULL,
    PRIMARY KEY (entity_id, intent_id, engine)
);

-- B3b: derive_serp_binary's output (per-company Google top-10 presence).
CREATE TABLE IF NOT EXISTS serp_binary (
    entity_id          TEXT NOT NULL,
    intent_id          INTEGER NOT NULL,
    in_google_top10    INTEGER NOT NULL,
    PRIMARY KEY (entity_id, intent_id)
);

-- Parallel alias-regex validation track (run_alias_regex's output).
CREATE TABLE IF NOT EXISTS alias_hits (
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    canonical_name   TEXT NOT NULL,
    PRIMARY KEY (run_id, canonical_name)
);

-- Convenience views over the finished pipeline. Defined last: they read the
-- B-stage tables above, which CREATE VIEW resolves at definition time.
-- DROP-then-CREATE (still idempotent) so re-applying this file refreshes the
-- definition instead of silently keeping an older one.

-- The frozen "final list": entities the A4 frame call put IN the vertical
-- AND the A4-prime search check affirmatively confirmed as a real vendor
-- (is_real = 1, entity_type = 'vendor'). Everything the pipeline was NOT
-- sure about -- meets_frame 'borderline' / 'unresolved', or a verification
-- that only failed open -- is excluded here and surfaced in
-- entity_list_uncertain instead. verification_confidence_low flags the
-- handful of INs whose search check self-reported < 0.5 confidence (kept in
-- the list, just marked).
DROP VIEW IF EXISTS final_entity_list;
CREATE VIEW final_entity_list AS
SELECT
    e.entity_id,
    e.canonical_name,
    COALESCE(v.verified_homepage_url, e.homepage_url)              AS homepage_url,
    e.segment,
    v.entity_type,
    v.confidence                                                  AS verification_confidence,
    CASE WHEN COALESCE(v.confidence, 1) < 0.5 THEN 1 ELSE 0 END    AS verification_confidence_low,
    (SELECT COUNT(*) FROM entity_surface_forms sf WHERE sf.entity_id = e.entity_id)                    AS n_surface_forms,
    (SELECT COUNT(*) FROM per_run_company p WHERE p.entity_id = e.entity_id AND p.mentioned = 1)       AS n_mention_runs,
    (SELECT COUNT(*) FROM per_run_company p WHERE p.entity_id = e.entity_id AND p.recommended = 1)     AS n_recommend_runs,
    (SELECT COUNT(*) FROM cell_rates c WHERE c.entity_id = e.entity_id)                                AS n_cells,
    (SELECT ROUND(AVG(c.mention_rate), 4) FROM cell_rates c WHERE c.entity_id = e.entity_id)           AS avg_mention_rate,
    (SELECT ROUND(AVG(c.recommendation_rate), 4) FROM cell_rates c WHERE c.entity_id = e.entity_id)    AS avg_recommendation_rate,
    (SELECT COUNT(DISTINCT q.engine) FROM eligibility_qualifying q WHERE q.entity_id = e.entity_id)    AS n_eligible_engines
FROM entities e
JOIN entity_verification v ON v.entity_id = e.entity_id
WHERE e.meets_frame = 'in'
  AND v.is_real = 1
  AND v.entity_type = 'vendor'
ORDER BY n_mention_runs DESC, e.canonical_name;

-- The complement: everything held back from final_entity_list because the
-- pipeline was not sure -- borderline / unresolved frame calls, plus any IN
-- whose verification did not affirmatively confirm a vendor. frame_reason
-- carries the one-line "why" for borderline rows.
DROP VIEW IF EXISTS entity_list_uncertain;
CREATE VIEW entity_list_uncertain AS
SELECT
    e.entity_id,
    e.canonical_name,
    COALESCE(v.verified_homepage_url, e.homepage_url) AS homepage_url,
    e.meets_frame,
    e.frame_reason,
    v.is_real,
    v.entity_type,
    v.confidence AS verification_confidence,
    (SELECT COUNT(*) FROM per_run_company p WHERE p.entity_id = e.entity_id AND p.mentioned = 1) AS n_mention_runs
FROM entities e
LEFT JOIN entity_verification v ON v.entity_id = e.entity_id
WHERE e.meets_frame IN ('borderline', 'unresolved')
   OR (e.meets_frame = 'in' AND (v.is_real IS NOT 1 OR v.entity_type IS NOT 'vendor'))
ORDER BY e.meets_frame, e.canonical_name;
