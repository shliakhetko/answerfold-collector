-- geo-collector schema
-- Applied once at initialization (INSTRUMENT.md section 13): a single definition,
-- not a migration sequence. Every statement is idempotent so init_db() can run
-- at the start of every process without risk to existing data.
--
-- Deliberately absent: any table computed after collection (url_class, per-run x
-- company mention/recommendation flags, rollups). Those belong to analysis/,
-- which may read this database, but this schema never anticipates them
-- (INSTRUMENT.md sections 3, 11, 12 -- the one-way collector -> analysis rule).

PRAGMA foreign_keys = ON;

-- 8.1 Runs: one row per model call, plus manual UI-calibration rows (is_manual=1)
-- that reuse whichever columns apply instead of a parallel table, since a manual
-- row shares almost every field with an API row.
CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT PRIMARY KEY,
    -- Which instrument build produced this row (section 2). Required on every row,
    -- including manual ones, so an old experiment stays traceable.
    instrument_commit       TEXT NOT NULL,
    -- Full ISO 8601 with timezone offset (section 8.1). Text, not epoch, so it
    -- stays human-legible and the offset is preserved exactly as recorded.
    timestamp               TEXT NOT NULL,
    -- No CHECK/enum: which engines exist is an experiment decision (section 6),
    -- not a schema fact.
    engine                  TEXT NOT NULL,
    model_requested         TEXT,
    model_returned          TEXT,
    -- intent_id / paraphrase_id / rep_index / day_index: plain integers, no CHECK
    -- range. DB.md's 1-7 / 1-5 / 1-3 / 1-7 are one experiment's declared counts,
    -- not schema-level constraints (section 4: "the instrument does not choose
    -- an experiment's counts").
    intent_id               INTEGER NOT NULL,
    paraphrase_id           INTEGER NOT NULL,
    -- Denormalized deliberately (section 8.1): a historical row must never depend
    -- on a prompt table that could later be edited.
    prompt_text             TEXT NOT NULL,
    rep_index               INTEGER NOT NULL,
    day_index               INTEGER NOT NULL,
    raw_response_text       TEXT,
    -- Nullable tri-state (0/1/NULL), not a plain boolean default: an adapter that
    -- cannot determine this must say so rather than imply "no" (section 7.3).
    search_fired            INTEGER,
    -- Stored exactly as the vendor returns it (section 6) -- opaque text, no
    -- assumed shape.
    fan_out_queries         TEXT,
    latency_ms              INTEGER,
    finish_reason           TEXT,
    -- NULL on success. A failed call is not an observation of absence (section
    -- 7.1) and must never be conflated with a genuine empty answer.
    error                   TEXT,
    -- Token counts parsed out of the vendor's raw usage object (see
    -- raw_response_envelope below for that source), one column per count so
    -- they're queryable without a JSON extract. Nullable: google_ai_mode has
    -- no token concept at all (docs/usage_paths.md), and a parse failure
    -- should leave these NULL rather than block the row. No cost/price
    -- column here -- no vendor returns a dollar figure (docs/usage_paths.md),
    -- so a USD cost is a derived quantity computed from these counts by
    -- analysis/aggregate/pricing.py, not a raw fact collection ever observes
    -- (INSTRUMENT.md sections 3, 11).
    tokens_input            INTEGER,
    tokens_output           INTEGER,
    tokens_total            INTEGER,
    -- reasoning_tokens/tool_tokens: a shared concept across vendors (each
    -- reports *something* here under its own name -- OpenAI's
    -- output_tokens_details.reasoning_tokens, Gemini's thoughts_token_count/
    -- tool_use_prompt_token_count, Anthropic's output_tokens_details.
    -- thinking_tokens), so one column each rather than per-vendor duplicates.
    -- NULL, not 0, when a given vendor doesn't report it for a row.
    reasoning_tokens        INTEGER,
    tool_tokens             INTEGER,
    -- The remaining fields are vendor-native, not cross-vendor concepts --
    -- e.g. OpenAI's cache_write_tokens and Anthropic's cache_creation_tokens
    -- are different mechanisms with different billing, so they get their own
    -- columns rather than being forced into a shared name (INSTRUMENT.md
    -- section 6: no cross-vendor normalization). Each is populated by
    -- exactly one engine and NULL for every other row.
    cached_tokens           INTEGER, -- openai_chatgpt
    cache_write_tokens      INTEGER, -- openai_chatgpt
    cached_content_tokens   INTEGER, -- google_gemini
    cache_creation_tokens   INTEGER, -- anthropic_claude
    cache_read_tokens       INTEGER, -- anthropic_claude
    web_search_requests     INTEGER, -- anthropic_claude
    web_fetch_requests      INTEGER, -- anthropic_claude
    -- Populated only for no-API cells (section 6).
    scraper_vendor          TEXT,
    scraper_endpoint        TEXT,
    -- The full unparsed response. Every other column here is a parse of it; if a
    -- parse is wrong, this is what makes it fixable without a rerun (section
    -- 8.1, applying the section 3 rule one level up). The only JSON column on
    -- this table -- everything else is flattened into its own typed column.
    raw_response_envelope   TEXT,
    -- Section 9's manual UI-calibration procedure reuses this table instead of a
    -- parallel schema, since a manual row differs from an API row only by having
    -- a screenshot and lacking fields like latency_ms/fan_out_queries.
    is_manual               INTEGER NOT NULL DEFAULT 0,
    screenshot_path         TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_engine_day ON runs(engine, day_index);
CREATE INDEX IF NOT EXISTS idx_runs_intent_paraphrase ON runs(intent_id, paraphrase_id);

-- 8.2 Sources: one row per cited source per run.
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    -- Order of appearance in the answer. Captured at write time because it is
    -- unrecoverable once sources are stored as a set (section 8.2).
    position    INTEGER NOT NULL,
    -- Exactly as returned. No url_normalized column: normalization is a
    -- post-collection concern (section 8.2) -- doing it live risks an
    -- unrecoverable overwrite if the normalization logic is wrong.
    url_raw     TEXT NOT NULL,
    domain      TEXT,
    title       TEXT,
    -- Same rationale as runs.extra, smaller scale: room for an occasional
    -- vendor-specific per-source field (e.g. a relevance score) without a
    -- schema change.
    extra       TEXT,
    UNIQUE (run_id, position)
);

CREATE INDEX IF NOT EXISTS idx_sources_run_id ON sources(run_id);
CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain);

-- 8.3 Population: not implemented for now -- removed along with
-- collector/runner/load_population.py and its storage.api functions
-- (INSTRUMENT.md section 8.3 still describes the intended design; nothing
-- here contradicts it, this table just doesn't exist until it's rebuilt).

-- 8.5 Once-per-window / daily items.

-- Resolved daily, not once (section 8.4). Append-only: a redirect state that
-- existed on a given day cannot be reconstructed later, so each day's job
-- inserts a new row rather than updating a prior one (section 10: "nothing is
-- edited in place").
CREATE TABLE IF NOT EXISTS domain_redirect_snapshot (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_domain    TEXT NOT NULL,
    -- Optional convenience join key matching runs.day_index. Nullable because
    -- checked_at is the real source of truth (a snapshot job might run outside
    -- a day-indexed experiment context, e.g. a dry run).
    day_index           INTEGER,
    checked_at          TEXT NOT NULL,
    resolved_url        TEXT,
    status_code         INTEGER,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_redirect_domain ON domain_redirect_snapshot(canonical_domain);

-- Third-party ranking snapshot, generalized beyond "google_top10": one
-- observation of an external ranking system's state for one entity/intent pair,
-- checked once during the window (section 8.5). ranking_system and metric are
-- free text, not an enum, since which systems/metrics matter (top-10 presence,
-- numeric rank, etc.) is an experiment decision, not a schema fact (section 1:
-- vertical- and engine-agnostic).
CREATE TABLE IF NOT EXISTS ranking_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- No REFERENCES population(entity_id): the population table doesn't
    -- exist right now (see the note above 8.5). Plain identifier until
    -- population is rebuilt.
    entity_id       TEXT NOT NULL,
    intent_id       INTEGER NOT NULL,
    ranking_system  TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           TEXT NOT NULL,
    checked_at      TEXT NOT NULL,
    raw             TEXT
);

CREATE INDEX IF NOT EXISTS idx_ranking_entity_intent ON ranking_snapshot(entity_id, intent_id);

-- Crawler access policy per domain (section 8.5). blocked is a nullable
-- tri-state: NULL = unverified, 0 = checked/not blocked, 1 = checked/blocked.
-- "Treated as unverified until checked directly" (section 8.5), so the schema
-- must not force a true/false default.
CREATE TABLE IF NOT EXISTS robots_ai_crawler_block (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    domain      TEXT NOT NULL,
    blocked     INTEGER,
    checked_at  TEXT NOT NULL,
    raw_notes   TEXT
);

CREATE INDEX IF NOT EXISTS idx_robots_domain ON robots_ai_crawler_block(domain);
