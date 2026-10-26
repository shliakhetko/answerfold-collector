"""Post-window auxiliary collection (Phase-0 batch).

The seven-day AI-search collection window closed 2026-08-26. These modules
collect the time-sensitive context that was never part of the daily run grid
and that decays if not captured now: the Google organic SERP grid, crawler
access policy (robots.txt / llms.txt / ai.txt), the resolution state of every
cited URL, company site inventories, and third-party directory presence.

Layering (INSTRUMENT.md section 12): this is analysis-layer code -- it may
import from collector/ (config, storage.db, runner.execute) and from
analysis/coding/ (the resolved entity roster), never the reverse. It never
touches data/01_raw/runs/*.jsonl or data/01_raw/collector.sqlite3 as a write target.

Collection discipline (study rules): store the raw response body/payload
verbatim, derive nothing. No classifiers, no normalization of existing raw
data, no aggregation, no regex extraction of results into booleans. Every
record carries the standard post-window envelope (analysis.aux.record).
"""
