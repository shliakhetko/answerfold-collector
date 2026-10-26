"""Reads the post-window Google top-10 SERP captures A1 needs.

data/02_aux/google_serp/i{intent_id}-p1.json (paraphrase_id 1 == P01, the
head phrase EXPERIMENT.md's A1 spec pins: "paraphrase P01 of each intent is
the head phrase. Fixed now, so the query cannot be chosen after seeing
which one helps" -- confirmed against config/prompts/prompts.csv, where
paraphrase_id 1 is the shortest/most generic wording for every intent) is
a SerpApi google-engine response; organic_results[].{position, link} is
what derive_serp_binary needs, no pre-extracted domain.

Since 2026-09-29 these files are covered by data/02_aux/MANIFEST.json and
hash-verified by read_frozen_inputs before this loader runs. Originally a
loose post-window aux file (collected 2026-09-06,
11 days after the window closed), same status as reading.py's own
CITED_URL_RESOLUTION_PATH. A missing or unreadable file for a given intent
is skipped with a printed warning, never a hard PhaseBInputError.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from collector import paths

GOOGLE_SERP_DIR = paths.data_rel(paths.AUX_GOOGLE_SERP_DIR)  # relative to the data root
HEAD_PARAPHRASE_ID = 1  # P01
INTENT_IDS_WITH_HEAD_PHRASE_FILES = range(1, 8)  # i1..i7-p1.json exist on disk (7 intents);
# this loader stays intent-agnostic -- A1 itself restricts to the unbranded set via
# analysis.phase_b.citation_panel.UNBRANDED_INTENT_IDS.

SERP_COLLECTION_LIMITATION = (
    "SERP data was collected 2026-09-06 (11 days after the 2026-08-20..26 collection window closed, "
    "post_window=true per data/02_aux/google_serp/index.jsonl), not during the collection window itself -- "
    "EXPERIMENT.md's A1 spec calls for data \"captured once during the collection week.\" This is a real "
    "deviation from the pre-registration, not a footnote: google_top10 reflects Google's ranking as of "
    "Sep 6, not as of the Aug 20-26 window."
)


def read_head_phrase_serp_rows(data_root: Path, serp_dir: Path | None = None) -> list[dict[str, Any]]:
    """[{"intent_id": int, "position": int, "url": str}, ...], shaped for
    analysis.coding.b3b_serp_derivation.derive_serp_binary. One entry per
    organic_results item across every intent's head-phrase file found on
    disk."""
    data_root = Path(data_root)
    serp_dir = GOOGLE_SERP_DIR if serp_dir is None else Path(serp_dir)  # the public deposit passes its own
    rows: list[dict[str, Any]] = []
    for intent_id in INTENT_IDS_WITH_HEAD_PHRASE_FILES:
        path = data_root / serp_dir / f"i{intent_id}-p{HEAD_PARAPHRASE_ID}.json"
        if not path.exists():
            print(f"serp_loader: no head-phrase SERP file for intent {intent_id} at {path} -- skipped")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"serp_loader: could not read/parse {path}: {exc} -- skipped")
            continue
        for result in data.get("organic_results", []):
            url = result.get("link")
            position = result.get("position")
            if url is None or position is None:
                continue
            rows.append({"intent_id": intent_id, "position": position, "url": url})
    return rows
