"""Append-only JSONL run log -- the pilot's storage backend in place of
SQLite (collector.storage.db / collector.storage.api's runs/sources tables,
which stay defined but uncalled by this write path).

One line per run, appended and flushed immediately. `run_id` is
deterministic (collector.runner.execute.build_run_id), so a retry of the
same run produces a second line carrying the same run_id -- nothing is
rewritten or deleted (INSTRUMENT.md section 10: "nothing is edited in
place"). Any reader takes the LAST line for a given run_id as authoritative;
this module never reads the file back to deduplicate.

No parsing, no field extraction, no normalization: `request` and
`response`/`error` are stored exactly as the caller hands them over.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from collector import paths
DEFAULT_RUNS_DIR = paths.RUNS_DIR

# Guards the open/write/flush/fsync sequence below against concurrent
# callers (scripts/collect/run_collection.py now runs one worker thread per engine).
# Lines here embed the full raw vendor response envelope (can be tens of KB),
# and a buffered writer can split that across multiple underlying OS write()
# syscalls at flush() time -- without this lock, two threads writing near-
# simultaneously could interleave and corrupt the file. Process-local is
# enough: only this one process ever writes to a given runs_dir.
_WRITE_LOCK = threading.Lock()


def append_run_line(
    *,
    run_id: str,
    ts_utc: str,
    engine: str,
    intent_id: int,
    paraphrase_id: int,
    rep_index: int,
    day_index: int,
    prompt_order_seed: int,
    position_in_pass: int,
    request: dict[str, Any] | None,
    response: dict[str, Any] | None = None,
    error: str | None = None,
    runs_dir: str | Path | None = None,
) -> Path:
    """Append one line to <runs_dir>/<date from ts_utc>.jsonl.

    Exactly one of `response`/`error` should be set: a failed run writes an
    `error` field instead of `response`, never both, never neither.

    `prompt_order_seed`/`position_in_pass` record the seeded-RNG shuffle that
    produced this rep's prompt order (scripts/collect/run_collection.py) -- required
    on every row so the pass order is fully reconstructable after the fact.
    """
    line: dict[str, Any] = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "engine": engine,
        "intent_id": intent_id,
        "paraphrase_id": paraphrase_id,
        "rep_index": rep_index,
        "day_index": day_index,
        "prompt_order_seed": prompt_order_seed,
        "position_in_pass": position_in_pass,
        "request": request,
    }
    if error is not None:
        line["error"] = error
    else:
        line["response"] = response

    target_dir = Path(runs_dir) if runs_dir is not None else DEFAULT_RUNS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.fromisoformat(ts_utc).date().isoformat()
    path = target_dir / f"{date}.jsonl"

    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    return path
