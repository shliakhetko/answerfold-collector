"""Persistent state for the detached (submit-tonight / collect-tomorrow)
coding-pipeline run.

`scripts/coding/submit_coding_batches.py` writes one of these next to the target
DB; `scripts/coding/collect_coding_batches.py` reads it, advances it one or more
phases per invocation, and rewrites it. The machine can be off between any
two invocations -- the vendor keeps the batch jobs and their results for
weeks, and every fetched result lands in the same LLM cache the normal
pipeline uses.

Phases:
  submitted        -- primary extractor batches are in flight (or there was
                      nothing to submit; batch_ids may be empty)
  primaries_done   -- every primary result is in the extraction cache
  arbiter_submitted -- the single arbiter batch is in flight
  arbiter_done     -- every arbiter result is in the cache; only local
                      consensus math + A3..B remain
  complete         -- the pipeline finished; report_path is written
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PHASES = ("submitted", "primaries_done", "arbiter_submitted", "arbiter_done", "complete")


def default_session_path(db_path: str | Path) -> Path:
    p = Path(db_path)
    return p.with_name(p.stem + ".batch-session.json")


@dataclass
class BatchSession:
    # everything collect_coding_batches needs to finish the run
    runs_dir: str
    db_path: str
    cache_dir: str
    unbranded_intents: list[int]
    inclusion_frame_path: str
    seed_list_path: str
    alt_domains_seed_path: str
    cited_url_resolution_path: str | None
    skip_entity_verification: bool
    report_path: str | None
    limit: int | None = None
    skip_eligibility_floor: bool = False
    stop_after: str | None = None

    phase: str = "submitted"
    primary_batch_ids: dict[str, str] = field(default_factory=dict)
    arbiter_batch_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    history: list[str] = field(default_factory=list)

    def advance(self, phase: str, note: str, now: str) -> None:
        assert phase in PHASES, phase
        self.phase = phase
        self.updated_at = now
        self.history.append(f"{now}  {phase}  {note}")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BatchSession":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
