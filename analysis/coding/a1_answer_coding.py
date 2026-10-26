"""A1: answer coding (Post-Collection Coding Pipeline Spec). 2-of-3 extractor
consensus on every company/product mention plus is_recommended, but the 3rd
vote is only ever spent on disagreements: the 2 PRIMARY_EXTRACTORS always
run in full; ARBITER_EXTRACTOR is called once per run, and only when the
primaries disagree on something, with a small targeted confirm/reject
prompt (analysis.coding.extractors.arbiter) instead of a full re-extraction.
The binary 2-of-3 inclusion decision is unaffected by this -- a candidate
both primaries already agree on reaches quorum without ever consulting the
arbiter, same as it always would have. Deterministic verification and
consensus grouping below run with no model calls.

PRIMARY_EXTRACTORS/ARBITER_EXTRACTOR (2026-09-02): Claude Haiku + Gemini
Flash-Lite are the two primaries, run as one vendor Batch job each; OpenAI
(gpt-5-mini) is the arbiter, run SYNCHRONOUSLY over just the disputed runs.
OpenAI's Batch API is unusable on this account -- every A1 batch failed
validation with "Cannot find file <id>, or organization does not have
access to it", zero requests run (2026-09-01/02) -- so OpenAI cannot be a
primary, but a plain synchronous Responses call works fine and the arbiter
only ever touches candidates the two primaries disagree on. This does make
two of the three voters' vendors (Anthropic, Google) appear among the
engines under test as before; noted in the ambiguity ledger. Reconfigure
these two constants + ARBITER_CALL_FNS if the split should change.

Output: the mentions_raw SQL table (analysis/coding/schema.sql), via
analysis.coding.storage -- run_id, surface_form, evidence_span,
is_recommended, votes.
"""
from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from analysis.coding import storage
from analysis.coding.cache import LLMCache, cache_key
from analysis.coding.extractors.anthropic_extractor import (
    call_anthropic_extractor,
    fetch_anthropic_batch_results,
    poll_anthropic_batch,
    submit_anthropic_batch,
)
from analysis.coding.extractors.arbiter import (
    ArbiterBatchItem,
    ArbiterResult,
    ArbiterVote,
    call_arbiter_vote,
    fetch_arbiter_batch_results,
    submit_arbiter_batch,
)
from analysis.coding.extractors.base import BatchItem, ExtractionResult, MentionCandidate
from analysis.coding.extractors.gemini_extractor import (
    call_gemini_extractor,
    fetch_gemini_batch_results,
    poll_gemini_batch,
    submit_gemini_batch,
)
from analysis.coding.extractors.models import EXTRACTOR_MODELS
from analysis.coding.extractors.openai_extractor import (
    call_openai_arbiter_vote,
    call_openai_extractor,
    fetch_openai_batch_results,
    poll_openai_batch,
    submit_openai_batch,
)
from analysis.coding.extractors.prompts import ARBITER_PROMPT_VERSION, PROMPT_VERSION
from analysis.coding.text_clean import strip_markdown_links

# name -> call_<x>_extractor -- same keys as extractors.models.EXTRACTOR_MODELS,
# since both are looked up by extractor name. All 3 stay registered here
# even though run_a1 only ever calls the 2 PRIMARY_EXTRACTORS through this
# map -- ARBITER_EXTRACTOR is called via ARBITER_CALL_FNS instead.
EXTRACTORS: dict[str, Callable[..., ExtractionResult]] = {
    "anthropic": call_anthropic_extractor,
    "openai": call_openai_extractor,
    "gemini": call_gemini_extractor,
}

PRIMARY_EXTRACTORS: tuple[str, str] = ("anthropic", "gemini")
ARBITER_EXTRACTOR = "openai"

# The arbiter vote call, per vendor -- ARBITER_EXTRACTOR selects one. OpenAI's
# is synchronous-only (see module docstring); Anthropic's also has a Batch
# path (arbiter.py) for an ARBITER_EXTRACTOR="anthropic" reconfig.
ARBITER_CALL_FNS: dict[str, Callable[..., ArbiterResult]] = {
    "anthropic": call_arbiter_vote,
    "openai": call_openai_arbiter_vote,
}
DEFAULT_ARBITER_CALL_FN = ARBITER_CALL_FNS[ARBITER_EXTRACTOR]


@dataclass(frozen=True)
class BatchExtractor:
    submit: Callable[[list[BatchItem]], str]
    poll: Callable[..., str]
    # Uniform (batch_id, items) -> {custom_id: ExtractionResult} across all 3
    # vendors, even though only Gemini's fetch actually needs `items` (its
    # results come back positional, not keyed -- see fetch_gemini_batch_results'
    # docstring); Anthropic/OpenAI's ignore it.
    fetch: Callable[[str, list[BatchItem]], dict[str, ExtractionResult]]


BATCH_EXTRACTORS: dict[str, BatchExtractor] = {
    "anthropic": BatchExtractor(
        submit_anthropic_batch, poll_anthropic_batch, lambda batch_id, items: fetch_anthropic_batch_results(batch_id)
    ),
    "openai": BatchExtractor(
        submit_openai_batch, poll_openai_batch, lambda batch_id, items: fetch_openai_batch_results(batch_id)
    ),
    "gemini": BatchExtractor(submit_gemini_batch, poll_gemini_batch, fetch_gemini_batch_results),
}

DEFAULT_BATCH_MAX_WAIT_SECONDS = 3600.0

# Consensus requires 2-of-3 extractors to agree (spec) -- not derived from
# len(EXTRACTORS) since a future 4th/5th extractor would change quorum math
# the spec never asked for; this is a fixed study parameter, same tier as
# ELIGIBILITY_FLOOR in b3_aggregates.py.
CONSENSUS_MIN_VOTES = 2


def verify_candidate(candidate: MentionCandidate, raw_response_text: str) -> bool:
    """Deterministic, no model (spec): drop if evidence_span is not a
    literal substring of raw_response_text, or surface_form is not a
    literal substring of evidence_span."""
    return candidate.evidence_span in raw_response_text and candidate.surface_form in candidate.evidence_span


def normalize_for_voting(surface_form: str) -> str:
    """Lowercase, strip punctuation and whitespace -- voting only (spec),
    never stored. "Smith.ai" and "Smith AI" both normalize to "smithai"."""
    return re.sub(r"[^a-z0-9]", "", surface_form.lower())


@dataclass(frozen=True)
class MentionRow:
    run_id: str
    surface_form: str
    evidence_span: str
    is_recommended: bool
    votes: tuple[str, ...]  # contributing extractor names, first-seen order


def _group_candidates(per_extractor: dict[str, list[MentionCandidate]]) -> dict[str, list[tuple[str, MentionCandidate]]]:
    """normalize_for_voting(surface_form) -> [(extractor_name, candidate), ...],
    one entry per extractor that emitted that normalized form (at most once
    each). Shared by consensus_mentions (keeps >=2-vote groups) and
    single_vote_mentions (keeps exactly-1-vote groups) so both read the same
    grouping."""
    groups: dict[str, list[tuple[str, MentionCandidate]]] = {}
    for extractor_name, candidates in per_extractor.items():
        seen_in_extractor: set[str] = set()
        for candidate in candidates:
            norm = normalize_for_voting(candidate.surface_form)
            if norm in seen_in_extractor:
                continue  # one vote per extractor per normalized form
            seen_in_extractor.add(norm)
            groups.setdefault(norm, []).append((extractor_name, candidate))
    return groups


def consensus_mentions(run_id: str, per_extractor: dict[str, list[MentionCandidate]]) -> list[MentionRow]:
    """per_extractor values must already be verify_candidate-filtered.

    Groups by normalize_for_voting(); keeps a group only if >=2 extractors
    contributed. surface_form/evidence_span on the output row are taken from
    whichever candidate was seen first (per_extractor's own iteration
    order, then within-extractor list order) -- a documented, arbitrary but
    deterministic tie-break, since the spec doesn't prescribe one.
    is_recommended is a strict majority (>half) of the *contributing*
    extractors' votes; an exact tie (possible with a 2-of-2 split) resolves
    to False, matching the frozen edge rule's own bias toward not
    recommending when a mention is ambiguous."""
    groups = _group_candidates(per_extractor)

    rows: list[MentionRow] = []
    for entries in groups.values():
        if len(entries) < CONSENSUS_MIN_VOTES:
            continue
        _winner_extractor, winner_candidate = entries[0]
        recommended_votes = sum(1 for _, c in entries if c.is_recommended)
        is_recommended = recommended_votes * 2 > len(entries)
        rows.append(
            MentionRow(
                run_id=run_id,
                surface_form=winner_candidate.surface_form,
                evidence_span=winner_candidate.evidence_span,
                is_recommended=is_recommended,
                votes=tuple(name for name, _ in entries),
            )
        )
    return rows


def single_vote_mentions(run_id: str, per_extractor: dict[str, list[MentionCandidate]]) -> list[dict]:
    """Candidates seen by exactly 1 of the 3 extractors -- did not reach the
    2-of-3 consensus quorum. Not part of mentions_raw.jsonl (that frozen
    schema only records consensus rows, per the spec); exposed only for the
    calibration spot-check sample (spec: "Sample ... 1-of-3 dropped cases,"
    analysis.coding.calibration.sample_spot_check)."""
    groups = _group_candidates(per_extractor)
    result: list[dict] = []
    for entries in groups.values():
        if len(entries) != 1:
            continue
        extractor_name, candidate = entries[0]
        result.append(
            {
                "run_id": run_id,
                "extractor": extractor_name,
                "surface_form": candidate.surface_form,
                "evidence_span": candidate.evidence_span,
                "is_recommended": candidate.is_recommended,
            }
        )
    return result


@dataclass
class ExtractorDropStats:
    verified: int = 0
    dropped: int = 0


@dataclass
class A1Stats:
    runs_processed: int = 0
    mentions_written: int = 0
    drop_stats_by_extractor: dict[str, ExtractorDropStats] = field(default_factory=dict)
    single_vote_groups: list[dict] = field(default_factory=list)
    # Cost/behavior visibility for the arbiter step: how often it was
    # actually needed (only when the 2 primaries disagreed), not just how
    # often it theoretically could have been called.
    arbiter_calls_made: int = 0
    arbiter_candidates_checked: int = 0
    arbiter_candidates_confirmed: int = 0

    def drop_rate(self, extractor_name: str) -> float | None:
        s = self.drop_stats_by_extractor.get(extractor_name)
        if s is None or (s.verified + s.dropped) == 0:
            return None
        return s.dropped / (s.verified + s.dropped)


def _extraction_cache_key(extractor_name: str, raw_response_text: str) -> str:
    model = EXTRACTOR_MODELS.get(extractor_name, extractor_name)
    return cache_key(input_text=raw_response_text, model=model, prompt_version=PROMPT_VERSION)


def _extraction_result_from_cache(cached: dict) -> ExtractionResult:
    return ExtractionResult(
        mentions=[MentionCandidate(**m) for m in cached["mentions"]],
        model_returned=cached.get("model_returned"),
        error=cached.get("error"),
        raw_envelope=cached.get("raw_envelope"),
        tokens_input=cached.get("tokens_input"),
        tokens_output=cached.get("tokens_output"),
    )


def _cached_extraction(
    extractor_name: str, call_fn: Callable[..., ExtractionResult], raw_response_text: str, cache: LLMCache
) -> ExtractionResult:
    key = _extraction_cache_key(extractor_name, raw_response_text)
    cached = cache.get(key)
    if cached is not None:
        return _extraction_result_from_cache(cached)
    result = call_fn(raw_response_text=raw_response_text)
    cache.set(key, asdict(result))
    return result


def _arbiter_cache_key(raw_response_text: str, candidates: list[dict]) -> str:
    """Covers both the text and the exact disputed candidate set (sorted, so
    key order never matters) -- a rerun with the same disagreement set hits
    cache; a changed candidate set (e.g. after a primary extractor's prompt
    changes) does not."""
    candidates_key_part = json.dumps(
        [{"surface_form": c["surface_form"], "evidence_span": c["evidence_span"]} for c in candidates],
        sort_keys=True,
    )
    return cache_key(
        input_text=raw_response_text + "\x00" + candidates_key_part,
        model=EXTRACTOR_MODELS[ARBITER_EXTRACTOR],
        prompt_version=ARBITER_PROMPT_VERSION,
    )


def _arbiter_result_from_cache(cached: dict) -> ArbiterResult:
    return ArbiterResult(
        votes=[ArbiterVote(**v) for v in cached["votes"]],
        model_returned=cached.get("model_returned"),
        error=cached.get("error"),
        raw_envelope=cached.get("raw_envelope"),
        tokens_input=cached.get("tokens_input"),
        tokens_output=cached.get("tokens_output"),
    )


def _cached_arbiter_call(
    call_fn: Callable[..., ArbiterResult],
    raw_response_text: str,
    candidates: list[dict],
    cache: LLMCache,
) -> ArbiterResult:
    key = _arbiter_cache_key(raw_response_text, candidates)
    cached = cache.get(key)
    if cached is not None:
        return _arbiter_result_from_cache(cached)
    result = call_fn(raw_response_text=raw_response_text, candidates=candidates)
    cache.set(key, asdict(result))
    return result


# --------------------------------------------------------------------------
# Detached batch flow (scripts/coding/submit_coding_batches.py + collect_coding_batches.py)
#
# The synchronous and use_batch_api paths both submit, wait, and collect in
# ONE process. These helpers split that in two: submit_primary_batches()
# fires the vendor jobs and returns their ids; a later process (possibly
# after the machine was off) polls and fetch_*()es them into the same LLM
# cache. Once every result is cached, run_a1(use_batch_api=True) does only
# the local consensus math -- no network. The arbiter batch depends on the
# primary results, so it is a second round: disputed_by_run_from_cache() ->
# submit_arbiter_batch_from_cache() -> fetch_arbiter_batch_into_cache().
# --------------------------------------------------------------------------

def _eligible_run_rows(run_rows: Iterable[dict]) -> list[dict]:
    return [r for r in run_rows if r.get("raw_response_text")]


def _cleaned_by_run(run_rows: Iterable[dict]) -> dict[str, str]:
    return {r["run_id"]: strip_markdown_links(r["raw_response_text"]) for r in _eligible_run_rows(run_rows)}


def _primary_pending(name: str, cleaned: dict[str, str], cache: LLMCache) -> list[BatchItem]:
    return [
        BatchItem(custom_id=run_id, raw_response_text=text)
        for run_id, text in cleaned.items()
        if cache.get(_extraction_cache_key(name, text)) is None
    ]


def submit_primary_batches(
    run_rows: Iterable[dict], *, cache: LLMCache, batch_extractors: dict[str, BatchExtractor] = BATCH_EXTRACTORS
) -> dict[str, str]:
    """One vendor batch per PRIMARY_EXTRACTOR, covering that primary's
    cache-miss runs. Returns {name: batch_id}; a primary already fully
    cached is omitted. Submits only -- never polls."""
    cleaned = _cleaned_by_run(run_rows)
    batch_ids: dict[str, str] = {}
    for name in PRIMARY_EXTRACTORS:
        pending = _primary_pending(name, cleaned, cache)
        if pending:
            batch_ids[name] = batch_extractors[name].submit(pending)
    return batch_ids


def primary_batch_statuses(
    batch_ids: dict[str, str], *, batch_extractors: dict[str, BatchExtractor] = BATCH_EXTRACTORS
) -> dict[str, str | None]:
    """{name: terminal_status_str} once a batch has finished, {name: None}
    while it is still running. One poll each, no waiting, never raises."""
    statuses: dict[str, str | None] = {}
    for name, batch_id in batch_ids.items():
        try:
            statuses[name] = batch_extractors[name].poll(batch_id, max_wait_seconds=0.0)
        except TimeoutError:
            statuses[name] = None
    return statuses


def fetch_primary_batches(
    run_rows: Iterable[dict],
    batch_ids: dict[str, str],
    *,
    cache: LLMCache,
    batch_extractors: dict[str, BatchExtractor] = BATCH_EXTRACTORS,
) -> None:
    """Fetch each submitted primary batch, writing every result (successes
    AND errors) into the extraction cache. Idempotent: a name already fully
    cached is skipped."""
    cleaned = _cleaned_by_run(run_rows)
    for name, batch_id in batch_ids.items():
        pending = _primary_pending(name, cleaned, cache)
        if not pending:
            continue
        fetched = batch_extractors[name].fetch(batch_id, pending)
        for item in pending:
            result = fetched.get(item.custom_id, ExtractionResult(error="missing_from_batch_result"))
            cache.set(_extraction_cache_key(name, item.raw_response_text), asdict(result))


def disputed_by_run_from_cache(run_rows: Iterable[dict], *, cache: LLMCache) -> dict[str, list[dict]]:
    """{run_id: disputed candidate list} computed from fully-cached primary
    results -- the same "named by exactly one primary" set _run_a1_batched
    builds in its Phase 2."""
    cleaned = _cleaned_by_run(run_rows)
    per_extractor_by_run: dict[str, dict[str, list[MentionCandidate]]] = {run_id: {} for run_id in cleaned}
    for name in PRIMARY_EXTRACTORS:
        for run_id, text in cleaned.items():
            cached = cache.get(_extraction_cache_key(name, text))
            result = _extraction_result_from_cache(cached) if cached is not None else ExtractionResult()
            per_extractor_by_run[run_id][name] = [c for c in result.mentions if verify_candidate(c, text)]
    disputed: dict[str, list[dict]] = {}
    for run_id, per_extractor in per_extractor_by_run.items():
        one_vote = single_vote_mentions(run_id, per_extractor)
        if one_vote:
            disputed[run_id] = one_vote
    return disputed


def _arbiter_pending(
    run_rows: Iterable[dict], disputed_by_run: dict[str, list[dict]], arbiter_cache: LLMCache
) -> list[ArbiterBatchItem]:
    cleaned = _cleaned_by_run(run_rows)
    return [
        ArbiterBatchItem(run_id, cleaned[run_id], disputed)
        for run_id, disputed in disputed_by_run.items()
        if run_id in cleaned and arbiter_cache.get(_arbiter_cache_key(cleaned[run_id], disputed)) is None
    ]


def submit_arbiter_batch_from_cache(
    run_rows: Iterable[dict],
    disputed_by_run: dict[str, list[dict]],
    *,
    arbiter_cache: LLMCache,
    submit_fn: Callable[..., str] = submit_arbiter_batch,
) -> str | None:
    """One arbiter batch for every disputed set not already cached. Returns
    the batch id, or None when nothing needs the arbiter."""
    pending = _arbiter_pending(run_rows, disputed_by_run, arbiter_cache)
    return submit_fn(pending) if pending else None


def fetch_arbiter_batch_into_cache(
    run_rows: Iterable[dict],
    disputed_by_run: dict[str, list[dict]],
    batch_id: str,
    *,
    arbiter_cache: LLMCache,
    fetch_fn: Callable[..., dict[str, ArbiterResult]] = fetch_arbiter_batch_results,
) -> None:
    pending = _arbiter_pending(run_rows, disputed_by_run, arbiter_cache)
    if not pending:
        return
    fetched = fetch_fn(batch_id)
    for item in pending:
        result = fetched.get(item.custom_id, ArbiterResult(error="missing_from_batch_result"))
        arbiter_cache.set(_arbiter_cache_key(item.raw_response_text, item.candidates), asdict(result))


def _adjudicate_sync(
    pending: list[ArbiterBatchItem],
    *,
    arbiter_cache: LLMCache,
    call_fn: Callable[..., ArbiterResult],
    max_workers: int,
) -> dict[str, ArbiterResult]:
    """Run call_fn over each pending disputed set (thread-pooled), cache
    every result, return {custom_id: ArbiterResult}. Shared by
    run_arbiter_sync_into_cache and _run_a1_batched's phase 2."""
    out: dict[str, ArbiterResult] = {}
    if not pending:
        return out

    def _one(item: ArbiterBatchItem) -> tuple[ArbiterBatchItem, ArbiterResult]:
        return item, call_fn(raw_response_text=item.raw_response_text, candidates=item.candidates)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for item, result in pool.map(_one, pending):
            arbiter_cache.set(_arbiter_cache_key(item.raw_response_text, item.candidates), asdict(result))
            out[item.custom_id] = result
    return out


def run_arbiter_sync_into_cache(
    run_rows: Iterable[dict],
    disputed_by_run: dict[str, list[dict]],
    *,
    arbiter_cache: LLMCache,
    call_fn: Callable[..., ArbiterResult] = DEFAULT_ARBITER_CALL_FN,
    max_workers: int = 8,
) -> int:
    """Adjudicate every disputed run SYNCHRONOUSLY (one small call per run,
    thread-pooled), writing ArbiterResults into the arbiter cache. Returns
    the number of new calls made. Idempotent -- a disputed set already
    cached is skipped. This is collect_coding_batches' replacement for the
    Anthropic arbiter Batch job now that ARBITER_EXTRACTOR is OpenAI, whose
    Batch API is unusable on this account; the arbiter only ever touches
    the (few hundred) runs the two primaries disagree on."""
    pending = _arbiter_pending(run_rows, disputed_by_run, arbiter_cache)
    _adjudicate_sync(pending, arbiter_cache=arbiter_cache, call_fn=call_fn, max_workers=max_workers)
    return len(pending)


def run_a1(
    run_rows: Iterable[dict],
    *,
    cache: LLMCache,
    arbiter_cache: LLMCache | None = None,
    extractors: dict[str, Callable[..., ExtractionResult]] = EXTRACTORS,
    arbiter_call_fn: Callable[..., ArbiterResult] = DEFAULT_ARBITER_CALL_FN,
    use_batch_api: bool = False,
    batch_extractors: dict[str, BatchExtractor] = BATCH_EXTRACTORS,
    batch_max_wait_seconds: float = DEFAULT_BATCH_MAX_WAIT_SECONDS,
    offline: bool = False,
    conn: sqlite3.Connection,
) -> A1Stats:
    """run_rows: iterable of {"run_id": str, "raw_response_text": str | None}.
    Clears the mentions_raw table before writing (spec: "Stages are
    independently rerunnable") -- not an append, so a rerun never
    duplicates rows.

    arbiter_cache defaults to `cache` if not given (existing single-cache
    callers keep working) -- scripts/coding/run_coding_pipeline.py passes a
    separate cache_dir so the arbiter's calls report as their own stage in
    scripts/coding/report_coding_pipeline_cost.py rather than being folded into
    whichever primary extractor's directory.

    use_batch_api=True switches to _run_a1_batched: submits every cache-miss
    run for a given primary as ONE vendor Batch API job (not one call per
    run), at each vendor's Batch API discount. The arbiter always runs
    synchronously over just the disputed runs (ARBITER_EXTRACTOR is OpenAI,
    whose Batch API is unusable on this account). Same consensus math either
    way."""
    arbiter_cache = arbiter_cache or cache
    if use_batch_api:
        return _run_a1_batched(
            run_rows, cache=cache, arbiter_cache=arbiter_cache, batch_extractors=batch_extractors,
            arbiter_call_fn=arbiter_call_fn, max_wait_seconds=batch_max_wait_seconds,
            offline=offline, conn=conn,
        )

    stats = A1Stats()
    storage.clear_mentions_raw(conn=conn)
    primaries = {name: extractors[name] for name in PRIMARY_EXTRACTORS}

    for run in run_rows:
        raw_response_text = run.get("raw_response_text")
        if not raw_response_text:
            continue  # nothing to extract from an errored/empty run
        run_id = run["run_id"]

        # Cleaned once, reused for every extractor/arbiter call and for
        # verify_candidate below -- extraction input and verification must
        # agree on the same text or evidence spans stop matching (see
        # text_clean.py's docstring).
        cleaned_text = strip_markdown_links(raw_response_text)

        per_extractor: dict[str, list[MentionCandidate]] = {}
        for name, call_fn in primaries.items():
            result = _cached_extraction(name, call_fn, cleaned_text, cache)
            verified = [c for c in result.mentions if verify_candidate(c, cleaned_text)]
            dropped = len(result.mentions) - len(verified)
            entry = stats.drop_stats_by_extractor.setdefault(name, ExtractorDropStats())
            entry.verified += len(verified)
            entry.dropped += dropped
            per_extractor[name] = verified

        # Candidates the 2 primaries already agree on will reach quorum via
        # consensus_mentions() below on their own 2 votes, without ever
        # touching the arbiter -- disputed is exactly what's left over:
        # named by only one primary.
        disputed = single_vote_mentions(run_id, per_extractor)
        arbiter_confirmed: list[MentionCandidate] = []
        arbiter_error: str | None = None
        if disputed:
            arbiter_result = _cached_arbiter_call(arbiter_call_fn, cleaned_text, disputed, arbiter_cache)
            arbiter_error = arbiter_result.error
            votes_by_norm = {normalize_for_voting(v.surface_form): v for v in arbiter_result.votes}
            for d in disputed:
                vote = votes_by_norm.get(normalize_for_voting(d["surface_form"]))
                if vote is not None and vote.confirmed:
                    arbiter_confirmed.append(
                        MentionCandidate(
                            surface_form=d["surface_form"], evidence_span=d["evidence_span"],
                            is_recommended=vote.is_recommended,
                        )
                    )
            stats.arbiter_calls_made += 1
            stats.arbiter_candidates_checked += len(disputed)
            stats.arbiter_candidates_confirmed += len(arbiter_confirmed)

        # A confirmed disputed candidate now appears in exactly 2 of these
        # 3 lists (its original primary + the arbiter) -> reaches quorum via
        # the same consensus_mentions() every other candidate goes through.
        # An already-2-vote-agreed candidate is unaffected: the arbiter's
        # list never contained it in the first place.
        per_extractor[ARBITER_EXTRACTOR] = arbiter_confirmed

        rows = consensus_mentions(run_id, per_extractor)
        if rows:
            storage.insert_mentions_raw(
                [{**asdict(row), "votes": list(row.votes)} for row in rows], conn=conn
            )
            stats.mentions_written += len(rows)

        final_single_votes = single_vote_mentions(run_id, per_extractor)
        for group in final_single_votes:
            # Every remaining single-vote item was part of `disputed` (the
            # arbiter's input covers 100% of it) -- checked=True unless the
            # arbiter call itself errored, in which case none of them were
            # actually reviewed.
            group["arbiter_checked"] = bool(disputed) and arbiter_error is None
        stats.single_vote_groups.extend(final_single_votes)
        stats.runs_processed += 1

    return stats


def _run_a1_batched(
    run_rows: Iterable[dict],
    *,
    cache: LLMCache,
    arbiter_cache: LLMCache,
    batch_extractors: dict[str, BatchExtractor],
    arbiter_call_fn: Callable[..., ArbiterResult] = DEFAULT_ARBITER_CALL_FN,
    max_wait_seconds: float,
    arbiter_max_workers: int = 8,
    offline: bool = False,
    conn: sqlite3.Connection,
) -> A1Stats:
    """Phases: (1) each primary's cache-miss runs go out as one vendor Batch
    job per primary; (2) once both primaries are in, every run's disputed
    set (candidates named by only one primary) is adjudicated by the
    synchronous arbiter (thread-pooled over just the disputed runs); (3)
    consensus + write, run by run -- identical math to run_a1's synchronous
    path. This function only changes how ExtractionResult/ArbiterResult get
    produced, not what happens with them afterward."""
    stats = A1Stats()
    storage.clear_mentions_raw(conn=conn)

    eligible_runs = [r for r in run_rows if r.get("raw_response_text")]
    cleaned_by_run: dict[str, str] = {
        r["run_id"]: strip_markdown_links(r["raw_response_text"]) for r in eligible_runs
    }

    # Phase 1: one batch job per primary, covering every cache-miss run.
    per_extractor_by_run: dict[str, dict[str, list[MentionCandidate]]] = {run_id: {} for run_id in cleaned_by_run}
    for name in PRIMARY_EXTRACTORS:
        batch_extractor = batch_extractors[name]
        by_run: dict[str, ExtractionResult] = {}
        pending: list[BatchItem] = []
        for run_id, text in cleaned_by_run.items():
            key = _extraction_cache_key(name, text)
            cached = cache.get(key)
            if cached is not None:
                by_run[run_id] = _extraction_result_from_cache(cached)
            else:
                pending.append(BatchItem(custom_id=run_id, raw_response_text=text))

        if pending and offline:
            raise RuntimeError(
                f"a1 offline mode: {len(pending)} runs missing from the {name} extraction cache "
                "-- the detached batch collect is incomplete (re-run collect_coding_batches)"
            )
        if pending:
            batch_id = batch_extractor.submit(pending)
            batch_extractor.poll(batch_id, max_wait_seconds=max_wait_seconds)
            fetched = batch_extractor.fetch(batch_id, pending)
            for item in pending:
                result = fetched.get(item.custom_id, ExtractionResult(error="missing_from_batch_result"))
                cache.set(_extraction_cache_key(name, item.raw_response_text), asdict(result))
                by_run[item.custom_id] = result

        for run_id, text in cleaned_by_run.items():
            result = by_run[run_id]
            verified = [c for c in result.mentions if verify_candidate(c, text)]
            entry = stats.drop_stats_by_extractor.setdefault(name, ExtractorDropStats())
            entry.verified += len(verified)
            entry.dropped += len(result.mentions) - len(verified)
            per_extractor_by_run[run_id][name] = verified

    # Phase 2: adjudicate every run's disputed set with the synchronous arbiter.
    disputed_by_run: dict[str, list[dict]] = {}
    for run_id, per_extractor in per_extractor_by_run.items():
        disputed = single_vote_mentions(run_id, per_extractor)
        if disputed:
            disputed_by_run[run_id] = disputed

    arbiter_confirmed_by_run: dict[str, list[MentionCandidate]] = {run_id: [] for run_id in cleaned_by_run}
    arbiter_error_by_run: dict[str, str | None] = {}
    if disputed_by_run:
        arbiter_by_run: dict[str, ArbiterResult] = {}
        pending_arbiter: list[ArbiterBatchItem] = []
        for run_id, disputed in disputed_by_run.items():
            key = _arbiter_cache_key(cleaned_by_run[run_id], disputed)
            cached = arbiter_cache.get(key)
            if cached is not None:
                arbiter_by_run[run_id] = _arbiter_result_from_cache(cached)
            else:
                pending_arbiter.append(ArbiterBatchItem(run_id, cleaned_by_run[run_id], disputed))

        if pending_arbiter and offline:
            raise RuntimeError(
                f"a1 offline mode: {len(pending_arbiter)} disputed sets missing from the arbiter cache "
                "-- the detached batch collect is incomplete (re-run collect_coding_batches)"
            )
        arbiter_by_run.update(
            _adjudicate_sync(
                pending_arbiter, arbiter_cache=arbiter_cache,
                call_fn=arbiter_call_fn, max_workers=arbiter_max_workers,
            )
        )

        for run_id, disputed in disputed_by_run.items():
            arbiter_result = arbiter_by_run[run_id]
            arbiter_error_by_run[run_id] = arbiter_result.error
            votes_by_norm = {normalize_for_voting(v.surface_form): v for v in arbiter_result.votes}
            confirmed = []
            for d in disputed:
                vote = votes_by_norm.get(normalize_for_voting(d["surface_form"]))
                if vote is not None and vote.confirmed:
                    confirmed.append(
                        MentionCandidate(
                            surface_form=d["surface_form"], evidence_span=d["evidence_span"],
                            is_recommended=vote.is_recommended,
                        )
                    )
            arbiter_confirmed_by_run[run_id] = confirmed
            stats.arbiter_calls_made += 1
            stats.arbiter_candidates_checked += len(disputed)
            stats.arbiter_candidates_confirmed += len(confirmed)

    # Phase 3: consensus + write, run by run -- identical math to run_a1's
    # synchronous path from this point on.
    for run_id, per_extractor in per_extractor_by_run.items():
        per_extractor[ARBITER_EXTRACTOR] = arbiter_confirmed_by_run[run_id]
        rows = consensus_mentions(run_id, per_extractor)
        if rows:
            storage.insert_mentions_raw([{**asdict(row), "votes": list(row.votes)} for row in rows], conn=conn)
            stats.mentions_written += len(rows)

        final_single_votes = single_vote_mentions(run_id, per_extractor)
        disputed = disputed_by_run.get(run_id, [])
        arbiter_error = arbiter_error_by_run.get(run_id)
        for group in final_single_votes:
            group["arbiter_checked"] = bool(disputed) and arbiter_error is None
        stats.single_vote_groups.extend(final_single_votes)
        stats.runs_processed += 1

    return stats
