from analysis.coding import storage
from analysis.coding.a1_answer_coding import (
    ARBITER_EXTRACTOR,
    PRIMARY_EXTRACTORS,
    ArbiterResult,
    ArbiterVote,
    BatchExtractor,
    ExtractionResult,
    MentionCandidate,
    consensus_mentions,
    normalize_for_voting,
    run_a1,
    single_vote_mentions,
    verify_candidate,
)
from analysis.coding.extractors.base import BatchItem
from analysis.coding.cache import LLMCache
from analysis.coding.db import init_coding_schema
from collector.storage import api
from collector.storage.db import get_connection, init_db

RAW_TEXT = "Goodcall is a strong choice. Unlike Vapi, which needs setup, Goodcall works out of the box."


def _fresh_conn(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    init_coding_schema(db_path)
    return get_connection(db_path)


def _seed_run(conn, run_id):
    api.insert_run(
        run_id=run_id, instrument_commit="deadbeef", timestamp="2026-08-24T10:00:00+00:00",
        engine="anthropic_claude", intent_id=1, paraphrase_id=1, prompt_text="x",
        rep_index=1, day_index=1, conn=conn,
    )
    conn.commit()


def _fake_extractor(mentions, *, model_returned="fake-model", tokens_input=10, tokens_output=5, error=None):
    def call_fn(*, raw_response_text):
        return ExtractionResult(
            mentions=list(mentions), model_returned=model_returned, error=error,
            tokens_input=tokens_input, tokens_output=tokens_output,
        )
    return call_fn


def _primary_extractors(a, b):
    """a, b map to PRIMARY_EXTRACTORS[0], PRIMARY_EXTRACTORS[1] in order."""
    name_a, name_b = PRIMARY_EXTRACTORS
    return {name_a: a, name_b: b}


def _fake_arbiter(votes, *, error=None, tokens_input=15, tokens_output=8):
    def call_fn(*, raw_response_text, candidates):
        return ArbiterResult(votes=list(votes), error=error, tokens_input=tokens_input, tokens_output=tokens_output)
    return call_fn


def _fake_batch_extractor(mentions_by_run_id, *, submit_calls=None):
    """mentions_by_run_id: {run_id: [MentionCandidate, ...]} -- fetch()
    returns this per item regardless of which primary it's registered as."""
    def submit(items):
        if submit_calls is not None:
            submit_calls.append([item.custom_id for item in items])
        return "batch-fake-id"

    def poll(batch_id, *, max_wait_seconds):
        return "ended"

    def fetch(batch_id, items):
        return {item.custom_id: ExtractionResult(mentions=list(mentions_by_run_id.get(item.custom_id, []))) for item in items}

    return BatchExtractor(submit=submit, poll=poll, fetch=fetch)


def _fake_arbiter_call_fn(votes_by_norm_text, *, seen=None):
    """votes_by_norm_text: {raw_response_text: [ArbiterVote, ...]}. The
    arbiter runs synchronously, one call per disputed run."""
    def call_fn(*, raw_response_text, candidates):
        if seen is not None:
            seen.append(raw_response_text)
        return ArbiterResult(votes=list(votes_by_norm_text.get(raw_response_text, [])))

    return call_fn


def _refusing_arbiter_call_fn(*args, **kwargs):
    raise AssertionError("arbiter should not be called")


# --------------------------------------------------------------------------
# verify_candidate
# --------------------------------------------------------------------------

def test_verify_candidate_accepts_literal_substrings():
    candidate = MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)
    assert verify_candidate(candidate, RAW_TEXT) is True


def test_verify_candidate_rejects_hallucinated_evidence_span():
    # evidence_span never appears in raw_response_text at all.
    candidate = MentionCandidate("Goodcall", "Goodcall cures cancer.", True)
    assert verify_candidate(candidate, RAW_TEXT) is False


def test_verify_candidate_rejects_surface_form_not_in_evidence_span():
    # evidence_span is real, but surface_form doesn't literally appear in it.
    candidate = MentionCandidate("Acme Corp", "Goodcall is a strong choice.", True)
    assert verify_candidate(candidate, RAW_TEXT) is False


# --------------------------------------------------------------------------
# normalize_for_voting
# --------------------------------------------------------------------------

def test_normalize_for_voting_collapses_spelling_variants():
    assert normalize_for_voting("Smith.ai") == normalize_for_voting("Smith AI") == "smithai"


def test_normalize_for_voting_is_case_and_whitespace_insensitive():
    assert normalize_for_voting("  Good Call  ") == normalize_for_voting("GOODCALL")


# --------------------------------------------------------------------------
# consensus_mentions
# --------------------------------------------------------------------------

def test_consensus_keeps_2_of_3_and_drops_1_of_3():
    per_extractor = {
        "anthropic": [MentionCandidate("Goodcall", "Goodcall is great.", True)],
        "openai": [MentionCandidate("Goodcall", "Goodcall is great.", True)],
        "gemini": [MentionCandidate("Vapi", "Vapi is different.", False)],  # only 1 vote
    }
    rows = consensus_mentions("run-1", per_extractor)
    assert len(rows) == 1
    assert rows[0].surface_form == "Goodcall"
    assert rows[0].votes == ("anthropic", "openai")


def test_consensus_majority_recommended_wins():
    per_extractor = {
        "anthropic": [MentionCandidate("Goodcall", "...", True)],
        "openai": [MentionCandidate("Goodcall", "...", True)],
        "gemini": [MentionCandidate("Goodcall", "...", False)],
    }
    rows = consensus_mentions("run-1", per_extractor)
    assert rows[0].is_recommended is True


def test_consensus_tie_resolves_to_not_recommended():
    per_extractor = {
        "anthropic": [MentionCandidate("Goodcall", "...", True)],
        "openai": [MentionCandidate("Goodcall", "...", False)],
    }
    rows = consensus_mentions("run-1", per_extractor)
    assert rows[0].is_recommended is False


def test_consensus_surface_form_uses_first_seen_spelling():
    per_extractor = {
        "anthropic": [MentionCandidate("Smith.ai", "Smith.ai is good.", True)],
        "openai": [MentionCandidate("Smith AI", "Smith AI is good.", True)],
    }
    rows = consensus_mentions("run-1", per_extractor)
    assert rows[0].surface_form == "Smith.ai"  # first-seen (dict iteration order)


def test_consensus_one_vote_per_extractor_even_with_duplicates():
    # An extractor emitting the same normalized surface_form twice must
    # still count as exactly one vote from that extractor.
    per_extractor = {
        "anthropic": [
            MentionCandidate("Goodcall", "Goodcall is great.", True),
            MentionCandidate("GoodCall", "GoodCall works well.", True),
        ],
        "openai": [MentionCandidate("Goodcall", "Goodcall is great.", True)],
    }
    rows = consensus_mentions("run-1", per_extractor)
    assert len(rows) == 1
    assert rows[0].votes == ("anthropic", "openai")


def test_consensus_no_mentions_produces_no_rows():
    assert consensus_mentions("run-1", {"anthropic": [], "openai": [], "gemini": []}) == []


# --------------------------------------------------------------------------
# single_vote_mentions
# --------------------------------------------------------------------------

def test_single_vote_mentions_returns_only_1_of_3_groups():
    per_extractor = {
        "anthropic": [MentionCandidate("Goodcall", "...", True)],
        "openai": [MentionCandidate("Goodcall", "...", True)],  # 2-of-3, excluded here
        "gemini": [MentionCandidate("Vapi", "Vapi is different.", False)],  # 1-of-3
    }
    result = single_vote_mentions("run-1", per_extractor)
    assert result == [
        {"run_id": "run-1", "extractor": "gemini", "surface_form": "Vapi", "evidence_span": "Vapi is different.", "is_recommended": False}
    ]


def test_single_vote_mentions_empty_when_everything_reaches_quorum():
    per_extractor = {
        "anthropic": [MentionCandidate("Goodcall", "...", True)],
        "openai": [MentionCandidate("Goodcall", "...", True)],
    }
    assert single_vote_mentions("run-1", per_extractor) == []


# --------------------------------------------------------------------------
# run_a1 -- both PRIMARY_EXTRACTORS always run; ARBITER_EXTRACTOR only runs
# when they disagree. Roles are passed in explicitly here, so these stay
# valid regardless of which vendors the constants name.
# --------------------------------------------------------------------------

def test_run_a1_happy_path_both_primaries_agree_arbiter_never_called(tmp_path):
    extractors = _primary_extractors(
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
    )
    # Any call at all is a bug: both primaries already agree, so the arbiter
    # should never be consulted.
    def arbiter_call_fn(*, raw_response_text, candidates):
        raise AssertionError("arbiter should not be called when primaries agree")

    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    stats = run_a1(
        [{"run_id": "run-1", "raw_response_text": RAW_TEXT}],
        cache=cache, extractors=extractors, arbiter_call_fn=arbiter_call_fn, conn=conn,
    )

    assert stats.runs_processed == 1
    assert stats.mentions_written == 1
    assert stats.arbiter_calls_made == 0
    rows = storage.list_mentions_raw(conn=conn)
    assert rows == [{
        "run_id": "run-1", "surface_form": "Goodcall",
        "evidence_span": "Goodcall is a strong choice.", "is_recommended": True,
        "votes": list(PRIMARY_EXTRACTORS),
    }]


def test_run_a1_disagreement_triggers_one_arbiter_call_that_confirms(tmp_path):
    # openai names Goodcall; gemini names nothing -- a disputed candidate the
    # arbiter then confirms, promoting it to a 2-vote consensus row.
    extractors = _primary_extractors(
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
        _fake_extractor([]),
    )
    arbiter_calls = {"n": 0}

    def arbiter_call_fn(*, raw_response_text, candidates):
        arbiter_calls["n"] += 1
        assert candidates == [{
            "run_id": "run-1", "extractor": PRIMARY_EXTRACTORS[0], "surface_form": "Goodcall",
            "evidence_span": "Goodcall is a strong choice.", "is_recommended": True,
        }]
        return ArbiterResult(votes=[ArbiterVote("Goodcall", confirmed=True, is_recommended=True)])

    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    stats = run_a1(
        [{"run_id": "run-1", "raw_response_text": RAW_TEXT}],
        cache=cache, extractors=extractors, arbiter_call_fn=arbiter_call_fn, conn=conn,
    )

    assert arbiter_calls["n"] == 1
    assert stats.arbiter_calls_made == 1
    assert stats.arbiter_candidates_checked == 1
    assert stats.arbiter_candidates_confirmed == 1
    rows = storage.list_mentions_raw(conn=conn)
    assert len(rows) == 1
    assert rows[0]["votes"] == [PRIMARY_EXTRACTORS[0], ARBITER_EXTRACTOR]
    assert stats.single_vote_groups == []


def test_run_a1_disagreement_triggers_one_arbiter_call_that_rejects(tmp_path):
    extractors = _primary_extractors(
        _fake_extractor([MentionCandidate("NotACompany", "NotACompany was mentioned.", True)]),
        _fake_extractor([]),
    )
    arbiter_call_fn = _fake_arbiter([ArbiterVote("NotACompany", confirmed=False, is_recommended=False)])
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    stats = run_a1(
        [{"run_id": "run-1", "raw_response_text": "NotACompany was mentioned."}],
        cache=cache, extractors=extractors, arbiter_call_fn=arbiter_call_fn, conn=conn,
    )

    assert stats.arbiter_calls_made == 1
    assert stats.arbiter_candidates_confirmed == 0
    assert storage.list_mentions_raw(conn=conn) == []
    assert stats.single_vote_groups == [
        {"run_id": "run-1", "extractor": PRIMARY_EXTRACTORS[0], "surface_form": "NotACompany",
         "evidence_span": "NotACompany was mentioned.", "is_recommended": True, "arbiter_checked": True}
    ]


def test_run_a1_arbiter_call_error_leaves_candidate_unchecked(tmp_path):
    extractors = _primary_extractors(
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
        _fake_extractor([]),
    )
    arbiter_call_fn = _fake_arbiter([], error="TimeoutError: simulated")
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    stats = run_a1(
        [{"run_id": "run-1", "raw_response_text": RAW_TEXT}],
        cache=cache, extractors=extractors, arbiter_call_fn=arbiter_call_fn, conn=conn,
    )

    assert storage.list_mentions_raw(conn=conn) == []
    assert stats.single_vote_groups[0]["arbiter_checked"] is False


def test_run_a1_drops_hallucinated_evidence_span_and_logs_it(tmp_path):
    # One primary hallucinates an evidence_span that isn't a literal
    # substring of raw_response_text -- the deterministic verification step
    # must drop it, independent of any real API call. With only 1 primary
    # left proposing "Goodcall", it becomes a disputed candidate.
    extractors = _primary_extractors(
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall cures every disease known to science.", True)]),
    )
    arbiter_call_fn = _fake_arbiter([ArbiterVote("Goodcall", confirmed=True, is_recommended=True)])
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    stats = run_a1(
        [{"run_id": "run-1", "raw_response_text": RAW_TEXT}],
        cache=cache, extractors=extractors, arbiter_call_fn=arbiter_call_fn, conn=conn,
    )

    rows = storage.list_mentions_raw(conn=conn)
    assert len(rows) == 1
    assert rows[0]["votes"] == [PRIMARY_EXTRACTORS[0], ARBITER_EXTRACTOR]
    assert stats.drop_stats_by_extractor[PRIMARY_EXTRACTORS[1]].dropped == 1
    assert stats.drop_stats_by_extractor[PRIMARY_EXTRACTORS[1]].verified == 0
    assert stats.drop_stats_by_extractor[PRIMARY_EXTRACTORS[0]].dropped == 0
    assert stats.drop_rate(PRIMARY_EXTRACTORS[1]) == 1.0
    assert stats.drop_rate(PRIMARY_EXTRACTORS[0]) == 0.0


def test_run_a1_skips_runs_with_no_text(tmp_path):
    extractors = _primary_extractors(_fake_extractor([]), _fake_extractor([]))
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)

    stats = run_a1([{"run_id": "run-errored", "raw_response_text": None}], cache=cache, extractors=extractors, conn=conn)

    assert stats.runs_processed == 0
    assert storage.list_mentions_raw(conn=conn) == []


def test_run_a1_uses_cache_on_second_call(tmp_path):
    call_count = {"n": 0}

    def counting_extractor(*, raw_response_text):
        call_count["n"] += 1
        return ExtractionResult(mentions=[MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)])

    extractors = _primary_extractors(counting_extractor, counting_extractor)
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")
    _seed_run(conn, "run-2")

    run_a1([{"run_id": "run-1", "raw_response_text": RAW_TEXT}], cache=cache, extractors=extractors, conn=conn)
    assert call_count["n"] == 2  # 2 primaries, no disagreement -- arbiter never called

    # Same text, different run_id -- cache is keyed on (model, prompt_version,
    # input_text), not run_id, so this must hit cache instead of calling again.
    run_a1([{"run_id": "run-2", "raw_response_text": RAW_TEXT}], cache=cache, extractors=extractors, conn=conn)
    assert call_count["n"] == 2


def test_run_a1_uses_cache_for_arbiter_on_second_call(tmp_path):
    extractors = _primary_extractors(
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
        _fake_extractor([]),
    )
    arbiter_calls = {"n": 0}

    def counting_arbiter(*, raw_response_text, candidates):
        arbiter_calls["n"] += 1
        return ArbiterResult(votes=[ArbiterVote("Goodcall", confirmed=True, is_recommended=True)])

    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")
    _seed_run(conn, "run-2")

    run_a1([{"run_id": "run-1", "raw_response_text": RAW_TEXT}], cache=cache, extractors=extractors, arbiter_call_fn=counting_arbiter, conn=conn)
    assert arbiter_calls["n"] == 1

    run_a1([{"run_id": "run-2", "raw_response_text": RAW_TEXT}], cache=cache, extractors=extractors, arbiter_call_fn=counting_arbiter, conn=conn)
    assert arbiter_calls["n"] == 1  # same text + same disputed set -- cache hit


def test_run_a1_extractors_receive_markdown_link_stripped_text(tmp_path):
    # google_ai_mode-style inline citation markdown -- the extractor must
    # see "Fathom is the overall best option.", not the raw link syntax.
    linked_text = "[Fathom](https://fathom.video/) is the overall best option."
    received: dict[str, str] = {}

    def recording_extractor(name):
        def call_fn(*, raw_response_text):
            received[name] = raw_response_text
            return ExtractionResult(mentions=[MentionCandidate("Fathom", "Fathom is the overall best option.", True)])
        return call_fn

    extractors = _primary_extractors(recording_extractor("a"), recording_extractor("b"))
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    stats = run_a1([{"run_id": "run-1", "raw_response_text": linked_text}], cache=cache, extractors=extractors, conn=conn)

    assert received == {"a": "Fathom is the overall best option.", "b": "Fathom is the overall best option."}
    # And the evidence_span (drawn from the cleaned text an extractor would
    # actually return) survives verify_candidate rather than being dropped
    # as a hallucination -- proving extraction input and verification agree.
    assert stats.mentions_written == 1
    rows = storage.list_mentions_raw(conn=conn)
    assert rows[0]["evidence_span"] == "Fathom is the overall best option."


def test_run_a1_rewrites_table_not_append(tmp_path):
    extractors = _primary_extractors(
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
        _fake_extractor([MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]),
    )
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    run_a1([{"run_id": "run-1", "raw_response_text": RAW_TEXT}], cache=cache, extractors=extractors, conn=conn)
    run_a1([{"run_id": "run-1", "raw_response_text": RAW_TEXT}], cache=cache, extractors=extractors, conn=conn)

    rows = storage.list_mentions_raw(conn=conn)
    assert len(rows) == 1  # not duplicated on rerun


# --------------------------------------------------------------------------
# run_a1(use_batch_api=True) -- _run_a1_batched: one job submission per
# primary (covering every run at once); the arbiter runs synchronously,
# one call per disputed run.
# --------------------------------------------------------------------------

def test_run_a1_batched_one_submit_per_primary_no_disagreement(tmp_path):
    mentions = {"run-1": [MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)]}
    submit_calls_a, submit_calls_b = [], []
    name_a, name_b = PRIMARY_EXTRACTORS
    batch_extractors = {
        name_a: _fake_batch_extractor(mentions, submit_calls=submit_calls_a),
        name_b: _fake_batch_extractor(mentions, submit_calls=submit_calls_b),
    }
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")

    stats = run_a1(
        [{"run_id": "run-1", "raw_response_text": RAW_TEXT}],
        cache=cache, use_batch_api=True, batch_extractors=batch_extractors,
        arbiter_call_fn=_refusing_arbiter_call_fn, conn=conn,
    )

    assert submit_calls_a == [["run-1"]]
    assert submit_calls_b == [["run-1"]]
    assert stats.arbiter_calls_made == 0
    rows = storage.list_mentions_raw(conn=conn)
    assert rows == [{
        "run_id": "run-1", "surface_form": "Goodcall",
        "evidence_span": "Goodcall is a strong choice.", "is_recommended": True,
        "votes": list(PRIMARY_EXTRACTORS),
    }]


def test_run_a1_batched_skips_cache_hit_runs_from_submission(tmp_path):
    name_a, name_b = PRIMARY_EXTRACTORS
    mentions = {
        "run-1": [MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)],
        "run-2": [MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)],
    }
    submit_calls_a: list = []
    batch_extractors = {
        name_a: _fake_batch_extractor(mentions, submit_calls=submit_calls_a),
        name_b: _fake_batch_extractor(mentions),
    }
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")
    _seed_run(conn, "run-2")

    run_a1(
        [{"run_id": "run-1", "raw_response_text": RAW_TEXT}],
        cache=cache, use_batch_api=True, batch_extractors=batch_extractors,
        arbiter_call_fn=_refusing_arbiter_call_fn, conn=conn,
    )
    assert submit_calls_a == [["run-1"]]

    # Second run, same text -- cache_key is (model, prompt_version, text),
    # not run_id, so run-2's text hits the same cache entry run-1 already
    # populated and must never be submitted.
    run_a1(
        [{"run_id": "run-2", "raw_response_text": RAW_TEXT}],
        cache=cache, use_batch_api=True, batch_extractors=batch_extractors,
        arbiter_call_fn=_refusing_arbiter_call_fn, conn=conn,
    )
    assert submit_calls_a == [["run-1"]]  # unchanged -- no second submission


def test_run_a1_batched_disagreement_triggers_one_sync_arbiter_call_per_disputed_run(tmp_path):
    name_a, name_b = PRIMARY_EXTRACTORS
    run2_text = "Unlike Vapi, which needs setup, it works."
    # run-1: only primary A names Goodcall; run-2: only primary A names Vapi.
    mentions_a = {
        "run-1": [MentionCandidate("Goodcall", "Goodcall is a strong choice.", True)],
        "run-2": [MentionCandidate("Vapi", "Unlike Vapi, which needs setup,", False)],
    }
    mentions_b: dict = {"run-1": [], "run-2": []}
    batch_extractors = {name_a: _fake_batch_extractor(mentions_a), name_b: _fake_batch_extractor(mentions_b)}
    seen: list = []
    arbiter_call_fn = _fake_arbiter_call_fn(
        {
            RAW_TEXT: [ArbiterVote("Goodcall", confirmed=True, is_recommended=True)],
            run2_text: [ArbiterVote("Vapi", confirmed=True, is_recommended=False)],
        },
        seen=seen,
    )
    cache = LLMCache(cache_dir=tmp_path / "cache")
    conn = _fresh_conn(tmp_path)
    _seed_run(conn, "run-1")
    _seed_run(conn, "run-2")

    stats = run_a1(
        [
            {"run_id": "run-1", "raw_response_text": RAW_TEXT},
            {"run_id": "run-2", "raw_response_text": run2_text},
        ],
        cache=cache, use_batch_api=True, batch_extractors=batch_extractors,
        arbiter_call_fn=arbiter_call_fn, conn=conn,
    )

    assert set(seen) == {RAW_TEXT, run2_text}  # one arbiter call per disputed run
    assert stats.arbiter_calls_made == 2
    rows = storage.list_mentions_raw(conn=conn)
    assert {r["run_id"]: r["votes"] for r in rows} == {
        "run-1": [name_a, ARBITER_EXTRACTOR], "run-2": [name_a, ARBITER_EXTRACTOR],
    }
