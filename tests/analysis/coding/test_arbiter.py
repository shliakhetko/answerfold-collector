import json

import pytest

from analysis.coding.extractors.arbiter import (
    ArbiterBatchItem,
    ArbiterResult,
    ArbiterVote,
    call_arbiter_vote,
    fetch_arbiter_batch_results,
    poll_arbiter_batch,
    submit_arbiter_batch,
)
from collector.config import AnthropicConfig
from tests.fakes import (
    FakeAnthropicBatchClient,
    FakeAnthropicBatches,
    FakeAnthropicClient,
    FakeMessages,
    fake_anthropic_batch_error_entry,
    fake_anthropic_batch_result_entry,
    fake_arbiter_response,
)

RAW_TEXT = "Goodcall is a strong choice. Unlike Vapi, which needs setup, Goodcall works out of the box."


def test_call_arbiter_vote_happy_path():
    votes_json = json.dumps(
        [{"surface_form": "Vapi", "confirmed": True, "is_recommended": False}]
    )
    client = FakeAnthropicClient(FakeMessages(response=fake_arbiter_response(votes_json)))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_arbiter_vote(
        raw_response_text=RAW_TEXT,
        candidates=[{"surface_form": "Vapi", "evidence_span": "Unlike Vapi, which needs setup,"}],
        config=cfg,
        client=client,
    )

    assert result.error is None
    assert result.votes == [ArbiterVote("Vapi", confirmed=True, is_recommended=False)]
    assert result.tokens_input == 150


def test_call_arbiter_vote_rejects_a_candidate():
    votes_json = json.dumps(
        [{"surface_form": "NotACompany", "confirmed": False, "is_recommended": False}]
    )
    client = FakeAnthropicClient(FakeMessages(response=fake_arbiter_response(votes_json)))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_arbiter_vote(
        raw_response_text=RAW_TEXT,
        candidates=[{"surface_form": "NotACompany", "evidence_span": "NotACompany was mentioned."}],
        config=cfg,
        client=client,
    )

    assert result.votes == [ArbiterVote("NotACompany", confirmed=False, is_recommended=False)]


def test_call_arbiter_vote_transport_failure_produces_error_not_exception():
    client = FakeAnthropicClient(FakeMessages(exc=TimeoutError("simulated timeout")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_arbiter_vote(
        raw_response_text=RAW_TEXT,
        candidates=[{"surface_form": "Vapi", "evidence_span": "Unlike Vapi, which needs setup,"}],
        config=cfg,
        client=client,
    )

    assert result.error == "TimeoutError: simulated timeout"
    assert result.votes == []


def test_call_arbiter_vote_empty_candidates_never_calls_the_model():
    client = FakeAnthropicClient(FakeMessages(exc=AssertionError("should never be called")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_arbiter_vote(raw_response_text=RAW_TEXT, candidates=[], config=cfg, client=client)

    assert result == ArbiterResult()


# --------------------------------------------------------------------------
# Arbiter batch (submit/poll/fetch) -- one Anthropic Message Batch covering
# every run's disputed set for the whole pipeline run.
# --------------------------------------------------------------------------

def test_submit_arbiter_batch_builds_one_request_per_run():
    batches = FakeAnthropicBatches(batch_id="batch_arb_1")
    client = FakeAnthropicBatchClient(batches)
    items = [
        ArbiterBatchItem("run-1", RAW_TEXT, [{"surface_form": "Vapi", "evidence_span": "Unlike Vapi, which needs setup,"}]),
        ArbiterBatchItem("run-2", "Some other text.", [{"surface_form": "Acme", "evidence_span": "Acme was named."}]),
    ]

    batch_id = submit_arbiter_batch(items, config=AnthropicConfig(api_key="sk-ant-test"), client=client)

    assert batch_id == "batch_arb_1"
    requests = batches.create_kwargs["requests"]
    assert [r["custom_id"] for r in requests] == ["run-1", "run-2"]
    assert "Vapi" in requests[0]["params"]["messages"][0]["content"]


def test_poll_arbiter_batch_polls_until_terminal_status():
    batches = FakeAnthropicBatches(statuses=("in_progress", "ended"))
    client = FakeAnthropicBatchClient(batches)

    status = poll_arbiter_batch(
        "batch_arb_1", config=AnthropicConfig(api_key="sk-ant-test"), client=client,
        poll_interval_seconds=0, max_wait_seconds=10,
    )

    assert status == "ended"


def test_poll_arbiter_batch_raises_timeout_error_rather_than_hanging():
    batches = FakeAnthropicBatches(statuses=("in_progress",))
    client = FakeAnthropicBatchClient(batches)

    with pytest.raises(TimeoutError):
        poll_arbiter_batch(
            "batch_arb_1", config=AnthropicConfig(api_key="sk-ant-test"), client=client,
            poll_interval_seconds=0, max_wait_seconds=0,
        )


def test_fetch_arbiter_batch_results_parses_succeeded_and_errored_entries():
    votes_json = json.dumps([{"surface_form": "Vapi", "confirmed": True, "is_recommended": False}])
    batches = FakeAnthropicBatches(results=[
        fake_anthropic_batch_result_entry(custom_id="run-1", text_json=votes_json),
        fake_anthropic_batch_error_entry(custom_id="run-2", result_type="expired"),
    ])
    client = FakeAnthropicBatchClient(batches)

    results = fetch_arbiter_batch_results("batch_arb_1", config=AnthropicConfig(api_key="sk-ant-test"), client=client)

    assert results["run-1"].error is None
    assert results["run-1"].votes == [ArbiterVote("Vapi", confirmed=True, is_recommended=False)]
    assert results["run-2"].error == "batch_result_expired"
