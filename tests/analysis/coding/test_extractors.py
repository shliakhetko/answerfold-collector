import json
from types import SimpleNamespace

import pytest

from analysis.coding.extractors.anthropic_extractor import (
    call_anthropic_extractor,
    fetch_anthropic_batch_results,
    poll_anthropic_batch,
    submit_anthropic_batch,
)
from analysis.coding.extractors.arbiter import ArbiterVote
from analysis.coding.extractors.base import BatchItem, MentionCandidate, parse_mentions_json
from analysis.coding.extractors.gemini_extractor import (
    call_gemini_extractor,
    fetch_gemini_batch_results,
    poll_gemini_batch,
    submit_gemini_batch,
)
from analysis.coding.extractors.openai_extractor import (
    call_openai_arbiter_vote,
    call_openai_extractor,
    fetch_openai_batch_results,
    poll_openai_batch,
    submit_openai_batch,
)
from collector.config import AnthropicConfig, GeminiConfig, OpenAIConfig
from tests.fakes import (
    FakeAnthropicBatchClient,
    FakeAnthropicBatches,
    FakeAnthropicClient,
    FakeClient,
    FakeGeminiBatchClient,
    FakeGeminiBatches,
    FakeGeminiClient,
    FakeGeminiModels,
    FakeMessages,
    FakeOpenAIBatchClient,
    FakeOpenAIBatches,
    FakeOpenAIUploads,
    FakeOpenAIFiles,
    FakeResponses,
    fake_anthropic_batch_error_entry,
    fake_anthropic_batch_result_entry,
    fake_anthropic_extraction_response,
    fake_gemini_extraction_response,
    fake_openai_extraction_response,
)

MENTIONS_JSON = json.dumps(
    [
        {"surface_form": "Goodcall", "evidence_span": "Goodcall is a strong choice.", "is_recommended": True},
        {"surface_form": "Vapi", "evidence_span": "Unlike Vapi, which needs setup", "is_recommended": False},
    ]
)


# --------------------------------------------------------------------------
# parse_mentions_json
# --------------------------------------------------------------------------

def test_parse_mentions_json_plain():
    mentions = parse_mentions_json(MENTIONS_JSON)
    assert mentions == [
        MentionCandidate("Goodcall", "Goodcall is a strong choice.", True),
        MentionCandidate("Vapi", "Unlike Vapi, which needs setup", False),
    ]


def test_parse_mentions_json_strips_markdown_code_fence():
    fenced = f"```json\n{MENTIONS_JSON}\n```"
    assert parse_mentions_json(fenced) == parse_mentions_json(MENTIONS_JSON)


def test_parse_mentions_json_empty_array():
    assert parse_mentions_json("[]") == []


def test_parse_mentions_json_repairs_invalid_backslash_escape():
    # gpt/haiku sometimes leave a raw backslash inside an evidence_span.
    bad = '[{"surface_form": "Foo", "evidence_span": "priced at \\$5 \\(flat\\)", "is_recommended": true}]'
    assert parse_mentions_json(bad) == [MentionCandidate("Foo", "priced at \\$5 \\(flat\\)", True)]


def test_parse_mentions_json_salvages_truncated_output():
    # Output cut off at max_tokens: 2 complete objects, then an unterminated
    # string in the third and no closing bracket.
    truncated = (
        '```json\n[\n'
        '  {"surface_form": "Goodcall", "evidence_span": "Goodcall is great.", "is_recommended": true},\n'
        '  {"surface_form": "Vapi", "evidence_span": "Vapi needs setup", "is_recommended": false},\n'
        '  {"surface_form": "Retell", "evidence_span": "Retell AI is a devel'
    )
    assert parse_mentions_json(truncated) == [
        MentionCandidate("Goodcall", "Goodcall is great.", True),
        MentionCandidate("Vapi", "Vapi needs setup", False),
    ]


def test_parse_mentions_json_raises_when_nothing_salvageable():
    with pytest.raises(json.JSONDecodeError):
        parse_mentions_json("not json at all {")


# --------------------------------------------------------------------------
# Anthropic extractor
# --------------------------------------------------------------------------

def test_anthropic_extractor_happy_path():
    client = FakeAnthropicClient(FakeMessages(response=fake_anthropic_extraction_response(MENTIONS_JSON)))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_anthropic_extractor(raw_response_text="...", config=cfg, client=client)

    assert result.error is None
    assert result.model_returned == "claude-haiku-4-5-20251001"
    assert result.mentions == [
        MentionCandidate("Goodcall", "Goodcall is a strong choice.", True),
        MentionCandidate("Vapi", "Unlike Vapi, which needs setup", False),
    ]
    assert result.tokens_input == 100
    assert result.tokens_output == 40


def test_anthropic_extractor_transport_failure_produces_error_not_exception():
    client = FakeAnthropicClient(FakeMessages(exc=TimeoutError("simulated timeout")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_anthropic_extractor(raw_response_text="...", config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.mentions == []


def test_anthropic_extractor_malformed_json_degrades_gracefully():
    client = FakeAnthropicClient(FakeMessages(response=fake_anthropic_extraction_response("not valid json")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_anthropic_extractor(raw_response_text="...", config=cfg, client=client)

    assert result.mentions == []
    assert result.error is not None
    # model_returned/tokens still land -- one malformed field never blanks
    # the whole result (same discipline as collector/engines/*.py's _safe()).
    assert result.model_returned == "claude-haiku-4-5-20251001"
    assert result.tokens_input == 100


def test_anthropic_extractor_sends_temperature_zero():
    messages = FakeMessages(response=fake_anthropic_extraction_response("[]"))
    client = FakeAnthropicClient(messages)
    cfg = AnthropicConfig(api_key="sk-ant-test")

    call_anthropic_extractor(raw_response_text="...", config=cfg, client=client)

    assert messages.last_kwargs["temperature"] == 0


# --------------------------------------------------------------------------
# OpenAI extractor
# --------------------------------------------------------------------------

def test_openai_extractor_happy_path():
    client = FakeClient(FakeResponses(response=fake_openai_extraction_response(MENTIONS_JSON)))
    cfg = OpenAIConfig(api_key="sk-test")

    result = call_openai_extractor(raw_response_text="...", config=cfg, client=client)

    assert result.error is None
    assert result.model_returned == "gpt-5-mini"
    assert result.mentions == [
        MentionCandidate("Goodcall", "Goodcall is a strong choice.", True),
        MentionCandidate("Vapi", "Unlike Vapi, which needs setup", False),
    ]
    assert result.tokens_input == 100
    assert result.tokens_output == 40


def test_openai_extractor_transport_failure_produces_error_not_exception():
    client = FakeClient(FakeResponses(exc=TimeoutError("simulated timeout")))
    cfg = OpenAIConfig(api_key="sk-test")

    result = call_openai_extractor(raw_response_text="...", config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.mentions == []


def test_call_openai_arbiter_vote_parses_votes():
    votes_json = '[{"surface_form": "Goodcall", "confirmed": true, "is_recommended": false}]'
    client = FakeClient(FakeResponses(response=fake_openai_extraction_response(votes_json)))
    result = call_openai_arbiter_vote(
        raw_response_text="Goodcall is great.",
        candidates=[{"surface_form": "Goodcall", "evidence_span": "Goodcall is great."}],
        config=OpenAIConfig(api_key="sk-test"), client=client,
    )
    assert result.error is None
    assert result.votes == [ArbiterVote(surface_form="Goodcall", confirmed=True, is_recommended=False)]


def test_call_openai_arbiter_vote_empty_candidates_never_calls_model():
    client = FakeClient(FakeResponses(exc=AssertionError("must not call the model")))
    result = call_openai_arbiter_vote(
        raw_response_text="x", candidates=[], config=OpenAIConfig(api_key="sk-test"), client=client,
    )
    assert result.votes == [] and result.error is None


# --------------------------------------------------------------------------
# Gemini extractor
# --------------------------------------------------------------------------

def test_gemini_extractor_happy_path():
    client = FakeGeminiClient(FakeGeminiModels(response=fake_gemini_extraction_response(MENTIONS_JSON)))
    cfg = GeminiConfig(api_key="sk-test")

    result = call_gemini_extractor(raw_response_text="...", config=cfg, client=client)

    assert result.error is None
    assert result.model_returned == "gemini-flash-lite-latest"
    assert result.mentions == [
        MentionCandidate("Goodcall", "Goodcall is a strong choice.", True),
        MentionCandidate("Vapi", "Unlike Vapi, which needs setup", False),
    ]
    assert result.tokens_input == 100
    assert result.tokens_output == 40


def test_gemini_extractor_transport_failure_produces_error_not_exception():
    client = FakeGeminiClient(FakeGeminiModels(exc=TimeoutError("simulated timeout")))
    cfg = GeminiConfig(api_key="sk-test")

    result = call_gemini_extractor(raw_response_text="...", config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.mentions == []


# --------------------------------------------------------------------------
# Anthropic batch (submit/poll/fetch)
# --------------------------------------------------------------------------

def test_submit_anthropic_batch_builds_one_request_per_item():
    batches = FakeAnthropicBatches(batch_id="batch_123")
    client = FakeAnthropicBatchClient(batches)
    items = [BatchItem("run-1", "text one"), BatchItem("run-2", "text two")]

    batch_id = submit_anthropic_batch(items, config=AnthropicConfig(api_key="sk-ant-test"), client=client)

    assert batch_id == "batch_123"
    requests = batches.create_kwargs["requests"]
    assert [r["custom_id"] for r in requests] == ["run-1", "run-2"]
    assert requests[0]["params"]["temperature"] == 0
    assert "text one" in requests[0]["params"]["messages"][0]["content"]


def test_poll_anthropic_batch_polls_until_terminal_status():
    batches = FakeAnthropicBatches(statuses=("in_progress", "in_progress", "ended"))
    client = FakeAnthropicBatchClient(batches)

    status = poll_anthropic_batch(
        "batch_123", config=AnthropicConfig(api_key="sk-ant-test"), client=client,
        poll_interval_seconds=0, max_wait_seconds=10,
    )

    assert status == "ended"
    assert batches._retrieve_calls == 3


def test_poll_anthropic_batch_raises_timeout_error_rather_than_hanging():
    batches = FakeAnthropicBatches(statuses=("in_progress",))  # never terminal
    client = FakeAnthropicBatchClient(batches)

    with pytest.raises(TimeoutError):
        poll_anthropic_batch(
            "batch_123", config=AnthropicConfig(api_key="sk-ant-test"), client=client,
            poll_interval_seconds=0, max_wait_seconds=0,
        )


def test_fetch_anthropic_batch_results_parses_succeeded_and_errored_entries():
    batches = FakeAnthropicBatches(results=[
        fake_anthropic_batch_result_entry(custom_id="run-1", text_json=MENTIONS_JSON),
        fake_anthropic_batch_error_entry(custom_id="run-2", result_type="expired"),
    ])
    client = FakeAnthropicBatchClient(batches)

    results = fetch_anthropic_batch_results("batch_123", config=AnthropicConfig(api_key="sk-ant-test"), client=client)

    assert set(results) == {"run-1", "run-2"}
    assert results["run-1"].error is None
    assert results["run-1"].mentions == [
        MentionCandidate("Goodcall", "Goodcall is a strong choice.", True),
        MentionCandidate("Vapi", "Unlike Vapi, which needs setup", False),
    ]
    assert results["run-2"].error == "batch_result_expired"


# --------------------------------------------------------------------------
# OpenAI batch (submit/poll/fetch)
# --------------------------------------------------------------------------

def test_submit_openai_batch_uploads_jsonl_and_creates_batch():
    batches = FakeOpenAIBatches(batch_id="batch_456")
    files = FakeOpenAIFiles(file_id="file_789")
    client = FakeOpenAIBatchClient(batches, files)
    items = [BatchItem("run-1", "text one"), BatchItem("run-2", "text two")]

    batch_id = submit_openai_batch(items, config=OpenAIConfig(api_key="sk-test"), client=client)

    assert batch_id == "batch_456"
    uploaded_jsonl = files.create_kwargs["file"][1].decode("utf-8")
    lines = [json.loads(line) for line in uploaded_jsonl.splitlines()]
    assert [line["custom_id"] for line in lines] == ["run-1", "run-2"]
    assert lines[0]["url"] == "/v1/responses"
    assert "text one" in lines[0]["body"]["input"]
    assert batches.create_kwargs["input_file_id"] == "file_789"


def test_submit_openai_batch_uses_chunked_upload_when_available():
    batches = FakeOpenAIBatches(batch_id="batch_456")
    files = FakeOpenAIFiles(file_id="file_should_not_be_used")
    uploads = FakeOpenAIUploads(file_id="file_chunked")
    client = FakeOpenAIBatchClient(batches, files, uploads=uploads)

    submit_openai_batch(
        [BatchItem("run-1", "text one")], config=OpenAIConfig(api_key="sk-test"), client=client,
    )

    assert files.create_kwargs is None  # single-shot upload path not taken
    assert uploads.chunked_kwargs["purpose"] == "batch"
    assert uploads.chunked_kwargs["bytes"] == len(uploads.chunked_kwargs["file"])
    assert batches.create_kwargs["input_file_id"] == "file_chunked"


def test_submit_openai_batch_waits_for_file_processing_before_creating_batch(monkeypatch):
    monkeypatch.setattr("analysis.coding.extractors.openai_extractor.time.sleep", lambda s: None)
    calls = {"retrieve": 0, "created_at": None}

    class _SlowFiles(FakeOpenAIFiles):
        def retrieve(self, file_id):
            calls["retrieve"] += 1
            status = "pending" if calls["retrieve"] < 3 else "processed"
            return SimpleNamespace(id=file_id, status=status)

    class _RecordingBatches(FakeOpenAIBatches):
        def create(self, **kwargs):
            calls["created_at"] = calls["retrieve"]
            return super().create(**kwargs)

    files = _SlowFiles(file_id="file_789")
    client = FakeOpenAIBatchClient(_RecordingBatches(batch_id="batch_456"), files)

    submit_openai_batch(
        [BatchItem("run-1", "t")], config=OpenAIConfig(api_key="sk-test"), client=client,
    )

    # batch was not created until the file left "pending"
    assert calls["retrieve"] >= 3
    assert calls["created_at"] == 3


def test_poll_openai_batch_polls_until_terminal_status():
    batches = FakeOpenAIBatches(statuses=("validating", "in_progress", "completed"))
    files = FakeOpenAIFiles()
    client = FakeOpenAIBatchClient(batches, files)

    status = poll_openai_batch(
        "batch_456", config=OpenAIConfig(api_key="sk-test"), client=client,
        poll_interval_seconds=0, max_wait_seconds=10,
    )

    assert status == "completed"


def test_fetch_openai_batch_results_parses_output_and_error_files():
    output_line = json.dumps({
        "custom_id": "run-1",
        "response": {"body": {
            "model": "gpt-5-mini-2025-08-07",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": MENTIONS_JSON}]}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }},
    })
    error_line = json.dumps({"custom_id": "run-2", "error": {"message": "rate limited"}})
    batches = FakeOpenAIBatches(output_file_id="out_1", error_file_id="err_1")
    files = FakeOpenAIFiles(content_by_id={"out_1": output_line, "err_1": error_line})
    client = FakeOpenAIBatchClient(batches, files)

    results = fetch_openai_batch_results("batch_456", config=OpenAIConfig(api_key="sk-test"), client=client)

    assert results["run-1"].error is None
    assert results["run-1"].tokens_input == 50
    assert results["run-1"].mentions == [
        MentionCandidate("Goodcall", "Goodcall is a strong choice.", True),
        MentionCandidate("Vapi", "Unlike Vapi, which needs setup", False),
    ]
    assert "rate limited" in results["run-2"].error


# --------------------------------------------------------------------------
# Gemini batch (submit/poll/fetch)
# --------------------------------------------------------------------------

def test_submit_gemini_batch_builds_inline_requests_and_returns_job_name():
    batches = FakeGeminiBatches(job_name="batches/fake-1")
    client = FakeGeminiBatchClient(batches)
    items = [BatchItem("run-1", "text one"), BatchItem("run-2", "text two")]

    job_name = submit_gemini_batch(items, config=GeminiConfig(api_key="sk-test"), client=client)

    assert job_name == "batches/fake-1"
    src = batches.create_kwargs["src"]
    assert len(src) == 2
    assert "text one" in src[0]["contents"][0]["parts"][0]["text"]


def test_poll_gemini_batch_polls_until_terminal_state():
    batches = FakeGeminiBatches(states=("JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"))
    client = FakeGeminiBatchClient(batches)

    state = poll_gemini_batch(
        "batches/fake-1", config=GeminiConfig(api_key="sk-test"), client=client,
        poll_interval_seconds=0, max_wait_seconds=10,
    )

    assert state == "JOB_STATE_SUCCEEDED"


def test_fetch_gemini_batch_results_zips_positionally():
    ok_response = fake_gemini_extraction_response(MENTIONS_JSON)
    inlined = [
        SimpleNamespace(response=ok_response, error=None),
        SimpleNamespace(response=None, error="quota exceeded"),
    ]
    batches = FakeGeminiBatches(inlined_responses=inlined)
    client = FakeGeminiBatchClient(batches)
    items = [BatchItem("run-1", "text one"), BatchItem("run-2", "text two")]

    results = fetch_gemini_batch_results("batches/fake-1", items, config=GeminiConfig(api_key="sk-test"), client=client)

    assert results["run-1"].error is None
    assert results["run-1"].mentions == [
        MentionCandidate("Goodcall", "Goodcall is a strong choice.", True),
        MentionCandidate("Vapi", "Unlike Vapi, which needs setup", False),
    ]
    assert "quota exceeded" in results["run-2"].error


def test_fetch_gemini_batch_results_raises_on_count_mismatch():
    batches = FakeGeminiBatches(inlined_responses=[SimpleNamespace(response=None, error=None)])
    client = FakeGeminiBatchClient(batches)
    items = [BatchItem("run-1", "text one"), BatchItem("run-2", "text two")]  # 2 items, 1 result

    with pytest.raises(ValueError):
        fetch_gemini_batch_results("batches/fake-1", items, config=GeminiConfig(api_key="sk-test"), client=client)
