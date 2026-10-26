import json

from collector.config import AnthropicConfig
from collector.engines.anthropic_claude import call_claude
from tests.fakes import FakeAnthropicClient, FakeMessages, fake_claude_success_response


def test_happy_path_parses_fields_and_sources():
    client = FakeAnthropicClient(FakeMessages(response=fake_claude_success_response()))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_claude(prompt_text="capital of France?", model_requested="claude-opus-5", config=cfg, client=client)

    assert result.error is None
    assert result.model_returned == "claude-opus-5"
    assert result.raw_response_text == "Paris is the capital of France."
    assert result.search_fired is True
    assert json.loads(result.fan_out_queries) == ["capital of France"]
    assert result.finish_reason == "end_turn"
    assert json.loads(result.api_cost_or_tokens) == {
        "input_tokens": 12,
        "output_tokens": 8,
        "server_tool_use": {"web_search_requests": 1},
    }
    assert result.raw_response_envelope == '{"id": "msg_fake", "stop_reason": "end_turn"}'
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.position == 1
    assert source.url_raw == "https://en.wikipedia.org/wiki/Paris"
    assert source.domain == "en.wikipedia.org"
    assert source.title == "Paris - Wikipedia"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.request_payload == {
        "model": "claude-opus-5",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "capital of France?"}],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "user_location": {"type": "approximate", "country": "US"},
            }
        ],
    }


def test_transport_failure_produces_error_result_not_exception():
    client = FakeAnthropicClient(FakeMessages(exc=TimeoutError("simulated timeout")))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_claude(prompt_text="q", model_requested="claude-opus-5", config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.raw_response_text is None
    assert result.search_fired is None
    assert result.sources == []
    assert result.latency_ms is not None


def test_no_search_call_reports_search_fired_false():
    response = fake_claude_success_response()
    response.content = [block for block in response.content if block.type == "text"]
    response.content[0].citations = []
    client = FakeAnthropicClient(FakeMessages(response=response))
    cfg = AnthropicConfig(api_key="sk-ant-test")

    result = call_claude(prompt_text="q", model_requested="claude-opus-5", config=cfg, client=client)

    assert result.error is None
    assert result.search_fired is False
    assert result.fan_out_queries is None
    assert result.sources == []


def test_web_search_max_uses_omitted_by_default():
    """No invented cap: Anthropic's own guidance says comparative questions
    (this study's subject) can need 10+ searches, and the consumer surface
    applies no cap either -- see collector/config.py's
    DEFAULT_ANTHROPIC_WEB_SEARCH_MAX_USES."""
    messages = FakeMessages(response=fake_claude_success_response())
    client = FakeAnthropicClient(messages)
    cfg = AnthropicConfig(api_key="sk-ant-test")

    call_claude(prompt_text="q", model_requested="claude-opus-5", config=cfg, client=client)

    tool = messages.last_kwargs["tools"][0]
    assert "max_uses" not in tool


def test_web_search_max_uses_sent_when_explicitly_configured():
    messages = FakeMessages(response=fake_claude_success_response())
    client = FakeAnthropicClient(messages)
    cfg = AnthropicConfig(api_key="sk-ant-test", web_search_max_uses=3)

    call_claude(prompt_text="q", model_requested="claude-opus-5", config=cfg, client=client)

    tool = messages.last_kwargs["tools"][0]
    assert tool["max_uses"] == 3
