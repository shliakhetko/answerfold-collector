import json

from collector.config import OpenAIConfig
from collector.engines.openai_chatgpt import call_chatgpt
from tests.fakes import FakeClient, FakeResponses, fake_success_response


def test_happy_path_parses_fields_and_sources():
    client = FakeClient(FakeResponses(response=fake_success_response()))
    cfg = OpenAIConfig(api_key="sk-test")

    result = call_chatgpt(prompt_text="capital of France?", model_requested="gpt-4o", config=cfg, client=client)

    assert result.error is None
    assert result.model_returned == "gpt-4o-search-preview"
    assert result.raw_response_text == "Paris is the capital of France."
    assert result.search_fired is True
    assert json.loads(result.fan_out_queries) == ["capital of France"]
    assert result.finish_reason == "completed"
    assert json.loads(result.api_cost_or_tokens) == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }
    assert result.raw_response_envelope == '{"id": "resp_fake", "status": "completed"}'
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.position == 1
    assert source.url_raw == "https://en.wikipedia.org/wiki/Paris"
    assert source.domain == "en.wikipedia.org"
    assert source.title == "Paris - Wikipedia"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.request_payload == {
        "model": "gpt-4o",
        "input": "capital of France?",
        "tools": [
            {
                "type": "web_search",
                "user_location": {"type": "approximate", "country": "US"},
            }
        ],
    }


def test_transport_failure_produces_error_result_not_exception():
    client = FakeClient(FakeResponses(exc=TimeoutError("simulated timeout")))
    cfg = OpenAIConfig(api_key="sk-test")

    result = call_chatgpt(prompt_text="q", model_requested="gpt-4o", config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.raw_response_text is None
    assert result.search_fired is None
    assert result.sources == []
    assert result.latency_ms is not None


def test_no_search_call_reports_search_fired_false():
    response = fake_success_response()
    response.output = [item for item in response.output if item.type != "web_search_call"]
    client = FakeClient(FakeResponses(response=response))
    cfg = OpenAIConfig(api_key="sk-test")

    result = call_chatgpt(prompt_text="q", model_requested="gpt-4o", config=cfg, client=client)

    assert result.error is None
    assert result.search_fired is False
    assert result.fan_out_queries is None
