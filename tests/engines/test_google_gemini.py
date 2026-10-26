import json

from collector.config import GeminiConfig
from collector.engines.google_gemini import call_gemini
from tests.fakes import FakeGeminiClient, FakeGeminiModels, fake_gemini_success_response


def test_happy_path_parses_fields_and_sources():
    client = FakeGeminiClient(FakeGeminiModels(response=fake_gemini_success_response()))
    cfg = GeminiConfig(api_key="gm-test")

    result = call_gemini(prompt_text="capital of France?", model_requested="gemini-2.5-flash", config=cfg, client=client)

    assert result.error is None
    assert result.model_returned == "gemini-2.5-flash"
    assert result.raw_response_text == "Paris is the capital of France."
    assert result.search_fired is True
    assert json.loads(result.fan_out_queries) == ["capital of France"]
    assert result.finish_reason == "STOP"
    assert json.loads(result.api_cost_or_tokens) == {
        "prompt_token_count": 12,
        "candidates_token_count": 8,
        "total_token_count": 20,
    }
    assert result.raw_response_envelope == '{"model_version": "gemini-2.5-flash"}'
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.position == 1
    assert source.url_raw == "https://en.wikipedia.org/wiki/Paris"
    assert source.domain == "en.wikipedia.org"
    assert source.title == "Paris - Wikipedia"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.request_payload == {
        "model": "gemini-2.5-flash",
        "contents": "capital of France?",
        "tools": [{"google_search": {}}],
        "tool_config": {
            "retrieval_config": {
                "lat_lng": {"latitude": 39.8283, "longitude": -98.5795},
                "language_code": "en",
            }
        },
    }


def test_transport_failure_produces_error_result_not_exception():
    client = FakeGeminiClient(FakeGeminiModels(exc=TimeoutError("simulated timeout")))
    cfg = GeminiConfig(api_key="gm-test")

    result = call_gemini(prompt_text="q", model_requested="gemini-2.5-flash", config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.raw_response_text is None
    assert result.search_fired is None
    assert result.sources == []
    assert result.latency_ms is not None


def test_no_grounding_metadata_reports_search_fired_false():
    response = fake_gemini_success_response()
    response.candidates[0].grounding_metadata = None
    client = FakeGeminiClient(FakeGeminiModels(response=response))
    cfg = GeminiConfig(api_key="gm-test")

    result = call_gemini(prompt_text="q", model_requested="gemini-2.5-flash", config=cfg, client=client)

    assert result.error is None
    assert result.search_fired is False
    assert result.fan_out_queries is None
    assert result.sources == []
