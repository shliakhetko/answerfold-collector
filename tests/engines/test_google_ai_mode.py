from collector.config import SerpApiConfig
from collector.engines.google_ai_mode import call_google_ai_mode
from tests.fakes import FakeSerpApiClient, fake_google_ai_mode_success_response


def test_happy_path_parses_fields_and_sources():
    client = FakeSerpApiClient(response=fake_google_ai_mode_success_response())
    cfg = SerpApiConfig(api_key="serp-test")

    result = call_google_ai_mode(prompt_text="capital of France?", model_requested="", config=cfg, client=client)

    assert result.error is None
    assert result.model_returned is None
    assert result.raw_response_text == "Paris is the capital of France."
    assert result.search_fired is True
    assert result.fan_out_queries is None
    assert result.api_cost_or_tokens is None
    assert result.finish_reason == "Success"
    assert result.scraper_vendor == "serpapi"
    assert result.scraper_endpoint == "google_ai_mode"
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.position == 1
    assert source.url_raw == "https://en.wikipedia.org/wiki/Paris"
    assert source.domain == "en.wikipedia.org"
    assert source.title == "Paris - Wikipedia"
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.request_payload == {
        "engine": "google_ai_mode",
        "q": "capital of France?",
        "gl": "us",
        "hl": "en",
        "no_cache": True,
    }


def test_request_payload_never_leaks_injected_api_key():
    """Regression test: serpapi.Client.request() mutates the params dict it
    receives in place, injecting the real api_key -- the adapter must pass a
    copy to .search(), never the dict it also stores/logs as
    request_payload, or the live secret leaks into the JSONL line."""
    client = FakeSerpApiClient(response=fake_google_ai_mode_success_response())
    cfg = SerpApiConfig(api_key="serp-real-secret")

    result = call_google_ai_mode(prompt_text="q", model_requested="", config=cfg, client=client)

    assert "api_key" not in result.request_payload
    assert result.request_payload == {
        "engine": "google_ai_mode",
        "q": "q",
        "gl": "us",
        "hl": "en",
        "no_cache": True,
    }


def test_transport_failure_produces_error_result_not_exception():
    client = FakeSerpApiClient(exc=TimeoutError("simulated timeout"))
    cfg = SerpApiConfig(api_key="serp-test")

    result = call_google_ai_mode(prompt_text="q", model_requested="", config=cfg, client=client)

    assert result.error == "TimeoutError: simulated timeout"
    assert result.raw_response_text is None
    assert result.sources == []
    assert result.latency_ms is not None
    # Unlike model_returned, the scraper identity is known even on failure --
    # it's a fact about which adapter ran, not something parsed from a response.
    assert result.scraper_vendor == "serpapi"
    assert result.scraper_endpoint == "google_ai_mode"


def test_no_results_still_search_fired_true():
    # There's no tri-state "did search fire" axis for this engine (unlike the
    # API-backed adapters): Google AI Mode is inherently a search product, so
    # a successful-but-empty response still reports search_fired=True.
    response = fake_google_ai_mode_success_response()
    response["references"] = []
    response["reconstructed_markdown"] = ""
    response["text_blocks"] = []
    client = FakeSerpApiClient(response=response)
    cfg = SerpApiConfig(api_key="serp-test")

    result = call_google_ai_mode(prompt_text="q", model_requested="", config=cfg, client=client)

    assert result.error is None
    assert result.search_fired is True
    assert result.sources == []
    assert result.raw_response_text is None


def test_200_response_with_error_status_populates_error():
    response = {
        "search_metadata": {"status": "Error"},
        "error": "Google hasn't returned any results for this query.",
    }
    client = FakeSerpApiClient(response=response)
    cfg = SerpApiConfig(api_key="serp-test")

    result = call_google_ai_mode(prompt_text="q", model_requested="", config=cfg, client=client)

    assert result.error == "Google hasn't returned any results for this query."
    assert result.raw_response_text is None
    assert result.sources == []
    assert result.search_fired is None
    assert result.raw_response_envelope is not None
