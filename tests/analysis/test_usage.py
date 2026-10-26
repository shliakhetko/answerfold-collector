"""Token-count and search-fired extraction, one case per engine. Response
shapes below are lifted from real pilot output so expected values match what
a real response actually contains."""
from analysis.aggregate.usage import extract_extra_usage_fields, extract_search_fired, extract_usage


def test_openai_chatgpt():
    response = {
        "usage": {
            "input_tokens": 314,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 266,
            "output_tokens_details": {"reasoning_tokens": 592},
            "total_tokens": 580,
        }
    }
    usage = extract_usage("openai_chatgpt", response)
    assert (usage.input, usage.output, usage.total) == (314, 266, 580)
    assert usage.reasoning_tokens == 592
    # Not broken out by this vendor -- never guessed.
    assert usage.tool_tokens is None


def test_google_gemini():
    response = {
        "usage_metadata": {
            "cache_tokens_details": None,
            "cached_content_token_count": None,
            "candidates_token_count": 1172,
            "prompt_token_count": 1046,
            "thoughts_token_count": 1599,
            "tool_use_prompt_token_count": 210,
            "total_token_count": 3817,
        }
    }
    usage = extract_usage("google_gemini", response)
    assert (usage.input, usage.output, usage.total) == (1046, 1172, 3817)
    assert usage.reasoning_tokens == 1599
    assert usage.tool_tokens == 210


def test_anthropic_claude():
    response = {
        "usage": {
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_tokens": 21814,
            "output_tokens": 1877,
            "output_tokens_details": {"thinking_tokens": 206},
            "server_tool_use": {"web_search_requests": 2},
        }
    }
    usage = extract_usage("anthropic_claude", response)
    # No vendor-reported total -- computed as input + output only, excluding
    # cache tokens (see analysis/aggregate/usage.py's comment on why).
    assert (usage.input, usage.output, usage.total) == (21814, 1877, 23691)
    assert usage.reasoning_tokens == 206
    # server_tool_use.web_search_requests is a search *count*, not a token
    # count -- Anthropic exposes no tool-token breakdown.
    assert usage.tool_tokens is None


def test_google_ai_mode_has_no_token_field():
    response = {"search_metadata": {"status": "Success"}, "text_blocks": []}
    usage = extract_usage("google_ai_mode", response)
    assert (usage.input, usage.output, usage.total, usage.reasoning_tokens, usage.tool_tokens) == (
        None,
        None,
        None,
        None,
        None,
    )


def test_missing_response_does_not_raise():
    assert extract_usage("openai_chatgpt", None) == extract_usage("openai_chatgpt", {})


def test_unknown_engine_returns_all_none():
    usage = extract_usage("some_future_engine", {"usage": {"input_tokens": 1}})
    assert (usage.input, usage.output, usage.total, usage.reasoning_tokens, usage.tool_tokens) == (
        None,
        None,
        None,
        None,
        None,
    )


def test_search_fired_openai_true():
    response = {"output": [{"type": "web_search_call"}, {"type": "message"}]}
    assert extract_search_fired("openai_chatgpt", response) is True


def test_search_fired_openai_false():
    response = {"output": [{"type": "message"}]}
    assert extract_search_fired("openai_chatgpt", response) is False


def test_search_fired_gemini_true():
    response = {
        "candidates": [
            {"grounding_metadata": {"web_search_queries": ["best CRM for a 5-person startup team"]}}
        ]
    }
    assert extract_search_fired("google_gemini", response) is True


def test_search_fired_gemini_false():
    response = {"candidates": [{"grounding_metadata": None}]}
    assert extract_search_fired("google_gemini", response) is False


def test_search_fired_anthropic_true():
    response = {"content": [{"type": "server_tool_use", "name": "web_search"}]}
    assert extract_search_fired("anthropic_claude", response) is True


def test_search_fired_anthropic_false():
    response = {"content": [{"type": "text", "text": "..."}]}
    assert extract_search_fired("anthropic_claude", response) is False


def test_search_fired_google_ai_mode_always_true():
    response = {"search_metadata": {"status": "Success"}}
    assert extract_search_fired("google_ai_mode", response) is True


def test_search_fired_none_when_no_response():
    # "unknown" (missing response, e.g. an errored run) must never collapse
    # into "did not search" -- same tri-state rule as AdapterResult.search_fired.
    assert extract_search_fired("openai_chatgpt", None) is None


def test_search_fired_none_for_unknown_engine():
    assert extract_search_fired("some_future_engine", {"output": []}) is None


def test_extra_usage_fields_openai():
    response = {
        "usage": {
            "input_tokens": 314,
            "input_tokens_details": {"cache_write_tokens": 4412, "cached_tokens": 100},
            "output_tokens": 266,
            "output_tokens_details": {"reasoning_tokens": 592},
        }
    }
    assert extract_extra_usage_fields("openai_chatgpt", response) == {
        "cached_tokens": 100,
        "cache_write_tokens": 4412,
    }


def test_extra_usage_fields_gemini():
    response = {"usage_metadata": {"cached_content_token_count": 50, "prompt_token_count": 16}}
    assert extract_extra_usage_fields("google_gemini", response) == {"cached_content_tokens": 50}


def test_extra_usage_fields_anthropic():
    response = {
        "usage": {
            "input_tokens": 21814,
            "output_tokens": 1877,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 500,
            "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0},
        }
    }
    # cache_creation_input_tokens=0 and web_fetch_requests=0 are real
    # reported zeros, not missing -- they must survive (only None is
    # dropped, see extract_extra_usage_fields's docstring).
    assert extract_extra_usage_fields("anthropic_claude", response) == {
        "cache_creation_tokens": 0,
        "cache_read_tokens": 500,
        "web_search_requests": 2,
        "web_fetch_requests": 0,
    }


def test_extra_usage_fields_google_ai_mode_always_empty():
    assert extract_extra_usage_fields("google_ai_mode", {"search_metadata": {"status": "Success"}}) == {}


def test_extra_usage_fields_missing_response():
    assert extract_extra_usage_fields("openai_chatgpt", None) == {}
