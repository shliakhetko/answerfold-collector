"""raw_response_text/model_returned/sources extraction, one case per engine.
Mirrors test_usage.py's style. See response_fields.py's module docstring for
the openai/gemini computed-property trap this suite guards against."""
from analysis.aggregate.response_fields import (
    extract_model_returned,
    extract_raw_response_text,
    extract_sources_rows,
)


def test_openai_chatgpt_text_and_model():
    response = {
        "model": "gpt-5.6-luna-2026-08-01",
        "output": [
            {"type": "web_search_call"},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Smith.ai is a popular choice."},
                ],
            },
        ],
    }
    assert extract_raw_response_text("openai_chatgpt", response) == "Smith.ai is a popular choice."
    assert extract_model_returned("openai_chatgpt", response) == "gpt-5.6-luna-2026-08-01"


def test_openai_chatgpt_never_reads_top_level_output_text_shortcut():
    # A stored response never has this key at all (output_text is a computed
    # SDK property, absent from model_dump()) -- if present anyway (e.g. a
    # hand-crafted fixture), it must still be ignored in favor of walking
    # `output`, since that's the only path a real stored response has.
    response = {
        "output_text": "SHOULD NOT BE READ",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "real text"}]}],
    }
    assert extract_raw_response_text("openai_chatgpt", response) == "real text"


def test_openai_chatgpt_missing_output_is_none():
    assert extract_raw_response_text("openai_chatgpt", {"model": "gpt-5.6-luna"}) is None


def test_openai_chatgpt_sources():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "...",
                        "annotations": [
                            {"type": "url_citation", "url": "https://smith.ai/pricing", "title": "Pricing"},
                            {"type": "other_annotation"},
                        ],
                    }
                ],
            }
        ]
    }
    sources = extract_sources_rows("openai_chatgpt", response)
    assert sources == [
        {"position": 1, "url_raw": "https://smith.ai/pricing", "domain": "smith.ai", "title": "Pricing"}
    ]


def test_google_gemini_text_and_model():
    response = {
        "model_version": "gemini-3.6-flash-002",
        "candidates": [{"content": {"parts": [{"text": "Goodcall handles this well."}]}}],
    }
    assert extract_raw_response_text("google_gemini", response) == "Goodcall handles this well."
    assert extract_model_returned("google_gemini", response) == "gemini-3.6-flash-002"


def test_google_gemini_never_reads_top_level_text_shortcut():
    response = {
        "text": "SHOULD NOT BE READ",
        "candidates": [{"content": {"parts": [{"text": "real text"}]}}],
    }
    assert extract_raw_response_text("google_gemini", response) == "real text"


def test_google_gemini_missing_candidates_is_none():
    assert extract_raw_response_text("google_gemini", {"model_version": "gemini-3.6-flash"}) is None


def test_google_gemini_sources():
    response = {
        "candidates": [
            {
                "grounding_metadata": {
                    "grounding_chunks": [
                        {"web": {"uri": "https://heyrosie.com/", "domain": "heyrosie.com", "title": "Rosie"}},
                        {"web": None},
                    ]
                }
            }
        ]
    }
    sources = extract_sources_rows("google_gemini", response)
    assert sources == [
        {"position": 1, "url_raw": "https://heyrosie.com/", "domain": "heyrosie.com", "title": "Rosie"}
    ]


def test_anthropic_claude_text_and_model():
    response = {
        "model": "claude-sonnet-5-20260601",
        "content": [
            {"type": "server_tool_use", "name": "web_search"},
            {"type": "text", "text": "Vapi is a developer platform."},
        ],
    }
    assert extract_raw_response_text("anthropic_claude", response) == "Vapi is a developer platform."
    assert extract_model_returned("anthropic_claude", response) == "claude-sonnet-5-20260601"


def test_anthropic_claude_sources():
    response = {
        "content": [
            {
                "type": "text",
                "text": "...",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://askrosie.ai/",
                        "title": "Rosie AI",
                    },
                    {"type": "other_citation_type"},
                ],
            }
        ]
    }
    sources = extract_sources_rows("anthropic_claude", response)
    assert sources == [
        {"position": 1, "url_raw": "https://askrosie.ai/", "domain": "askrosie.ai", "title": "Rosie AI"}
    ]


def test_google_ai_mode_text_prefers_reconstructed_markdown():
    response = {"reconstructed_markdown": "# Best options\n\nSmith.ai", "text_blocks": []}
    assert extract_raw_response_text("google_ai_mode", response) == "# Best options\n\nSmith.ai"


def test_google_ai_mode_text_falls_back_to_text_blocks():
    response = {"text_blocks": [{"type": "paragraph", "snippet": "Try Goodcall."}]}
    assert extract_raw_response_text("google_ai_mode", response) == "Try Goodcall."


def test_google_ai_mode_model_returned_always_none():
    assert extract_model_returned("google_ai_mode", {"reconstructed_markdown": "x"}) is None


def test_google_ai_mode_sources():
    response = {"references": [{"link": "https://vapi.ai/", "title": "Vapi"}, {"title": "no link"}]}
    sources = extract_sources_rows("google_ai_mode", response)
    assert sources == [{"position": 1, "url_raw": "https://vapi.ai/", "domain": "vapi.ai", "title": "Vapi"}]


def test_missing_response_returns_none_and_empty_list():
    assert extract_raw_response_text("openai_chatgpt", None) is None
    assert extract_model_returned("openai_chatgpt", None) is None
    assert extract_sources_rows("openai_chatgpt", None) == []


def test_unknown_engine_returns_none_and_empty_list():
    assert extract_raw_response_text("some_future_engine", {"output": []}) is None
    assert extract_model_returned("some_future_engine", {"model": "x"}) is None
    assert extract_sources_rows("some_future_engine", {"references": []}) == []
