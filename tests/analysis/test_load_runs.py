import json
import sqlite3

from analysis.aggregate.load_runs import LoadStats, is_max_tokens_truncated, load_jsonl_dir, parse_run_record
from collector.storage.db import init_db


def _openai_record(run_id: str = "openai_chatgpt-i1-p1-d1-r1", output_tokens: int = 266) -> dict:
    return {
        "run_id": run_id,
        "ts_utc": "2026-08-13T11:07:41.873421+00:00",
        "engine": "openai_chatgpt",
        "intent_id": 1,
        "paraphrase_id": 1,
        "rep_index": 1,
        "day_index": 1,
        "request": {"model": "gpt-4o", "input": "what's the best CRM?", "tools": [{"type": "web_search"}]},
        "response": {
            "usage": {"input_tokens": 314, "output_tokens": output_tokens, "total_tokens": 314 + output_tokens}
        },
    }


def _google_ai_mode_record(run_id: str = "google_ai_mode-i1-p1-d1-r1") -> dict:
    return {
        "run_id": run_id,
        "ts_utc": "2026-08-13T11:09:48.767398+00:00",
        "engine": "google_ai_mode",
        "intent_id": 1,
        "paraphrase_id": 1,
        "rep_index": 1,
        "day_index": 1,
        "request": {"engine": "google_ai_mode", "q": "what's the best CRM?"},
        "response": {"search_metadata": {"status": "Success"}, "text_blocks": []},
    }


def test_parse_run_record_maps_tokens_and_prompt_text():
    kwargs = parse_run_record(_openai_record(), instrument_commit="deadbeef")
    assert kwargs["tokens_input"] == 314
    assert kwargs["tokens_output"] == 266
    assert kwargs["tokens_total"] == 580
    assert kwargs["prompt_text"] == "what's the best CRM?"
    assert kwargs["instrument_commit"] == "deadbeef"
    assert "api_cost_or_tokens" not in kwargs
    assert "extra" not in kwargs
    # No "output" key with a message/output_text block in this fixture's
    # response -- a real "no text", not "unknown" (response is present).
    assert kwargs["raw_response_text"] is None
    assert kwargs["model_returned"] is None
    # No "output" key in this fixture's response -- a real "did not search",
    # not "unknown" (response is present, just has no web_search_call item).
    assert kwargs["search_fired"] is False
    # Nothing in this fixture's usage object reports any of these -- every
    # detail column is its own flat kwarg, NULL (not merged/omitted) when
    # not reported.
    for field in (
        "reasoning_tokens", "tool_tokens", "cached_tokens", "cache_write_tokens",
        "cached_content_tokens", "cache_creation_tokens", "cache_read_tokens",
        "web_search_requests", "web_fetch_requests",
    ):
        assert kwargs[field] is None


def test_parse_run_record_populates_raw_response_text_and_model_returned():
    record = _openai_record()
    record["response"]["model"] = "gpt-5.6-luna-2026-08-01"
    record["response"]["output"] = [
        {"type": "message", "content": [{"type": "output_text", "text": "Smith.ai is a popular choice."}]}
    ]
    kwargs = parse_run_record(record, instrument_commit="deadbeef")
    assert kwargs["raw_response_text"] == "Smith.ai is a popular choice."
    assert kwargs["model_returned"] == "gpt-5.6-luna-2026-08-01"


def test_parse_run_record_flattens_reasoning_and_vendor_fields():
    record = _openai_record()
    record["response"]["usage"]["output_tokens_details"] = {"reasoning_tokens": 592}
    record["response"]["usage"]["input_tokens_details"] = {"cache_write_tokens": 4412, "cached_tokens": 0}
    record["response"]["output"] = [{"type": "web_search_call"}]
    kwargs = parse_run_record(record, instrument_commit="deadbeef")
    assert kwargs["search_fired"] is True
    # reasoning_tokens (from extract_usage) and cache_write_tokens/cached_tokens
    # (from extract_extra_usage_fields) land as separate flat kwargs, each its
    # own schema.sql column -- cached_tokens=0 is a real reported zero and
    # survives; tool_tokens stayed None (not reported by this vendor).
    assert kwargs["reasoning_tokens"] == 592
    assert kwargs["cache_write_tokens"] == 4412
    assert kwargs["cached_tokens"] == 0
    assert kwargs["tool_tokens"] is None
    assert "extra" not in kwargs


def test_parse_run_record_google_ai_mode_scraper_fields():
    kwargs = parse_run_record(_google_ai_mode_record(), instrument_commit="deadbeef")
    assert kwargs["scraper_vendor"] == "serpapi"
    assert kwargs["scraper_endpoint"] == "google_ai_mode"
    assert (kwargs["tokens_input"], kwargs["tokens_output"], kwargs["tokens_total"]) == (None, None, None)
    # Unconditionally True on a successful parse -- Google AI Mode has no
    # non-search mode (collector/engines/google_ai_mode.py).
    assert kwargs["search_fired"] is True


def test_parse_run_record_extracts_finish_reason_per_engine():
    anthropic_record = _openai_record()
    anthropic_record["engine"] = "anthropic_claude"
    anthropic_record["response"] = {"stop_reason": "max_tokens"}
    assert parse_run_record(anthropic_record, instrument_commit="deadbeef")["finish_reason"] == "max_tokens"

    openai_incomplete = _openai_record()
    openai_incomplete["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    assert parse_run_record(openai_incomplete, instrument_commit="deadbeef")["finish_reason"] == "max_output_tokens"

    gemini_record = _openai_record()
    gemini_record["engine"] = "google_gemini"
    gemini_record["response"] = {"candidates": [{"finish_reason": "MAX_TOKENS"}]}
    assert parse_run_record(gemini_record, instrument_commit="deadbeef")["finish_reason"] == "MAX_TOKENS"

    assert parse_run_record(_google_ai_mode_record(), instrument_commit="deadbeef")["finish_reason"] == "Success"


def test_is_max_tokens_truncated_matches_each_vendors_own_signal():
    assert is_max_tokens_truncated("anthropic_claude", "max_tokens") is True
    assert is_max_tokens_truncated("anthropic_claude", "end_turn") is False
    assert is_max_tokens_truncated("openai_chatgpt", "max_output_tokens") is True
    assert is_max_tokens_truncated("openai_chatgpt", "completed") is False
    assert is_max_tokens_truncated("google_gemini", "MAX_TOKENS") is True
    assert is_max_tokens_truncated("google_gemini", "STOP") is False
    # No token concept for this engine -- never truncated this way.
    assert is_max_tokens_truncated("google_ai_mode", "max_tokens") is False
    assert is_max_tokens_truncated("anthropic_claude", None) is False


def test_load_jsonl_dir_inserts_and_dedupes_retries(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    second_line = _openai_record(output_tokens=266)
    second_line["response"]["usage"]["output_tokens_details"] = {"reasoning_tokens": 592}
    second_line["response"]["output"] = [
        {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": "Smith.ai is a popular choice.",
                    "annotations": [{"type": "url_citation", "url": "https://smith.ai/", "title": "Smith.ai"}],
                }
            ],
        }
    ]
    with (runs_dir / "2026-08-13.jsonl").open("w") as f:
        # Two lines, same run_id -- a retried run. The second (later)
        # line's token counts must be the ones that land.
        f.write(json.dumps(_openai_record(output_tokens=100)) + "\n")
        f.write(json.dumps(second_line) + "\n")
        f.write(json.dumps(_google_ai_mode_record()) + "\n")

    stats = load_jsonl_dir(runs_dir, db_path=db_path, instrument_commit="deadbeef")
    assert stats == LoadStats(inserted=2, skipped_existing=0, files=1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT tokens_output, tokens_total, reasoning_tokens, raw_response_text FROM runs WHERE run_id = ?",
        ("openai_chatgpt-i1-p1-d1-r1",),
    ).fetchone()
    assert (row["tokens_output"], row["tokens_total"]) == (266, 314 + 266)
    # A real SQL column now, not something buried in a JSON `extra` blob.
    assert row["reasoning_tokens"] == 592
    assert row["raw_response_text"] == "Smith.ai is a popular choice."

    # sources rows land once, from the winning (second) line only.
    source_rows = conn.execute(
        "SELECT url_raw, domain, title FROM sources WHERE run_id = ?",
        ("openai_chatgpt-i1-p1-d1-r1",),
    ).fetchall()
    assert [dict(r) for r in source_rows] == [
        {"url_raw": "https://smith.ai/", "domain": "smith.ai", "title": "Smith.ai"}
    ]
    conn.close()

    # Re-running against the same files must not raise or duplicate rows --
    # every run_id is now already present.
    stats_again = load_jsonl_dir(runs_dir, db_path=db_path, instrument_commit="deadbeef")
    assert stats_again == LoadStats(inserted=0, skipped_existing=2, files=1)

    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM sources WHERE run_id = ?", ("openai_chatgpt-i1-p1-d1-r1",)
    ).fetchone()[0]
    assert count == 1  # not duplicated on rerun
    conn.close()
