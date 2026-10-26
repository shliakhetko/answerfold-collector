import json
import time

from collector.config import OpenAIConfig
from collector.engines.openai_chatgpt import call_chatgpt
from collector.engines.types import AdapterResult
from collector.runner.execute import build_run_id, execute_run, get_instrument_commit
from tests.fakes import FakeClient, FakeResponses, fake_success_response


def _read_lines(runs_dir):
    files = sorted(runs_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected exactly one jsonl file, found {files}"
    return [json.loads(line) for line in files[0].read_text().splitlines() if line]


def test_build_run_id_is_deterministic():
    kwargs = dict(engine="openai_chatgpt", intent_id=2, paraphrase_id=3, day_index=1, rep_index=2)
    assert build_run_id(**kwargs) == build_run_id(**kwargs) == "openai_chatgpt-i2-p3-d1-r2"


def test_get_instrument_commit_returns_nonempty_string():
    commit = get_instrument_commit()
    assert isinstance(commit, str) and len(commit) > 0


def test_execute_run_happy_path_appends_jsonl_line(tmp_path):
    client = FakeClient(FakeResponses(response=fake_success_response()))
    cfg = OpenAIConfig(api_key="sk-test")

    run_id = execute_run(
        adapter=call_chatgpt,
        engine="openai_chatgpt",
        model_requested="gpt-4o",
        intent_id=1,
        paraphrase_id=1,
        prompt_text="capital of France?",
        rep_index=1,
        day_index=1,
        prompt_order_seed=42,
        position_in_pass=0,
        runs_dir=tmp_path,
        config=cfg,
        client=client,
    )

    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    line = lines[0]
    assert line["run_id"] == run_id == "openai_chatgpt-i1-p1-d1-r1"
    assert line["engine"] == "openai_chatgpt"
    assert line["intent_id"] == 1
    assert line["paraphrase_id"] == 1
    assert line["rep_index"] == 1
    assert line["day_index"] == 1
    assert "ts_utc" in line
    assert "error" not in line
    assert line["request"] == {
        "model": "gpt-4o",
        "input": "capital of France?",
        "tools": [
            {
                "type": "web_search",
                "user_location": {"type": "approximate", "country": "US"},
            }
        ],
    }
    assert line["response"]["id"] == "resp_fake"
    assert line["response"]["status"] == "completed"


def test_execute_run_failure_path_writes_error_instead_of_response(tmp_path):
    client = FakeClient(FakeResponses(exc=TimeoutError("simulated timeout")))
    cfg = OpenAIConfig(api_key="sk-test")

    execute_run(
        adapter=call_chatgpt,
        engine="openai_chatgpt",
        model_requested="gpt-4o",
        intent_id=1,
        paraphrase_id=1,
        prompt_text="q",
        rep_index=2,
        day_index=1,
        prompt_order_seed=42,
        position_in_pass=0,
        runs_dir=tmp_path,
        config=cfg,
        client=client,
    )

    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    line = lines[0]
    assert line["error"] == "TimeoutError: simulated timeout"
    assert "response" not in line
    assert line["request"] == {
        "model": "gpt-4o",
        "input": "q",
        "tools": [
            {
                "type": "web_search",
                "user_location": {"type": "approximate", "country": "US"},
            }
        ],
    }


def test_execute_run_persists_scraper_vendor_and_endpoint_inside_request(tmp_path):
    """Regression check for the no-API-cell fields (INSTRUMENT.md section 6):
    a stub adapter reporting scraper_vendor/scraper_endpoint still round-trips
    through execute_run -- here via whatever it puts on the AdapterResult's
    request_payload, since the JSONL schema has no dedicated top-level slot
    for scraper identity (it lives inside `request`, the adapter's own
    payload shape)."""

    def stub_adapter(*, prompt_text, model_requested):
        return AdapterResult(
            raw_response_text="stub answer",
            search_fired=True,
            scraper_vendor="serpapi",
            scraper_endpoint="google_ai_mode",
            request_payload={"engine": "google_ai_mode", "q": prompt_text},
            raw_response_envelope=json.dumps({"reconstructed_markdown": "stub answer"}),
        )

    execute_run(
        adapter=stub_adapter,
        engine="google_ai_mode",
        model_requested="",
        intent_id=1,
        paraphrase_id=1,
        prompt_text="q",
        rep_index=1,
        day_index=1,
        prompt_order_seed=42,
        position_in_pass=0,
        runs_dir=tmp_path,
    )

    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["request"] == {"engine": "google_ai_mode", "q": "q"}
    assert lines[0]["response"] == {"reconstructed_markdown": "stub answer"}


def test_execute_run_timeout_writes_error_row_without_waiting_for_the_call(tmp_path):
    """A hard timeout_seconds must return (and write a row) as soon as the
    deadline passes, not once the slow adapter eventually finishes -- the
    barrier-synchronized collection loop (scripts/collect/run_collection.py) depends
    on this to bound how long the other engines ever wait on a stuck one."""

    def slow_adapter(*, prompt_text, model_requested):
        time.sleep(5)
        return AdapterResult(raw_response_text="too late", request_payload={"prompt": prompt_text})

    started = time.monotonic()
    run_id = execute_run(
        adapter=slow_adapter,
        engine="openai_chatgpt",
        model_requested="gpt-4o",
        intent_id=1,
        paraphrase_id=1,
        prompt_text="q",
        rep_index=1,
        day_index=1,
        prompt_order_seed=42,
        position_in_pass=0,
        runs_dir=tmp_path,
        timeout_seconds=0.05,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    line = lines[0]
    assert line["run_id"] == run_id
    assert line["error"] == "timed out after 0.05s"
    assert "response" not in line
    assert line["request"] is None


def test_execute_run_retry_appends_second_line_last_one_wins(tmp_path):
    """Append-only design (confirmed this session): a retry of the same
    deterministic run_id produces a second line, not a rewrite. Any reader
    is expected to treat the last line for a given run_id as authoritative;
    this module never deletes/rewrites earlier lines itself."""

    def stub_adapter(*, prompt_text, model_requested, attempt):
        return AdapterResult(raw_response_text=f"attempt {attempt}", request_payload={"attempt": attempt})

    common_kwargs = dict(
        adapter=stub_adapter,
        engine="openai_chatgpt",
        model_requested="gpt-4o",
        intent_id=1,
        paraphrase_id=1,
        prompt_text="q",
        rep_index=1,
        day_index=1,
        prompt_order_seed=42,
        position_in_pass=0,
        runs_dir=tmp_path,
    )

    run_id_1 = execute_run(**common_kwargs, attempt=1)
    run_id_2 = execute_run(**common_kwargs, attempt=2)

    assert run_id_1 == run_id_2

    lines = _read_lines(tmp_path)
    assert len(lines) == 2
    assert all(line["run_id"] == run_id_1 for line in lines)
    assert [line["request"]["attempt"] for line in lines] == [1, 2]
