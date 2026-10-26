import json
import time
from pathlib import Path

from collector.engines import registry
from collector.engines.types import AdapterResult
from scripts.collect import run_collection

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _fake_adapter(*, prompt_text: str, model_requested: str) -> AdapterResult:
    return AdapterResult(
        model_returned=model_requested,
        raw_response_text="ok",
        search_fired=False,
        request_payload={"model": model_requested, "input": prompt_text},
        raw_response_envelope=json.dumps({"model": model_requested, "text": "ok"}),
    )


def _slow_fake_adapter(*, prompt_text: str, model_requested: str) -> AdapterResult:
    time.sleep(0.2)
    return _fake_adapter(prompt_text=prompt_text, model_requested=model_requested)


def _read_lines(runs_dir):
    files = sorted(runs_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected exactly one jsonl file, found {files}"
    return [json.loads(line) for line in files[0].read_text().splitlines() if line]


def test_run_collection_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.ADAPTERS, "openai_chatgpt", _fake_adapter)
    runs_dir = tmp_path / "runs"

    run_collection.main(
        [
            "--day-index", "1",
            "--prompts", str(FIXTURES / "prompts.csv"),
            "--experiment-config", str(FIXTURES / "experiment.yaml"),
            "--runs-dir", str(runs_dir),
            "--prompt-order-seed", "42",
        ]
    )

    lines = _read_lines(runs_dir)
    assert len(lines) == 1
    line = lines[0]
    assert line["run_id"] == "openai_chatgpt-i1-p1-d1-r1"
    assert line["response"]["text"] == "ok"
    assert line["request"]["model"] == "gpt-4o-search-preview"
    assert line["prompt_order_seed"] == 42
    assert line["position_in_pass"] == 0


def test_run_collection_runs_engines_concurrently_and_merges_lines(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.ADAPTERS, "openai_chatgpt", _fake_adapter)
    monkeypatch.setitem(registry.ADAPTERS, "anthropic_claude", _fake_adapter)
    runs_dir = tmp_path / "runs"

    run_collection.main(
        [
            "--day-index", "1",
            "--prompts", str(FIXTURES / "prompts.csv"),
            "--experiment-config", str(FIXTURES / "experiment_multi_engine.yaml"),
            "--runs-dir", str(runs_dir),
            "--prompt-order-seed", "42",
        ]
    )

    lines = _read_lines(runs_dir)
    assert len(lines) == 2
    assert {line["engine"] for line in lines} == {"openai_chatgpt", "anthropic_claude"}
    assert {line["prompt_order_seed"] for line in lines} == {42}


def test_run_collection_runs_engines_in_parallel_not_serially(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.ADAPTERS, "openai_chatgpt", _slow_fake_adapter)
    monkeypatch.setitem(registry.ADAPTERS, "anthropic_claude", _slow_fake_adapter)

    start = time.monotonic()
    run_collection.main(
        [
            "--day-index", "1",
            "--prompts", str(FIXTURES / "prompts.csv"),
            "--experiment-config", str(FIXTURES / "experiment_multi_engine.yaml"),
            "--runs-dir", str(tmp_path / "runs"),
            "--prompt-order-seed", "42",
        ]
    )
    elapsed = time.monotonic() - start

    # Serial would take >= 2 * 0.2s; parallel should land close to ~0.2s.
    # Generous threshold to avoid flakiness on a loaded machine.
    assert elapsed < 0.35


def test_run_collection_shares_shuffled_pass_order_across_engines(tmp_path, monkeypatch):
    """The pass order is computed once from the seed and shared by every
    engine -- all engines must ask the *same* prompt at a given
    (rep_index, position_in_pass), even though the order itself is shuffled
    away from file order."""
    monkeypatch.setitem(registry.ADAPTERS, "openai_chatgpt", _fake_adapter)
    monkeypatch.setitem(registry.ADAPTERS, "anthropic_claude", _fake_adapter)
    runs_dir = tmp_path / "runs"

    run_collection.main(
        [
            "--day-index", "1",
            "--prompts", str(FIXTURES / "prompts_multi.csv"),
            "--experiment-config", str(FIXTURES / "experiment_multi_engine_3reps.yaml"),
            "--runs-dir", str(runs_dir),
            "--prompt-order-seed", "7",
        ]
    )

    lines = _read_lines(runs_dir)
    assert len(lines) == 2 * 3 * 5  # 2 engines x 3 reps x 5 prompts

    by_position: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for line in lines:
        key = (line["rep_index"], line["position_in_pass"])
        by_position.setdefault(key, set()).add((line["intent_id"], line["paraphrase_id"]))

    # Every (rep, position) key was seen by both engines and they agree on
    # which prompt lived there -- exactly one (intent_id, paraphrase_id).
    assert len(by_position) == 3 * 5
    assert all(len(prompts_seen) == 1 for prompts_seen in by_position.values())

    # Not just file order: at least one pass departs from the CSV's
    # (intent_id, paraphrase_id) sequence for this seed.
    file_order = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    for rep_index in (1, 2, 3):
        pass_order = [next(iter(by_position[(rep_index, pos)])) for pos in range(5)]
        if pass_order != file_order:
            break
    else:
        assert False, "expected at least one shuffled pass to differ from file order"


def test_run_collection_prompt_order_is_deterministic_given_seed(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.ADAPTERS, "openai_chatgpt", _fake_adapter)
    monkeypatch.setitem(registry.ADAPTERS, "anthropic_claude", _fake_adapter)

    def order_for(runs_dir):
        run_collection.main(
            [
                "--day-index", "1",
                "--prompts", str(FIXTURES / "prompts_multi.csv"),
                "--experiment-config", str(FIXTURES / "experiment_multi_engine_3reps.yaml"),
                "--runs-dir", str(runs_dir),
                "--prompt-order-seed", "99",
            ]
        )
        lines = _read_lines(runs_dir)
        return sorted(
            (line["engine"], line["rep_index"], line["position_in_pass"], line["intent_id"], line["paraphrase_id"])
            for line in lines
        )

    assert order_for(tmp_path / "run_a") == order_for(tmp_path / "run_b")


def test_run_collection_timeout_writes_error_row_and_does_not_block_other_engine(tmp_path, monkeypatch):
    def hanging_adapter(*, prompt_text: str, model_requested: str) -> AdapterResult:
        time.sleep(2)
        return _fake_adapter(prompt_text=prompt_text, model_requested=model_requested)

    monkeypatch.setitem(registry.ADAPTERS, "openai_chatgpt", _fake_adapter)
    monkeypatch.setitem(registry.ADAPTERS, "anthropic_claude", hanging_adapter)
    monkeypatch.setattr(run_collection, "PER_REQUEST_TIMEOUT_SECONDS", 0.05)
    runs_dir = tmp_path / "runs"

    start = time.monotonic()
    run_collection.main(
        [
            "--day-index", "1",
            "--prompts", str(FIXTURES / "prompts.csv"),
            "--experiment-config", str(FIXTURES / "experiment_multi_engine.yaml"),
            "--runs-dir", str(runs_dir),
            "--prompt-order-seed", "42",
        ]
    )
    elapsed = time.monotonic() - start

    # Bounded by the injected 0.05s timeout, not the adapter's 2s sleep --
    # the barrier must not let the slow engine block the fast one.
    assert elapsed < 1.0

    lines = _read_lines(runs_dir)
    assert len(lines) == 2
    by_engine = {line["engine"]: line for line in lines}
    assert by_engine["openai_chatgpt"]["response"]["text"] == "ok"
    assert "timed out after 0.05s" in by_engine["anthropic_claude"]["error"]
    assert by_engine["anthropic_claude"]["request"] is None


def test_run_collection_rejects_day_index_outside_window(tmp_path):
    try:
        run_collection.main(
            [
                "--day-index", "5",
                "--prompts", str(FIXTURES / "prompts.csv"),
                "--experiment-config", str(FIXTURES / "experiment.yaml"),
                "--runs-dir", str(tmp_path / "runs"),
                "--prompt-order-seed", "42",
            ]
        )
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "outside the declared window" in str(exc)
