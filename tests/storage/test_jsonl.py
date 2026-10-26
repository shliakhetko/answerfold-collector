import json
import threading

from collector.storage.jsonl import append_run_line


def test_append_run_line_concurrent_writes_produce_valid_jsonl(tmp_path):
    """Regression test for jsonl.py's write lock: N threads calling
    append_run_line concurrently against the same runs_dir/date must never
    interleave/corrupt each other's line, even with large per-line payloads
    that risk spanning multiple underlying write() syscalls on flush."""
    n_threads = 20
    big_payload = {"raw_text": "x" * 50_000}

    def write_one(i: int) -> None:
        append_run_line(
            run_id=f"stress-{i}",
            ts_utc="2026-08-19T00:00:00+00:00",
            engine="openai_chatgpt",
            intent_id=1,
            paraphrase_id=1,
            rep_index=1,
            day_index=1,
            prompt_order_seed=42,
            position_in_pass=i,
            request={"i": i},
            response=big_payload,
            runs_dir=tmp_path,
        )

    threads = [threading.Thread(target=write_one, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == n_threads

    parsed = [json.loads(line) for line in lines]  # raises if any line is malformed/interleaved
    assert {p["run_id"] for p in parsed} == {f"stress-{i}" for i in range(n_threads)}
