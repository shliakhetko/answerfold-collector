import json

from analysis.coding import storage
from analysis.coding.a3_entity_resolution import EntityResolutionResult, EntityRow
from analysis.coding.a3b_entity_verification import VerificationResult, VerificationRow
from analysis.coding.a4_entity_classification import ClassificationResult, ClassificationRow
from analysis.coding.a5_url_classification import URLClassificationResult, URLClassificationRow
from analysis.coding.extractors.base import ExtractionResult, MentionCandidate
from collector.storage.db import get_connection
from scripts.coding.run_coding_pipeline import run_pipeline
from tests.fakes import FakeHomepageFetcher

RAW_TEXT_1 = "Otter.ai is a strong choice for meeting notes."
RAW_TEXT_2 = "Otter.ai works well for sales calls too."


def _record(run_id: str, *, raw_text: str, rep_index: int) -> dict:
    return {
        "run_id": run_id,
        "ts_utc": "2026-08-24T10:00:00+00:00",
        "engine": "anthropic_claude",
        "intent_id": 1,
        "paraphrase_id": 1,
        "rep_index": rep_index,
        "day_index": 1,
        "request": {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "best AI meeting note taker"}]},
        "response": {
            "model": "claude-sonnet-5",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": raw_text,
                    "citations": [
                        {"type": "web_search_result_location", "url": "https://otter.ai/", "title": "Otter.ai"}
                    ],
                }
            ],
        },
    }


def _fake_extractor(*, raw_response_text: str) -> ExtractionResult:
    # Whole text as evidence_span always passes verify_candidate as long as
    # it literally contains "Otter.ai" -- true for both fixture texts here.
    return ExtractionResult(mentions=[MentionCandidate("Otter.ai", raw_response_text, True)])


FAKE_EXTRACTORS = {"anthropic": _fake_extractor, "openai": _fake_extractor, "gemini": _fake_extractor}


def _fake_entity_resolution(*, surface_form_candidates):
    if not surface_form_candidates:
        return EntityResolutionResult(entities=[])
    return EntityResolutionResult(
        entities=[
            EntityRow(entity_id="otter", canonical_name="Otter.ai", surface_forms=("Otter.ai",), homepage_url="https://otter.ai/")
        ]
    )


def _fake_verification(*, canonical_name, surface_forms, evidence_spans, candidate_domains):
    return VerificationResult(
        verification=VerificationRow(
            is_real=True, entity_type="vendor",
            verified_homepage_url="https://otter.ai/", disambiguation_note=None, confidence=0.9,
        )
    )


def _fake_classification(*, canonical_name, homepage_url, homepage_text, inclusion_frame_text):
    return ClassificationResult(classification=ClassificationRow("in", None, "done-for-you", "https://otter.ai/"))


def _fake_url_classification(*, url_normalized, title):
    return URLClassificationResult(classification=URLClassificationRow("vendor page", True, 0.9))


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_run_pipeline_chains_every_stage_end_to_end(tmp_path):
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(_record("run-1", raw_text=RAW_TEXT_1, rep_index=1)) + "\n")
        f.write(json.dumps(_record("run-2", raw_text=RAW_TEXT_2, rep_index=2)) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    db_path = tmp_path / "coding_trial.sqlite3"

    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=db_path,
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_fake_entity_resolution,
        verification_call_fn=_fake_verification,
        classification_call_fn=_fake_classification,
        url_classification_call_fn=_fake_url_classification,
        fetcher=FakeHomepageFetcher({"https://otter.ai/": "Otter.ai transcribes and summarizes meetings."}),
    )

    # A1
    assert report.runs_processed == 2
    assert report.runs_skipped_no_text == 0
    assert report.mentions_written == 2

    conn = get_connection(db_path)
    mentions = storage.list_mentions_raw(conn=conn)
    assert {m["surface_form"] for m in mentions} == {"Otter.ai"}

    # A2 (surface_forms view)
    surface_form_rows = storage.list_surface_forms(conn=conn)
    assert surface_form_rows == [{"surface_form": "Otter.ai", "mention_count": 2, "run_ids": "run-1;run-2"}]

    # A3 + A4 (real fetch, via FakeHomepageFetcher)
    entity_rows = storage.list_entities(conn=conn)
    assert len(entity_rows) == 1
    assert entity_rows[0]["entity_id"] == "otter"
    assert entity_rows[0]["meets_frame"] == "in"
    assert report.entities_resolved == 1
    assert report.in_frame_count == 1
    assert report.out_of_frame_count == 0

    # A5
    url_rows = storage.list_url_registry(conn=conn)
    assert len(url_rows) == 1
    assert url_rows[0]["url_class"] == "vendor page"

    # B1/B2 -- persisted to their tables
    per_run_company_rows = conn.execute("SELECT run_id, mentioned FROM per_run_company").fetchall()
    assert {row["run_id"] for row in per_run_company_rows} == {"run-1", "run-2"}
    assert all(row["mentioned"] == 1 for row in per_run_company_rows)

    rollup_rows = conn.execute("SELECT run_id FROM per_run_rollups").fetchall()
    assert len(rollup_rows) == 2

    # B3 -- only 2 mentions/engine, well under ELIGIBILITY_FLOOR (5), so
    # nothing qualifies yet -- exercising that this trial's small volume
    # correctly produces an empty (not broken) eligibility report.
    assert report.qualifying_by_engine == {}
    assert report.cell_rates == []
    assert conn.execute("SELECT COUNT(*) FROM cell_rates").fetchone()[0] == 0

    conn.close()

    # trial_report.json round-trips the same report
    report_path = db_path.parent / f"{db_path.stem}_report.json"
    on_disk = json.loads(report_path.read_text())
    assert on_disk == report.to_dict()


def test_run_pipeline_rerun_against_same_db_path_does_not_raise(tmp_path):
    # Regression test: entities gets cleared+rewritten early (A3), but
    # per_run_company (which REFERENCES entities) used to only get cleared
    # near the end of the pipeline -- on a SECOND run against a db_path that
    # already has a previous run's per_run_company rows, A3's clear_entities()
    # would hit "FOREIGN KEY constraint failed" before ever reaching the
    # point that clears per_run_company. A first run against an empty db
    # never exercises this (nothing to conflict with yet).
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(_record("run-1", raw_text=RAW_TEXT_1, rep_index=1)) + "\n")
        f.write(json.dumps(_record("run-2", raw_text=RAW_TEXT_2, rep_index=2)) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    db_path = tmp_path / "coding_trial.sqlite3"
    kwargs = dict(
        runs_dir=runs_dir,
        db_path=db_path,
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_fake_entity_resolution,
        verification_call_fn=_fake_verification,
        classification_call_fn=_fake_classification,
        url_classification_call_fn=_fake_url_classification,
        fetcher=FakeHomepageFetcher({"https://otter.ai/": "Otter.ai transcribes and summarizes meetings."}),
    )

    run_pipeline(**kwargs)
    second_report = run_pipeline(**kwargs)  # must not raise sqlite3.IntegrityError

    assert second_report.entities_resolved == 1
    conn = get_connection(db_path)
    per_run_company_rows = conn.execute("SELECT run_id FROM per_run_company").fetchall()
    assert {row["run_id"] for row in per_run_company_rows} == {"run-1", "run-2"}  # not duplicated
    conn.close()


def _record_with_citations(run_id: str, *, raw_text: str, rep_index: int, citations: list[dict]) -> dict:
    rec = _record(run_id, raw_text=raw_text, rep_index=rep_index)
    rec["response"]["content"][0]["citations"] = citations
    return rec


def test_run_pipeline_applies_cited_url_resolution(tmp_path):
    # Two Gemini-style grounding-redirect wrappers that both resolve to the
    # same real page, plus one citation that resolved to a 404.
    wrap_a = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA"
    wrap_b = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/BBB"
    dead = "https://otter.ai/blog/removed-post"

    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(_record_with_citations(
            "run-1", raw_text=RAW_TEXT_1, rep_index=1,
            citations=[{"type": "web_search_result_location", "url": wrap_a, "title": "Otter"}],
        )) + "\n")
        f.write(json.dumps(_record_with_citations(
            "run-2", raw_text=RAW_TEXT_2, rep_index=2,
            citations=[
                {"type": "web_search_result_location", "url": wrap_b, "title": "Otter dup"},
                {"type": "web_search_result_location", "url": dead, "title": "gone"},
            ],
        )) + "\n")

    resolution_path = tmp_path / "resolution.jsonl"
    with resolution_path.open("w") as f:
        for wrap in (wrap_a, wrap_b):
            f.write(json.dumps({
                "url": wrap, "status_bucket": "ok",
                "final_url": "https://otter.ai/pricing",
                "final_registrable_domain": "otter.ai",
            }) + "\n")
        f.write(json.dumps({
            "url": dead, "status_bucket": "http_404",
            "final_url": "https://otter.ai/blog/removed-post",
            "final_registrable_domain": "otter.ai",
        }) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    db_path = tmp_path / "coding_trial.sqlite3"
    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=db_path,
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=resolution_path,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_fake_entity_resolution,
        verification_call_fn=_fake_verification,
        classification_call_fn=_fake_classification,
        url_classification_call_fn=_fake_url_classification,
        fetcher=FakeHomepageFetcher({"https://otter.ai/": "Otter.ai transcribes and summarizes meetings."}),
    )

    stats = report.cited_url_resolution
    assert stats["citations_total"] == 3
    assert stats["citations_swapped"] == 2  # both wrappers
    assert stats["citations_dead"] == 1
    assert stats["unique_urls_before"] == 3
    assert stats["unique_urls_after"] == 2  # wrappers collapsed to one otter.ai/pricing

    conn = get_connection(db_path)

    # A5: wrappers became one real row; the 404 recorded as "other", no call.
    url_rows = {r["url_normalized"]: r for r in storage.list_url_registry(conn=conn)}
    assert "https://otter.ai/pricing" in url_rows
    assert not any("vertexaisearch" in u for u in url_rows)
    assert url_rows["https://otter.ai/blog/removed-post"]["url_class"] == "other"

    # audit table carries the full mapping
    resolution_rows = {r["cited_url"]: r for r in storage.list_cited_url_resolution(conn=conn)}
    assert resolution_rows[wrap_a]["final_url"] == "https://otter.ai/pricing"
    assert resolution_rows[dead]["status_bucket"] == "http_404"
    conn.close()


def test_run_pipeline_a4prime_excludes_non_vendor_before_frame_check(tmp_path):
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(_record("run-1", raw_text="Otter.ai and G2 both came up.", rep_index=1)) + "\n")
        f.write(json.dumps(_record("run-2", raw_text="Otter.ai and G2 again.", rep_index=2)) + "\n")

    def two_entity_resolution(*, surface_form_candidates):
        return EntityResolutionResult(entities=[
            EntityRow(entity_id="otter", canonical_name="Otter.ai", surface_forms=("Otter.ai",), homepage_url="https://otter.ai/"),
            EntityRow(entity_id="g2", canonical_name="G2", surface_forms=("G2",), homepage_url=None),
        ])

    def _extractor(*, raw_response_text):
        return ExtractionResult(mentions=[
            MentionCandidate("Otter.ai", raw_response_text, True),
            MentionCandidate("G2", raw_response_text, False),
        ])

    def verification(*, canonical_name, surface_forms, evidence_spans, candidate_domains):
        if canonical_name == "G2":
            return VerificationResult(verification=VerificationRow(True, "directory", "https://www.g2.com", None, 0.95))
        return VerificationResult(verification=VerificationRow(True, "vendor", "https://otter.ai/", None, 0.9))

    classification_calls = []

    def tracking_classification(*, canonical_name, homepage_url, homepage_text, inclusion_frame_text):
        classification_calls.append(canonical_name)
        return ClassificationResult(classification=ClassificationRow("in", None, "done-for-you", "https://otter.ai/"))

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    db_path = tmp_path / "coding_trial.sqlite3"
    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=db_path,
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors={"anthropic": _extractor, "openai": _extractor, "gemini": _extractor},
        entity_resolution_call_fn=two_entity_resolution,
        verification_call_fn=verification,
        classification_call_fn=tracking_classification,
        url_classification_call_fn=_fake_url_classification,
        fetcher=FakeHomepageFetcher({"https://otter.ai/": "Otter.ai transcribes meetings."}),
    )

    # G2 never reached A4's frame check; Otter did.
    assert classification_calls == ["Otter.ai"]
    assert report.entity_verification["real_vendors"] == 1
    assert report.entity_verification["excluded_non_vendor"] == 1
    assert report.entity_verification["by_type"] == {"vendor": 1, "directory": 1}

    conn = get_connection(db_path)
    entities = {e["entity_id"]: e for e in storage.list_entities(conn=conn)}
    assert entities["g2"]["meets_frame"] == "out"
    assert entities["otter"]["meets_frame"] == "in"
    verifications = {v["entity_id"]: v for v in storage.list_entity_verification(conn=conn)}
    assert verifications["g2"]["entity_type"] == "directory"
    conn.close()


def test_run_pipeline_run_limit_caps_processed_runs(tmp_path):
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        for i in range(1, 6):
            f.write(json.dumps(_record(f"run-{i}", raw_text=RAW_TEXT_1, rep_index=i)) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=tmp_path / "c.sqlite3",
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_fake_entity_resolution,
        verification_call_fn=_fake_verification,
        classification_call_fn=_fake_classification,
        url_classification_call_fn=_fake_url_classification,
        fetcher=FakeHomepageFetcher({"https://otter.ai/": "Otter.ai transcribes meetings."}),
        run_limit=2,
    )
    assert report.runs_processed == 2  # 5 on disk, only 2 coded


def test_run_pipeline_stop_after_a1_writes_mentions_only(tmp_path):
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(_record("run-1", raw_text=RAW_TEXT_1, rep_index=1)) + "\n")
        f.write(json.dumps(_record("run-2", raw_text=RAW_TEXT_2, rep_index=2)) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    def _boom(**_k):
        raise AssertionError("stages after A1 must not run")

    db_path = tmp_path / "c.sqlite3"
    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=db_path,
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_boom,
        verification_call_fn=_boom,
        classification_call_fn=_boom,
        url_classification_call_fn=_boom,
        fetcher=FakeHomepageFetcher({}),
        stop_after="a1",
    )

    assert report.stopped_after == "a1"
    assert report.mentions_written == 2
    assert report.entities_resolved == 0 and report.cell_rates == []

    conn = get_connection(db_path)
    assert len(storage.list_mentions_raw(conn=conn)) == 2
    assert storage.list_entities(conn=conn) == []
    conn.close()

    on_disk = json.loads((db_path.parent / "c_report.json").read_text())
    assert on_disk["stopped_after"] == "a1"


def test_run_pipeline_stop_after_a3b_runs_verification_but_not_a4(tmp_path):
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(_record("run-1", raw_text=RAW_TEXT_1, rep_index=1)) + "\n")
        f.write(json.dumps(_record("run-2", raw_text=RAW_TEXT_2, rep_index=2)) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "STUB frame -- must not be read at stop_after=a3b")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    def _boom_a4(**_k):
        raise AssertionError("A4 classification must not run at stop_after=a3b")

    db_path = tmp_path / "c.sqlite3"
    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=db_path,
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_fake_entity_resolution,
        verification_call_fn=_fake_verification,
        classification_call_fn=_boom_a4,
        url_classification_call_fn=_boom_a4,
        fetcher=FakeHomepageFetcher({}),
        stop_after="a3b",
    )

    assert report.stopped_after == "a3b"
    assert report.entities_resolved > 0
    assert report.entity_verification["ran"] is True
    assert report.cell_rates == []

    conn = get_connection(db_path)
    assert len(storage.list_entities(conn=conn)) > 0
    assert len(storage.list_entity_verification(conn=conn)) > 0
    assert storage.list_url_registry(conn=conn) == []
    conn.close()


def test_run_pipeline_skip_entity_verification_flag(tmp_path):
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(_record("run-1", raw_text=RAW_TEXT_1, rep_index=1)) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    def exploding_verification(**_kwargs):
        raise AssertionError("verification must not run when skipped")

    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=tmp_path / "out.sqlite3",
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_fake_entity_resolution,
        verification_call_fn=exploding_verification,
        classification_call_fn=_fake_classification,
        url_classification_call_fn=_fake_url_classification,
        fetcher=FakeHomepageFetcher({"https://otter.ai/": "Otter.ai transcribes and summarizes meetings."}),
        skip_entity_verification=True,
    )

    assert report.entity_verification["ran"] is False
    conn = get_connection(tmp_path / "out.sqlite3")
    assert storage.list_entity_verification(conn=conn) == []
    # A4 still classified the entity normally (fail-open)
    assert storage.list_entities(conn=conn)[0]["meets_frame"] == "in"
    conn.close()


def test_run_pipeline_skips_runs_with_no_text(tmp_path):
    runs_dir = tmp_path / "runs_trial"
    runs_dir.mkdir()
    errored_record = _record("run-err", raw_text="", rep_index=1)
    errored_record["response"] = None
    errored_record["error"] = "timed out"
    with (runs_dir / "2026-08-24.jsonl").open("w") as f:
        f.write(json.dumps(errored_record) + "\n")

    inclusion_frame_path = tmp_path / "inclusion_frame.yaml"
    _write(inclusion_frame_path, "trial-only stub frame")
    seed_list_path = tmp_path / "seed.csv"
    _write(seed_list_path, "canonical_name,homepage_domain\n")
    alt_domains_seed_path = tmp_path / "alt_domains_seed.csv"
    _write(alt_domains_seed_path, "entity_id,alt_domain,source_note\n")

    report = run_pipeline(
        runs_dir=runs_dir,
        db_path=tmp_path / "out.sqlite3",
        unbranded_intent_ids={1},
        inclusion_frame_path=inclusion_frame_path,
        seed_list_path=seed_list_path,
        alt_domains_seed_path=alt_domains_seed_path,
        cache_dir=tmp_path / "llm_cache",
        cited_url_resolution_path=None,
        instrument_commit="test-commit",
        extractors=FAKE_EXTRACTORS,
        entity_resolution_call_fn=_fake_entity_resolution,
        verification_call_fn=_fake_verification,
        classification_call_fn=_fake_classification,
        url_classification_call_fn=_fake_url_classification,
        fetcher=FakeHomepageFetcher({}),
    )

    assert report.runs_processed == 0
    assert report.runs_skipped_no_text == 1
    assert report.mentions_written == 0
    assert report.entities_resolved == 0
