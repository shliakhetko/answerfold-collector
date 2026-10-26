from analysis.phase_b.figures import s3
from analysis.phase_b.reading import FrozenInputs


def _run(run_id, engine, *, error=None, raw_response_text="some text", is_manual=None, search_fired=True, paraphrase_id=None):
    return {
        "run_id": run_id,
        "engine": engine,
        "error": error,
        "raw_response_text": raw_response_text,
        "is_manual": is_manual,
        "search_fired": search_fired,
        "paraphrase_id": paraphrase_id,
    }


def _inputs(raw_runs, sources_by_run):
    return FrozenInputs(
        entities=[],
        alt_domains_rows=[],
        entity_verification_rows=[],
        mentions_raw=[],
        raw_runs=raw_runs,
        sources_by_run=sources_by_run,
    )


def test_s3_basic_share_with_and_without_sources():
    raw_runs = [
        _run("r1", "openai_chatgpt"),
        _run("r2", "openai_chatgpt"),
        _run("r3", "openai_chatgpt"),
    ]
    sources_by_run = {"r1": [{"url_raw": "https://x.com"}], "r2": [], "r3": []}
    result = s3.compute(_inputs(raw_runs, sources_by_run))
    assert result.status == "ok"
    assert result.denominator_after_exclusion["openai_chatgpt"] == 3
    assert result.n_included["openai_chatgpt"] == 3
    assert result.value["openai_chatgpt"] == 2 / 3
    assert result.n_excluded["openai_chatgpt"] == {}


def test_s3_excludes_collection_error():
    raw_runs = [_run("r1", "google_gemini", error="500 INTERNAL"), _run("r2", "google_gemini")]
    sources_by_run = {"r2": []}
    result = s3.compute(_inputs(raw_runs, sources_by_run))
    assert result.denominator_after_exclusion["google_gemini"] == 1  # what value was actually computed over
    assert result.n_included["google_gemini"] == 1
    assert result.n_excluded["google_gemini"] == {"collection_error": 1}
    assert result.value["google_gemini"] == 1.0  # the one usable run had zero sources


def test_s3_excludes_no_response_text():
    # a second, healthy engine keeps the figure as a whole "ok" so this
    # exercises the per-engine untestable slice, not the all-engines
    # collapse (covered separately below)
    raw_runs = [
        _run("r1", "anthropic_claude", raw_response_text=""),
        _run("r2", "anthropic_claude", raw_response_text=None),
        _run("r3", "openai_chatgpt"),
    ]
    result = s3.compute(_inputs(raw_runs, sources_by_run={"r3": []}))
    assert result.status == "ok"
    assert result.denominator_after_exclusion["anthropic_claude"] == 0  # nothing usable -> the per-engine slice is untestable
    assert result.n_included["anthropic_claude"] == 0
    assert result.n_excluded["anthropic_claude"] == {"no_response_text": 2}
    assert result.value["anthropic_claude"] is None  # untestable slice, not a fabricated 0/0


def test_s3_denominator_always_equals_n_included_for_every_engine():
    """The invariant the author display surface's integrity check relies
    on: denominator_after_exclusion is always the true divisor value was
    computed over, which for S3 is n_included -- never the pre-exclusion
    pool, even when real exclusions happened."""
    raw_runs = [
        _run("r1", "google_gemini", error="500 INTERNAL"),
        _run("r2", "google_gemini"),
        _run("r3", "google_gemini", raw_response_text=""),
    ]
    result = s3.compute(_inputs(raw_runs, sources_by_run={"r2": []}))
    for engine in result.denominator_after_exclusion:
        assert result.n_included[engine] == result.denominator_after_exclusion[engine]


def test_s3_all_engines_zero_usable_collapses_to_untestable_figure():
    raw_runs = [_run("r1", "anthropic_claude", raw_response_text="")]
    result = s3.compute(_inputs(raw_runs, sources_by_run={}))
    assert result.status == "untestable"
    assert result.reason_untestable is not None
    assert result.value is None
    assert result.denominator_after_exclusion is None


def test_s3_excludes_manual_ui_calibration_rows():
    raw_runs = [_run("r1", "google_ai_mode", is_manual=1), _run("r2", "google_ai_mode")]
    sources_by_run = {"r2": [{"url_raw": "https://x.com"}]}
    result = s3.compute(_inputs(raw_runs, sources_by_run))
    assert result.denominator_after_exclusion["google_ai_mode"] == 1  # what value was actually computed over
    assert result.n_included["google_ai_mode"] == 1
    assert result.n_excluded["google_ai_mode"] == {"manual_ui_calibration": 1}


def test_s3_search_fired_is_supporting_table_only_not_folded_into_value():
    raw_runs = [_run("r1", "openai_chatgpt", search_fired=False)]
    sources_by_run = {"r1": []}
    result = s3.compute(_inputs(raw_runs, sources_by_run))
    # search_fired=False does not exclude the run or change the share
    assert result.denominator_after_exclusion["openai_chatgpt"] == 1
    assert result.n_included["openai_chatgpt"] == 1
    assert result.value["openai_chatgpt"] == 1.0
    row = result.supporting_table[0]
    assert row["search_fired"] is False


def test_s3_threshold_and_reason_untestable_are_explicit_none():
    result = s3.compute(_inputs([_run("r1", "openai_chatgpt")], {"r1": []}))
    assert result.threshold is None
    assert result.reason_untestable is None
    assert result.status == "ok"


def test_s3_per_engine_breakdown_independent():
    raw_runs = [_run("r1", "openai_chatgpt"), _run("r2", "google_gemini")]
    sources_by_run = {"r1": [], "r2": [{"url_raw": "https://x.com"}]}
    result = s3.compute(_inputs(raw_runs, sources_by_run))
    assert set(result.value) == {"openai_chatgpt", "google_gemini"}
    assert result.value["openai_chatgpt"] == 1.0
    assert result.value["google_gemini"] == 0.0


def test_s3_supporting_table_has_one_row_per_run():
    raw_runs = [_run("r1", "openai_chatgpt"), _run("r2", "openai_chatgpt", error="x")]
    result = s3.compute(_inputs(raw_runs, sources_by_run={"r1": []}))
    assert len(result.supporting_table) == 2


def test_s3_supporting_table_carries_real_paraphrase_id_column():
    raw_runs = [_run("r1", "openai_chatgpt", paraphrase_id=3)]
    result = s3.compute(_inputs(raw_runs, sources_by_run={"r1": []}))
    assert result.supporting_table[0]["paraphrase_id"] == 3


def test_s3_notes_identity_true_when_zero_sources_exactly_matches_no_search():
    raw_runs = [
        _run("r1", "google_gemini", search_fired=False),  # no sources, no search
        _run("r2", "google_gemini", search_fired=True),  # has sources, searched
    ]
    sources_by_run = {"r1": [], "r2": [{"url_raw": "https://x.com"}]}
    result = s3.compute(_inputs(raw_runs, sources_by_run))
    identity = result.notes["search_fired_zero_sources_identity"]["google_gemini"]
    assert identity == {"n_zero_sources": 1, "n_no_search": 1, "n_both": 1, "identical_sets": True}


def test_s3_notes_identity_false_when_zero_sources_and_no_search_diverge():
    # google_ai_mode-shaped case: search_fired always True, but a run can
    # still come back with zero sources for an unrelated reason
    raw_runs = [_run("r1", "google_ai_mode", search_fired=True)]
    result = s3.compute(_inputs(raw_runs, sources_by_run={"r1": []}))
    identity = result.notes["search_fired_zero_sources_identity"]["google_ai_mode"]
    assert identity["identical_sets"] is False
    assert identity == {"n_zero_sources": 1, "n_no_search": 0, "n_both": 0, "identical_sets": False}


def test_s3_notes_identity_computed_independently_per_engine():
    raw_runs = [
        _run("r1", "google_gemini", search_fired=False),
        _run("r2", "google_ai_mode", search_fired=True),
    ]
    sources_by_run = {"r1": [], "r2": []}
    result = s3.compute(_inputs(raw_runs, sources_by_run))
    identities = result.notes["search_fired_zero_sources_identity"]
    assert identities["google_gemini"]["identical_sets"] is True
    assert identities["google_ai_mode"]["identical_sets"] is False
