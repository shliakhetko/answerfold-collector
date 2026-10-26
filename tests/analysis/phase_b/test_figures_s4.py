from analysis.phase_b.figures import FIGURES, s4
from analysis.phase_b.reading import FrozenInputs


def _inputs(raw_runs):
    return FrozenInputs(
        entities=[], alt_domains_rows=[], entity_verification_rows=[],
        mentions_raw=[], raw_runs=raw_runs, sources_by_run={},
    )


def _run(run_id, engine, intent, para, search_fired, error=None):
    return {
        "run_id": run_id, "engine": engine, "intent_id": intent, "paraphrase_id": para,
        "day_index": 1, "rep_index": 1, "error": error, "raw_response_text": "text",
        "is_manual": None, "search_fired": search_fired,
    }


def test_s4_registered():
    assert "S4" in FIGURES


def test_s4_no_runs_is_untestable():
    assert s4.compute(_inputs([])).status == "untestable"


def test_s4_share_per_cell_and_denominators():
    runs = [
        _run("a", "google_gemini", 1, 3, False),
        _run("b", "google_gemini", 1, 3, True),
        _run("c", "google_gemini", 1, 3, False),
        _run("d", "google_gemini", 2, 1, True),
        _run("e", "google_gemini", 2, 1, None),          # unknown -> excluded
        _run("f", "google_gemini", 2, 1, True, error="boom"),  # errored -> excluded
    ]
    result = s4.compute(_inputs(runs))
    v = result.value["google_gemini"]
    assert v["i1-p3"] == 2 / 3
    assert v["i2-p1"] == 0.0
    assert result.denominator_before_exclusion["google_gemini"] == 6
    assert result.denominator_after_exclusion["google_gemini"] == 4
    assert result.n_included["google_gemini"] == 4
    assert result.n_excluded["google_gemini"]["search_fired_not_reported"] == 1
    assert result.threshold is None
    assert "s3-search-fired-zero-sources-identity-noticed-2026-09-12" in result.notes["ledger_refs"]
