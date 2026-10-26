from analysis.phase_b.figures import ui_calibration as uc
from analysis.phase_b.reading import FrozenInputs
from analysis.phase_b.ui_calibration import extract_answer_text


def _entity(eid, name):
    return {"entity_id": eid, "canonical_name": name, "surface_forms": name, "meets_frame": "in"}


def _api(run_id, text):
    return {"run_id": run_id, "engine": "google_gemini", "intent_id": 1, "paraphrase_id": 1, "day_index": 1,
            "rep_index": 1, "error": None, "raw_response_text": text, "is_manual": None}


def _ui(key, text, miss=None):
    return {"run_key": key, "ui": "gemini", "engine": "google_gemini", "intent_id": 1, "paraphrase_id": 1,
            "rep": 1, "text": text, "extraction_miss": miss}


def _inputs(ui_runs, api_runs, eyeball=None):
    return FrozenInputs(
        entities=[_entity("smith", "Smith.ai"), _entity("goodcall", "Goodcall")], alt_domains_rows=[],
        entity_verification_rows=[], mentions_raw=[], raw_runs=api_runs, sources_by_run={},
        ui_test_runs=ui_runs, ui_eyeball=eyeball or {},
    )


def test_raw_counts_same_matcher_both_sides_and_miss_listing():
    ui = [_ui("a", "Try Smith.ai and Goodcall"), _ui("b", "Only Smith.ai"), _ui("c", None, "container_matches_0_expected_1")]
    api = [_api("r1", "Smith.ai is good"), _api("r2", "nothing")]
    r = uc.compute(_inputs(ui, api))
    cell = r.value["google_gemini"]["i1-p1"]
    assert cell["ui_runs"] == 3 and cell["ui_runs_parsed"] == 2 and cell["api_runs"] == 2
    assert cell["companies_named_in_ui"] == 2 and cell["of_which_also_named_in_api"] == 1
    assert cell["ui_only_companies"] == ["goodcall"]
    by = {c["entity_id"]: c for c in cell["per_company"]}
    assert by["smith"]["ui_runs_with"] == "2 of 2" and by["smith"]["api_runs_with"] == "1 of 2"
    assert r.n_excluded == {"container_extraction_miss": 1}
    assert r.notes["extraction_misses"][0]["run_key"] == "c"
    assert r.threshold is None and r.verdict is None
    assert "Not a finding" in r.notes["caveats"]


def test_extraction_uses_the_container_only():
    html = ("<html><nav>Smith.ai in the sidebar</nav><model-response><message-content>"
            "<div class='markdown'>Goodcall answer</div></message-content></model-response></html>")
    text, miss = extract_answer_text("gemini", html)
    assert text == "Goodcall answer" and miss is None
    assert extract_answer_text("gemini", "<html></html>")[1] == "container_matches_0_expected_1"


def test_eyeball_summary_counts():
    eye = {"runs": {"a": {"png_checked": True, "container_extraction": "ok", "generic_phrase_false_positive": ["AI Receptionist"],
                          "source_chip_only": [], "ambiguous_or_duplicate_entity": [], "visible_vendor_not_matched": []}}}
    r = uc.compute(_inputs([_ui("a", "Smith.ai")], [_api("r1", "x")], eye))
    s = r.notes["eyeball"]
    assert s["n_checked_by_eye_against_png"] == 1 and s["n_runs_strict_agreement"] == 0
    assert s["n_runs_agreement_ignoring_generic_phrase_hits"] == 1
