"""Personal-data cleaning of the publish package (scripts/publish/clean_personal_data.py).

Runs on a tiny synthetic package. Needs the spaCy model en_core_web_lg (the
cleaning dependency group); skips when it is not installed.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

spacy = pytest.importorskip("spacy")
pytest.importorskip("presidio_analyzer")
if not spacy.util.is_package("en_core_web_lg"):
    pytest.skip("en_core_web_lg is not installed", allow_module_level=True)

from scripts.publish import clean_personal_data as cp  # noqa: E402

CHATGPT_TEXT = (
    "Founded by John Doe, Goodcall is a solid pick. Email jane.roe@gmail.com or support@goodcall.com, "
    "call (415) 555-0132. Office: 1600 Amphitheatre Parkway, Mountain View, CA 94043. "
    "Profiles: https://www.linkedin.com/in/john-doe-123/?trk=a and https://linkedin.com/in/john-doe-123 and "
    "https://www.linkedin.com/in/mary-major/ and https://www.linkedin.com/company/goodcall and "
    "https://www.facebook.com/Goodcall/posts/1. Ask Michael Johnson about it."
)


def _write_csv(path: Path, rows: list[dict], comment: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if comment:
            f.write(comment + "\n")
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_package(root: Path) -> Path:
    _write_csv(root / "data/coding/company_table.csv", [
        {"entity_id": "goodcall", "canonical_name": "Goodcall", "homepage_url": "https://goodcall.com/", "meets_frame": "in",
         "in_seed_list": "False", "segment": "", "evidence_url": "", "frame_reason": "", "surface_forms": "Goodcall;GoodCall"},
    ], comment="# company_table")
    _write_csv(root / "data/coding/company_verification.csv", [
        {"entity_id": "goodcall", "is_real": "True", "entity_type": "vendor", "verified_homepage_url": "https://goodcall.com/",
         "disambiguation_note": "Goodcall was started by Bob Marley in Austin.", "confidence": "0.9"},
    ])
    _write_csv(root / "data/coding/url_classes.csv", [
        {"url_normalized": "https://linkedin.com/in/john-doe-123", "url_class": "other", "publisher_is_in_category": "0",
         "classifier_confidence": "0.5", "settled_class": "other", "settled_publisher_is_in_category": "0"},
    ], comment="# url_classes")
    (root / "config/prompts").mkdir(parents=True)
    _write_csv(root / "config/prompts/prompts.csv", [
        {"intent_id": "7", "paraphrase_id": "2", "prompt_text": "what are some alternatives to Goodcall", "branded": "1"},
    ])
    record = {
        "run_id": "openai_chatgpt-i1-p1-d1-r1", "engine": "openai_chatgpt",
        "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": CHATGPT_TEXT, "annotations": [
            {"type": "url_citation", "url": "https://www.linkedin.com/in/john-doe-123/?trk=a", "title": "John Doe - LinkedIn"}]}]}]},
    }
    gemini = {"run_id": "google_gemini-i1-p1-d1-r1", "engine": "google_gemini", "response": {"candidates": [{
        "grounding_metadata": {"grounding_chunks": [{"web": {"title": "linkedin.com", "uri": "https://www.linkedin.com/in/mary-major"}}]}}]}}
    (root / "data/raw").mkdir(parents=True)
    (root / "data/raw/2026-08-20.jsonl").write_text(json.dumps(record) + "\n" + json.dumps(gemini) + "\n", encoding="utf-8")
    (root / "cleaning").mkdir()
    return root


def run(tmp_path: Path, name: str, remove: str | None = None) -> tuple[Path, Path, cp.Result]:
    root = make_package(tmp_path / name)
    if remove:
        (root / cp.REMOVE_LIST).write_text("# manual\n" + remove + "\n", encoding="utf-8")
    private = tmp_path / f"{name}_private"
    result = cp.clean_package(root, private, log=lambda *_: None)
    return root, private, result


def test_two_runs_on_the_same_input_give_identical_outputs(tmp_path):
    a_root, a_private, _ = run(tmp_path, "a")
    b_root, b_private, _ = run(tmp_path, "b")
    assert cp.tree_digest(a_root) == cp.tree_digest(b_root)
    assert cp.tree_digest(a_private) == cp.tree_digest(b_private)


def test_all_five_types_are_replaced_and_the_keep_list_holds(tmp_path):
    root, private, result = run(tmp_path, "p")
    record = json.loads((root / "data/raw/2026-08-20.jsonl").read_text(encoding="utf-8").splitlines()[0])
    text = record["response"]["output"][0]["content"][0]["text"]
    for gone in ("John Doe", "jane.roe", "555-0132", "1600 Amphitheatre", "john-doe-123", "mary-major", "Michael Johnson"):
        assert gone not in text, gone
    for token in ("[PERSON]", "[EMAIL]", "[PHONE]", "[ADDRESS]", "[PROFILE-1]", "[PROFILE-2]"):
        assert token in text, token
    assert "Goodcall" in text and "[EMAIL]@goodcall.com" in text  # keep-list name and domain never removed
    assert "https://www.linkedin.com/company/goodcall" in text  # company LinkedIn page untouched
    assert "https://www.facebook.com/Goodcall/posts/1" in text  # a company's own profile page is kept
    types = {h.type for h in result.removed}
    assert {cp.EMAIL, cp.PHONE, cp.ADDRESS, cp.PERSON, cp.PROFILE} <= types
    assert any(h.reason == cp.REASON_COMPANY_URL for h in result.kept)


def test_profile_tokens_are_stable_one_per_page_and_never_shared(tmp_path):
    root, _, _ = run(tmp_path, "t")
    lines = (root / "data/raw/2026-08-20.jsonl").read_text(encoding="utf-8").splitlines()
    chatgpt, gemini = json.loads(lines[0]), json.loads(lines[1])
    text = chatgpt["response"]["output"][0]["content"][0]["text"]
    # the same page (with and without query, www, trailing slash) gets one number; a different page another
    assert text.count("[PROFILE-1]") == 2 and text.count("[PROFILE-2]") == 1
    assert chatgpt["response"]["output"][0]["content"][0]["annotations"][0]["url"] == "https://www.linkedin.com/[PROFILE-1]"
    assert gemini["response"]["candidates"][0]["grounding_metadata"]["grounding_chunks"][0]["web"]["uri"] == "https://www.linkedin.com/[PROFILE-2]"
    # the classified URL table gets the same token, so citations still join to their class
    rows = list(csv.DictReader(l for l in (root / "data/coding/url_classes.csv").open(encoding="utf-8") if not l.startswith("#")))
    assert rows[0]["url_normalized"] == "https://linkedin.com/[PROFILE-1]"
    assert (root / "data/coding/url_classes.csv").read_text(encoding="utf-8").startswith("# url_classes")


def test_keep_list_is_generated_and_remove_list_is_honoured(tmp_path):
    root, _, result = run(tmp_path, "r", remove="solid pick")
    keep = (root / cp.KEEP_LIST).read_text(encoding="utf-8").splitlines()
    for term in ("Goodcall", "GoodCall", "goodcall.com", "ChatGPT", "Claude", "Gemini", "Google AI Mode"):
        assert term in keep, term
    text = json.loads((root / "data/raw/2026-08-20.jsonl").read_text(encoding="utf-8").splitlines()[0])["response"]["output"][0]["content"][0]["text"]
    assert "solid pick" not in text and "[REMOVED]" in text
    assert any(h.type == cp.REMOVED for h in result.removed)


def test_private_outputs_hold_values_and_the_shipped_report_does_not(tmp_path):
    root, private, result = run(tmp_path, "o")
    log = list(csv.DictReader((private / "cleaning_log.csv").open(encoding="utf-8")))
    assert list(log[0]) == cp.LOG_COLUMNS
    assert list(csv.DictReader((private / "kept_hits.csv").open(encoding="utf-8")).fieldnames) == cp.KEPT_COLUMNS
    report = (root / cp.REPORT).read_text(encoding="utf-8")
    private_report = (private / "CLEANING_REPORT_private.md").read_text(encoding="utf-8")
    for section in ("Removed and kept, per type and engine", "Most frequent removed values", "Kept hits with score",
                    "Possible misses", "What this cleaning cannot catch"):
        assert section in report and section in private_report, section
    for hit in result.removed:
        assert hit.original not in report, hit.original
    assert "John Doe" in private_report
    assert not list(root.rglob("cleaning_log.csv")) and not list(root.rglob("kept_hits.csv"))


def test_private_dir_inside_the_package_is_refused(tmp_path):
    root = make_package(tmp_path / "x")
    with pytest.raises(SystemExit):
        cp.clean_package(root, root / "private", log=lambda *_: None)


def test_profile_handle_rules():
    assert cp.profile_handle("www.linkedin.com", "/in/someone/") == "someone"
    assert cp.profile_handle("www.linkedin.com", "/company/acme") is None
    assert cp.profile_handle("x.com", "/someone/status/1") == "someone"
    assert cp.profile_handle("analytics.twitter.com", "/i/") is None
    assert cp.profile_handle("www.facebook.com", "/tr/") is None
    assert cp.profile_handle("www.facebook.com", "/groups/123/posts/4") is None
    assert cp.profile_handle("www.facebook.com", "/chris.greene.1/posts/2") == "chris.greene.1"
    assert cp.profile_handle("www.instagram.com", "/reel/abc/") is None
    assert cp.profile_handle("www.tiktok.com", "/@someone/video/1") == "someone"
    assert cp.profile_handle("www.youtube.com", "/watch") is None


def test_one_word_person_hits_score_below_the_threshold():
    assert cp.person_score("Voicemail", 0.85) < cp.REMOVE_AT_OR_ABOVE
    assert cp.person_score("john doe", 0.85) < cp.REMOVE_AT_OR_ABOVE
    assert cp.person_score("John Doe", 0.85) >= cp.REMOVE_AT_OR_ABOVE


# ---------------------------------------------------------------- the built package (skips when publish/ has no cleaning)

from collector import paths  # noqa: E402

PUBLISH = paths.PUBLISH_DIR
needs_cleaned_publish = pytest.mark.skipif(not (PUBLISH / cp.REPORT).exists(), reason="publish/ has not been cleaned")


@needs_cleaned_publish
def test_cleaning_ships_code_lists_and_report_and_no_private_log():
    for rel in (cp.KEEP_LIST, cp.REMOVE_LIST, cp.REPORT, cp.CLEANING_DIR / "clean_personal_data.py"):
        assert (PUBLISH / rel).exists(), rel
    assert (PUBLISH / cp.REMOVE_LIST).read_text(encoding="utf-8").startswith("#")
    names = {p.name for p in PUBLISH.rglob("*") if p.is_file()}
    assert not names & {"cleaning_log.csv", "kept_hits.csv", "CLEANING_REPORT_private.md"}
    # publish/ is frozen as deposited on Zenodo; the source may have moved on since.
    assert (PUBLISH / cp.CLEANING_DIR / "clean_personal_data.py").is_file()


@needs_cleaned_publish
def test_shipped_report_holds_no_removed_value():
    private_dir = os.environ.get("ANSWERFOLD_CLEANING_PRIVATE_DIR")
    log = Path(private_dir) / "cleaning_log.csv" if private_dir else None
    if log is None or not log.exists():
        pytest.skip("set ANSWERFOLD_CLEANING_PRIVATE_DIR to the folder holding the private cleaning log")
    report = (PUBLISH / cp.REPORT).read_text(encoding="utf-8")
    originals = {r["original"] for r in csv.DictReader(log.open(encoding="utf-8")) if len(r["original"]) >= 6}
    assert originals and not [o for o in originals if o in report]
