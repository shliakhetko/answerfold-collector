import hashlib
import json

import pytest

from analysis.phase_b.reading import PhaseBInputError, read_frozen_inputs
from collector import paths


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _build_valid_repo(tmp_path):
    """Minimal but complete frozen-input fixture: one entity, one mention,
    one run with one source, checksums computed to match."""
    entity_csv = "# entity_registry\nentity_id,canonical_name,homepage_url,meets_frame,in_seed_list,segment,evidence_url,frame_reason,surface_forms\nacme,Acme,https://acme.com,in,True,done-for-you,,,Acme\n"
    alt_csv = "# alt_domains\nentity_id,alt_domain,source_note\n"
    entity_verification_csv = "# entity_verification\nentity_id,is_real,entity_type,verified_homepage_url,disambiguation_note,confidence\n"
    mentions_jsonl = json.dumps({"run_id": "r1", "surface_form": "Acme", "evidence_span": "x", "is_recommended": False, "votes": ["a"]}) + "\n"
    url_registry_csv = "# url_registry\nurl_normalized,url_class,publisher_is_in_category,classifier_confidence,settled_class,settled_publisher_is_in_category\n"
    report_json = "{}"
    spot_check_csv = "# spot_check\nurl_normalized,classifier_confidence,model_url_class,model_publisher_is_in_category,human_class,human_publisher_is_in_category,note\n"

    coding_dir = tmp_path / paths.rel(paths.CODED_DIR)
    files = {
        "entity_registry.csv": entity_csv,
        "alt_domains.csv": alt_csv,
        "entity_verification.csv": entity_verification_csv,
        "mentions_raw.jsonl": mentions_jsonl,
        "url_registry_frozen.csv": url_registry_csv,
        "url_classification_spot_check.csv": spot_check_csv,
    }
    artifacts = []
    for name, content in files.items():
        path = _write(coding_dir / name, content)
        artifacts.append({"path": f"data/03_coded/{name}", "sha256": _sha256_bytes(content.encode()), "bytes": len(content), "lines": content.count("\n")})

    report_path = _write(tmp_path / paths.rel(paths.CODING_REPORT_JSON), report_json)
    artifacts.append({"path": "data/03_coded/coding_a1_report.json", "sha256": _sha256_bytes(report_json.encode()), "bytes": len(report_json), "lines": 0})

    _write(coding_dir / "MANIFEST.json", json.dumps({"artifacts": artifacts}))

    raw_record = {
        "run_id": "r1",
        "ts_utc": "2026-08-20T14:00:00+00:00",
        "engine": "openai_chatgpt",
        "intent_id": 1,
        "paraphrase_id": 1,
        "rep_index": 1,
        "day_index": 1,
        "request": {"model": "gpt-5"},
        "response": {"output": []},
    }
    raw_line = json.dumps(raw_record) + "\n"
    raw_path = _write(tmp_path / paths.rel(paths.RUNS_DIR) / "2026-08-20.jsonl", raw_line)
    # data/01_raw/raw_frozen mirrors runs/ per the real convention
    frozen_path = _write(tmp_path / paths.rel(paths.RAW_FROZEN_DIR) / "2026-08-20.jsonl", raw_line)
    raw_manifest = {
        "collector_version": "test/0.0.0+deadbeef",
        "raw_files": [{"path": "data/01_raw/raw_frozen/2026-08-20.jsonl", "sha256": _sha256_bytes(raw_line.encode()), "bytes": len(raw_line), "lines": 1}],
    }
    _write(tmp_path / paths.rel(paths.RAW_MANIFEST), json.dumps(raw_manifest))
    _write(tmp_path / paths.rel(paths.AUX_MANIFEST), json.dumps({"files": []}))
    _write(tmp_path / paths.rel(paths.UI_TEST_DIR) / "MANIFEST.json", json.dumps({"files": []}))
    return tmp_path


def test_read_frozen_inputs_succeeds_on_valid_fixture(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    inputs = read_frozen_inputs(repo_root, repo_root / "data")
    assert len(inputs.entities) == 1
    assert len(inputs.mentions_raw) == 1
    assert len(inputs.raw_runs) == 1
    assert inputs.raw_runs[0]["run_id"] == "r1"
    assert inputs.inputs_manifest["instrument_commit"] == "test/0.0.0+deadbeef"
    assert inputs.entity_verification_rows == []


def test_read_frozen_inputs_no_resolution_file_is_a_noop(tmp_path):
    """data/02_aux/cited_urls/resolution.jsonl is a post-window aux snapshot,
    not a checksummed Phase-A artifact -- its absence must never raise,
    only skip the swap (matches load_resolution_map's own contract)."""
    repo_root = _build_valid_repo(tmp_path)
    inputs = read_frozen_inputs(repo_root, repo_root / "data")
    assert inputs.inputs_manifest["cited_url_resolution_rows_loaded"] == 0
    assert inputs.sources_by_run == {"r1": []}


def test_read_frozen_inputs_swaps_reachable_citation_via_resolution_file(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    raw_record = {
        "run_id": "r2",
        "ts_utc": "2026-08-20T14:00:00+00:00",
        "engine": "google_gemini",
        "intent_id": 1,
        "paraphrase_id": 1,
        "rep_index": 1,
        "day_index": 1,
        "request": {"model": "gemini"},
        "response": {
            "candidates": [
                {
                    "grounding_metadata": {
                        "grounding_chunks": [
                            {"web": {"uri": "https://vertexaisearch.example/redirect/abc", "title": "wrapped"}}
                        ]
                    }
                }
            ]
        },
    }
    raw_line_r1 = (repo_root / paths.rel(paths.RAW_FROZEN_DIR) / "2026-08-20.jsonl").read_text(encoding="utf-8")
    raw_line_r2 = json.dumps(raw_record) + "\n"
    combined = raw_line_r1 + raw_line_r2
    (repo_root / paths.rel(paths.RAW_FROZEN_DIR) / "2026-08-20.jsonl").write_text(combined, encoding="utf-8")
    (repo_root / paths.rel(paths.RUNS_DIR) / "2026-08-20.jsonl").write_text(combined, encoding="utf-8")
    raw_manifest = {
        "collector_version": "test/0.0.0+deadbeef",
        "raw_files": [
            {
                "path": "data/01_raw/raw_frozen/2026-08-20.jsonl",
                "sha256": _sha256_bytes(combined.encode()),
                "bytes": len(combined),
                "lines": 2,
            }
        ],
    }
    _write(repo_root / paths.rel(paths.RAW_MANIFEST), json.dumps(raw_manifest))

    resolution_line = json.dumps(
        {
            "url": "https://vertexaisearch.example/redirect/abc",
            "status_bucket": "ok",
            "final_url": "https://realpublisher.com/post",
            "final_registrable_domain": "realpublisher.com",
        }
    ) + "\n"
    _write(repo_root / paths.rel(paths.AUX_DIR) / "cited_urls" / "resolution.jsonl", resolution_line)

    inputs = read_frozen_inputs(repo_root, repo_root / "data")
    assert inputs.inputs_manifest["cited_url_resolution_rows_loaded"] == 1
    r2_sources = inputs.sources_by_run.get("r2", [])
    assert r2_sources
    assert r2_sources[0]["url_raw"] == "https://realpublisher.com/post"
    assert r2_sources[0]["domain"] == "realpublisher.com"
    assert r2_sources[0]["url_cited"] == "https://vertexaisearch.example/redirect/abc"


def test_read_frozen_inputs_loads_url_registry_rows_with_bool_coercion(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    url_registry_csv = (
        "# url_registry\n"
        "url_normalized,url_class,publisher_is_in_category,classifier_confidence,"
        "settled_class,settled_publisher_is_in_category\n"
        "https://acme.com,vendor page,1,0.95,vendor page,1\n"
        "https://reviews.example,review site,0,0.80,review site,0\n"
    )
    path = repo_root / paths.rel(paths.CODED_DIR) / "url_registry_frozen.csv"
    path.write_text(url_registry_csv, encoding="utf-8")
    manifest_path = repo_root / paths.rel(paths.CODED_DIR) / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        if entry["path"].endswith("url_registry_frozen.csv"):
            entry["sha256"] = _sha256_bytes(url_registry_csv.encode())
            entry["bytes"] = len(url_registry_csv)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    inputs = read_frozen_inputs(repo_root, repo_root / "data")
    assert len(inputs.url_registry_rows) == 2
    acme = next(r for r in inputs.url_registry_rows if r["url_normalized"] == "https://acme.com")
    assert acme["publisher_is_in_category"] is True
    assert acme["settled_publisher_is_in_category"] is True
    reviews = next(r for r in inputs.url_registry_rows if r["url_normalized"] == "https://reviews.example")
    # real bool, never the raw CSV string -- "0" is truthy as a Python str,
    # so this must be the coerced False, not the literal "0" passed through
    assert reviews["publisher_is_in_category"] is False
    assert isinstance(reviews["publisher_is_in_category"], bool)


def test_read_frozen_inputs_applies_gemini_title_fallback_when_never_captured(tmp_path):
    """A Gemini citation with no row at all in resolution.jsonl (the aux
    capture never matched it) still gets recovered via the title field --
    the same condition the old is_dead-based check would have missed."""
    repo_root = _build_valid_repo(tmp_path)
    wrapped = "https://vertexaisearch.example/redirect/xyz"
    raw_record = {
        "run_id": "r3",
        "ts_utc": "2026-08-20T14:00:00+00:00",
        "engine": "google_gemini",
        "intent_id": 1,
        "paraphrase_id": 1,
        "rep_index": 1,
        "day_index": 1,
        "request": {"model": "gemini"},
        "response": {
            "candidates": [
                {
                    "grounding_metadata": {
                        "grounding_chunks": [{"web": {"uri": wrapped, "title": "acme.com"}}]
                    }
                }
            ]
        },
    }
    raw_line_r1 = (repo_root / paths.rel(paths.RAW_FROZEN_DIR) / "2026-08-20.jsonl").read_text(encoding="utf-8")
    raw_line_r3 = json.dumps(raw_record) + "\n"
    combined = raw_line_r1 + raw_line_r3
    (repo_root / paths.rel(paths.RAW_FROZEN_DIR) / "2026-08-20.jsonl").write_text(combined, encoding="utf-8")
    (repo_root / paths.rel(paths.RUNS_DIR) / "2026-08-20.jsonl").write_text(combined, encoding="utf-8")
    raw_manifest = {
        "collector_version": "test/0.0.0+deadbeef",
        "raw_files": [
            {
                "path": "data/01_raw/raw_frozen/2026-08-20.jsonl",
                "sha256": _sha256_bytes(combined.encode()),
                "bytes": len(combined),
                "lines": 2,
            }
        ],
    }
    _write(repo_root / paths.rel(paths.RAW_MANIFEST), json.dumps(raw_manifest))

    inputs = read_frozen_inputs(repo_root, repo_root / "data")
    r3_sources = inputs.sources_by_run.get("r3", [])
    assert r3_sources
    assert r3_sources[0]["resolution_method"] == "title_fallback"
    assert r3_sources[0]["url_raw"] == "https://acme.com/"
    assert r3_sources[0]["domain"] == "acme.com"
    assert inputs.inputs_manifest["cited_url_resolution"]["citations_title_fallback_recovered"] == 1


def test_read_frozen_inputs_raises_on_phase_a_checksum_mismatch(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    # corrupt a manifested Phase-A file after its checksum was recorded
    (repo_root / paths.rel(paths.CODED_DIR) / "mentions_raw.jsonl").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(PhaseBInputError, match="checksum mismatch"):
        read_frozen_inputs(repo_root, repo_root / "data")


def test_read_frozen_inputs_raises_on_missing_phase_a_file(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    (repo_root / paths.rel(paths.CODED_DIR) / "alt_domains.csv").unlink()
    with pytest.raises(PhaseBInputError, match="missing"):
        read_frozen_inputs(repo_root, repo_root / "data")


def test_read_frozen_inputs_raises_on_raw_manifest_checksum_mismatch(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    (repo_root / paths.rel(paths.RAW_FROZEN_DIR) / "2026-08-20.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PhaseBInputError, match="checksum mismatch"):
        read_frozen_inputs(repo_root, repo_root / "data")


def test_read_frozen_inputs_raises_on_missing_manifest(tmp_path):
    repo_root = tmp_path
    with pytest.raises(PhaseBInputError, match="manifest not found"):
        read_frozen_inputs(repo_root, repo_root / "data")


def _manifest_entry(path, rel):
    return {"path": rel, "sha256": _sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size}


def test_no_data_path_fails_loudly_and_has_no_default(tmp_path, monkeypatch):
    from analysis.phase_b.reading import resolve_data_root

    monkeypatch.delenv("GEO_DATA_ROOT", raising=False)
    with pytest.raises(PhaseBInputError, match="no data path"):
        resolve_data_root(None)
    with pytest.raises(PhaseBInputError, match="does not exist"):
        resolve_data_root(tmp_path / "nope")
    monkeypatch.setenv("GEO_DATA_ROOT", str(tmp_path))
    assert resolve_data_root(None) == tmp_path


def test_run_phase_b_main_stops_without_data_path(monkeypatch, capsys):
    run_phase_b = pytest.importorskip("scripts.results.run_phase_b")

    monkeypatch.delenv("GEO_DATA_ROOT", raising=False)
    assert run_phase_b.main([]) == 1
    assert "no data path" in capsys.readouterr().err


def test_aux_manifest_mismatch_and_missing_file_stop_the_build(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    f = _write(repo_root / paths.rel(paths.AUX_DIR) / "roster_provisional.csv", "entity_id,primary_domain\n")
    manifest = repo_root / paths.rel(paths.AUX_MANIFEST)
    manifest.write_text(json.dumps({"files": [_manifest_entry(f, "data/02_aux/roster_provisional.csv")]}))
    read_frozen_inputs(repo_root, repo_root / "data")  # intact -> ok
    f.write_text("tampered\n")
    with pytest.raises(PhaseBInputError, match="aux: .*checksum mismatch"):
        read_frozen_inputs(repo_root, repo_root / "data")
    f.unlink()
    with pytest.raises(PhaseBInputError, match="aux: .*missing"):
        read_frozen_inputs(repo_root, repo_root / "data")


def test_ui_manifest_mismatch_stops_the_build(tmp_path):
    repo_root = _build_valid_repo(tmp_path)
    f = _write(repo_root / paths.rel(paths.UI_TEST_DIR) / "I01_P01" / "gemini_I01_P01_R1.html", "<html></html>")
    (repo_root / paths.rel(paths.UI_TEST_DIR) / "MANIFEST.json").write_text(
        json.dumps({"files": [_manifest_entry(f, "data/01_raw/ui_test/I01_P01/gemini_I01_P01_R1.html")]})
    )
    read_frozen_inputs(repo_root, repo_root / "data")
    f.write_text("<html>changed</html>")
    with pytest.raises(PhaseBInputError, match="ui_test: .*checksum mismatch"):
        read_frozen_inputs(repo_root, repo_root / "data")


def test_runs_manifest_paths_map_to_raw_frozen_under_the_data_root(tmp_path):
    from analysis.phase_b.reading import resolve_data_path

    assert resolve_data_path(tmp_path, paths.rel(paths.RUNS_DIR) / "2026-08-20.jsonl") == tmp_path / paths.data_rel(paths.RAW_FROZEN_DIR) / "2026-08-20.jsonl"
    assert resolve_data_path(tmp_path, paths.rel(paths.CODED_DIR) / "x.csv") == tmp_path / paths.data_rel(paths.CODED_DIR) / "x.csv"
