"""Smoke + behaviour tests for the scripts.aux.* CLIs. No network, no API
keys: the collectors are exercised only through --dry-run and the
confirmation gate (which refuses on a non-TTY without --confirm)."""
import json

import pytest

from analysis.coding import storage as coding_storage
from analysis.coding.db import init_coding_schema
from collector.storage.db import get_connection, init_db
from scripts.aux import (
    aux_summary,
    audit_collection,
    build_roster,
    collect_cited_urls,
    collect_directories,
    collect_google_serp,
    collect_robots,
    collect_site_inventory,
    freeze_window,
    update_ambiguity_ledger,
)

RUN_LINE = {
    "run_id": "openai_chatgpt-i1-p1-d1-r1",
    "ts_utc": "2026-08-20T14:00:05+00:00",
    "engine": "openai_chatgpt",
    "intent_id": 1,
    "paraphrase_id": 1,
    "rep_index": 1,
    "day_index": 1,
    "prompt_order_seed": 1,
    "position_in_pass": 0,
    "request": {"model": "gpt-5.6-luna"},
    "response": {
        "model": "gpt-5.6-luna",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "You should try Goodcall, it is a great AI receptionist for small business owners." * 3,
                        "annotations": [{"type": "url_citation", "url": "https://goodcall.com/", "title": "Goodcall"}],
                    }
                ],
            },
            {"type": "web_search_call", "action": {"queries": ["best ai receptionist"]}},
        ],
    },
}


@pytest.fixture
def runs_dir(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    (d / "2026-08-20.jsonl").write_text(json.dumps(RUN_LINE) + "\n")
    return d


@pytest.fixture
def coding_db(tmp_path):
    db = tmp_path / "coding.sqlite3"
    init_db(db)
    init_coding_schema(db)
    conn = get_connection(db)
    coding_storage.insert_entities(
        [
            {"entity_id": "goodcall", "canonical_name": "Goodcall", "homepage_url": "https://goodcall.com", "surface_forms": ["Goodcall"]},
            {"entity_id": "verifonly", "canonical_name": "VerifOnly", "homepage_url": None, "surface_forms": ["VerifOnly"]},
        ],
        conn=conn,
    )
    coding_storage.update_entity_classification("goodcall", meets_frame="in", in_seed_list=True, segment="done-for-you", evidence_url=None, conn=conn)
    coding_storage.update_entity_classification("verifonly", meets_frame="in", in_seed_list=False, segment="done-for-you", evidence_url=None, conn=conn)
    coding_storage.insert_entity_verification(
        [{"entity_id": "verifonly", "is_real": 1, "entity_type": "vendor",
          "verified_homepage_url": "https://verifonly.example", "disambiguation_note": None, "confidence": 0.9}],
        conn=conn,
    )
    conn.commit()
    conn.close()
    return db


def test_freeze_window_writes_manifest_and_frozen_copy(tmp_path, runs_dir, monkeypatch):
    frozen = tmp_path / "frozen"
    manifest = tmp_path / "raw" / "MANIFEST.json"
    monkeypatch.setattr(freeze_window, "FROZEN_DIR", frozen)
    monkeypatch.setattr(freeze_window, "MANIFEST_PATH", manifest)
    freeze_window.main(["--runs-dir", str(runs_dir), "--no-tag"])
    m = json.loads(manifest.read_text())
    assert m["totals"]["lines"] == 1
    assert m["raw_files"][0]["unique_run_ids"] == 1
    assert (frozen / "2026-08-20.jsonl").exists()


def test_audit_report_has_all_sections(tmp_path, runs_dir):
    report = tmp_path / "audit.md"
    audit_collection.main(["--runs-dir", str(runs_dir), "--prompts", "does-not-exist.csv", "--experiment-config", "nope.yaml", "--report", str(report)])
    text = report.read_text()
    for heading in ["Grid coverage", "Timestamp distribution", "model_returned", "search_fired", "Cited URL", "Permanent gaps"]:
        assert heading in text


def test_build_roster_from_coding_db(tmp_path, coding_db):
    out = tmp_path / "roster.csv"
    build_roster.main(["--coding-db", str(coding_db), "--seed-list", "missing.csv", "--out", str(out)])
    text = out.read_text()
    assert text.startswith("# PROVISIONAL")
    assert "goodcall,Goodcall,goodcall.com" in text
    # homepage-less entity picks up the A4' verified domain
    assert "verifonly,VerifOnly,verifonly.example" in text


def test_ledger_is_idempotent(tmp_path):
    aux_root = tmp_path / "aux"
    report = tmp_path / "ledger.md"
    update_ambiguity_ledger.main(["--aux-root", str(aux_root), "--report", str(report)])
    update_ambiguity_ledger.main(["--aux-root", str(aux_root), "--report", str(report)])
    rows = (aux_root / "ambiguity_ledger.jsonl").read_text().strip().splitlines()
    ids = [json.loads(r)["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert "schedule-deviation-day4-2026-08-23" in ids


def test_ledger_render_from_jsonl_missing_entry_leaves_markdown_unchanged(tmp_path, capsys):
    aux_root = tmp_path / "aux"
    report = tmp_path / "ledger.md"
    update_ambiguity_ledger.main(["--aux-root", str(aux_root), "--report", str(report)])
    full = report.read_text()
    # a hand-appended markdown entry the jsonl does not know about
    hand = "## 2026-12-01 — hand-entry\n\n**Situation.** s\n\n**Decision.** d\n\n**Reason.** r\n"
    report.write_text(full + "\n" + hand)
    before = report.read_text()
    update_ambiguity_ledger.main(["--aux-root", str(aux_root), "--report", str(report)])
    assert report.read_text() == before
    out = capsys.readouterr().out
    assert "WARNING" in out and "2026-12-01 \u2014 hand-entry" in out


def test_ledger_roundtrips_list_style_entries():
    e = {"id": "x", "date": "2026-01-01", "situation": "s", "decision": "\n- a\n- b", "reason": "r"}
    md = update_ambiguity_ledger.render_md([e])
    assert "**Decision.**\n- a\n- b\n\n**Reason.** r" in md


def test_summary_runs_on_empty_aux(tmp_path):
    aux_root = tmp_path / "aux"
    aux_root.mkdir()
    report = tmp_path / "summary.md"
    manifest = tmp_path / "MANIFEST.json"
    aux_summary.main(["--aux-root", str(aux_root), "--report", str(report), "--manifest", str(manifest)])
    assert "Auxiliary collection summary" in report.read_text()
    assert "aux" in json.loads(manifest.read_text())


@pytest.mark.parametrize(
    "mod",
    [collect_google_serp, collect_robots, collect_cited_urls, collect_site_inventory, collect_directories],
)
def test_collectors_dry_run_make_no_calls(mod, tmp_path, runs_dir, coding_db, capsys):
    roster_csv = tmp_path / "roster.csv"
    build_roster.main(["--coding-db", str(coding_db), "--seed-list", "missing.csv", "--out", str(roster_csv)])
    argv = ["--dry-run", "--aux-root", str(tmp_path / "aux")]
    if mod is collect_google_serp:
        argv += ["--prompts", str(runs_dir.parent / "p.csv")]
        (runs_dir.parent / "p.csv").write_text("intent_id,paraphrase_id,prompt_text,branded\n1,1,best ai receptionist,0\n")
    elif mod is collect_cited_urls:
        argv += ["--runs-dir", str(runs_dir)]
    else:
        argv += ["--roster", str(roster_csv)]
    mod.main(argv)
    assert "dry-run" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("mod", [collect_robots, collect_site_inventory, collect_directories])
def test_collectors_refuse_without_confirm_on_non_tty(mod, tmp_path, coding_db, capsys):
    roster_csv = tmp_path / "roster.csv"
    build_roster.main(["--coding-db", str(coding_db), "--seed-list", "missing.csv", "--out", str(roster_csv)])
    mod.main(["--roster", str(roster_csv), "--aux-root", str(tmp_path / "aux")])
    assert "blocked" in capsys.readouterr().out.lower()
