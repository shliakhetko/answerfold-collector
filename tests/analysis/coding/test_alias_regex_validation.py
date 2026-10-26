from analysis.coding.alias_regex_validation import compare_to_ensemble, load_alias_list, run_alias_regex

ENTITY_REGISTRY_ROWS = [
    {"entity_id": "goodcall", "canonical_name": "Goodcall", "surface_forms": "Goodcall;GoodCall"},
    {"entity_id": "vapi", "canonical_name": "Vapi", "surface_forms": "Vapi"},
]


def _write_alias_list(path, rows):
    lines = ["canonical_name,alias_regex"]
    for canonical_name, pattern in rows:
        lines.append(f"{canonical_name},{pattern}")
    path.write_text("\n".join(lines) + "\n")


def test_load_alias_list_compiles_patterns(tmp_path):
    alias_path = tmp_path / "alias_list.csv"
    _write_alias_list(alias_path, [("Goodcall", r"(?i)\bgoodcall\b")])
    aliases = load_alias_list(alias_path)
    assert len(aliases) == 1
    assert aliases[0][0] == "Goodcall"
    assert aliases[0][1].search("I recommend GOODCALL for this.")


def test_run_alias_regex_finds_matches_case_insensitively(tmp_path):
    alias_path = tmp_path / "alias_list.csv"
    _write_alias_list(alias_path, [("Goodcall", r"(?i)\bgoodcall\b"), ("Vapi", r"(?i)\bvapi\b")])
    run_rows = [
        {"run_id": "run-1", "raw_response_text": "Goodcall is a strong choice, unlike Vapi."},
        {"run_id": "run-2", "raw_response_text": "No relevant companies here."},
    ]
    hits = run_alias_regex(run_rows, alias_path)
    assert {(h["run_id"], h["canonical_name"]) for h in hits} == {("run-1", "Goodcall"), ("run-1", "Vapi")}


def test_run_alias_regex_the_deliberate_rosie_gap(tmp_path):
    # config/alias_list.csv's real known gap: a bare "Rosie" mention matches
    # neither askrosie.ai nor heyrosie.com's regex, by design -- this test
    # documents that the mechanism behaves as intended, not a bug.
    alias_path = tmp_path / "alias_list.csv"
    _write_alias_list(alias_path, [("AskRosie", r"(?i)\baskrosie\.ai\b"), ("HeyRosie", r"(?i)\bheyrosie\.com\b")])
    run_rows = [{"run_id": "run-1", "raw_response_text": "Rosie is a great AI phone assistant."}]
    assert run_alias_regex(run_rows, alias_path) == []


def test_compare_to_ensemble_recall_and_diffs():
    alias_hits = [
        {"run_id": "run-1", "canonical_name": "Goodcall"},
        {"run_id": "run-2", "canonical_name": "Goodcall"},  # regex found, ensemble will miss
    ]
    mentions_raw_rows = [
        {"run_id": "run-1", "surface_form": "Goodcall"},  # ensemble agrees on run-1
        {"run_id": "run-3", "surface_form": "Vapi"},  # ensemble found, regex missed
    ]
    seed_list = {"goodcall"}

    report = compare_to_ensemble(alias_hits, mentions_raw_rows, ENTITY_REGISTRY_ROWS, seed_list)

    assert report.ensemble_recall_on_seed_list == 0.5  # 1 of 2 seed-list regex hits confirmed by ensemble
    assert report.regex_found_ensemble_missed == [("run-2", "Goodcall")]
    assert report.ensemble_found_regex_missed == [("run-3", "Vapi")]


def test_compare_to_ensemble_no_seed_hits_gives_none_recall():
    report = compare_to_ensemble([], [], ENTITY_REGISTRY_ROWS, seed_list=set())
    assert report.ensemble_recall_on_seed_list is None
