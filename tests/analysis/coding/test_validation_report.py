from analysis.coding.a1_answer_coding import A1Stats, ExtractorDropStats
from analysis.coding.alias_regex_validation import ComparisonReport
from analysis.coding.validation_report import render_validation_report, write_validation_report


def test_render_includes_drop_rates_self_judging_and_comparison():
    a1_stats = A1Stats(
        runs_processed=2, mentions_written=3,
        drop_stats_by_extractor={"anthropic": ExtractorDropStats(verified=8, dropped=2)},
    )
    comparison = ComparisonReport(
        ensemble_recall_on_seed_list=0.5,
        regex_found_ensemble_missed=[("run-2", "Goodcall")],
        ensemble_found_regex_missed=[("run-3", "Vapi")],
    )

    text = render_validation_report(a1_stats=a1_stats, comparison=comparison)

    assert "# Validation Report" in text
    assert "anthropic: dropped 2 of 10 candidates (20.0%)" in text
    assert "Self-judging limitation" in text
    assert "run-2: Goodcall" in text
    assert "run-3: Vapi" in text
    assert "50.0%" in text


def test_render_handles_empty_stats_and_no_recall():
    text = render_validation_report(a1_stats=A1Stats(), comparison=ComparisonReport(ensemble_recall_on_seed_list=None))
    assert "(no A1 run recorded)" in text
    assert "n/a" in text
    assert "(none)" in text


def test_write_validation_report_writes_file(tmp_path):
    output_path = tmp_path / "validation_report.md"
    text = write_validation_report(
        a1_stats=A1Stats(), comparison=ComparisonReport(ensemble_recall_on_seed_list=None), output_path=output_path
    )
    assert output_path.read_text() == text
