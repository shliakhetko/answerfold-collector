"""Assembles validation_report.md, one of Phase A's five frozen artifacts:
A1 extractor drop rates, the self-judging limitation disclosure (spec:
"either swap it out or write the limitation" -- this pipeline writes it,
since the 3 extractors necessarily share vendors with the 4 engines under
test), and the alias-regex-vs-ensemble comparison.
"""
from __future__ import annotations

from pathlib import Path

from analysis.coding.a1_answer_coding import A1Stats
from analysis.coding.alias_regex_validation import ComparisonReport

from collector import paths
DEFAULT_OUTPUT_PATH = paths.VALIDATION_REPORT_MD

SELF_JUDGING_LIMITATION = """\
A1's three extractors (Anthropic, OpenAI, Gemini -- analysis/coding/extractors/)
reuse this project's already-installed SDKs, the same three vendors behind
three of the four engines under test in config/experiment.yaml (openai_chatgpt,
google_gemini, anthropic_claude; Google doubly so via google_ai_mode). This
is unavoidable without dropping an extractor entirely, since there is no
fourth already-integrated vendor to swap in.

Per the spec, this is disclosure, not disqualification: extraction asks a
model to identify a literal surface_form and evidence_span and is
deterministically verified afterward (evidence_span/surface_form must be
literal substrings of the answer text, or the row is dropped) -- a
hallucinated favorable extraction is caught by that substring check, not by
the model grading its own vendor's output the way a quality judge would.
"""


def render_validation_report(*, a1_stats: A1Stats, comparison: ComparisonReport) -> str:
    lines: list[str] = ["# Validation Report", ""]

    lines += ["## A1 extractor drop rates", ""]
    if a1_stats.drop_stats_by_extractor:
        for name in sorted(a1_stats.drop_stats_by_extractor):
            stats = a1_stats.drop_stats_by_extractor[name]
            total = stats.verified + stats.dropped
            rate = a1_stats.drop_rate(name)
            rate_str = f"{rate:.1%}" if rate is not None else "n/a"
            lines.append(f"- {name}: dropped {stats.dropped} of {total} candidates ({rate_str})")
    else:
        lines.append("(no A1 run recorded)")
    lines.append("")

    lines += ["## Self-judging limitation", "", SELF_JUDGING_LIMITATION.strip(), ""]

    lines += ["## Alias regex validation", ""]
    recall = comparison.ensemble_recall_on_seed_list
    recall_str = f"{recall:.1%}" if recall is not None else "n/a (no regex hits on the seed list)"
    lines.append(f"- Ensemble recall on the seeded prospect list: {recall_str}")
    lines.append("")
    lines += ["### Cases regex found and the ensemble missed", ""]
    if comparison.regex_found_ensemble_missed:
        for run_id, name in comparison.regex_found_ensemble_missed:
            lines.append(f"- {run_id}: {name}")
    else:
        lines.append("(none)")
    lines.append("")
    lines += ["### Cases the ensemble found and regex missed", ""]
    if comparison.ensemble_found_regex_missed:
        for run_id, name in comparison.ensemble_found_regex_missed:
            lines.append(f"- {run_id}: {name}")
    else:
        lines.append("(none)")
    lines.append("")

    return "\n".join(lines)


def write_validation_report(
    *, a1_stats: A1Stats, comparison: ComparisonReport, output_path: Path = DEFAULT_OUTPUT_PATH
) -> str:
    text = render_validation_report(a1_stats=a1_stats, comparison=comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text
