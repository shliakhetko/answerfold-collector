"""Calibration and checks (Post-Collection Coding Pipeline Spec). No model
calls of its own -- sampling and arithmetic over what earlier stages already
produced. Every function here is a manual-review AID, not something that
gates or blocks a pipeline run by itself.
"""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def sample_calibration_runs(run_rows: list[dict[str, Any]], *, n: int = 40, seed: int | None = None) -> list[str]:
    """spec: "Hand-label 40 randomly sampled runs for both surface_form and
    is_recommended. Tune prompts against those only." Returns sorted run_ids
    so the sample is reproducible and diffable given the same seed."""
    rng = random.Random(seed)
    run_ids = [r["run_id"] for r in run_rows]
    return sorted(rng.sample(run_ids, min(n, len(run_ids))))


@dataclass(frozen=True)
class SpotCheckSample:
    two_of_three: list[dict[str, Any]] = field(default_factory=list)
    one_of_three_dropped: list[dict[str, Any]] = field(default_factory=list)
    random_slice: list[dict[str, Any]] = field(default_factory=list)


def sample_spot_check(
    mentions_raw_rows: list[dict[str, Any]],
    single_vote_groups: list[dict[str, Any]],
    *,
    n_2of3: int,
    n_1of3_dropped: int,
    n_random: int,
    seed: int | None = None,
) -> SpotCheckSample:
    """spec: "Sample 2-of-3 consensus cases, 1-of-3 dropped cases, and a
    random slice. Errors concentrate in disagreements." -- "2-of-3" here
    means exactly-2-vote consensus rows (a real disagreement the majority
    rule resolved), not 3-of-3 full agreement, since disagreement is where
    the spec says errors concentrate. single_vote_groups comes from
    analysis.coding.a1_answer_coding.A1Stats.single_vote_groups -- those
    1-of-3 candidates never appear in mentions_raw.jsonl itself."""
    rng = random.Random(seed)
    two_of_three_pool = [m for m in mentions_raw_rows if len(m.get("votes", [])) == 2]
    return SpotCheckSample(
        two_of_three=rng.sample(two_of_three_pool, min(n_2of3, len(two_of_three_pool))),
        one_of_three_dropped=rng.sample(single_vote_groups, min(n_1of3_dropped, len(single_vote_groups))),
        random_slice=rng.sample(mentions_raw_rows, min(n_random, len(mentions_raw_rows))),
    )


def sample_low_confidence_urls(url_registry_rows: list[dict[str, Any]], *, fraction: float = 0.10) -> list[dict[str, Any]]:
    """spec: "sample low classifier_confidence URLs per the frozen 10%
    rule" -- the lowest-confidence `fraction` of classified URLs, sorted
    ascending. Rows with no confidence value (A5 not run, or that URL's
    classification errored) are excluded, not treated as confidence 0."""
    scored = [r for r in url_registry_rows if r.get("classifier_confidence") not in (None, "")]
    scored.sort(key=lambda r: float(r["classifier_confidence"]))
    if not scored:
        return []
    n = max(1, round(len(scored) * fraction))
    return scored[:n]


def _truthy(value: str | None) -> bool | None:
    v = (value or "").strip().lower()
    if v == "":
        return None
    return v in ("1", "true", "yes")


def apply_overrides(
    spot_check_rows: list[dict[str, Any]], *, threshold: float = 0.70
) -> dict[str, tuple[str, bool | None]]:
    """2026-09-12 A5 close-out: url_normalized -> (settled_class,
    settled_publisher_is_in_category) for every spot-check row carrying a
    human label. Narrow rule (pre-registered, not this function's to
    relax): a human label overrides the model only where the ORIGINAL
    classifier_confidence was below `threshold` AND a human_class is
    present. A calibration-band (>= threshold) disagreement is evidence for
    compute_agreement, never an override -- calibration measures trust in
    the untouched majority, it does not relabel it. Rows without a
    classifier_confidence, or below threshold but never reviewed
    (human_class blank), produce no override; the caller is expected to
    pass the model's own url_class/publisher_is_in_category through
    unchanged for those."""
    overrides: dict[str, tuple[str, bool | None]] = {}
    for row in spot_check_rows:
        conf = row.get("classifier_confidence")
        human_class = (row.get("human_class") or "").strip()
        if conf in (None, "") or not human_class:
            continue
        if float(conf) < threshold:
            overrides[row["url_normalized"]] = (human_class, _truthy(row.get("human_publisher_is_in_category")))
    return overrides


def compute_agreement(spot_check_rows: list[dict[str, Any]], *, threshold: float = 0.70) -> dict[str, Any]:
    """Per-band (below/at-or-above `threshold`) agreement rate between
    human_* and model_* labels on a reviewed spot-check sample, counting
    only rows with a non-blank human_class (an unreviewed sampled row is
    excluded, not treated as disagreement -- same convention as
    sample_low_confidence_urls excluding rows with no confidence value)."""

    def _band_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        reviewed = [r for r in rows if (r.get("human_class") or "").strip()]
        if not reviewed:
            return {"n_sampled": len(rows), "n_reviewed": 0, "class_agree_rate": None, "publisher_agree_rate": None, "both_agree_rate": None}
        class_agree = pub_agree = both_agree = 0
        for r in reviewed:
            class_match = r["human_class"].strip() == (r.get("model_url_class") or "").strip()
            pub_match = _truthy(r.get("human_publisher_is_in_category")) == _truthy(r.get("model_publisher_is_in_category"))
            class_agree += class_match
            pub_agree += pub_match
            both_agree += class_match and pub_match
        n = len(reviewed)
        return {
            "n_sampled": len(rows),
            "n_reviewed": n,
            "class_agree_rate": class_agree / n,
            "publisher_agree_rate": pub_agree / n,
            "both_agree_rate": both_agree / n,
        }

    below = [r for r in spot_check_rows if r.get("classifier_confidence") not in (None, "") and float(r["classifier_confidence"]) < threshold]
    above = [r for r in spot_check_rows if r.get("classifier_confidence") not in (None, "") and float(r["classifier_confidence"]) >= threshold]
    return {"below_threshold": _band_stats(below), "above_threshold_calibration": _band_stats(above)}


@dataclass(frozen=True)
class CostSummary:
    n_cache_entries: int
    total_tokens_input: int
    total_tokens_output: int
    mean_tokens_input: float | None
    mean_tokens_output: float | None


def measure_actual_token_cost(cache_dir: Path, *, n: int | None = 20) -> CostSummary:
    """spec: "Measure real input and output token counts on 20 actual runs
    across all 3 extractors before committing to the full pass. Do not
    extrapolate from assumed counts." Reads tokens_input/tokens_output
    straight out of cached results (analysis.coding.cache.LLMCache entries
    already carry the real measured counts from the calls that produced
    them) -- no re-call. cache_dir should be the parent llm_cache/ directory
    (or one extractor's subdirectory) so this covers every extractor's
    cached calls found under it."""
    entries: list[dict[str, Any]] = []
    for path in sorted(cache_dir.rglob("*.json")):
        with path.open(encoding="utf-8") as f:
            entries.append(json.load(f))
    if n is not None:
        entries = entries[:n]

    tokens_input = [e["tokens_input"] for e in entries if e.get("tokens_input") is not None]
    tokens_output = [e["tokens_output"] for e in entries if e.get("tokens_output") is not None]
    return CostSummary(
        n_cache_entries=len(entries),
        total_tokens_input=sum(tokens_input),
        total_tokens_output=sum(tokens_output),
        mean_tokens_input=statistics.fmean(tokens_input) if tokens_input else None,
        mean_tokens_output=statistics.fmean(tokens_output) if tokens_output else None,
    )
