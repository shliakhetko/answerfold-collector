"""Estimated USD cost per run, derived from token counts.

Analysis-layer only (INSTRUMENT.md sections 3, 11): no vendor response in
this study returns a dollar figure, so cost is not a raw fact collection
observes -- it is computed here, after the fact, from
token counts and a hand-maintained price list. It is never written back into
the collector's SQLite schema; notebooks/aggregate.ipynb computes it as a
pandas column at read time.

Rates below are illustrative, sourced from each vendor's public pricing
page, not re-checked against a live quote. Vendor prices change without
notice -- re-verify per model before a real experiment reports a cost
figure.

Keyed by a substring match against model_requested/model_returned rather
than an exact string, since the reported model (e.g. "gpt-4o-2024-08-06") is
often more specific than the pinned identifier requested (e.g. "gpt-4o") --
and for some vendors' "-latest" aliases, the resolved version number lands
in the middle of the name (e.g. "gemini-3.5-flash-lite"), not just appended
at the end, so a plain prefix check isn't enough.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    input_per_million: float
    output_per_million: float


# Longest/most specific prefix should be checked first by callers -- see
# _match_rate below, which sorts by prefix length.
PRICING: dict[str, Rate] = {
    "gpt-4o": Rate(input_per_million=2.50, output_per_million=10.00),
    "gemini-flash-latest": Rate(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash": Rate(input_per_million=0.30, output_per_million=2.50),
    "claude-sonnet": Rate(input_per_million=3.00, output_per_million=15.00),
    # analysis/coding/'s cheap/fast-tier judge & extractor models (A1/A3/A4/A5)
    # -- sourced live 2026-08-26 (Anthropic's own pricing page; third-party
    # aggregators for OpenAI/Gemini), not re-checked against a live quote.
    # Re-verify before a real experiment reports a cost figure, same as
    # every other rate in this table.
    "claude-haiku-4-5": Rate(input_per_million=1.00, output_per_million=5.00),
    "gpt-5-mini": Rate(input_per_million=0.25, output_per_million=2.00),
    # Not "gemini-flash-lite": the "-latest" alias this repo pins
    # (extractors/models.py) resolves to a versioned model_returned like
    # "gemini-3.5-flash-lite" -- the version number lands *before* "flash",
    # not after, so this can only ever match as a substring, never a
    # prefix. Confirmed against real cache data (data/_trial/coded/llm_cache).
    "flash-lite": Rate(input_per_million=0.10, output_per_million=0.40),
}

# Same three coding-pipeline models, at each vendor's Batch API rate (a flat
# 50% discount on both input and output tokens, confirmed live 2026-08-26 for
# all three vendors) -- selected via compute_cost_usd(..., batch=True).
BATCH_PRICING: dict[str, Rate] = {
    "claude-haiku-4-5": Rate(input_per_million=0.50, output_per_million=2.50),
    # A4-prime (a3b_entity_verification.py) runs on claude-sonnet with the
    # web_search tool. Batch rate is the same flat 50% on tokens; the
    # per-search tool fee ($10/1k searches, Anthropic) is NOT modelled here.
    "claude-sonnet": Rate(input_per_million=1.50, output_per_million=7.50),
    "gpt-5-mini": Rate(input_per_million=0.125, output_per_million=1.00),
    "flash-lite": Rate(input_per_million=0.05, output_per_million=0.20),
}


def _match_rate(model: str, table: dict[str, Rate]) -> Rate | None:
    """Substring match, not prefix-only: a "-latest"-style pinned identifier
    (see extractors/models.py) can resolve to a model_returned with a
    version number spliced into the middle (e.g. "gemini-3.5-flash-lite"),
    not just appended at the end. Longest matching key wins so a more
    specific entry (e.g. "claude-haiku-4-5") beats a shorter generic one."""
    candidates = [key for key in table if key in model]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return table[best]


def _missing(value: object) -> bool:
    """True for None or float('nan') -- a SQL NULL round-tripped through
    pandas (e.g. notebooks/aggregate.ipynb's pd.read_sql_query) becomes NaN,
    not None, so `value is None` alone misses it. Checked with isinstance
    first since NaN != NaN would otherwise need `math.isnan`, which raises
    on a non-float input like a model name string."""
    return value is None or (isinstance(value, float) and value != value)


def compute_cost_usd(
    model: str | None,
    tokens_input: int | None,
    tokens_output: int | None,
    *,
    batch: bool = False,
) -> float | None:
    """Returns None (never fabricates a number) when the model isn't in
    PRICING, or either token count is missing -- e.g. google_ai_mode, or any
    row where usage extraction failed. batch=True prices against
    BATCH_PRICING (each vendor's Batch API rate) instead of standard
    synchronous pricing."""
    if _missing(model) or not isinstance(model, str) or _missing(tokens_input) or _missing(tokens_output):
        return None
    rate = _match_rate(model, BATCH_PRICING if batch else PRICING)
    if rate is None:
        return None
    return (
        tokens_input / 1_000_000 * rate.input_per_million
        + tokens_output / 1_000_000 * rate.output_per_million
    )
