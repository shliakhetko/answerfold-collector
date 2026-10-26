"""Pinned model IDs for A1's three extractors -- cheap/fast tier, distinct
from the pinned models under test in config/experiment.yaml (this study's engines,
not the extractors judging them).

Live config (2026-09-02): anthropic + gemini are the two PRIMARY_EXTRACTORS
(one Batch job each); openai is the synchronous ARBITER_EXTRACTOR. All three
IDs exercised live on this account -- anthropic + gemini via successful A1
Batch jobs, openai (gpt-5-mini -> gpt-5-mini-2025-08-07) via a synchronous
Responses smoke test. gpt-5-mini rejects the `temperature` param (reasoning
tier); the extractor/arbiter calls omit it, same as this study's own
openai_chatgpt engine adapter.
"""
from __future__ import annotations

EXTRACTOR_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5-mini",
    "gemini": "gemini-flash-lite-latest",
}
