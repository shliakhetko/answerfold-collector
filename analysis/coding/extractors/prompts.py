"""A1 extraction prompt, shared verbatim across all three extractors --
spec: "Extraction and recommendation go in one call because both need the
full answer text."

PROMPT_VERSION is part of the cache key (analysis.coding.cache.cache_key)
and the spec's freeze/disclosure requirement ("Prompts frozen and
versioned before the real run. Calibrate on 40 hand-labelled runs only.").
"v0-unfrozen": this prompt has NOT gone through that calibration pass yet --
do not treat it as final, and bump this version (never silently edit the
prompt under the same version) once it has.
"""
from __future__ import annotations

PROMPT_VERSION = "v0-unfrozen"

_EXTRACTION_PROMPT_TEMPLATE = """You are extracting company or product mentions from one AI search engine's answer to a buyer's question.

For every company or product name mentioned in the TEXT below, return one JSON object with exactly these keys:
- "surface_form": the name exactly as written in TEXT (a literal, character-for-character substring of TEXT).
- "evidence_span": the exact sentence or clause of TEXT containing that mention. Must be a literal, character-for-character substring of TEXT -- do not paraphrase, summarize, or fix typos.
- "is_recommended": true only if the company is presented as an option for the buyer's own question. false if it is named only as someone else's competitor, named only in a caveat or comparison, or named only in an unrelated aside. A company that is not mentioned cannot be recommended.

Return a JSON array of these objects and nothing else -- no markdown code fence, no commentary. Return an empty array [] if no company or product is named.

TEXT:
{raw_response_text}"""


def build_extraction_prompt(raw_response_text: str) -> str:
    return _EXTRACTION_PROMPT_TEMPLATE.format(raw_response_text=raw_response_text)


ARBITER_PROMPT_VERSION = "v0-unfrozen"

_ARBITER_PROMPT_TEMPLATE = """You are checking a short list of candidate company/product mentions against the AI search engine answer they were pulled from. Each candidate was named by only one of two other reviewers, so you're the tie-breaker.

For each candidate below, decide:
- "confirmed": true only if the candidate is a genuine company or product name actually presented as an option (or a competitor/caveat/aside, per the recommendation rule below) in TEXT. false if it's not really a distinct company/product, or the text doesn't support it.
- "is_recommended": true only if the company is presented as an option for the buyer's own question. false if it is named only as someone else's competitor, named only in a caveat or comparison, or named only in an unrelated aside. Only meaningful when "confirmed" is true; set to false otherwise.

Return a JSON array with exactly one object per candidate, in the same order, each with exactly these keys: "surface_form", "confirmed", "is_recommended". No markdown code fence, no commentary.

TEXT:
{raw_response_text}

CANDIDATES:
{candidates_json}"""


def build_arbiter_prompt(raw_response_text: str, candidates_json: str) -> str:
    return _ARBITER_PROMPT_TEMPLATE.format(raw_response_text=raw_response_text, candidates_json=candidates_json)
