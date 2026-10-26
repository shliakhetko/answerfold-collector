"""B1: per run x company (Post-Collection Coding Pipeline Spec). No model --
pure joins over Phase A's frozen artifacts plus raw JSONL. Restricted to
entities where meets_frame == "in" (spec: "Do not let entities failing the
frame enter any denominator. They are a count."). "borderline" and
"unresolved" are excluded from the denominator too -- an ambiguous frame
call is not the same as a confirmed member.

Produces one row per (run_id, in-frame entity_id) pair -- a full cross
product, not just the mentioned pairs -- because B3's mention_rate needs a
real denominator (every run an entity could have been mentioned in, not
only the runs where it was).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from analysis.coding.text_clean import strip_markdown_links


def in_frame_entities(entity_registry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in entity_registry_rows if row.get("meets_frame") == "in"]


def surface_form_to_entity_id(entities: list[dict[str, Any]]) -> dict[str, str]:
    """Exact surface_form (as stored in mentions_raw.jsonl, never
    normalized) -> entity_id, built only from in-frame entities -- a mention
    of an out-of-frame entity's surface form simply has no entry here and is
    dropped by the caller, matching the spec's "not a denominator" rule."""
    mapping: dict[str, str] = {}
    for entity in entities:
        for surface_form in (entity.get("surface_forms") or "").split(";"):
            if surface_form:
                mapping[surface_form] = entity["entity_id"]
    return mapping


def _domain_of(url_or_domain: str) -> str:
    """Lowercased, www-stripped host -- accepts either a bare domain
    (alt_domains.csv) or a full URL (homepage_url)."""
    host = urlparse(url_or_domain).netloc or url_or_domain
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def entity_domains(entities: list[dict[str, Any]], alt_domains_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """entity_id -> set of normalized domains (own homepage + alt_domains.csv
    rows for that entity) -- the own_domain_cited join key."""
    domains: dict[str, set[str]] = {}
    for entity in entities:
        entity_id = entity["entity_id"]
        homepage = entity.get("homepage_url")
        domains[entity_id] = {_domain_of(homepage)} if homepage else set()
    for row in alt_domains_rows:
        entity_id = row.get("entity_id")
        alt_domain = row.get("alt_domain")
        if entity_id in domains and alt_domain:
            domains[entity_id].add(_domain_of(alt_domain))
    return domains


@dataclass(frozen=True)
class PerRunCompanyRow:
    run_id: str
    entity_id: str
    mentioned: bool
    recommended: bool
    own_domain_cited: bool
    linked: bool
    mention_without_own_citation: bool
    first_mention_position: int | None



def _span_position(cleaned_text: str, mention: dict[str, Any]) -> int:
    """Where a mention's quoted sentence starts in the link-stripped answer
    text, -1 if absent. In the publish package the quote of a Gemini or AI
    Mode mention is withheld (Google's terms) and the position, computed the
    same way on the original text, ships as evidence_span_position."""
    span = mention.get("evidence_span")
    if span is None and "evidence_span_position" in mention:
        return int(mention["evidence_span_position"])
    return cleaned_text.find(span)

def build_per_run_company(
    run_rows: list[dict[str, Any]],
    mentions_raw_rows: list[dict[str, Any]],
    entity_registry_rows: list[dict[str, Any]],
    sources_by_run: dict[str, list[dict[str, Any]]],
    alt_domains_rows: list[dict[str, Any]],
) -> list[PerRunCompanyRow]:
    entities = in_frame_entities(entity_registry_rows)
    sf_to_entity = surface_form_to_entity_id(entities)
    domains_by_entity = entity_domains(entities, alt_domains_rows)

    # (run_id, entity_id) -> list of mentions_raw rows attributable to that
    # entity (an entity can have >1 surface form mentioned in the same run).
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for mention in mentions_raw_rows:
        entity_id = sf_to_entity.get(mention["surface_form"])
        if entity_id is None:
            continue  # out-of-frame or unresolved surface form -- never a denominator
        grouped.setdefault((mention["run_id"], entity_id), []).append(mention)

    rows: list[PerRunCompanyRow] = []
    for run in run_rows:
        run_id = run["run_id"]
        # A1 (a1_answer_coding.py) extracts evidence_span from markdown-
        # link-stripped text, so first_mention_position below must search
        # the same cleaned text -- searching the untouched raw text would
        # silently miss any span that used to straddle a stripped link.
        raw_response_text = strip_markdown_links(run.get("raw_response_text") or "")
        cited_domains = {
            _domain_of(source["domain"] or source.get("url_raw", ""))
            for source in sources_by_run.get(run_id, [])
            if source.get("domain") or source.get("url_raw")
        }

        for entity in entities:
            entity_id = entity["entity_id"]
            entity_mentions = grouped.get((run_id, entity_id), [])
            mentioned = bool(entity_mentions)
            recommended = mentioned and any(m["is_recommended"] for m in entity_mentions)

            # "linked": every row in the sources table was extracted from an
            # actual citation/annotation (collector/engines/*.py's
            # _extract_sources), never a plain-text domain mention with no
            # href -- so for this data model, a cited own domain is always a
            # link. Kept as a separate field (not silently collapsed into
            # own_domain_cited) in case a future source type breaks that
            # equivalence.
            own_domain_cited = bool(domains_by_entity.get(entity_id) and domains_by_entity[entity_id] & cited_domains)
            linked = own_domain_cited

            first_mention_position = None
            if mentioned:
                positions = [
                    _span_position(raw_response_text, m)
                    for m in entity_mentions
                    if _span_position(raw_response_text, m) != -1
                ]
                first_mention_position = min(positions) if positions else None

            rows.append(
                PerRunCompanyRow(
                    run_id=run_id,
                    entity_id=entity_id,
                    mentioned=mentioned,
                    recommended=recommended,
                    own_domain_cited=own_domain_cited,
                    linked=linked,
                    mention_without_own_citation=mentioned and not own_domain_cited,
                    first_mention_position=first_mention_position,
                )
            )
    return rows
