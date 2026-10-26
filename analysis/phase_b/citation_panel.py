"""Phase B shared intermediate: the run x citation panel with url_class
attached.

One row per citation instance -- run_id, engine, intent_id, paraphrase_id,
position, url_normalized, domain, url_class, publisher_is_in_category --
restricted to usable runs (analysis.phase_b.exclusions.usable_run_ids,
reused verbatim) and the six unbranded intents (1-6; intent 7 is branded
and out of scope for every figure that reads this panel).

url_class/publisher_is_in_category come from A5's classification, frozen
2026-09-12 at data/03_coded/url_registry_frozen.csv and loaded onto
FrozenInputs.url_registry_rows, keyed by url_normalized -- the same key
A5 itself deduplicated sources on. This is the one url_class join: S1, K2
and X1 all read this panel and none of them recomputes a class.

Registers no study figure and computes no share -- same discipline as
domain_ownership.py's build_source_ownership: this module only builds the
join and reports on how it was built.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from analysis.coding.b1_per_run_company import in_frame_entities, surface_form_to_entity_id
from analysis.coding.registrable_domain import registrable_domain
from analysis.coding.url_normalize import normalize_url
from analysis.phase_b.domain_ownership import entity_owned_domains
from analysis.phase_b.exclusions import usable_run_ids
from analysis.phase_b.reading import FrozenInputs

UNBRANDED_INTENT_IDS = frozenset({1, 2, 3, 4, 5, 6})


@dataclass(frozen=True)
class CitationRow:
    run_id: str
    engine: str
    intent_id: int | None
    paraphrase_id: int | None
    position: int | None
    url_normalized: str
    domain: str | None
    url_class: str | None  # None = no url_registry match (see build report)
    publisher_is_in_category: bool | None
    # True when A5's raw classification said False (or the settled
    # override said False) but this row's domain matches a known in-frame
    # entity's own domain, so publisher_is_in_category above was forced to
    # True (2026-09-23, round 2 review) -- per-citation auditability
    # for the domain-ownership override, not just the aggregate count in
    # CitationPanelBuildReport.
    publisher_is_in_category_overridden: bool
    resolution_status: str | None
    resolution_method: str | None  # "title_fallback" or None

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine": self.engine,
            "intent_id": self.intent_id,
            "paraphrase_id": self.paraphrase_id,
            "position": self.position,
            "url_normalized": self.url_normalized,
            "domain": self.domain,
            "url_class": self.url_class,
            "publisher_is_in_category": self.publisher_is_in_category,
            "publisher_is_in_category_overridden": self.publisher_is_in_category_overridden,
            "resolution_status": self.resolution_status,
            "resolution_method": self.resolution_method,
        }


@dataclass(frozen=True)
class CitationPanelBuildReport:
    """No headline number lives here -- a build report, not a figure
    record, same convention as domain_ownership.BuildReport."""

    usable_runs_total: int
    runs_in_scope: int  # usable AND intent_id in UNBRANDED_INTENT_IDS
    rows_produced: int  # == citations_total, restricted to runs_in_scope
    url_class_counts: dict[str, int]  # includes an "unclassified" bucket
    citations_unclassified: int
    # Split of the unclassified count: title-fallback synthetic URLs are
    # *guaranteed* unmatched (url_registry_frozen.csv was frozen
    # 2026-09-12, 11 days before the Gemini title fallback existed) -- a
    # different, expected reason from "this real citation was simply
    # never classified," so the two are never miscounted together.
    citations_unclassified_title_fallback: int
    citations_unclassified_other: int
    engines_in_scope: dict[str, int]  # citation count per engine
    # domain -> {"title_fallback": n, "other": n} for every unclassified
    # citation's registrable domain (2026-09-23, round 2 review) -- a
    # permanent, inspectable version of "what exactly is in the 200-ish
    # title_fallback citations and the handful of genuinely-never-matched
    # ones," instead of a one-off query re-run every time someone asks.
    # Closing the title_fallback slice fully means re-running A5
    # classification on these domains (a real LLM call) -- not done here,
    # flagged as a follow-up requiring the owner's go-ahead.
    unclassified_domains_by_reason: dict[str, dict[str, int]]
    # How many classified citations had publisher_is_in_category forced
    # True by the domain-ownership override (2026-09-23, round 2 review)
    # -- A5's raw per-URL judgment is inconsistent on a vendor's
    # own content-marketing pages even though the domain is unambiguous;
    # see all_in_frame_owned_domains's docstring for the evidence.
    citations_publisher_override_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "usable_runs_total": self.usable_runs_total,
            "runs_in_scope": self.runs_in_scope,
            "rows_produced": self.rows_produced,
            "url_class_counts": self.url_class_counts,
            "citations_unclassified": self.citations_unclassified,
            "citations_unclassified_title_fallback": self.citations_unclassified_title_fallback,
            "citations_unclassified_other": self.citations_unclassified_other,
            "engines_in_scope": self.engines_in_scope,
            "unclassified_domains_by_reason": self.unclassified_domains_by_reason,
            "citations_publisher_override_count": self.citations_publisher_override_count,
        }


def url_class_publisher_crosstab(
    rows: list["CitationRow"],
) -> dict[str, dict[str, dict[str, int]]]:
    """engine -> url_class (raw label, "unclassified" for None) ->
    publisher_is_in_category ("true"/"false"/"null") -> citation count.

    The permanent, auditable answer to "every distinct url_class label
    value with counts per engine, and how publisher_is_in_category splits
    within it" -- both S1 and X1 read this so the crosstab is computed
    once, identically, not reimplemented per figure (2026-09-23, owner
    review Fix 1). String keys throughout (not bool/None) so this survives
    a JSON round-trip through the seam file unchanged."""
    tab: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        cls_key = row.url_class if row.url_class is not None else "unclassified"
        pub_key = "true" if row.publisher_is_in_category is True else (
            "false" if row.publisher_is_in_category is False else "null"
        )
        engine_tab = tab.setdefault(row.engine, {})
        cls_tab = engine_tab.setdefault(cls_key, {"true": 0, "false": 0, "null": 0})
        cls_tab[pub_key] += 1
    return tab


def all_in_frame_owned_domains(inputs: FrozenInputs) -> set[str]:
    """The union of every in-frame entity's owned domains
    (analysis.phase_b.domain_ownership.entity_owned_domains, the existing
    matching rule, reused per entity rather than a new join).

    Two consumers (2026-09-23, round 2 review):
    - K2's Rule 2 (excluded_population): a citation's domain counts as
      own-domain if it's in this set, regardless of which entity that
      particular run was actually about.
    - build_citation_panel's publisher_is_in_category override (Fix D):
      A5's raw per-URL judgment turns out to be inconsistent on a known
      vendor's own content-marketing pages -- e.g. goodcall.com/pricing
      reads publisher_is_in_category=True but goodcall.com/answering-
      services/ai-answering-service-dental-practices (also Goodcall's own
      domain) reads False, split cleanly by page style, not randomly (see
      the 2026-09-23 s1-x1-publisher-override ledger entry). For a domain
      we already have ground truth on via this set, that ground truth
      overrides the classifier's per-URL guess."""
    entities = in_frame_entities(inputs.entities)
    verified_homepage_by_entity = {
        r["entity_id"]: r["verified_homepage_url"]
        for r in inputs.entity_verification_rows
        if r.get("verified_homepage_url")
    }
    malformed: list[str] = []
    owned: set[str] = set()
    for entity in entities:
        owned |= set(entity_owned_domains(entity, inputs.alt_domains_rows, malformed, verified_homepage_by_entity))
    return owned


def mentioned_own_domains_by_run(inputs: FrozenInputs) -> dict[str, set[str]]:
    """run_id -> union of owned domains (entity_owned_domains, reused not
    reimplemented) for every in-frame entity mentioned in that run.

    Two consumers (2026-09-23, round 2 review): S1's
    vendor_share_excluding_mentioned_own_domain_by_engine (Fix 6) and K2's
    Rule 1 / excluded_self_citation (Fix B) -- a citation to the domain of
    the company actually being discussed in that specific run is self-
    referential, a different question from "does this domain belong to
    any population company at all" (K2's Rule 2 / all_in_frame_owned_
    domains above)."""
    entities = in_frame_entities(inputs.entities)
    sf_to_entity = surface_form_to_entity_id(entities)
    entities_by_id = {e["entity_id"]: e for e in entities}
    verified_homepage_by_entity = {
        r["entity_id"]: r["verified_homepage_url"]
        for r in inputs.entity_verification_rows
        if r.get("verified_homepage_url")
    }
    malformed: list[str] = []
    owned_by_entity: dict[str, set[str]] = {}

    entity_ids_by_run: dict[str, set[str]] = defaultdict(set)
    for m in inputs.mentions_raw:
        entity_id = sf_to_entity.get(m.get("surface_form"))
        if entity_id is not None:
            entity_ids_by_run[m["run_id"]].add(entity_id)

    result: dict[str, set[str]] = {}
    for run_id, entity_ids in entity_ids_by_run.items():
        domains: set[str] = set()
        for entity_id in entity_ids:
            if entity_id not in owned_by_entity:
                entity = entities_by_id.get(entity_id)
                owned_by_entity[entity_id] = (
                    set(entity_owned_domains(entity, inputs.alt_domains_rows, malformed, verified_homepage_by_entity))
                    if entity is not None
                    else set()
                )
            domains |= owned_by_entity[entity_id]
        result[run_id] = domains
    return result


def build_citation_panel(inputs: FrozenInputs) -> tuple[list[CitationRow], CitationPanelBuildReport]:
    usable_ids = usable_run_ids(inputs.raw_runs)
    runs_by_id = {r["run_id"]: r for r in inputs.raw_runs}
    owned_domains = all_in_frame_owned_domains(inputs)

    registry_by_normalized: dict[str, dict[str, Any]] = {
        row["url_normalized"]: row for row in inputs.url_registry_rows if row.get("url_normalized")
    }

    rows: list[CitationRow] = []
    url_class_counts: dict[str, int] = {}
    citations_unclassified_title_fallback = 0
    citations_unclassified_other = 0
    engines_in_scope: dict[str, int] = {}
    runs_in_scope = 0
    unclassified_domains_by_reason: dict[str, dict[str, int]] = {}
    citations_publisher_override_count = 0

    for run_id in sorted(usable_ids):
        run = runs_by_id.get(run_id, {})
        intent_id = run.get("intent_id")
        if intent_id not in UNBRANDED_INTENT_IDS:
            continue
        runs_in_scope += 1
        engine = run.get("engine")

        for source in inputs.sources_by_run.get(run_id, []):
            url_raw = source.get("url_raw") or ""
            url_normalized = normalize_url(url_raw)
            registry_row = registry_by_normalized.get(url_normalized)

            overridden = False
            if registry_row is not None:
                url_class = registry_row.get("settled_class") or registry_row.get("url_class")
                settled_pub = registry_row.get("settled_publisher_is_in_category")
                publisher_is_in_category = (
                    settled_pub if settled_pub is not None else registry_row.get("publisher_is_in_category")
                )
                # Domain-ownership override (2026-09-23, round 2 review):
                # A5's raw per-URL judgment is inconsistent on a known
                # vendor's own content-marketing pages (see
                # all_in_frame_owned_domains's docstring) -- for a domain
                # we already have ground truth on, that ground truth wins.
                # url_class (content-type label) is left untouched; only
                # the publisher-identity field is corrected.
                if publisher_is_in_category is not True:
                    citation_domain = registrable_domain(url_normalized)
                    if citation_domain in owned_domains:
                        publisher_is_in_category = True
                        overridden = True
                        citations_publisher_override_count += 1
            else:
                url_class = None
                publisher_is_in_category = None
                reason = "title_fallback" if source.get("resolution_method") == "title_fallback" else "other"
                if reason == "title_fallback":
                    citations_unclassified_title_fallback += 1
                else:
                    citations_unclassified_other += 1
                unclassified_domain = registrable_domain(url_normalized) or url_normalized
                domain_counts = unclassified_domains_by_reason.setdefault(
                    unclassified_domain, {"title_fallback": 0, "other": 0}
                )
                domain_counts[reason] += 1

            class_key = url_class if url_class is not None else "unclassified"
            url_class_counts[class_key] = url_class_counts.get(class_key, 0) + 1
            engines_in_scope[engine] = engines_in_scope.get(engine, 0) + 1

            rows.append(
                CitationRow(
                    run_id=run_id,
                    engine=engine,
                    intent_id=intent_id,
                    paraphrase_id=run.get("paraphrase_id"),
                    position=source.get("position"),
                    url_normalized=url_normalized,
                    domain=source.get("domain"),
                    url_class=url_class,
                    publisher_is_in_category=publisher_is_in_category,
                    publisher_is_in_category_overridden=overridden,
                    resolution_status=source.get("resolution_status"),
                    resolution_method=source.get("resolution_method"),
                )
            )

    report = CitationPanelBuildReport(
        usable_runs_total=len(usable_ids),
        runs_in_scope=runs_in_scope,
        rows_produced=len(rows),
        url_class_counts=url_class_counts,
        citations_unclassified=citations_unclassified_title_fallback + citations_unclassified_other,
        citations_unclassified_title_fallback=citations_unclassified_title_fallback,
        citations_unclassified_other=citations_unclassified_other,
        engines_in_scope=engines_in_scope,
        unclassified_domains_by_reason=unclassified_domains_by_reason,
        citations_publisher_override_count=citations_publisher_override_count,
    )
    return rows, report
