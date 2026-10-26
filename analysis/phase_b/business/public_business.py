"""Business figures for the public report and its appendix (final edit pass, 2026-10-12).

Exploratory and derived, like the other business_figures_*.json files; nothing here feeds or changes
figures.json. Everything is local computation on stored data (the frozen citation panel, the registry,
mentions_raw, the raw runs and the stored figures); no model call.

Blocks (all per engine, with numerators and denominators):

* source_classes: S1's classified citations split into vendor-published, review sites, news and press,
  forums, video and social and other, plus exact shares for the registrable domains reddit.com and
  youtube.com (these are NOT separate classes). Same classified-citation path and denominator as S1
  (business_figures.json engines.*.source_mix); build_public_figures asserts the counts equal it.
* urls: vendor share on unique cited URLs next to the citation-count share, and the top 5 single URLs.
  url_normalized drops the query string, so every YouTube video collapses into youtube.com/watch; that entry
  carries is_aggregated_video=True and must not be used in single-URL claims.
* vendor_page_types: the modal vendor page type per engine (from rest X1).
* non_vendor_domains: top 10 domains whose citation is not Vendor-published, with count and share.
* vendor_domains: the top 10 vendor-published domains per engine split in three shades (see below).
* search: web searches per answer per engine, the ChatGPT site: share, two example query sets.
* wording3: the verbatim Gemini wording 3 of the unbranded general question and its no-search rate.
* k1: top 5 names per engine (display names) and which are newly in frame after the 2026-10-12 relabel.
* models, study: model_returned counts and the counts a method box needs, in one place.

Vendor-domain shades (each vendor-published citation of a top-10 domain lands in exactly one):
1. named_in_scope: the page is on the domain of an in-scope company (registry meets_frame == 'in') that is
   named in that same answer. This is S1's own_page + own_listicle count (self_reference_count); asserted.
2. out_of_scope_vendor: the page is on the domain of a registry entity the frame labels out, borderline or
   unresolved (meets_frame != 'in'). "Phone or voice vendor" is A5's own judgment on the citation
   (publisher_is_in_category: the publisher is itself a company selling an AI receptionist / AI voice agent
   product), the same judgment that puts the citation in the vendor-published class at all; an entity the
   frame calls out for another reason (for example adjacent_product) is still a phone or voice vendor here.
   Whether that entity was named in the answer is recorded (named_in_answer_count), but the shade does not
   depend on it.
3. other_vendor_not_named: any other vendor-published citation. Split in two for the record: the domain
   belongs to an in-scope company that is not named in that answer (in_scope_entity_not_named), or no
   registry entity owns the domain (no_registry_entity).
Domain to entity: entity_owned_domains (homepage, else verified homepage, plus alt_domains) of EVERY registry
entity, reduced with registrable_domain. A domain owned by several entities goes first to the entity whose own
registry homepage_url it is, then to one whose verified homepage it is, then to one that lists it as an alternate
domain (alias rows come last); ties go to an in-scope entity,
then to the first by entity_id; all candidates are listed in entity_ids.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import json
from pathlib import Path
import statistics
from collections import Counter, defaultdict
from typing import Any

from analysis.coding.b1_per_run_company import in_frame_entities, surface_form_to_entity_id
from analysis.coding.registrable_domain import registrable_domain
from analysis.phase_b import exclusions
from analysis.phase_b.business import s1_business as s1b
from analysis.phase_b.business import source_classes as sc
from analysis.phase_b.business.extremes import ENGINE_ORDER, extreme, ordered
from analysis.phase_b.citation_panel import (
    UNBRANDED_INTENT_IDS,
    CitationRow,
    build_citation_panel,
    mentioned_own_domains_by_run,
)
from analysis.phase_b.domain_ownership import entity_owned_domains
from analysis.phase_b.reading import FrozenInputs
from collector import paths

OPENAI, AI_MODE, GEMINI, CLAUDE = ENGINE_ORDER
NEWLY_IN_FRAME = ("retell-ai", "dograh", "pipecat")  # in frame only since the 2026-10-12 relabel
TOP_URLS_N = 5
TOP_NON_VENDOR_N = 10
TOP_VENDOR_DOMAINS_N = s1b.TOP_DOMAINS_N
K1_TOP_N = 5
# Shown for an engine whose response carries no model name (Google AI Mode, collected through a scraper).
NO_MODEL_LABEL = "no model name returned, collected through a scraper"
# The 2026-10-12 relabel in numbers: config/relabel_record_2026-10-12.json (tracked), checked against the ledger
# entry frame-label-fix-2026-10-12-outcome by a test, so the counts cannot drift from the text of the ledger.
RELABEL_RECORD_PATH = paths.CONFIG_DIR / "relabel_record_2026-10-12.json"


def _relabel(record_path=RELABEL_RECORD_PATH) -> dict[str, Any]:
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    return {
        "date": rec["date"],
        "calls": len(rec["calls_made"]),
        "invalid_again_decided_by_hand": len(rec["invalid_again_decided_by_hand"]),
        "accepted_model_output": len(rec["accepted_model_output"]),
        "set_by_hand_no_call": len(rec["set_by_hand_no_call"]),
        "ledger_entry": rec["ledger_entry"],
    }


WORDING_INTENT_ID, WORDING_PARAPHRASE_ID = 1, 3
EXAMPLE_INTENT_ID, EXAMPLE_PARAPHRASE_ID = 1, 1
YOUTUBE_PATH_NOTE = (
    "url_normalized drops the query string, so every YouTube video collapses to youtube.com/watch; "
    "that entry is many videos, not one URL."
)
# Public source classes: the four classes the report names, the video-and-social class, and the rest.
PUBLIC_CLASS_KEYS = {
    sc.VENDOR: "vendor_published",
    sc.REVIEW: "review_sites",
    sc.NEWS: "news_and_press",
    sc.FORUMS: "forums",
    sc.VIDEO_SOCIAL: "video_and_social",
}
OTHER_KEY = "other"
NOTE = (
    "Exploratory and derived; separate from figures.json. Counts come from the frozen citation panel, "
    "registry, mentions_raw and raw runs; shares are fractions of stored counts. Company names appear only as "
    "display names of cited domains and top-5 entries."
)


class PublicBusinessError(Exception):
    """A stored figure no longer matches what this file recomputes, or an input has the wrong shape."""


def _share(count: int | float, total: int | float) -> float | None:
    return count / total if total else None


def _cell(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "share": _share(count, total)}


def _run_meta(inputs: FrozenInputs) -> dict[str, dict[str, Any]]:
    return {r["run_id"]: r for r in inputs.raw_runs}


def _classified_by_engine(rows: list[CitationRow]) -> dict[str, list[CitationRow]]:
    out: dict[str, list[CitationRow]] = defaultdict(list)
    for r in rows:
        if r.url_class is not None:
            out[r.engine].append(r)
    return out


# --- 1. source classes -----------------------------------------------------------------------------

def _source_classes(by_engine: dict[str, list[CitationRow]], domain_map: dict[str, str]) -> dict[str, Any]:
    engines: dict[str, Any] = {}
    for e in ordered(by_engine):
        rows = by_engine[e]
        n = len(rows)
        classes = Counter(sc.display_class(r, domain_map) for r in rows)
        domains = Counter(sc.row_domain(r) for r in rows)
        public = {key: _cell(classes.get(cls, 0), n) for cls, key in PUBLIC_CLASS_KEYS.items()}
        other = n - sum(c["count"] for c in public.values())
        public[OTHER_KEY] = _cell(other, n)
        engines[e] = {
            "n_classified": n,
            "classes": public,
            "display_classes": {cls: _cell(classes.get(cls, 0), n) for cls in sc.DISPLAY_CLASSES},
            "domains": {
                "reddit_com": _cell(domains.get("reddit.com", 0), n),
                "youtube_com": _cell(domains.get("youtube.com", 0), n),
                "youtu_be": _cell(domains.get("youtu.be", 0), n),
            },
        }
    return {"engines": engines}


def _check_against_s1(source: dict[str, Any], s1_business: dict[str, Any]) -> None:
    """The counts must be S1's: same classified-citation total, vendor-published count and class counts."""
    for e, block in source["engines"].items():
        stored = s1_business["engines"][e]
        if block["n_classified"] != stored["classified_citations"]:
            raise PublicBusinessError(f"{e}: classified citations {block['n_classified']} != S1 {stored['classified_citations']}")
        if block["classes"]["vendor_published"]["count"] != stored["vendor_published"]["count"]:
            raise PublicBusinessError(f"{e}: vendor-published count differs from S1")
        for cls, cell in block["display_classes"].items():
            if cell["count"] != stored["source_mix"][cls]["count"]:
                raise PublicBusinessError(f"{e}: class {cls!r} count differs from S1's source_mix")


# --- 2. URLs ------------------------------------------------------------------------------------------

def _is_aggregated_video(url: str) -> bool:
    domain = registrable_domain(url) or ""
    return domain in {"youtube.com", "youtu.be"}


def _urls(by_engine: dict[str, list[CitationRow]]) -> dict[str, Any]:
    engines: dict[str, Any] = {}
    for e in ordered(by_engine):
        rows = by_engine[e]
        counts: Counter = Counter(r.url_normalized for r in rows)
        runs: dict[str, set[str]] = defaultdict(set)
        meta: dict[str, CitationRow] = {}
        for r in rows:
            runs[r.url_normalized].add(r.run_id)
            meta.setdefault(r.url_normalized, r)
        vendor_urls = {u for u, r in meta.items() if r.publisher_is_in_category is True}
        vendor_citations = sum(1 for r in rows if r.publisher_is_in_category is True)
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_URLS_N]
        top_urls = [
            {
                "url": u, "domain": sc.row_domain(meta[u]), "citations": c, "distinct_runs": len(runs[u]),
                "share_of_citations": _share(c, len(rows)),
                "url_class": meta[u].url_class, "vendor_published": meta[u].publisher_is_in_category is True,
                "is_aggregated_video": _is_aggregated_video(u),
            }
            for u, c in top
        ]
        engines[e] = {
            "citations": _cell(len(rows), len(rows)),
            "vendor_citations": _cell(vendor_citations, len(rows)),
            "unique_urls": len(counts),
            "unique_vendor_urls": _cell(len(vendor_urls), len(counts)),
            "top_urls": top_urls,
            "top_urls_citations": sum(t["citations"] for t in top_urls),
            "top_urls_share": _share(sum(t["citations"] for t in top_urls), len(rows)),
            "aggregated_video_entries": sum(t["is_aggregated_video"] for t in top_urls),
        }
    return {"video_note": YOUTUBE_PATH_NOTE, "engines": engines}


def _check_urls(urls: dict[str, Any], figures: dict[str, Any]) -> None:
    stored = figures["figures"]["S1"]["notes"]["unique_url_vendor_share_by_engine"]
    for e, block in urls["engines"].items():
        got = block["unique_vendor_urls"]["share"]
        if abs(got - stored[e]) > 1e-12:
            raise PublicBusinessError(f"{e}: unique-URL vendor share {got} != stored S1 {stored[e]}")


def _vendor_page_types(rest: dict[str, Any]) -> dict[str, Any]:
    x1 = rest["x1"]["engines"]
    engines = {
        e: {"modal_class": x1[e]["modal_class"], "modal_share": x1[e]["modal_share"],
            "modal_margin": x1[e]["modal_margin"], "weak_modal": x1[e]["weak_modal"],
            "raw_vendor_share": x1[e]["raw_vendor_share"], "n_classified": x1[e]["n_classified"]}
        for e in ordered(x1)
    }
    return {
        "engines": engines,
        "flags": {
            "product_pages_lead_in_openai_chatgpt_and_google_ai_mode": all(engines[e]["modal_class"] == "vendor page" for e in (OPENAI, AI_MODE)),
            "vendor_listicles_lead_in_google_gemini_and_anthropic_claude": all(engines[e]["modal_class"] == "vendor listicle" for e in (GEMINI, CLAUDE)),
            "google_ai_mode_lead_is_weak": engines[AI_MODE]["weak_modal"],
            "anthropic_claude_modal_share_about_three_quarters": abs(engines[CLAUDE]["modal_share"] - 0.75) <= 0.03,
        },
    }


# --- 3. non-vendor domains --------------------------------------------------------------------------

def _non_vendor_domains(by_engine: dict[str, list[CitationRow]], domain_map: dict[str, str]) -> dict[str, Any]:
    engines: dict[str, Any] = {}
    for e in ordered(by_engine):
        rows = by_engine[e]
        n = len(rows)
        counts: Counter = Counter()
        cls_of: dict[str, str] = {}
        for r in rows:
            cls = sc.display_class(r, domain_map)
            if cls == sc.VENDOR:
                continue
            d = sc.row_domain(r)
            counts[d] += 1
            cls_of[d] = cls
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_NON_VENDOR_N]
        engines[e] = {
            "n_classified": n,
            "n_non_vendor": sum(counts.values()),
            "distinct_domains": len(counts),
            "top_domains": [
                {"domain": d, "display_class": cls_of[d], "count": c, "total": n, "share": _share(c, n)}
                for d, c in top
            ],
        }
    return {"engines": engines}


# --- 4. vendor domains in three shades ----------------------------------------------------------------

def _all_surface_forms(inputs: FrozenInputs) -> dict[str, str]:
    """surface_form -> entity_id over EVERY registry entity (as K1 does), with in-scope entities winning a
    clash so the in-scope mapping equals analysis.coding.b1_per_run_company's."""
    mapping = surface_form_to_entity_id(inputs.entities)
    mapping.update(surface_form_to_entity_id(in_frame_entities(inputs.entities)))
    return mapping


def _entity_by_domain(inputs: FrozenInputs) -> dict[str, list[dict[str, Any]]]:
    verified = {
        r["entity_id"]: r["verified_homepage_url"] for r in inputs.entity_verification_rows if r.get("verified_homepage_url")
    }
    owners: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    source_rank = {"homepage": 0, "verified_homepage": 1}
    malformed: list[str] = []
    for entity in inputs.entities:
        for domain, source in entity_owned_domains(entity, inputs.alt_domains_rows, malformed, verified).items():
            owners[domain].append((source_rank.get(source, 2), entity))
    # Primary owner first: the entity whose registry homepage_url is the domain, then one whose verified
    # homepage is, then one that only lists it as an alternate domain; ties go to an in-scope entity, then to
    # entity_id. Alias rows (for example a product name that lists the company's domain) come after.
    return {
        domain: [en for _, en in sorted(found, key=lambda t: (t[0], t[1]["meets_frame"] != "in", t[1]["entity_id"]))]
        for domain, found in owners.items()
    }


def _named_by_run(inputs: FrozenInputs, usable: set[str]) -> dict[str, set[str]]:
    sf = _all_surface_forms(inputs)
    named: dict[str, set[str]] = defaultdict(set)
    for m in inputs.mentions_raw:
        entity_id = sf.get(m.get("surface_form"))
        if entity_id is not None and m["run_id"] in usable:
            named[m["run_id"]].add(entity_id)
    return named


def _vendor_domains(
    inputs: FrozenInputs, by_engine: dict[str, list[CitationRow]], usable: set[str], meta: dict[str, dict[str, Any]],
    entity_rates: dict[tuple[str, str], int], unbranded_totals: dict[str, int],
) -> dict[str, Any]:
    owners = _entity_by_domain(inputs)
    named = _named_by_run(inputs, usable)
    mentioned_own = mentioned_own_domains_by_run(inputs)
    engines: dict[str, Any] = {}
    for e in ordered(by_engine):
        vendor_rows = [r for r in by_engine[e] if r.publisher_is_in_category is True]
        totals: Counter = Counter(sc.row_domain(r) for r in vendor_rows)
        top = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_VENDOR_DOMAINS_N]
        top_domains = {d for d, _ in top}
        shades: dict[str, Counter] = defaultdict(Counter)
        out_named: Counter = Counter()
        other_split: dict[str, Counter] = defaultdict(Counter)
        for r in vendor_rows:
            d = sc.row_domain(r)
            if d not in top_domains:
                continue
            if s1b.self_reference_part(r, mentioned_own) != s1b.OTHER_VENDOR:
                shades[d]["named_in_scope"] += 1
                continue
            entities = owners.get(d, [])
            primary = entities[0] if entities else None
            if primary is not None and primary["meets_frame"] != "in":
                shades[d]["out_of_scope_vendor"] += 1
                if primary["entity_id"] in named.get(r.run_id, set()):
                    out_named[d] += 1
            else:
                shades[d]["other_vendor_not_named"] += 1
                other_split[d]["in_scope_entity_not_named" if primary is not None else "no_registry_entity"] += 1
        domains = []
        for d, total in top:
            entities = owners.get(d, [])
            primary = entities[0] if entities else None
            s = shades[d]
            record: dict[str, Any] = {
                "domain": d, "count": total,
                "named_in_scope": s["named_in_scope"],
                "out_of_scope_vendor": s["out_of_scope_vendor"],
                "other_vendor_not_named": s["other_vendor_not_named"],
                "other_vendor_split": {
                    "in_scope_entity_not_named": other_split[d]["in_scope_entity_not_named"],
                    "no_registry_entity": other_split[d]["no_registry_entity"],
                },
                "entity_id": primary["entity_id"] if primary else None,
                "entity_name": primary["canonical_name"] if primary else None,
                "entity_ids": [en["entity_id"] for en in entities],
                "entity_meets_frame": primary["meets_frame"] if primary else None,
                "entity_in_scope": bool(primary and primary["meets_frame"] == "in"),
                "out_of_scope_named_in_answer_count": out_named[d],
                # Two-shade view used by the public chart: named in that answer (any company in the registry) or not.
                "named_in_answer": s["named_in_scope"] + out_named[d],
                "not_named_in_answer": total - s["named_in_scope"] - out_named[d],
            }
            if primary is not None:
                record["entity_runs_naming_it"] = _cell(
                    entity_rates.get((e, primary["entity_id"]), 0), unbranded_totals[e])
            else:
                record["entity_runs_naming_it"] = None
            if s["named_in_scope"] + s["out_of_scope_vendor"] + s["other_vendor_not_named"] != total:
                raise PublicBusinessError(f"{e}/{d}: shades do not sum to the domain total")
            domains.append(record)
        engines[e] = {"vendor_citations": len(vendor_rows), "top_domains": domains}
    return {"engines": engines}


def _check_vendor_domains(vd: dict[str, Any], s1_business: dict[str, Any]) -> None:
    for e, block in vd["engines"].items():
        stored = s1_business["engines"][e]["vendor_published"]["top_domains"]
        got = block["top_domains"]
        if [(d["domain"], d["count"]) for d in got] != [(d["domain"], d["count"]) for d in stored]:
            raise PublicBusinessError(f"{e}: top vendor domains differ from S1's")
        for g, s in zip(got, stored):
            if g["named_in_scope"] != s["self_reference_count"]:
                raise PublicBusinessError(f"{e}/{g['domain']}: named_in_scope != S1 self_reference_count")


# --- 5. search behaviour --------------------------------------------------------------------------------

def gemini_queries(response: dict[str, Any]) -> list[str]:
    candidate = (response.get("candidates") or [{}])[0] or {}
    metadata = candidate.get("grounding_metadata") or {}
    return [q for q in (metadata.get("web_search_queries") or []) if isinstance(q, str)]


def chatgpt_search_calls(response: dict[str, Any]) -> list[list[str]]:
    """One list of query strings per web_search_call output item that carried queries (open_page and
    find_in_page actions carry none and are not searches)."""
    calls: list[list[str]] = []
    for item in response.get("output") or []:
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        queries = action.get("queries") or ([action["query"]] if action.get("query") else [])
        queries = [q for q in queries if isinstance(q, str)]
        if queries:
            calls.append(queries)
    return calls


def claude_queries(response: dict[str, Any]) -> list[str]:
    out = []
    for block in response.get("content") or []:
        if block.get("type") == "server_tool_use" and block.get("name") == "web_search":
            q = (block.get("input") or {}).get("query")
            if isinstance(q, str):
                out.append(q)
    return out


def _queries_for(engine: str, response: dict[str, Any]) -> list[str] | None:
    """Flat list of search query strings of one answer; None when the engine does not store them."""
    if engine == GEMINI:
        return gemini_queries(response)
    if engine == OPENAI:
        return [q for call in chatgpt_search_calls(response) for q in call]
    if engine == CLAUDE:
        return claude_queries(response)
    return None


def _search(inputs: FrozenInputs, usable: set[str]) -> dict[str, Any]:
    runs = sorted((r for r in inputs.raw_runs if r["run_id"] in usable), key=lambda r: r["run_id"])
    per_engine: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    ai_mode_keys: set[str] = set()
    for run in runs:
        envelope = run.get("raw_response_envelope")
        response = json.loads(envelope) if envelope else {}
        per_engine[run["engine"]].append((run, response))
        if run["engine"] == AI_MODE and isinstance(response, dict):
            ai_mode_keys |= set(response)
    engines: dict[str, Any] = {}
    for e in ordered(per_engine):
        items = per_engine[e]
        if e == AI_MODE:
            engines[e] = {
                "available": False,
                "reason": "the SerpApi google_ai_mode response stores no list of search queries",
                "response_keys_seen": sorted(ai_mode_keys),
                "n_answers": len(items),
            }
            continue
        counts = [len(_queries_for(e, resp) or []) for _, resp in items]
        searched = [c for c in counts if c > 0]
        block: dict[str, Any] = {
            "available": True,
            "n_answers": len(items),
            "n_queries": sum(counts),
            "mean_queries_per_answer": _share(sum(counts), len(items)),
            "n_answers_with_search": len(searched),
            "mean_queries_per_searching_answer": _share(sum(searched), len(searched)),
            "max_queries_in_one_answer": max(counts) if counts else 0,
        }
        if e == OPENAI:
            calls = [len(chatgpt_search_calls(resp)) for _, resp in items]
            all_queries = [q for _, resp in items for q in (_queries_for(e, resp) or [])]
            site = [q for q in all_queries if "site:" in q.lower()]
            answers_with_site = sum(
                1 for _, resp in items if any("site:" in q.lower() for q in (_queries_for(e, resp) or []))
            )
            block.update(
                {
                    "n_search_calls": sum(calls),
                    "mean_search_calls_per_answer": _share(sum(calls), len(items)),
                    "site_restricted_queries": _cell(len(site), len(all_queries)),
                    "answers_with_site_query": _cell(answers_with_site, len(searched)),
                }
            )
        engines[e] = block
    return {"engines": engines, "examples": _search_examples(per_engine)}


def _search_examples(per_engine: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> dict[str, Any]:
    """Rule: the first usable run of the engine for intent 1, paraphrase 1, ordered by day then repetition
    (that is day 1, repetition 1 unless that run is unusable). Run id and exact query strings are stored."""
    examples: dict[str, Any] = {}
    for e in (GEMINI, OPENAI):
        pool = sorted(
            (it for it in per_engine.get(e, []) if it[0]["intent_id"] == EXAMPLE_INTENT_ID and it[0]["paraphrase_id"] == EXAMPLE_PARAPHRASE_ID),
            key=lambda it: (it[0]["day_index"], it[0]["rep_index"], it[0]["run_id"]),
        )
        if not pool:
            raise PublicBusinessError(f"no usable {e} run for the example question")
        run, response = pool[0]
        entry: dict[str, Any] = {
            "run_id": run["run_id"], "day_index": run["day_index"], "rep_index": run["rep_index"],
            "question": run["prompt_text"], "queries": _queries_for(e, response),
        }
        if e == OPENAI:
            entry["query_groups"] = chatgpt_search_calls(response)
        examples[e] = entry
    return {
        "rule": "first usable run for intent 1, paraphrase 1, ordered by day then repetition",
        "intent_id": EXAMPLE_INTENT_ID, "paraphrase_id": EXAMPLE_PARAPHRASE_ID,
        "engines": examples,
    }


# --- 6. Gemini wording 3 --------------------------------------------------------------------------------

def _wording3(inputs: FrozenInputs, usable: set[str], rest: dict[str, Any]) -> dict[str, Any]:
    with paths.PROMPTS_CSV.open(newline="", encoding="utf-8") as f:
        prompts = list(csv.DictReader(f))
    rows = [p for p in prompts if int(p["intent_id"]) == WORDING_INTENT_ID and int(p["paraphrase_id"]) == WORDING_PARAPHRASE_ID]
    if len(rows) != 1:
        raise PublicBusinessError("config/prompts/prompts.csv has no unique intent 1, paraphrase 3 row")
    text = rows[0]["prompt_text"]
    gem = [r for r in inputs.raw_runs if r["run_id"] in usable and r["engine"] == GEMINI]
    skipped = [r for r in gem if r["search_fired"] is False]
    w3 = [r for r in gem if r["paraphrase_id"] == WORDING_PARAPHRASE_ID]
    w3_unbranded = [r for r in w3 if r["intent_id"] in UNBRANDED_INTENT_IDS]
    w3_skipped = [r for r in w3 if r["search_fired"] is False]
    w3_q1 = [r for r in w3 if r["intent_id"] == WORDING_INTENT_ID]
    per_intent = {
        str(i): _cell(sum(1 for r in w3 if r["intent_id"] == i and r["search_fired"] is False), sum(1 for r in w3 if r["intent_id"] == i))
        for i in sorted({r["intent_id"] for r in w3})
    }
    mean_unbranded = statistics.fmean(per_intent[str(i)]["share"] for i in sorted(UNBRANDED_INTENT_IDS))
    stored = rest["s4"]["gemini_top_paraphrase"]
    if stored["paraphrase_id"] != WORDING_PARAPHRASE_ID or abs(stored["mean_rate_unbranded"] - mean_unbranded) > 1e-12:
        raise PublicBusinessError("wording-3 no-search rate differs from rest S4")
    if len(skipped) != rest["s4"]["engines"][GEMINI]["n_not_fired"]:
        raise PublicBusinessError("Gemini no-search run count differs from rest S4")
    return {
        "intent_id": WORDING_INTENT_ID, "paraphrase_id": WORDING_PARAPHRASE_ID, "prompt_text": text,
        "gemini_runs_usable": len(gem),
        "gemini_no_search_runs": len(skipped),
        "gemini_no_search_runs_from_wording3": _cell(len(w3_skipped), len(skipped)),
        "wording3_runs_that_skip_search": _cell(len(w3_skipped), len(w3)),
        "wording3_unbranded_runs_that_skip_search": _cell(
            sum(1 for r in w3_unbranded if r["search_fired"] is False), len(w3_unbranded)),
        "wording3_question1_runs_that_skip_search": _cell(sum(1 for r in w3_q1 if r["search_fired"] is False), len(w3_q1)),
        "wording3_by_intent": per_intent,
        "mean_unbranded_rate": mean_unbranded,
        "other_wordings_no_search_runs": len(skipped) - len(w3_skipped),
    }


# --- 7. K1 top 5 ---------------------------------------------------------------------------------------

def _k1(inputs: FrozenInputs, figures: dict[str, Any]) -> dict[str, Any]:
    f = figures["figures"]["K1"]
    names = {e["entity_id"]: e["canonical_name"] for e in inputs.entities}
    engines: dict[str, Any] = {}
    for e in ordered(f["value"]):
        total = f["denominator_after_exclusion"][e]
        top = f["notes"]["top_5_ranked"][e][:K1_TOP_N]
        entries = [
            {"rank": i, "entity_id": t["entity_id"], "name": names[t["entity_id"]], "mentions": t["count"],
             "mentions_total": total, "share": t["share"], "newly_in_frame": t["entity_id"] in NEWLY_IN_FRAME}
            for i, t in enumerate(top, start=1)
        ]
        engines[e] = {
            "mentions_total": total,
            "count_to_half": f["notes"]["count_to_50pct"][e],
            "top5_share": f["value"][e],
            "top5": entries,
            "newly_in_frame_in_top5": [t["entity_id"] for t in entries if t["newly_in_frame"]],
        }
    entered = sorted({i for b in engines.values() for i in b["newly_in_frame_in_top5"]})
    return {
        "newly_in_frame_after_relabel": list(NEWLY_IN_FRAME),
        "engines": engines,
        "newly_in_frame_entities_in_any_top5": entered,
        "flags": {
            "retell_ai_in_a_top5": "retell-ai" in entered,
            "dograh_in_a_top5": "dograh" in entered,
            "pipecat_in_a_top5": "pipecat" in entered,
            "newly_in_frame_entity_in_a_top5": bool(entered),
        },
    }


# --- 8. models and study counts -----------------------------------------------------------------------

def _models(inputs: FrozenInputs, usable: set[str]) -> dict[str, Any]:
    returned: dict[str, Counter] = defaultdict(Counter)
    requested: dict[str, Counter] = defaultdict(Counter)
    for r in inputs.raw_runs:
        if r["run_id"] in usable:
            returned[r["engine"]][r["model_returned"] or "(none)"] += 1
            requested[r["engine"]][r["model_requested"] or "(none)"] += 1
    return {
        "engines": {
            e: {
                "n_usable": sum(returned[e].values()),
                "model_returned": dict(sorted(returned[e].items(), key=lambda kv: (-kv[1], kv[0]))),
                "model_requested": dict(sorted(requested[e].items(), key=lambda kv: (-kv[1], kv[0]))),
                "model_label": ", ".join(
                    n if n != "(none)" else NO_MODEL_LABEL for n in sorted(returned[e], key=lambda n: (-returned[e][n], n))),
            }
            for e in ordered(returned)
        }
    }


def _a1_rule(figures: dict[str, Any]) -> dict[str, Any]:
    """The Google top-10 rule of A1 (a question counts only if 20% to 80% of the qualifying companies are in
    Google's top ten for it), and how many questions met it, per engine."""
    from analysis.phase_b.figures import a1 as a1m

    included = figures["figures"]["A1"]["notes"]["included_intents_by_engine"]
    return {
        "low": a1m.INTENT_INCLUSION_LOW, "high": a1m.INTENT_INCLUSION_HIGH,
        "questions_meeting_rule_by_engine": {e: len(v) for e, v in included.items()},
        "questions_meeting_rule_total": sum(len(v) for v in included.values()),
        "untestable_in_every_engine": all(not v for v in included.values()),
        # The planned statistic (point-biserial r) is stored as null for every engine.
        "stored_value_null_in_every_engine": all(v is None for v in figures["figures"]["A1"]["value"].values()),
    }


def _window(window: dict[str, Any]) -> dict[str, Any]:
    start = datetime.date.fromisoformat(window["start"])
    end = datetime.date.fromisoformat(window["end"])
    return {**window, "month_label": f"{start:%B %Y}", "range_label": f"{start.day} to {end.day} {end:%B %Y}"}


# --- 8b. V1 without no-search runs ----------------------------------------------------------------------

def _v1_sensitivity(inputs: FrozenInputs, figures: dict[str, Any]) -> dict[str, Any]:
    """V1 (engine versus industry, analysis/phase_b/figures/v1.py, same method and primary population: companies
    qualifying in at least one engine) recomputed on the rates panel built from usable runs that did NOT skip the
    web search (search_fired is not False). The panel, the eligibility floor and the ratios are rebuilt on that
    smaller set. The same code on all usable runs must reproduce the stored V1 (asserted)."""
    from analysis.phase_b.figures import v1 as v1m
    from analysis.phase_b.rates_panel import build_rates_panel

    def run(runs: list[dict[str, Any]]) -> dict[str, Any]:
        cells, _report = build_rates_panel(dataclasses.replace(inputs, raw_runs=runs))
        engines = sorted({c.engine for c in cells})
        rate_by_key = {(c.entity_id, c.intent_id, c.engine): c.mention_rate for c in cells}
        qual = {c.entity_id for c in cells if c.qualifies}
        summary = v1m._summarize(sorted(qual), engines, rate_by_key)
        median = summary["median"]
        return {
            "n_companies_considered": summary["n_companies_considered"],
            "engine_matters_more": summary["n_above_line_engine_matters_more"],
            "industry_matters_more": summary["n_below_line_industry_matters_more"],
            "never_named_general": summary["n_excluded_engine_spread_zero"],
            "n_compared": summary["n_included"],
            "median_ratio": median,
            "engine_over_industry_median": (1 / median) if median else None,
        }

    all_usable = [r for r in inputs.raw_runs]
    baseline = run(all_usable)
    stored = figures["figures"]["V1"]
    if abs(baseline["median_ratio"] - stored["value"]) > 1e-9 or \
            baseline["engine_matters_more"] != stored["notes"]["n_above_line_engine_matters_more"]:
        raise PublicBusinessError("V1 recomputed on all runs differs from the stored V1")
    usable = exclusions.usable_run_ids(inputs.raw_runs)
    kept = [r for r in inputs.raw_runs if r["run_id"] not in usable or r["search_fired"] is not False]
    removed = sum(1 for r in inputs.raw_runs if r["run_id"] in usable and r["search_fired"] is False)
    filtered = run(kept)
    planned = figures["figures"]["V1"]["threshold"]["falsifier_at_or_below"]
    return {
        "planned_falsifier_at_or_below": planned,
        "stored_median_ratio_at_or_below_planned_line": stored["value"] <= planned,
        "method": "same as V1 (population SD of mention rate; median of industry spread over engine spread), "
                  "rates and eligibility rebuilt without usable runs where search_fired is False",
        "no_search_runs_removed": removed,
        "all_runs": baseline,
        "without_no_search_runs": filtered,
        "same_conclusion": filtered["engine_matters_more"] > filtered["industry_matters_more"]
        and filtered["median_ratio"] < 1,
    }


def _study(
    inputs: FrozenInputs, usable: set[str], figures: dict[str, Any], rest: dict[str, Any], named: dict[str, set[str]],
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entities = {e["entity_id"]: e for e in inputs.entities}
    n_in = len(in_frame_entities(inputs.entities))
    unbranded_runs = {rid for rid in usable if meta[rid]["intent_id"] in UNBRANDED_INTENT_IDS}
    out_named_engine: dict[str, set[str]] = defaultdict(set)
    out_named_all: set[str] = set()
    out_named_any_intent: set[str] = set()
    for rid, ids in named.items():
        for i in ids:
            if entities[i]["meets_frame"] == "in":
                continue
            out_named_any_intent.add(i)
            if rid in unbranded_runs:
                out_named_all.add(i)
                out_named_engine[meta[rid]["engine"]].add(i)
    stored_out = figures["figures"]["K1"]["notes"]["out_of_frame_named_by_engine"]
    for e, ids in out_named_engine.items():
        if len(ids) != stored_out[e]["total_out_of_frame_entities"]:
            raise PublicBusinessError(f"{e}: out-of-scope entities named {len(ids)} != K1 {stored_out[e]['total_out_of_frame_entities']}")
    method = rest["method"]
    cell_counts: Counter = Counter((r["engine"], r["intent_id"], r["paraphrase_id"], r["day_index"]) for r in inputs.raw_runs)
    runs_per_cell = sorted(set(cell_counts.values()))
    per_engine_planned = Counter(r["engine"] for r in inputs.raw_runs)
    reasons = Counter(exclusions.exclusion_reason(r) or "usable" for r in inputs.raw_runs)
    unusable = [
        {"run_id": r["run_id"], "engine": r["engine"], "reason": exclusions.exclusion_reason(r)}
        for r in inputs.raw_runs if exclusions.exclusion_reason(r) is not None
    ]
    n_intents, n_par, per_day, days = method["n_intents"], method["paraphrases_per_intent"], method["runs_per_day"], method["days"]
    return {
        "in_scope_companies_in_registry": n_in,
        "in_scope_companies_frozen_baseline": method["roster_size"],
        "registry_matches_frozen_baseline": n_in == method["roster_size"],
        "qualifying_companies_by_engine": dict(method["qualifying_companies"]),
        "eligibility_floor_mentions": method["eligibility_floor"],
        "out_of_scope_entities_named_unbranded": len(out_named_all),
        "out_of_scope_entities_named_any_intent": len(out_named_any_intent),
        "out_of_scope_entities_named_by_engine": {e: len(out_named_engine[e]) for e in ordered(out_named_engine)},
        "planned_runs": method["planned_runs"]["total"],
        "planned_runs_by_engine": {e: per_engine_planned[e] for e in ordered(per_engine_planned)},
        "usable_runs": len(usable),
        "runs_not_usable": len(inputs.raw_runs) - len(usable),
        "not_usable_runs": unusable,
        "exclusion_reasons": dict(reasons),
        "questions": n_intents, "unbranded_questions": method["n_unbranded_intents"],
        "wordings_per_question": n_par, "runs_per_day": per_day, "days": days,
        "runs_per_question_wording_day_values": runs_per_cell,
        "schedule_product": n_intents * n_par * per_day * days,
        "schedule_matches_planned_per_engine": all(v == n_intents * n_par * per_day * days for v in per_engine_planned.values()),
        "runs_per_question_wording_engine": per_day * days,
        "window": _window(method["window"]),
        "relabel": _relabel(),
        "frame_fixed_date": method["inclusion_frame_frozen"],
        "a1_rule": _a1_rule(figures),
        "ui_runs_per_question": rest["derived"]["ui_calibration"]["counts"]["ui_runs_per_prompt"],
        "api_runs_per_question_wording": rest["derived"]["ui_calibration"]["counts"]["api_runs_per_prompt"],
    }


# --- derived flags --------------------------------------------------------------------------------------

def _derive(result: dict[str, Any]) -> dict[str, Any]:
    src = result["source_classes"]["engines"]
    names = ordered(src)
    google = [e for e in (AI_MODE, GEMINI) if e in src]
    others = [e for e in names if e not in google]
    vendor = {e: src[e]["classes"]["vendor_published"]["share"] for e in names}
    review = {e: src[e]["classes"]["review_sites"]["share"] for e in names}
    yt = {e: src[e]["domains"]["youtube_com"]["share"] for e in names}
    rd = {e: src[e]["domains"]["reddit_com"]["share"] for e in names}
    forums = {e: src[e]["classes"]["forums"]["share"] for e in names}
    urls = result["urls"]["engines"]
    unique_vs_citation = {e: urls[e]["unique_vendor_urls"]["share"] - urls[e]["vendor_citations"]["share"] for e in names}
    search = result["search"]["engines"]
    w3 = result["wording3"]
    return {
        "vendor_share": extreme(vendor),
        "review_sites_share": extreme(review),
        "youtube_share": extreme(yt),
        "youtube_share_google_engines": extreme({e: yt[e] for e in google}),
        "reddit_share": extreme(rd),
        "reddit_share_google_engines": extreme({e: rd[e] for e in google}),
        "unique_vendor_minus_citation_vendor_share": unique_vs_citation,
        "unique_url_vendor_share": extreme({e: urls[e]["unique_vendor_urls"]["share"] for e in names}),
        "flags": {
            "vendor_published_largest_public_class_in_every_engine": all(
                src[e]["classes"]["vendor_published"]["count"] > max(
                    c["count"] for k, c in src[e]["classes"].items() if k != "vendor_published") for e in names),
            "vendor_share_above_55_percent_in_every_engine": all(v > 0.55 for v in vendor.values()),
            "review_sites_under_1_percent_in_every_engine": all(v < 0.01 for v in review.values()),
            "youtube_present_in_both_google_engines": all(src[e]["domains"]["youtube_com"]["count"] > 0 for e in google),
            "youtube_absent_in_other_engines": all(src[e]["domains"]["youtube_com"]["count"] == 0 for e in others),
            "youtube_above_10_percent_in_both_google_engines": all(yt[e] > 0.10 for e in google),
            "reddit_present_in_both_google_engines": all(src[e]["domains"]["reddit_com"]["count"] > 0 for e in google),
            "reddit_absent_in_other_engines": all(src[e]["domains"]["reddit_com"]["count"] == 0 for e in others),
            "reddit_under_4_percent_in_every_engine": all(v < 0.04 for v in rd.values()),
            "reddit_below_youtube_in_both_google_engines": all(rd[e] < yt[e] for e in google),
            "forums_class_only_in_google_engines": all(src[e]["classes"]["forums"]["count"] > 0 for e in google)
            and all(src[e]["classes"]["forums"]["count"] == 0 for e in others),
            "unique_url_vendor_share_above_70_percent_in_every_engine": all(
                urls[e]["unique_vendor_urls"]["share"] > 0.70 for e in names),
            "unique_vendor_share_above_citation_share_in_gemini": unique_vs_citation[GEMINI] > 0,
            "unique_vendor_share_above_citation_share_in_both_google_engines": all(unique_vs_citation[e] > 0 for e in google),
            "unique_vendor_share_below_citation_share_in_chatgpt_and_claude": all(unique_vs_citation[e] < 0 for e in (OPENAI, CLAUDE)),
            "top_url_is_aggregated_video_in_google_engines": all(urls[e]["top_urls"][0]["is_aggregated_video"] for e in google),
            "top_url_is_a_single_page_in_chatgpt_and_claude": not any(urls[e]["top_urls"][0]["is_aggregated_video"] for e in (OPENAI, CLAUDE)),
            "google_ai_mode_stores_no_search_queries": not search[AI_MODE]["available"],
            "chatgpt_runs_more_queries_per_answer_than_gemini": search[OPENAI]["mean_queries_per_answer"] > search[GEMINI]["mean_queries_per_answer"],
            "chatgpt_site_restricted_queries_a_majority": search[OPENAI]["site_restricted_queries"]["share"] > 0.5,
            "wording3_skips_search_in_most_runs": w3["wording3_runs_that_skip_search"]["share"] > 0.5,
            "wording3_accounts_for_most_gemini_no_search_runs": w3["gemini_no_search_runs_from_wording3"]["share"] > 0.5,
            "wording3_question1_skips_search_in_over_90_percent_of_runs": w3["wording3_question1_runs_that_skip_search"]["share"] > 0.9,
            "retell_ai_in_a_top5": result["k1"]["flags"]["retell_ai_in_a_top5"],
            "registry_matches_frozen_baseline": result["study"]["registry_matches_frozen_baseline"],
            "a1_untestable_as_planned_in_every_engine": result["study"]["a1_rule"]["untestable_in_every_engine"],
            "a1_stored_values_null_in_every_engine": result["study"]["a1_rule"]["stored_value_null_in_every_engine"],
            "v1_median_ratio_at_or_below_planned_line": result["v1_sensitivity"]["stored_median_ratio_at_or_below_planned_line"],
            "v1_without_no_search_runs_engine_matters_more": result["v1_sensitivity"]["same_conclusion"],
        },
    }


# --- entry point ----------------------------------------------------------------------------------------

def build_public_figures(
    inputs: FrozenInputs, figures: dict[str, Any], rest: dict[str, Any], s1_business: dict[str, Any],
    domain_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """`figures` is figures.json, `rest` business_figures_rest.json, `s1_business` business_figures.json
    (all already built from the same registry). The S1 counts are asserted, never copied."""
    domain_map = sc.load_domain_map() if domain_map is None else domain_map
    rows, _report = build_citation_panel(inputs)
    by_engine = _classified_by_engine(rows)
    usable = exclusions.usable_run_ids(inputs.raw_runs)
    meta = _run_meta(inputs)
    named = _named_by_run(inputs, usable)
    unb_by_engine: dict[str, set[str]] = defaultdict(set)
    for rid in usable:
        if meta[rid]["intent_id"] in UNBRANDED_INTENT_IDS:
            unb_by_engine[meta[rid]["engine"]].add(rid)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for rid, ids in named.items():
        e = meta[rid]["engine"]
        if rid in unb_by_engine.get(e, set()):
            for i in ids:
                counts[(e, i)] += 1
    unb_totals = {e: len(ids) for e, ids in unb_by_engine.items()}

    source = _source_classes(by_engine, domain_map)
    _check_against_s1(source, s1_business)
    urls = _urls(by_engine)
    _check_urls(urls, figures)
    vendor_domains = _vendor_domains(inputs, by_engine, usable, meta, counts, unb_totals)
    _check_vendor_domains(vendor_domains, s1_business)

    result: dict[str, Any] = {
        "figure_id": "public_business",
        "status": "exploratory_derived",
        "note": NOTE,
        "engine_order": list(ENGINE_ORDER),
        "source_classes": source,
        "urls": urls,
        "vendor_page_types": _vendor_page_types(rest),
        "non_vendor_domains": _non_vendor_domains(by_engine, domain_map),
        "vendor_domains": vendor_domains,
        "search": _search(inputs, usable),
        "wording3": _wording3(inputs, usable, rest),
        "k1": _k1(inputs, figures),
        "models": _models(inputs, usable),
        "study": _study(inputs, usable, figures, rest, named, meta),
        "v1_sensitivity": _v1_sensitivity(inputs, figures),
    }
    result["derived"] = _derive(result)
    return result
