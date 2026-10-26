#!/usr/bin/env python3
"""Personal-data cleaning of the publish package. Deterministic: the same
input gives the same output (files are read in sorted order, rows in order,
no randomness, no model call other than the local spaCy model).

    uv sync --group cleaning                                  # once: the optional dependencies
    uv run --group cleaning python cleaning/clean_personal_data.py --private-dir <folder outside the package>

The dependencies (presidio-analyzer, presidio-anonymizer, spaCy with the
en_core_web_lg model, phonenumbers) are the optional `cleaning` dependency
group in pyproject.toml; the one command (scripts/build_publish_package.py)
does not need them.

What it replaces, in every shipped text field (ChatGPT and Claude answer
text, quotes, page titles, search queries and thinking text in data/raw/,
the app-check texts, coded-mention quotes, page titles in the cited-URL
resolution file, free-text notes in the tables and the by-eye check), in
every cited URL (all engines), and in robots.txt comments:

    email address               -> [EMAIL]      (at a keep-list domain only the
                                                 part before @ is replaced)
    phone number (phonenumbers,
      region US)                -> [PHONE]
    US street address           -> [ADDRESS]
    person name (spaCy PERSON)  -> [PERSON]     (two or more words with a capital;
                                                 a one-word hit scores below the
                                                 threshold and is kept and logged)
    personal profile URL        -> scheme and host kept, the rest becomes
                                   /[PROFILE-n]: one number per distinct page
                                   (the URL without query, fragment, www or a
                                   trailing slash, the way the figures count
                                   pages), numbered in order of first
                                   appearance (sorted file, row, position)
    a term in remove_list.txt   -> [REMOVED]

Never replaced: anything that overlaps a term of keep_list.txt (every
company name, alias and domain in the shipped tables, the engine names and
the vendors named in the prompts), linkedin.com/company/ pages, and a
profile URL whose handle is a keep-list name or domain label.

A detection is replaced when its score is at least REMOVE_AT_OR_ABOVE.
Everything detected and not replaced is logged with its reason.

Outputs:
    cleaning/keep_list.txt                 generated from the shipped tables
    cleaning/CLEANING_REPORT.md            counts and pointers, no original value
    <private dir>/cleaning_log.csv         every replacement, with the original
    <private dir>/kept_hits.csv            every detection that was kept, with the reason
    <private dir>/CLEANING_REPORT_private.md   the full report with values

The private dir has no default: pass --private-dir, a folder outside the
package. It receives the removed values, so never share it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse, urlunparse

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

CLEANING_DIR = Path("cleaning")
KEEP_LIST = CLEANING_DIR / "keep_list.txt"
REMOVE_LIST = CLEANING_DIR / "remove_list.txt"
REPORT = CLEANING_DIR / "CLEANING_REPORT.md"
SPACY_MODEL = "en_core_web_lg"
REMOVE_AT_OR_ABOVE = 0.5
SINGLE_WORD_PERSON_SCORE = 0.45
REPORT_KEPT_AT_OR_ABOVE = 0.5
TOP_N = 30
CONTEXT_CHARS = 60
MIN_KEEP_TERM = 3

EMAIL, PHONE, ADDRESS, PERSON, PROFILE, REMOVED = "EMAIL", "PHONE", "ADDRESS", "PERSON", "PROFILE_URL", "REMOVE_LIST"
TYPES = (EMAIL, PHONE, ADDRESS, PERSON, PROFILE, REMOVED)
TOKEN = {EMAIL: "[EMAIL]", PHONE: "[PHONE]", ADDRESS: "[ADDRESS]", PERSON: "[PERSON]", REMOVED: "[REMOVED]"}
PRESIDIO_ENTITY = {"EMAIL_ADDRESS": EMAIL, "PHONE_NUMBER": PHONE, "US_STREET_ADDRESS": ADDRESS, "PERSON": PERSON}
REASON_KEEP, REASON_LOW, REASON_COMPANY_URL = "keep-list", "below threshold", "company profile URL"

TEXT_ENGINES = ("openai_chatgpt", "anthropic_claude")
ENGINE_NAMES = ("OpenAI", "ChatGPT", "OpenAI ChatGPT", "Google", "Gemini", "Google Gemini", "Google AI Mode", "AI Mode",
                "Anthropic", "Claude", "Anthropic Claude")
# Keys of a ChatGPT or Claude window row whose string values are text (everything else gets the URL rule only).
RAW_TEXT_KEYS = {"text", "cited_text", "title", "query", "queries", "thinking", "pattern"}
MISS_TRIGGERS = re.compile(r"(?i)\b(founder|ceo|owner|attorney|call us|email us|contact)\b|\bDr\.|@|\d{7,}")
TOKEN_ON_LINE = re.compile(r"\[(EMAIL|PHONE|ADDRESS|PERSON|REMOVED|PROFILE-\d+)\]")

# --- personal profile URLs ---------------------------------------------------------------------------------
PROFILE_URL = re.compile(
    r"(?<![A-Za-z0-9.\-/@])(?P<scheme>https?://)?"
    r"(?P<host>(?:[A-Za-z0-9-]+\.)*(?:linkedin|x|twitter|facebook|instagram|tiktok|youtube)\.com)"
    r"(?P<path>/[^\s\"'<>()\[\]\\|,`]*)",
    re.IGNORECASE,
)
MAIN_HOSTS = {"", "www", "m", "mobile", "web", "mbasic", "business"}
RESERVED = {
    "x": {"i", "intent", "share", "home", "search", "hashtag", "explore", "settings", "login", "signup", "tos", "privacy",
          "messages", "notifications", "compose", "status", "about"},
    "facebook": {"tr", "csp", "ajax", "browser_reporting", "groups", "pages", "events", "sharer", "sharer.php", "share",
                 "plugins", "dialog", "watch", "fb-answers", "login", "login.php", "help", "policies", "policy.php",
                 "privacy", "hashtag", "photo.php", "photo", "permalink.php", "story.php", "business", "marketplace",
                 "gaming", "reel", "reels", "videos", "media", "legal", "ads", "about", "l.php", "v2.0", "terms",
                 "notes", "search", "settings", "places", "public", "careers", "watchparty", "live", "stories", "home.php",
                 "signals", "pg", "hashtag", "fundraisers", "messenger", "lite", "unsupportedbrowser"},
    "instagram": {"p", "reel", "reels", "explore", "stories", "tv", "error", "accounts", "about", "developer", "legal",
                  "direct", "web", "static"},
}
RESERVED["twitter"] = RESERVED["x"]
PERSONAL_FACEBOOK_PREFIXES = {"people", "profile.php"}


def person_score(span: str, score: float) -> float:
    """spaCy's PERSON score (0.85), lowered to SINGLE_WORD_PERSON_SCORE (below the
    threshold: kept and logged) when the span is one word or has no capital letter.
    A lone first or last name rarely identifies anyone, and in this data most
    one-word PERSON hits are product names and headings ("Voicemail")."""
    words = [w for w in re.split(r"\s+", span.strip()) if w]
    if len(words) < 2 or not any(ch.isupper() for ch in span):
        return SINGLE_WORD_PERSON_SCORE
    return round(score, 4)


def normalize_page(url: str) -> str:
    """The page a URL points to, the way the figures count pages
    (analysis/coding/url_normalize.py): no query or fragment, host lower case
    without www, no trailing slash."""
    parsed = urlparse(url if "://" in url else "https://" + url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return urlunparse((parsed.scheme, host, parsed.path.rstrip("/"), "", "", ""))


def profile_handle(host: str, path: str) -> str | None:
    """The account handle when host + path is a personal profile page, else None."""
    labels = host.lower().split(".")
    site, sub = labels[-2], ".".join(labels[:-2])
    segments = [s for s in path.split("/") if s]
    if not segments:
        return None
    first = segments[0]
    if site == "linkedin":
        return segments[1] if first.lower() == "in" and len(segments) > 1 else None
    if site in ("tiktok", "youtube"):
        return first[1:] if first.startswith("@") and len(first) > 1 else None
    if sub not in MAIN_HOSTS:
        return None
    if site == "facebook" and first.lower() in PERSONAL_FACEBOOK_PREFIXES:
        return segments[1] if first.lower() == "people" and len(segments) > 1 else first
    if site in RESERVED and first.lower() not in RESERVED[site] and not first.startswith(("?", "#")):
        if site in ("x", "twitter") and not re.fullmatch(r"[A-Za-z0-9_]{1,15}", first):
            return None
        if site == "instagram" and not re.fullmatch(r"[A-Za-z0-9._]{1,30}", first):
            return None
        if site == "facebook" and not re.fullmatch(r"[A-Za-z0-9.\-]{2,80}", first):
            return None
        return first
    return None


# --- logging -----------------------------------------------------------------------------------------------

@dataclass
class Hit:
    file: str
    row_id: str
    engine: str
    type: str
    original: str
    replacement: str
    before: str
    after: str
    score: float
    reason: str = ""


LOG_COLUMNS = ["file", "row_id", "engine", "type", "original", "replacement", "context_before", "context_after", "score"]
KEPT_COLUMNS = LOG_COLUMNS + ["reason"]


@dataclass
class Result:
    removed: list[Hit] = field(default_factory=list)
    kept: list[Hit] = field(default_factory=list)
    misses: list[tuple[str, str, str, str]] = field(default_factory=list)  # file, row id, engine, line
    files_changed: list[str] = field(default_factory=list)
    texts_scanned: int = 0
    urls_scanned: int = 0


# --- keep list ---------------------------------------------------------------------------------------------

def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def _host(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url if "://" in url else "https://" + url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return host or None


def build_keep_list(root: Path) -> list[str]:
    """Every company name, alias (surface form), slug and domain in the shipped
    tables, the engine names, and the vendors named in the prompts."""
    root = Path(root)
    terms: set[str] = set(ENGINE_NAMES)
    registry_names: set[str] = set()
    for row in _csv_rows(root / "data/coding/company_table.csv"):
        for name in [row.get("canonical_name") or "", row.get("entity_id") or ""] + (row.get("surface_forms") or "").split(";"):
            if name.strip():
                registry_names.add(name.strip())
        for url_key in ("homepage_url", "evidence_url"):
            if _host(row.get(url_key)):
                terms.add(_host(row.get(url_key)))
    for row in _csv_rows(root / "data/coding/company_alt_domains.csv"):
        if row.get("alt_domain"):
            terms.add(row["alt_domain"].strip().lower())
    for row in _csv_rows(root / "data/coding/company_verification.csv"):
        if _host(row.get("verified_homepage_url")):
            terms.add(_host(row.get("verified_homepage_url")))
    for row in _csv_rows(root / "data/aux/roster_provisional.csv"):
        if (row.get("canonical_name") or "").strip():
            registry_names.add(row["canonical_name"].strip())
        for d in [row.get("primary_domain") or ""] + (row.get("alt_domains") or "").split(";"):
            if d.strip():
                terms.add(d.strip().lower())
    terms |= registry_names
    for row in _csv_rows(root / "config/prompts/prompts.csv"):
        if row.get("branded") == "1":
            text = row.get("prompt_text") or ""
            terms |= {n for n in registry_names if re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", text)}
    return sorted({t for t in terms if len(t) >= MIN_KEEP_TERM}, key=lambda t: (t.lower(), t))


def write_keep_list(root: Path, terms: list[str]) -> None:
    header = [
        "# Keep list for cleaning/clean_personal_data.py: a detection that overlaps one of these terms is never replaced.",
        "# Generated from the shipped tables (company names, aliases, slugs, domains), the engine names and the vendors",
        "# named in the prompts. Regenerated on every run; edit remove_list.txt instead.",
    ]
    (root / KEEP_LIST).write_text("\n".join(header + terms) + "\n", encoding="utf-8")


def read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def term_regex(terms: Iterable[str]) -> re.Pattern[str] | None:
    parts = sorted({t for t in terms if t}, key=lambda t: (-len(t), t))
    if not parts:
        return None
    body = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in parts)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE)


def alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# --- the cleaner -------------------------------------------------------------------------------------------

class Cleaner:
    def __init__(self, keep_terms: list[str], remove_terms: list[str], private_dir: Path):
        self.keep_re = term_regex(keep_terms)
        self.remove_re = term_regex(remove_terms)
        self.keep_domains = {t.lower() for t in keep_terms if re.fullmatch(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)}
        self.keep_handles = {alnum(t) for t in keep_terms if len(alnum(t)) >= MIN_KEEP_TERM}
        self.keep_handles |= {alnum(d.split(".")[0]) for d in self.keep_domains if len(alnum(d.split(".")[0])) >= MIN_KEEP_TERM}
        self.profile_numbers: dict[str, int] = {}
        self.result = Result()
        self._analysis_cache: dict[str, list[tuple[str, int, int, float]]] = {}
        self._analyzer = None
        self._anonymizer = None
        self.private_dir = private_dir

    # -- engines (loaded once) ------------------------------------------------------------------------------
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer, self._anonymizer = build_engines()
        return self._analyzer

    def analyze(self, text: str) -> list[tuple[str, int, int, float]]:
        """(type, start, end, score) for every detection in text, cached by text."""
        if text not in self._analysis_cache:
            results = self.analyzer().analyze(
                text=text, language="en", entities=list(PRESIDIO_ENTITY), score_threshold=0.0
            )
            found = sorted(
                {(PRESIDIO_ENTITY[r.entity_type], r.start, r.end, person_score(text[r.start:r.end], float(r.score))
                  if r.entity_type == "PERSON" else round(float(r.score), 4)) for r in results},
                key=lambda d: (d[1], d[2], d[0], -d[3]),
            )
            self._analysis_cache[text] = found
        return self._analysis_cache[text]

    # -- profile URLs -----------------------------------------------------------------------------------------
    def profile_replacements(self, text: str) -> list[tuple[int, int, str, str, float, str]]:
        """(start, end, type, replacement, score, kept_reason) for profile URLs in text."""
        out = []
        for m in PROFILE_URL.finditer(text):
            handle = profile_handle(m.group("host"), m.group("path"))
            if handle is None:
                continue
            original = m.group(0)
            if alnum(handle) in self.keep_handles:
                out.append((m.start(), m.end(), PROFILE, original, 1.0, REASON_COMPANY_URL))
                continue
            key = normalize_page(original)
            if key not in self.profile_numbers:
                self.profile_numbers[key] = len(self.profile_numbers) + 1
            replacement = f"{m.group('scheme') or ''}{m.group('host')}/[PROFILE-{self.profile_numbers[key]}]"
            out.append((m.start(), m.end(), PROFILE, replacement, 1.0, ""))
        return out

    # -- one string -------------------------------------------------------------------------------------------
    def clean(self, text: str, *, file: str, row_id: str, engine: str = "", mode: str = "text") -> str:
        """mode: "text" (all types), "url" (profile URLs only), "comment" (email and phone only)."""
        if not isinstance(text, str) or not text:
            return text
        candidates: list[tuple[int, int, str, str, float, str]] = []  # start, end, type, replacement, score, kept reason
        candidates += self.profile_replacements(text)
        if mode == "url":
            self.result.urls_scanned += 1
        else:
            self.result.texts_scanned += 1
            url_spans = [(m.start(), m.end()) for m in re.finditer(r"https?://[^\s<>\"'()\[\]]+", text)]
            keep_spans = [(m.start(), m.end()) for m in self.keep_re.finditer(text)] if self.keep_re else []
            for kind, start, end, score in self.analyze(text):
                if mode == "comment" and kind not in (EMAIL, PHONE):
                    continue
                if kind != EMAIL and any(s < end and start < e for s, e in url_spans):
                    continue  # inside a URL: the URL rule applies there
                original = text[start:end]
                if kind == EMAIL:
                    domain = original.rsplit("@", 1)[-1].lower().rstrip(".")
                    keep_domain = any(domain == d or domain.endswith("." + d) for d in self.keep_domains)
                    replacement = TOKEN[EMAIL] + "@" + original.rsplit("@", 1)[-1] if keep_domain else TOKEN[EMAIL]
                    reason = "" if score >= REMOVE_AT_OR_ABOVE else REASON_LOW
                elif any(s < end and start < e for s, e in keep_spans):
                    replacement, reason = TOKEN[kind], REASON_KEEP
                else:
                    replacement, reason = TOKEN[kind], ("" if score >= REMOVE_AT_OR_ABOVE else REASON_LOW)
                candidates.append((start, end, kind, replacement, score, reason))
            if self.remove_re is not None:
                for m in self.remove_re.finditer(text):
                    candidates.append((m.start(), m.end(), REMOVED, TOKEN[REMOVED], 1.0, ""))
        if not candidates:
            if mode == "text":
                self.scan_misses(text, file=file, row_id=row_id, engine=engine)
            return text
        removals = [c for c in candidates if not c[5]]
        removals.sort(key=lambda c: (c[0], -(c[1] - c[0]), -c[4], c[2]))
        chosen: list[tuple[int, int, str, str, float, str]] = []
        for c in removals:
            if chosen and c[0] < chosen[-1][1]:
                continue  # overlaps an earlier, longer or stronger replacement
            chosen.append(c)
        for c in sorted(candidates, key=lambda c: (c[0], c[1], c[2])):
            hit = Hit(file, row_id, engine, c[2], text[c[0]:c[1]], c[3], text[max(0, c[0] - CONTEXT_CHARS):c[0]],
                      text[c[1]:c[1] + CONTEXT_CHARS], c[4], c[5])
            if c[5]:
                self.result.kept.append(hit)
            elif c in chosen:
                self.result.removed.append(hit)
        cleaned = apply_replacements(text, chosen, self._anonymizer_engine()) if chosen else text
        if mode == "text":
            self.scan_misses(cleaned, file=file, row_id=row_id, engine=engine)
        return cleaned

    def scan_misses(self, text: str, *, file: str, row_id: str, engine: str) -> None:
        for line in text.splitlines():
            if MISS_TRIGGERS.search(line) and not TOKEN_ON_LINE.search(line):
                self.result.misses.append((file, row_id, engine, line.strip()[:300]))

    def _anonymizer_engine(self):
        if self._anonymizer is None:
            self._analyzer, self._anonymizer = build_engines()
        return self._anonymizer


INSTALL_HINT = "the cleaning dependencies are an optional group: run `uv sync --group cleaning` (or `uv run --group cleaning ...`)"


def check_dependencies() -> None:
    """Stop with the install command when the optional cleaning group is missing."""
    import importlib.util

    missing = [m for m in ("presidio_analyzer", "presidio_anonymizer", "spacy", "phonenumbers", SPACY_MODEL)
               if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"STOP: missing {', '.join(missing)}; {INSTALL_HINT}")


def build_engines():
    """Presidio analyzer (spaCy en_core_web_lg for PERSON, a pattern recognizer
    for email, phonenumbers for US phone numbers, a pattern recognizer for US
    street addresses) and the presidio anonymizer."""
    import phonenumbers
    from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer, RecognizerRegistry, RecognizerResult
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import SpacyRecognizer
    from presidio_anonymizer import AnonymizerEngine

    class UsPhoneRecognizer(EntityRecognizer):
        """phonenumbers' matcher, region US. A valid number scores 0.75; a
        possible but not valid one 0.4 (below the threshold, so kept and logged)."""

        def __init__(self):
            super().__init__(supported_entities=["PHONE_NUMBER"], name="UsPhoneRecognizer", supported_language="en")

        def load(self) -> None:
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            out = []
            for match in phonenumbers.PhoneNumberMatcher(text, "US", leniency=phonenumbers.Leniency.POSSIBLE):
                score = 0.75 if phonenumbers.is_valid_number(match.number) else 0.4
                out.append(RecognizerResult("PHONE_NUMBER", match.start, match.end, score))
            return out

    suffix = (r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Parkway|Pkwy|"
              r"Highway|Hwy|Circle|Cir|Terrace|Ter|Square|Sq|Trail|Trl|Plaza|Plz)\.?")
    street = rf"\b\d{{1,6}}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][A-Za-z0-9'.-]*\s+){{1,4}}{suffix}"
    unit = r"(?:,?\s+(?:Suite|Ste\.?|Unit|Apt\.?|#)\s*[A-Za-z0-9-]+)?"
    city_state_zip = r",?\s+[A-Z][A-Za-z.\s]{1,30},\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?"
    address = PatternRecognizer(
        supported_entity="US_STREET_ADDRESS",
        name="UsStreetAddressRecognizer",
        patterns=[
            Pattern("street, city, state and ZIP", street + unit + city_state_zip, 0.85),
            Pattern("street only", street + unit + r"\b", 0.6),
        ],
    )
    email = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        name="EmailPatternRecognizer",
        patterns=[Pattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 1.0)],
    )
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
        "ner_model_configuration": {
            "labels_to_ignore": ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP",
                                 "ORDINAL", "ORG", "PERCENT", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"],
        },
    })
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.add_recognizer(SpacyRecognizer(supported_language="en", supported_entities=["PERSON"]))
    registry.add_recognizer(email)
    registry.add_recognizer(UsPhoneRecognizer())
    registry.add_recognizer(address)
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])
    for model in getattr(nlp_engine, "nlp", {}).values():
        model.max_length = max(model.max_length, 5_000_000)
    return analyzer, AnonymizerEngine()


def apply_replacements(text: str, chosen: list[tuple[int, int, str, str, float, str]], anonymizer) -> str:
    """Replace each chosen span with its own replacement through the presidio
    anonymizer (one operator per span, so numbered and partial replacements work)."""
    from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

    results = [RecognizerResult(f"R{i}", c[0], c[1], c[4]) for i, c in enumerate(chosen)]
    operators = {f"R{i}": OperatorConfig("replace", {"new_value": c[3]}) for i, c in enumerate(chosen)}
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


# --- files -------------------------------------------------------------------------------------------------

def _walk(node: Any, fn: Callable[[str, str | None], str], key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, fn, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, fn, key) for v in node]
    if isinstance(node, str):
        return fn(node, key)
    return node


def clean_raw(cleaner: Cleaner, root: Path) -> None:
    for path in sorted((root / "data" / "raw").glob("*.jsonl")):
        rel = path.relative_to(root).as_posix()
        out_lines, changed = [], False
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.strip():
                out_lines.append(line)
                continue
            record = json.loads(line)
            engine, run_id = record.get("engine", ""), record.get("run_id", "")
            text_row = engine in TEXT_ENGINES

            def fn(value: str, key: str | None) -> str:
                mode = "text" if text_row and key in RAW_TEXT_KEYS else "url"
                return cleaner.clean(value, file=rel, row_id=run_id, engine=engine, mode=mode)

            new = {k: (_walk(v, fn, k) if k == "response" else v) for k, v in record.items()}
            if new != record:
                changed = True
                out_lines.append(json.dumps(new) + "\n")
            else:
                out_lines.append(line)
        if changed:
            path.write_text("".join(out_lines), encoding="utf-8")
            cleaner.result.files_changed.append(rel)


def clean_jsonl(cleaner: Cleaner, root: Path, rel: str, *, text_keys: set[str], row_id: Callable[[dict, int], str],
                engine_of: Callable[[dict], str], dump: Callable[[dict], str], text_engines_only: bool) -> None:
    path = root / rel
    if not path.exists():
        return
    out_lines, changed = [], False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), start=1):
        if not line.strip():
            out_lines.append(line)
            continue
        row = json.loads(line)
        engine = engine_of(row)
        rid = row_id(row, n)

        def fn(value: str, key: str | None) -> str:
            text_ok = key in text_keys and (not text_engines_only or engine in TEXT_ENGINES)
            return cleaner.clean(value, file=rel, row_id=rid, engine=engine, mode="text" if text_ok else "url")

        new = _walk(row, fn)
        if new != row:
            changed = True
            out_lines.append(dump(new) + "\n")
        else:
            out_lines.append(line)
    if changed:
        path.write_text("".join(out_lines), encoding="utf-8")
        cleaner.result.files_changed.append(rel)


def clean_csv(cleaner: Cleaner, root: Path, rel: str, *, text_cols: set[str], url_cols: set[str], id_col: str) -> None:
    path = root / rel
    if not path.exists():
        return
    raw = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines(keepends=True)
    comments = [l for l in lines if l.startswith("#")]
    body = "".join(l for l in lines if not l.startswith("#"))
    reader = csv.DictReader(io.StringIO(body, newline=""))
    fields = list(reader.fieldnames or [])
    rows = list(reader)
    changed = False
    for n, row in enumerate(rows, start=1):
        rid = row.get(id_col) or f"row {n}"
        for col in fields:
            value = row.get(col)
            if not value:
                continue
            mode = "text" if col in text_cols else "url" if col in url_cols else None
            if mode is None:
                continue
            new = cleaner.clean(value, file=rel, row_id=rid, mode=mode)
            if new != value:
                row[col] = new
                changed = True
    if changed:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)
        path.write_text("".join(comments) + buf.getvalue(), encoding="utf-8")
        cleaner.result.files_changed.append(rel)


def clean_eyeball(cleaner: Cleaner, root: Path) -> None:
    rel = "data/ui/eyeball_check.json"
    path = root / rel
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(data)
    out["runs"] = {
        run_key: _walk(entry, lambda v, k, rk=run_key: cleaner.clean(v, file=rel, row_id=rk, mode="text"))
        for run_key, entry in data.get("runs", {}).items()
    }
    if out != data:
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        cleaner.result.files_changed.append(rel)


def clean_robots(cleaner: Cleaner, root: Path) -> None:
    """robots.txt: email and phone in comments only (the rules themselves are what the crawler pre-check reads)."""
    for path in sorted((root / "data" / "aux" / "robots").glob("*/robots.txt")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        out = []
        for line in text.splitlines(keepends=True):
            rule, sep, comment = line.partition("#")
            if sep and comment.strip():
                comment = cleaner.clean(comment, file=rel, row_id=path.parent.name, mode="comment")
            out.append(rule + sep + comment)
        new = "".join(out)
        if new != text:
            path.write_text(new, encoding="utf-8")
            cleaner.result.files_changed.append(rel)


def _mention_engine(row: dict) -> str:
    return str(row.get("run_id") or "").split("-", 1)[0]


def clean_package(root: Path, private_dir: Path, log=print) -> Result:
    root = Path(root).resolve()
    private_dir = Path(private_dir).expanduser().resolve()
    if private_dir == root or root in private_dir.parents:
        raise SystemExit(f"STOP: the private dir {private_dir} is inside the package")
    check_dependencies()
    keep_terms = build_keep_list(root)
    write_keep_list(root, keep_terms)
    if not (root / REMOVE_LIST).exists():
        (root / REMOVE_LIST).write_text(
            "# Manual additions: one term per line. Every occurrence in a cleaned text field is replaced with [REMOVED],\n"
            "# even when it overlaps a keep-list term. Lines starting with # are ignored.\n",
            encoding="utf-8",
        )
    remove_terms = read_list(root / REMOVE_LIST)
    cleaner = Cleaner(keep_terms, remove_terms, private_dir)
    log(f"keep list: {len(keep_terms)} terms; remove list: {len(remove_terms)} terms")

    clean_raw(cleaner, root)
    log(f"raw window cleaned: {len(cleaner.result.removed)} replacements so far")
    clean_jsonl(cleaner, root, "data/coding/mentions_coded.jsonl", text_keys={"evidence_span"},
                row_id=lambda r, n: f"{r.get('run_id')} line {n}", engine_of=_mention_engine,
                dump=lambda r: json.dumps(r, sort_keys=True), text_engines_only=True)
    clean_jsonl(cleaner, root, "data/ui/ui_runs.jsonl", text_keys={"text"}, row_id=lambda r, n: str(r.get("run_key")),
                engine_of=lambda r: str(r.get("engine") or ""), dump=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True),
                text_engines_only=True)
    clean_jsonl(cleaner, root, "data/aux/cited_urls/resolution.jsonl", text_keys={"page_title"},
                row_id=lambda r, n: f"line {n}", engine_of=lambda r: "", dump=json.dumps, text_engines_only=False)
    clean_csv(cleaner, root, "data/coding/company_table.csv", text_cols={"frame_reason"},
              url_cols={"homepage_url", "evidence_url"}, id_col="entity_id")
    clean_csv(cleaner, root, "data/coding/company_verification.csv", text_cols={"disambiguation_note"},
              url_cols={"verified_homepage_url"}, id_col="entity_id")
    clean_csv(cleaner, root, "data/coding/company_alt_domains.csv", text_cols={"source_note"}, url_cols=set(), id_col="entity_id")
    clean_csv(cleaner, root, "data/coding/url_classes.csv", text_cols=set(), url_cols={"url_normalized"}, id_col="url_normalized")
    clean_csv(cleaner, root, "data/aux/roster_provisional.csv", text_cols={"notes"}, url_cols=set(), id_col="entity_id")
    clean_csv(cleaner, root, "data/aux/google_serp_top10.csv", text_cols=set(), url_cols={"url"}, id_col="question_id")
    clean_eyeball(cleaner, root)
    clean_robots(cleaner, root)
    write_outputs(root, private_dir, cleaner)
    log(f"cleaning: {len(cleaner.result.removed)} replaced, {len(cleaner.result.kept)} kept, "
        f"{len(cleaner.result.misses)} possible misses, {len(cleaner.result.files_changed)} files changed")
    return cleaner.result


# --- outputs -----------------------------------------------------------------------------------------------

def _hit_row(h: Hit, with_reason: bool) -> list[Any]:
    row = [h.file, h.row_id, h.engine, h.type, h.original, h.replacement, h.before, h.after, f"{h.score:.2f}"]
    return row + [h.reason] if with_reason else row


def _write_private_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
    path.chmod(0o600)


def _counts_table(result: Result) -> list[str]:
    engines = sorted({h.engine or "(no engine)" for h in result.removed + result.kept})
    removed = Counter((h.type, h.engine or "(no engine)") for h in result.removed)
    kept = Counter((h.type, h.engine or "(no engine)") for h in result.kept)
    lines = ["| Type | Engine | Removed | Kept |", "|---|---|---|---|"]
    for t in TYPES:
        for e in engines:
            if removed[(t, e)] or kept[(t, e)]:
                lines.append(f"| {t} | {e} | {removed[(t, e)]:,} | {kept[(t, e)]:,} |")
    lines.append(f"| all | all | {len(result.removed):,} | {len(result.kept):,} |")
    return lines


CANNOT_CATCH = [
    "Names spaCy does not tag as a person (unusual names, names in lower case, names written as initials), one-word "
    "names (scored below the threshold on purpose, see Rules), and names inside URLs that are not a LinkedIn /in/, X, "
    "Twitter, Facebook, Instagram, TikTok or YouTube profile.",
    "A person whose name is also a company name, alias or domain on the keep list: anything that overlaps a keep-list "
    "term is kept by design.",
    "Personal details that are not one of the five types: job titles, employers, towns, ages, photos, or a person "
    "described without a name.",
    "Phone numbers outside the US numbering plan that phonenumbers does not read as a US number, and street "
    "addresses that do not follow the US number + street + suffix pattern.",
    "Profile pages on sites other than the ones listed, and URLs that identify a person in another way (a blog path, "
    "a query string). The url_key hash in the resolution file is left as it is.",
    "Text in fields that are not scanned as text: HTTP headers in the resolution file, robots.txt rules (only their "
    "comments are scanned), the documents under docs/ and the code.",
    "Gemini and Google AI Mode answer text is not in the package at all, so it needs no cleaning.",
]


def write_outputs(root: Path, private_dir: Path, cleaner: Cleaner) -> None:
    result = cleaner.result
    private_dir.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)
    _write_private_csv(private_dir / "cleaning_log.csv", LOG_COLUMNS, [_hit_row(h, False) for h in result.removed])
    _write_private_csv(private_dir / "kept_hits.csv", KEPT_COLUMNS, [_hit_row(h, True) for h in result.kept])
    kept_high = [h for h in result.kept if h.score >= REPORT_KEPT_AT_OR_ABOVE]

    def report(private: bool) -> str:
        lines = [
            "# Personal-data cleaning report" + (" (PRIVATE copy, holds original values)" if private else ""),
            "",
            "Written by `cleaning/clean_personal_data.py`. Same input gives the same output. "
            f"Replacement threshold: score at least {REMOVE_AT_OR_ABOVE}. Text fields scanned: {result.texts_scanned:,}; "
            f"URL fields scanned: {result.urls_scanned:,}; files changed: {len(result.files_changed):,}; distinct "
            f"profile pages replaced: {len(cleaner.profile_numbers):,}.",
            "",
            "To rerun: `uv sync --group cleaning`, then `uv run --group cleaning python cleaning/clean_personal_data.py "
            "--private-dir <a folder outside the package>`. "
            "It regenerates `cleaning/keep_list.txt` and honours `cleaning/remove_list.txt`.",
            "",
            "## Rules",
            "",
            "- E-mail address: `[EMAIL]`; at a keep-list domain only the part before @ is replaced, so the company domain stays.",
            "- Phone number (phonenumbers, region US): `[PHONE]`; a possible but not valid number scores below the threshold.",
            "- US street address (number, street, suffix, optional unit, city, state and ZIP): `[ADDRESS]`.",
            f"- Person (spaCy PERSON through presidio): `[PERSON]` for a name of two or more words with a capital; a one-word "
            f"hit scores {SINGLE_WORD_PERSON_SCORE} (below the threshold), because a lone first or last name rarely "
            "identifies anyone and most one-word hits in this data are product names and headings.",
            "- Personal profile URL (LinkedIn /in/, X, Twitter, Facebook, Instagram, TikTok @, YouTube @): scheme and host "
            "stay, the rest becomes `/[PROFILE-n]`, one number per distinct page in order of first appearance. A profile "
            "whose handle is a keep-list name or domain label is a company page and stays, as do linkedin.com/company/ pages.",
            "- A term in `remove_list.txt`: `[REMOVED]`, always.",
            "- Anything overlapping a keep-list term (company names, aliases, slugs and domains, engine names, vendors in "
            "the prompts) is kept, except e-mail addresses (see above) and remove-list terms.",
            "",
            "## Removed and kept, per type and engine",
            "",
            *_counts_table(result),
            "",
            "## Most frequent removed values",
            "",
        ]
        if private:
            for t in TYPES:
                top = Counter(h.original for h in result.removed if h.type == t).most_common(TOP_N)
                if not top:
                    continue
                lines += [f"### {t}", ""] + [f"- {n:,} x `{v}`" for v, n in top] + [""]
        else:
            lines += [f"The {TOP_N} most frequent removed values per type are listed in the private copy of this report "
                      "and in the private cleaning log (kept by the author, outside the package). They are not shipped.", ""]
        lines += [f"## Kept hits with score at least {REPORT_KEPT_AT_OR_ABOVE} ({len(kept_high):,})", ""]
        by_reason = Counter(h.reason for h in kept_high)
        lines += [f"- {reason}: {n:,}" for reason, n in sorted(by_reason.items())] + [""]
        if private:
            lines += ["| Type | Score | Reason | File | Row | Value |", "|---|---|---|---|---|---|"]
            lines += [f"| {h.type} | {h.score:.2f} | {h.reason} | {h.file} | {h.row_id} | `{h.original}` |" for h in kept_high]
        else:
            lines += ["| Type | Score | Reason | File | Row |", "|---|---|---|---|---|"]
            lines += [f"| {h.type} | {h.score:.2f} | {h.reason} | {h.file} | {h.row_id} |" for h in kept_high]
        lines += ["", "", "## Possible misses", "",
                  "Lines of cleaned text that mention a founder, CEO, owner, Dr., attorney, call us, email us, contact, an @ "
                  f"or a run of seven or more digits, and carry no replacement token: {len(result.misses):,}."]
        if private:
            lines += [""] + [f"- {f} | {r} | {e} | {line}" for f, r, e, line in result.misses]
        else:
            lines += ["The lines themselves are in the private copy of this report."]
        lines += ["", "## What this cleaning cannot catch", ""] + [f"- {item}" for item in CANNOT_CATCH]
        return "\n".join(lines) + "\n"

    (root / REPORT).write_text(report(False), encoding="utf-8")
    private_report = private_dir / "CLEANING_REPORT_private.md"
    private_report.write_text(report(True), encoding="utf-8")
    private_report.chmod(0o600)


def tree_digest(root: Path) -> str:
    """sha256 over every file under root (path and bytes), in sorted order."""
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Personal-data cleaning of the publish package")
    parser.add_argument("--root", default=str(PACKAGE_ROOT), help="package root (default: the folder holding cleaning/)")
    parser.add_argument("--private-dir", default=None,
                        help="required: folder outside the package for the log of removed values")
    args = parser.parse_args(argv)
    if not args.private_dir:
        print("STOP: --private-dir is required. Pass a folder outside the package; the log of removed "
              "values goes there and must never be shared.", file=sys.stderr)
        return 2
    clean_package(Path(args.root), Path(args.private_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
