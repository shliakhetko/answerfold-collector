"""Environment-derived configuration for engine adapters.

Loads `.env` from the repo root (anchored by path, independent of CWD) via
python-dotenv, then reads named environment variables. Every tunable used by
an adapter has a name, a documented default constant, and exactly one
definition here -- this is what satisfies "no magic numbers": nothing in
collector/engines/*.py is a bare literal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

# override=False: a real process env var always wins over .env (useful for
# CI / ad-hoc overrides without editing the file). Safe to import repeatedly;
# python-dotenv is idempotent.
load_dotenv(dotenv_path=ENV_PATH, override=False)

# --- Named env vars -------------------------------------------------------
# API keys are never shared across vendors -- each is a distinct secret.
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_SERPAPI_API_KEY = "SERPAPI_API_KEY"

# Shared across every adapter that builds an HTTP/SDK client (OpenAI, Gemini,
# Anthropic, SerpApi) -- one client-construction timeout, not four copies of
# the same number. If one vendor ever genuinely needs a different timeout,
# split it back out into its own ENV_*_TIMEOUT_SECONDS at that point; nothing
# here should be pre-split for a need that doesn't exist yet.
ENV_TIMEOUT_SECONDS = "TIMEOUT_SECONDS"

# Shared across the adapters whose SDK client takes a retry count (OpenAI,
# Gemini, Anthropic). Does NOT apply to SerpApi -- SerpApiConfig has no
# max_retries field; the official `serpapi` client's constructor doesn't
# expose one to wire it to (see SerpApiConfig below).
ENV_MAX_RETRIES = "MAX_RETRIES"

# Shared across the adapters where this is an optional response-length cap
# with a real vendor default when unset (OpenAI, Gemini). Does NOT apply to
# Anthropic -- `max_tokens` is a REQUIRED parameter on every Anthropic
# request, not an optional cap, so it keeps its own ANTHROPIC_MAX_TOKENS
# below with a real default rather than "blank = vendor default". Does NOT
# apply to SerpApi -- there is no token concept for that engine at all.
ENV_MAX_OUTPUT_TOKENS = "MAX_OUTPUT_TOKENS"

# Anthropic-only: required on every request (see above), so this cannot be
# folded into MAX_OUTPUT_TOKENS's "blank = vendor default" convention.
ENV_ANTHROPIC_MAX_TOKENS = "ANTHROPIC_MAX_TOKENS"
# Anthropic-only: caps how many searches Claude may run per request -- an
# operational/cost lever with no equivalent concept on any other engine.
# Optional and unset by default (see DEFAULT_ANTHROPIC_WEB_SEARCH_MAX_USES):
# the consumer surface doesn't cap this either, and Anthropic's own guidance
# is that comparative/multi-entity questions -- exactly this study's subject
# -- can legitimately need 10+ searches, so an invented cap would silently
# truncate real answers rather than reflect what a buyer actually sees
# (INSTRUMENT.md section 6).
ENV_ANTHROPIC_WEB_SEARCH_MAX_USES = "ANTHROPIC_WEB_SEARCH_MAX_USES"

# --- Documented defaults ---------------------------------------------------
DEFAULT_TIMEOUT_SECONDS: float = 60.0
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_MAX_OUTPUT_TOKENS: int | None = None  # None = vendor default; no invented cap

# Anthropic's API requires max_tokens on every request -- there is no
# vendor-default/None option like OpenAI/Gemini's max_output_tokens, so this
# must be a real, documented number rather than None. Raised from 4096 to
# 8192: a "best X for Y" prompt listing several vendors can run past 4096
# output tokens and get cut off mid-list (stop_reason: max_tokens), which
# undercounts vendors near the bottom in a way that looks like a real
# finding. See analysis.aggregate.load_runs.is_max_tokens_truncated for the
# downstream flag that excludes any row still truncated despite this cap.
DEFAULT_ANTHROPIC_MAX_TOKENS: int = 8192
# None = no cap sent to the API; Claude decides how many searches to run,
# same as the consumer surface. See ENV_ANTHROPIC_WEB_SEARCH_MAX_USES for why
# this must not default to an invented number.
DEFAULT_ANTHROPIC_WEB_SEARCH_MAX_USES: int | None = None


class ConfigError(RuntimeError):
    """Raised at construction time for missing/invalid setup (e.g. no API
    key, an unparseable env var). Distinct from the per-run adapter contract
    in INSTRUMENT.md section 7.1 ("does not raise"), which covers failures
    during the actual request only -- a missing key means collection cannot
    start at all, which should crash loudly, not produce a DB row."""


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid number") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid integer") from exc


def _get_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid integer") from exc


@dataclass(frozen=True)
class OpenAIConfig:
    # repr=False: the key must never be leaked via an accidental
    # print(config) or a log.debug(config) somewhere downstream.
    api_key: str = field(repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        api_key = os.environ.get(ENV_OPENAI_API_KEY)
        if not api_key:
            raise ConfigError(
                f"{ENV_OPENAI_API_KEY} is not set. Add it to .env (see "
                ".env.example) or export it before running the OpenAI adapter."
            )
        return cls(
            api_key=api_key,
            timeout_seconds=_get_float(ENV_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS),
            max_retries=_get_int(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES),
            max_output_tokens=_get_optional_int(ENV_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS),
        )


@dataclass(frozen=True)
class GeminiConfig:
    # repr=False: the key must never be leaked via an accidental
    # print(config) or a log.debug(config) somewhere downstream.
    api_key: str = field(repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        api_key = os.environ.get(ENV_GEMINI_API_KEY)
        if not api_key:
            raise ConfigError(
                f"{ENV_GEMINI_API_KEY} is not set. Add it to .env (see "
                ".env.example) or export it before running the Gemini adapter."
            )
        return cls(
            api_key=api_key,
            timeout_seconds=_get_float(ENV_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS),
            max_retries=_get_int(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES),
            max_output_tokens=_get_optional_int(ENV_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS),
        )


@dataclass(frozen=True)
class AnthropicConfig:
    # repr=False: the key must never be leaked via an accidental
    # print(config) or a log.debug(config) somewhere downstream.
    api_key: str = field(repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS
    web_search_max_uses: int | None = DEFAULT_ANTHROPIC_WEB_SEARCH_MAX_USES

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        api_key = os.environ.get(ENV_ANTHROPIC_API_KEY)
        if not api_key:
            raise ConfigError(
                f"{ENV_ANTHROPIC_API_KEY} is not set. Add it to .env (see "
                ".env.example) or export it before running the Anthropic adapter."
            )
        return cls(
            api_key=api_key,
            timeout_seconds=_get_float(ENV_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS),
            max_retries=_get_int(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES),
            max_tokens=_get_int(ENV_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MAX_TOKENS),
            web_search_max_uses=_get_optional_int(
                ENV_ANTHROPIC_WEB_SEARCH_MAX_USES, DEFAULT_ANTHROPIC_WEB_SEARCH_MAX_USES
            ),
        )


@dataclass(frozen=True)
class SerpApiConfig:
    # repr=False: the key must never be leaked via an accidental
    # print(config) or a log.debug(config) somewhere downstream.
    api_key: str = field(repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "SerpApiConfig":
        api_key = os.environ.get(ENV_SERPAPI_API_KEY)
        if not api_key:
            raise ConfigError(
                f"{ENV_SERPAPI_API_KEY} is not set. Add it to .env (see "
                ".env.example) or export it before running the SerpApi adapter."
            )
        return cls(
            api_key=api_key,
            timeout_seconds=_get_float(ENV_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS),
        )
