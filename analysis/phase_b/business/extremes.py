"""Lowest and highest value with the engine that holds it, shared by the S1 and A1/V1 business figures.

Ties go to the engine that comes first in the fixed study order (the same order as the charts, see
scripts/results/user_charts/style.py; a test keeps the two equal). Engines outside that order sort after it,
by name. Values that are None (a ratio with a zero denominator) are skipped."""
from __future__ import annotations

from typing import Any

ENGINE_ORDER = ("openai_chatgpt", "google_ai_mode", "google_gemini", "anthropic_claude")


def ordered(engines) -> list[str]:
    """Engines in the fixed study order."""
    return sorted(engines, key=lambda e: (ENGINE_ORDER.index(e) if e in ENGINE_ORDER else len(ENGINE_ORDER), e))


def extreme(values: dict[str, Any]) -> dict[str, Any]:
    keys = [k for k in ordered(values) if values[k] is not None]
    if not keys:
        return {"min": None, "max": None}
    low = min(keys, key=values.get)
    high = max(keys, key=values.get)
    return {"min": {"engine": low, "value": values[low]}, "max": {"engine": high, "value": values[high]}}
