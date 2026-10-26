"""Maps an experiment's declared engine name (experiment_config.yaml) to its
adapter callable. Adding an engine means adding one entry here plus one
adapter module (INSTRUMENT.md section 12) -- nothing else changes.
"""
from __future__ import annotations

from collector.engines.anthropic_claude import call_claude
from collector.engines.google_ai_mode import call_google_ai_mode
from collector.engines.google_gemini import call_gemini
from collector.engines.openai_chatgpt import call_chatgpt

ADAPTERS = {
    "openai_chatgpt": call_chatgpt,
    "google_gemini": call_gemini,
    "anthropic_claude": call_claude,
    "google_ai_mode": call_google_ai_mode,
}
