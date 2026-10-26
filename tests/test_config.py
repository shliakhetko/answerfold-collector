import pytest

from collector.config import (
    ENV_ANTHROPIC_API_KEY,
    ENV_GEMINI_API_KEY,
    ENV_OPENAI_API_KEY,
    ENV_SERPAPI_API_KEY,
    AnthropicConfig,
    ConfigError,
    GeminiConfig,
    OpenAIConfig,
    SerpApiConfig,
)


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv(ENV_OPENAI_API_KEY, raising=False)
    with pytest.raises(ConfigError, match=ENV_OPENAI_API_KEY):
        OpenAIConfig.from_env()


def test_from_env_reads_tunables(monkeypatch):
    monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-test")
    monkeypatch.setenv("TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "256")

    config = OpenAIConfig.from_env()

    assert config.api_key == "sk-test"
    assert config.timeout_seconds == 12.5
    assert config.max_retries == 5
    assert config.max_output_tokens == 256


def test_from_env_defaults_when_tunables_unset(monkeypatch):
    monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-test")
    monkeypatch.delenv("TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MAX_RETRIES", raising=False)
    monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)

    config = OpenAIConfig.from_env()

    assert config.timeout_seconds == 60.0
    assert config.max_retries == 2
    assert config.max_output_tokens is None


def test_api_key_not_in_repr():
    config = OpenAIConfig(api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(config)


def test_shared_tunables_apply_across_vendors(monkeypatch):
    """TIMEOUT_SECONDS and MAX_RETRIES are read by OpenAI, Gemini, and
    Anthropic alike -- one env var, not three per-vendor copies."""
    monkeypatch.setenv(ENV_OPENAI_API_KEY, "sk-test")
    monkeypatch.setenv(ENV_GEMINI_API_KEY, "gm-test")
    monkeypatch.setenv(ENV_ANTHROPIC_API_KEY, "sk-ant-test")
    monkeypatch.setenv("TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("MAX_RETRIES", "5")

    for config in (OpenAIConfig.from_env(), GeminiConfig.from_env(), AnthropicConfig.from_env()):
        assert config.timeout_seconds == 12.5
        assert config.max_retries == 5


def test_serpapi_config_uses_shared_timeout_only(monkeypatch):
    """SerpApiConfig has no max_retries field -- MAX_RETRIES doesn't apply
    to it, only the shared TIMEOUT_SECONDS does."""
    monkeypatch.setenv(ENV_SERPAPI_API_KEY, "serp-test")
    monkeypatch.setenv("TIMEOUT_SECONDS", "12.5")

    config = SerpApiConfig.from_env()

    assert config.timeout_seconds == 12.5
    assert not hasattr(config, "max_retries")


def test_anthropic_max_tokens_is_independent_of_shared_max_output_tokens(monkeypatch):
    """ANTHROPIC_MAX_TOKENS is a separate, required-with-a-real-default env
    var -- it must not be affected by the shared MAX_OUTPUT_TOKENS, which is
    an optional cap that doesn't apply to Anthropic."""
    monkeypatch.setenv(ENV_ANTHROPIC_API_KEY, "sk-ant-test")
    monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)

    config = AnthropicConfig.from_env()

    assert config.max_tokens == 8192
