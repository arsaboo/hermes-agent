"""Tests for Local Ollama provider integration."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import _PROVIDER_LABELS, _PROVIDER_ALIASES, normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider
from agent.model_metadata import _URL_TO_PROVIDER, _PROVIDER_PREFIXES


# ── Provider Registry ──

class TestLocalOllamaProviderRegistry:
    def test_ollama_in_registry(self):
        assert "ollama" in PROVIDER_REGISTRY

    def test_ollama_config(self):
        pconfig = PROVIDER_REGISTRY["ollama"]
        assert pconfig.id == "ollama"
        assert pconfig.name == "Local Ollama"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "http://localhost:11434/v1"

    def test_ollama_env_vars(self):
        pconfig = PROVIDER_REGISTRY["ollama"]
        assert pconfig.api_key_env_vars == ("OLLAMA_LOCAL_API_KEY",)
        assert pconfig.base_url_env_var == "OLLAMA_LOCAL_BASE_URL"

    def test_ollama_base_url(self):
        assert "localhost:11434" in PROVIDER_REGISTRY["ollama"].inference_base_url


# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_KEY",
    "OLLAMA_LOCAL_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestLocalOllamaAliases:
    def test_explicit_ollama(self):
        assert resolve_provider("ollama") == "ollama"

    def test_ollama_local_alias(self):
        """ollama-local alias routes to ollama."""
        assert normalize_provider("ollama-local") == "ollama"

    def test_ollama_cloud_stays_cloud(self):
        """ollama-cloud is distinct from local ollama."""
        assert resolve_provider("ollama-cloud") == "ollama-cloud"


# ── Credential Resolution ──

class TestLocalOllamaCredentials:
    def test_resolve_without_api_key(self):
        """Local Ollama works without any API key configured."""
        creds = resolve_api_key_provider_credentials("ollama")
        assert creds["provider"] == "ollama"
        assert creds["api_key"] == "ollama"  # dummy token
        assert creds["base_url"] == "http://localhost:11434/v1"

    def test_resolve_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_LOCAL_BASE_URL", "http://192.168.1.100:11434/v1")
        creds = resolve_api_key_provider_credentials("ollama")
        assert creds["base_url"] == "http://192.168.1.100:11434/v1"

    def test_resolve_with_explicit_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_LOCAL_API_KEY", "my-custom-key")
        creds = resolve_api_key_provider_credentials("ollama")
        assert creds["api_key"] == "my-custom-key"

    def test_runtime_ollama_local(self):
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="ollama")
        assert result["provider"] == "ollama"
        assert result["api_mode"] == "chat_completions"
        assert result["base_url"] == "http://localhost:11434/v1"


# ── Model Discovery ──

class TestLocalOllamaModelDiscovery:
    def test_no_static_model_list(self):
        """Local Ollama models are discovered dynamically — no static list."""
        from hermes_cli.models import _PROVIDER_MODELS
        assert "ollama" not in _PROVIDER_MODELS

    def test_provider_label(self):
        assert "ollama" in _PROVIDER_LABELS
        assert _PROVIDER_LABELS["ollama"] == "Local Ollama"

    def test_fetch_local_models_from_api(self, tmp_path, monkeypatch):
        """Live API returns models from local Ollama."""
        from hermes_cli.models import fetch_local_ollama_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("hermes_cli.models.fetch_api_models", return_value=["gemma3:4b", "qwen3.5:27b"]):
            result = fetch_local_ollama_models(force_refresh=True)

        assert result == ["gemma3:4b", "qwen3.5:27b"]

    def test_uses_disk_cache(self, tmp_path, monkeypatch):
        """Second call returns cached results without hitting the API."""
        from hermes_cli.models import fetch_local_ollama_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("hermes_cli.models.fetch_api_models", return_value=["model-a"]) as mock_api:
            first = fetch_local_ollama_models(force_refresh=True)
            assert first == ["model-a"]
            assert mock_api.call_count == 1

            second = fetch_local_ollama_models()
            assert second == ["model-a"]
            assert mock_api.call_count == 1  # no extra API call

    def test_force_refresh_bypasses_cache(self, tmp_path, monkeypatch):
        from hermes_cli.models import fetch_local_ollama_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("hermes_cli.models.fetch_api_models", return_value=["model-a"]) as mock_api:
            fetch_local_ollama_models(force_refresh=True)
            fetch_local_ollama_models(force_refresh=True)
            assert mock_api.call_count == 2

    def test_stale_cache_on_failure(self, tmp_path, monkeypatch):
        """If API fails, stale cache is returned."""
        from hermes_cli.models import fetch_local_ollama_models, _save_local_ollama_cache

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Pre-populate a stale cache
        _save_local_ollama_cache(["stale-model"])

        # Make it stale
        cache_path = tmp_path / "ollama_local_models_cache.json"
        with open(cache_path) as f:
            data = json.load(f)
        data["cached_at"] = 0
        with open(cache_path, "w") as f:
            json.dump(data, f)

        with patch("hermes_cli.models.fetch_api_models", return_value=None):
            result = fetch_local_ollama_models(force_refresh=True)

        assert result == ["stale-model"]

    def test_empty_on_total_failure(self, tmp_path, monkeypatch):
        from hermes_cli.models import fetch_local_ollama_models

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("hermes_cli.models.fetch_api_models", return_value=None):
            result = fetch_local_ollama_models(force_refresh=True)

        assert result == []


# ── Model Normalization ──

class TestLocalOllamaModelNormalization:
    def test_passthrough_with_tag(self):
        """Local Ollama is a passthrough provider — model:tag preserved."""
        assert normalize_model_for_provider("gemma3:4b", "ollama") == "gemma3:4b"

    def test_passthrough_bare_name(self):
        assert normalize_model_for_provider("llama3.3", "ollama") == "llama3.3"


# ── URL-to-Provider Mapping ──

class TestLocalOllamaUrlMapping:
    def test_localhost_url_to_provider(self):
        assert _URL_TO_PROVIDER.get("localhost:11434") == "ollama"

    def test_loopback_url_to_provider(self):
        assert _URL_TO_PROVIDER.get("127.0.0.1:11434") == "ollama"

    def test_provider_prefix(self):
        assert "ollama" in _PROVIDER_PREFIXES


# ── providers.py New System ──

class TestLocalOllamaProvidersNew:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "ollama" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["ollama"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_override == "http://localhost:11434/v1"
        assert overlay.base_url_env_var == "OLLAMA_LOCAL_BASE_URL"

    def test_label_override(self):
        from hermes_cli.providers import _LABEL_OVERRIDES
        assert _LABEL_OVERRIDES.get("ollama") == "Local Ollama"

    def test_get_label(self):
        from hermes_cli.providers import get_label
        assert get_label("ollama") == "Local Ollama"

    def test_get_provider(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("ollama")
        assert pdef is not None
        assert pdef.id == "ollama"
        assert pdef.transport == "openai_chat"


# ── Auxiliary Model ──

class TestLocalOllamaAuxiliary:
    def test_aux_model_defined(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "ollama" in _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS["ollama"] == "gemma3:4b"
