"""The OpenRouter client must be created lazily, not at import time."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# lib.constants reads data/stopwords.txt at import time; data/ is not committed.
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "stopwords.txt").exists(),
    reason="requires data/stopwords.txt",
)


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm_client = importlib.import_module("lib.llm_client")
    # Stop get_client() from reading a developer's local .env file.
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: None)
    llm_client.get_client.cache_clear()
    yield llm_client
    llm_client.get_client.cache_clear()


@pytest.mark.parametrize(
    "module", ["lib.query_enhancement", "lib.augmented_generation", "lib.reranking"]
)
def test_llm_modules_import_without_api_key(no_api_key, module):
    importlib.reload(importlib.import_module(module))


def test_get_client_raises_only_when_called(no_api_key):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        no_api_key.get_client()


def test_get_client_is_cached(no_api_key, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert no_api_key.get_client() is no_api_key.get_client()
