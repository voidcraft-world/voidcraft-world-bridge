"""Every test starts from a stock machine: no model-server env vars set.

The developer running these may well have VOIDCRAFT_LOCAL_LLM_URL or a pinned
model exported; without this, probing tests would quietly test their setup.
"""
import pytest

RUNTIME_ENV = ("VOIDCRAFT_LOCAL_LLM_URL", "VOIDCRAFT_OLLAMA_URL", "VOIDCRAFT_LOCAL_LLM_MODEL")


@pytest.fixture(autouse=True)
def _stock_runtime_env(monkeypatch, tmp_path):
    for name in RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    # No other compute hosts unless a test writes some: never the developer's own hosts.json.
    monkeypatch.setenv("VOIDCRAFT_LOCAL_LLM_HOSTS", str(tmp_path / "hosts.json"))
