"""The default-model rule: strongest installed chat model, by the preference order, never an embedder."""
from __future__ import annotations

from voidcraft_world_bridge.local_llm.models import is_chat_model, matches, pick_model, resolve_requested


def m(name: str, family: str = "qwen35") -> dict:
    return {"name": name, "details": {"family": family}}


def test_tagged_preference_matches_its_quant_variants_only():
    assert matches("qwen3.6:35b-a3b", "qwen3.6:35b-a3b")
    assert matches("qwen3.6:35b-a3b", "qwen3.6:35b-a3b-q8_0")
    assert not matches("qwen3.6:35b-a3b", "qwen3.6:35b")
    assert not matches("qwen3.6:35b-a3b", "qwen3.6:27b")


def test_untagged_preference_matches_any_tag_of_that_model():
    assert matches("qwen3.8", "qwen3.8:latest")
    assert matches("qwen3.8", "qwen3.8:27b")
    assert not matches("qwen3.8", "qwen3.80:27b")


def test_picks_by_order_not_by_size():
    installed = [m("qwen3.8:27b"), m("qwen3.6:35b-a3b"), m("qwen3.5:9b")]
    assert pick_model(installed) == "qwen3.6:35b-a3b"


def test_never_falls_back_to_an_embedding_model():
    assert pick_model([m("nomic-embed-text:latest", "nomic-bert")]) is None
    assert not is_chat_model(m("qwen3-embedding:0.6b", "qwen3"))


def test_unknown_chat_model_is_the_fallback():
    assert pick_model([m("mistral-small:24b", "mistral3"), m("bge-m3", "bert")]) == "mistral-small:24b"


def test_pin_wins_only_when_installed():
    installed = [m("qwen3.5:9b"), m("qwen3.8:27b")]
    assert pick_model(installed, pinned="qwen3.5:9b") == "qwen3.5:9b"
    assert pick_model(installed, pinned="llama9:1t") == "qwen3.8:27b"


def test_a_named_model_must_be_installed_the_plugin_never_pulls():
    installed = [m("qwen3.5:9b")]
    assert resolve_requested("qwen3.5:9b", installed, None) == ("qwen3.5:9b", None)
    assert resolve_requested("qwen3.8", installed, None) == (None, "model_not_installed")
    assert resolve_requested(None, [], None) == (None, "no_model")
