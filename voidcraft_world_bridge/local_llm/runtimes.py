"""Which model server the plugin talks to — found, not configured.

Someone who already runs Ollama or LM Studio should not have to tell the bridge
so. With nothing set, the plugin asks the well-known local ports in order and
uses the first that answers like a model server:

    Ollama      http://127.0.0.1:11434        (native API)
    LM Studio   http://127.0.0.1:1234/v1      (OpenAI-compatible)
    llama.cpp   http://127.0.0.1:8080/v1      (OpenAI-compatible, llama-server)

Ollama is asked first because it is the most common, and its native API sets
the context size per request, which the OpenAI shape cannot.

Setting a URL turns probing off — that one server, and only it, is asked:

    VOIDCRAFT_LOCAL_LLM_URL   any OpenAI-compatible server (vLLM, Jan, a custom port)
    VOIDCRAFT_OLLAMA_URL      an Ollama on a non-default port

The probe only ever reaches loopback. A URL that points elsewhere is the user's
explicit choice, made in their own environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from voidcraft_world_bridge.local_llm import ollama, openai_compat
from voidcraft_world_bridge.local_llm.transport import Opener


@dataclass(frozen=True)
class Runtime:
    """One model server: which adapter speaks to it, where, and what to call it."""

    adapter: ModuleType
    base: str
    label: str

    @property
    def kind(self) -> str:
        return self.adapter.KIND

    def list_models(self, opener: Opener | None = None) -> list[dict]:
        return self.adapter.list_models(self.base, opener)

    def version(self, opener: Opener | None = None) -> str | None:
        return self.adapter.version(self.base, opener)

    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        return self.adapter.chat(self.base, model, messages, **kwargs)

    def missing_model_hint(self, model: str) -> str:
        return self.adapter.missing_model_hint(model)

    @property
    def no_model_hint(self) -> str:
        return self.adapter.NO_MODEL_HINT

    def describe(self) -> str:
        return f"{self.label} at {self.base}"


PROBED = (
    Runtime(ollama, ollama.DEFAULT_URL, "Ollama"),
    Runtime(openai_compat, "http://127.0.0.1:1234/v1", "LM Studio"),
    Runtime(openai_compat, "http://127.0.0.1:8080/v1", "llama.cpp"),
)


def candidates() -> tuple[Runtime, ...]:
    """The servers to ask, in order. One entry when a URL is configured."""
    explicit = openai_compat.configured_url()
    if explicit:
        return (Runtime(openai_compat, explicit, openai_compat.LABEL),)
    if ollama.base_url() != ollama.DEFAULT_URL:
        return (Runtime(ollama, ollama.base_url(), ollama.LABEL),)
    return PROBED
