"""Agent runtimes: a model reached some other way than a local server.

A `/chat` body with an `agent: {runtime, …}` field goes to the agent runtime of
that name instead of the local model, on its own queue and worker. The plugin
ships with NONE; a host that embeds the bridge registers its own by passing
`agent_runtimes=[…]` to `LocalLlmPlugin` (or `builtin_plugins`). A body naming a
runtime that is not registered is a 400.

An agent runtime satisfies `AgentRuntime` below. Duck-typed: nothing inherits
it, and the plugin never checks it at runtime — a host that runs mypy gets its
runtimes checked against it, which is the point of writing it down as a type.

The plugin checks the shared contract before `validate_agent` runs: an agent
takes exactly one system and one user message, and no tools.
"""
from __future__ import annotations

from typing import Protocol


class AgentRuntime(Protocol):
    @property
    def name(self) -> str: ...            # the `agent.runtime` value it answers to
    @property
    def default_model(self) -> str: ...   # the model a body that names none gets
    def model_error(self, model: str) -> str | None: ...                      # why a model is not one it takes
    def validate_agent(self, raw: dict) -> tuple[dict | None, str | None]: ...  # its `agent` fields cleaned, or why not
    def list_agents(self) -> list[dict]: ...                                  # rows for `GET /status?agents=1`
    def run(self, request: dict) -> dict: ...                                 # the chat result, or raise AgentRefused


class AgentRefused(Exception):
    """The call must not run, or failed: `code` becomes the job's error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
