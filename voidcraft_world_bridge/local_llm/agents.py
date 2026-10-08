"""Agent runtimes: a model reached some other way than a local server.

A `/chat` body with an `agent: {runtime, …}` field goes to the agent runtime of
that name instead of the local model, on its own queue and worker. The plugin
ships with NONE; a host that embeds the bridge registers its own by passing
`agent_runtimes=[…]` to `LocalLlmPlugin` (or `builtin_plugins`). A body naming a
runtime that is not registered is a 400.

An agent runtime is duck-typed:

    name: str                                   # the `agent.runtime` value it answers to
    default_model: str                          # the model a body that names none gets
    model_error(model) -> str | None            # why a model is not one it takes
    validate_agent(raw: dict) -> (dict | None, str | None)
                                                # its own `agent` fields, cleaned, or why not
    list_agents() -> list[dict]                 # rows for `GET /status?agents=1`
    run(request: dict) -> dict                  # the chat result, or raise AgentRefused

The plugin checks the shared contract before `validate_agent` runs: an agent
takes exactly one system and one user message, and no tools.
"""
from __future__ import annotations


class AgentRefused(Exception):
    """The call must not run, or failed: `code` becomes the job's error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
