"""The plugins every bridge mounts without being told to.

Today that is `local-llm`: the reason a stranger runs this bridge at all is to
let voidcraft.world reach their own model, so it must work with zero config. A
plugin directory configured under the same name is skipped, never mounted twice
(`load_plugins`, `preloaded`).
"""
from __future__ import annotations

from collections.abc import Iterable

from voidcraft_world_bridge.local_llm.agents import AgentRuntime
from voidcraft_world_bridge.local_llm.plugin import PLUGIN_NAME as LOCAL_LLM
from voidcraft_world_bridge.local_llm.plugin import LocalLlmPlugin
from voidcraft_world_bridge.plugins import Plugin, PluginContext


def builtin_plugins(context: PluginContext, *,
                    agent_runtimes: Iterable[AgentRuntime] = ()) -> dict[str, Plugin]:
    """name → plugin for every built-in. `agent_runtimes` are handed to
    `local-llm` (see `local_llm/agents.py`); a stock bridge registers none."""
    return {LOCAL_LLM: LocalLlmPlugin(context, agent_runtimes=agent_runtimes)}
