"""Socket-free plugin contract client. This tests dispatch, not HTTP Host/CORS guards."""
from __future__ import annotations

from voidcraft_world_bridge.plugins import Plugin, dispatch_plugin, validate_plugin


class PluginClient:
    def __init__(self, plugin: Plugin) -> None:
        validate_plugin(plugin, plugin.name)
        self.plugin = plugin

    def request(self, method: str, path: str, *, query: dict[str, list[str]] | None = None,
                body: dict | None = None, headers: dict[str, str] | None = None) -> tuple:
        if method not in ("GET", "POST"):
            raise ValueError("plugin client supports GET and POST")
        return dispatch_plugin({self.plugin.name: self.plugin}, method, self.plugin.name,
                               path, query or {}, body, headers)
