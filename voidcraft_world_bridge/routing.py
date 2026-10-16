"""An optional route table; existing duck-typed plugins need no changes."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

Handler = Callable[[dict[str, list[str]], dict | None], tuple]


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    handler: Handler
    description: str = ""


class RoutedPlugin:
    def __init__(self, name: str, routes: list[Route]) -> None:
        self.name = name
        self._routes: dict[tuple[str, str], Route] = {}
        for route in routes:
            if route.method not in ("GET", "POST") or not route.path.startswith("/"):
                raise ValueError("routes require GET or POST and an absolute subpath")
            key = (route.method, route.path)
            if key in self._routes or not callable(route.handler):
                raise ValueError("duplicate route or non-callable handler")
            self._routes[key] = route

    def routes(self) -> list[str]:
        return list(dict.fromkeys(f"/plugin/{self.name}{path}" for _, path in self._routes))

    def operations(self) -> list[dict]:
        return [{"method": r.method, "path": f"/plugin/{self.name}{r.path}",
                 "description": r.description} for r in self._routes.values()]

    def _dispatch(self, method: str, subpath: str, query: dict[str, list[str]], body: dict | None) -> tuple:
        route = self._routes.get((method, subpath))
        if route is not None:
            return route.handler(query, body)
        if any(path == subpath for _, path in self._routes):
            return 405, {"error": "method not allowed"}
        known = ", ".join(f"{method} {path}" for method, path in self._routes)
        return 404, {"error": f"unknown route — {known}"}

    def handle_get(self, subpath: str, query: dict[str, list[str]]) -> tuple:
        return self._dispatch("GET", subpath, query, None)

    def handle_post(self, subpath: str, query: dict[str, list[str]], body: dict | None) -> tuple:
        return self._dispatch("POST", subpath, query, body)
