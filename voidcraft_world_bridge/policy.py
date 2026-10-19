"""Optional caller restrictions, intersected with the host's guards at dispatch."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})
PROXY_HEADER_PREFIXES = ("x-forwarded-", "tailscale-")
PROXY_HEADERS = frozenset({"forwarded", "via", "x-real-ip"})
TYPED_URL = "none"


@dataclass(frozen=True)
class Refusal:
    code: str
    message: str
    allowed_origins: tuple[str, ...] = ()


def host_hostname(host_header: str) -> str | None:
    try:
        return urlsplit(f"//{host_header}").hostname
    except ValueError:
        return None


@dataclass(frozen=True)
class CallerPolicy:
    """A plugin may narrow access, never replace the host's authorization.

    Origins are supplied by the plugin. None leaves the host's origin policy
    in charge; an empty set permits native callers only. These headers are
    browser/proxy checks, not authentication against other local processes.
    """

    allowed_origins: frozenset[str] | None = None
    loopback_only: bool = True
    reject_proxy: bool = True
    reject_no_cors: bool = True
    subject: str = "plugin"

    def refusal(self, headers: Mapping[str, str] | None) -> Refusal | None:
        named = {str(k).lower(): v for k, v in (headers or {}).items()}
        if self.reject_proxy and any(
            k in PROXY_HEADERS or k.startswith(PROXY_HEADER_PREFIXES) for k in named
        ):
            return Refusal("host_refused", f"The {self.subject} is never served through a proxy")
        host = named.get("host")
        if self.loopback_only and host is not None and host_hostname(host) not in LOOPBACK_HOSTNAMES:
            return Refusal("host_refused", f"The {self.subject} answers on this machine only")
        origin = named.get("origin")
        if origin is not None:
            if self.allowed_origins is None or origin in self.allowed_origins:
                return None
            return Refusal("origin_refused", f"This page's origin may not read the {self.subject}",
                           tuple(sorted(self.allowed_origins)))
        fetch_site = named.get("sec-fetch-site")
        if self.reject_no_cors and fetch_site is not None and fetch_site != TYPED_URL:
            return Refusal("no_cors_refused", f"A browser request without an Origin may not read the {self.subject}")
        return None
