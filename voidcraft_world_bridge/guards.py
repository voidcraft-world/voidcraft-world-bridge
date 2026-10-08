"""Who may talk to the bridge: the Host (DNS-rebinding) and Origin guards.

Every request a browser sends to a loopback server passes through these, and
they are the whole reason a page on voidcraft.world may reach this machine while
a page anywhere else may not. Pure functions over header values, so the tests
need no socket.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

# The single Origin allowlist for EVERY route: no Origin (curl, CLI, native
# apps), a deployed VoidCraft origin, or ANY loopback origin — the dev SPA roams
# ports (vite falls back to 5174+ when 5173 is taken; mkcert flips the scheme),
# and a loopback page adds nothing an attacker doesn't already have.
#
# One guard, every route — read routes included. A host's read routes can leak
# real content (a terminal pane's visible rows, a process's cwd), and wildcard
# CORS on those means ANY page the user has open can read them cross-origin. The
# browser's Private Network Access preflight is no backstop: it is answered
# permissively for allowlisted origins, and Firefox/Safari don't implement it.
ALLOWED_BROWSER_ORIGINS = frozenset({
    "https://voidcraft.world",
    "https://www.voidcraft.world",
    "https://dev.voidcraft.world",
})

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Extra Host values beyond loopback (`tailscale serve` remote access). The
# second name is the one this bridge was born under, still honoured so an
# existing setup keeps working.
ALLOWED_HOSTS_ENV = "VOIDCRAFT_BRIDGE_ALLOWED_HOSTS"
LEGACY_ALLOWED_HOSTS_ENV = "TERMINAL_BRIDGE_ALLOWED_HOSTS"


def is_loopback_origin(origin: str) -> bool:
    """True for http(s)://localhost|127.0.0.1|[::1] on any port; malformed → False."""
    try:
        parts = urlsplit(origin)
        return parts.scheme in ("http", "https") and parts.hostname in LOOPBACK_HOSTS
    except ValueError:
        return False


def origin_allowed(origin: str | None) -> bool:
    """Whether a request carrying this Origin may be served at all.

    `None` passes so that non-browser clients work: curl, CLI subcommands and
    native apps send no Origin.

    An absent Origin is NOT by itself proof of a non-browser client — see
    `is_no_cors_browser_request`, which expensive routes use to tell the two
    apart. This function is deliberately left permissive because routes that
    are merely *readable* also serve legitimate no-Origin browser requests (a
    plugin page loaded in an iframe, a hand-typed debug URL).
    """
    return origin is None or origin in ALLOWED_BROWSER_ORIGINS or is_loopback_origin(origin)


def is_no_cors_browser_request(origin: str | None, sec_fetch_site: str | None) -> bool:
    """True for a browser request that omitted Origin because it is `no-cors`.

    `<img>`, `<script src>`, `<iframe src>` and `fetch(url, {mode:'no-cors'})`
    all issue REAL cross-origin GETs while sending no Origin header — so the
    "absent Origin means a native client" reading in `origin_allowed` does not
    hold on its own. What does hold is that every browser making such a request
    sends `Sec-Fetch-Site` (Chrome 76+, Firefox 90+, Safari 16.4+), and no
    curl/URLSession/CLI caller sends it at all.

    For routes where the request is EXPENSIVE rather than merely readable: the
    attacker cannot read a no-cors response, so the risk there is not disclosure
    but the work performed (a held thread, a subprocess per call).
    """
    return origin is None and sec_fetch_site is not None


def host_header_hostname(host_header: str) -> str | None:
    """The hostname from a Host header, port and IPv6 brackets stripped."""
    try:
        return urlsplit(f"//{host_header}").hostname
    except ValueError:
        return None


def allowed_remote_hosts() -> frozenset[str]:
    """Extra Host values accepted beyond loopback, from the environment.

    Comma-separated in `VOIDCRAFT_BRIDGE_ALLOWED_HOSTS` (or the legacy
    `TERMINAL_BRIDGE_ALLOWED_HOSTS`; both are read). A full URL is accepted and
    reduced to its hostname, so `https://mymac.tailXXXX.ts.net` works.
    """
    raw = ",".join(os.environ.get(name, "") for name in (ALLOWED_HOSTS_ENV, LEGACY_ALLOWED_HOSTS_ENV))
    hosts: set[str] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "//" in token:
            token = token.split("//", 1)[1]
        parsed = host_header_hostname(token)
        if parsed:
            hosts.add(parsed)
    return frozenset(hosts)


def host_allowed(host_header: str | None) -> bool:
    """Whether the Host this request was addressed to may be served.

    THE DNS-REBINDING GUARD, and the only defence against it — a loopback bind
    is not one. An attacker page on `rebind.evil.com` (1s TTL, re-resolved to
    127.0.0.1) makes the browser open a genuine same-origin connection to this
    server, so every Origin/CORS check passes by construction: the attacker's
    own hostname IS the origin. Only the Host tells us the request was
    addressed to a name we do not answer to.

    Absent Host passes (HTTP/1.0 clients omit it; a browser never does).

    Non-loopback names must be listed explicitly in the env var. **A `*.ts.net`
    suffix rule would be wrong here**: Tailscale *funnel* publishes
    `<machine>.<tailnet>.ts.net` to the public internet, so an attacker with a
    funneled node owns a perfectly real `.ts.net` name. Pin the machine.
    """
    if host_header is None:
        return True
    hostname = host_header_hostname(host_header)
    if hostname is None:
        return False
    return hostname in LOOPBACK_HOSTS or hostname in allowed_remote_hosts()
