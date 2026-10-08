"""DNS-rebinding and no-cors-GET guards.

Both close holes that the Origin guard structurally cannot:

* Under DNS rebinding the attacker's own hostname becomes the origin, so every
  same-origin check passes by construction. Only the Host header shows that the
  request was addressed to a name this server does not answer to.
* `<img>`, `<script src>`, `<iframe src>` and `fetch(…, {mode:'no-cors'})` all
  send NO Origin, which `origin_allowed` reads as "a native client". That
  reading is what made `GET /wait` a remote fork-bomb.
"""

import pytest

from voidcraft_world_bridge.guards import (
    ALLOWED_HOSTS_ENV,
    LEGACY_ALLOWED_HOSTS_ENV,
    allowed_remote_hosts,
    host_allowed,
    host_header_hostname,
    is_no_cors_browser_request,
    origin_allowed,
)


@pytest.fixture(autouse=True)
def _clear_allowlist(monkeypatch):
    """Every test starts from the shipped default: loopback only."""
    monkeypatch.delenv(ALLOWED_HOSTS_ENV, raising=False)
    monkeypatch.delenv(LEGACY_ALLOWED_HOSTS_ENV, raising=False)


def test_legacy_env_name_still_allows_a_host(monkeypatch):
    # The name this bridge was born under keeps an existing tailnet setup working.
    monkeypatch.setenv(LEGACY_ALLOWED_HOSTS_ENV, "mymac.tailXXXX.ts.net")
    assert host_allowed("mymac.tailXXXX.ts.net:8443") is True


def test_origin_allowlist():
    assert origin_allowed(None) is True  # curl, CLI, native apps
    assert origin_allowed("https://voidcraft.world") is True
    assert origin_allowed("http://localhost:5174") is True
    assert origin_allowed("https://evil.example") is False
    assert origin_allowed("https://voidcraft.world.evil.example") is False
    assert origin_allowed("null") is False


# --- host_allowed ------------------------------------------------------------


def test_loopback_hosts_allowed_on_any_port():
    for host in (
        "127.0.0.1:7682",
        "localhost:7682",
        "[::1]:7682",
        "127.0.0.1",
        "localhost",
        "LOCALHOST:7682",  # hostname is case-normalized
    ):
        assert host_allowed(host) is True, host


def test_absent_host_allowed():
    # HTTP/1.0 clients omit it; a browser never does.
    assert host_allowed(None) is True


def test_rebinding_hostnames_rejected():
    # The whole point: these resolve to 127.0.0.1 at connect time, so the
    # socket, the bind and every Origin check are all satisfied.
    for host in (
        "rebind.evil.com:7682",
        "evil.example",
        "localhost.evil.com:7682",  # loopback-lookalike
        "127.0.0.1.evil.com:7682",
        "attacker.tailscale-lookalike.net",
    ):
        assert host_allowed(host) is False, host


def test_funnelled_ts_net_is_not_trusted_by_default():
    # Deliberately NOT a `*.ts.net` suffix rule: Tailscale funnel publishes
    # <machine>.<tailnet>.ts.net publicly, so an attacker with a funneled node
    # owns a real one. Same reasoning as the frontend's rack-remote.ts pin.
    assert host_allowed("attacker.tailXXXX.ts.net") is False


def test_configured_remote_host_allowed(monkeypatch):
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "mymac.tailXXXX.ts.net")
    assert host_allowed("mymac.tailXXXX.ts.net") is True
    assert host_allowed("mymac.tailXXXX.ts.net:8443") is True
    # Still only the pinned name — a sibling on the same tailnet is not implied.
    assert host_allowed("other.tailXXXX.ts.net") is False


def test_allowlist_accepts_urls_and_lists(monkeypatch):
    monkeypatch.setenv(
        ALLOWED_HOSTS_ENV,
        " https://mymac.tailXXXX.ts.net:8443 , box.local ,, ",
    )
    assert allowed_remote_hosts() == frozenset({"mymac.tailxxxx.ts.net", "box.local"})
    assert host_allowed("mymac.tailXXXX.ts.net") is True
    assert host_allowed("box.local:7682") is True


def test_malformed_host_rejected():
    assert host_allowed("[::1") is False  # malformed IPv6 → ValueError path
    assert host_allowed("") is False


def test_host_header_hostname_strips_port_and_brackets():
    assert host_header_hostname("127.0.0.1:7682") == "127.0.0.1"
    assert host_header_hostname("[::1]:7682") == "::1"
    assert host_header_hostname("Example.COM") == "example.com"


# --- is_no_cors_browser_request ----------------------------------------------


def test_native_client_is_not_a_no_cors_request():
    # curl, CLI tools and native apps send neither header.
    assert is_no_cors_browser_request(None, None) is False


def test_no_origin_plus_sec_fetch_is_a_browser():
    # <img src>, <script src>, <iframe src>, fetch(mode:'no-cors').
    for site in ("cross-site", "same-site", "none", "same-origin"):
        assert is_no_cors_browser_request(None, site) is True, site


def test_cors_request_with_origin_is_left_to_the_origin_guard():
    # The deployed SPA reaching loopback sends Origin AND Sec-Fetch-Site:
    # cross-site. It must not be caught here — origin_allowed admits it.
    assert is_no_cors_browser_request("https://voidcraft.world", "cross-site") is False
    assert is_no_cors_browser_request("http://localhost:5173", "cross-site") is False
    # An origin that is NOT allowlisted is still the Origin guard's job to
    # reject, not this one's.
    assert is_no_cors_browser_request("https://evil.example", "cross-site") is False
