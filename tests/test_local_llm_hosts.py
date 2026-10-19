"""Compute hosts: the file, the alias gate, routing between machines, and the /hosts route.

Tunnels are faked here (`FakeTunnel`); `test_local_llm_tunnel.py` drives the real
`SshTunnel` against a fake `ssh`. Every machine's model server is a FakeOllama,
told apart by the port its tunnel reports.
"""
from __future__ import annotations

import os
import stat
import urllib.parse
from dataclasses import dataclass

import pytest
from fake_ollama import FakeOllama

from voidcraft_world_bridge.local_llm import hosts as hosts_mod
from voidcraft_world_bridge.local_llm import jobs as jobs_mod
from voidcraft_world_bridge.local_llm.plugin import LocalLlmPlugin
from voidcraft_world_bridge.local_llm.tunnel import DOWN, UP

LOCAL_PORT = 11434


@dataclass
class Context:
    start_dir: str = "."
    log: object = None


class FakeTunnel:
    def __init__(self, port: int | None, error: str | None = None) -> None:
        self.port, self.error = port, error
        self.closed = False
        self.ensured = 0

    def snapshot(self):
        return (UP if self.port else DOWN), self.port, self.error

    def ensure(self):
        self.ensured += 1
        return self.snapshot()

    def wait_up(self, timeout=None):
        return self.port

    def close(self):
        self.closed = True


class Machines:
    """An opener that answers as whichever FakeOllama owns the request's port."""

    def __init__(self, by_port: dict[int, FakeOllama]) -> None:
        self.by_port = by_port

    def __call__(self, request, timeout=None):
        port = urllib.parse.urlsplit(request.full_url).port
        return self.by_port[port](request, timeout)


def model(name: str) -> dict:
    return {"name": name, "details": {"family": name.split(":")[0]}}


SSH_CONFIG = """\
Host studio
  HostName studio.lan
Host spare box-2  # two aliases on one line
Host *
  ServerAliveInterval 30
Host !nope
Host=equals
"""


@pytest.fixture
def ssh_config(tmp_path):
    path = tmp_path / "ssh_config"
    path.write_text(SSH_CONFIG)
    return path


def make(tmp_path, ssh_config, remotes: dict[str, tuple[FakeOllama, FakeTunnel]],
         local: FakeOllama | None = None) -> tuple[LocalLlmPlugin, Machines]:
    hosts_mod.save_hosts([hosts_mod.RemoteHost(name, name) for name in remotes], tmp_path / "hosts.json")
    machines = Machines({LOCAL_PORT: local or FakeOllama(models=[model("qwen3.8:latest")])})
    tunnels = {}
    for name, (fake, tunnel) in remotes.items():
        tunnels[name] = tunnel
        if tunnel.port:
            machines.by_port[tunnel.port] = fake
    plugin = LocalLlmPlugin(Context(), opener=machines, start_worker=False,
                            hosts_path=tmp_path / "hosts.json", ssh_config_path=ssh_config,
                            tunnel_factory=lambda remote: tunnels[remote.name])
    return plugin, machines


CHAT = {"system": "You command the red side.", "user": "Your turn.", "think": False}


# -- the file and the alias gate -------------------------------------------------

@pytest.mark.parametrize("alias", ["-oProxyCommand=touch /tmp/x", "a b", "a;rm", "", "local", "x" * 70,
                                   "$(id)", "studio\n", ".hidden", None, 7])
def test_an_alias_that_could_be_an_option_or_a_command_is_refused(alias):
    assert hosts_mod.alias_problem(alias)


@pytest.mark.parametrize("alias", ["studio", "box-2.lan", "Gpu_1"])
def test_ordinary_aliases_are_accepted(alias):
    assert hosts_mod.alias_problem(alias) is None


def test_only_literal_host_lines_are_aliases(ssh_config):
    assert hosts_mod.configured_aliases(ssh_config) == {"studio", "spare", "box-2", "equals"}


def test_no_ssh_config_means_no_aliases(tmp_path):
    assert hosts_mod.configured_aliases(tmp_path / "missing") == set()


def test_a_missing_or_corrupt_file_is_no_hosts(tmp_path):
    assert hosts_mod.load_hosts(tmp_path / "missing.json") == []
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert hosts_mod.load_hosts(corrupt) == []


def test_bad_entries_are_dropped_and_the_first_of_a_name_wins(tmp_path):
    path = tmp_path / "hosts.json"
    path.write_text('{"version": 1, "hosts": ['
                    '{"ssh": "studio"}, {"ssh": "studio", "port": 1}, {"ssh": "-oX=y"},'
                    '{"ssh": "spare", "kind": "openai"}, {"ssh": "box", "port": 99999}, "nope"]}')
    assert hosts_mod.load_hosts(path) == [
        hosts_mod.RemoteHost("studio", "studio", "ollama", 11434),
        hosts_mod.RemoteHost("spare", "spare", "openai", 8080),
    ]


def test_the_file_round_trips_and_is_owner_only(tmp_path):
    path = tmp_path / "nested" / "hosts.json"
    hosts = [hosts_mod.RemoteHost("studio", "studio")]
    hosts_mod.save_hosts(hosts, path)
    assert hosts_mod.load_hosts(path) == hosts
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# -- routing ---------------------------------------------------------------------

def test_status_lists_every_host_and_where_each_model_is(tmp_path, ssh_config):
    remote = FakeOllama(models=[model("gpt-oss:20b"), model("qwen3.8:latest")])
    plugin, _ = make(tmp_path, ssh_config, {"studio": (remote, FakeTunnel(5001))})
    status, body = plugin.handle_get("/status", {})
    assert status == 200
    assert [h["name"] for h in body["hosts"]] == ["local", "studio"]
    local, studio = body["hosts"]
    assert local["local"] is True and local["tunnel"] == "none"
    assert studio == {"name": "studio", "local": False, "kind": "ollama", "reachable": True,
                      "models": ["gpt-oss:20b", "qwen3.8:latest"], "queue": 0, "running": False,
                      "tok_s": None, "prefill_tok_s": None, "tunnel": "up", "error": None}
    assert [m["name"] for m in body["models"]] == ["qwen3.8:latest", "gpt-oss:20b"]
    assert body["model_hosts"] == {"qwen3.8:latest": ["local", "studio"], "gpt-oss:20b": ["studio"]}
    assert body["picked_model"] == "qwen3.8:latest", "the pick stays this machine's"


def test_with_no_hosts_file_status_has_only_this_machine(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config, {})
    _, body = plugin.handle_get("/status", {})
    assert [h["name"] for h in body["hosts"]] == ["local"]
    assert body["model_hosts"] == {"qwen3.8:latest": ["local"]}


def test_a_down_tunnel_is_unreachable_and_says_why(tmp_path, ssh_config):
    tunnel = FakeTunnel(None, "ssh exited (255): Permission denied (publickey)")
    plugin, _ = make(tmp_path, ssh_config, {"studio": (FakeOllama(), tunnel)})
    _, body = plugin.handle_get("/status", {})
    studio = body["hosts"][1]
    assert studio["reachable"] is False and studio["tunnel"] == "down"
    assert "Permission denied" in studio["error"]
    assert tunnel.ensured, "status starts a tunnel that is down"


def test_a_named_host_answers_and_the_job_says_so(tmp_path, ssh_config):
    remote = FakeOllama(models=[model("gpt-oss:20b")])
    plugin, machines = make(tmp_path, ssh_config, {"studio": (remote, FakeTunnel(5001))})
    status, body = plugin.handle_post("/chat", {}, {**CHAT, "model": "gpt-oss:20b", "host": "studio"})
    assert status == 202 and body["host"] == "studio"
    assert plugin._lanes["studio"].store.run_next() is True
    _, job = plugin.handle_get("/job", {"id": [body["job_id"]]})
    assert job["state"] == jobs_mod.DONE and job["host"] == "studio"
    assert job["timings"] == {"prompt_tokens": 120, "prompt_s": 0.6, "output_tokens": 30, "output_s": 2.0}
    assert "timings" not in job["result"], "the result keeps the adapter's shape"
    assert any(path == "/api/chat" for path, _ in remote.requests)
    assert not any(path == "/api/chat" for path, _ in machines.by_port[LOCAL_PORT].requests)
    _, status_body = plugin.handle_get("/status", {})
    assert status_body["hosts"][1]["tok_s"] == 15.0
    assert status_body["hosts"][1]["prefill_tok_s"] == 200.0


def test_a_model_only_another_machine_has_goes_there(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config,
                     {"studio": (FakeOllama(models=[model("gpt-oss:20b")]), FakeTunnel(5001))})
    _, body = plugin.handle_post("/chat", {}, {**CHAT, "model": "gpt-oss:20b"})
    assert body["host"] == "studio"


def test_a_shared_model_goes_to_the_least_busy_host_and_ties_stay_here(tmp_path, ssh_config):
    shared = [model("qwen3.8:latest")]
    plugin, _ = make(tmp_path, ssh_config, {"studio": (FakeOllama(models=shared), FakeTunnel(5001))},
                     local=FakeOllama(models=shared))
    ask = {**CHAT, "model": "qwen3.8:latest"}
    assert plugin.handle_post("/chat", {}, ask)[1]["host"] == "local"
    assert plugin.handle_post("/chat", {}, ask)[1]["host"] == "studio"
    assert plugin.handle_post("/chat", {}, ask)[1]["host"] == "local"


def test_a_question_naming_no_model_stays_here(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config, {"studio": (FakeOllama(), FakeTunnel(5001))})
    assert plugin.handle_post("/chat", {}, dict(CHAT))[1]["host"] == "local"


def test_a_model_nobody_has_fails_here_the_way_it_always_did(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config, {"studio": (FakeOllama(), FakeTunnel(5001))})
    _, body = plugin.handle_post("/chat", {}, {**CHAT, "model": "nope:1b"})
    assert body["host"] == "local"
    plugin._jobs.run_next()
    _, job = plugin.handle_get("/job", {"id": [body["job_id"]]})
    assert job["error"]["code"] == "model_not_installed"


def test_naming_a_host_is_refused_at_once_when_it_cannot_answer(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config, {
        "studio": (FakeOllama(models=[model("qwen3.5:9b")]), FakeTunnel(5001)),
        "spare": (FakeOllama(), FakeTunnel(None, "ssh exited (255)")),
    })
    assert plugin.handle_post("/chat", {}, {**CHAT, "host": "nowhere"})[1]["error"] == "host_unknown"
    status, body = plugin.handle_post("/chat", {}, {**CHAT, "host": "spare"})
    assert (status, body["error"]) == (409, "host_unreachable")
    status, body = plugin.handle_post("/chat", {}, {**CHAT, "host": "studio", "model": "gpt-oss:20b"})
    assert (status, body["error"]) == (409, "model_not_on_host")
    status, body = plugin.handle_post("/chat", {}, {**CHAT, "host": "-oProxyCommand=x"})
    assert status == 400


def test_each_host_has_its_own_bounded_queue(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config,
                     {"studio": (FakeOllama(models=[model("gpt-oss:20b")]), FakeTunnel(5001))})
    ask = {**CHAT, "host": "studio", "model": "gpt-oss:20b"}
    for _ in range(jobs_mod.MAX_PENDING):
        assert plugin.handle_post("/chat", {}, ask)[0] == 202
    status, body = plugin.handle_post("/chat", {}, ask)
    assert status == 429 and "studio" in body["message"]
    assert plugin.handle_post("/chat", {}, dict(CHAT))[0] == 202, "this machine's queue is separate"


def test_a_host_that_slept_mid_queue_fails_its_job_not_the_bridge(tmp_path, ssh_config):
    tunnel = FakeTunnel(5001)
    plugin, _ = make(tmp_path, ssh_config, {"studio": (FakeOllama(models=[model("gpt-oss:20b")]), tunnel)})
    _, body = plugin.handle_post("/chat", {}, {**CHAT, "host": "studio", "model": "gpt-oss:20b"})
    tunnel.port, tunnel.error = None, "ssh exited (255): Broken pipe"
    plugin._lanes["studio"].store.run_next()
    _, job = plugin.handle_get("/job", {"id": [body["job_id"]]})
    assert job["error"]["code"] == "host_unreachable" and "Broken pipe" in job["error"]["message"]


# -- POST /hosts -------------------------------------------------------------------

LOOPBACK_HEADERS = {"host": "127.0.0.1:7682", "origin": "https://voidcraft.world"}


def edit(plugin, body, headers=None):
    return plugin.handle_request("POST", "/hosts", {}, body, headers or dict(LOOPBACK_HEADERS))


def test_adding_a_configured_alias_saves_it_and_starts_its_tunnel(tmp_path, ssh_config):
    tunnel = FakeTunnel(5001)
    plugin, machines = make(tmp_path, ssh_config, {})
    machines.by_port[5001] = FakeOllama(models=[model("gpt-oss:20b")])
    plugin._tunnel_factory = lambda remote: tunnel
    status, body = edit(plugin, {"action": "add", "ssh": "studio"})
    assert status == 200 and body["ok"] is True
    assert [h["name"] for h in body["hosts"]] == ["local", "studio"]
    assert hosts_mod.load_hosts(tmp_path / "hosts.json") == [hosts_mod.RemoteHost("studio", "studio")]
    assert tunnel.ensured
    assert edit(plugin, {"action": "add", "ssh": "studio"})[0] == 200, "adding it again is a no-op"


@pytest.mark.parametrize("alias", ["-oProxyCommand=sh", "unlisted", "*", "nope", "local"])
def test_an_alias_not_literally_in_the_ssh_config_is_refused(tmp_path, ssh_config, alias):
    plugin, _ = make(tmp_path, ssh_config, {})
    status, body = edit(plugin, {"action": "add", "ssh": alias})
    assert (status, body["error"]) == (400, "bad_alias")
    assert not (tmp_path / "hosts.json").read_text().count('"ssh"')


def test_removing_a_host_closes_its_tunnel_and_keeps_its_answers(tmp_path, ssh_config):
    tunnel = FakeTunnel(5001)
    plugin, _ = make(tmp_path, ssh_config, {"studio": (FakeOllama(models=[model("gpt-oss:20b")]), tunnel)})
    _, asked = plugin.handle_post("/chat", {}, {**CHAT, "host": "studio", "model": "gpt-oss:20b"})
    plugin._lanes["studio"].store.run_next()
    status, body = edit(plugin, {"action": "remove", "name": "studio"})
    assert status == 200 and [h["name"] for h in body["hosts"]] == ["local"]
    assert tunnel.closed
    assert hosts_mod.load_hosts(tmp_path / "hosts.json") == []
    assert plugin.handle_get("/job", {"id": [asked["job_id"]]})[0] == 200


def test_this_machine_cannot_be_removed_and_unknown_hosts_are_named(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config, {})
    assert edit(plugin, {"action": "remove", "name": "local"})[1]["error"] == "cannot_remove_local"
    assert edit(plugin, {"action": "remove", "name": "ghost"})[1]["error"] == "host_unknown"
    assert edit(plugin, {"action": "rename"})[0] == 400


@pytest.mark.parametrize("headers", [
    {"host": "laptop.example.ts.net", "origin": "https://voidcraft.world"},
    {**LOOPBACK_HEADERS, "x-forwarded-for": "10.0.0.2"},
    {"host": "127.0.0.1:7682", "sec-fetch-site": "cross-site"},
])
def test_the_host_list_is_edited_from_this_machine_only(tmp_path, ssh_config, headers):
    plugin, _ = make(tmp_path, ssh_config, {})
    status, _ = edit(plugin, {"action": "add", "ssh": "studio"}, headers)
    assert status == 403
    assert hosts_mod.load_hosts(tmp_path / "hosts.json") == []


def test_other_routes_are_not_narrowed_by_the_hosts_policy(tmp_path, ssh_config):
    plugin, _ = make(tmp_path, ssh_config, {})
    status, _ = plugin.handle_request("GET", "/status", {}, None, {"host": "laptop.example.ts.net"})
    assert status == 200
