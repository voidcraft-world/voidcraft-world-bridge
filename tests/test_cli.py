"""The command line: `status` finds nothing on a port nobody holds, and takes
`--port` on either side of the subcommand."""

from __future__ import annotations

import socket

import pytest

from voidcraft_world_bridge import __version__
from voidcraft_world_bridge.cli import main


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_status_on_a_silent_port_exits_1_and_says_so(capsys):
    port = _free_port()
    assert main(["status", "--port", str(port)]) == 1
    assert f"nothing answering on 127.0.0.1:{port}" in capsys.readouterr().out


def test_port_is_accepted_on_either_side_of_the_subcommand(capsys):
    # `status --port N` used to be "unrecognized arguments"; and a subparser
    # default must not overwrite a port the root parser already read.
    port = _free_port()
    assert main(["--port", str(port), "status"]) == 1
    assert f":{port}" in capsys.readouterr().out
    assert main(["status", "--port", str(port)]) == 1
    assert f":{port}" in capsys.readouterr().out


def test_version_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
