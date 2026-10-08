"""voidcraft-world-bridge: a loopback server that lets voidcraft.world reach your machine.

It mounts plugins under `/plugin/<name>/` behind a Host + Origin allowlist, so a
page on voidcraft.world can talk to something on this machine — a local language
model, first — while a page anywhere else cannot.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("voidcraft-world-bridge")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"
