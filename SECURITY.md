# Security

The bridge is a loopback HTTP server that lets a page on voidcraft.world reach a model server on
the machine it runs on. Its whole security model is in `voidcraft_world_bridge/guards.py` and the
README's "Why it exists" section: loopback bind, a Host guard against DNS rebinding, an Origin
allowlist, per-origin CORS, and bounded request bodies.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository:
<https://github.com/voidcraft-world/voidcraft-world-bridge/security/advisories/new>.
Please do not open a public issue for something exploitable. You will get an answer within a week.

## What counts

- A request served from an origin that is not voidcraft.world or loopback, or a CORS header echoed
  to one.
- A request accepted for a `Host` other than loopback or a name the user listed explicitly.
- Any way to make the bridge bind beyond `127.0.0.1`.
- A plugin POST body accepted past the core cap or the plugin's declared cap.
- The `local-llm` plugin reaching anything other than the probed loopback ports or the one URL the
  user configured, or keeping a prompt after the job is answered.

## What does not

- Setting `VOIDCRAFT_BRIDGE_ALLOWED_HOSTS` to a public name, or publishing the bridge through a
  public tunnel. The README says not to; that is the user's own choice.
- A plugin the user chose to load. A plugin runs with the bridge's own privileges, by design.

## Supported versions

The latest release on PyPI.
