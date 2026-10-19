# Changelog

Notable changes to `voidcraft-world-bridge`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer. The contracts
clients depend on are additive-only, so a breaking change is a major version.

## [Unreleased]

### Added

- Compute hosts: other machines' model servers answer questions too, each on its own queue, reached
  through an SSH port forward to an alias in your SSH config (nothing binds to the network).
  `/status` gains `hosts` and `model_hosts`; `POST /chat` takes `host`; jobs report `host` and
  `timings`; `POST /hosts` adds or removes one; `VOIDCRAFT_LOCAL_LLM_HOSTS` names the file.

## [0.1.2] - 2026-10-19

- Add optional caller policies, declarative routes, managed local services, reusable jobs and a socket-free plugin test client.
- Validate plugin factories, advertise startup diagnostics, and add `check-plugin DIR`.
- Close plugin resources with the server; reject negative body lengths and malformed JSON responses safely.
- Clean up rejected factory results and invalidate stopped service endpoints. Keep child HTTP traffic
  direct to loopback, without proxy or redirect credential forwarding.

### Added

- `POST /chat` takes `think` as a reasoning level (`"low"`, `"medium"`, `"high"`) as well as a boolean, and
  passes it to Ollama as given. A model that cannot turn reasoning off (gpt-oss ignores `false`) needs the level.

## [0.1.1] - 2026-10-08

### Added

- `examples/hello-plugin`: a complete plugin to copy, exercised by the test suite.
- `max_body_bytes` on a plugin and the 4 KB default are now in the README.

### Fixed

- A `tools` entry that is not an object is now a `400`, not a `500`.
- `voidcraft-world-bridge status --port N` is accepted; before, `--port` had to come first.
- A `Content-Length` header that is not a number is a `400`; it used to drop the connection.
- A plugin directory is appended to `sys.path`, never put first, so a stray module in it cannot
  shadow the standard library for the whole process.
- The `Server` header names the bridge only, not the Python version it runs on.

## [0.1.0] - 2026-10-08

### Added

- The bridge: a loopback HTTP server with a Host guard, an Origin allowlist, per-origin CORS,
  `GET /snapshot`, and plugins mounted under `/plugin/<name>/`.
- The built-in `local-llm` plugin: finds Ollama, LM Studio or llama.cpp (or the one server you
  point it at), picks the strongest installed chat model, and answers `POST /chat` as a job.
- `voidcraft-world-bridge status`, `--port`, `--version`.

[Unreleased]: https://github.com/voidcraft-world/voidcraft-world-bridge/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/voidcraft-world/voidcraft-world-bridge/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/voidcraft-world/voidcraft-world-bridge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/voidcraft-world/voidcraft-world-bridge/releases/tag/v0.1.0
