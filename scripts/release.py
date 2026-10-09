"""Cut a release: `prepare` opens the release PR, `tag` publishes it once merged.

    uv run python scripts/release.py prepare patch       # or minor, major, X.Y.Z
    uv run python scripts/release.py tag                 # on main, after the PR merges
    uv run python scripts/release.py sync-consumers      # move dependents to the latest release

Every subcommand takes `--dry-run`: preflight and plan, nothing changed.

Why two steps. The release commit (version bump, CHANGELOG, lock) is reviewed and passes CI like
any change. The tag is the one irreversible act — PyPI never accepts the same version twice, and
`publish.yml` uploads on any `v*` tag — so it gets its own preflight and a confirmation.

`tag` is re-runnable. If the tag is already on the remote it skips to verification, so a release
that was merged but never tagged, or a publish that timed out, is finished by running it again.

Consumers. A project that depends on this package (a host that embeds the bridge) can follow
every release: list its directories in `VOIDCRAFT_BRIDGE_CONSUMERS` (`os.pathsep`-separated) or
pass `--consumer`, and `tag` ends by moving each one's lock to the new version. The lock change
is left uncommitted: the consumer owns its own history.

Stdlib only, like the package. It drives `git`, `uv` and `gh`, which a maintainer already has.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "voidcraft-world-bridge"
REPO_URL = "https://github.com/voidcraft-world/voidcraft-world-bridge"
PYPI_JSON = f"https://pypi.org/pypi/{PACKAGE}/json"
CONSUMERS_ENV = "VOIDCRAFT_BRIDGE_CONSUMERS"
PORT_ENV = "VOIDCRAFT_BRIDGE_PORT"
DEFAULT_PORT = 7682
# The repository is public; a commit with a personal address publishes it for good.
NOREPLY_SUFFIX = "@users.noreply.github.com"
BUMPS = ("patch", "minor", "major")

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
PROJECT_VERSION_RE = re.compile(r'^version = "([^"]*)"', re.MULTILINE)
HEADING_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.MULTILINE)
UNRELEASED_HEADING = "## [Unreleased]\n"
UNRELEASED_LINK_RE = re.compile(r"^\[Unreleased\]: (\S+)/compare/v(\d+\.\d+\.\d+)\.\.\.HEAD$", re.MULTILINE)


class ReleaseError(Exception):
    """A preflight or verification failed; the message says what to do."""


# --- Pure: versions, pyproject, CHANGELOG, lock files ---------------------------------------


def parse_version(text: str) -> tuple[int, int, int]:
    match = VERSION_RE.match(text.strip())
    if not match:
        raise ReleaseError(f"not a release version (X.Y.Z): {text!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def bump_version(current: str, spec: str) -> str:
    """`spec` is `patch`/`minor`/`major`, or an explicit X.Y.Z above `current`."""
    major, minor, patch = parse_version(current)
    if spec == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if spec == "minor":
        return f"{major}.{minor + 1}.0"
    if spec == "major":
        return f"{major + 1}.0.0"
    if parse_version(spec) <= (major, minor, patch):
        raise ReleaseError(f"{spec} is not above the current version {current}")
    return spec


def project_version(pyproject: str) -> str:
    # A regex, not tomllib: tomllib is 3.11+ and the suite runs on 3.10. The first
    # `version = ` at column 0 is [project]'s; no other table here sets one.
    match = PROJECT_VERSION_RE.search(pyproject)
    if not match:
        raise ReleaseError("pyproject.toml has no `version = ` line")
    return match.group(1)


def set_project_version(pyproject: str, version: str) -> str:
    project_version(pyproject)  # raises when there is nothing to replace
    return PROJECT_VERSION_RE.sub(f'version = "{version}"', pyproject, count=1)


def changelog_versions(changelog: str) -> list[str]:
    """The released versions, newest first, as their headings list them."""
    return HEADING_RE.findall(changelog)


def unreleased_notes(changelog: str) -> str:
    """The body under `## [Unreleased]`, up to the next version heading."""
    start = changelog.find(UNRELEASED_HEADING)
    if start < 0:
        raise ReleaseError("CHANGELOG.md has no `## [Unreleased]` heading")
    body_start = start + len(UNRELEASED_HEADING)
    nxt = changelog.find("\n## [", body_start)
    return changelog[body_start:] if nxt < 0 else changelog[body_start:nxt + 1]


def rewrite_changelog(changelog: str, version: str, date: str) -> str:
    """Move `[Unreleased]` under `## [version] - date`, open a fresh empty `[Unreleased]`,
    and point the compare links at the new tag."""
    notes = unreleased_notes(changelog)
    if not notes.strip():
        raise ReleaseError("`## [Unreleased]` in CHANGELOG.md is empty: say what this release changes first")
    link = UNRELEASED_LINK_RE.search(changelog)
    if not link:
        raise ReleaseError("CHANGELOG.md has no `[Unreleased]: <repo>/compare/vX.Y.Z...HEAD` link")
    repo, previous = link.groups()
    start = changelog.find(UNRELEASED_HEADING) + len(UNRELEASED_HEADING)
    released = f"\n## [{version}] - {date}\n\n" + notes.lstrip("\n")
    out = changelog[:start] + released + changelog[start + len(notes):]
    links = (f"[Unreleased]: {repo}/compare/v{version}...HEAD\n"
             f"[{version}]: {repo}/compare/v{previous}...v{version}")
    return UNRELEASED_LINK_RE.sub(links, out, count=1)


def locked_version(lock: str, package: str = PACKAGE) -> str | None:
    """The version of `package` a `uv.lock` pins, or None when it is not in the lock."""
    match = re.search(rf'^\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "([^"]+)"', lock, re.MULTILINE)
    return match.group(1) if match else None


def declared_constraint(pyproject: str, package: str = PACKAGE) -> str | None:
    """The requirement string a consumer declares for `package`, e.g. `>=0.1,<0.2`."""
    match = re.search(rf'"{re.escape(package)}((?:\[[^\]]*\])?[^"]*)"', pyproject)
    return match.group(1).strip() if match else None


def parse_consumers(env_value: str, extra: list[str] | None = None) -> list[Path]:
    """Consumer directories from the env var plus `--consumer`, expanded, deduplicated, in order."""
    raw = [*env_value.split(os.pathsep), *(extra or [])]
    seen: dict[Path, None] = {}
    for part in raw:
        if part.strip():
            seen.setdefault(Path(part.strip()).expanduser().resolve(), None)
    return list(seen)


def check_consumer_locked(consumer: Path, target: str) -> str:
    """The version `consumer` now locks; raises when it is not `target`, naming why."""
    pyproject = consumer / "pyproject.toml"
    lock = consumer / "uv.lock"
    if not pyproject.is_file() or not lock.is_file():
        raise ReleaseError(f"{consumer}: needs a pyproject.toml and a uv.lock")
    constraint = declared_constraint(pyproject.read_text())
    if constraint is None:
        raise ReleaseError(f"{consumer}: does not depend on {PACKAGE}")
    locked = locked_version(lock.read_text())
    if locked == target:
        return locked
    # `uv lock -P` stays inside the declared range, so a release outside it leaves the old
    # pin. Widening is deliberate: on 0.x a minor bump may break the host.
    raise ReleaseError(
        f"{consumer}: locks {locked}, not {target}. Its requirement `{PACKAGE}{constraint}` "
        f"excludes {target}? Widen it deliberately, then run `sync-consumers` again.")


# --- Impure: processes, network --------------------------------------------------------------


def say(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], cwd: Path = ROOT, *, capture: bool = True, check: bool = True) -> str:
    """Run a command. Captured output is returned; on failure, it is the error."""
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture, check=False)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        raise ReleaseError(f"`{' '.join(cmd)}` failed ({proc.returncode})" + (f":\n{detail}" if detail else ""))
    return (proc.stdout or "").strip() if capture else ""


def act(dry_run: bool, description: str, cmd: list[str], cwd: Path = ROOT) -> str:
    """A step that changes something: printed under --dry-run, run otherwise."""
    if dry_run:
        say(f"  would {description}: {' '.join(cmd)}")
        return ""
    say(f"→ {description}")
    return run(cmd, cwd)


def pypi_versions() -> set[str]:
    try:
        with urllib.request.urlopen(PYPI_JSON, timeout=15) as response:
            return set(json.load(response)["releases"])
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return set()
        raise ReleaseError(f"PyPI answered {err.code} for {PYPI_JSON}") from err
    except urllib.error.URLError as err:
        raise ReleaseError(f"cannot reach PyPI: {err.reason}") from err


def latest_release(versions: set[str]) -> str:
    released = [v for v in versions if VERSION_RE.match(v)]
    if not released:
        raise ReleaseError(f"{PACKAGE} has no release on PyPI yet")
    return max(released, key=parse_version)


def running_bridge_version() -> str | None:
    """The core version a bridge on the default port reports in `/snapshot`, if one is up."""
    port = os.environ.get(PORT_ENV) or str(DEFAULT_PORT)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot", timeout=2) as response:
            version = json.load(response).get("bridge", {}).get("version")
    except (OSError, ValueError, AttributeError):
        return None
    return version if isinstance(version, str) else None


def read(name: str) -> str:
    return (ROOT / name).read_text()


def tag_name(version: str) -> str:
    return f"v{version}"


def remote_has_tag(tag: str) -> bool:
    return bool(run(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"]))


def preflight_main() -> str:
    """On `main`, clean, and even with `origin/main`. Returns HEAD's SHA."""
    run(["git", "fetch", "--quiet", "--tags", "origin", "main"])
    if run(["git", "status", "--porcelain"]):
        raise ReleaseError("the working tree is not clean: commit or stash first")
    branch = run(["git", "branch", "--show-current"])
    if branch != "main":
        raise ReleaseError(f"on `{branch}`, not `main`")
    head, upstream = run(["git", "rev-parse", "HEAD"]), run(["git", "rev-parse", "origin/main"])
    if head != upstream:
        raise ReleaseError("`main` is not even with `origin/main`: pull (or push) first")
    return head


def preflight_identity() -> None:
    email = run(["git", "config", "user.email"], check=False)
    if not email.endswith(NOREPLY_SUFFIX):
        raise ReleaseError(f"git user.email is not a GitHub noreply address (…{NOREPLY_SUFFIX}); "
                           "this history is public. Set it for this repo: `git config user.email …`")


def typos_pin() -> str:
    """The `typos@X.Y.Z` that CI runs, read from ci.yml so the two cannot drift."""
    match = re.search(r"uvx (typos@[\d.]+)", read(".github/workflows/ci.yml"))
    if not match:
        raise ReleaseError("ci.yml has no `uvx typos@X.Y.Z` step")
    return match.group(1)


def run_gates(version: str) -> None:
    """The CI ladder, locally: lint, types, spelling, tests, then build and smoke the wheel."""
    for cmd in (["uv", "lock", "--check"], ["uv", "run", "ruff", "check", "."], ["uv", "run", "mypy"],
                ["uvx", typos_pin()], ["uv", "run", "pytest", "-q"]):
        say(f"→ gate: {' '.join(cmd)}")
        run(cmd)
    with tempfile.TemporaryDirectory() as out:
        say("→ gate: uv build, then the wheel installs and runs")
        run(["uv", "build", "--out-dir", out])
        wheel = str(next(Path(out).glob("*.whl")))
        printed = run(["uvx", "--from", wheel, PACKAGE, "--version"])
        if version not in printed:
            raise ReleaseError(f"the wheel reports {printed!r}, not {version}")
        # With nothing listening, `status` exits 1 — proof cli → server → plugins import from the wheel.
        status = subprocess.run(["uvx", "--from", wheel, PACKAGE, "status", "--port", "7799"],
                                cwd=ROOT, capture_output=True, check=False)
        if status.returncode != 1:
            raise ReleaseError(f"the wheel's `status` exited {status.returncode}, expected 1")


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


# --- prepare ---------------------------------------------------------------------------------


def prepare(spec: str, dry_run: bool) -> None:
    preflight_main()
    preflight_identity()
    current = project_version(read("pyproject.toml"))
    version = bump_version(current, spec)
    tag = tag_name(version)
    today = datetime.date.today().isoformat()
    notes = unreleased_notes(read("CHANGELOG.md")).strip()
    changelog = rewrite_changelog(read("CHANGELOG.md"), version, today)  # raises on an empty [Unreleased]
    if run(["git", "tag", "--list", tag]) or remote_has_tag(tag):
        raise ReleaseError(f"{tag} already exists")
    if version in pypi_versions():
        raise ReleaseError(f"{version} is already on PyPI")

    branch = f"release/{tag}"
    say(f"Release {current} → {version} on `{branch}`:\n\n{notes}\n")
    if dry_run:
        say("(dry run: no files, branches or PRs changed)")
        return

    run(["git", "switch", "--quiet", "-c", branch])
    (ROOT / "pyproject.toml").write_text(set_project_version(read("pyproject.toml"), version))
    (ROOT / "CHANGELOG.md").write_text(changelog)
    run(["uv", "lock"])
    run_gates(version)
    run(["git", "add", "pyproject.toml", "CHANGELOG.md", "uv.lock"])
    run(["git", "commit", "--quiet", "-m", f"Release {version}"])
    run(["git", "push", "--quiet", "-u", "origin", branch])
    with tempfile.TemporaryDirectory() as scratch:
        body = Path(scratch) / "body.md"
        body.write_text(f"{notes}\n\nAfter this merges, on `main`: `uv run python scripts/release.py tag`.\n")
        url = run(["gh", "pr", "create", "--base", "main", "--title", f"Release {version}", "--body-file", str(body)])
    say(f"✓ {url}\nAfter it merges: git switch main && git pull && uv run python scripts/release.py tag")


# --- tag -------------------------------------------------------------------------------------


def wait_for_ci(sha: str) -> None:
    runs = json.loads(run(["gh", "run", "list", "--commit", sha, "--workflow", "ci.yml",
                           "--json", "databaseId,status,conclusion"]))
    if not runs:
        raise ReleaseError(f"no `ci` run for {sha[:7]} yet: push it, or wait for GitHub to start one")
    latest = runs[0]
    if latest["status"] != "completed":
        say(f"→ waiting for ci run {latest['databaseId']}")
        run(["gh", "run", "watch", str(latest["databaseId"]), "--exit-status"], capture=False)
    elif latest["conclusion"] != "success":
        raise ReleaseError(f"ci on {sha[:7]} concluded `{latest['conclusion']}`")


def watch_publish(tag: str) -> None:
    for _ in range(30):
        runs = json.loads(run(["gh", "run", "list", "--workflow", "publish.yml", "--branch", tag,
                               "--json", "databaseId"]))
        if runs:
            say(f"→ watching publish run {runs[0]['databaseId']}")
            run(["gh", "run", "watch", str(runs[0]["databaseId"]), "--exit-status"], capture=False)
            return
        time.sleep(5)
    raise ReleaseError(f"no publish run started for {tag}: check the Actions tab")


def verify_published(version: str, dry_run: bool) -> None:
    say(f"→ verifying {version}")
    for attempt in range(1, 31):
        if version in pypi_versions():
            break
        if dry_run or attempt == 30:
            raise ReleaseError(f"{version} is not on PyPI ({PYPI_JSON})")
        time.sleep(10)
    printed = run(["uvx", "--refresh-package", PACKAGE, f"{PACKAGE}@{version}", "--version"])
    if version not in printed:
        raise ReleaseError(f"`uvx {PACKAGE}@{version} --version` printed {printed!r}")
    assets = json.loads(run(["gh", "release", "view", tag_name(version), "--json", "assets"]))["assets"]
    if len(assets) < 2:
        raise ReleaseError(f"the GitHub Release {tag_name(version)} has {len(assets)} file(s), expected wheel + sdist")
    say(f"✓ {version}: on PyPI, runs from uvx, GitHub Release has {len(assets)} files")


def tag(yes: bool, dry_run: bool, consumers: list[str]) -> None:
    head = preflight_main()
    version = project_version(read("pyproject.toml"))
    released = changelog_versions(read("CHANGELOG.md"))
    if not released or released[0] != version:
        raise ReleaseError(f"pyproject.toml says {version}, the newest CHANGELOG heading says "
                           f"{released[0] if released else 'nothing'}: run `prepare` first")
    ref = tag_name(version)

    if remote_has_tag(ref):
        say(f"{ref} is already on the remote: verifying what it published")
    else:
        preflight_identity()  # the annotated tag records the tagger
        if run(["git", "tag", "--list", ref]):
            raise ReleaseError(f"{ref} exists locally but not on the remote: inspect it, then "
                               f"`git push origin {ref}` or `git tag -d {ref}`")
        if version in pypi_versions():
            raise ReleaseError(f"{version} is on PyPI but {ref} is not on the remote: investigate before tagging")
        wait_for_ci(head)
        say(f"Tag {ref} at {head[:7]} and publish {version} to PyPI. This cannot be undone.")
        if dry_run:
            say("(dry run: stopping before the tag)")
            sync_consumers(version, consumers, dry_run=True)
            return
        if not yes and not confirm("Publish?"):
            raise ReleaseError("not confirmed")
        act(dry_run, f"tag {ref}", ["git", "tag", "-a", ref, head, "-m", f"{PACKAGE} {version}"])
        act(dry_run, f"push {ref}", ["git", "push", "--quiet", "origin", ref])
        watch_publish(ref)

    verify_published(version, dry_run)
    sync_consumers(version, consumers, dry_run)


# --- sync-consumers --------------------------------------------------------------------------


def sync_consumers(version: str | None, extra: list[str], dry_run: bool) -> None:
    consumers = parse_consumers(os.environ.get(CONSUMERS_ENV, ""), extra)
    target = version or latest_release(pypi_versions())
    if not consumers:
        say(f"no consumers to update (set {CONSUMERS_ENV} or pass --consumer)")
    failures = []
    for consumer in consumers:
        try:
            say(f"✓ {consumer}: already on {check_consumer_locked(consumer, target)}")
            continue
        except ReleaseError:
            pass  # not on the target yet (or not a consumer: the re-check below says which)
        lock = consumer / "uv.lock"
        before = locked_version(lock.read_text()) if lock.is_file() else None
        try:
            # `-P pkg==X` pins exactly; outside the declared range uv refuses, and the
            # re-check turns that into a message naming the requirement to widen.
            act(dry_run, f"move {consumer} from {before} to {target}",
                ["uv", "lock", "--upgrade-package", f"{PACKAGE}=={target}", "--refresh-package", PACKAGE], consumer)
            act(dry_run, f"sync {consumer}", ["uv", "sync"], consumer)
        except ReleaseError as err:
            constraint = declared_constraint((consumer / "pyproject.toml").read_text()) \
                if (consumer / "pyproject.toml").is_file() else None
            failures.append(f"{consumer}: cannot lock {target} under `{PACKAGE}{constraint or ''}`; "
                            f"widen it deliberately if that is the cause.\n{err}")
            continue
        if not dry_run:
            try:
                check_consumer_locked(consumer, target)
                say(f"✓ {consumer}: {before} → {target} (uv.lock changed, not committed)")
            except ReleaseError as err:
                failures.append(str(err))
    running = running_bridge_version()
    if running and parse_version(running) < parse_version(target):
        port = os.environ.get(PORT_ENV) or DEFAULT_PORT
        say(f"! the bridge on :{port} runs {running}: restart it to pick up {target}")
    if failures:
        raise ReleaseError("\n".join(failures))


# --- CLI -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare", help="bump, CHANGELOG, gates, then the release PR")
    p_prepare.add_argument("spec", help="patch | minor | major | X.Y.Z")
    p_tag = sub.add_parser("tag", help="tag main, publish, verify, then sync consumers")
    p_tag.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")
    p_sync = sub.add_parser("sync-consumers", help="move each consumer's lock to a release")
    p_sync.add_argument("--version", help="the release to lock (default: the latest on PyPI)")
    for p in (p_tag, p_sync):
        p.add_argument("--consumer", action="append", default=[], metavar="DIR",
                       help=f"a project to move to the release, on top of ${CONSUMERS_ENV}")
    for p in (p_prepare, p_tag, p_sync):
        p.add_argument("--dry-run", action="store_true", help="preflight and plan; change nothing")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.spec, args.dry_run)
        elif args.command == "tag":
            tag(args.yes, args.dry_run, args.consumer)
        else:
            sync_consumers(args.version, args.consumer, args.dry_run)
    except ReleaseError as err:
        say(f"✗ {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
