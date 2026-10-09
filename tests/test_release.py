"""The release script's pure parts, and the repo state it relies on.

The script itself is a maintainer tool, run by hand; what it computes — the next version,
the CHANGELOG rewrite, whether a consumer's lock moved — is checked here, so a release never
discovers a bug in it on tag day."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release", ROOT / "scripts" / "release.py")
assert _spec and _spec.loader
release = importlib.util.module_from_spec(_spec)
sys.modules["release"] = release
_spec.loader.exec_module(release)

REPO = "https://github.com/voidcraft-world/voidcraft-world-bridge"
CHANGELOG = f"""# Changelog

Intro.

## [Unreleased]

### Fixed

- A thing.

## [0.1.1] - 2026-10-08

### Added

- Older.

[Unreleased]: {REPO}/compare/v0.1.1...HEAD
[0.1.1]: {REPO}/compare/v0.1.0...v0.1.1
[0.1.0]: {REPO}/releases/tag/v0.1.0
"""


@pytest.mark.parametrize(("spec", "expected"), [
    ("patch", "0.1.2"), ("minor", "0.2.0"), ("major", "1.0.0"), ("0.3.0", "0.3.0"),
])
def test_bump_version(spec, expected):
    assert release.bump_version("0.1.1", spec) == expected


@pytest.mark.parametrize("spec", ["0.1.1", "0.1.0", "1.0", "v0.2.0", "next"])
def test_bump_version_refuses_a_non_increase_or_a_bad_spec(spec):
    with pytest.raises(release.ReleaseError):
        release.bump_version("0.1.1", spec)


def test_project_version_reads_and_writes_the_project_table_only():
    pyproject = '[project]\nname = "x"\nversion = "0.1.1"\n\n[tool.mypy]\npython_version = "3.10"\n'
    assert release.project_version(pyproject) == "0.1.1"
    bumped = release.set_project_version(pyproject, "0.2.0")
    assert release.project_version(bumped) == "0.2.0"
    assert 'python_version = "3.10"' in bumped


def test_rewrite_changelog_moves_unreleased_and_rewrites_links():
    out = release.rewrite_changelog(CHANGELOG, "0.1.2", "2026-10-10")
    assert release.changelog_versions(out) == ["0.1.2", "0.1.1"]
    assert release.unreleased_notes(out).strip() == ""
    assert "## [Unreleased]\n\n## [0.1.2] - 2026-10-10\n\n### Fixed\n\n- A thing.\n\n## [0.1.1]" in out
    assert f"[Unreleased]: {REPO}/compare/v0.1.2...HEAD\n[0.1.2]: {REPO}/compare/v0.1.1...v0.1.2\n" in out
    assert f"[0.1.1]: {REPO}/compare/v0.1.0...v0.1.1" in out  # older links untouched


def test_rewrite_changelog_refuses_an_empty_unreleased():
    empty = CHANGELOG.replace("### Fixed\n\n- A thing.\n\n", "")
    with pytest.raises(release.ReleaseError, match="empty"):
        release.rewrite_changelog(empty, "0.1.2", "2026-10-10")


def test_the_repo_is_release_consistent():
    """pyproject's version is the newest CHANGELOG heading and every heading has a link —
    the state `tag` checks, held on every PR instead of found on tag day."""
    changelog = (ROOT / "CHANGELOG.md").read_text()
    versions = release.changelog_versions(changelog)
    assert versions[0] == release.project_version((ROOT / "pyproject.toml").read_text())
    for version in versions:
        assert f"\n[{version}]: {REPO}/" in changelog, version
    assert release.UNRELEASED_LINK_RE.search(changelog).group(2) == versions[0]


LOCK = """version = 1

[[package]]
name = "voidcraft-world-bridge"
version = "{v}"
source = {{ registry = "https://pypi.org/simple" }}
"""


def _consumer(tmp_path, constraint=">=0.1,<0.2", locked="0.1.0"):
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "host"\ndependencies = [\n    "voidcraft-world-bridge{constraint}",\n]\n')
    (tmp_path / "uv.lock").write_text(LOCK.format(v=locked))
    return tmp_path


def test_lock_and_constraint_readers(tmp_path):
    consumer = _consumer(tmp_path)
    assert release.locked_version((consumer / "uv.lock").read_text()) == "0.1.0"
    assert release.declared_constraint((consumer / "pyproject.toml").read_text()) == ">=0.1,<0.2"
    assert release.locked_version('[[package]]\nname = "other"\nversion = "1.0.0"\n') is None


def test_a_consumer_on_the_release_passes(tmp_path):
    assert release.check_consumer_locked(_consumer(tmp_path, locked="0.1.1"), "0.1.1") == "0.1.1"


def test_a_consumer_left_behind_names_its_requirement(tmp_path):
    # What `uv lock -P` leaves when 0.2.0 is outside `<0.2`: the old pin.
    with pytest.raises(release.ReleaseError, match=r"locks 0\.1\.0, not 0\.2\.0.*>=0\.1,<0\.2"):
        release.check_consumer_locked(_consumer(tmp_path), "0.2.0")


def test_a_directory_that_does_not_depend_on_the_bridge_is_refused(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n')
    (tmp_path / "uv.lock").write_text("version = 1\n")
    with pytest.raises(release.ReleaseError, match="does not depend"):
        release.check_consumer_locked(tmp_path, "0.1.1")


def test_parse_consumers_expands_dedupes_and_keeps_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    env = f"{a}{release.os.pathsep}{release.os.pathsep} {b} {release.os.pathsep}{a}"
    assert release.parse_consumers(env, [str(b), str(tmp_path / "c")]) == [a, b, tmp_path / "c"]
    assert release.parse_consumers("") == []


def test_main_dry_run_refuses_cleanly(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise release.ReleaseError("stop here")

    monkeypatch.setattr(release, "preflight_main", boom)
    assert release.main(["tag", "--dry-run"]) == 1
    assert "✗ stop here" in capsys.readouterr().out
