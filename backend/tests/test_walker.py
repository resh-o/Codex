"""Tests for file discovery + filtering (`walker.walk_repo`)."""

from pathlib import Path

import pytest

from app.ingestion.walker import EXCLUDE_DIRS, INCLUDE_EXTENSIONS, walk_repo


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// content\n", encoding="utf-8")


@pytest.fixture()
def sample_tree(tmp_path: Path) -> Path:
    """A small repo-like tree mixing keepers and junk."""
    # Files we expect to keep.
    _touch(tmp_path / "app.py")
    _touch(tmp_path / "src" / "index.ts")
    _touch(tmp_path / "src" / "component.tsx")
    _touch(tmp_path / "src" / "util.js")
    _touch(tmp_path / "src" / "widget.jsx")

    # Files we expect to drop by extension.
    _touch(tmp_path / "README.md")
    _touch(tmp_path / "data.json")
    _touch(tmp_path / "styles.css")

    # Directories we expect to prune entirely.
    _touch(tmp_path / "node_modules" / "lib" / "index.js")
    _touch(tmp_path / ".git" / "config.py")
    _touch(tmp_path / ".venv" / "site.py")
    _touch(tmp_path / "__pycache__" / "cached.py")
    _touch(tmp_path / "dist" / "bundle.js")
    _touch(tmp_path / "build" / "out.js")
    _touch(tmp_path / ".next" / "server.js")
    _touch(tmp_path / ".github" / "workflow.js")  # dotfile dir
    return tmp_path


def test_includes_only_target_extensions(sample_tree: Path) -> None:
    result = set(walk_repo(sample_tree))
    assert result == {
        "app.py",
        "src/index.ts",
        "src/component.tsx",
        "src/util.js",
        "src/widget.jsx",
    }


def test_excludes_non_source_extensions(sample_tree: Path) -> None:
    result = walk_repo(sample_tree)
    for rel in result:
        assert Path(rel).suffix in INCLUDE_EXTENSIONS


def test_prunes_junk_directories(sample_tree: Path) -> None:
    result = walk_repo(sample_tree)
    joined = "\n".join(result)
    for junk in ["node_modules", ".git", ".venv", "__pycache__", "dist", "build", ".next", ".github"]:
        assert junk not in joined


def test_returns_posix_relative_paths(sample_tree: Path) -> None:
    result = walk_repo(sample_tree)
    assert all("\\" not in rel for rel in result)
    assert all(not Path(rel).is_absolute() for rel in result)


def test_results_are_sorted(sample_tree: Path) -> None:
    result = walk_repo(sample_tree)
    assert result == sorted(result)


def test_exclude_dirs_is_a_configurable_constant() -> None:
    # The exclude list is a module-level constant, not inlined, so it can be
    # extended without touching the walk logic.
    assert "node_modules" in EXCLUDE_DIRS
    assert isinstance(EXCLUDE_DIRS, frozenset)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        walk_repo(tmp_path / "does-not-exist")
