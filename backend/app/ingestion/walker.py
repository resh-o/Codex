"""File discovery + filtering.

Walks a checked-out repo and returns the source files worth chunking, skipping
vendored code, build output, and VCS metadata.
"""

from __future__ import annotations

from pathlib import Path

# Extensions we know how to chunk.  Kept in sync with the grammars wired up in
# ``chunker`` -- adding a language means touching both.
INCLUDE_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".ts", ".tsx", ".js", ".jsx"}
)

# Directory names to skip wholesale.  Made a module-level constant (not inlined)
# so it is trivial to extend for a new stack.  Any directory whose name starts
# with "." is also excluded (see ``_is_excluded_dir``).
EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        "venv",
        ".venv",
        "__pycache__",
        "dist",
        "build",
        ".next",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".idea",
        ".vscode",
        "site-packages",
        "vendor",
    }
)


def _is_excluded_dir(name: str) -> bool:
    """Whether a directory ``name`` should be pruned from the walk."""
    if name in EXCLUDE_DIRS:
        return True
    # Any dotfile directory (e.g. ".github", ".cache") is junk for our purposes.
    return name.startswith(".") and name not in {".", ".."}


def walk_repo(root: Path | str) -> list[str]:
    """Return source file paths under ``root``, relative to ``root``.

    Paths use POSIX ('/') separators regardless of platform, so downstream
    ``file_path`` values are stable and portable.

    Excluded: anything under a directory in :data:`EXCLUDE_DIRS` or any dotfile
    directory; anything without an extension in :data:`INCLUDE_EXTENSIONS`.
    Results are sorted for deterministic ordering.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    results: list[str] = []
    # os.walk-style pruning via Path: iterate directories ourselves so we can
    # prune excluded subtrees instead of walking into them.
    stack: list[Path] = [root_path]
    while stack:
        current = stack.pop()
        for entry in sorted(current.iterdir()):
            if entry.is_dir():
                if not _is_excluded_dir(entry.name):
                    stack.append(entry)
            elif entry.is_file() and entry.suffix in INCLUDE_EXTENSIONS:
                results.append(entry.relative_to(root_path).as_posix())

    results.sort()
    return results
