"""Repo cloning / updating via GitPython.

Wraps GitPython so callers get clean, typed exceptions instead of raw
``git.exc`` internals leaking up through the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import git
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

logger = logging.getLogger(__name__)

# Where clones live by default.  Overridable per-call and via env var so the
# Docker container and local runs can point somewhere writable.
DEFAULT_WORKDIR = Path(os.environ.get("CODEX_WORKDIR", Path.cwd() / ".codex_repos"))


class CloneError(RuntimeError):
    """Raised when a repo cannot be cloned or updated.

    Carries a human-readable message; the original GitPython exception (if any)
    is attached as ``__cause__`` for debugging.
    """


@dataclass(frozen=True)
class CloneResult:
    """Outcome of a successful clone/pull."""

    local_path: Path
    commit_sha: str
    repo_url: str


def _slug_for(repo_url: str) -> str:
    """Derive a stable, filesystem-safe directory name for a repo URL.

    Uses the last path segment plus a short hash of the full URL so that two
    repos with the same name from different hosts/owners don't collide.
    """
    parsed = urlparse(repo_url)
    tail = parsed.path.rstrip("/").split("/")[-1] or "repo"
    tail = re.sub(r"\.git$", "", tail)
    tail = re.sub(r"[^A-Za-z0-9._-]", "-", tail) or "repo"
    digest = hashlib.sha1(repo_url.encode("utf-8")).hexdigest()[:10]
    return f"{tail}-{digest}"


def clone_or_update(
    repo_url: str,
    workdir: Path | str | None = None,
    depth: int | None = 1,
) -> CloneResult:
    """Clone ``repo_url`` into ``workdir``, or pull if it already exists.

    Parameters
    ----------
    repo_url:
        A git-cloneable URL (https or ssh).
    workdir:
        Directory under which the repo is checked out.  Defaults to
        :data:`DEFAULT_WORKDIR`.  The repo lands in ``workdir/<slug>``.
    depth:
        Shallow-clone depth.  ``1`` (default) is plenty for chunking a snapshot;
        pass ``None`` for a full history clone.

    Returns
    -------
    CloneResult
        Local path and the resolved HEAD commit SHA.

    Raises
    ------
    CloneError
        For any git-level failure (invalid URL, auth, network, etc.).
    """
    if not repo_url or not repo_url.strip():
        raise CloneError("repo_url must be a non-empty string")

    base = Path(workdir) if workdir is not None else DEFAULT_WORKDIR
    base.mkdir(parents=True, exist_ok=True)
    dest = base / _slug_for(repo_url)

    try:
        if dest.exists():
            repo = _pull_existing(dest, repo_url)
        else:
            logger.info("Cloning %s -> %s", repo_url, dest)
            clone_kwargs: dict = {}
            if depth is not None:
                clone_kwargs["depth"] = depth
            repo = git.Repo.clone_from(repo_url, dest, **clone_kwargs)

        commit_sha = repo.head.commit.hexsha
    except GitCommandError as exc:
        raise CloneError(_explain_git_error(repo_url, exc)) from exc
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise CloneError(
            f"Local path for {repo_url!r} is not a valid git repository: {exc}"
        ) from exc
    except CloneError:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise CloneError(f"Unexpected error cloning {repo_url!r}: {exc}") from exc

    return CloneResult(local_path=dest, commit_sha=commit_sha, repo_url=repo_url)


def _pull_existing(dest: Path, repo_url: str) -> git.Repo:
    """Open an existing checkout and fast-forward it to the remote HEAD."""
    try:
        repo = git.Repo(dest)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise CloneError(
            f"Existing path {dest} is not a git repo; remove it and retry: {exc}"
        ) from exc

    logger.info("Updating existing checkout at %s", dest)
    try:
        repo.remotes.origin.pull()
    except (GitCommandError, AttributeError) as exc:
        # AttributeError guards against a repo with no 'origin' remote.
        raise CloneError(
            f"Failed to pull latest changes for {repo_url!r}: {exc}"
        ) from exc
    return repo


def _explain_git_error(repo_url: str, exc: GitCommandError) -> str:
    """Turn a raw GitCommandError into a friendlier, actionable message."""
    stderr = (exc.stderr or "").lower()
    if any(s in stderr for s in ("could not resolve host", "network", "timed out")):
        reason = "network error (could not reach the host)"
    elif any(s in stderr for s in ("authentication failed", "permission denied", "403")):
        reason = "authentication failed (private repo or bad credentials)"
    elif any(s in stderr for s in ("not found", "repository not found", "does not exist", "404")):
        reason = "repository not found (check the URL)"
    else:
        reason = "git command failed"
    return f"Could not clone {repo_url!r}: {reason}."
