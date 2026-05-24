"""Git metadata helpers."""

import subprocess
from pathlib import Path


def get_git_commit(repo_dir: str | Path = ".") -> str:
    """Return the current git commit hash when available.

    Args:
    ----
        repo_dir: Repository directory.

    Returns:
    -------
        Git commit hash or ``unknown`` when git metadata is not available.

    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
