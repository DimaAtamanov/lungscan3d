"""Path utilities."""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Created or existing directory as a ``Path`` object.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
