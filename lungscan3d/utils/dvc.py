"""DVC integration helpers."""

import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _run_dvc_command(command: list[str]) -> bool:
    """Run a DVC command and report whether it succeeded.

    Args:
    ----
        command: Full command line as a list of arguments.

    Returns:
    -------
        ``True`` when DVC completed successfully, otherwise ``False``.

    """
    LOGGER.info("Running DVC command: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        LOGGER.warning("DVC command failed: %s", error)
        return False
    LOGGER.info("DVC command finished successfully")
    return True


def dvc_pull(target: str | Path | None = None, remote: str | None = None) -> bool:
    """Pull data or model artifacts via DVC.

    Args:
    ----
        target: Optional DVC target to pull.
        remote: Optional DVC remote name.

    Returns:
    -------
        ``True`` when DVC completed successfully, otherwise ``False``.

    """
    command = ["dvc", "pull"]
    if remote is not None:
        command.extend(["-r", remote])
    if target is not None:
        command.append(str(target))
    return _run_dvc_command(command)


def dvc_push(target: str | Path | None = None, remote: str | None = None) -> bool:
    """Push data or model artifacts via DVC.

    Args:
    ----
        target: Optional DVC target to push.
        remote: Optional DVC remote name.

    Returns:
    -------
        ``True`` when DVC completed successfully, otherwise ``False``.

    """
    command = ["dvc", "push"]
    if remote is not None:
        command.extend(["-r", remote])
    if target is not None:
        command.append(str(target))
    return _run_dvc_command(command)


def dvc_add(target: str | Path) -> bool:
    """Add a target to DVC tracking.

    Args:
    ----
        target: File or directory to track with DVC.

    Returns:
    -------
        ``True`` when DVC completed successfully, otherwise ``False``.

    """
    return _run_dvc_command(["dvc", "add", str(target)])
