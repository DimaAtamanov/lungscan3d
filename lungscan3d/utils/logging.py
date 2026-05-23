"""Logging helpers for command-line workflows."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure human-readable console logging.

    Args:
    ----
        level: Logging level name, for example ``INFO`` or ``DEBUG``.

    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
        return
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
