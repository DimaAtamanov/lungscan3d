"""Split helpers."""

import numpy as np


def split_indices(
    num_items: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train/validation/test indices.

    Args:
    ----
        num_items: Number of items to split.
        train_fraction: Fraction assigned to train split.
        val_fraction: Fraction assigned to validation split.
        test_fraction: Fraction assigned to test split.
        seed: Random seed.

    Returns:
    -------
        Train, validation, and test index arrays.

    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    train_end = int(num_items * train_fraction)
    val_end = train_end + int(num_items * val_fraction)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]
