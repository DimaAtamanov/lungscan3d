"""Split helpers."""

import json
from pathlib import Path

import numpy as np


def _validate_split_fractions(
    train_fraction: float, val_fraction: float, test_fraction: float
) -> None:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")


def split_indices(
    num_items: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train/validation/test indices by rows."""
    _validate_split_fractions(train_fraction, val_fraction, test_fraction)
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    train_end = int(num_items * train_fraction)
    val_end = train_end + int(num_items * val_fraction)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def split_indices_by_group(
    groups: np.ndarray | list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create row indices while keeping each patient/series in one split only.

    Args:
    ----
        groups: One group id per sample, normally LUNA16 ``seriesuid`` values.
        train_fraction: Approximate fraction of unique groups assigned to train.
        val_fraction: Approximate fraction of unique groups assigned to validation.
        test_fraction: Approximate fraction of unique groups assigned to test.
        seed: Random seed.

    Returns:
    -------
        Train, validation, and test row indices. No group value appears in more
        than one output split.

    """
    _validate_split_fractions(train_fraction, val_fraction, test_fraction)
    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        raise ValueError("groups must be a one-dimensional array")
    if len(group_array) == 0:
        raise ValueError("groups must not be empty")

    unique_groups = np.unique(group_array.astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    train_end = int(len(unique_groups) * train_fraction)
    val_end = train_end + int(len(unique_groups) * val_fraction)
    train_groups = set(unique_groups[:train_end])
    val_groups = set(unique_groups[train_end:val_end])
    test_groups = set(unique_groups[val_end:])

    group_strings = group_array.astype(str)
    train_idx = np.flatnonzero(np.isin(group_strings, list(train_groups)))
    val_idx = np.flatnonzero(np.isin(group_strings, list(val_groups)))
    test_idx = np.flatnonzero(np.isin(group_strings, list(test_groups)))
    return train_idx, val_idx, test_idx


def assert_disjoint_groups(
    groups: np.ndarray | list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Raise when a group leaks across train/validation/test splits."""
    group_array = np.asarray(groups).astype(str)
    split_groups = [
        set(group_array[train_idx]),
        set(group_array[val_idx]),
        set(group_array[test_idx]),
    ]
    if split_groups[0] & split_groups[1]:
        raise ValueError("Patient/group leakage between train and validation splits")
    if split_groups[0] & split_groups[2]:
        raise ValueError("Patient/group leakage between train and test splits")
    if split_groups[1] & split_groups[2]:
        raise ValueError("Patient/group leakage between validation and test splits")


def save_split_indices(
    output_dir: str | Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray | list[str] | None = None,
) -> None:
    """Persist split indices and optional group ids for reproducibility."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "train_idx.npy", np.asarray(train_idx, dtype=np.int64))
    np.save(directory / "val_idx.npy", np.asarray(val_idx, dtype=np.int64))
    np.save(directory / "test_idx.npy", np.asarray(test_idx, dtype=np.int64))
    if groups is None:
        return
    group_array = np.asarray(groups).astype(str)
    summary = {
        "train_groups": sorted(set(group_array[train_idx])),
        "val_groups": sorted(set(group_array[val_idx])),
        "test_groups": sorted(set(group_array[test_idx])),
    }
    (directory / "groups.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
