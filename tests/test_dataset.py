import numpy as np

from lungscan3d.data.dataset import PatchDataset
from lungscan3d.data.splits import split_indices


def test_patch_dataset_returns_expected_shapes():
    volumes = np.zeros((4, 1, 32, 32, 32), dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    dataset = PatchDataset(volumes, labels)

    volume, label = dataset[1]

    assert len(dataset) == 4
    assert tuple(volume.shape) == (1, 32, 32, 32)
    assert tuple(label.shape) == (1,)
    assert label.item() == 1.0


def test_split_indices_are_reproducible():
    first = split_indices(100, 0.7, 0.15, 0.15, seed=42)
    second = split_indices(100, 0.7, 0.15, 0.15, seed=42)

    assert all(
        np.array_equal(left, right) for left, right in zip(first, second, strict=True)
    )
    assert sum(len(split) for split in first) == 100
