import numpy as np
import pytest

from lungscan3d.data.splits import assert_disjoint_groups, split_indices_by_group


def test_split_indices_by_group_prevents_patient_leakage():
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"])

    train_idx, val_idx, test_idx = split_indices_by_group(groups, 0.6, 0.2, 0.2, seed=7)

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(groups)
    assert_disjoint_groups(groups, train_idx, val_idx, test_idx)


def test_assert_disjoint_groups_rejects_leakage():
    groups = np.array(["p1", "p1", "p2"])

    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_groups(groups, np.array([0]), np.array([1]), np.array([2]))
