import numpy as np
from lungscan3d.training.hard_negative_mining import select_hard_negative_indices


def test_select_hard_negative_indices_keeps_high_probability_negatives():
    labels = np.array([0, 0, 1, 0, 1], dtype=np.float32)
    probabilities = np.array([0.9, 0.2, 0.8, 0.7, 0.1], dtype=np.float32)

    indices = select_hard_negative_indices(
        labels=labels,
        probabilities=probabilities,
        top_fraction=0.5,
        min_probability=0.5,
    )

    assert indices.tolist() == [0]
