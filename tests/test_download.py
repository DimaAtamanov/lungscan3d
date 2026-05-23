"""Tests for data download helpers."""

import pytest

from lungscan3d.data.download import parse_subset_selection


def test_parse_subset_selection_from_string() -> None:
    assert parse_subset_selection(subsets="0,2,2", max_subsets=None) == [0, 2]


def test_parse_subset_selection_from_max_subsets() -> None:
    assert parse_subset_selection(subsets=None, max_subsets=3) == [0, 1, 2]


def test_parse_subset_selection_rejects_invalid_id() -> None:
    with pytest.raises(ValueError, match="subset ids"):
        parse_subset_selection(subsets="10", max_subsets=None)
