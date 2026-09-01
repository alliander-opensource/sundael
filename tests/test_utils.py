# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Test utility functions."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from sundael.utils import mean_gt_zero, np_apply_along_axis


@pytest.fixture
def a() -> npt.NDArray[np.float64]:
    """Test array fixture."""
    return np.random.rand(5, 3)


@pytest.mark.parametrize("axis", (0, 1))
def test_np_max(a: npt.NDArray[np.float64], axis: Literal[0, 1]) -> None:
    """Test Python version of numba-accelerated max()."""
    np.testing.assert_equal(np_apply_along_axis.py_func(np.max, axis, a), np.max(a, axis))


@pytest.mark.parametrize("axis", (0, 1))
def test_np_min(a: npt.NDArray[np.float64], axis: Literal[0, 1]) -> None:
    """Test Python version of numba-accelerated min()."""
    np.testing.assert_equal(np_apply_along_axis.py_func(np.min, axis, a), np.min(a, axis))


@pytest.mark.parametrize("axis,expected", ((0, [2.0, 10, np.nan]), (1, [10.0, 2.0])))
def test_mean_lt_zero(axis: Literal[0, 1], expected: list[float]) -> None:
    """Test calculation of mean without taking zero into account."""
    df = pd.DataFrame([[-1, 10, 0], [2, 0, 0]])
    np.testing.assert_array_almost_equal(mean_gt_zero(df, axis=axis), expected)
