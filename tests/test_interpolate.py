import numpy as np
import pytest

from dxquant.interpolate import Spline1D, linear_interpolate


def test_linear_interpolate_matches_line() -> None:
    x = [0.0, 1.0, 2.0]
    y = [0.0, 2.0, 4.0]
    pts = [0.5, 1.5]
    np.testing.assert_allclose(linear_interpolate(x, y, pts), [1.0, 3.0])


def test_spline1d_values_and_derivatives() -> None:
    x = [0.0, 1.0, 2.0, 3.0]
    y = [value**2 for value in x]
    spline = Spline1D(x, y)

    vals = spline.value([1.5])
    ders = spline.derivative([1.5])
    np.testing.assert_allclose(vals, [2.25], rtol=1e-6)
    np.testing.assert_allclose(ders, [3.0], rtol=1e-6)


def test_spline_reduces_degree_when_insufficient_points() -> None:
    spline = Spline1D([0.0, 1.0], [0.0, 1.0], degree=3)
    assert spline.degree == 1  # downgraded to linear


def test_spline_raises_on_unsorted_x() -> None:
    with pytest.raises(ValueError):
        Spline1D([0.0, 0.0], [1.0, 1.0])
