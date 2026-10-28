import datetime as dt

import numpy as np
import pytest

from dxquant.curves import ConstantShortRateCurve, InterpolatedZeroCurve


def test_constant_short_rate_curve_discount() -> None:
    ref = dt.date(2024, 1, 1)
    curve = ConstantShortRateCurve(name="const", reference_date=ref, rate=0.05)
    df = curve.discount_factor(dt.date(2025, 1, 1))
    delta = curve.year_fraction(dt.date(2025, 1, 1))
    expected = np.exp(-0.05 * delta)
    np.testing.assert_allclose(df, expected, rtol=1e-9)


def test_constant_forward_rate_symmetry() -> None:
    ref = dt.date(2024, 1, 1)
    curve = ConstantShortRateCurve(name="const", reference_date=ref, rate=0.03)
    fwd = curve.forward_rate(dt.date(2024, 7, 1), dt.date(2025, 7, 1))
    np.testing.assert_allclose(fwd, 0.03)


def test_interpolated_zero_curve_discount() -> None:
    ref = dt.date(2024, 1, 1)
    nodes = [
        (dt.date(2024, 7, 1), 0.01),
        (dt.date(2025, 1, 1), 0.015),
        (dt.date(2026, 1, 1), 0.02),
    ]
    curve = InterpolatedZeroCurve("zc", reference_date=ref, nodes=nodes)
    df = curve.discount_factor(dt.date(2025, 7, 1))
    t = curve._time_from_date(dt.date(2025, 7, 1))
    r = curve.zero_rate(dt.date(2025, 7, 1))
    np.testing.assert_allclose(df, np.exp(-r * t), rtol=1e-6)


def test_interpolated_curve_requires_two_nodes() -> None:
    ref = dt.date(2024, 1, 1)
    with pytest.raises(ValueError):
        InterpolatedZeroCurve("zc", reference_date=ref, nodes=[(ref, 0.01)])


def test_instantaneous_forward_rate_matches_derivative() -> None:
    ref = dt.date(2024, 1, 1)
    nodes = [
        (dt.date(2024, 7, 1), 0.02),
        (dt.date(2025, 1, 1), 0.03),
        (dt.date(2026, 1, 1), 0.04),
    ]
    curve = InterpolatedZeroCurve("zc", reference_date=ref, nodes=nodes)
    target = dt.date(2025, 7, 1)
    f_inst = curve.instantaneous_forward(target)
    forward_rate = curve.forward_rate(target, dt.date(2025, 7, 2))
    np.testing.assert_allclose(f_inst, forward_rate, rtol=1e-3)
