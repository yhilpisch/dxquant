import datetime as dt

import numpy as np
import pytest

from dxquant import time


def test_ensure_datetime_array_handles_dates_and_datetimes() -> None:
    """Dates should be promoted to datetimes and sorted."""

    times = [
        dt.date(2024, 1, 3),
        dt.datetime(2024, 1, 1, 12, 0),
        dt.date(2024, 1, 2),
    ]
    arr = time.ensure_datetime_array(times)

    assert arr[0] == dt.datetime(2024, 1, 1, 12, 0)  # earliest entry
    assert arr[-1] == dt.datetime(2024, 1, 3, 0, 0)  # promoted date


def test_year_fractions_matches_expected() -> None:
    """Year fractions should align with ACT/365 day-count conventions."""

    origin = dt.date(2024, 1, 1)
    dates = [
        origin,
        origin + dt.timedelta(days=30),
        origin + dt.timedelta(days=60),
    ]
    fractions = time.year_fractions(dates, day_count=365.0)

    expected = np.array([0.0, 30 / 365.0, 60 / 365.0])
    np.testing.assert_allclose(fractions, expected)


def test_uniform_time_grid() -> None:
    grid = time.uniform_time_grid(1.0, 4)
    np.testing.assert_allclose(grid, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_year_fraction_act_365_default() -> None:
    start = dt.date(2024, 1, 1)
    end = dt.date(2025, 1, 1)
    assert time.year_fraction(start, end) == pytest.approx(366 / 365.0)


def test_year_fraction_act_360() -> None:
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 4, 1)  # 91 days
    assert time.year_fraction(
        start, end, time.DayCount.ACT_360
    ) == pytest.approx(91 / 360.0)


def test_year_fraction_thirty_360() -> None:
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 7, 1)  # exactly 6 months on bond basis -> 0.5
    assert time.year_fraction(
        start, end, time.DayCount.THIRTY_360
    ) == pytest.approx(0.5)


def test_year_fraction_accepts_raw_denominator_and_datetime() -> None:
    start = dt.datetime(2024, 1, 1, 12, 0)
    end = dt.date(2024, 1, 31)
    assert time.year_fraction(start, end, 365.25) == pytest.approx(30 / 365.25)


def test_year_fraction_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        time.year_fraction(dt.date(2024, 2, 1), dt.date(2024, 1, 1))
