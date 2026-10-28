import datetime as dt

import pytest

from dxquant.calendar import BusinessDayCalendar


@pytest.fixture(scope="module")
def calendar() -> BusinessDayCalendar:
    """NYC-style calendar with a custom holiday (2024-07-04)."""

    return BusinessDayCalendar(
        name="NYC",
        holidays=[dt.date(2024, 7, 4)],
    )


def test_is_business_day(calendar: BusinessDayCalendar) -> None:
    assert calendar.is_business_day(dt.date(2024, 7, 3))  # regular Wednesday
    assert not calendar.is_business_day(dt.date(2024, 7, 4))  # custom holiday
    assert not calendar.is_business_day(
        dt.date(2024, 7, 6)
    )  # Saturday via weekmask


def test_add_business_days(calendar: BusinessDayCalendar) -> None:
    start = dt.date(2024, 7, 3)
    shifted = calendar.add_business_days(start, 1)
    assert shifted == dt.date(2024, 7, 5)  # July 4th skipped


def test_business_days_between(calendar: BusinessDayCalendar) -> None:
    start = dt.date(2024, 7, 1)
    end = dt.date(2024, 7, 5)
    assert calendar.business_days_between(start, end) == 3  # excludes end date
    assert calendar.business_days_between(start, end, inclusive=True) == 4


def test_schedule(calendar: BusinessDayCalendar) -> None:
    dates = calendar.schedule(dt.date(2024, 7, 1), dt.date(2024, 7, 8))
    assert dates[0] == dt.date(2024, 7, 1)
    assert dates[-1] == dt.date(2024, 7, 8)
    assert dt.date(2024, 7, 4) not in dates  # holiday skipped
