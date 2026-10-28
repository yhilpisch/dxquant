import datetime as dt

from dxquant.curves import ConstantShortRateCurve
from dxquant.env import MarketEnvironment


def test_market_environment_add_and_get() -> None:
    env = MarketEnvironment(name="test", pricing_date=dt.date(2024, 1, 1))
    env.add_constant("sigma", 0.2)
    env.add_list("assets", ["SPX", "NDX"])

    curve = ConstantShortRateCurve("disc", dt.date(2024, 1, 1), 0.03)
    env.add_curve("discount", curve)

    assert env.get_constant("sigma") == 0.2
    assert env.get_list("assets") == ["SPX", "NDX"]
    assert env.get_curve("discount") is curve


def test_market_environment_merge() -> None:
    env_a = MarketEnvironment(name="A", pricing_date=dt.date(2024, 1, 1))
    env_a.add_constant("vol", 0.3)

    env_b = MarketEnvironment(name="B", pricing_date=dt.date(2024, 1, 1))
    env_b.add_constant("vol", 0.2)
    env_b.add_list("assets", ["EURUSD"])

    env_a.merge(env_b)
    assert env_a.get_constant("vol") == 0.2  # overwritten
    assert env_a.get_list("assets") == ["EURUSD"]
