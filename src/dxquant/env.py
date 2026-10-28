# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Market environment containers tracking constants, lists, and curves.

This module supersedes ``market_environment`` from the legacy ``dx`` package
with dataclass-backed,
type-hinted storage plus inline comments for auditability.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field

from .curves import ConstantShortRateCurve, InterpolatedZeroCurve

CurveType = ConstantShortRateCurve | InterpolatedZeroCurve

__all__ = ["MarketEnvironment"]


@dataclass
class MarketEnvironment:
    """
    Container bundling pricing date, constants, lists, and curve objects.

    Parameters
    ----------
    name:
        Identifier for logging/reporting.
    pricing_date:
        Date associated with the environment snapshot.
    constants/lists/curves:
        Optional initial mappings; defaults to empty dictionaries.
    """

    name: str
    pricing_date: dt.date | dt.datetime
    constants: MutableMapping[str, float] = field(default_factory=dict)
    lists: MutableMapping[str, list[str]] = field(default_factory=dict)
    curves: MutableMapping[str, CurveType] = field(default_factory=dict)

    def add_constant(self, key: str, value: float) -> None:
        """Insert or update a scalar constant."""

        self.constants[key] = value

    def get_constant(self, key: str) -> float:
        """Return a named scalar; unknown names raise KeyError."""

        return self.constants[key]

    def add_list(self, key: str, values: Iterable[str]) -> None:
        """Store a list of identifiers (e.g., underlyings)."""

        self.lists[key] = list(values)

    def get_list(self, key: str) -> list[str]:
        """Return the stored list of identifiers for a key."""

        return self.lists[key]

    def add_curve(self, key: str, curve: CurveType) -> None:
        """Attach a discount/zero curve."""

        self.curves[key] = curve

    def get_curve(self, key: str) -> CurveType:
        """Return a named deterministic discount curve."""

        return self.curves[key]

    def merge(self, other: MarketEnvironment) -> None:
        """
        Merge another environment into this one, overriding duplicate keys.

        Useful when layering product-specific data on top of global market data.
        """

        self.constants.update(other.constants)
        self.lists.update(other.lists)
        self.curves.update(other.curves)

    def snapshot(self) -> Mapping[str, Mapping[str, object]]:
        """
        Return copied mappings; referenced curve objects are shared.
        """

        return {
            "constants": dict(self.constants),
            "lists": {key: list(values) for key, values in self.lists.items()},
            "curves": dict(self.curves),
        }
