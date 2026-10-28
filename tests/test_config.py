"""
Smoke tests for the top-level DXQuant package.
"""

from dxquant import DXAConfig, __version__, build_default_config


def test_version_semver_prefix() -> None:
    """
    Version strings should follow a semantic prefix to simplify release
    automation.
    """

    assert __version__.startswith("0."), (
        __version__
    )  # guard against accidental version bumps


def test_build_default_config_is_reproducible() -> None:
    """Default config helper should always return the same baseline object."""

    cfg = build_default_config()
    assert isinstance(cfg, DXAConfig)
    assert cfg.rng_seed is None  # engines decide whether to enforce seeding
    assert cfg.metadata == {}  # explicit empty metadata for clarity
