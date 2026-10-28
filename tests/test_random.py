import numpy as np

from dxquant.random import sn_random_numbers


def test_sn_random_numbers_shape_and_seed() -> None:
    """Random numbers should respect shape and seeding semantics."""

    out = sn_random_numbers((2, 3, 4), seed=42)
    assert out.shape == (2, 3, 4)

    twin = sn_random_numbers((2, 3, 4), seed=42)
    np.testing.assert_allclose(out, twin)  # deterministic when seed identical


def test_sn_random_numbers_antithetic_moment_matching() -> None:
    """Antithetic variates should yield near-zero mean and unit variance."""

    samples = sn_random_numbers(
        (1, 1000), antithetic=True, moment_matching=True, seed=7
    )
    samples = samples.squeeze(0)

    np.testing.assert_allclose(samples.mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(samples.std(ddof=0), 1.0, atol=1e-10)
