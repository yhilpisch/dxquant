import numpy as np

from dxquant.calibration import (
    LeastSquaresCalibration,
    calibrate_least_squares,
    solve_scalar,
)
from dxquant.fourier import FourierPricer
from dxquant.processes import HestonProcess


def test_solve_scalar_brent() -> None:
    root = solve_scalar(lambda x: x**2 - 2, bracket=(0.0, 2.0))
    np.testing.assert_allclose(root, np.sqrt(2), rtol=1e-10)


def test_least_squares_calibration() -> None:
    target = np.array([1.0, 4.0, 9.0])

    def model(params: np.ndarray) -> np.ndarray:
        a, b = params
        x = np.array([1.0, 2.0, 3.0])
        return np.asarray(a * x**2 + b, dtype=float)

    result = calibrate_least_squares(model, target, guess=[1.0, 0.0])
    np.testing.assert_allclose(result, np.array([1.0, 0.0]), rtol=1e-9)

    lsq = LeastSquaresCalibration(
        target=target, model=model, initial_parameters=np.array([0.8, 0.2])
    )
    fitted = lsq.run()
    np.testing.assert_allclose(
        fitted.x, np.array([1.0, 0.0]), rtol=1e-9, atol=1e-12
    )


def test_heston_parameter_recovery_from_synthetic_smile() -> None:
    spot = 100.0
    rate = 0.01
    maturity = 1.0
    true_params = np.array([2.0, 0.04, 0.5, -0.4])
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)

    def price_surface(params: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, rho = params
        heston = HestonProcess(
            kappa=kappa, theta=theta, sigma=sigma, rho=rho, drift=rate
        )
        phi = heston.characteristic_function(
            tau=maturity, spot=spot, variance=theta
        )
        discount = np.exp(-rate * maturity)
        fp = FourierPricer(dampening=1.2, integration_limit=150.0, steps=6000)
        return np.asarray(
            [fp.price_call(phi, spot, strike, discount) for strike in strikes],
            dtype=float,
        )

    target_prices = price_surface(true_params)
    lower = [1e-4, 1e-4, 1e-4, -0.99]
    upper = [5.0, 1.0, 2.0, 0.99]
    recovered = calibrate_least_squares(
        price_surface,
        target_prices,
        guess=[1.5, 0.03, 0.6, -0.2],
        bounds=(lower, upper),
    )
    np.testing.assert_allclose(recovered, true_params, rtol=2e-1, atol=5e-3)
