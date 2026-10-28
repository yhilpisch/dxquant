Numerical contracts
===================

DXQuant 0.2 is an unreleased alpha with explicit supported domains. The local
regression suite checks analytical identities, independent synthetic fixtures,
and representative stochastic regimes. It does not establish complete legacy
parity or production validation across all parameter combinations.

Dates and curves
----------------

Dates and datetimes represent calendar dates for year fractions; clock times
are discarded. Scalar ``year_fraction`` rejects reversed intervals. Vector
``year_fractions`` deliberately sorts and anchors at the earliest date. Curves,
swaps, and Monte Carlo accept ``DayCount`` or a finite positive denominator.
Choose consistent simulation/discount/accrual clocks explicitly.

Constant short rates remain nonnegative. Interpolated continuously compounded
zero rates may be negative down to -100%. Splines extrapolate by default;
``Spline1D(extrapolate=False)`` rejects out-of-domain queries. Linear
interpolation clamps to its endpoint values. These are explicit numerical
policies, not fitted arbitrage constraints.

The simple zero bootstrap uses a regular year-fraction coupon grid and requires
all prior coupon discount factors and enough nodes for interpolation. It is not
a general market-instrument bootstrap. Projection bootstrapping instead uses
anchored regular calendar coupons and requires a par quote at every coupon
maturity. Missing, duplicate, off-grid, or stub quotes are rejected. Covered
projection intervals compound log-linear accumulation on calendar-day
coordinates (including nonadditive 30/360 month ends);
extrapolation is rejected. Swaps optionally use separate projection forwards,
with projection and fixed accrual conventions applied to their respective legs.

Processes and exercise
----------------------

GBM/Merton increments are sampled in log space. Merton compounds every Poisson
arrival and compensates expected jump drift. Diffusion/count/mark streams are
separate. Seeds replay the current implementation; corrected implementations
intentionally differ from earlier seeded outputs.

Heston uses projected Euler variance and previous variance for the log-spot
increment. Its zero-vol-of-vol limit uses exact deterministic integrated
variance. CIR similarly uses projected Euler rates with exact deterministic
zero-sigma paths and stable bond analytics. Finite-step projection is an
approximation; compare grid refinements for the parameters being used.

The square-root jump process retains DXQuant's additive signed ``exp(Y) - 1`` marks,
no jump compensator, and projection at zero. It differs from the legacy
level-scaled jump model. The mean-reverting process has level-proportional
volatility, not additive-volatility Vasicek dynamics.

SABR samples lognormal volatility and uses its previous value in the forward
increment. Beta=1 uses a log update; beta=0 permits signed normal forwards;
intermediate beta uses an absorbing zero Euler boundary. The Hagan lognormal
volatility function is asymptotic and restricted to positive forward/strike/
alpha, beta in [0, 1], nonnegative expiry/vol-of-vol, and absolute rho below one.

American puts include immediate exercise and expiry. Quadratic in-sample
Longstaff-Schwartz regression remains a finite exercise-grid approximation.
Asian averages and barrier monitoring use the supplied grid, including inception;
barriers have no continuous-monitoring correction.

Monte Carlo uncertainty and sensitivities
-----------------------------------------

``MonteCarloResult.sampling_scheme`` identifies the error contract. IID paths
use the usual mean-error estimator; even antithetic paths use independent pair
means. Single moment-matched batches, unknown schemes, or insufficient/odd
pair counts report ``NaN`` rather than an IID standard error. Known deterministic
pricing returns zero. Independently seeded whole-batch replications can measure
moment-matched estimator dispersion. Sample-error estimates are not guarantees
of coverage in every tail/parameter regime.

Finite differences use common draws within a calculation, even when a caller's
engine has no seed. Temporary clones preserve the caller's process/seed. Bump
size remains a numerical choice; inspect stability across bumps and path counts.
The process drift is explicit and is not inferred from a discount curve.

Portfolio risk and currencies
-----------------------------

Portfolio PV sums quantity/FX-scaled values. Payoff VaR/ES is dispersion of
``price - discounted_payoff``; it is not future mark-to-market or historical
market-risk VaR. Stochastic multi-position risk needs matching path counts,
a common maturity/pricing date, and an explicit shared ``scenario_set``.
The caller certifies ordered joint scenarios and their dependence model.
A label neither creates nor validates correlations. Without it, aggregate risk
methods reject and risk-table aggregate metrics are ``NaN`` with metadata.

Risk-table and parallel Greeks include position quantity and FX in base-currency
units per local spot unit. Multi-asset/pathwise Greek coverage is explicitly
limited. A risk table records base currency and original position currency
separately. Scenario PnL subtracts the unshocked base value. No stochastic FX,
shared simulation service, or future mark-to-market risk engine is provided.

Fourier integration and optimization
------------------------------------

Carr-Madan calls require a normalized log-price characteristic function with
finite positive real first/damping moments. Controls and price bounds are
validated; quadrature/tail failures raise errors. Consecutive doubled-tail
checks diagnose cutoff sensitivity but are not a rigorous error bound for an
arbitrary characteristic function. Exact deterministic CFs from package factories
handle expiry/zero volatility directly. The CF carries its price level; ``spot``
is retained and validated for call-signature compatibility.

Mean-variance inputs must be finite, complete return observations with unique
asset columns and finite fully invested weights. Optimization is long-only;
manual portfolio statistics may use valid signed weights. Returns are copied
and statistics cached on construction; create a new portfolio for changed data.
Calibration requires finite matching target/model vectors and explicit valid
bounds. Identifiability and optimizer convergence require independent assessment.
