# DXQuant tutorials

Open a tutorial below and select **Runtime → Run all** in Google Colab.
Use a standard CPU runtime. The first code cell installs DXQuant and Matplotlib
if needed; the remaining cells run with synthetic inputs. No GPU, Drive mount,
API credentials, or manual path edits are required. Internet access is needed
for installation.

| Tutorial | Google Colab |
|---|---|
| [Notebook 0 · Quickstart](00_dxquant_quickstart.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/00_dxquant_quickstart.ipynb) |
| [Notebook 1 · Introduction](01_dxquant_introduction.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/01_dxquant_introduction.ipynb) |
| [Notebook 2 · Monte Carlo Pricing and Greeks](02_dxquant_monte_carlo_and_greeks.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/02_dxquant_monte_carlo_and_greeks.ipynb) |
| [Notebook 3 · Calibration and Fourier Pricing](03_dxquant_calibration_and_fourier.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/03_dxquant_calibration_and_fourier.ipynb) |
| [Notebook 4 · Curves, Swaps, and CIR Short Rates](04_dxquant_rates_curves_and_swaps.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/04_dxquant_rates_curves_and_swaps.ipynb) |
| [Notebook 5 · Multi-Asset Models and Basket Options](05_dxquant_multi_asset_and_baskets.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/05_dxquant_multi_asset_and_baskets.ipynb) |
| [Notebook 6 · Heston Pricing: Fourier vs Monte Carlo](06_dxquant_heston_fourier_and_monte_carlo.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/06_dxquant_heston_fourier_and_monte_carlo.ipynb) |
| [Notebook 7 · Portfolio Risk and FX Conversion](07_dxquant_portfolio_risk_and_fx.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/07_dxquant_portfolio_risk_and_fx.ipynb) |
| [Notebook 8 · Time Grids and Business-Day Calendars](08_dxquant_time_and_calendars.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/08_dxquant_time_and_calendars.ipynb) |
| [Notebook 9 · Random Numbers and Variance Reduction](09_dxquant_random_numbers_and_variance_reduction.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/09_dxquant_random_numbers_and_variance_reduction.ipynb) |
| [Notebook 10 · Process Zoo and Simulation Diagnostics](10_dxquant_processes_zoo.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/10_dxquant_processes_zoo.ipynb) |
| [Notebook 11 · Interpolation and Curve Queries](11_dxquant_interpolation_and_curves.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/11_dxquant_interpolation_and_curves.ipynb) |
| [Notebook 12 · Validation and Failure Modes](12_dxquant_validation_and_failure_modes.ipynb) | [Open in Colab](https://colab.research.google.com/github/yhilpisch/dxquant/blob/main/notebooks/12_dxquant_validation_and_failure_modes.ipynb) |

Before the first PyPI release, setup installs the public GitHub `main` branch
from the repository root. The maintainer confirmed a complete
Quickstart run on Colab on 2026-10-28. The remaining 12 tutorials still need
hosted-runtime checks; the local execution evidence is recorded with the headless helper below.

For local Jupyter, open a notebook from a checkout. If DXQuant is absent, the
same setup cell installs the local package. An existing DXQuant installation
is reused, so start with a fresh runtime/environment to test the current source.
Plotting examples keep their `ENABLE_PLOTS` switch; set it to `True` to show plots.

## Validate all tutorials headlessly

After cloning the repository, run these commands from its root. The helper
lives at `tools/validate_colab.py`:

```bash
python tools/validate_colab.py
python tools/validate_colab.py --execute
```

The default performs formal checks only: the exact 13-notebook inventory,
notebook schemas, unique IDs when present, cleared outputs/execution counts,
Python syntax, line width/whitespace, consistent installation cells, correct
Colab links, and absence of execution-bypass tags. It does not install packages
or run notebook cells.

`--execute` first requires all formal checks to pass, installs the cloned
package and its runtime/plot dependencies into the current Python environment,
then executes every notebook in a fresh kernel of that interpreter. It enables
plots in memory, uses a headless backend, and continues with later notebooks if
one fails. Source notebooks are never overwritten. A failed check, notebook
error, skipped code cell, or timeout produces a nonzero exit status.

To retain versions, revision/dirty status, source hashes, timings, and errors in JSON:

```bash
python tools/validate_colab.py --execute --timeout 300 \
    --report /tmp/dxquant-colab-report.json
```

`--timeout` is a positive limit in seconds **per cell**, not per notebook.
If runner dependencies are missing, install them with
`python -m pip install nbformat nbclient ipykernel`.

For a Colab notebook cell, assuming the clone is `/content/dxquant`:

```python
import subprocess
import sys

validator = "/content/dxquant/tools/validate_colab.py"
subprocess.run([sys.executable, validator], check=True)
subprocess.run(
    [sys.executable, validator, "--execute", "--report",
     "/content/dxquant-colab-report.json"],
    check=True,
)
```

Run these as separate optional cells after cloning. They are deliberately not
part of Quickstart's Run all sequence, which would recursively launch the
validator when it executes Quickstart. Download the report from Colab's Files
pane to preserve evidence after the runtime ends. The helper's local success
is not a claim that all tutorials have already passed on hosted Colab.
