#!/usr/bin/env python3
"""Validate DXQuant tutorials statically or execute them headlessly.

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-Powered by different LLMs

Usage: python tools/validate_colab.py [--execute] [--report PATH]
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from nbformat import NotebookNode

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (
    "00_dxquant_quickstart.ipynb",
    "01_dxquant_introduction.ipynb",
    "02_dxquant_monte_carlo_and_greeks.ipynb",
    "03_dxquant_calibration_and_fourier.ipynb",
    "04_dxquant_rates_curves_and_swaps.ipynb",
    "05_dxquant_multi_asset_and_baskets.ipynb",
    "06_dxquant_heston_fourier_and_monte_carlo.ipynb",
    "07_dxquant_portfolio_risk_and_fx.ipynb",
    "08_dxquant_time_and_calendars.ipynb",
    "09_dxquant_random_numbers_and_variance_reduction.ipynb",
    "10_dxquant_processes_zoo.ipynb",
    "11_dxquant_interpolation_and_curves.ipynb",
    "12_dxquant_validation_and_failure_modes.ipynb",
)
SOURCE = (
    "dxquant @ git+https://github.com/yhilpisch/dxquant.git@main"
)
COLAB = "https://colab.research.google.com/github/yhilpisch/dxquant/"


@dataclass
class Result:
    """Record a notebook's checks and any failures."""

    notebook: str
    phase: str
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def validate(path: Path) -> NotebookNode:
    """Check schema, clean cells, syntax, and the tutorial's Colab contract."""
    import nbformat

    raw = json.loads(path.read_text(encoding="utf-8"))
    identifiers = [c["id"] for c in raw.get("cells", []) if "id" in c]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate cell IDs")
    nbformat.validate(raw)
    if raw["nbformat"] != 4:
        raise ValueError("Notebook format 4 required")
    notebook = cast(
        "NotebookNode", nbformat.reads(json.dumps(raw), as_version=4)
    )
    code = [c for c in notebook.cells if c.cell_type == "code"]
    if not code:
        raise ValueError("No executable cells")
    setup = ast.parse(code[0].source)
    constants = [
        n.value for n in ast.walk(setup) if isinstance(n, ast.Constant)
    ]
    if SOURCE not in constants or "matplotlib>=3.8" not in constants:
        raise ValueError("First code cell must set up DXQuant/Matplotlib")
    if not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "check_call"
        for n in ast.walk(setup)
    ):
        raise ValueError("Setup cell has no installation command")
    badge = f"{COLAB}blob/main/notebooks/{path.name}"
    if not any(
        badge in c.source for c in notebook.cells if c.cell_type == "markdown"
    ):
        raise ValueError("Missing or incorrect Colab link")
    for number, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            continue
        if cell.execution_count is not None or cell.outputs:
            raise ValueError(f"Cell {number}: saved execution/output")
        tags = set(cell.metadata.get("tags", []))
        if tags & {"skip-execution", "raises-exception"}:
            raise ValueError(f"Cell {number}: execution-bypass tag")
        for line_number, line in enumerate(cell.source.splitlines(), 1):
            if len(line) > 80 or line.rstrip() != line or "\t" in line:
                raise ValueError(
                    f"Cell {number}, line {line_number}: width/whitespace"
                )
        compile(cell.source, f"{path.name}:cell-{number}", "exec")
    return notebook


def execute(notebook: NotebookNode, timeout: int) -> None:
    """Run all cells with plotting in a fresh kernel of this interpreter."""
    from nbclient import NotebookClient

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace(
                "ENABLE_PLOTS = False", "ENABLE_PLOTS = True"
            )
    with tempfile.TemporaryDirectory(prefix="dxquant-notebook-") as temp:
        env = dict(
            os.environ,
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(Path(temp) / "matplotlib"),
            IPYTHONDIR=str(Path(temp) / "ipython"),
        )
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=False,
            force_raise_errors=True,
            record_timing=False,
            resources={"metadata": {"path": str(ROOT / "notebooks")}},
        )
        client.km = client.create_kernel_manager()
        spec = client.km.kernel_spec
        if spec is None:
            raise RuntimeError("Python kernel specification unavailable")
        spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client.execute(env=env)
        for number, cell in enumerate(notebook.cells, 1):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            if cell.execution_count is None:
                raise RuntimeError(f"Cell {number} was not executed")
            if any(o.output_type == "error" for o in cell.outputs):
                raise RuntimeError(f"Cell {number} returned an error")


def environment() -> dict[str, object]:
    """Capture runtime versions and the checked-out revision for the report."""
    versions: dict[str, str | None] = {}
    for name in (
        "dxquant",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "nbformat",
        "nbclient",
        "ipykernel",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty: bool | None = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "versions": versions,
        "root": str(ROOT),
        "git_commit": revision,
        "git_dirty": dirty,
        "validator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> int:
    """Validate every tutorial and return a nonzero status on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Install this checkout and run every notebook with plotting",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds per cell (default 300)",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.report and args.report.suffix.lower() != ".json":
        parser.error("--report must name a .json file")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    results = []
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    names = {p.name for p in paths}
    fingerprints = {p.name: sha256(p.read_bytes()).hexdigest() for p in paths}
    if names != set(EXPECTED):
        results.append(
            Result(
                "inventory",
                "formal",
                errors=[
                    f"Missing: {sorted(set(EXPECTED) - names)}; "
                    f"unexpected: {sorted(names - set(EXPECTED))}"
                ],
            )
        )
    notebooks = {}
    setups = set()
    for path in paths:
        start = time.perf_counter()
        result = Result(path.name, "formal")
        try:
            notebook = validate(path)
            notebooks[path.name] = notebook
            setups.add(
                next(c.source for c in notebook.cells if c.cell_type == "code")
            )
        except Exception as exc:
            result.errors.append(f"{type(exc).__name__}: {exc}")
        result.seconds = time.perf_counter() - start
        results.append(result)
        status = "FAIL" if result.errors else "PASS"
        print(f"{status} formal {path.name}", flush=True)
    if len(setups) > 1:
        results.append(
            Result(
                "setup cells",
                "formal",
                errors=["Setup cells differ across tutorials"],
            )
        )
    if args.execute and not any(r.errors for r in results):
        try:
            import ipykernel  # noqa: F401
            import nbclient  # noqa: F401

            print(
                "Installing this checkout and plotting dependencies...",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    str(ROOT),
                    "matplotlib>=3.8",
                ],
                check=True,
            )
        except Exception as exc:
            results.append(
                Result(
                    "runtime setup",
                    "execute",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            )
        else:
            for name, notebook in notebooks.items():
                start = time.perf_counter()
                result = Result(name, "execute")
                print(f"RUN  {name}", flush=True)
                try:
                    execute(notebook, args.timeout)
                except Exception as exc:
                    result.errors.append(f"{type(exc).__name__}: {exc}")
                result.seconds = time.perf_counter() - start
                results.append(result)
                status = "FAIL" if result.errors else "PASS"
                print(
                    f"{status} execute {name} ({result.seconds:.1f}s)",
                    flush=True,
                )
    failed = [r for r in results if r.errors]
    for result in failed:
        print(f"\n{result.phase}: {result.notebook}", file=sys.stderr)
        print("\n".join(result.errors), file=sys.stderr)
    if any("ModuleNotFoundError" in e for r in failed for e in r.errors):
        print(
            "Install runner dependencies: python -m pip install "
            "nbformat nbclient ipykernel",
            file=sys.stderr,
        )
    passed = sum(r.phase == "execute" and not r.errors for r in results)
    mode = f"; executed {passed}/{len(EXPECTED)}" if args.execute else ""
    print(
        f"{'FAIL' if failed else 'PASS'}: {len(paths)} notebooks{mode}",
        flush=True,
    )
    if args.report:
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execute": args.execute,
            "timeout_per_cell": args.timeout,
            "success": not failed,
            "environment": environment(),
            "results": [asdict(r) for r in results],
            "notebook_sha256": fingerprints,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {args.report.resolve()}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
