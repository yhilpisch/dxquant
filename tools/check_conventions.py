"""Check strict line width, public docstrings, and annotated defaults."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sources() -> Iterator[tuple[str, str, bool]]:
    """Yield maintained Python files and notebook code cells."""

    files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
        ROOT / "docs/conf.py",
        *sorted((ROOT / "tools").glob("*.py")),
    ]
    for path in files:
        yield str(path.relative_to(ROOT)), path.read_text(), "src" in path.parts
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"], 1):
            if cell["cell_type"] == "code":
                label = f"{path.relative_to(ROOT)}:cell-{index}"
                yield label, "".join(cell["source"]), False


def offsets(text: str) -> list[int]:
    """Return zero-based offsets for source line boundaries."""

    result = [0]
    for line in text.splitlines(keepends=True):
        result.append(result[-1] + len(line.encode("utf-8")))
    return result


def check_defaults(tree: ast.AST, text: str) -> list[str]:
    """Require no horizontal spaces around annotated argument defaults."""

    failures = []
    starts = offsets(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args]
        pairs: list[tuple[ast.arg, ast.expr | None]] = list(
            zip(
                arguments[-len(node.args.defaults) :],
                node.args.defaults,
                strict=False,
            )
        )
        pairs += list(
            zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
        )
        for argument, default in pairs:
            annotation = argument.annotation
            if annotation is None or default is None:
                continue
            if (
                annotation.end_lineno is None
                or annotation.end_col_offset is None
            ):
                continue
            begin = (
                starts[annotation.end_lineno - 1] + annotation.end_col_offset
            )
            end = starts[default.lineno - 1] + default.col_offset
            gap = text.encode("utf-8")[begin:end].decode("utf-8")
            if re.search(r"[ \t]=|=[ \t]", gap):
                failures.append(f"{argument.lineno}: annotated default spacing")
    return failures


def check_docstrings(node: ast.Module | ast.ClassDef) -> list[str]:
    """Check the library's visible modules, classes, and callable methods."""

    failures = []
    if not ast.get_docstring(node):
        failures.append(f"{getattr(node, 'lineno', 1)}: missing docstring")
    for child in node.body:
        if not isinstance(child, (ast.ClassDef, ast.FunctionDef)):
            continue
        if child.name.startswith("_") and child.name != "__call__":
            continue
        if isinstance(child, ast.ClassDef):
            failures.extend(check_docstrings(child))
        else:
            overload = any(
                isinstance(d, ast.Name) and d.id == "overload"
                for d in child.decorator_list
            )
            if not overload and not ast.get_docstring(child):
                failures.append(
                    f"{child.lineno}: missing {child.name} docstring"
                )
    return failures


def main() -> int:
    """Print convention violations and return a failing exit status."""

    failures = []
    count = 0
    for label, text, library in sources():
        count += 1
        for index, line in enumerate(text.splitlines(), 1):
            if len(line) > 80:
                failures.append(f"{label}:{index}: line exceeds 80 characters")
        tree = ast.parse(text, filename=label)
        issues = check_defaults(tree, text)
        if library:
            issues += check_docstrings(tree)
        failures.extend(f"{label}:{message}" for message in issues)
    for failure in failures:
        print(failure)
    print(
        f"Conventions: {count} Python files/cells; {len(failures)} violations."
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
