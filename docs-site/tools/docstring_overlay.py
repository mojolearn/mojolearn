"""Stage the installed mojolearn release with newer docstrings from this repo.

Docstrings change between releases (a docs-only edit should not wait for a wheel). This copies the
installed package to <out>/mojolearn and replaces each Python file with the repo's version ONLY when
the two are the same code once docstrings are removed, so the API reference documents exactly the
released code with the current text. Any file whose code differs keeps the released version.
Usage: python tools/docstring_overlay.py <repo root> <out dir>
"""
# cpu-route: documentation build tooling, never imported by the mojolearn runtime.
import ast
import pathlib
import shutil
import sys

import mojolearn


def code_only(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def main():
    repo, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    installed = pathlib.Path(mojolearn.__file__).parent
    staged = out / "mojolearn"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(installed, staged, symlinks=True)
    taken = kept = 0
    for released in staged.rglob("*.py"):
        source = repo / "python" / "mojolearn" / released.relative_to(staged)
        if not source.exists():
            continue
        new = source.read_text()
        if new == released.read_text():
            continue
        try:
            same = code_only(new) == code_only(released.read_text())
        except SyntaxError:
            same = False
        if same:
            released.write_text(new)
            taken += 1
        else:
            kept += 1
    print(f"docstring_overlay: {taken} files with newer docstrings, {kept} with newer code kept as released")


if __name__ == "__main__":
    main()
