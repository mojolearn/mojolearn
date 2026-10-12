#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Decide whether two commits hold the same bit-producing code.

Used by tools/admit_identity_columns.sh when the vendor records name different commits. A record
is a function of the kernels, the bindings and the Python routing layer; it cannot depend on
docs, boards, release notes, the orchestration tools, the docs site, the research release tree,
or package metadata. 2026-10-12: the AMD column was recorded at bac431c11 and the NVIDIA column
at a8fd6d926, 80 Python files apart, every one a docstring rewrite for docs.mojolearn.org plus
the project URLs in pyproject.toml. Re-recording a vendor for docstrings is the double work
Andrew forbade on 2026-10-10, so docstring-only Python changes count as identical code: each
changed .py file is parsed on both sides and compared with its docstrings removed.

Usage: identity_code_equal.py <commit-a> <commit-b>   exit 0 = identical, 1 = differs (lists the files)
"""
import ast
import subprocess
import sys

# Paths that cannot change a record's bits.
INERT = ("bench/results/", "docs/", "docs-site/", "release/", "tools/", "experiments/", "paper/",
         "archive/", "parked/", "CHANGELOG.md", "python/pyproject.toml", "pyproject.toml", ".github/")


def _strip_docstrings(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def _show(commit, path):
    try:
        return subprocess.check_output(["git", "show", f"{commit}:{path}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def differing_files(a, b):
    names = subprocess.check_output(["git", "diff", "--name-only", a, b, "--"]).decode().split("\n")
    out = []
    for path in names:
        if not path or path.startswith(INERT) or path.endswith(".md"):
            continue
        if path.endswith(".py"):
            sa, sb = _show(a, path), _show(b, path)
            if sa is not None and sb is not None:
                try:
                    if _strip_docstrings(sa) == _strip_docstrings(sb):
                        continue
                except SyntaxError:
                    pass
        out.append(path)
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().split("\n")[-1], file=sys.stderr)
        return 2
    files = differing_files(argv[1], argv[2])
    if files:
        print("\n".join(files))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
