#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The static half of the release gate: the checks a box with no GPU can honestly make.

Run by `.github/workflows/wheel-ci.yml`. Every check here is a fact about the
SOURCE TREE or about a wheel's INVENTORY, never about a kernel, because the
runner that runs it has no GPU and a check that cannot run must say so rather
than be allowed to fail quietly.

WHY THIS FILE EXISTS. Wheel CI tried to build the real wheel on a hosted macOS
runner for its whole history and failed 124 times out of 127; the three greens
were on 2026-08-20 and one of them was, by its own commit message, "a green CI
run in which all five interpreters had failed". Two releases were cut past that
red gate and both shipped a defect. A gate that verifies less and is GREEN is
worth more than one that pretends to build a GPU wheel and never can.

    inventory <package-dir> [<extra-root.py> ...]
        Every .py in the shipped package must be REACHABLE by import from a
        root. `mojolearn/torch_ops.py`, 1,138 lines, shipped in the 0.2.x and
        0.3.x wheels having never been imported, compiled or executed by
        anything (CHANGELOG.md, "Removed", 2026-08-31). `packages =
        ["mojolearn"]` in pyproject.toml sweeps the directory, so nothing
        downstream could have noticed. This walks the import graph instead.

    versions <repo-root>
        python/pyproject.toml, python/mojolearn/_version.py and CITATION.cff
        must agree. The release workflow's "Version agreement" step reads the
        first two only, and a wrong CITATION.cff `version` is minted
        permanently into a DOI.

    pins <repo-root>
        Every bindings/build_*.sh must pin BOTH cpu baselines, and MACOS_FLOOR
        must equal setup.py's DEFAULT_MACOS_TARGET. macOS pinned
        `--target-cpu apple-m1` from 0.1.0; Linux pinned nothing until
        2026-08-31, and that is exactly why 0.3.0's Linux wheel carried
        unconditional AVX-512 and SIGILLed on every host without it. This
        gate cannot catch a bad BINARY -- `packaging/linux/isa_baseline_linux.py`
        on the shipped artifact is the only thing that can, and it belongs in
        the release path -- but it does stop the SOURCE regressing to the
        state that produced one.
"""

import ast
import pathlib
import re
import sys


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------
def _intra_package_imports(path, pkg):
    """Names of sibling modules `path` imports from within `pkg`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # `from . import x, y`  /  `from .x import y`  /  `from .x.y import z`
            if node.level and not node.module:
                out.update(a.name for a in node.names)
            elif node.level and node.module:
                out.add(node.module.split(".")[0])
            elif node.module and node.module.split(".")[0] == pkg:
                parts = node.module.split(".")
                if len(parts) > 1:
                    out.add(parts[1])
                else:
                    out.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == pkg and len(parts) > 1:
                    out.add(parts[1])
    return out


def inventory(argv):
    pkgdir = pathlib.Path(argv[0]).resolve()
    pkg = pkgdir.name
    extra_roots = [pathlib.Path(p).resolve() for p in argv[1:]]

    present = {p.stem: p for p in sorted(pkgdir.glob("*.py"))}
    if "__init__" not in present:
        print(f"FAIL {pkgdir} has no __init__.py", file=sys.stderr)
        return 1

    # ROOTS. `__init__` is what `import mojolearn` runs. `__main__` is
    # `python -m mojolearn` and is the ONLY importer of `_verify`. The extra
    # roots are the top-level modules pyproject.toml ships outside the
    # package -- today `mojolearn_diagnostics`, which pyproject.toml names as
    # the `mojolearn` console script and which is deliberately kept OUT of
    # the package so it still runs when importing the package is the thing a
    # user needs to diagnose.
    roots = ["__init__", "__main__"]
    queue = [r for r in roots if r in present]
    # A standalone top-level module may pull package modules in too.
    for f in extra_roots:
        queue.extend(_intra_package_imports(f, pkg))

    reached = set()
    while queue:
        name = queue.pop()
        if name in reached:
            continue
        reached.add(name)
        if name in present:
            queue.extend(_intra_package_imports(present[name], pkg))

    orphans = sorted(set(present) - reached)
    if orphans:
        print(
            "FAIL these modules ship in the wheel and NOTHING imports them:\n"
            + "".join(f"    {pkg}/{o}.py\n" for o in orphans)
            + "  `packages = [\"mojolearn\"]` sweeps the directory, so an\n"
            "  abandoned module is not absent from the wheel -- it SHIPS.\n"
            "  mojolearn/torch_ops.py did, for two minor releases, having\n"
            "  never been imported by anything. Delete it, or import it.",
            file=sys.stderr,
        )
        return 1
    print(f"  inventory: {len(present)} modules in {pkg}/, all reachable by import")
    return 0


# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------
def _one(pattern, path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    m = re.search(pattern, text, re.M)
    if not m:
        raise SystemExit(f"FAIL {path}: no line matching {pattern!r}")
    return m.group(1)


def versions(argv):
    root = pathlib.Path(argv[0]).resolve()
    found = {
        "python/pyproject.toml": _one(
            r'^version = "([^"]+)"', root / "python/pyproject.toml"),
        "python/mojolearn/_version.py": _one(
            r'^__version__ = "([^"]+)"', root / "python/mojolearn/_version.py"),
        "CITATION.cff": _one(
            r'^version: *"?([^"\s]+)"?', root / "CITATION.cff"),
    }
    if len(set(found.values())) != 1:
        print("FAIL the version is written in three places and they disagree:",
              file=sys.stderr)
        for k, v in found.items():
            print(f"    {v:<12} {k}", file=sys.stderr)
        print("  A wrong CITATION.cff version is minted permanently into a DOI.",
              file=sys.stderr)
        return 1
    print(f"  versions: {next(iter(found.values()))} in all three places")
    return 0


# --------------------------------------------------------------------------
# pins
# --------------------------------------------------------------------------
def pins(argv):
    root = pathlib.Path(argv[0]).resolve()
    target = _one(r'^DEFAULT_MACOS_TARGET = "([^"]+)"', root / "python/setup.py")
    scripts = sorted((root / "bindings").glob("build*.sh"))
    if not scripts:
        print("FAIL no bindings/build*.sh found", file=sys.stderr)
        return 1
    bad = 0
    for s in scripts:
        text = s.read_text(encoding="utf-8")
        floor = re.search(r'^MACOS_FLOOR="([^"]+)"', text, re.M)
        if not floor:
            print(f"FAIL {s.name}: no MACOS_FLOOR", file=sys.stderr); bad += 1
        elif floor.group(1) != target:
            print(f"FAIL {s.name}: MACOS_FLOOR {floor.group(1)} but "
                  f"setup.py tags {target}. One wheel carries one tag and the "
                  f"tag is the floor of everything inside.", file=sys.stderr)
            bad += 1
        # THE LINE THAT WAS MISSING WHEN 0.3.0 SHIPPED. Without it `mojo build`
        # targets the build box's own CPU; four build boxes in a row happened
        # to have AVX-512 and the wheel SIGILLed everywhere else.
        if not re.search(r'--target-cpu \$\{MOJOLEARN_LINUX_CPU:-x86-64-v3\}', text):
            print(f"FAIL {s.name}: does not pin the Linux x86-64 baseline to "
                  f"x86-64-v3. 0.3.0 shipped without this and SIGILLed on "
                  f"every host without AVX-512.", file=sys.stderr)
            bad += 1
        if "apple-m1" not in text:
            print(f"FAIL {s.name}: does not pin --target-cpu apple-m1",
                  file=sys.stderr)
            bad += 1
    if bad:
        return 1
    print(f"  pins: {len(scripts)} build scripts, macOS floor {target}, "
          f"Linux baseline x86-64-v3")
    return 0


CHECKS = {"inventory": inventory, "versions": versions, "pins": pins}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CHECKS:
        print(f"usage: wheel_ci.py {{{'|'.join(CHECKS)}}} ...", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(CHECKS[sys.argv[1]](sys.argv[2:]))
