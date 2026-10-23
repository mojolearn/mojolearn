#!/usr/bin/env python3
"""Resolve admission's host bindings without importing the product package."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def host_bindings(package, families):
    """Return the actual files selected by the package's binding-path resolver.

    Load only the standalone import-plumbing module: package initialization
    would load native extensions before this preflight could diagnose them.
    A private file has the same precedence here as it does during import.
    """
    package = Path(package)
    spec = importlib.util.spec_from_file_location(
        "identity_host_preflight_merged", package / "_merged.py",
    )
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)
    result = {}
    for family in families:
        path = Path(resolver.binding_path(
            str(package / "host"), f"_mojolearn_{family}_host",
        ))
        # Index membership alone is insufficient; a missing build artifact
        # (or a directory named .so) must not pass the host readiness check.
        result[family] = {"path": str(path), "exists": path.is_file()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("families", nargs="+")
    args = parser.parse_args(argv)
    try:
        bindings = host_bindings(args.package, args.families)
        if args.report:
            args.report.write_text(json.dumps({"bindings": bindings}, indent=2) + "\n")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"identity_host_preflight: cannot resolve host bindings: {exc}", file=sys.stderr)
        return 2
    for family, record in bindings.items():
        if not record["exists"]:
            print(family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
