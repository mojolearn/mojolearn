#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run source-contract test modules (files that read sources and import no
mojolearn binding) without importing the package: `python3
tools/run_source_tests.py <test file> ...`. Exit status is the failure count."""
import importlib.util
import sys
import traceback

failures = 0
for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location("source_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in sorted(dir(module)):
        if name.startswith("test_") and callable(getattr(module, name)):
            try:
                getattr(module, name)()
                print("PASS", path.rsplit("/", 1)[-1], name)
            except Exception:
                failures += 1
                print("FAIL", path.rsplit("/", 1)[-1], name)
                traceback.print_exc(limit=3)
sys.exit(failures)
