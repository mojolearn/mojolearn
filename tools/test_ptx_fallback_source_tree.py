# SPDX-License-Identifier: Apache-2.0
"""Collect python/mojolearn/tests/test_ptx_fallback.py from outside the package.

That file needs no built binding: it loads `_backend` under a private alias with
the package __init__ skipped. Collected in place, pytest imports it as
`mojolearn.tests.test_ptx_fallback`, which runs `mojolearn/__init__.py` first,
and on a source-only tree (no bindings/build*.sh run) that refuses with
"no identical binary exists" before a single case is reached. This file loads
the same module by path under a private name and exposes its TestCase classes
unchanged, so the same cases run on a source-only checkout (a CPU pod) as well.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "python/mojolearn/tests/test_ptx_fallback.py"
_NAME = "_mojolearn_ptx_fallback_cases"

_spec = importlib.util.spec_from_file_location(_NAME, _PATH)
_cases = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = _cases
_spec.loader.exec_module(_cases)

for _name, _value in vars(_cases).items():
    if isinstance(_value, type) and issubclass(_value, unittest.TestCase) and _value.__module__ == _NAME:
        globals()[_name] = _value

if __name__ == "__main__":
    unittest.main()
