# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests: CatBoost / XGBoost / LightGBM parameter names on the
gradient-boosting classes (python/mojolearn/_boosting_param_names.py).

The table and its generated page; the compiled library is not imported.
"""
import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _table_module():
    spec = importlib.util.spec_from_file_location(
        "_boosting_param_names", ROOT / "python" / "mojolearn" / "_boosting_param_names.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- the table

def test_table_names_are_unique():
    names = [row[0] for row in _table_module().BOOSTING_PARAMETER_NAMES]
    assert len(names) == len(set(names))


def test_refused_rows_carry_their_alternative():
    for name, canonical, libraries, same, note in _table_module().BOOSTING_PARAMETER_NAMES:
        assert libraries
        if not same:
            assert canonical in note, name


def test_generated_page_is_current():
    result = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "gen_boosting_parameter_names.py"),
                             "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
