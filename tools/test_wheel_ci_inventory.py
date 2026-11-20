# SPDX-License-Identifier: Apache-2.0
"""`packaging/wheel_ci.py inventory`: the import forms it counts as reaching a module."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wheel_ci", ROOT / "packaging" / "wheel_ci.py")
wheel_ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wheel_ci)


def _package(tmp_path, files):
    pkg = tmp_path / "python" / "mojolearn"
    pkg.mkdir(parents=True)
    for name, text in files.items():
        (pkg / name).write_text(text, encoding="utf-8")
    return pkg


_GETATTR = (
    "def __getattr__(name):\n"
    "    import importlib\n"
    "    where = _CPU_INFERENCE[name]\n"
    "    if where is None:\n"
    "        return importlib.import_module('.' + name, __name__)\n"
    "    return getattr(importlib.import_module(where, __name__), name)\n"
)


def test_first_touch_table_reaches_string_and_none_entries(tmp_path, capsys):
    pkg = _package(tmp_path, {
        "__init__.py": "_CPU_INFERENCE = {'Host': '._host_impl', 'language_model': None}\n" + _GETATTR,
        "_host_impl.py": "from . import _helper\n",
        "_helper.py": "",
        "language_model.py": "",
    })
    assert wheel_ci.inventory([str(pkg)]) == 0
    assert "all reachable by import" in capsys.readouterr().out


def test_string_table_without_import_module_getattr_is_inert(tmp_path, capsys):
    pkg = _package(tmp_path, {
        "__init__.py": "_NAMES = {'Host': '._host_impl', 'language_model': None}\n",
        "_host_impl.py": "",
        "language_model.py": "",
    })
    assert wheel_ci.inventory([str(pkg)]) == 1
    err = capsys.readouterr().err
    assert "mojolearn/_host_impl.py" in err and "mojolearn/language_model.py" in err


def test_adapted_module_is_reached_through_host_surface(tmp_path):
    files = {
        "__init__.py": "from . import host_surface\n",
        "host_surface.py": "ADAPTED_MODULES = {'_b': dict(family='b', module='_adapter')}\n",
        "_adapter.py": "",
    }
    assert wheel_ci.inventory([str(_package(tmp_path / "a", files))]) == 0
    files["host_surface.py"] = "OTHER = {'_b': dict(family='b', module='_adapter')}\n"
    assert wheel_ci.inventory([str(_package(tmp_path / "b", files))]) == 1


def test_unimported_module_still_fails(tmp_path):
    pkg = _package(tmp_path, {"__init__.py": "", "_orphan.py": ""})
    assert wheel_ci.inventory([str(pkg)]) == 1
