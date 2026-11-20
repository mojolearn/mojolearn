# SPDX-License-Identifier: Apache-2.0
"""`mojolearn.check(estimator, X)`: the GPU-versus-CPU check under its public name, at the Python boundary.

No binding and no GPU. The package's own `__getattr__` and `_LAZY_NAMES` are taken from `python/mojolearn/__init__.py`
and run on a stub `mojolearn` module (the stub install of test_check_both.py) whose `__path__` is the source tree, so
touching `mojolearn.check` imports the real `_verify_check.py` and `_check_both.py` the way an installed package does.
What is held here: the name is exported and routed lazily (importing the package imports neither module), it is the
very function `mojolearn._verify_check.check` (the private path keeps working, one implementation), it gives the same
answers and refusals as that path, and its docstring is user documentation.
"""
import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python/mojolearn"
INIT = PKG / "__init__.py"

_spec = importlib.util.spec_from_file_location("check_both_cases", Path(__file__).with_name("test_check_both.py"))
cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cases)

_CHECK_MODULES = ("mojolearn._verify_check", "mojolearn._check_both")


def _init_tree():
    return ast.parse(INIT.read_text())


def _literal(name):
    for node in _init_tree().body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a literal in mojolearn/__init__.py")


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A stub `mojolearn` package that routes names with the real `__init__.py` `__getattr__`."""
    for name in _CHECK_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def make(**install):
        stub = cases.install(tmp_path, **install)
        ml = types.ModuleType("mojolearn")
        ml.__path__ = [str(PKG)]
        ml.__package__ = "mojolearn"
        vars(ml).update(vars(stub))
        getattr_node = next(node for node in _init_tree().body
                            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__")
        vars(ml).update(_LAZY_NAMES=_literal("_LAZY_NAMES"), _importlib=importlib, _NOT_YET={},
                        _families=types.SimpleNamespace(EAGER_BEFORE=()), tokenizer=None)
        exec(compile(ast.Module(body=[getattr_node], type_ignores=[]), str(INIT), "exec"), vars(ml))
        monkeypatch.setitem(sys.modules, "mojolearn", ml)
        return ml

    yield make
    for name in _CHECK_MODULES:
        sys.modules.pop(name, None)


def labelled(monkeypatch):
    """Fixed device labels, as test_check_both's `command` sets them (no nvidia-smi on a test runner)."""
    cb = sys.modules["mojolearn._check_both"]
    real = cb.side_environment
    monkeypatch.setattr(cb, "side_environment",
                        lambda ml, side, describe_device=None, cpu_model=None:
                        real(ml, side, describe_device=cases.LABELS["describe_device"],
                             cpu_model=cases.LABELS["cpu_model"]))


def test_check_is_exported_and_routed_to_the_one_check_lazily():
    assert "check" in _literal("__all__")
    assert _literal("_LAZY_NAMES")["check"] == "._verify_check"
    imported = set()
    for node in _init_tree().body:   # module level only: what `import mojolearn` runs
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not imported & {"_verify_check", "_check_both", "mojolearn._verify_check", "mojolearn._check_both"}


def test_touching_the_name_imports_the_check_and_nothing_before(package):
    ml = package()
    assert not any(name in sys.modules for name in _CHECK_MODULES), "routing must not import on its own"
    check = ml.check
    assert check is sys.modules["mojolearn._verify_check"].check, "one function: the private path is the same object"
    assert vars(ml)["check"] is check, "resolved once, then a plain attribute"
    assert sys.modules["mojolearn._verify_check"].cb is sys.modules["mojolearn._check_both"]


def test_the_public_name_matches_and_reports_like_the_private_path(package, monkeypatch):
    ml = package()
    check = ml.check
    labelled(monkeypatch)
    X, est = cases.data()
    result = check(est, X)
    assert result["ran"] and result["method"] == "predict" and est.fits == 0
    assert result["verdict"]["verdict"] == "MATCH" and result["verdict"]["match"]
    private = sys.modules["mojolearn._verify_check"]
    again = private.check(est, X)
    assert again["verdict"] == result["verdict"]
    assert again["sides"]["gpu"]["outputs"] == result["sides"]["gpu"]["outputs"]
    assert "MATCH:" in private.format_check(result)


def test_the_public_name_finds_a_mismatch(package, monkeypatch):
    ml = package(nudge=3)
    check = ml.check
    labelled(monkeypatch)
    X, est = cases.data()
    result = check(est, X)
    assert result["ran"] and result["verdict"]["verdict"] == "MISMATCH"
    assert result["verdict"]["first"]["index"] == [3]


@pytest.mark.parametrize("install,needle", [
    (dict(mode="fast"), "identical mode only"),
    (dict(vendor="cpu"), None),
])
def test_the_public_name_keeps_the_refusals(package, monkeypatch, install, needle):
    ml = package(**install)
    check = ml.check
    labelled(monkeypatch)
    X, est = cases.data()
    est.predict = lambda X: pytest.fail("a refused check must not call the estimator")
    result = check(est, X)
    assert not result["ran"] and "verdict" not in result
    cb = sys.modules["mojolearn._check_both"]
    assert needle in result["reason"] if needle else result["reason"] == cb.NO_GPU


def test_the_docstring_is_user_documentation():
    tree = ast.parse((PKG / "_verify_check.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")
    doc = " ".join(ast.get_docstring(node).split())
    for section in ("Parameters", "Returns", "Examples"):
        assert section in doc, section
    assert "In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs" in doc
    assert "mojolearn.check(" in doc
    assert "_verify" not in doc and "_check_both" not in doc, "no private module path in public docs"
    assert "fast" not in doc, "the public docs speak of identical only"
