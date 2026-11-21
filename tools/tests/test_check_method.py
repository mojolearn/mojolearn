# SPDX-License-Identifier: Apache-2.0
"""`estimator.check(X)`: the GPU-versus-CPU check as a method, at the Python boundary.

No binding and no GPU. The real `CheckMixin` class statement is taken from `python/mojolearn/_mode.py` and run in a
namespace that stands for `mojolearn._mode`, under the stub `mojolearn` package of test_check_public.py (its
`__path__` is the source tree), so calling `check` imports the real `_verify_check.py` and `_check_both.py` the way an
installed package does. What is held here: every estimator reaches the method by inheritance (read from the class
statements, on a sample of families and on the family with its own base), the method is a door onto the one function
`mojolearn.check` is (same answers, same refusals, the report printed), it is not a parameter, and neither
`import mojolearn` nor the class statements load the check modules.
"""
import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python/mojolearn"
MODE = PKG / "_mode.py"

_spec = importlib.util.spec_from_file_location("check_public_cases", Path(__file__).with_name("test_check_public.py"))
public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(public)
cases = public.cases
package = public.package          # the stub-package fixture, shared
_CHECK_MODULES = public._CHECK_MODULES


def _tree(path):
    return ast.parse(path.read_text())


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _check_mixin():
    """The real `CheckMixin` class statement, run as `mojolearn._mode` would run it."""
    node = _class(_tree(MODE), "CheckMixin")
    namespace = {"__name__": "mojolearn._mode", "__package__": "mojolearn"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODE), "exec"), namespace)
    return namespace["CheckMixin"]


def _estimator(mixin):
    class Estimator(cases.GpuModel, mixin):
        pass
    X, est = cases.data()
    return X, Estimator(est.coef)


# --------------------------------------------------------------------------
# which classes have it: the class statements
# --------------------------------------------------------------------------

def _module_name(path):
    rel = path.relative_to(PKG).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _index():
    """module -> (class name -> base expressions, name -> (module, name) it is imported or routed from)."""
    out = {}
    for path in PKG.rglob("*.py"):
        if "tests" in path.relative_to(PKG).parts:
            continue
        name = _module_name(path)
        tree = ast.parse(path.read_text())
        classes, names = {}, {}
        here = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = [ast.unparse(b) for b in node.bases]
            elif isinstance(node, ast.ImportFrom):
                base = here[:len(here) - node.level + 1] if node.level else []
                module = ".".join([p for p in base if p] + ([node.module] if node.module else []))
                for alias in node.names:
                    names[alias.asname or alias.name] = (module, alias.name)
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == "_LAZY_EXPORTS"):
                for key, (module, attr) in ast.literal_eval(node.value).items():
                    names[key] = (".".join([p for p in here if p] + [module]), attr)
        out[name] = (classes, names)
    return out


def _reaches(index, module, name, target=("_mode", "CheckMixin"), seen=()):
    if (module, name) == target:
        return True
    if (module, name) in seen:
        return False
    seen += ((module, name),)
    classes, names = index.get(module, ({}, {}))
    if name in classes:
        return any(_reaches(index, module, base.split("[")[0], target, seen) for base in classes[name]
                   if "." not in base)
    if name in names:
        return _reaches(index, *names[name], target, seen)
    return False


SAMPLE = ("Ridge", "LogisticRegression", "KMeans", "PCA", "RandomForestClassifier", "StandardScaler", "SVC",
          "CNNClassifier")


def test_every_mode_mixin_reaches_the_one_method():
    tree = _tree(MODE)
    assert [ast.unparse(b) for b in _class(tree, "PublicOutputsMixin").bases] == ["CheckMixin"]
    for name, parent in (("SklearnTagsMixin", "PublicOutputsMixin"), ("ParamsMixin", "SklearnTagsMixin"),
                         ("NumericModeMixin", "ParamsMixin")):
        assert parent in [ast.unparse(b) for b in _class(tree, name).bases], name
    defined = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "check"]
    assert len(defined) == 1, "one method, no copies"


def test_a_sample_of_public_estimators_inherits_it():
    lazy = public._literal("_LAZY_NAMES")
    index = _index()
    missing = [name for name in SAMPLE if not _reaches(index, lazy[name].lstrip("."), name)]
    assert missing == []
    searches = PKG / "model_selection.py"
    for path in PKG.rglob("*.py"):    # the families with their own base mix in the same class, never a copy
        if "tests" not in path.relative_to(PKG).parts and path not in (MODE, searches):
            assert "def check(self, X" not in path.read_text(), path
    # The one other definition (2026-11-18): a search has no device of its own, so `_BaseSearch.check` is the
    # check of its fitted `best_estimator_` (the same `_verify_check.check`, imported on the call), defined once
    # on the search base and nowhere else in that module.
    tree = _tree(searches)
    defined = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "check"]
    assert len(defined) == 1, "one search check, no copies"
    assert defined[0] in _class(tree, "_BaseSearch").body
    body = ast.unparse(defined[0])
    assert "from ._verify_check import check, format_check" in body
    assert "check(best, X, method=method)" in body
    assert "best_estimator_" in body


# --------------------------------------------------------------------------
# what it does
# --------------------------------------------------------------------------

def test_the_class_statement_and_the_package_load_no_check_module(package):
    package()
    mixin = _check_mixin()
    _estimator(mixin)
    assert not any(name in sys.modules for name in _CHECK_MODULES), "the method imports the check when called"
    imported = set()
    for node in _tree(MODE).body:     # module level of _mode.py: what `import mojolearn` runs
        if isinstance(node, ast.ImportFrom):
            imported.update([node.module or ""] + [alias.name for alias in node.names])
    assert not imported & {"_verify_check", "_check_both", "check", "format_check"}


def _loaded(ml, monkeypatch):
    """Touch `mojolearn.check` (loads the check modules) and fix the device labels, as test_check_public does."""
    function = ml.check
    public.labelled(monkeypatch)
    return function


def test_the_method_is_a_door_onto_the_one_check(package, monkeypatch, capsys):
    ml = package()
    X, est = _estimator(_check_mixin())
    function = _loaded(ml, monkeypatch)
    private = sys.modules["mojolearn._verify_check"]
    assert function is private.check, "mojolearn.check is the function the method calls"
    calls = []
    answer = dict(format="mojolearn.check-both.v1", ran=False, reason="recorded")
    monkeypatch.setattr(private, "check", lambda *a, **k: calls.append((a, k)) or answer)
    monkeypatch.setattr(private, "format_check", lambda result: "REPORT " + result["reason"])
    assert est.check(X, method="predict", fit=(X,)) is answer
    assert calls == [((est, X), dict(method="predict", fit=(X,)))]
    assert capsys.readouterr().out == "REPORT recorded\n"


def test_the_method_matches_prints_and_answers_like_mojolearn_check(package, monkeypatch, capsys):
    ml = package()
    X, est = _estimator(_check_mixin())
    function = _loaded(ml, monkeypatch)
    result = est.check(X)
    printed = capsys.readouterr().out
    assert result["ran"] and result["method"] == "predict" and est.fits == 0
    assert result["verdict"]["verdict"] == "MATCH"
    cb = sys.modules["mojolearn._check_both"]
    assert printed == cb.format_check(result) + "\n" and "MATCH:" in printed
    again = function(est, X)
    assert again["verdict"] == result["verdict"]
    assert again["sides"]["gpu"]["outputs"] == result["sides"]["gpu"]["outputs"]


def test_the_method_finds_a_mismatch(package, monkeypatch, capsys):
    ml = package(nudge=3)
    X, est = _estimator(_check_mixin())
    _loaded(ml, monkeypatch)
    result = est.check(X)
    assert result["ran"] and result["verdict"]["verdict"] == "MISMATCH"
    assert result["verdict"]["first"]["index"] == [3]
    assert "MISMATCH" in capsys.readouterr().out


@pytest.mark.parametrize("install,needle", [
    (dict(mode="fast"), "identical mode only"),
    (dict(vendor="cpu"), None),
])
def test_the_method_keeps_the_refusals(package, monkeypatch, capsys, install, needle):
    ml = package(**install)
    X, est = _estimator(_check_mixin())
    _loaded(ml, monkeypatch)
    est.predict = lambda X: pytest.fail("a refused check must not call the estimator")
    result = est.check(X)
    assert not result["ran"] and "verdict" not in result
    cb = sys.modules["mojolearn._check_both"]
    assert needle in result["reason"] if needle else result["reason"] == cb.NO_GPU
    assert result["reason"] in capsys.readouterr().out


def test_it_is_not_a_parameter_or_a_tag():
    tree = _tree(MODE)
    # get_params reads the constructor signature (ParamsMixin._get_param_names); the tags read named class
    # attributes (sklearn_tags). Neither enumerates methods, and `check` is no constructor argument.
    names = _class(tree, "ParamsMixin")
    getter = next(n for n in names.body if isinstance(n, ast.FunctionDef) and n.name == "_get_param_names")
    assert "signature" in ast.unparse(getter) and "dir(" not in ast.unparse(getter)
    tags = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sklearn_tags")
    assert "dir(" not in ast.unparse(tags) and "'check'" not in ast.unparse(tags)
    method = next(n for n in ast.walk(_class(tree, "CheckMixin")) if isinstance(n, ast.FunctionDef))
    assert [a.arg for a in method.args.args] == ["self", "X"]
    assert [a.arg for a in method.args.kwonlyargs] == ["method", "fit"]


def test_the_docstring_is_user_documentation():
    method = next(n for n in ast.walk(_class(_tree(MODE), "CheckMixin")) if isinstance(n, ast.FunctionDef))
    doc = " ".join(ast.get_docstring(method).split())
    for section in ("Parameters", "Returns", "Examples"):
        assert section in doc, section
    assert "In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs" in doc
    assert ".check(X)" in doc and "MATCH" in doc and "MISMATCH" in doc
    # private module names only: a bare substring test also matched the public `mojolearn.host_model` ("host_model")
    for private in ("_verify", "_check_both", "_mode"):
        assert not re.search(r"(?<![A-Za-z0-9])" + private, doc), private
    assert "fast" not in doc, "the public docs speak of identical only"
