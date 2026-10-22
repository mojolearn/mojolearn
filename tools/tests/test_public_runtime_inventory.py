# SPDX-License-Identifier: Apache-2.0
"""tools/public_runtime_inventory.py over a small fake package (static; imports nothing).

Run: python3 tools/tests/test_public_runtime_inventory.py   (or pytest)
"""
import importlib.util
import sys
import tempfile
import textwrap
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "public_runtime_inventory.py"
SPEC = importlib.util.spec_from_file_location("public_runtime_inventory", TOOL)
INV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INV)

FILES = {
    "__init__.py": '''
        from .est import (Native, Stateless, Orchestrated, Helper, Python, Driven, Flow, Typed,
                          Refuses, Dispatched, native_function)
        from .owned import Owned, Connected, Imported, Getattr, Wrapped
        __all__ = ["Native", "Stateless", "Orchestrated", "Helper", "Python", "Driven", "Flow", "Typed",
                   "Refuses", "Dispatched", "native_function", "LIMIT",
                   "Owned", "Connected", "Imported", "Getattr", "Wrapped"]
        LIMIT = 4
    ''',
    "_backend.py": '''
        def binding(name, mode=None):
            return __import__(name)
    ''',
    "_buffer.py": '''
        def empty(shape, dtype):
            return bytearray(4)

        def as_f32(X):
            for row in X:
                pass
            return X
    ''',
    "_prog.py": '''
        from . import _backend

        class Prog:
            def __init__(self):
                self.stages = []

            def stage(self, name):
                self.stages.append(name)

            def run(self):
                return _backend.binding("_mojolearn_x").run_program(self.stages)
    ''',
    "_tok.py": '''
        from . import _backend

        def _binding():
            return _Loaded(_backend.binding("_mojolearn_tok"))

        class _Loaded:
            def __init__(self, module):
                self._module = module
    ''',
    "_bridge.py": '''
        from ._tok import _binding

        def connect(owner):
            # ABI connection at import: (entry, context)
            entry = _binding()._module.owner_call
            return entry, {"owner": owner}
    ''',
    "owned.py": '''
        from . import _backend, _bridge
        from ._tok import _binding as _owned_binding


        class Owned:
            def fit(self, X, y=None):
                if X is None:
                    raise ValueError("X is required")
                return _NATIVE("fit", (self, X, y))

            def predict(self, X):
                return self._prepare(X)

            def _prepare(self, X):
                rows = [row for row in X]
                return _NATIVE("predict", (self, rows))


        class Connected:
            def fit(self, X, y=None):
                return _OWNER(self, "fit", (X, y), _OWNER_CONTEXT)

            def predict(self, X):
                _OWNER = len
                return _OWNER(X)


        class Getattr:
            def fit(self, X, y=None):
                return _GETATTR(self, X)


        _NATIVE = _owned_binding()._module.small_call
        _GETATTR = getattr(_backend.binding("_mojolearn_x"), "getattr_fit")
        _OWNER, _OWNER_CONTEXT = _bridge.connect(Connected)

        from .wrapped import Imported, Wrapped
    ''',
    "wrapped.py": '''
        from .owned import _NATIVE


        def _work(X):
            total = 0
            for row in X:
                total += row
            return total


        class Imported:
            def fit(self, X, y=None):
                return _NATIVE("fit", (self, X))


        class Wrapped:
            def fit(self, X, y=None):
                return _NATIVE("fit", (self, _work(X)))
    ''',
    "est.py": '''
        import numpy as np
        from . import _backend
        from ._buffer import as_f32, empty
        from ._prog import Prog


        class _Base:
            def _bind(self):
                return _backend.binding("_mojolearn_x", None)


        class Native(_Base):
            def fit(self, X, y=None):
                """Docstrings are not statements."""
                if X is None:
                    raise ValueError("X is required")
                return self._bind().x_fit(self, as_f32(X), y)

            def predict(self, X):
                return _backend.binding("_mojolearn_x").x_predict(self, X)


        class Stateless:
            def fit(self, X, y=None):
                return self


        class Orchestrated(_Base):
            def fit(self, X, y=None):
                b = self._bind()
                out = empty((4,), "<f4")
                b.stage_one(X, out)
                return self._finish(b, out)

            def _finish(self, b, out):
                b.stage_two(out)
                return self


        class Helper(_Base):
            def fit(self, X, y=None):
                return self._fit_with_tree_start(X, 0)

            def _fit_with_tree_start(self, X, start):
                for i in range(len(X)):
                    X[i] = X[i] * 2
                return self._bind().x_fit(X)


        class Python:
            def fit(self, X, y=None):
                self.mean_ = np.mean(X, axis=0)
                return self

            def transform(self, X):
                return [row for row in X]


        class Driven(_Base):
            def fit(self, X, y=None):
                b = self._bind()
                for layer in self.layers:  # driver-loop: over the model's layers
                    b.layer_fit(layer, X)
                return self


        def _call(binding, X):
            return binding.flow_entry(X)


        class Flow(_Base):
            def fit(self, X, y=None):
                return _call(self._bind(), X)


        class Typed:
            def fit(self, X, y=None):
                pr = Prog()
                pr.stage("a")
                return pr.run()


        class Refuses:
            def fit(self, X, y=None):
                return _backend.binding("_mojolearn_x").x_fit(X)

            def save(self, path):
                """Not available."""
                raise NotImplementedError("pickle it")


        _ENTRY, _CONTEXT = _backend.binding("_mojolearn_x").connect_public()


        class Dispatched:
            def fit(self, X, y=None):
                return _ENTRY(self, "fit", (X, y), _CONTEXT)


        def native_function(X):
            return _backend.binding("_mojolearn_x").entry(X)
    ''',
}


def build(root):
    base = root / "python" / "mojolearn"
    base.mkdir(parents=True)
    for name, text in FILES.items():
        (base / name).write_text(textwrap.dedent(text).lstrip())
    return root


def rows_by_key(root):
    return {(r.public.rsplit(".", 1)[-1], r.method): r for r in INV.inventory_rows(root)}


def test_verdicts_and_counts():
    with tempfile.TemporaryDirectory() as tmp:
        rows = rows_by_key(build(Path(tmp)))
    native = rows[("Native", "fit")]
    assert (native.verdict, native.native_calls, native.native_entries) == ("native", 1, 1)
    assert native.statements == 3 and native.shell == 1 and native.helpers == 0
    assert rows[("Native", "predict")].verdict == "native"
    assert rows[("native_function", INV.FUNCTION_OP)].verdict == "native"
    assert rows[("Stateless", "fit")].verdict == "native"            # a no-op
    orch = rows[("Orchestrated", "fit")]
    assert (orch.verdict, orch.native_calls, orch.native_entries, orch.allocs, orch.helpers) == (
        "orchestrated", 2, 2, 1, 1)
    helper = rows[("Helper", "fit")]                                   # one call into a Python helper
    assert helper.verdict == "python" and helper.untagged_loops == 1 and helper.native_calls == 1
    python = rows[("Python", "fit")]
    assert (python.verdict, python.numpy, python.native_calls) == ("python", 1, 0)
    assert rows[("Python", "transform")].verdict == "python"
    driven = rows[("Driven", "fit")]
    assert (driven.verdict, driven.loops, driven.untagged_loops) == ("orchestrated", 1, 0)
    flow = rows[("Flow", "fit")]                                       # binding passed to a helper
    assert (flow.verdict, flow.native_calls, flow.helpers) == ("orchestrated", 1, 1)
    typed = rows[("Typed", "fit")]                                     # methods on a constructed object
    assert typed.native_calls == 1 and typed.helpers == 3 and typed.verdict == "orchestrated"
    assert ("Refuses", "save") not in rows and rows[("Refuses", "fit")].verdict == "native"
    dispatched = rows[("Dispatched", "fit")]                           # the staged one-call shell
    assert (dispatched.verdict, dispatched.native_calls) == ("native", 1)
    assert not any(key[0] == "LIMIT" for key in rows)                  # constants are not operations


def test_module_level_entries():
    """A module-level name bound once at import to a compiled entry is a native entry when called."""
    with tempfile.TemporaryDirectory() as tmp:
        rows = rows_by_key(build(Path(tmp)))
    owned = rows[("Owned", "fit")]                     # `_NATIVE = _binding()._module.small_call`
    assert (owned.verdict, owned.native_calls, owned.helpers) == ("native", 1, 0)
    assert rows[("Owned", "predict")].verdict == "python"      # a Python helper doing work stays python
    connected = rows[("Connected", "fit")]             # `_OWNER, _CTX = _bridge.connect(...)`
    assert (connected.verdict, connected.native_calls) == ("native", 1)
    assert rows[("Connected", "predict")].native_calls == 0    # a local rebinding shadows the entry
    assert rows[("Getattr", "fit")].verdict == "native"        # getattr(binding, "entry")
    assert rows[("Imported", "fit")].verdict == "native"       # `from .owned import _NATIVE`
    wrapped = rows[("Wrapped", "fit")]                 # one entry call, but a Python helper does work
    assert wrapped.verdict == "python" and wrapped.native_calls == 1 and wrapped.helpers == 1


def test_baseline_and_check():
    with tempfile.TemporaryDirectory() as tmp:
        root = build(Path(tmp))
        baseline = root / "baseline.tsv"
        assert INV.main(["--root", str(root), "--baseline", "--baseline-file", str(baseline)]) == 0
        assert INV.main(["--root", str(root), "--check", "--baseline-file", str(baseline)]) == 0
        records = INV.read_tsv(baseline)
        assert {r["verdict"] for r in records} == {"native", "orchestrated", "python"}
        # a native method that starts allocating in Python fails the check
        est = root / "python/mojolearn/est.py"
        est.write_text(est.read_text().replace(
            "        return _backend.binding(\"_mojolearn_x\").x_predict(self, X)",
            "        out = empty((4,), \"<f4\")\n"
            "        return _backend.binding(\"_mojolearn_x\").x_predict(self, X, out)"))
        assert INV.main(["--root", str(root), "--check", "--baseline-file", str(baseline)]) == 1
        problems, _ = INV.check(INV.records_of(INV.inventory_rows(root)), records)
        assert any("regressed native -> orchestrated" in p for p in problems)
        assert any("non-native public methods grew" in p for p in problems)


def test_summary_counts_families():
    records = [
        {"method": "fit", "class": "m.A", "verdict": "native", "family": "f", "native_calls": "1",
         "py_statements": "2"},
        {"method": "predict", "class": "m.A", "verdict": "python", "family": "f", "native_calls": "0",
         "py_statements": "9"},
    ]
    text = INV.summarize(records)
    assert "TOTAL" in text and text.splitlines()[1].split()[:7] == ["f", "1", "2", "1", "0", "1", "0"]


def test_endpoint_ledger_includes_lifecycle_descriptors_aliases_and_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        root = build(Path(tmp))
        base = root / "python/mojolearn"
        (base / "__init__.py").write_text("from .lifecycle import Child, Child as Alias\n__all__ = ['Child', 'Alias']\n")
        (base / "lifecycle.py").write_text(textwrap.dedent('''
            raise RuntimeError("inventory must never import this module")
            class Parent:
                def __init__(self, width, /, *, copy=True):
                    self.width = width
                def state_dict(self):
                    return {"width": self.width}
                def save(self, path):
                    raise NotImplementedError("format unavailable")
                @property
                def value(self):
                    return self.width
                @value.setter
                def value(self, width):
                    self.width = width
                def _private(self):
                    return None
            class Child(Parent):
                checkpoint = Parent.state_dict
                def backward(self, gradient, *, retain_graph=False):
                    return gradient
        '''))
        report = INV.endpoint_inventory(root)
        rows = {r["method"]: r for r in report["endpoints"]}
        assert set(rows) == {"__init__", "state_dict", "save", "value:get", "value:set", "checkpoint", "backward"}
        assert rows["save"]["always_refuses"]
        assert rows["__init__"]["kind"] == "constructor"
        assert rows["checkpoint"]["kind"] == "method_alias"
        assert rows["__init__"]["parameters"] == [
            dict(name="width", kind="positional_only", required=True, default=None),
            dict(name="copy", kind="keyword_only", required=False, default="True")]
        assert rows["value:get"]["source_ast_sha256"] != rows["value:set"]["source_ast_sha256"]
        assert all(set(r["aliases"]) == {"mojolearn.Child", "mojolearn.Alias"} for r in rows.values())
        assert all(e["status"] == "unreviewed" and e["records"] == []
                   for r in rows.values() for e in r["evidence"].values())
        out = root / "endpoints.json"
        assert INV.main(["--root", str(root), "--endpoints-json", str(out)]) == 0
        assert __import__('json').loads(out.read_text())["summary"]["endpoints"] == 7


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("ok", name)
    sys.exit(0)
