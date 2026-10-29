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
    assert (native.verdict, native.native_calls, native.native_entries) == ("orchestrated", 1, 1)
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
    assert (owned.verdict, owned.native_calls, owned.helpers) == ("orchestrated", 1, 0)
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
        assert INV.main(["--root", str(root), "--check", "--baseline-file", str(baseline)]) == 1
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
        assert any("zero required regardless of baseline" in p for p in problems)


def test_existing_baseline_debt_cannot_authorize_product_execution():
    row = {"class": "mojolearn.Estimator", "method": "fit", "module:line": "est.py:1"}
    for verdict in ("python", "orchestrated", "unresolved"):
        debt = dict(row, verdict=verdict)
        problems, _ = INV.check([debt], [debt])
        assert problems and any("zero required" in p for p in problems)
    native = dict(row, verdict="native")
    assert INV.check([native], [dict(row, verdict="python")])[0] == []


def test_returned_framework_callable_is_audited_without_an_export():
    with tempfile.TemporaryDirectory() as tmp:
        root = build(Path(tmp))
        folder = root / "python/mojolearn/importers"
        folder.mkdir()
        (folder / "__init__.py").write_text("")
        path = folder / "_program.py"
        path.write_text(textwrap.dedent('''
            from .. import _backend
            class ImportedProgram:
                def run(self, X):
                    X = [x for x in X]
                    return _backend.binding("_mojolearn").run(X)
        '''))
        rows = INV.records_of(INV.inventory_rows(root))
        returned = [r for r in rows if r["class"].endswith(".ImportedProgram")]
        assert len(returned) == 1 and returned[0]["verdict"] == "python"
        path.write_text("class ImportedProgram:\n    pass\n")
        rows = INV.records_of(INV.inventory_rows(root))
        returned = [r for r in rows if r["class"].endswith(".ImportedProgram")]
        assert len(returned) == 1 and returned[0]["verdict"] == "unresolved"


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


def endpoint_fixture(root, source, exports=("Estimator",)):
    base = root / "python/mojolearn"
    base.mkdir(parents=True)
    (base / "__init__.py").write_text("from .est import " + ", ".join(exports) + "\n__all__ = " + repr(list(exports)) + "\n")
    (base / "_backend.py").write_text("def binding(name): return __import__(name)\n")
    (base / "est.py").write_text("from . import _backend\n_NATIVE = _backend.binding('_mojolearn_test').run\n" + textwrap.dedent(source))
    return base


def test_gate_includes_omitted_query_methods_constructors_and_properties():
    cases = {
        "kneighbors": "def kneighbors(self, X):\n    return [float(x) for x in X]",
        "radius_neighbors": "def radius_neighbors(self, X):\n    return [float(x) for x in X]",
        "search": "def search(self, X):\n    return [float(x) for x in X]",
        "__init__": "def __init__(self, count):\n    self.count = int(count)",
        "value:get": "@property\ndef value(self):\n    return self._value",
    }
    for method, body in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            endpoint_fixture(root, "class Estimator:\n" + textwrap.indent(body, "    ") + "\n")
            report = INV.endpoint_inventory(root)
            assert any(r["method"] == method and r["runtime_status"] == "prohibited_source" for r in report["endpoints"]), report
            assert INV.main(["--root", str(root), "--check"]) == 1


def test_gate_allows_only_tag_metadata_not_calls_into_it():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = endpoint_fixture(root, '''
            class Estimator:
                def __sklearn_tags__(self):
                    tags = dict(estimator_type='regressor')
                    return tags
                def fit(self, X):
                    return _NATIVE(self, X)
        ''')
        assert INV.main(["--root", str(root), "--check"]) == 0
        report = INV.endpoint_inventory(root)
        tags = next(r for r in report["endpoints"] if r["method"] == "__sklearn_tags__")
        assert tags["static_verdict"] == "python" and tags["runtime_status"] == "allowed_tag_metadata"
        path = base / "est.py"
        path.write_text(path.read_text().replace("_NATIVE(self, X)", "_NATIVE(self, self.__sklearn_tags__())"))
        assert INV.main(["--root", str(root), "--check"]) == 1
        path.write_text("class Estimator:\n    def __sklearn_tags__(self): return self.predict([1])\n    def predict(self, X): return [x + 1 for x in X]\n")
        tags = next(r for r in INV.endpoint_inventory(root)["endpoints"] if r["method"] == "__sklearn_tags__")
        assert tags["runtime_status"] == "prohibited_source", tags


def test_unresolved_exports_and_non_algorithm_product_types_cannot_disappear():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = endpoint_fixture(root, "class Estimator:\n    def read(self, X): return [x + 1 for x in X]\n")
        manifest = root / "packaging/native_only/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"inventory": [{"algorithm": "mojolearn.Estimator", "symbol_origin": "mojolearn.est.Estimator", "status": "not_algorithm"}]}))
        assert INV.inventory_rows(root) == []
        assert INV.main(["--root", str(root), "--check"]) == 1
        (base / "__init__.py").write_text("__all__ = ['MissingFactoryProduct']\n")
        rows = INV.endpoint_inventory(root)["endpoints"]
        assert any(r["kind"] == "unresolved_export" for r in rows)
        assert INV.main(["--root", str(root), "--check"]) == 1


def test_import_time_native_function_replacement_supersedes_dead_body():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, '''
            class Estimator:
                def search(self, X):
                    return [float(x) for x in X]
            def install(cls):
                def search(self, X):
                    return _NATIVE(self, X)
                cls.search = search
            install(Estimator)
        ''')
        row = INV.endpoint_inventory(root)["endpoints"][0]
        assert row["runtime_status"] == "native_boundary", row
        assert row["definition"] != row["declared_definition"]
        assert row["import_connections"] and INV.main(["--root", str(root), "--check"]) == 0


def test_native_named_factory_result_is_unresolved_until_installed_slot_qualified():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, '''
            class Estimator:
                def search(self, X):
                    return [float(x) for x in X]
            def install(cls):
                cls.search = _NATIVE('make_callable')
            install(Estimator)
        ''')
        row = INV.endpoint_inventory(root)["endpoints"][0]
        assert row["runtime_status"] == "unresolved", row
        assert row["declared_definition"] and row["definition"] is None
        assert "installed-slot" in row["qualification_debt"][0]
        assert INV.main(["--root", str(root), "--check"]) == 1


def test_class_decorator_installed_constructor_is_followed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, '''
            def connect(kind):
                def decorate(cls):
                    def initialize(self, **kwargs):
                        return _NATIVE(self, kwargs)
                    cls.__init__ = initialize
                    return cls
                return decorate
            @connect('plain')
            class Estimator:
                def __init__(self, count):
                    self.count = int(count)
        ''')
        row = next(r for r in INV.endpoint_inventory(root)["endpoints"] if r["method"] == "__init__")
        assert row["runtime_status"] == "native_boundary", row
        assert INV.main(["--root", str(root), "--check"]) == 0


def test_returned_factory_operations_and_dynamic_decorators_stay_visible():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = endpoint_fixture(root, '''
            class _Result:
                def search(self, X):
                    return [float(x) for x in X]
            def factory():
                return _Result()
        ''', exports=("factory",))
        report = INV.endpoint_inventory(root)
        assert any(r["returned_operation"] and r["method"] == "search" for r in report["endpoints"])
        assert INV.check_endpoints(report)
        (base / "__init__.py").write_text("from .est import Estimator\n__all__=['Estimator']\n")
        (base / "est.py").write_text("from somewhere import dynamic_install\n@dynamic_install\nclass Estimator:\n    def fit(self, X): return self\n")
        rows = INV.endpoint_inventory(root)["endpoints"]
        assert any(r["kind"] == "unresolved_installer" for r in rows)
        assert INV.main(["--root", str(root), "--check"]) == 1


def test_gate_report_keeps_all_failures_while_console_is_bounded():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, "class Estimator:\n" + "\n".join(
            f"    def query_{n}(self, X): return [x + 1 for x in X]" for n in range(45)) + "\n")
        report = root / "report.json"
        assert INV.main(["--root", str(root), "--check", "--endpoints-json", str(report)]) == 1
        data = json.loads(report.read_text())
        assert data["gate"]["prohibited_or_unresolved"] == 45
        assert len(data["gate"]["problems"]) == 45


def test_native_name_and_per_call_binding_selection_are_not_native_proof():
    for operation in ("_native_missing(self, X)", "_backend.binding('_mojolearn_test').run(self, X)"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            endpoint_fixture(root, "class Estimator:\n    def search(self, X): return " + operation + "\n")
            row = INV.endpoint_inventory(root)["endpoints"][0]
            assert row["runtime_status"] != "native_boundary", row
            assert INV.main(["--root", str(root), "--check"]) == 1


def test_transitive_replaced_method_body_is_not_walked_as_live_python():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, '''
            class Estimator:
                def search(self, X): return self.query(X)
                def query(self, X): return [float(x) for x in X]
            Estimator.query = _NATIVE
        ''')
        rows = {r['method']:r for r in INV.endpoint_inventory(root)['endpoints']}
        assert rows['query']['runtime_status'] == 'native_boundary', rows
        assert rows['search']['runtime_status'] == 'native_boundary', rows


def test_unresolved_custom_descriptor_is_not_silently_omitted():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, '''
            from somewhere import Descriptor
            class Estimator:
                value = Descriptor()
        ''')
        rows = INV.endpoint_inventory(root)['endpoints']
        assert len(rows) == 1 and rows[0]['kind'] == 'unresolved_class_attribute', rows
        assert INV.main(['--root', str(root), '--check']) == 1


def test_class_slot_replacement_does_not_relabel_same_named_module_function():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = endpoint_fixture(root, '''
            from . import helpers
            class Estimator:
                def search(self, X): return helpers.query(X)
                def query(self, X): return [float(x) for x in X]
            Estimator.query = _NATIVE
        ''')
        (base / 'helpers.py').write_text('def query(X): return [float(x) for x in X]\n')
        rows = {r['method']:r for r in INV.endpoint_inventory(root)['endpoints']}
        assert rows['query']['runtime_status'] == 'native_boundary', rows
        assert rows['search']['runtime_status'] == 'prohibited_source', rows


def _rows(root):
    return {r["method"]: r for r in INV.endpoint_inventory(root)["endpoints"]}


def test_connection_namespace_fields_are_compiled_entries():
    source = """
        from types import SimpleNamespace
        def _connect(mode):
            module = _backend.binding('_mojolearn_x')
            result = SimpleNamespace(call=module.ops_call, binding=module)
            return result
        _C = {}
        for _mode in ('identical', 'fast'):
            try:
                _C[_mode] = _connect(_mode)
            except ImportError as _error:
                _C[_mode] = type(_error).__name__ + ': ' + str(_error)
        _c = _C['identical']
        if isinstance(_c, str):
            _c = _C['fast']
        _CALL = _c.call
        _RUN = _c.binding.ops_run
        class Estimator:
            def fit(self, X):
                return _CALL(self, X)
            def predict(self, X):
                return _RUN(self, X)
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, source)
        rows = _rows(root)
        assert rows["fit"]["runtime_status"] == "native_boundary", rows["fit"]
        assert rows["fit"]["native_entries"] == ["ops_call"]
        assert rows["predict"]["native_entries"] == ["ops_run"]
    with tempfile.TemporaryDirectory() as tmp:
        # a field rebound to anything but an entry is not native
        root = Path(tmp)
        endpoint_fixture(root, source.replace("            return result", "            result.call = len\n            return result"))
        assert _rows(root)["fit"]["runtime_status"] != "native_boundary"


def test_installer_branch_decided_by_literal_kind():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, """
            import inspect
            _SPECIAL = ('forest',)
            class Estimator:
                def fit(self, X):
                    return _NATIVE(self, X)
            class Forest:
                def fit(self, X):
                    return [float(x) for x in X]
            def register(cls, kind):
                special = kind in _SPECIAL
                if special:
                    for name in ('fit', 'decision_function'):
                        method = getattr(cls, name, None)
                        if inspect.isfunction(method):
                            setattr(cls, name, inspect.unwrap(method))
                elif kind == 'other':
                    cls.decision_function = len
            register(Estimator, 'plain')
            register(Forest, 'forest')
        """, exports=("Estimator", "Forest"))
        endpoints = INV.endpoint_inventory(root)["endpoints"]
        names = {r["endpoint"]: r for r in endpoints}
        assert not any(r["method"] == "decision_function" for r in endpoints), sorted(names)
        assert names["mojolearn.Estimator.fit"]["runtime_status"] == "native_boundary"
        assert names["mojolearn.Forest.fit"]["runtime_status"] == "prohibited_source"


def test_class_creation_hook_wrappers_are_python():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = endpoint_fixture(root, """
            from ._mode import SklearnTagsMixin
            class Estimator(SklearnTagsMixin):
                def fit(self, X):
                    return _NATIVE(self, X)
            class Owned(SklearnTagsMixin):
                _mojolearn_native_public = True
                def fit(self, X):
                    return _NATIVE(self, X)
        """, exports=("Estimator", "Owned"))
        (base / "_mode.py").write_text(textwrap.dedent("""
            _OUTPUTS = ("predict",)
            _FITS = ("fit",)
            _NEEDS_FIT = ("predict",)
            _LOADS = ("load",)
            class PublicOutputsMixin:
                def __init_subclass__(cls, **kw):
                    super().__init_subclass__(**kw)
            class SklearnTagsMixin(PublicOutputsMixin):
                def __init_subclass__(cls, **kw):
                    super().__init_subclass__(**kw)
        """))
        rows = {r["endpoint"]: r for r in INV.endpoint_inventory(root)["endpoints"]}
        wrapped = rows["mojolearn.Estimator.fit"]
        assert wrapped["runtime_status"] == "prohibited_source", wrapped
        assert wrapped["class_creation_wrapper"] == "mojolearn.est.Estimator"
        assert rows["mojolearn.Owned.fit"]["runtime_status"] == "native_boundary"


def test_dataclass_generated_slots_and_later_native_replacement():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, """
            from dataclasses import dataclass
            @dataclass(frozen=True, init=False, repr=False, eq=False)
            class Estimator:
                def run(self):
                    return _NATIVE(self)
            def _set(self, name, value):
                return _NATIVE(self, name, value)
            Estimator.__setattr__ = _set
        """)
        rows = _rows(root)
        assert rows["__delattr__"]["runtime_status"] == "prohibited_source", rows["__delattr__"]
        assert rows["__delattr__"]["generated_by"] == "dataclasses.dataclass"
        assert rows["__setattr__"]["runtime_status"] == "native_boundary", rows["__setattr__"]
        assert "__init__" not in rows and "__repr__" not in rows


def test_property_factory_accessors_are_measured():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, """
            def _word(k):
                def get(self):
                    return _NATIVE(self, k)
                def put(self, value):
                    return _NATIVE(self, k, value)
                return property(get, put)
            class Estimator:
                t = _word(0)
                def _report(name):
                    def get(self):
                        return self.__dict__[name]
                    return property(get)
                seen_ = _report("seen_")
                del _report
        """)
        rows = _rows(root)
        assert rows["t:get"]["runtime_status"] == "native_boundary", rows["t:get"]
        assert rows["t:set"]["runtime_status"] == "native_boundary"
        assert rows["seen_:get"]["runtime_status"] == "prohibited_source", rows["seen_:get"]


def test_compiled_descriptor_factory_slot_is_its_entry():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        endpoint_fixture(root, """
            from types import MethodType as _MethodType
            class Estimator:
                def run(self):
                    return _NATIVE(self)
            def install(cls):
                module = _backend.binding('_mojolearn_x')
                cls.__del__ = module.classical_attribute_descriptor(_MethodType(module.finalize_call, {}))
                cls.__getattribute__ = module.other_factory(_MethodType(module.finalize_call, {}))
            install(Estimator)
        """)
        rows = _rows(root)
        assert rows["__del__"]["runtime_status"] == "native_boundary", rows["__del__"]
        assert rows["__del__"]["native_entries"] == ["finalize_call"]
        assert rows["__getattribute__"]["runtime_status"] == "unresolved", rows["__getattribute__"]



def test_native_constructor_keyword_is_local_to_its_defining_class():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = endpoint_fixture(root, """
            from ._mode import NumericModeMixin
            class Native(NumericModeMixin, _native_constructor=True):
                def __init__(self, *, numeric_mode=None):
                    return _NATIVE(self, numeric_mode)
                def fit(self, X):
                    return _NATIVE(self, X)
            class Passive(Native):
                pass
            class Own(Native):
                def __init__(self, *, numeric_mode=None):
                    return _NATIVE(self, numeric_mode)
            class Disabled(NumericModeMixin, _native_constructor=False):
                def __init__(self, *, numeric_mode=None):
                    return _NATIVE(self, numeric_mode)
            class MissingMode(NumericModeMixin, _native_constructor=True):
                def __init__(self):
                    return _NATIVE(self)
            class Unknown(NumericModeMixin, _native_constructor=True, metaclass=type):
                def __init__(self, *, numeric_mode=None):
                    return _NATIVE(self, numeric_mode)
        """, exports=("Native", "Passive", "Own", "Disabled", "MissingMode", "Unknown"))
        mode = textwrap.dedent("""
            _OUTPUTS = ("predict",)
            _FITS = ("fit",)
            _NEEDS_FIT = ("predict",)
            _LOADS = ("load",)
            def _install_numeric_mode(cls, *, native_constructor=False):
                if native_constructor:
                    return
                cls.__init__ = make_wrapper(cls.__init__)
            class PublicOutputsMixin:
                def __init_subclass__(cls, **kw):
                    super().__init_subclass__(**kw)
            class SklearnTagsMixin(PublicOutputsMixin):
                def __init_subclass__(cls, **kw):
                    super().__init_subclass__(**kw)
            class NumericModeMixin(SklearnTagsMixin):
                def __init_subclass__(cls, *, _native_constructor=False, **kw):
                    super().__init_subclass__(**kw)
                    _install_numeric_mode(cls, native_constructor=_native_constructor)
        """)
        (base / "_mode.py").write_text(mode)
        rows = {row["endpoint"]: row for row in INV.endpoint_inventory(root)["endpoints"]}
        for name in ("Native", "Passive"):
            row = rows["mojolearn." + name + ".__init__"]
            assert row["runtime_status"] == "native_boundary", row
            assert "class_creation_wrapper" not in row, row
        for name in ("Own", "Disabled", "MissingMode"):
            row = rows["mojolearn." + name + ".__init__"]
            assert row["runtime_status"] == "prohibited_source", row
            assert row["class_creation_wrapper"] == "mojolearn.est." + name, row
        assert rows["mojolearn.Native.fit"]["class_creation_wrapper"] == "mojolearn.est.Native"
        for name in ("Native", "Passive", "Disabled"):
            assert "mojolearn." + name + ".<class-construction>" not in rows
        assert rows["mojolearn.Unknown.<class-construction>"]["runtime_status"] == "unresolved"
        assert rows["mojolearn.MissingMode.<class-construction>"]["runtime_status"] == "unresolved"
        # A matching keyword alone cannot exempt a constructor when the indexed
        # hook does not implement/forward the skip protocol.
        (base / "_mode.py").write_text(mode.replace(
            "_install_numeric_mode(cls, native_constructor=_native_constructor)", "_install_numeric_mode(cls)"))
        rows = {row["endpoint"]: row for row in INV.endpoint_inventory(root)["endpoints"]}
        assert rows["mojolearn.Native.__init__"]["class_creation_wrapper"] == "mojolearn.est.Native"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("ok", name)
    sys.exit(0)
