"""Public handoff and import-time route admission; no package/binary import."""
import ast
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python/mojolearn"
QUERIES = (
    ("neighbors.py", "NearestNeighbors", "kneighbors", 3),
    ("neighbors.py", "RadiusNeighbors", "radius_neighbors", 4),
    ("_ivf_impl.py", "IVFIndex", "search", 2),
)


class Hostile:
    def __getattribute__(self, name):
        raise AssertionError("Python inspected query input: " + name)

    def __bool__(self):
        raise AssertionError("Python converted query flag")

    def __int__(self):
        raise AssertionError("Python converted query count")

    def __iter__(self):
        raise AssertionError("Python iterated query input")


@pytest.mark.parametrize("filename,classname,name,argc", QUERIES)
def test_public_query_forwards_untouched_to_one_native_call(filename, classname, name, argc):
    tree = ast.parse((PACKAGE / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert [ast.unparse(n) for n in method.decorator_list] == ["_native_query_method"]
    assert sum(isinstance(n, ast.Call) for n in ast.walk(method)) == 1
    decorator_tree = ast.parse((PACKAGE / "_native_robust_bridge.py").read_text())
    decorator = next(n for n in decorator_tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_method")
    calls, sentinel, context = [], object(), object()

    def native(model, operation, args, received_context):
        calls.append((model, operation, args, received_context))
        return sentinel

    namespace = {"_NATIVE_QUERY": native, "_QUERY_CONTEXT": context}
    exec(compile(ast.Module([decorator], type_ignores=[]), "native_method", "exec"), namespace)
    namespace["_native_query_method"] = namespace["native_method"]
    exec(compile(ast.Module([method], type_ignores=[]), filename, "exec"), namespace)
    method = namespace[name]
    for flag in ("_mojolearn_public_out", "_mojolearn_tracks_fit", "_mojolearn_needs_fit", "_mojolearn_validates"):
        assert getattr(method, flag) is True
    model, value = Hostile(), Hostile()
    assert method(model, *([value] * argc)) is sentinel
    assert len(calls) == 1 and calls[0][0] is model and calls[0][1] == name
    assert calls[0][3] is context and all(arg is value for arg in calls[0][2])


def bridge_namespace():
    tree = ast.parse((PACKAGE / "_native_neighbor_queries.py").read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    namespace = {"__package__": "neighbor_query_test_package"}
    exec(compile(tree, "_native_neighbor_queries.py", "exec"), namespace)
    return namespace


def test_proxy_resolution_occurs_only_while_capturing_builtin_entries():
    ns = bridge_namespace()
    resolutions = []

    class Proxy:
        def __getattr__(self, name):
            resolutions.append(name)
            if name == "search":
                return len
            raise ImportError(name)

    proxy = Proxy()
    result = ns["_capture"](proxy, "test.native", ("search", "optional"))
    assert type(result) is ModuleType and vars(result)["search"] is len
    assert resolutions == ["search", "optional"]
    assert "__getattr__" not in vars(result)
    assert ns["_BINDING_ALIASES"][id(proxy)] == (proxy, result)
    assert ns["_BINDING_ALIASES"][id(result)] == (result, result)
    assert result.search([1]) == 1 and resolutions == ["search", "optional"]


def test_python_callable_cannot_enter_native_route_table():
    ns = bridge_namespace()
    module = ModuleType("test.invalid")
    module.search = lambda value: value
    with pytest.raises(TypeError, match="must be compiled"):
        ns["_capture"](module, "invalid", ("search",))
    assert id(module) not in ns["_BINDING_ALIASES"]


def test_cpu_routes_use_only_explicit_host_loader(monkeypatch):
    import sys
    ns = bridge_namespace()
    package = ModuleType("neighbor_query_test_package")
    backend = ModuleType(package.__name__ + "._backend")
    calls = []
    backend.default_mode = lambda: "identical"

    def host(name):
        calls.append(name)
        module = ModuleType(name)
        module.knn_search = len
        return module

    def gpu(*args):
        raise AssertionError("CPU query asked for a GPU route")

    backend.load_host_module, backend.binding = host, gpu
    package._backend = backend
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    routes = ns["_routes"](("base",), {"_mojolearn": "core_host"})
    assert calls == ["core_host"] and tuple(routes["base"]) == ("identical",)
    assert routes["base"]["identical"].knn_search is len


def test_native_owner_replaces_lazy_python_getter_and_keeps_host_classes():
    tree = ast.parse((PACKAGE / "_ivf_impl.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IVFIndex")
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in cls.body)
    host = (PACKAGE / "_classical_host.py").read_text()
    for name in ("HostNearestNeighbors", "HostKNeighborsClassifier", "HostKNeighborsRegressor", "HostRadiusNeighbors", "HostIVFIndex"):
        assert name in host[host.index("_connect_neighbor_host({"):]
    native = (ROOT / "bindings/neighbor_query_native.mojo").read_text()
    assert "Python.import_module" not in native
    for path in ("ivf_flat_index_export", "ivf_flat_index_prepare", "ivf_flat_index_search", "ivf_flat_search"):
        assert path in native
    assert "_resident_owner" in native and "_ivf_filter" in native


def test_shared_scalar_admission_does_not_import_python_at_runtime():
    scalar = (ROOT / "bindings/host_inference_scalars.mojo").read_text()
    assert 'Python.import_module("builtins")' not in scalar
