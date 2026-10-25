"""CPU-only public-door and callback-regression tests; no package/binary import."""
import ast
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python/mojolearn"
SHELLS = {
    "parallel_neighbors_reference.py": {"ReferenceShardedNeighbors": (
        "__init__", "kneighbors", "predict", "predict_proba", "close", "__enter__", "__exit__")},
    "parallel_neighbors.py": {"ParallelQueries": ("__init__", "query", "close", "__enter__", "__exit__")},
    "parallel_ivf.py": {"DistributedIVFIndex": ("__init__", "from_index", "search", "close", "__enter__", "__exit__")},
}


def load_shell(filename):
    calls = []
    result = object()

    def native(operation, args):
        calls.append((operation, args))
        return None if operation.endswith("_init") else result

    tree = ast.parse((PACKAGE / filename).read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    namespace = {"_native_parallel_shard": native}
    exec(compile(tree, filename, "exec"), namespace)
    return namespace, calls, result


class Hostile:
    """The Python door must pass even invalid inputs untouched to admission."""
    def __getattribute__(self, name):
        raise AssertionError("Python inspected input: " + name)

    def __bool__(self):
        raise AssertionError("Python tested input truth")

    def __int__(self):
        raise AssertionError("Python converted input")

    def __iter__(self):
        raise AssertionError("Python iterated input")


@pytest.mark.parametrize("filename", SHELLS)
def test_public_methods_have_exactly_one_native_handoff(filename):
    tree = ast.parse((PACKAGE / filename).read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in SHELLS[filename][node.name]:
                continue
            body = [n for n in method.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            assert len(body) == 1, (filename, method.name)
            assert isinstance(body[0], (ast.Return, ast.Expr))
            call = body[0].value
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            assert call.func.id == "_native_parallel_shard"
            assert len([n for n in ast.walk(call) if isinstance(n, ast.Call)]) == 1


def test_reference_queries_forward_without_touching_estimator_or_output():
    ns, calls, result = load_shell("parallel_neighbors_reference.py")
    value = Hostile()
    model = ns["ReferenceShardedNeighbors"](value, devices=value,
        reference_rows_per_shard=value, query_rows_per_shard=value)
    assert calls[-1][0] == "reference_init"
    assert all(arg is value for arg in calls[-1][1][1:])
    assert model.kneighbors(value, value, value) is result
    assert calls[-1] == ("reference_query", (model, value, "kneighbors", value, value))
    assert model.predict(value) is result
    assert calls[-1] == ("reference_query", (model, value, "predict", None, True))
    assert model.predict_proba(value) is result
    assert model.__enter__() is result
    model.close()
    assert calls[-1] == ("holder_close", (model,))


def test_query_keywords_and_default_method_reach_native_unchanged():
    ns, calls, result = load_shell("parallel_neighbors.py")
    value = Hostile()
    model = ns["ParallelQueries"](value, devices=value, rows_per_shard=value)
    assert calls[-1] == ("queries_init", (model, value, value, value))
    assert model.query(value, method=value, n_neighbors=value, sort_results=value) is result
    operation, args = calls[-1]
    assert operation == "parallel_queries" and args[:3] == (model, value, value)
    assert args[3]["n_neighbors"] is value and args[3]["sort_results"] is value
    assert model.query() is result
    assert calls[-1][1] == (model, None, None, {})


def test_ivf_admission_and_scaler_inverse_never_run_python_protocols():
    ns, calls, result = load_shell("parallel_ivf.py")
    cls = ns["DistributedIVFIndex"]
    value = Hostile()
    assert cls.from_index(value, devices=value) is result
    assert calls[-1] == ("ivf_from_index", (cls, value, value))
    ns, calls, result = load_shell("parallel_preprocessing.py")
    assert ns["transform_scaler"](value, value, devices=value, columns_per_shard=value, inverse=value) is result
    assert calls[-1] == ("transform_scaler", (value, value, value, value, value))


def test_binding_snapshot_retains_compiled_entries_and_refuses_python_callbacks():
    tree = ast.parse((PACKAGE / "_native_parallel_shard.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_connect_binding")
    binding = ModuleType("example")
    binding.query = len  # Native builtin stands in for an extension door.
    backend_calls = []
    backend = SimpleNamespace(binding=lambda *args: backend_calls.append(args) or binding)
    ns = {"_backend": backend}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "connection", "exec"), ns)
    connected = ns["_connect_binding"]("example", ("query",), ("optional",))
    assert type(connected) is ModuleType and connected.query is len
    assert "__getattr__" not in vars(connected)
    binding.query = lambda value: value
    assert connected.query is len and len(backend_calls) == 1
    with pytest.raises(TypeError, match="requires compiled"):
        ns["_connect_binding"]("example", ("query",))


def test_worker_has_no_python_callback_registry_or_dynamic_estimator_call():
    owner = (ROOT / "bindings/parallel_shard_native.mojo").read_text()
    worker = (ROOT / "bindings/parallel_neighbor_worker.mojo").read_text()
    assert not (PACKAGE / "_parallel_shard_ops.py").exists()
    for forbidden in ('operation == "py"', '_t("py_ops")', "_py_request", "py_call(function, args)"):
        assert forbidden not in owner
    assert "parallel_neighbor_worker(operation, args," in owner
    for forbidden in ("Python.import_module", "getattr(state", "_extension()", "_partial_search(", ".kneighbors(", ".predict("):
        assert forbidden not in worker
    assert 'family_entry(binding, "ivf_flat_partial_search", context)' in worker
    assert "finally:\n            knn_release(model)" in worker
    assert "parallel_public_result(result, method, kind, context)" in owner


def test_distributed_receipt_inventory_names_only_existing_modules():
    path = PACKAGE / "_verify_distributed.py"
    spec = importlib.util.spec_from_file_location("isolated_distributed_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all((PACKAGE / name).is_file() for name in module.PROFILE_FILES)
    assert "_parallel_shard_ops.py" not in module.PROFILE_FILES
