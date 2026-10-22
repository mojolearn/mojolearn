"""Exercise the real inventory functions without importing a native runtime."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def source_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


def inventory(modules):
    ns = dict(__name__="mojolearn._verify", sys=SimpleNamespace(modules=modules),
              os=os, time=time, hashlib=hashlib, _SHA256_MEMO={})
    return source_functions("python/mojolearn/_verify.py", ("binding_artifacts", "_sha256"), ns)["binding_artifacts"]


@pytest.mark.parametrize("name", ["mojolearn._mojolearn", "mojolearn._host._mojolearn_training_host",
                                  "mojolearn._fast._mojolearn_training"])
@pytest.mark.parametrize("source", ["__file__", "__spec__.origin"])
def test_actual_loaded_alias_and_origin_are_hashed(tmp_path, name, source):
    path = tmp_path / "native.so"
    path.write_bytes(b"actual loaded fixture")
    fields = {"__file__": str(path)} if source == "__file__" else {"__spec__": SimpleNamespace(origin=str(path))}
    rows = inventory({name: SimpleNamespace(**fields)})()
    assert rows == [dict(module=name, file=str(path.resolve()), path_source=source,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size,
                         mtime=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
                         artifact_source_commit="unverified")]


@pytest.mark.parametrize("kind", ["no-path", "missing", "empty", "conflict", "python"])
def test_unresolved_native_module_is_an_error(tmp_path, kind):
    path = tmp_path / ("native.py" if kind == "python" else "native.so")
    if kind != "missing":
        path.write_bytes(b"" if kind == "empty" else b"fixture")
    fields = {} if kind == "no-path" else {"__spec__": SimpleNamespace(origin=str(path))}
    if kind == "conflict":
        fields["__file__"] = str(tmp_path / "other.so")
    with pytest.raises(RuntimeError, match="provenance unresolved"):
        inventory({"mojolearn._host._mojolearn_training_host": SimpleNamespace(**fields)})()


def test_routing_placeholders_are_skipped_but_unknown_stubs_are_not(tmp_path):
    class HostBinding(ModuleType):
        pass
    path = tmp_path / "native.so"
    path.write_bytes(b"loaded")
    modules = {"mojolearn._backend": SimpleNamespace(_HostBinding=HostBinding),
               "mojolearn._mojolearn_training": HostBinding("proxy"),
               "mojolearn._host._mojolearn_training_host": SimpleNamespace(__spec__=SimpleNamespace(origin=str(path)))}
    assert len(inventory(modules)()) == 1
    modules["mojolearn._mojolearn_unknown"] = ModuleType("unknown")
    with pytest.raises(RuntimeError, match="provenance unresolved"):
        inventory(modules)()


def test_installed_but_unloaded_files_never_enter_the_inventory(tmp_path):
    (tmp_path / "_mojolearn_unused.so").write_bytes(b"not loaded")
    with pytest.raises(RuntimeError, match="no loaded native"):
        inventory({})()
    # A controller may load no native files when its workers do all operations.
    assert inventory({})(require_loaded=False) == []


@pytest.mark.parametrize("report,complete", [({}, False), ({"bindings": []}, False),
    ({"bindings": [{}], "bindings_error": "unresolved"}, False),
    ({"bindings": [{}]}, False),
    ({"bindings": [dict(module="native", file="native.so", sha256="a" * 64, size=1)]}, True)])
def test_report_cannot_claim_complete_with_unresolved_native_bytes(report, complete):
    fn = source_functions("python/mojolearn/_verify_all.py", ("_native_provenance_complete",), {})["_native_provenance_complete"]
    assert fn(report) is complete
    assert report["native_provenance_complete"] is complete


@pytest.mark.parametrize("binding,error", [(None, None), ({}, None),
    ({"module": "native", "file": "native.so", "sha256": "x" * 64, "size": 1}, None),
    ({"module": "native", "file": "native.so", "sha256": "a" * 64, "size": 1}, "unresolved")])
def test_new_protocol_reference_rejects_missing_or_bad_native_receipts(binding, error):
    spec = importlib.util.spec_from_file_location("owner_reference_admit", ROOT / "python/mojolearn/_verify_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(cells={"ols/base": {"verdict": "STABLE", "hashes": ["a" * 16]}},
                  mode="identical", commit="1334beeedd92", vendor="cpu-amd-epyc-9754-128-core-processor",
                  package=dict(binding_provenance_protocol="loaded-native-v2", bindings=[] if binding is None else [binding], bindings_error=error),
                  host={"families": {"_mojolearn_estimators_host": {"column": "cpu", "sabotage": False}}})
    assert module.admit(record, "records/cpu.json") == "loaded native binding provenance incomplete"
