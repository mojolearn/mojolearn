"""Non-building source checks. Actual public execution remains a separate gate."""
from pathlib import Path

from check_generation_native_boundary import ROOT, audit


def source(path):
    return (ROOT / path).read_text()


def test_native_dependency_closure_has_no_missing_source():
    seen, problems = audit()
    assert "transformer.transformer_binding_impl" in seen
    assert not problems, problems


def test_optional_pure_root_assertion_finds_transitive_imports(tmp_path):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "entry.mojo").write_text("from runtime.bridge import run\n")
    (tmp_path / "runtime" / "bridge.mojo").write_text("from std.python import PythonObject as Hidden\n")
    _, problems = audit(tmp_path, ("runtime.entry",), require_native_only=True)
    assert (("runtime.entry", "runtime.bridge", "std.python"), "provider/Python dependency") in problems


def test_optional_pure_root_assertion_rejects_cpython_spelling(tmp_path):
    (tmp_path / "entry.mojo").write_text('from std.ffi import external_call\nexternal_call["PyObject_Call", Int](a, b, c)\n')
    _, problems = audit(tmp_path, ("entry",), require_native_only=True)
    assert (("entry",), "Python/CPython helper token in native code") in problems


def test_optional_pure_root_assertion_and_missing_dependencies(tmp_path):
    (tmp_path / "entry.mojo").write_text("from providers.python.hidden import run\nfrom absent.native import run\n")
    _, problems = audit(tmp_path, ("entry",), require_native_only=True)
    assert {reason for _, reason in problems} == {"provider/Python dependency", "unresolved local dependency"}


def test_default_dependency_inventory_allows_native_capi_in_providers(tmp_path):
    (tmp_path / "providers").mkdir()
    (tmp_path / "entry.mojo").write_text("from providers.execution import run\n")
    (tmp_path / "providers" / "execution.mojo").write_text(
        'from std.python import PythonObject\nfrom std.ffi import external_call\n'
        'def run(value):\n    return external_call["Py_INCREF", Int](value)\n')
    seen, problems = audit(tmp_path, ("entry",))
    assert "providers.execution" in seen
    assert not problems, problems


# Lane transformer-session-native (2026-11-12): the generation session ABI
# (providers/python/generation_session_pyapi.mojo, transformer/session_abi.mojo,
# transformer/native_session_call.mojo, `transformer_native_api_v1`) is deleted;
# a layer's session forward is the block table's op 11 on the session handle.
def test_session_reserved_before_gil_release_and_released_on_both_outcomes():
    run = source("providers/python/transformer_block_session_pyapi.mojo").split("def transformer_block_session_run(")[1]
    assert run.index("NEURAL_BLOCK_RESERVE") < run.index("with GILReleased")
    assert run.count("NEURAL_BLOCK_RELEASE") == 2
    assert "except problem:" in run
    native = source("transformer/native_block_dispatch.mojo")
    assert "return transformer_session_forward_native(owner[], expanded, p, opts)" in native
    session = source("transformer/session_native_dispatch.mojo")
    assert session.count("return _transformer_run_session(owner, a,") == 1
    assert "owner.clear()" in session


def test_gpu_layer_session_uses_native_vectors():
    public = source("bindings/causal_lm_public.mojo")
    assert 'getenv("MOJOLEARN_GENERATION_NATIVE_SESSION")' not in public
    branch = public.split('elif device15 or (py_contains(e, "block_session")')[1].split("        else:\n", 1)[0]
    assert "transformer_block_session_run(session," in branch
    assert "py_ints(" not in branch
    assert "py_call(" not in branch
    assert 'not py_contains(e, "resident") or not _resident_norm_ok(model)' in public
    # No-final-norm admission is a separately compiled opt-in capability.
    # Its helper preserves the default refusal while supporting that flag.
    norm = public.split("def _resident_norm_ok(", 1)[1].split("\ndef ", 1)[0]
    assert "comptime if _LM_RESIDENT_GENERATE:" in norm
    assert "return model._norm is not None" in norm
    assert 'comptime _LM_RESIDENT_GENERATE = is_defined["MOJOLEARN_IDN_LM_RESIDENT_GENERATE"]()' in public
