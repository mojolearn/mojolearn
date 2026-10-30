"""Non-building source checks. Actual public execution remains a separate gate."""
from pathlib import Path

from check_generation_native_boundary import ROOT, audit


def source(path):
    return (ROOT / path).read_text()


def test_native_dependency_closure_has_no_explicit_evaluation_or_missing_source():
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


def test_default_inventory_checks_provider_execution_instead_of_exempting_folder(tmp_path):
    (tmp_path / "providers").mkdir()
    (tmp_path / "entry.mojo").write_text("from providers.algorithm import run\n")
    (tmp_path / "providers" / "algorithm.mojo").write_text(
        'from std.python import Python as P\ndef run():\n    return P.evaluate("1+2")\n')
    _, problems = audit(tmp_path, ("entry",))
    assert problems == [(("entry", "providers.algorithm"), "explicit Python evaluator call P.evaluate at line 3")]


def test_connection_copies_table_once_and_retains_both_owners():
    provider = source("providers/python/generation_session_pyapi.mojo")
    assert '"PyCapsule_IsValid"' in provider and '"PyCapsule_GetPointer"' in provider
    assert "GenerationNativeProvider(module, capsule, factory, factory.__self__, admit_session, table[0])" in provider
    assert "GenerationNativeSession(provider, session, handle)" in provider
    assert "held[].provider is provider and held[].owner is session" in provider


def test_busy_is_reserved_before_gil_release_and_finalized_on_both_outcomes():
    provider = source("providers/python/generation_session_pyapi.mojo").split("def generation_native_forward(")[1]
    assert provider.index("generation_session_forward(0,") < provider.index("with GILReleased")
    assert provider.index("with GILReleased") < provider.index("generation_session_forward(1,")
    assert provider.count("generation_session_forward(2,") == 2
    assert "except problem:" in provider
    native = source("transformer/session_native_dispatch.mojo")
    refusal = native.split("    except problem:")[1]
    assert "busy = False" not in refusal  # never unlock a refused reservation


def test_old_and_direct_doors_share_the_exact_numerical_execution_function():
    door = source("bindings/_mojolearn_transformer.mojo")
    body = door.split("def transformer_session_forward_binding(")[1].split("def transformer_native_api_binding")[0]
    native = source("transformer/session_native_dispatch.mojo")
    assert "transformer_session_forward_native(owner[], a, p, opts)" in body
    assert "transformer_session_forward_native(owner[], a, p, opts)" in native
    assert native.count("return _transformer_run_session(owner, a,") == 1
    assert "owner.clear()" in native
    assert "error_sink(error_context, Int(bytes.unsafe_ptr()), len(bytes))" in native


def test_candidate_stays_opt_in_and_uses_native_vectors():
    public = source("bindings/causal_lm_public.mojo")
    assert public.count('getenv("MOJOLEARN_GENERATION_NATIVE_SESSION")') == 1
    assert 'mode == NUMERIC_IDENTICAL and (String(getenv("MOJOLEARN_GENERATION_NATIVE_SESSION")) == "1"' in public
    branch = public.split('if native_provider is not None and not head_profile')[1].split("        else:\n", 1)[0]
    assert "generation_native_forward(linked, a_words, p_words)" in branch
    assert "py_ints(" not in branch
    assert 'py_call(e["forward_session"]' not in branch
    assert 'not py_contains(e, "causal_lm_session_run") or not _resident_norm_ok(model)' in public
    # No-final-norm admission is a separately compiled opt-in capability.
    # Its helper preserves the default refusal while supporting that flag.
    norm = public.split("def _resident_norm_ok(", 1)[1].split("\ndef ", 1)[0]
    assert "comptime if _LM_RESIDENT_GENERATE:" in norm
    assert "return model._norm is not None" in norm
    assert 'comptime _LM_RESIDENT_GENERATE = is_defined["MOJOLEARN_IDN_LM_RESIDENT_GENERATE"]()' in public
