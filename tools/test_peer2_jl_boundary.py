"""Source controls for the public JL scalar door; runtime probes are separate."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_jl_only_forwards_original_inputs():
    source = ast.parse((ROOT / "python/mojolearn/_expansion_decomp.py").read_text())
    method = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                  and n.name == "johnson_lindenstrauss_min_dim")
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert len(calls) == 1
    assert calls[0].func.id == "_JL_CALL"
    assert [n.id for n in calls[0].args[2].elts] == ["n_samples", "eps"]


def test_native_jl_has_no_python_numeric_helper_or_caller_protocol_dispatch():
    owner = (ROOT / "bindings/jl_native_pack.mojo").read_text()
    # 2f67f62a2e: both scalars go through the admitted word decoder (eps as a float, n_samples as a count),
    # with the class word read once from the import-resolved table instead of a Python call per scalar.
    assert 'var e = jl_scalar_word(eps._obj_ptr, jl_type_word(eps._obj_ptr), True,' in owner
    assert 'var n = jl_scalar_word(n_samples._obj_ptr, jl_type_word(n_samples._obj_ptr), False,' in owner
    assert 'PyLong_FromDouble' in owner
    for callback in ['import_module("operator")', 'b.float', 'b.int', 'py_call_typed', '__float__', '__int__']:
        assert callback not in owner
    assert '(e * e) / 2.0 - _cube_rounded(e) / 3.0' in owner
    assert '4.0 * _pinned_log(n) / denominator' in owner


def test_scalar_metadata_requires_exact_native_admission_before_buffer_decode():
    provider = (ROOT / "providers/python/jl_scalar_pyapi_unpack.mojo").read_text()
    assert '_one_of(cls, context["scalar_types"])' in provider
    assert 'if not scalar and not array:' in provider
    assert provider.index('if not scalar and not array:') < provider.index('native_scalar_buffer(value)')
    assert 'decoded.complex' in provider
    assert 'real.highest() + real.scale >= 0' in provider
    # 2f67f62a2e: a built-in float is read through the CPython C API, never a Python conversion protocol.
    assert 'table[].PyFloat_AsDouble(value)' in provider
    assert 'Float64(py=' not in provider
    assert 'return bitcast[DType.float64](real.float64_bits())' in provider
    assert 'value.__' not in provider


def test_random_projection_uses_the_same_admitted_jl_owner():
    owner = (ROOT / "bindings/rp_native_door.mojo").read_text()
    assert 'jl_min_dim(PythonObject(n), eps, context)' in owner
    bridge = (ROOT / "python/mojolearn/_native_jl_bridge.py").read_text()
    assert 'context = _numpy_free_context()' in bridge
    assert 'context["native_module_type"] = ModuleType' in bridge
