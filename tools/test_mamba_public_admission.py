"""Mamba public boundary source regressions; native runtime is tested separately."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ast.parse((ROOT / "python/mojolearn/_mamba_impl.py").read_text())
NATIVE = (ROOT / "bindings/mamba_public_native.mojo").read_text()


def _method(cls, name):
    owner = next(n for n in PUBLIC.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = method.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def _single_native_call(cls, name, operation):
    body = _method(cls, name)
    assert len(body) == 1 and isinstance(body[0], ast.Return), (cls, name)
    call = body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert call.func.attr == "mamba_native_call"
    assert call.args[0].value == operation


def test_all_mamba_state_constructors_are_one_native_operation():
    for cls in ("Mamba1State", "Mamba2State", "Mamba3State"):
        _single_native_call(cls, "__init__", "state_init")


def test_decode_session_properties_lifecycle_and_outputs_are_native():
    for cls in ("Mamba1DecodeSession", "_Mamba2DecodeSession", "_Mamba3DecodeSession"):
        for method, operation in (("state", "session_state"), ("__enter__", "session_enter"),
                                  ("__exit__", "session_exit"), ("__del__", "session_drop")):
            _single_native_call(cls, method, operation)
    for cls in ("_Mamba2DecodeSession", "_Mamba3DecodeSession"):
        _single_native_call(cls, "step", "session_step_public")


def test_record_fields_are_admitted_before_attribute_access():
    body = NATIVE.split("def _mamba_record_fields(")[1].split("def _mamba_scalar(")[0]
    assert body.index("py_type(value) is not expected") < body.index("return _fields(value)")
    assert "__getattribute__" in body and "__setattr__" in body and "__getattr__" in body
    assert "replaced field descriptor" in body
    assert "block.__dict__" not in NATIVE and "state.__dict__" not in NATIVE
    assert "session.__dict__" not in NATIVE


def test_checkpoint_admission_precedes_publication():
    body = NATIVE.split("def mamba3_setstate(")[1].split("def _shape_of(")[0]
    assert body.index("py_type(state) is not bi.dict") < body.index("state.keys()")
    assert body.index("py_type(keys[i]) is not bi.str") < body.index("candidate_fields.update")
    assert body.index("neural_block_fields(candidate, context)") < body.index("fields.update, candidate_fields")


def test_mutable_state_buffers_have_retained_operation_snapshots():
    body = NATIVE.split("def _mamba_buffer(")[1].split("def _mamba_state_names(")[0]
    assert "return neural_array_snapshot(value, context)" in body
    assert body.index("admit_native_buffer_exporter(value)") < body.index(".memoryview, value")
    forward = NATIVE.split("def _mamba3_forward_captured(")[1].split("def mamba3_allocate(")[0]
    assert "_m3_state_call(block, x, state, step, entries, fields, captured)" in forward
    assert 'py_setitem(slots, "buffered_tokens", captured["buffered_tokens"])' in forward
    assert 'py_setitem(slots, "pending", captured["pending"])' in forward


def test_session_reports_are_retained_before_returning_output():
    body = NATIVE.split('if operation == "session_step_public":')[1].split('elif operation != "setstate":')[0]
    assert body.index('py_setitem(block_fields, "h_last_", result[1])') < body.index("return result[0]")
    for name in ("k_last_", "v_last_", "theta_last_"):
        assert f'py_setitem(block_fields, "{name}"' in body
