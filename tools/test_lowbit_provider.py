"""Public lowbit frontdoor and operation-local admission source guards."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_all_public_lowbit_bodies_enter_native_once():
    tree=ast.parse((ROOT/'python/mojolearn/lowbit.py').read_text())
    seen=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.FunctionDef):continue
        body=node.body
        if isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant):body=body[1:]
        assert len(body)==1 and isinstance(body[0],ast.Return),node.name
        call=body[0].value
        assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='_NATIVE',node.name
        seen.append(node.name)
    assert len(seen)==11


def test_lowbit_uses_existing_numerical_provider_not_public_python_wrappers():
    source=(ROOT/'providers/python/lowbit_weight_pyapi.mojo').read_text()
    assert 'lin_connected_binding(context)' in source and 'lin_call(' in source
    assert 'neural_descriptor_connect' not in source and 'struct ' not in source
    assert 'linalg_ops_operation(' not in source
    assert 'low_pack_admitted(input^' in source
    assert 'def low_binding(mut binding: PythonObject' in source
    assert source.count('lin_connected_binding(context)')==1
    assert 'if binding is None:' in source
    assert 'low_admit(native_instance_item(fields' in source
    assert 'native_storage_allocate(context[' in source
    assert 'family_low_' in (ROOT/'bindings/family_native.mojo').read_text()


def test_once_captured_foreign_metadata_retains_zero_copy_owner():
    source=(ROOT/'bindings/linalg_ops_buffers.mojo').read_text()
    section=source[source.index('struct LinAdmission'):]
    assert section.count('AdmittedBorrowedBuffer(value)')==1
    assert 'classical_admit_value(value, context)' in section
    assert 'PyMemoryView_FromMemory' in section
    assert 'PythonObject(alloc=LinBufferPin(input.pin^))' in section
    assert 'py_setattr(array, "_pin", pin)' in section
    assert 'py_setattr(array, "_store", input.keep)' in section
    assert 'return lin_borrow_array(input^, typestr, context)' in section
    assert 'lin_convert(input.view, typestr, result.addr, name)' in section


def test_sequence_target_preserves_direct_float_storage_and_other_callers():
    source=(ROOT/'bindings/sequential_inputs.mojo').read_text()
    assert 'float_target: Bool = False' in source
    assert 'context, float_target)' in source
    assert 'if not float_target and (' in source
    assert 'native_array_from_values(String("f"), values)' in source
    low=(ROOT/'providers/python/lowbit_weight_pyapi.mojo').read_text()
    assert 'sequential_sequence_f32(value, context)' in low
