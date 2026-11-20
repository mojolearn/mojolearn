"""Public lowbit frontdoor and operation-local admission source guards."""
import ast
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from boundary_keys_plain import plain  # noqa: E402


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
    # bw2-linalg: the door reads held keys and leaf codes; plain() gives the by-name form.
    rel='providers/python/lowbit_weight_pyapi_door.mojo'
    source=plain(rel, (ROOT/rel).read_text())
    assert 'lin_connected_binding(context)' in source and 'lin_call(' in source
    assert 'neural_descriptor_connect' not in source and 'struct ' not in source
    assert 'linalg_ops_operation(' not in source
    assert 'low_pack_admitted(input, input_pin^' in source
    assert 'def low_binding(mut binding: PythonObject' in source
    assert source.count('lin_connected_binding(context)')==1
    assert 'if binding is None:' in source
    assert 'low_admit(native_instance_item(fields' in source
    assert 'native_storage_allocate(context[' in source
    # 83f9f8918f: family_low_* resolve to FR_LOW in the one operation table; the family door routes FR_LOW here
    # (bw2-linalg: by its leaf code; a name the table lacks gets the owner's unknown-operation text).
    assert 'elif operation.startswith("family_low_"):\n        owner = FR_LOW' in (ROOT/'bindings/native_operation_codes.mojo').read_text()
    assert ('if owner == FR_LOW:\n        if leaf == OP_LEAF_NONE:\n            lowbit_weight_unknown_operation(operation)\n'
            '        return lowbit_weight_operation(model, leaf, args, context)') in (ROOT/'bindings/family_native_door.mojo').read_text()


def test_once_captured_foreign_metadata_retains_zero_copy_owner():
    source=(ROOT/'bindings/admitted_input_unpack.mojo').read_text()
    section=source[source.index('struct AdmittedTypedPin'):]
    assert section.count('AdmittedBorrowedBuffer(value)')==1
    assert 'classical_admit_value(value, context)' in section
    assert 'PyMemoryView_FromMemory' in section
    # The admitted pin is moved out of the admission and into the Python pin
    # object (the mode-parameterized AdmittedBufferPin, a2b5592a6d).
    assert 'swap(held_pin, input.pin)' in section
    assert 'PythonObject(alloc=AdmittedBufferPin[](held_pin^))' in section
    assert 'py_setattr(array, "_pin", pin)' in section
    assert 'py_setattr(array, "_store", input.keep)' in section
    assert 'return _typed_borrow_array(view, input^, typestr, context)' in section
    assert 'admitted_typed_convert(view, typestr, result.addr, name)' in section


def test_sequence_target_preserves_direct_float_storage_and_other_callers():
    # 3a2e196fca: the sequence capture reads each scalar once; a float32 target stores integers as floats directly.
    source=(ROOT/'bindings/sequential_inputs_door.mojo').read_text()
    assert 'float32_target: Bool = False' in source                     # every other caller keeps its kinds
    assert '_sequence_capture_impl[False](mods, source, context, float_target, items)' in source
    assert 'elif kind == 2 or float_target:' in source
    f32=source.split('def sequential_sequence_f32(mut mods', 1)[1].split('\ndef ', 1)[0]
    assert 'sequential_sequence_native(mods, value, context, True)' in f32
    low=(ROOT/'providers/python/lowbit_weight_pyapi_door.mojo').read_text()
    assert 'sequential_sequence_f32(value, context)' in low
