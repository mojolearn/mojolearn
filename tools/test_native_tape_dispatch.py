"""Immutable tape ownership source contracts; runtime qualification is separate."""
from pathlib import Path
from check_generation_native_boundary import audit
ROOT = Path(__file__).resolve().parents[1]


def test_tape_execution_closures_are_native():
    for module in ('transformer.native_tape_dispatch', 'mamba.native_tape_dispatch'):
        _, failures = audit(ROOT, (module,))
        assert not failures, failures


def test_tape_consumption_and_original_numerical_owners_are_preserved():
    transformer = (ROOT/'transformer/native_tape_dispatch.mojo').read_text()
    mamba = (ROOT/'mamba/native_tape_dispatch.mojo').read_text()
    for source in (transformer, mamba):
        assert 'tape[].consumed = True' in source
        assert 'requires a reserved owner' in source
        assert source.index('if op == 4:') < source.index('if tape[].consumed:')
    assert '_transformer_run_session[True, True]' in transformer
    assert '_transformer_saved_backward(tape[], a)' in transformer
    assert transformer.index('tape[].consumed = True') < transformer.index('_transformer_saved_backward(tape[], a)')
    assert '_m3_prefill_run(tape[].owner' in mamba and 'force_retain=True' in mamba
    assert '_m3_backward_tape_run(tape[].owner, a)' in mamba
    backward = mamba.split('if op != 3')[1]
    assert backward.index('tape[].consumed = True') < backward.index('_m3_backward_tape_run')
    assert 'tape[].owner.usable = False' in backward and 'tape[].owner.release()' in backward


def test_provider_retains_tape_origin_instead_of_rereading_replaced_bindings():
    source = (ROOT/'bindings/neural_prefill_native.mojo').read_text()
    assert 'native, PythonObject(opaque), PythonObject(tape_base)' in source
    assert source.count('neural_descriptor_reserved_execute(') == 3
    backward = source.split('def neural_prefill_backward')[1].split('def neural_prefill_close')[0]
    assert 'var base = Int(py=tape[6])' in backward
    assert 'immutable tape belongs to another block family' in backward
    assert 'neural_descriptor_reserved_execute(tape[4], tape[1], Int(py=tape[5])' in backward
    assert 'neural_descriptor_handle(' not in backward
    for family, owner in (('mamba', 'Mamba3ForwardTape'), ('transformer', 'TransformerForwardTape')):
        shell = (ROOT/f'bindings/_mojolearn_{family}.mojo').read_text()
        assert f'tape.downcast_value_ptr[{owner}]()' in shell
