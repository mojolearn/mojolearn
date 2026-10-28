"""Source assertions for the reproduced pooled-output callback repair."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_logits_captures_output_native_address_once():
    source = (ROOT / 'bindings/byte_lm_trainer_native.mojo').read_text()
    body = source.split('def byte_lm_logits(')[1].split('def _registry(')[0]
    assert body.count('_logits_output_address(out, context)') == 1
    assert 'classical_address(out)' not in body
    assert 'unsafe_from_address=output_address' in body
    assert 'neural_copy_row_prefixes(Int(py=classical_address(result)), output_address,' in body
    helper = source.split('def _logits_output_address(')[1].split('def _pooled_logits(')[0]
    assert 'py_call(context["byte_lm_addr_getter"], output, cls)' in helper
    assert 'cls is not context["array_class"] and cls is not context["byte_lm_pooled_class"]' in helper
    assert 'sequential_address' not in helper


def test_pooled_hooks_are_pinned_after_native_finalizer_and_checked_before_writes():
    source = (ROOT / 'python/mojolearn/_byte_lm_impl.py').read_text()
    assert 'address_slot = vars(_Array)["_addr"]' in source
    assert 'type(address_slot) is not MemberDescriptorType' in source
    assert source.index('_PooledLogits.__del__ = module.classical_attribute_descriptor') < source.index('layout = tuple((base, tuple((name, vars(base).get(name))')
    native = (ROOT / 'bindings/byte_lm_trainer_native.mojo').read_text()
    pooled = native.split('def _pooled_logits(')[1].split('def byte_lm_pooled_finalize(')[0]
    assert pooled.index('_admit_pooled_logits_layout(context)') < pooled.index('classical_wrap_array(')
