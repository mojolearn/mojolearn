"""Source admission guards; runtime dtype behavior is covered after integration."""
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / 'providers/python/array_metadata_pyapi.mojo'


def test_dtype_overrides_are_optional_and_reject_before_cast_descriptor():
    source = SOURCE.read_text().split('def array_metadata_once(', 1)[1]
    signature, body = source.split(' raises -> ArrayMetadata:', 1)
    assert 'allow_non_native: Bool = False' in signature
    assert 'complex_error: String = ""' in signature
    assert 'f32_error: String = ""' in signature
    assert signature.index('allow_non_native:') < signature.index('complex_error:') < signature.index('f32_error:')
    assert body.index('raise Error(complex_error)') < body.index('var kind = cast_in_kind(')
    assert body.index('raise Error(f32_error)') < body.index('var kind = cast_in_kind(')
    for code in (13, 14, 15):
        assert f'code == {code}' in body
    # A caller opting into the established complex error gets it before
    # rank refusal, while default callers retain the early rank check.
    assert 'if nd > 2 and complex_error.byte_length() == 0:' in body


def test_readonly_is_captured_independently_of_exclusive_ownership():
    source = SOURCE.read_text()
    assert source.count('array_slot_once(value, "_readonly", context)') == 1
    assert 'owned_writable = owner[1] is None and not readonly' in source
    assert 'self.readonly = readonly' in source
    for name in ('replace_f32', 'replace_f64'):
        method = source.split('def ' + name + '(', 1)[1].split('\n    def ', 1)[0].split('\ndef ', 1)[0]
        assert 'self.readonly = False' in method
        assert 'self.owner =' not in method
