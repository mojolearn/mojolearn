"""Whole-operation audits and independent identity oracles for native helpers."""
import array
import hashlib
import struct

import pytest

from mojolearn import cross_vendor as cv
from mojolearn._verify_native_runtime import audit_native_call


def test_native_fold_preserves_logical_order_signed_zero_and_nan_contract():
    a = array.array('f', [1.0, -1e-39, 1e-39, float('nan')])
    b = array.array('f', [2 ** -24, -0.0, 0.0, 2.0])
    c = array.array('f', [2 ** -24, -0.0, 0.0, -2.0])
    inputs = [a.tobytes(), b.tobytes(), c.tobytes()]
    before = tuple(inputs)
    result = audit_native_call(cv.ordered_fold, inputs)
    result.assert_native()
    # Explicit integer words independently establish the non-associative
    # left fold, signed-zero FTZ, and the native canonical-NaN contract.
    assert struct.unpack('<4I', result.result) == (0x3f800000, 0x80000000, 0, 0x7fc00000)
    assert tuple(inputs) == before
    pair = audit_native_call(cv.fold_pair, inputs[1], inputs[2])
    pair.assert_native()
    prefixed = audit_native_call(cv.ordered_fold, [inputs[0]], prefix=pair.result)
    prefixed.assert_native()
    assert struct.unpack('<I', prefixed.result[:4])[0] == 0x3f800001


def test_native_hash_matches_independent_sha256_and_field_order():
    state = dict(parameters=array.array('f', [1.0, -0.0]), m=array.array('f', [3.0]),
                 v=array.array('f', [7.0]), flags=array.array('i', [1, 0]))
    expected = hashlib.sha256(b''.join(state[k].tobytes() for k in
                              ('parameters', 'm', 'v', 'flags'))).hexdigest()
    result = audit_native_call(cv.state_hash, state)
    result.assert_native()
    assert result.result == expected


@pytest.mark.parametrize('parts,prefix', [([], None), ([b'abc'], None),
                                        ([b'\x00' * 4, b'\x00' * 8], None)])
def test_native_fold_admission_refusals_have_no_python_work(parts, prefix):
    result = audit_native_call(cv.ordered_fold, parts, prefix=prefix)
    assert isinstance(result.exception, ValueError)
    report = result.report()
    assert not report['python_frames'], report
    assert not report['forbidden_native_calls'], report
    assert not report['wrapper_work'], report
    assert not report['unobserved_existing_threads'], report
