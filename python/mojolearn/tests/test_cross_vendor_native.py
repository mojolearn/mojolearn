"""Whole-operation audits and independent identity oracles for native helpers."""
import array
import hashlib

from mojolearn import cross_vendor as cv
from mojolearn._verify_native_runtime import audit_native_call


def test_native_hash_matches_independent_sha256_and_field_order():
    state = dict(parameters=array.array('f', [1.0, -0.0]), m=array.array('f', [3.0]),
                 v=array.array('f', [7.0]), flags=array.array('i', [1, 0]))
    expected = hashlib.sha256(b''.join(state[k].tobytes() for k in
                              ('parameters', 'm', 'v', 'flags'))).hexdigest()
    result = audit_native_call(cv.state_hash, state)
    result.assert_native()
    assert result.result == expected
