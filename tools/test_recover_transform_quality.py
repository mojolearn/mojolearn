import copy
import hashlib
import numpy as np
import pytest

import recover_transform_quality as R


def test_input_verification_uses_original_shape_dtype_and_canonical_hash():
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    metadata = {"arrays": {"X": dict(shape=[3, 2], dtype="float32",
        sha256=R.A._tool("classical_two_datasets").sha256_array(a))}}
    R.verify_inputs({"X": a}, metadata)
    for bad in (a + 1, a.astype(np.float64), a.reshape(2, 3)):
        with pytest.raises(ValueError):
            R.verify_inputs({"X": bad}, metadata)


def test_scored_output_digest_rejects_modified_or_differently_named_arrays():
    a = np.arange(6, dtype=np.float32)
    h = hashlib.sha256(b"pred" + a.tobytes()).hexdigest()[:16]
    R.verify_outputs({"pred": a}, h)
    for bad in ({"pred": a + 1}, {"different_output": a}):
        with pytest.raises(ValueError):
            R.verify_outputs(bad, h)


def test_recovery_cannot_change_timing_quality_or_worker_parameters():
    race = dict(cells=[dict(arm="ours", measurement={"id": "original"},
                           times_ms=[2.], median_ms=2., quality={"output_shape": "3x2"},
                           params={"scale": True})])
    R.verify_observations(race, copy.deepcopy(race))
    for key, value in (("times_ms", [1.]), ("quality", {"r2": 1.}), ("params", {"scale": False})):
        altered = copy.deepcopy(race)
        altered["cells"][0][key] = value
        with pytest.raises(ValueError):
            R.verify_observations(altered, race)
