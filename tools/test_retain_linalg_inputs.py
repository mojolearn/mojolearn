import copy
import hashlib

import numpy as np
import pytest

import bench_board_algos as A
import retain_linalg_inputs as R


def record(lane):
    arrays = A.lu_system(8) if lane in ("lu-solve", "lu-factor") else (A.sym_system(8),)
    metadata = [dict(shape=list(a.shape), dtype=str(a.dtype),
                     sha256=hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()) for a in arrays]
    cell = dict(status="ok", params=dict(raw_params=dict(seed=A.SEED, input_arrays=metadata)))
    return dict(family="algos", lane=lane, cells=[cell, copy.deepcopy(cell)]), arrays


@pytest.mark.parametrize("lane", ["cholesky", "eigh", "lu-solve", "lu-factor"])
def test_reconstruction_matches_every_original_input_byte(lane):
    rec, original = record(lane)
    restored = R.generated_inputs(rec, A)
    assert len(restored) == len(original)
    for (_, actual), expected in zip(restored, original):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


def test_changed_hash_or_disagreeing_workers_are_refused():
    rec, _ = record("cholesky")
    rec["cells"][1]["params"]["raw_params"]["input_arrays"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different input arrays"):
        R.generated_inputs(rec, A)
    rec["cells"] = [rec["cells"][1]]
    with pytest.raises(ValueError, match="differ from original"):
        R.generated_inputs(rec, A)
    assert R.generated_inputs(dict(family="algos", lane="qr"), A) == []
