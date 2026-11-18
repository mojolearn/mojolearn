# SPDX-License-Identifier: Apache-2.0
"""Device-resident fitted state of the decomposition owners (lane
ne1-a-decomp-resident, workstream A of
docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md; rows N310, N325, N350, N351).

Per family: fit, query twice (one upload: the second query holds no new
device buffer), write a fitted array in place (NumPy's read-only error),
replace it (the record is dropped, the query still returns the same bits),
and delete the model (the resident buffers are returned). IncrementalPCA also
checks that a partial_fit chain gives the same bits as before the resident
state (a fresh model fitted on the same batches without queries in between).
These are the source checks; the runtime lifecycle checks run against the
installed wheel in python/mojolearn/tests/test_decomp_resident_lifecycle.py
(the verify smoke's tests step), skipping where the device bindings are not
built.
"""
import gc
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text()


class ResidentSourceContract(unittest.TestCase):
    def test_shared_record_helpers(self):
        kit = _read("bindings/decomp_kit_native_door.mojo")
        for token in ("def decomp_resident_find(", "def decomp_resident_hold(", "def decomp_resident_operand(",
                      "def decomp_resident_adopt(", "def decomp_resident_drop(", "model_generation",
                      "fitted_freeze_any", "resident_capsule(free, handle)", "def decomp_door_resident_keep(",
                      "def decomp_door_resident_derived("):
            self.assertIn(token, kit)
        tail = kit.split("def decomp_resident_generation(", 1)[1]
        self.assertNotIn("py_setitem(fields", tail)  # never in the model's __dict__
        self.assertNotIn("hash(", tail)

    def test_doors_use_the_shared_helpers(self):
        als = _read("bindings/als_native_door.mojo")
        self.assertIn("door.owner = model", als)
        self.assertIn("decomp_door_resident_derived(", als)
        self.assertIn("decomp_door_resident_keep(", als)
        rp = _read("bindings/rp_native_door.mojo")
        self.assertIn("decomp_resident_operand(", rp)
        self.assertIn("decomp_resident_drop(model)", rp)
        # IncrementalPCA keeps its resident state through its typed owner (79f7ee66f4): the door adopts the
        # fit's buffers, reuses them in partial_fit and transform, and drops them on refit; the owner's records
        # are keyed by the model generation, so a replaced array is never served stale.
        ipca = _read("bindings/ipca_native_door.mojo")
        for token in ("ipca_owner_adopt(", "ipca_owner_cached(", "ipca_owner_operand(", "ipca_owner_drop(model)"):
            self.assertIn(token, ipca)
        owner = _read("bindings/ipca_owner_boundary_door.mojo")
        self.assertIn("model_generation(model)", owner)
        self.assertNotIn("py_setitem(fields", owner)  # never in the model's __dict__
        # TruncatedSVD: one native lease owner per fitted state, dropped on refit; the query carries the lease.
        tsvd = _read("bindings/tsvd_native_door.mojo")
        for token in ("tsvd_owner_lease(", "tsvd_owner_drop(model)", "lease.lease > 0, lease.lease"):
            self.assertIn(token, tsvd)

    def test_bindings_register_the_entries(self):
        est = _read("bindings/_mojolearn_estimators.mojo")
        for name in ("tsvd_resident_lease", "tsvd_resident_stats", "tsvd_transform_resident",
                     "tsvd_inverse_transform_resident"):
            self.assertIn(f'("{name}")', est)
        self.assertIn("add_type[TsvdResidentLease]", est)
        self.assertIn('("x_decomp_dev_live")', _read("bindings/_mojolearn_x_decomp.mojo"))
        host = _read("bindings/_mojolearn_estimators_host.mojo")
        self.assertNotIn("tsvd_transform_resident", host)

    def test_same_bodies_as_the_per_call_route(self):
        est = _read("decomposition/estimator.mojo")
        body = est.split("def tsvd_transform_host(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("tsvd_transform_dev(", body)
        body = est.split("def inverse_transform_host(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("inverse_transform_dev(", body)
        state = _read("decomposition/tsvd_resident_state.mojo")
        self.assertIn("tsvd_transform_dev(", state)
        self.assertIn("inverse_transform_dev(", state)
        owner = _read("x_decomp/ipca_owner.mojo")
        self.assertIn("kit.keep(", owner)
        self.assertIn("_state_id[RES](prev_dev", owner)
        self.assertIn("matrix_dev", _read("x_decomp/rp_owner.mojo"))


if __name__ == "__main__":
    unittest.main()
