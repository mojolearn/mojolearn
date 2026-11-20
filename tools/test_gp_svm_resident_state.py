# SPDX-License-Identifier: Apache-2.0
"""Device-resident fitted state of the Gaussian processes and kernel SVMs
(lane ne1-a-gp-svm, workstream A of docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md).

Per family: fit, predict twice (one upload), write a fitted array in place
(NumPy's read-only error), replace it (the record is dropped; the predict
still returns the same bits), refit (a new lease, one more upload) and delete
the model (the live lease count falls). These are the source checks;
the runtime lifecycle checks run against the installed wheel in
python/mojolearn/tests/test_gp_svm_resident_lifecycle.py (the verify smoke's
tests step), skipping where the device bindings are not built.
"""
import gc
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text()


class ResidentSourceContract(unittest.TestCase):
    def test_store_has_cap_lru_and_lock(self):
        src = _read("core/fitted_resident_store.mojo")
        for token in ("FITTED_CAP_DIVISOR", "get_memory_info", "least recently used eviction",
                      "_store_lock(store)", "fitted_lease_close", "ctx_cache_key"):
            self.assertIn(token, src)
        self.assertNotIn("hash", src.split('"""', 2)[2].lower().replace("no content hash", ""))

    def test_doors_take_a_lease_and_freeze(self):
        door = _read("bindings/fitted_resident_door.mojo")
        for token in ("prepared_get", "prepared_put", "model_generation", "fitted_freeze_any", "_base is None"):
            self.assertIn(token, door)
        self.assertNotIn("py_setitem(fields", door)  # never in the model's __dict__
        # The GP doors take their lease through the shared GP lease unpack, which keys the record by the model
        # generation and freezes the mirrored arrays; the lease word rides in the native predict request.
        lease = _read("bindings/gp_resident_boundary_unpack.mojo")
        for token in ("prepared_get(", "prepared_put(", "model_generation(model)", "fitted_freeze_any("):
            self.assertIn(token, lease)
        self.assertNotIn("py_setitem(fields", lease)
        self.assertIn("gp_resident_lease(model, _GPR_RESIDENT_SLOT,", _read("bindings/gp_regressor_native_door.mojo"))
        self.assertIn("gp_resident_lease(model, _GPC_RESIDENT_SLOT,", _read("bindings/gp_classifier_native_door.mojo"))
        svc = _read("bindings/svc_kernel_native_door.mojo")
        self.assertIn('"svc_predict_resident"', svc)
        self.assertIn('"svr_predict_resident"', svc)

    def test_bindings_register_the_entries(self):
        gp = _read("bindings/_mojolearn_gp.mojo")
        for name in ("gp_resident_lease", "gp_resident_stats", "gpr_predict_resident", "gpc_predict_all_resident"):
            self.assertIn(f'("{name}")', gp)
        self.assertIn('add_type[GpResidentLease]', gp)
        svm = _read("bindings/_mojolearn_svm.mojo")
        for name in ("svm_resident_lease", "svm_resident_stats", "svc_predict_resident", "svr_predict_resident"):
            self.assertIn(f'("{name}")', svm)
        self.assertIn('add_type[SvmResidentLease]', svm)

    def test_same_launches_as_the_per_call_route(self):
        gpr = _read("gaussian_process/gpr_resident.mojo")
        body = gpr.split("def gpr_predict_ptr_device(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("gpr_predict_dev(", body)
        self.assertIn("gpr_predict_dev(", _read("gaussian_process/gp_resident_state.mojo"))
        svc = _read("svm/impl/svc_impl.mojo")
        body = svc.split("def svc_predict(\n", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("svc_predict_dev(", body)
        # svc_predict_dev is svc_predict_dev_into returned as a host list; the resident route calls the same body.
        dev = svc.split("def svc_predict_dev(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("svc_predict_dev_into(", dev)
        self.assertIn("svc_predict_dev_into(", _read("svm/svm_resident_state.mojo"))


if __name__ == "__main__":
    unittest.main()
