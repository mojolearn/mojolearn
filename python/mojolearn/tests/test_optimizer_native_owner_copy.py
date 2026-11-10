"""Copy/checkpoint boundaries preserve public buffers and independent native owners."""
import copy
import pickle

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
@pytest.mark.parametrize("resident", [False, True])
def test_optimizer_copy_reconstructs_native_owner_without_process_handles(cls, method, resident):
    owner = cls(np.array([1., -.5, .25], dtype="f4"), lr=.01,
                numeric_mode="identical", resident=resident)
    owner.step(np.array([.1, -.2, .3], dtype="f4"))
    state = audit_native_call(owner.__getstate__).assert_native()
    assert "_native_optimizer_owner" not in state
    assert not {"_res", "_res_binding", "_res_host", "_host_fresh", "_pristine", "_p_dev", "_g_dev"} & vars(owner).keys()
    clone = copy.copy(owner) if method == "copy" else (
        copy.deepcopy(owner) if method == "deepcopy" else pickle.loads(pickle.dumps(owner)))
    assert vars(clone)["_native_optimizer_owner"] is not vars(owner)["_native_optimizer_owner"]
    assert clone.t == owner.t
    assert np.asarray(clone.params[0]).tobytes() == np.asarray(owner.params[0]).tobytes()
    assert np.asarray(clone.exp_avg).tobytes() == np.asarray(owner.exp_avg).tobytes()
    if method == "copy":
        assert clone.params is owner.params
    else:
        assert not np.shares_memory(np.asarray(clone.params[0]), np.asarray(owner.params[0]))
    step = owner.t
    audit_native_call(clone.step, np.array([.2, -.1, .4], dtype="f4")).assert_native()
    assert clone.t == step + 1 and owner.t == step
    if method != "copy":
        audit_native_call(owner.step, np.array([.2, -.1, .4], dtype="f4")).assert_native()
        assert np.asarray(clone.params[0]).tobytes() == np.asarray(owner.params[0]).tobytes()
        assert np.asarray(clone.exp_avg).tobytes() == np.asarray(owner.exp_avg).tobytes()


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
def test_optimizer_restores_legacy_host_control_as_native_values(cls):
    owner = cls(np.ones(3, dtype="f4"), numeric_mode="identical", resident=False)
    snapshot = owner.__getstate__()
    snapshot["_pristine"] = snapshot.pop("_native_optimizer_pristine")
    snapshot.update(_res=None, _res_binding=None, _res_host=False, _host_fresh=True,
                    _p_dev=False, _g_dev=False)
    clone = object.__new__(cls)
    audit_native_call(clone.__setstate__, snapshot).assert_native()
    assert not {"_res", "_res_binding", "_pristine", "_p_dev", "_g_dev"} & vars(clone).keys()
    audit_native_call(clone.step, np.array([.1, .2, .3], dtype="f4")).assert_native()
    assert clone.t == 1 and owner.t == 0


@pytest.mark.parametrize("cls", [T.SGD, T.Adam, T.AdamW])
def test_optimizer_parameter_reconstruction_creates_fresh_native_control(cls):
    owner = cls(np.ones(3, dtype="f4"), numeric_mode="identical", resident=False)
    owner.step(np.array([.1, .2, .3], dtype="f4"))
    parameters = audit_native_call(owner.get_params, False).assert_native()
    clone = audit_native_call(cls, **parameters).assert_native()
    assert clone.t == 0 and owner.t == 1
    assert vars(clone)["_native_optimizer_owner"] is not vars(owner)["_native_optimizer_owner"]
