# SPDX-License-Identifier: Apache-2.0
"""Boundary of `device=` on model selection's public functions and splitters
(lane vdf-msel): the keyword is keyword-only with default None, a bad word is
refused in the shared words before any work, a splitter stores its device as
the estimators do (get_params returns it, None included; clone and pickle
carry it), and a cv tool refuses a member constructed with a different device.
No numbers: the digests are verify's."""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn import model_selection as ms

BAD = "device must be 'gpu', 'cpu' or None"
GPU = mojolearn.vendor() != "cpu"

X = np.arange(40, dtype=np.float32).reshape(20, 2)
Y = np.arange(20, dtype=np.float32)
LABELS = np.array([0, 1] * 10, dtype=np.int64)

FUNCTIONS = {
    "cross_val_score": lambda d: ms.cross_val_score(mojolearn.Ridge(), X, Y, cv=2, device=d),
    "cross_validate": lambda d: ms.cross_validate(mojolearn.Ridge(), X, Y, cv=2, device=d),
    "cross_val_predict": lambda d: ms.cross_val_predict(mojolearn.Ridge(), X, Y, cv=2, device=d),
    "learning_curve": lambda d: ms.learning_curve(mojolearn.Ridge(), X, Y, cv=2, train_sizes=[1.0], device=d),
    "validation_curve": lambda d: ms.validation_curve(mojolearn.Ridge(), X, Y, param_name="alpha",
                                                      param_range=[1.0], cv=2, device=d),
    "permutation_test_score": lambda d: ms.permutation_test_score(mojolearn.Ridge(), X, Y, cv=2,
                                                                  n_permutations=2, device=d),
    "train_test_split": lambda d: ms.train_test_split(X, Y, random_state=0, device=d),
    "check_cv": lambda d: ms.check_cv(2, LABELS, classifier=True, device=d),
    "split_descriptor": lambda d: ms.split_descriptor(X, Y, estimator=mojolearn.Ridge(), cv=2, device=d),
    "default_folds": lambda d: ms.default_folds(LABELS, 2, classifier=True, device=d),
}

SPLITTERS = {
    "KFold": lambda d: ms.KFold(2, device=d),
    "StratifiedKFold": lambda d: ms.StratifiedKFold(2, device=d),
    "GroupKFold": lambda d: ms.GroupKFold(2, device=d),
    "StratifiedGroupKFold": lambda d: ms.StratifiedGroupKFold(2, device=d),
    "TimeSeriesSplit": lambda d: ms.TimeSeriesSplit(2, device=d),
    "LeaveOneOut": lambda d: ms.LeaveOneOut(device=d),
    "LeavePOut": lambda d: ms.LeavePOut(2, device=d),
    "LeaveOneGroupOut": lambda d: ms.LeaveOneGroupOut(device=d),
    "LeavePGroupsOut": lambda d: ms.LeavePGroupsOut(2, device=d),
    "RepeatedKFold": lambda d: ms.RepeatedKFold(n_splits=2, n_repeats=2, device=d),
    "RepeatedStratifiedKFold": lambda d: ms.RepeatedStratifiedKFold(n_splits=2, n_repeats=2, device=d),
    "ShuffleSplit": lambda d: ms.ShuffleSplit(2, device=d),
    "GroupShuffleSplit": lambda d: ms.GroupShuffleSplit(2, device=d),
    "StratifiedShuffleSplit": lambda d: ms.StratifiedShuffleSplit(2, device=d),
    "PredefinedSplit": lambda d: ms.PredefinedSplit([0, 1, 0, 1], device=d),
}


def _keyword_only_none(callable_):
    parameter = inspect.signature(callable_).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_function_signature(name):
    _keyword_only_none(getattr(ms, name))


@pytest.mark.parametrize("name", sorted(SPLITTERS))
def test_splitter_signature(name):
    _keyword_only_none(getattr(ms, name).__init__)


@pytest.mark.parametrize("bad", ["tpu", 0])
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_function_bad_word(name, bad):
    with pytest.raises(ValueError, match=BAD):
        FUNCTIONS[name](bad)


@pytest.mark.parametrize("bad", ["tpu", 0])
@pytest.mark.parametrize("name", sorted(SPLITTERS))
def test_splitter_bad_word_at_construction(name, bad):
    with pytest.raises(ValueError, match=BAD):
        SPLITTERS[name](bad)


# vdf2-public: the splitters keep the estimators' rule of
# test_device_parameter.py (b9795dbe2d: "the default is None and get_params
# returns what was passed"; clone and pickle carry it). This test asserted
# that device=None left no field (44b842efd3), which contradicted that rule.
@pytest.mark.parametrize("name", sorted(SPLITTERS))
def test_splitter_stores_its_device(name):
    import pickle
    for device in (None, "cpu"):
        splitter = SPLITTERS[name](device)
        assert splitter.get_params()["device"] == device
        assert splitter.device == device
        assert type(splitter)(**splitter.get_params()).get_params() == splitter.get_params()
        assert pickle.loads(pickle.dumps(splitter)).get_params() == splitter.get_params()


@pytest.mark.parametrize("name", sorted(SPLITTERS))
def test_splitter_refuses_a_bad_word_before_its_other_arguments(name):
    # the device is admitted before any other argument is read
    build = {"PredefinedSplit": lambda: ms.PredefinedSplit(None, device="tpu"),
             "LeavePOut": lambda: ms.LeavePOut(None, device="tpu"),
             "LeavePGroupsOut": lambda: ms.LeavePGroupsOut(None, device="tpu")}.get(
        name, lambda: SPLITTERS[name]("tpu"))
    with pytest.raises(ValueError, match=BAD):
        build()


@pytest.mark.skipif(GPU, reason="a CPU-only installation refuses device='gpu'")
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_gpu_refused_on_cpu_only(name):
    with pytest.raises(ValueError, match="needs a GPU build of mojolearn"):
        FUNCTIONS[name]("gpu")


@pytest.mark.skipif(not GPU, reason="a member device='gpu' is refused at construction on a CPU-only installation")
def test_member_with_another_device_is_refused():
    with pytest.raises(ValueError, match="cross_val_score was given device='cpu' but its estimator Ridge was "
                                         "constructed with device='gpu'"):
        ms.cross_val_score(mojolearn.Ridge(device="gpu"), X, Y, cv=2, device="cpu")
