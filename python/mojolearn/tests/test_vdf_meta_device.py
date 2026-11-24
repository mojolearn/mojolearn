# SPDX-License-Identifier: Apache-2.0
"""`device=` on the meta-estimators and OneClassSVM (lane vdf-meta): the boundary only.

Words and exception types, never numbers (the digests are verify's). Per class: the constructor takes
`device`, defaulting to None; a bad word is refused in the shared words; `device="cpu"` is admitted on
every installation (the one table answers True); and for the meta-estimators a member constructed with
another explicit device is refused in `function_device_member`'s words before anything is fitted.
"""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn import _device_table
from mojolearn import (
    CalibratedClassifierCV, DecisionTreeClassifier, DecisionTreeRegressor, MultiOutputClassifier,
    MultiOutputRegressor, OneClassSVM, OneVsRestClassifier, StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
)


def _tree(cls, **params):
    return cls(max_depth=2, **params)


# class -> a constructor taking the member keyword arguments
BUILD = {
    VotingClassifier: lambda **m: VotingClassifier([("a", _tree(DecisionTreeClassifier, **m))]),
    VotingRegressor: lambda **m: VotingRegressor([("a", _tree(DecisionTreeRegressor, **m))]),
    StackingClassifier: lambda **m: StackingClassifier([("a", _tree(DecisionTreeClassifier, **m))], cv=2),
    StackingRegressor: lambda **m: StackingRegressor([("a", _tree(DecisionTreeRegressor, **m))], cv=2),
    MultiOutputClassifier: lambda **m: MultiOutputClassifier(_tree(DecisionTreeClassifier, **m)),
    MultiOutputRegressor: lambda **m: MultiOutputRegressor(_tree(DecisionTreeRegressor, **m)),
    OneVsRestClassifier: lambda **m: OneVsRestClassifier(_tree(DecisionTreeClassifier, **m)),
    CalibratedClassifierCV: lambda **m: CalibratedClassifierCV(_tree(DecisionTreeClassifier, **m), cv=2),
    OneClassSVM: lambda **m: OneClassSVM(),
}
CLASSES = list(BUILD)
META = [cls for cls in CLASSES if cls is not OneClassSVM]
MULTI = (MultiOutputClassifier, MultiOutputRegressor)
REGRESSORS = (VotingRegressor, StackingRegressor, MultiOutputRegressor)


def _data(cls):
    rng = np.random.default_rng(0)
    x = rng.random((24, 3), dtype=np.float32)
    if cls in MULTI:
        y = np.stack([(x[:, 0] > 0.5), (x[:, 1] > 0.5)], axis=1)
        return x, (y.astype(np.float32) if cls in REGRESSORS else y.astype(np.int64))
    if cls in REGRESSORS:
        return x, x[:, 0].astype(np.float32)
    return x, (x[:, 0] > 0.5).astype(np.int64)


@pytest.mark.parametrize("cls", CLASSES)
def test_device_is_a_parameter_defaulting_to_none(cls):
    parameter = inspect.signature(cls.__init__).parameters["device"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert BUILD[cls]().get_params()["device"] is None


@pytest.mark.parametrize("cls", CLASSES)
def test_the_table_connects_the_class(cls):
    assert _device_table.answer(cls) is True


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("word", ["tpu", 0])
def test_a_bad_word_is_refused_in_the_shared_words(cls, word):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        BUILD[cls]().set_params(device=word)


@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_is_admitted_on_every_installation(cls):
    # Was refused beside a GPU set ("its CPU implementation is not connected to device= yet").
    model = BUILD[cls]().set_params(device="cpu")
    assert model.get_params()["device"] == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
def test_gpu_on_a_cpu_only_installation_is_refused_in_the_estimators_words(cls):
    if mojolearn.vendor() != "cpu":
        pytest.skip("holds on a CPU-only installation only")
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn"):
        BUILD[cls]().set_params(device="gpu")


@pytest.mark.parametrize("cls", META)
def test_a_member_with_another_explicit_device_is_refused_before_the_fit(cls):
    # Needs both words admitted: a GPU installation (a CPU-only one refuses device='gpu' at construction).
    if mojolearn.vendor() == "cpu":
        pytest.skip("holds on a GPU installation only")
    x, y = _data(cls)
    model = BUILD[cls](device="gpu").set_params(device="cpu")
    with pytest.raises(ValueError, match=f"{cls.__name__} was given device='cpu' but its estimator DecisionTree"
                                         "(Classifier|Regressor) was constructed with device='gpu'"):
        model.fit(x, y)


def test_a_nested_refusal_names_the_meta_estimator_that_handed_the_device():
    # Needs both words admitted: a GPU installation.
    if mojolearn.vendor() == "cpu":
        pytest.skip("holds on a GPU installation only")
    x, y = _data(VotingClassifier)
    inner = StackingClassifier([("a", _tree(DecisionTreeClassifier, device="gpu"))], cv=2)
    outer = VotingClassifier([("s", inner)], device="cpu")
    with pytest.raises(ValueError, match="VotingClassifier was given device='cpu', handed to StackingClassifier, "
                                         "but its estimator DecisionTreeClassifier was constructed with device='gpu'"):
        outer.fit(x, y)


def test_a_search_over_a_meta_estimator_with_a_device_hands_it_to_members_that_name_none():
    from mojolearn import GridSearchCV
    x, y = _data(VotingClassifier)
    voting = VotingClassifier([("a", _tree(DecisionTreeClassifier))], device="cpu")
    search = GridSearchCV(voting, {"voting": ["hard"]}, cv=2).fit(x, y)
    best = search.best_estimator_
    assert best.get_params()["device"] == "cpu"
    assert all(member.get_params()["device"] == "cpu" for member in best.estimators_)
