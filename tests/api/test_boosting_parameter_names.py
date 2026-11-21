# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests: CatBoost / XGBoost / LightGBM parameter names on the
gradient-boosting classes (python/mojolearn/_boosting_param_names.py).

Construction and set_params only; nothing here fits a model.
"""
import importlib.util
import inspect
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _table_module():
    spec = importlib.util.spec_from_file_location(
        "_boosting_param_names", ROOT / "python" / "mojolearn" / "_boosting_param_names.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ the classes

mojolearn = pytest.importorskip("mojolearn")
ADAPTERS = (mojolearn.GradientBoostingClassifier, mojolearn.GradientBoostingRegressor)
ALL = ADAPTERS + (mojolearn.GradientBoosting,)


def test_every_canonical_name_is_a_parameter():
    params = set(inspect.signature(mojolearn.GradientBoosting.__init__).parameters)
    for name, canonical, *_ in _table_module().BOOSTING_PARAMETER_NAMES:
        assert canonical in params, (name, canonical)
        assert name not in params, name


@pytest.mark.parametrize("cls", ALL)
def test_signature_and_get_params_report_canonical_names(cls):
    params = cls().get_params()
    assert "iterations" not in params and "aliases" not in params
    assert "n_estimators" in params


@pytest.mark.parametrize("cls", ALL)
def test_constructor_stores_under_the_canonical_name(cls):
    model = cls(iterations=50, eta=0.1, depth=4, reg_lambda=2.0, random_seed=7)
    params = model.get_params()
    assert params["n_estimators"] == 50
    assert params["learning_rate"] == 0.1
    assert params["max_depth"] == 4
    assert params["l2_leaf_reg"] == 2.0
    assert params["random_state"] == 7


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("name", ["num_boost_round", "num_trees", "n_iter", "num_iterations"])
def test_other_tree_count_names(cls, name):
    assert cls(**{name: 30}).get_params()["n_estimators"] == 30


@pytest.mark.parametrize("cls", ALL)
def test_python_keyword_name_is_accepted_through_a_dict(cls):
    assert cls(**{"lambda": 4.0}).get_params()["l2_leaf_reg"] == 4.0


@pytest.mark.parametrize("cls", ALL)
def test_two_names_same_value_is_accepted(cls):
    assert cls(n_estimators=40, iterations=40).get_params()["n_estimators"] == 40
    assert cls(iterations=40, num_trees=40).get_params()["n_estimators"] == 40


@pytest.mark.parametrize("cls", ALL)
def test_two_names_different_values_is_refused(cls):
    with pytest.raises(ValueError, match="n_estimators was given twice"):
        cls(n_estimators=40, iterations=50)
    with pytest.raises(ValueError, match="given twice"):
        cls(iterations=40, num_trees=50)


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("name,alternative", [
    ("max_bin", "border_count"), ("early_stopping_rounds", "od_wait"), ("gamma", "min_split_gain"),
    ("colsample_bylevel", "feature_fraction"), ("min_child_samples", "min_data_in_leaf"),
    ("task_type", "device"), ("boosting", "boosting_type"),
])
def test_different_meaning_is_refused_with_its_alternative(cls, name, alternative):
    with pytest.raises(ValueError, match=alternative):
        cls(**{name: 1})


@pytest.mark.parametrize("cls", ALL)
def test_unknown_name_is_still_a_type_error(cls):
    with pytest.raises(TypeError, match="not_a_parameter"):
        cls(not_a_parameter=1)


@pytest.mark.parametrize("cls", ALL)
def test_set_params_takes_other_names(cls):
    model = cls().set_params(iterations=60, depth=5)
    params = model.get_params()
    assert params["n_estimators"] == 60 and params["max_depth"] == 5


@pytest.mark.parametrize("cls", ALL)
def test_set_params_refuses_different_meaning_and_conflict(cls):
    with pytest.raises(ValueError, match="border_count"):
        cls().set_params(max_bin=32)
    with pytest.raises(ValueError, match="given twice"):
        cls().set_params(n_estimators=10, iterations=20)


@pytest.mark.parametrize("cls", ALL)
def test_set_params_unknown_name_keeps_its_error(cls):
    with pytest.raises(ValueError, match="not_a_parameter"):
        cls().set_params(not_a_parameter=1)


def test_gradient_boosting_only_names():
    model = mojolearn.GradientBoosting(loss_function="Logloss", categorical_feature=[0])
    params = model.get_params()
    assert params["loss"] == "Logloss" and list(params["cat_features"]) == [0]
    for cls in ADAPTERS:
        with pytest.raises(TypeError, match="loss_function"):
            cls(loss_function="Logloss")


def test_gradient_boosting_positional_conflict_rule():
    # A canonical value equal to its default yields to the other name; any
    # other canonical value with a different other-name value is refused.
    assert mojolearn.GradientBoosting(max_depth=6, depth=8).get_params()["max_depth"] == 8
    with pytest.raises(ValueError, match="max_depth was given twice"):
        mojolearn.GradientBoosting(max_depth=4, depth=8)


@pytest.mark.parametrize("cls", ALL)
def test_clone_round_trips_canonical_names(cls):
    base = pytest.importorskip("sklearn.base")
    model = cls(iterations=25, eta=0.2)
    copy = base.clone(model)
    assert copy.get_params()["n_estimators"] == 25
    assert copy.get_params()["learning_rate"] == 0.2
