"""GradientBoosting's SymmetricTree defaults are CatBoost's GPU learner's
(lane/catboost-parity, 2026-09-19). Host-only: every check reads the
resolved constructor state or the packed parameter list; no native fit runs.

Reference: catboost 1.2.10 at the pinned source 54a8143a. The learning-rate
checks hold the formula to the values a CatBoost 1.2.10 CPU install reported
from `get_all_params()` on this Mac (the one arm of their auto-selection
that runs here; the GPU arm uses the same formula with the GPU rows).
"""
import struct

import pytest

from mojolearn import GradientBoosting, GradientBoostingClassifier, GradientBoostingRegressor
from mojolearn import ensemble


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


# (loss, rows, iterations, use_best_model, boost_from_average) -> the
# learning_rate CatBoost 1.2.10 CPU reported from get_all_params(), task_type
# CPU, 5 features, no eval set (catboost CPU resolves boost_from_average True
# for RMSE and False for Logloss, options_helper.cpp:353-374).
CATBOOST_CPU_REPORTED = [
    (('RMSE', 1000, 1000, False, True), 0.040943000465631485),
    (('RMSE', 1000, 100, False, True), 0.2661829888820648),
    (('RMSE', 60000, 20, False, True), 0.5),
    (('RMSE', 60000, 300, False, True), 0.208078995347023),
    (('Logloss', 1000, 1000, False, False), 0.010301999747753143),
    (('Logloss', 1000, 100, False, False), 0.08510100096464157),
    (('Logloss', 60000, 20, False, False), 0.5),
    (('Logloss', 60000, 300, False, False), 0.17851899564266205),
]


@pytest.mark.parametrize('key,reported', CATBOOST_CPU_REPORTED)
def test_auto_learning_rate_formula_matches_catboost_cpu(key, reported):
    rate = ensemble.catboost_auto_learning_rate(*key, table=ensemble._CPU_AUTO_LEARNING_RATE)
    assert f32(rate) == reported


def test_gpu_auto_learning_rate_rows():
    # hand-evaluated from options_helper.cpp:221-262 (GPU rows)
    assert ensemble.catboost_auto_learning_rate('Logloss', 10000, 1000, False, False) == 0.029701
    assert ensemble.catboost_auto_learning_rate('RMSE', 200000, 1000, False, True) == 0.080863
    # Unknown targets have no row (`GetTargetType`, :179-192)
    assert ensemble.catboost_auto_learning_rate('CrossEntropy', 1000, 1000, False, False) is None
    assert ensemble.catboost_auto_learning_rate('MAE', 1000, 1000, False, True) is None
    # MultiClass has no boost_from_average=True row
    assert ensemble.catboost_auto_learning_rate('MultiClass', 1000, 1000, False, True) is None


def test_c_round_is_half_away_from_zero():
    assert ensemble._c_round(0.0000025, 6) == 0.000003
    assert ensemble._c_round(0.1234565, 6) in (0.123456, 0.123457)  # binary64 input
    assert ensemble._c_round(0.5, 6) == 0.5


def test_symmetric_defaults_are_catboost_gpu():
    m = GradientBoosting()
    assert m.n_estimators == 1000                  # boosting_options.cpp:13
    assert m.bootstrap_type == 'Bayesian'          # bootstrap_options.h:18
    assert m.bagging_temperature == 1.0            # bootstrap_options.h:16
    assert m.max_depth == 6 and m.border_count == 128 and m.nan_mode == 'Min'
    assert m.score_function == 'Cosine'
    assert m.learning_rate is None and m.random_strength is None
    params = m._params(1000, 5, 0)
    assert params[5] == 1000
    assert params[8] == 3.0                        # l2 unset -> 3.0
    assert params[25] == 1.0                       # oblivious_tree_options.cpp:17
    # RMSE, no eval set: use_best_model False, boost_from_average True
    assert params[7] == ensemble.catboost_auto_learning_rate('RMSE', 1000, 1000, False, True)
    assert params[16] == -1                        # 1000 iterations: the loss decides


@pytest.mark.parametrize('policy', ['Depthwise', 'Lossguide'])
def test_non_symmetric_policies_keep_their_defaults(policy):
    m = GradientBoosting(grow_policy=policy, loss='Logloss')
    assert m.n_estimators == 100
    assert m.bootstrap_type is None
    params = m._params(1000, 5, 0)
    assert params[7] == 0.03 and params[25] == 0.0
    assert params[16] == -1                        # no small-iteration rule


def test_learning_rate_auto_is_switched_off_as_theirs():
    base = dict(loss='Logloss', n_estimators=500)
    auto = ensemble.catboost_auto_learning_rate('Logloss', 4000, 500, False, False)
    assert GradientBoosting(**base)._params(4000, 5, 0)[7] == auto
    # options_helper.cpp:273-278: any of these set keeps 0.03
    for extra in (dict(l2_leaf_reg=3.0), dict(leaf_estimation_method='Newton'),
                  dict(leaf_estimation_iterations=10)):
        assert GradientBoosting(**base, **extra)._params(4000, 5, 0)[7] == 0.03
    assert GradientBoosting(**base, learning_rate=0.2)._params(4000, 5, 0)[7] == 0.2
    assert GradientBoosting(loss='CrossEntropy', n_estimators=500)._params(4000, 5, 0)[7] == 0.03


def test_learning_rate_key_reads_use_best_model_and_boost_from_average():
    m = GradientBoosting(loss='RMSE', n_estimators=1000)
    table = ensemble.catboost_auto_learning_rate
    assert m._resolved_learning_rate(5000) == table('RMSE', 5000, 1000, False, True)
    # an eval set with a non-constant target turns use_best_model on
    assert m._resolved_learning_rate(5000, 100, False) == table('RMSE', 5000, 1000, True, True)
    assert m._resolved_learning_rate(5000, 100, True) == table('RMSE', 5000, 1000, False, True)
    off = GradientBoosting(loss='RMSE', n_estimators=1000, boost_from_average=False)
    assert off._resolved_learning_rate(5000) == table('RMSE', 5000, 1000, False, False)


def test_small_iteration_leaf_count():
    # options_helper.cpp:290-307: < 200 iterations and < 20 features -> 1
    assert GradientBoosting(loss='Logloss', n_estimators=100)._params(1000, 19, 0)[16] == 1
    assert GradientBoosting(loss='Logloss', n_estimators=100)._params(1000, 20, 0)[16] == -1
    assert GradientBoosting(loss='Logloss', n_estimators=200)._params(1000, 5, 0)[16] == -1
    assert GradientBoosting(loss='Logloss', n_estimators=100,
                            leaf_estimation_iterations=10)._params(1000, 5, 0)[16] == 10
    # the 1 is resolved after the learning rate, so it does not switch it off
    m = GradientBoosting(loss='Logloss', n_estimators=100)
    assert m._params(1000, 5, 0)[7] == ensemble.catboost_auto_learning_rate(
        'Logloss', 1000, 100, False, False)


@pytest.mark.parametrize('loss', ['QueryRMSE', 'PairLogit', 'YetiRank'])
def test_querywise_default_bootstrap_is_refused_by_name(loss):
    with pytest.raises(NotImplementedError, match="bootstrap_type='No'"):
        GradientBoosting(loss=loss)
    assert GradientBoosting(loss=loss, bootstrap_type='No').bootstrap_type == 'No'


def test_default_bayesian_refuses_subsample_in_catboost_words():
    with pytest.raises(ValueError, match="default bootstrap_type='Bayesian'"):
        GradientBoosting(subsample=0.5)


def test_adapters_inherit_the_learner_defaults():
    for cls, loss in ((GradientBoostingClassifier, 'Logloss'), (GradientBoostingRegressor, 'RMSE')):
        learner = cls()._new_learner(None)
        assert learner.loss == loss
        assert learner.n_estimators == 1000
        assert learner.bootstrap_type == 'Bayesian'
        assert learner.learning_rate is None and learner.l2_leaf_reg is None
        assert learner._params(1000, 5, 0)[25] == 1.0
