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


# ---------------------------------------------------------------------------
# boosting_type (Ordered boosting) -- host-only: the resolution rule, the
# refusals and the parameter tail, through a fake binding
# ---------------------------------------------------------------------------

import struct as _struct
import numpy as np


class _FakeGbdt:
    """Captures what `fit` hands `gbdt_fit`; answers a one-tree text."""

    def __init__(self):
        self.calls = []

    def gbdt_fit(self, *args):
        self.calls.append(args)
        return ["mojolearn-model 2\n", 0, False, [0.5], []]

    def gbdt_model_dim(self, text):
        return 1


def _fit_fake(model, n_rows=64, **fit_kw):
    fake = _FakeGbdt()
    model._bind = lambda name: fake
    X = np.arange(n_rows * 3, dtype=np.float32).reshape(n_rows, 3) / 7
    y = (np.arange(n_rows) % 2).astype(np.float32)
    model.fit(X, y, **fit_kw)
    return fake.calls[-1]


@pytest.mark.parametrize('n_estimators,rows,want', [
    (1000, 49999, 'Ordered'), (1000, 50000, 'Plain'), (500, 100, 'Ordered'), (499, 100, 'Plain'),
])
def test_boosting_type_default_is_catboost_gpu(n_estimators, rows, want):
    # catboost_options.cpp:802-807 then defaults_helper.h:33-42
    assert GradientBoosting(loss='Logloss', n_estimators=n_estimators)._resolved_boosting_type(rows) == want


@pytest.mark.parametrize('kw', [
    dict(loss='MultiClass'), dict(loss='MultiClassOneVsAll'), dict(score_function='L2'),
    dict(score_function='NewtonL2'), dict(grow_policy='Depthwise', loss='Logloss'),
    dict(grow_policy='Lossguide', loss='Logloss'), dict(use_pointwise_searcher=True),
    dict(feature_fraction=0.5),
])
def test_boosting_type_unset_resolves_plain_where_theirs_does(kw):
    assert GradientBoosting(n_estimators=1000, **kw)._resolved_boosting_type(100) == 'Plain'


@pytest.mark.parametrize('kw,exc,match', [
    (dict(grow_policy='Depthwise', loss='Logloss'), ValueError, 'nonsymmetric'),
    (dict(loss='MultiClass'), ValueError, "can't be used with ordered"),
    (dict(score_function='L2'), ValueError, "can't be used with ordered"),
    (dict(leaf_estimation_method='Exact', loss='MAE'), ValueError, 'Exact leaf estimation'),
    (dict(loss='QueryRMSE', bootstrap_type='No'), NotImplementedError, 'query grouping'),
    (dict(use_pointwise_searcher=True), ValueError, 'doc-parallel'),
    (dict(feature_fraction=0.5), NotImplementedError, 'feature_fraction'),
])
def test_explicit_ordered_refusals_by_name(kw, exc, match):
    with pytest.raises(exc, match=match):
        GradientBoosting(boosting_type='Ordered', **kw)


def test_fold_options_are_ordered_only():
    with pytest.raises(ValueError, match='greater than 1'):
        GradientBoosting(fold_len_multiplier=1.0)
    with pytest.raises(ValueError, match='read only by Ordered'):
        GradientBoosting(boosting_type='Plain', fold_len_multiplier=3.0)
    with pytest.raises(ValueError, match='read only by Ordered'):
        # unset boosting type resolving Plain at fit time (100 iterations)
        _fit_fake(GradientBoosting(n_estimators=100, fold_permutation_block=16))
    # permutation_count is Ordered's too, and refused where the fit is Plain
    GradientBoosting(permutation_count=2)
    with pytest.raises(ValueError, match='permutation_count'):
        GradientBoosting(boosting_type='Plain', permutation_count=2)
    with pytest.raises(ValueError, match='permutation_count'):
        _fit_fake(GradientBoosting(n_estimators=100, permutation_count=2))


def test_ordered_fit_sends_the_ordered_tail():
    m = GradientBoosting(loss='Logloss', n_estimators=20, boosting_type='Ordered',
                         fold_len_multiplier=1.5, fold_permutation_block=16)
    args = _fit_fake(m)
    strs = args[7]
    assert strs[4:] == ['GreedyLogSum', 'Ordered',
                        str(_struct.unpack('<Q', _struct.pack('<d', 1.5))[0]), '16']
    assert m.boosting_type_ == 'Ordered'
    # a default Plain fit keeps the four-string call
    p = GradientBoosting(n_estimators=100)
    assert len(_fit_fake(p)[7]) == 4 and p.boosting_type_ == 'Plain'


def test_ordered_fit_takes_an_eval_set():
    # their test cursor is restated (dynamic_boosting.h:423-430): the eval
    # arrays cross with the Ordered tail, as a Plain fit's do
    m = GradientBoosting(loss='Logloss', n_estimators=600)
    args = _fit_fake(m, eval_set=(np.ones((4, 3), np.float32), np.array([0, 1, 0, 1], np.float32)))
    assert m.boosting_type_ == 'Ordered'
    assert args[6][20] == 4          # n_eval_rows
    assert args[7][5] == 'Ordered'


# ---- boost_from_average on MAE / Quantile / MAPE (lane/catboost-parity) ----
# `get_scale_and_bias()[1]` of CatBoost 1.2.10 CPU (iterations=1, depth=2,
# boost_from_average=True, thread_count=1) on the cases `_bias_cases` draws,
# as float64 bits, recorded on this Mac 2026-09-19. Both branches of their
# CalcSampleQuantile (below and at 100 rows), rounded targets (the delta
# adjust's tie arms, and a MAPE median on a -0.0 target).
CATBOOST_CPU_BIAS_BITS = [
    '000000c01677f63f',
    '000000602b1bc3bf',
    '000000c01677f63f',
    '000000803a77e23f',
    '000000e0fdffef3f',
    '000000e0fdffefbf',
    '000000e0fdffef3f',
    '0000000000000080',
    '000000c035d4eb3f',
    '00000000f496fcbf',
    '000000c035d4eb3f',
    '00000000076ad83f',
    '000000000100f03f',
    '000000e0fdffefbf',
    '000000000100f03f',
    '000000000000f03f',
    '00000060b24ce13f',
    '00000000b650f5bf',
    '00000060b24ce13f',
    '00000000166cb13f',
    '000000a0f7c6b0be',
    '00000000ffffffbf',
    '000000a0f7c6b0be',
    '000000000000e039',
    '000000c0593def3f',
    '000000c08956f2bf',
    '000000c0593def3f',
    '000000803bd1d23f',
    '000000000100f03f',
    '000000e0fdffefbf',
    '000000000100f03f',
    '0000000000000000',
    '000000c0f63af03f',
    '0000008012e6f0bf',
    '000000c0f63af03f',
    '0000008058e5d73f',
    '000000e0fdffef3f',
    '000000e0fdffefbf',
    '000000e0fdffef3f',
    '000000000000c839',
]


def _bias_cases():
    rng = np.random.default_rng(3)
    for n in (37, 99, 100, 1000, 5003):
        for kind in ("cont", "ties"):
            X = rng.normal(size=(n, 4)).astype(np.float32)
            y = rng.normal(size=n).astype(np.float32) * 3 + 1
            if kind == "ties":
                y = np.round(y).astype(np.float32)
            for loss, alpha in (("MAE", None), ("Quantile", 0.25), ("Quantile", 0.5), ("MAPE", None)):
                yield X, y, loss, alpha


def test_boost_from_average_bias_is_catboost_cpu_bits():
    got = []
    for X, y, loss, alpha in _bias_cases():
        extra = {} if alpha is None else {"loss_alpha": alpha}
        m = GradientBoosting(n_estimators=1, max_depth=2, loss=loss, random_strength=0.0,
                             bootstrap_type="No", **extra).fit(X, y)
        got.append(np.float64(m.bias_).tobytes().hex())
    assert got == CATBOOST_CPU_BIAS_BITS
