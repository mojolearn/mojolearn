#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""E2 driver: the SUB-FEATURE MATRIX, one identity card per cell.

E1 (`tools/e1_traced_fit.py`) certified ONE configuration per family --
ET classification, RF regression, symmetric-tree RMSE -- across three GPU
vendors. This driver is the sweep the paper's claim actually needs: every
loss, bootstrap, score function, leaf estimator, searcher, bin width,
categorical path, NaN mode and forest criterion the Python surface
exposes, each fitted at a fixed seed on byte-identical inputs, each
leaving a per-stage card and a prediction hash. Run it on each machine at
the same commit under NUMERIC_IDENTICAL (see tools/e1_bootstrap.sh, which
calls it after the E1 driver), then `tools/e2_matrix_diff.py` turns the
two directories into the IDENTICAL / DIVERGENT-at-stage / REFUSED table.

THREE VERDICTS PER CELL, ALL OF THEM RESULTS:
  identical   -- prediction hash equal AND every card stage equal
  divergent   -- first differing stage named by the card differ
  refused     -- the estimator raised BY NAME (a REFUSE arm, an option the
                 port does not carry); both machines refusing with the same
                 message is the passing result for that cell

EACH CELL RUNS IN ITS OWN SUBPROCESS. A device fault in one configuration
must not erase the sixty cards behind it, and a refused cell must not be
confused with a crashed one: the parent records `crashed` with the exit
code, distinct from `refused` with the message.

INPUTS ARE A PURE FUNCTION OF THE SEED, integer-exact where a float op
could go through a platform BLAS or libm (the first E1 run's finding):
numpy's MT19937 stream is platform-independent, the targets come from
integer matmul scaled by a power of two, the planted NaNs and category
codes come from integer hashing. `inputs` hashes in e2_cells.json prove
the inputs matched before any fit is compared.

TWO PREDICTION HASHES where a link function exists: `predictions` is the
RAW output (device arithmetic only) and `proba` is `predict_proba`, whose
sigmoid for Logloss/CrossEntropy is numpy's float64 `exp` ON THE HOST
(ensemble.py `predict_proba`). A proba-only divergence is therefore a
host-libm finding, not a device one; the two hashes keep those apart.

usage: PYTHONPATH=python python3 tools/e2_matrix_fit.py <out_dir> [--only SUBSTR]
       (internal) ... --cell NAME <out_dir>
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

SEED = 20260822
N_ROWS, N_COLS = 20000, 24


def sha256_of(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def make_inputs():
    """Every array here is bit-identical on every platform by construction;
    see the module docstring. Keep the E1 recipe for X/y_reg/y_clf
    VERBATIM so the E1 cells in this matrix reproduce the E1 hashes."""
    rng = np.random.RandomState(SEED)
    X = rng.rand(N_ROWS, N_COLS).astype(np.float32)
    w = rng.rand(N_COLS).astype(np.float32)
    q = (X * 1024.0).astype(np.int64)
    wq = (w * 1024.0).astype(np.int64)
    y_reg = ((q @ wq).astype(np.float64) * 2.0 ** -20).astype(np.float32)
    s = q.sum(axis=1)
    y_clf = (s > np.median(s)).astype(np.float32)
    assert float(y_reg.min()) > 0.0, "positive-target losses need y > 0"

    # a probability-valued target for CrossEntropy: s < 24*1024 < 2^15,
    # so s * 2^-15 is exact in float32
    y_prob = (s.astype(np.float64) * 2.0 ** -15).astype(np.float32)
    # three classes at exact integer tertile cuts of s
    srt = np.sort(s)
    c1, c2 = srt[N_ROWS // 3], srt[(2 * N_ROWS) // 3]
    y_mc = ((s > c1).astype(np.int64) + (s > c2).astype(np.int64)).astype(
        np.float32)
    # hashed integer sample weights 1..5
    idx = np.arange(N_ROWS, dtype=np.int64)
    sw = (1 + (idx * 7919) % 5).astype(np.float32)
    # planted NaNs at hashed cells: row i gets a NaN in column
    # (i*2654435761 mod 24) when (i*40503 mod 97) == 0 -- ~1% of rows
    X_nan = X.copy()
    hit = ((idx * 40503) % 97) == 0
    cols = ((idx * 2654435761) % N_COLS)[hit]
    X_nan[idx[hit], cols] = np.nan
    # categorical codes in columns 0 and 1: cardinality 3 (one-hot
    # territory) and 50 (CTR territory), integer-hashed from the bins --
    # AND A TARGET THAT DEPENDS ON THEM. The first smoke of this matrix
    # fitted the cat cells against y_reg, whose signal lives in the other
    # 22 columns, and the greedy never picked a cat split: cat_features,
    # one_hot_features and permutation_count all produced the SAME model
    # as the all-numeric fit, which read as "cat_features is inert". It is
    # not; the fixture was (`uniform test data hides permutation`, the
    # house rule). y_regcat adds an exact integer offset per code so the
    # categorical columns are the strongest features in the cell.
    X_cat = X.copy()
    code0 = q[:, 0] % 3
    code1 = (q[:, 1] * 7 + q[:, 2]) % 50
    X_cat[:, 0] = code0.astype(np.float32)
    X_cat[:, 1] = code1.astype(np.float32)
    y_regcat = (y_reg.astype(np.float64) + 8.0 * code0
                + (code1 % 7)).astype(np.float32)
    y_clfcat = ((code0 + code1 % 7) > 4).astype(np.float32)
    # a disjoint eval set for the overfitting-detector cells
    Xe = rng.rand(5000, N_COLS).astype(np.float32)
    qe = (Xe * 1024.0).astype(np.int64)
    ye_reg = ((qe @ wq).astype(np.float64) * 2.0 ** -20).astype(np.float32)
    # ROW-COUNT REGIMES: several kernels dispatch on n_rows (ET's sampler
    # boundaries at 128/9216 draws, the GBDT small-bin drivers, block and
    # chunk counts that are f(n_rows) by design). 200k rows, same recipe,
    # drawn AFTER the arrays above so every existing hash is untouched.
    Xb = rng.rand(200000, N_COLS).astype(np.float32)
    qb = (Xb * 1024.0).astype(np.int64)
    yb_reg = ((qb @ wq).astype(np.float64) * 2.0 ** -20).astype(np.float32)
    sb = qb.sum(axis=1)
    yb_clf = (sb > np.median(sb)).astype(np.float32)
    return dict(X=X, y_reg=y_reg, y_clf=y_clf, y_prob=y_prob, y_mc=y_mc,
                sw=sw, X_nan=X_nan, X_cat=X_cat, y_regcat=y_regcat,
                y_clfcat=y_clfcat, Xe=Xe, ye_reg=ye_reg,
                Xb=Xb, yb_reg=yb_reg, yb_clf=yb_clf)


# ---------------------------------------------------------------- the cells
# name -> (kind, spec). kind selects the family; spec is the constructor
# kwargs plus a few driver keys prefixed with '_' (which X/y, fit kwargs,
# whether to hash predict_proba). Names are STABLE -- they are the row
# labels of the cross-vendor table and the card file names.

def gb(**kw):
    return ("gbdt", kw)


def et(cls, **kw):
    return ("et_" + cls, kw)


def rf(cls, **kw):
    return ("rf_" + cls, kw)


GB_BASE = dict(n_estimators=20, max_depth=6, learning_rate=0.3,
               border_count=128, random_state=7)

CELLS = {}


def cell(name, kind_spec):
    assert name not in CELLS, name
    CELLS[name] = kind_spec


# --- the E1 four, verbatim (they must reproduce the E1 hashes) ------------
cell("et_clf", et("clf", n_estimators=10, max_depth=8, random_state=7,
                  _y="y_clf", _proba=True))
cell("rf_reg", rf("reg", n_estimators=10, max_depth=8, random_state=7,
                  _y="y_reg"))
cell("gbdt_rmse", gb(loss="RMSE", _y="y_reg", **GB_BASE))
cell("gbdt_logloss", gb(loss="Logloss", _y="y_clf", _proba=True, **GB_BASE))

# --- GBDT: every loss (symmetric arm, greedy searcher, Cosine) ------------
cell("gbdt_crossentropy", gb(loss="CrossEntropy", _y="y_prob", _proba=True,
                             **GB_BASE))
cell("gbdt_quantile", gb(loss="Quantile", _y="y_reg", **GB_BASE))
cell("gbdt_quantile_a03", gb(loss="Quantile", loss_alpha=0.3, _y="y_reg",
                             **GB_BASE))
cell("gbdt_mae", gb(loss="MAE", _y="y_reg", **GB_BASE))
cell("gbdt_loglinquantile", gb(loss="LogLinQuantile", _y="y_reg", **GB_BASE))
cell("gbdt_mape", gb(loss="MAPE", _y="y_reg", **GB_BASE))
cell("gbdt_poisson", gb(loss="Poisson", _y="y_reg", **GB_BASE))
cell("gbdt_lq", gb(loss="Lq", loss_q=1.5, _y="y_reg", **GB_BASE))
cell("gbdt_expectile", gb(loss="Expectile", loss_alpha=0.3, _y="y_reg",
                          **GB_BASE))
cell("gbdt_tweedie", gb(loss="Tweedie", loss_variance_power=1.5, _y="y_reg",
                        **GB_BASE))
# delta 6 on a target in [3.7, 10.6]: the first trees see residuals on
# BOTH sides of delta, so the quadratic and linear arms are both live.
# delta 1 saturated every row (Der2 = 0 everywhere, Cosine ties every
# split, the cursor oscillates) and the 20-tree prediction was EXACTLY
# ZERO -- CatBoost CPU does the same (DEVIATION 257's audit), so it is
# their semantics, but an all-zero output certifies nothing
cell("gbdt_huber", gb(loss="Huber", loss_delta=6.0, _y="y_reg", **GB_BASE))
cell("gbdt_huber_d1_saturated", gb(loss="Huber", loss_delta=1.0, _y="y_reg",
                                   **GB_BASE))
cell("gbdt_multiclass", gb(loss="MultiClass", _y="y_mc", _proba=True,
                           **GB_BASE))

# --- GBDT: bootstrap types (RMSE) ---------------------------------------
cell("gbdt_rmse_bayesian", gb(loss="RMSE", bootstrap_type="Bayesian",
                              _y="y_reg", **GB_BASE))
cell("gbdt_rmse_bernoulli", gb(loss="RMSE", bootstrap_type="Bernoulli",
                               subsample=0.66, _y="y_reg", **GB_BASE))
cell("gbdt_rmse_poissonboot", gb(loss="RMSE", bootstrap_type="Poisson",
                                 subsample=0.66, _y="y_reg", **GB_BASE))
cell("gbdt_rmse_noboot", gb(loss="RMSE", bootstrap_type="No", _y="y_reg",
                            **GB_BASE))
cell("gbdt_logloss_bayesian", gb(loss="Logloss", bootstrap_type="Bayesian",
                                 _y="y_clf", **GB_BASE))

# --- GBDT: score functions ----------------------------------------------
cell("gbdt_rmse_l2", gb(loss="RMSE", score_function="L2", _y="y_reg",
                        **GB_BASE))
cell("gbdt_rmse_newtoncosine", gb(loss="RMSE", score_function="NewtonCosine",
                                  _y="y_reg", **GB_BASE))
cell("gbdt_rmse_newtonl2", gb(loss="RMSE", score_function="NewtonL2",
                              _y="y_reg", **GB_BASE))
cell("gbdt_logloss_newtonl2", gb(loss="Logloss", score_function="NewtonL2",
                                 _y="y_clf", **GB_BASE))
cell("gbdt_logloss_newtoncosine", gb(loss="Logloss",
                                     score_function="NewtonCosine",
                                     _y="y_clf", **GB_BASE))

# --- GBDT: leaf estimation overrides -------------------------------------
cell("gbdt_logloss_gradient", gb(loss="Logloss",
                                 leaf_estimation_method="Gradient",
                                 _y="y_clf", **GB_BASE))
cell("gbdt_logloss_newton3", gb(loss="Logloss",
                                leaf_estimation_iterations=3,
                                _y="y_clf", **GB_BASE))
# Exact IS the loss default for Quantile/MAE (CatBoost's choice), so the
# overrides that exercise a different estimator are Newton and Gradient
# DEVIATION 257: CatBoost REFUSES Newton for Quantile/MAE/MAPE/LogLinQuantile
# (EnsureNewtonIsAvailable); this cell's passing result is REFUSED= on every
# column
cell("gbdt_quantile_newton", gb(loss="Quantile", leaf_estimation_method="Newton",
                                _y="y_reg", **GB_BASE))
cell("gbdt_mae_gradient", gb(loss="MAE", leaf_estimation_method="Gradient",
                             _y="y_reg", **GB_BASE))
cell("gbdt_logloss_exact", gb(loss="Logloss", leaf_estimation_method="Exact",
                              _y="y_clf", **GB_BASE))
cell("gbdt_rmse_gradient", gb(loss="RMSE", leaf_estimation_method="Gradient",
                              _y="y_reg", **GB_BASE))

# --- GBDT: random strength, pointwise searcher ---------------------------
cell("gbdt_rmse_rs1", gb(loss="RMSE", random_strength=1.0, _y="y_reg",
                         **GB_BASE))
cell("gbdt_rmse_pointwise", gb(loss="RMSE", use_pointwise_searcher=True,
                               _y="y_reg", **GB_BASE))
cell("gbdt_logloss_pointwise", gb(loss="Logloss", use_pointwise_searcher=True,
                                  _y="y_clf", **GB_BASE))
cell("gbdt_rmse_pointwise_rs1", gb(loss="RMSE", use_pointwise_searcher=True,
                                   random_strength=1.0, _y="y_reg",
                                   **GB_BASE))
cell("gbdt_rmse_pointwise_bayesian", gb(loss="RMSE",
                                        use_pointwise_searcher=True,
                                        bootstrap_type="Bayesian",
                                        _y="y_reg", **GB_BASE))
cell("gbdt_multiclass_pointwise", gb(loss="MultiClass",
                                     use_pointwise_searcher=True,
                                     _y="y_mc", _proba=True, **GB_BASE))

# --- GBDT: categorical paths (one-hot + CTRs) ----------------------------
# measured 2026-08-23 on the signal-bearing target: all-numeric, cat [0,1],
# cat [1]+one_hot [0], cat [0], permutation_count 1 and 2 are SIX distinct
# models, so every knob here reaches the engine
cell("gbdt_rmse_cat", gb(loss="RMSE", cat_features=[0, 1], _X="X_cat",
                         _y="y_regcat", **GB_BASE))
cell("gbdt_rmse_cat_numeric_control", gb(loss="RMSE", _X="X_cat",
                                         _y="y_regcat", **GB_BASE))
cell("gbdt_logloss_cat", gb(loss="Logloss", cat_features=[0, 1], _X="X_cat",
                            _y="y_clfcat", _proba=True, **GB_BASE))
cell("gbdt_rmse_cat_onehot", gb(loss="RMSE", cat_features=[1],
                                one_hot_features=[0], _X="X_cat",
                                _y="y_regcat", **GB_BASE))
cell("gbdt_rmse_cat_perm1", gb(loss="RMSE", cat_features=[0, 1],
                               permutation_count=1, _X="X_cat",
                               _y="y_regcat", **GB_BASE))
cell("gbdt_rmse_cat_perm2", gb(loss="RMSE", cat_features=[0, 1],
                               permutation_count=2, _X="X_cat",
                               _y="y_regcat", **GB_BASE))
cell("gbdt_rmse_cat_estperm0", gb(loss="RMSE", cat_features=[0, 1],
                                  permutation_count=2,
                                  ctr_estimation_permutation_id=0,
                                  _X="X_cat", _y="y_regcat", **GB_BASE))
cell("gbdt_rmse_cat_pointwise", gb(loss="RMSE", cat_features=[0, 1],
                                   use_pointwise_searcher=True,
                                   _X="X_cat", _y="y_regcat", **GB_BASE))
cell("gbdt_multiclass_cat", gb(loss="MultiClass", cat_features=[0, 1],
                               _X="X_cat", _y="y_mc", _proba=True,
                               **GB_BASE))

# --- GBDT: NaN modes, bin widths, depth, misc ----------------------------
cell("gbdt_rmse_nan_min", gb(loss="RMSE", nan_mode="Min", _X="X_nan",
                             _y="y_reg", **GB_BASE))
cell("gbdt_rmse_nan_max", gb(loss="RMSE", nan_mode="Max", _X="X_nan",
                             _y="y_reg", **GB_BASE))
# 2026-08-23, the three owed knobs (E3_RESULTS.md): 'Forbidden' on a NaN
# matrix is a REFUSAL (a certified answer, same message on every vendor)
# and on a clean matrix is an ordinary fit through the same code (equal
# to gbdt_rmse by construction, and certified so); the Bayesian bootstrap
# at temperature 0 (every weight (-log u)^0 = 1, so equal to gbdt_rmse /
# gbdt_rmse_noboot by construction -- measured, da34f396f968e546) and 3 (the
# bagging_temperature -> weight = (-log(u))^t arm, DEVIATION 258's
# portable log/pow on the device); and the 'Simple' leaf estimator.
cell("gbdt_rmse_nan_forbidden_refused", gb(loss="RMSE", nan_mode="Forbidden",
                                           _X="X_nan", _y="y_reg", **GB_BASE))
cell("gbdt_rmse_nan_forbidden_clean", gb(loss="RMSE", nan_mode="Forbidden",
                                         _y="y_reg", **GB_BASE))
cell("gbdt_rmse_bayesian_temp0", gb(loss="RMSE", bootstrap_type="Bayesian",
                                    bagging_temperature=0.0, _y="y_reg",
                                    **GB_BASE))
cell("gbdt_rmse_bayesian_temp3", gb(loss="RMSE", bootstrap_type="Bayesian",
                                    bagging_temperature=3.0, _y="y_reg",
                                    **GB_BASE))
# (not RMSE: there 'Simple' IS the one Newton step the default takes and
# the cell would equal gbdt_rmse bit for bit -- measured, da34f396f968e546
# both -- so it could not show reach; Quantile's default is not Simple)
cell("gbdt_quantile_simple", gb(loss="Quantile", leaf_estimation_method="Simple",
                                _y="y_reg", **GB_BASE))
cell("gbdt_logloss_simple", gb(loss="Logloss", leaf_estimation_method="Simple",
                               _y="y_clf", _proba=True, **GB_BASE))
for bc in (1, 15, 64, 254):
    spec = dict(GB_BASE)
    spec["border_count"] = bc
    cell(f"gbdt_rmse_bins{bc}", gb(loss="RMSE", _y="y_reg", **spec))
spec = dict(GB_BASE)
spec["max_depth"] = 10
cell("gbdt_rmse_depth10", gb(loss="RMSE", _y="y_reg", **spec))
spec = dict(GB_BASE)
spec["max_depth"] = 2
cell("gbdt_rmse_depth2", gb(loss="RMSE", _y="y_reg", **spec))
# True is CatBoost's default for RMSE (the smoke measured True == unset),
# so False is the arm that moves
cell("gbdt_rmse_no_boost_from_avg", gb(loss="RMSE", boost_from_average=False,
                                       _y="y_reg", **GB_BASE))
cell("gbdt_rmse_sample_weight", gb(loss="RMSE", _y="y_reg",
                                   _fit=dict(sample_weight="sw"), **GB_BASE))
cell("gbdt_logloss_class_weights", gb(loss="Logloss", class_weights=[1.0, 2.0],
                                      _y="y_clf", **GB_BASE))
cell("gbdt_logloss_border", gb(loss="Logloss", loss_border=0.3, _y="y_prob",
                               **GB_BASE))
cell("gbdt_rmse_l2reg0", gb(loss="RMSE", l2_leaf_reg=0.0, _y="y_reg",
                            **GB_BASE))
cell("gbdt_rmse_od_iter", gb(loss="RMSE", od_type="Iter", od_wait=3,
                             use_best_model=True, _y="y_reg",
                             _fit=dict(eval_set=("Xe", "ye_reg")), **GB_BASE))
cell("gbdt_rmse_od_inctodec", gb(loss="RMSE", od_type="IncToDec",
                                 od_pvalue=0.5, _y="y_reg",
                                 _fit=dict(eval_set=("Xe", "ye_reg")),
                                 **GB_BASE))
# 200000 (the default) and 0 both mean "every row" on 20k rows; 5000
# exercises the border-search subsample
cell("gbdt_rmse_borders_sub5k", gb(loss="RMSE", border_build_max_samples=5000,
                                   _y="y_reg", **GB_BASE))

# --- GBDT: grow policies (DEVIATION 259, on the Python surface 2026-08-23)
# Depthwise and Lossguide build NON-SYMMETRIC trees through
# TGreedySubsetsSearcher<TNonSymmetricTree> and the estimator; the model
# text carries `ntree` records and the apply is their
# TAddModelDocParallel<TNonSymmetricTree>. Lossguide's score function is
# left UNSET so the cell takes CatBoost's own GPU default there, NewtonL2
# (catboost_options.cpp:980-991); `_lossguide_cosine` pins Cosine so the
# leafwise Cosine calcer (LOSSGUIDE.md DEVIATION 318) is in the sweep too.
cell("gbdt_rmse_depthwise", gb(loss="RMSE", grow_policy="Depthwise",
                               _y="y_reg", **GB_BASE))
cell("gbdt_logloss_depthwise", gb(loss="Logloss", grow_policy="Depthwise",
                                  _y="y_clf", _proba=True, **GB_BASE))
cell("gbdt_rmse_lossguide", gb(loss="RMSE", grow_policy="Lossguide",
                               _y="y_reg", **GB_BASE))
cell("gbdt_rmse_lossguide_leaves8", gb(loss="RMSE", grow_policy="Lossguide",
                                       max_leaves=8, _y="y_reg", **GB_BASE))
cell("gbdt_rmse_lossguide_cosine", gb(loss="RMSE", grow_policy="Lossguide",
                                      score_function="Cosine", _y="y_reg",
                                      **GB_BASE))
cell("gbdt_rmse_depthwise_minleaf200", gb(loss="RMSE", grow_policy="Depthwise",
                                          min_data_in_leaf=200, _y="y_reg",
                                          **GB_BASE))
# the two CatBoost REFUSES, mirrored by name: the doc-parallel pointwise
# searcher is oblivious-only (pointwise_non_symmetric.cpp:5), and their GPU
# registers no (MultiClass, Lossguide) trainer (multiclass.cpp:5-14,
# train.cpp:279). The passing verdict for both is REFUSED on every column.
cell("gbdt_rmse_depthwise_pointwise", gb(loss="RMSE", grow_policy="Depthwise",
                                         use_pointwise_searcher=True,
                                         _y="y_reg", **GB_BASE))
cell("gbdt_multiclass_lossguide", gb(loss="MultiClass", grow_policy="Lossguide",
                                     _y="y_mc", _proba=True, **GB_BASE))

# --- row-count regime: 200k rows (added for E2 round 2, 2026-08-23) -------
cell("gbdt_rmse_200k", gb(loss="RMSE", _X="Xb", _y="yb_reg", **GB_BASE))
cell("gbdt_logloss_200k", gb(loss="Logloss", _X="Xb", _y="yb_clf", **GB_BASE))
cell("gbdt_rmse_pointwise_200k", gb(loss="RMSE", use_pointwise_searcher=True,
                                    _X="Xb", _y="yb_reg", **GB_BASE))
cell("et_clf_200k", et("clf", n_estimators=10, max_depth=8, random_state=7,
                       _X="Xb", _y="yb_clf"))
cell("et_reg_200k", et("reg", n_estimators=10, max_depth=8, random_state=7,
                       _X="Xb", _y="yb_reg"))
cell("rf_reg_200k", rf("reg", n_estimators=10, max_depth=8, random_state=7,
                       _X="Xb", _y="yb_reg"))
cell("rf_clf_200k", rf("clf", n_estimators=10, max_depth=8, random_state=7,
                       _X="Xb", _y="yb_clf"))

# --- Extra Trees ----------------------------------------------------------
ET_BASE = dict(n_estimators=10, max_depth=8, random_state=7)
cell("et_reg", et("reg", _y="y_reg", **ET_BASE))
cell("et_clf_cpu", et("clf", device="cpu", _y="y_clf", _proba=True, **ET_BASE))
cell("et_reg_cpu", et("reg", device="cpu", _y="y_reg", **ET_BASE))
cell("et_clf_bootstrap", et("clf", bootstrap=True, _y="y_clf", **ET_BASE))
cell("et_clf_allfeat", et("clf", max_features=1.0, _y="y_clf", **ET_BASE))
cell("et_reg_halffeat", et("reg", max_features=0.5, _y="y_reg", **ET_BASE))
cell("et_clf_deep", et("clf", n_estimators=10, max_depth=None, random_state=7,
                       _y="y_clf"))
cell("et_clf_minleaf5", et("clf", min_samples_leaf=5, _y="y_clf", **ET_BASE))
cell("et_reg_minsplit20", et("reg", min_samples_split=20, _y="y_reg",
                             **ET_BASE))
cell("et_clf_3class", et("clf", _y="y_mc", _proba=True, **ET_BASE))
cell("et_clf_maxleaf", et("clf", max_leaf_nodes=64, _y="y_clf", **ET_BASE))
cell("et_reg_mid", et("reg", min_impurity_decrease=0.001, _y="y_reg",
                      **ET_BASE))
cell("et_clf_entropy", et("clf", criterion="entropy", _y="y_clf", **ET_BASE))
# DEVIATION 459/460 (2026-08-23): the two REFUSED= cells above became running
# cells; these three join them -- sklearn's alias of entropy (must hash equal
# to et_clf_entropy), the regressor's bootstrap, and bootstrap with sklearn's
# max_samples resolved to a count (0.5 -> 10000 rows per tree).
cell("et_clf_logloss_alias", et("clf", criterion="log_loss", _y="y_clf",
                                **ET_BASE))
cell("et_reg_bootstrap", et("reg", bootstrap=True, _y="y_reg", **ET_BASE))
cell("et_clf_bootstrap_maxsamples", et("clf", bootstrap=True, max_samples=0.5,
                                       _y="y_clf", **ET_BASE))

# --- Random Forest ---------------------------------------------------------
RF_BASE = dict(n_estimators=10, max_depth=8, random_state=7)
cell("rf_clf", rf("clf", _y="y_clf", _proba=True, **RF_BASE))
cell("rf_clf_entropy", rf("clf", criterion="entropy", _y="y_clf", **RF_BASE))
cell("rf_reg_poisson", rf("reg", criterion="poisson", _y="y_reg", **RF_BASE))
cell("rf_reg_gamma", rf("reg", criterion="gamma", _y="y_reg", **RF_BASE))
cell("rf_reg_invgauss", rf("reg", criterion="inverse_gaussian", _y="y_reg",
                           **RF_BASE))
cell("rf_clf_nobootstrap", rf("clf", bootstrap=False, _y="y_clf", **RF_BASE))
cell("rf_reg_maxsamples", rf("reg", max_samples=0.5, _y="y_reg", **RF_BASE))
cell("rf_clf_allfeat", rf("clf", max_features=1.0, _y="y_clf", **RF_BASE))
cell("rf_reg_bins32", rf("reg", n_bins=32, _y="y_reg", **RF_BASE))
cell("rf_clf_deep", rf("clf", n_estimators=10, max_depth=None, random_state=7,
                       _y="y_clf"))
cell("rf_reg_mid", rf("reg", min_impurity_decrease=0.001, _y="y_reg",
                      **RF_BASE))
cell("rf_clf_3class", rf("clf", _y="y_mc", _proba=True, **RF_BASE))
# RENAMED 2026-09-01 (DEVIATION 408), and the fit does not move. This cell
# passed sklearn's `max_leaf_nodes` while the wrapper aliased it onto cuML's
# `max_leaves`, so it has always set slot 5 and has always measured cuML's
# breadth-first cap. The alias is now refused by name and the cap is a
# keyword under its own name, so the SAME value reaches the SAME slot and
# the cell's hash is unchanged -- a `REFUSED=` here would be a regression.
# Its extratrees twin above tests sklearn's semantics under sklearn's name;
# the two cells finally exercise two algorithms rather than one twice.
cell("rf_clf_maxleaf", rf("clf", max_leaves=64, _y="y_clf", **RF_BASE))
cell("rf_reg_minleaf5", rf("reg", min_samples_leaf=5, _y="y_reg", **RF_BASE))
cell("rf_clf_streams1", rf("clf", n_streams=1, _y="y_clf", **RF_BASE))


# ---------------------------------------------------------------- one cell
def run_cell(name, out_dir):
    kind, spec = CELLS[name]
    spec = dict(spec)
    data = make_inputs()
    X = data[spec.pop("_X", "X")]
    y = data[spec.pop("_y")]
    want_proba = spec.pop("_proba", False)
    fit_kw = {}
    for k, v in spec.pop("_fit", {}).items():
        if k == "eval_set":
            fit_kw[k] = (data[v[0]], data[v[1]])
        else:
            fit_kw[k] = data[v]

    import mojolearn
    from mojolearn.extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    from mojolearn.randomforest import (RandomForestClassifier,
                                        RandomForestRegressor)
    ctor = {
        "gbdt": mojolearn.GradientBoosting,
        "et_clf": ExtraTreesClassifier, "et_reg": ExtraTreesRegressor,
        "rf_clf": RandomForestClassifier, "rf_reg": RandomForestRegressor,
    }[kind]

    card = os.path.join(out_dir, name + ".card")
    # THE TRACE APPENDS. The E1 driver's four fits write cards of the same
    # names into the same directory, and a card holding two fits is one
    # the differ refuses (and one this matrix's differ mis-read as
    # agreement before it learned to split them). Truncate first.
    if os.path.exists(card):
        os.remove(card)
    entry = {"kind": kind, "spec": {k: v for k, v in spec.items()}}
    t0 = time.time()
    os.environ["MOJOLEARN_IDENTITY_TRACE"] = card
    try:
        model = ctor(**spec).fit(X, y, **fit_kw)
        pred = model.predict(X)
        entry["predictions"] = sha256_of(pred)
        if want_proba:
            entry["proba"] = sha256_of(model.predict_proba(X))
        # the GBDT loss curves are device reductions (functionValue per
        # iteration; the held-out curve when an eval_set was given) --
        # a second certified output beside the predictions
        for attr in ("loss_curve_", "test_loss_curve_"):
            cur = getattr(model, attr, None)
            if cur is not None:
                entry[attr.rstrip("_")] = sha256_of(
                    np.asarray(cur, dtype=np.float64))
        for attr in ("best_iteration_", "stopped_early_"):
            if hasattr(model, attr):
                entry[attr.rstrip("_")] = int(getattr(model, attr))
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        # a REFUSE arm raising BY NAME is a result; the extratrees and
        # randomforest wrappers raise plain Exception/NotImplementedError
        # with "is not ported" / "refused" text, the gbdt wrapper raises
        # ValueError/NotImplementedError
        if isinstance(exc, (NotImplementedError, ValueError)) or any(
                t in str(exc) for t in ("not ported", "refus", "not supported",
                                        "not reachable", "does not support")):
            entry["refused"] = msg
        else:  # a real failure, also a result, named
            entry["error"] = msg
            entry["traceback"] = traceback.format_exc()[-2000:]
        return entry
    finally:
        os.environ.pop("MOJOLEARN_IDENTITY_TRACE", None)
        entry["seconds"] = round(time.time() - t0, 2)
    entry["card"] = os.path.basename(card) if os.path.exists(card) else None
    # the train-here-infer-there artifact, where the family serializes
    model_path = os.path.join(out_dir, name + ".model.npz")
    try:
        model.save(model_path)
        with open(model_path, "rb") as fh:
            entry["model"] = hashlib.sha256(fh.read()).hexdigest()
    except Exception as exc:
        entry["model_unsaved"] = f"{type(exc).__name__}: {exc}"
    return entry


def _numeric_mode():
    """What the package LOADED ('fast' | 'identical'), read back from the
    gbdt binary's compile-time answer -- the mode is a build define now
    (MOJOLEARN_NUMERIC_MODE=identical selects python/mojolearn/identical/)."""
    import mojolearn
    return mojolearn.numeric_mode()



def _commit(repo):
    """The commit, or the archive's own identity when there is no `.git`.

    DEVIATION 1935, 2026-08-28. This was a bare
    `subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)`, and on
    a box that received the tree as a `git archive` -- which is exactly what
    `tools/gemm_remote_leg.sh` ships, because the repo is private and the box
    has no credentials -- it raises CalledProcessError 128 and takes the whole
    driver down with it.

    That is not a hypothetical. The NVIDIA column of the 2026-08-28 round
    (`cc499f7`) built all ten identical bindings and then lost phase 3's
    traced fits, phase 4's E2 tree matrix and phase 7's E2U matrix to this
    line, so the round had an Apple column, an AMD column, and an NVIDIA
    column with no matrix in it -- for a provenance string, not for anything
    numeric. The DigitalOcean leg never hit it because it ships a `git
    bundle` and clones it, so its boxes do have a `.git`.

    The fallbacks are the same evidence the leg already writes, in the order
    they are trustworthy: the bootstrap's own `commit.txt`, then
    MOJOLEARN_COMMIT from the environment, then the honest string "unknown".
    A provenance field must never be able to end a measurement.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    for cand in (os.path.join(repo, "commit.txt"),):
        try:
            with open(cand) as fh:
                v = fh.read().strip()
            if v:
                return v
        except Exception:
            pass
    return os.environ.get("MOJOLEARN_COMMIT", "unknown")

def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--cell":
        name, out_dir = argv[1], os.path.abspath(argv[2])
        entry = run_cell(name, out_dir)
        with open(os.path.join(out_dir, name + ".cell.json"), "w") as fh:
            json.dump(entry, fh, indent=2, sort_keys=True)
        return

    out_dir = os.path.abspath(argv[0])
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]
    os.makedirs(out_dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = make_inputs()
    record = {
        "commit": _commit(repo),
        "numeric_mode": _numeric_mode(),
        "platform": platform.platform(),
        "machine": platform.node(),
        "inputs": {k: sha256_of(v) for k, v in data.items()},
        "cells": {},
    }
    names = [n for n in CELLS if only is None or only in n]
    print(f"E2 matrix: {len(names)} cells -> {out_dir}")
    for i, name in enumerate(names, 1):
        cell_json = os.path.join(out_dir, name + ".cell.json")
        if os.path.exists(cell_json):
            os.remove(cell_json)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cell", name,
             out_dir], cwd=repo, capture_output=True, text=True)
        dt = round(time.time() - t0, 1)
        if os.path.exists(cell_json):
            with open(cell_json) as fh:
                entry = json.load(fh)
        else:
            entry = {"crashed": proc.returncode,
                     "stderr_tail": proc.stderr[-1500:]}
        record["cells"][name] = entry
        verdict = ("REFUSED" if "refused" in entry else
                   "ERROR" if "error" in entry else
                   "CRASHED" if "crashed" in entry else
                   "pred " + entry["predictions"][:16])
        print(f"[{i:2d}/{len(names)}] {name:32s} {verdict:40s} {dt}s",
              flush=True)
        if "stderr_tail" in entry:
            print(entry["stderr_tail"][-600:])
        # write after every cell so a killed run still leaves a record
        with open(os.path.join(out_dir, "e2_cells.json"), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
    print("wrote", os.path.join(out_dir, "e2_cells.json"))


if __name__ == "__main__":
    main()
