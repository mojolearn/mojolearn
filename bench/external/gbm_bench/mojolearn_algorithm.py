# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""mojolearn adapters for NVIDIA's gbm-bench.

This file is injected into a gbm-bench checkout by
`bench/external/patch_gbm_bench.py`. Each mojolearn arm deliberately mirrors
the gbm-bench arm of the library it is a port of -- the GBDT arm mirrors
`CatAlgorithm`, the forest arm mirrors `SkRandomForestAlgorithm` -- because
the whole point of running someone else's harness is that our arm is
configured the way theirs is and not the way we would have chosen. Where a
parameter has no mojolearn equivalent the difference is recorded in
`PARITY_NOTES` rather than silently dropped.

Arms registered here:

  mojolearn-gbdt-gpu   the CatBoost oblivious-tree port on Metal; the
                       side-by-side comparator is gbm-bench's own `cat-cpu`
  mojolearn-et-gpu     the cuML-design ExtraTrees port on Metal; the
                       side-by-side comparators are `skl-et-cpu` (the
                       algorithm's definition) and `lgbm-et-cpu` (LightGBM's
                       nearest forest)
  mojolearn-rf-gpu     the cuML RandomForest port on Metal (quantile
                       splits, with-replacement bootstrap); comparators are
                       gbm-bench's own `skrf` and `lgbm-rf-cpu`
  skl-et-cpu           sklearn ExtraTrees, an ADDED arm (gbm-bench ships
                       `skrf-cpu` but no ExtraTrees)
  lgbm-et-cpu          LightGBM boosting_type='rf' + extra_trees=true, the
                       closest thing LightGBM offers to an ExtraTrees forest
  lgbm-rf-cpu          LightGBM boosting_type='rf', a plain random forest

`device='gpu'` is explicit on every mojolearn arm, never 'auto', so a run
that cannot reach the accelerator fails loudly instead of quietly reporting
a CPU number under a GPU label. There is no mojolearn CPU arm because the
library has no CPU path at all.
"""

import numpy as np

import mojolearn
from mojolearn.extratrees import ExtraTreesClassifier, ExtraTreesRegressor
from mojolearn.randomforest import (
    RandomForestClassifier as MojoRFClassifier,
    RandomForestRegressor as MojoRFRegressor,
)
import sklearn.ensemble as sken
import lightgbm as lgb

from algorithms import Algorithm, Timer, shared_params
from datasets import LearningTask


# Differences between these arms and the gbm-bench arms they mirror. Anything
# added here belongs in the writeup beside the numbers.
PARITY_NOTES = {
    "border_count": (
        "gbm-bench's cat-cpu passes no border_count, so CatBoost CPU runs "
        "its default 254. mojolearn's default is 128 (CatBoost's GPU "
        "default). The arm sets border_count=254 EXPLICITLY so both arms "
        "quantize identically and the device is the only variable."
    ),
    "reg_lambda": (
        "gbm-bench's shared reg_lambda=1 is CatBoost's l2_leaf_reg. Same "
        "quantity, mojolearn spells it the CatBoost way."
    ),
    "scale_pos_weight": (
        "CatBoost's binary-only scale_pos_weight=w is class_weights=[1, w] "
        "in mojolearn. Same reweighting, different spelling."
    ),
    "multiclass": (
        "mojolearn's GradientBoosting python surface has no MultiClass; the "
        "Mojo layer implements it but the wrapper is one-dimensional. The "
        "GBDT arm REFUSES multiclass datasets by name instead of silently "
        "fitting something else. The ExtraTrees arm handles multiclass."
    ),
    "boost_from_average": (
        "PORTED IN THE ENGINE 2026-08-22 (gbdt/metrics/"
        "optimal_const_for_loss.mojo; check-bfa-oracle proves the bias "
        "bit-equal to CatBoost's own get_scale_and_bias on every arm). "
        "The arm passes NOTHING: both libraries resolve the same "
        "data-dependent default (auto-true for RMSE, false for Logloss -- "
        "Logloss is NOT on their AdjustBoostFromAverageDefaultValue list, "
        "which an earlier version of this note got wrong). The y-centering "
        "shim this arm used to carry is deleted."
    ),
    "rf-quantile-splits": (
        "mojolearn-rf-gpu is the cuML RandomForest port: splits are "
        "searched over at most 128 per-feature QUANTILES (cuML's design), "
        "while sklearn's skrf searches exact thresholds. Faster and a "
        "different algorithm; accuracy sits beside the timing so the "
        "reader can weigh both. Bootstrap defaults match sklearn's RF "
        "(with-replacement, n rows per tree); LightGBM's rf mode cannot "
        "match that (see lgbm-rf-bagging)."
    ),
    "lgbm-rf-bagging": (
        "LightGBM's rf boosting REQUIRES bagging_freq>0 and "
        "bagging_fraction<1 (it refuses to run otherwise), so lgbm-et-cpu "
        "and lgbm-rf-cpu use bagging_fraction=0.632/bagging_freq=1, the "
        "conventional bootstrap-expectation stand-in. mojolearn's "
        "ExtraTrees, like sklearn's, gives every tree ALL rows "
        "(bootstrap=False default). Row sampling therefore differs between "
        "the lgbm forest arms and the other two BY LIGHTGBM'S OWN "
        "CONSTRAINT; skl-et-cpu is the like-for-like comparator."
    ),
    "lgbm-rf-features": (
        "LightGBM's rf mode runs feature_fraction=1.0 (every feature at "
        "every node) where mojolearn-rf-gpu, skrf and the RF definition "
        "use max_features='sqrt'. MEASURED on covtype 2026-08-22: this "
        "asymmetry IS the F1/Precision gap -- at max_features=1.0 our arm "
        "scores Acc 0.7567 / F1 0.7478 / Prec 0.7548 against their "
        "0.7170 / 0.7466 / 0.7880, ahead on accuracy and F1. n_bins "
        "128 vs their max_bin 255 was probed the same day and moves "
        "NOTHING on covtype or year (255-bin deltas within 0.0003), so "
        "quantization is exonerated. The arms keep their own defaults in "
        "the table (each mirrors its own library's forest); this note is "
        "the decoder."
    ),
    "skl-threads": (
        "gbm-bench's own sklearn arms pass no n_jobs, which runs sklearn on "
        "ONE core. skl-et-cpu is our ADDED arm and it gets n_jobs=<cpus> "
        "(the same count every other arm receives), because beating a "
        "single-core sklearn is not a result."
    ),
}


def _forest_params(args):
    """The forest-arm shape of shared_params, exactly how gbm-bench's own
    SkRandomForestAlgorithm builds it: drop the boosting-only knobs, keep
    max_depth, take ntrees."""
    params = shared_params.copy()
    del params["reg_lambda"]
    del params["learning_rate"]
    params["n_estimators"] = args.ntrees
    params.update(args.extra)
    return params


class MojolearnGbdtGPUAlgorithm(Algorithm):
    """The CatBoost oblivious-tree port, configured as gbm-bench's
    CatAlgorithm configures CatBoost."""

    def configure(self, data, args):
        params = {
            "n_estimators": args.ntrees,
            "max_depth": shared_params["max_depth"],
            "learning_rate": shared_params["learning_rate"],
            "l2_leaf_reg": shared_params["reg_lambda"],
            "border_count": 254,  # PARITY_NOTES["border_count"]
        }
        if data.learning_task == LearningTask.REGRESSION:
            params["loss"] = "RMSE"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["loss"] = "Logloss"
            # CatAlgorithm's scale_pos_weight, spelled the mojolearn way.
            positives = np.count_nonzero(data.y_train)
            if positives:
                params["class_weights"] = [
                    1.0, len(data.y_train) / positives
                ]
        else:
            raise NotImplementedError(PARITY_NOTES["multiclass"])
        params.update(args.extra)
        return params

    def fit(self, data, args):
        params = self.configure(data, args)
        model = mojolearn.GradientBoosting(**params)
        X = np.ascontiguousarray(data.X_train, dtype=np.float32)
        y = np.ascontiguousarray(data.y_train, dtype=np.float32)
        with Timer() as t:
            # boost_from_average is the ENGINE's now, resolved by the same
            # data-dependent rule CatBoost applies to its own arm --
            # PARITY_NOTES["boost_from_average"].
            self.model = model.fit(X, y)
        return t.interval

    def test(self, data):
        X = np.ascontiguousarray(data.X_test, dtype=np.float32)
        if data.learning_task == LearningTask.CLASSIFICATION:
            # gbm-bench's binary metrics expect a positive-class
            # probability, matching lgb.Booster.predict for "binary".
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class MojolearnExtraTreesGPUAlgorithm(Algorithm):
    """The cuML-design ExtraTrees port, configured as gbm-bench's
    SkRandomForestAlgorithm configures its forest."""

    def _estimator(self, data, params):
        if data.learning_task == LearningTask.REGRESSION:
            return ExtraTreesRegressor(device="gpu", **params)
        return ExtraTreesClassifier(device="gpu", **params)

    def fit(self, data, args):
        params = _forest_params(args)
        model = self._estimator(data, params)
        X = np.ascontiguousarray(data.X_train, dtype=np.float32)
        y = data.y_train
        if data.learning_task != LearningTask.REGRESSION:
            y = y.astype(np.int64)
        with Timer() as t:
            self.model = model.fit(X, y)
        return t.interval

    def test(self, data):
        X = np.ascontiguousarray(data.X_test, dtype=np.float32)
        if data.learning_task == LearningTask.CLASSIFICATION:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class MojolearnRandomForestGPUAlgorithm(Algorithm):
    """The cuML RandomForest port, configured as gbm-bench's own `skrf`
    arm configures its forest (`_forest_params`), with cuML's quantile
    splits -- PARITY_NOTES["rf-quantile-splits"]."""

    def _estimator(self, data, params):
        if data.learning_task == LearningTask.REGRESSION:
            return MojoRFRegressor(device="gpu", **params)
        return MojoRFClassifier(device="gpu", **params)

    def fit(self, data, args):
        params = _forest_params(args)
        model = self._estimator(data, params)
        X = np.ascontiguousarray(data.X_train, dtype=np.float32)
        y = data.y_train
        if data.learning_task != LearningTask.REGRESSION:
            y = y.astype(np.int64)
        with Timer() as t:
            self.model = model.fit(X, y)
        return t.interval

    def test(self, data):
        X = np.ascontiguousarray(data.X_test, dtype=np.float32)
        if data.learning_task == LearningTask.CLASSIFICATION:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class SkRandomForestCPUAllCoresAlgorithm(Algorithm):
    """sklearn's RandomForest with n_jobs=<cpus>: the fair multicore RF
    baseline. An ADDED arm -- gbm-bench's own `skrf` passes no n_jobs and
    therefore runs sklearn on ONE core, and beating a single-core sklearn
    is not a result (PARITY_NOTES["skl-threads"])."""

    def _estimator(self, data, params):
        if data.learning_task == LearningTask.REGRESSION:
            return sken.RandomForestRegressor(**params)
        return sken.RandomForestClassifier(**params)

    def fit(self, data, args):
        params = _forest_params(args)
        params["n_jobs"] = args.cpus if args.cpus else -1
        model = self._estimator(data, params)
        with Timer() as t:
            self.model = model.fit(data.X_train, data.y_train)
        return t.interval

    def test(self, data):
        if data.learning_task == LearningTask.CLASSIFICATION:
            return self.model.predict_proba(data.X_test)[:, 1]
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class SkExtraTreesCPUAlgorithm(Algorithm):
    """sklearn's ExtraTrees, the algorithm's definition and therefore the
    like-for-like comparator for mojolearn-et-gpu. An ADDED arm; gbm-bench
    ships skrf-cpu but no ExtraTrees."""

    def _estimator(self, data, params):
        if data.learning_task == LearningTask.REGRESSION:
            return sken.ExtraTreesRegressor(**params)
        return sken.ExtraTreesClassifier(**params)

    def fit(self, data, args):
        params = _forest_params(args)
        params["n_jobs"] = args.cpus if args.cpus else -1
        model = self._estimator(data, params)
        with Timer() as t:
            self.model = model.fit(data.X_train, data.y_train)
        return t.interval

    def test(self, data):
        if data.learning_task == LearningTask.CLASSIFICATION:
            return self.model.predict_proba(data.X_test)[:, 1]
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class LgbmForestAlgorithm(Algorithm):
    """LightGBM in its rf boosting mode, parameterized the way gbm-bench's
    LgbmAlgorithm parameterizes LightGBM wherever a knob carries over."""

    extra_trees = False
    # None = LightGBM's default (cpu). The NVIDIA arms set "cuda" -- their
    # modern CUDA learner, not the legacy OpenCL "gpu" device. On a box
    # where their GPU arm exists, benchmarking only their CPU arm would be
    # choosing the weaker opponent on the vendor's own hardware.
    device_type = None

    def configure(self, data, args):
        params = shared_params.copy()
        del params["learning_rate"]  # rf boosting ignores it
        params.update({
            "boosting_type": "rf",
            "extra_trees": self.extra_trees,
            # PARITY_NOTES["lgbm-rf-bagging"]
            "bagging_freq": 1,
            "bagging_fraction": 0.632,
            "max_leaves": 256,
            "nthread": args.cpus,
            "verbose": -1,
        })
        if self.device_type is not None:
            params["device_type"] = self.device_type
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "regression"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary"
            params["scale_pos_weight"] = (
                len(data.y_train) / np.count_nonzero(data.y_train)
            )
        elif data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            params["objective"] = "multiclass"
            params["num_class"] = np.max(data.y_test) + 1
        params.update(args.extra)
        return params

    def fit(self, data, args):
        dtrain = lgb.Dataset(data.X_train, data.y_train,
                             free_raw_data=False)
        params = self.configure(data, args)
        with Timer() as t:
            self.model = lgb.train(params, dtrain, args.ntrees)
        return t.interval

    def test(self, data):
        if data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            prob = self.model.predict(data.X_test)
            return np.argmax(prob, axis=1)
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class LgbmExtraTreesCPUAlgorithm(LgbmForestAlgorithm):
    extra_trees = True


class LgbmRandomForestCPUAlgorithm(LgbmForestAlgorithm):
    extra_trees = False


class LgbmExtraTreesGPUAlgorithm(LgbmForestAlgorithm):
    """Their CUDA learner in ET mode -- the honest NVIDIA opponent.
    Requires a CUDA build of LightGBM; the pip wheel raises, and a raise
    must surface as a refusal, not as a missing row."""

    extra_trees = True
    device_type = "cuda"


class LgbmRandomForestGPUAlgorithm(LgbmForestAlgorithm):
    """Their CUDA learner in plain rf mode -- see LgbmExtraTreesGPU."""

    extra_trees = False
    device_type = "cuda"
