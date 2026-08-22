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
  lgbm-rf-cpu          LightGBM boosting_type='rf', a plain random forest;
                       registered now so the comparator exists, but there is
                       NO mojolearn-rf arm yet -- see PARITY_NOTES["no-rf"]

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
        "CatBoost's RMSE default boosts from the target mean; mojolearn "
        "refuses boost_from_average by name (the cursor seeds at zero, "
        "OPTIONS.md). For RMSE the two are the same arithmetic, so the arm "
        "centers y before fit and adds the mean back at predict, INSIDE the "
        "fit timer. Without this the arm spots CatBoost mean(y)*0.9^ntrees "
        "of head start and the accuracy column measures the seed, not the "
        "trees (measured: MAE 243 at 20 trees on year). The Logloss analog "
        "(log-odds prior) cannot be emulated by relabeling and remains a "
        "recorded gap on binary datasets."
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
        self._y_offset = 0.0
        X = np.ascontiguousarray(data.X_train, dtype=np.float32)
        with Timer() as t:
            y = np.ascontiguousarray(data.y_train, dtype=np.float32)
            if data.learning_task == LearningTask.REGRESSION:
                # PARITY_NOTES["boost_from_average"]; inside the timer
                # because it is part of this arm's training work.
                self._y_offset = float(np.mean(y, dtype=np.float64))
                y = (y - np.float32(self._y_offset)).astype(np.float32)
            self.model = model.fit(X, y)
        return t.interval

    def test(self, data):
        X = np.ascontiguousarray(data.X_test, dtype=np.float32)
        if data.learning_task == LearningTask.CLASSIFICATION:
            # gbm-bench's binary metrics expect a positive-class
            # probability, matching lgb.Booster.predict for "binary".
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X) + self._y_offset

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
