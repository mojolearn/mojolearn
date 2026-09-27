"""THE LINEAR LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `linear` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_linear",) once bindings/build_x_linear.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_linear", binding="_mojolearn_x_linear_host",
                        routes="_mojolearn_x_linear", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_linear",)
FAMILIES = (
    dict(
        family="x_linear",
        binding="_mojolearn_x_linear_host",
        routes="_mojolearn_x_linear",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "x-sgd-clf", "x-sgd-reg",
            "x-glm-poisson", "x-glm-gamma", "x-glm-tweedie",
            "x-huber",
            "x-bayes-ridge", "x-ard",
            "x-lars", "x-lasso-lars",
            "x-quantile",
            "x-perceptron",
            "x-pa-clf", "x-pa-reg",
            "x-sgd-ocsvm",
            "x-ridge-clf", "x-ridge-cv",
            "x-lasso-cv", "x-enet-cv",
            "x-logistic-cv",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "SGDClassifier", "SGDRegressor",
            "PoissonRegressor", "GammaRegressor", "TweedieRegressor",
            "HuberRegressor",
            "BayesianRidge", "ARDRegression",
            "Lars", "LassoLars",
            "QuantileRegressor",
            "Perceptron",
            "PassiveAggressiveClassifier", "PassiveAggressiveRegressor",
            "SGDOneClassSVM",
            "RidgeClassifier", "RidgeCV",
            "LassoCV", "ElasticNetCV",
            "LogisticRegressionCV",
        ),
        display="the linear expansion lane (SGD, GLMs, Huber, Bayesian, LARS, quantile, CV, isotonic)",
        host_modules=(
            "x_linear/ops.mojo", "x_linear/dispatch.mojo", "x_linear/sgd.mojo",
            "x_linear/glm.mojo", "x_linear/lbfgs.mojo", "x_linear/huber.mojo",
            "x_linear/bayes.mojo", "x_linear/lars.mojo",
            "x_linear/quantile.mojo", "x_linear/ridge.mojo",
            "x_linear/cd.mojo", "x_linear/logcv.mojo",
        ),
        exports=(
            "x_linear_host_numeric_mode", "x_linear_host_vendor", "x_linear_host_column", "x_linear_host_sabotage",
            "x_linear_fit", "x_linear_decision", "x_linear_numeric_mode", "x_linear_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the linear expansion lane's CPU route (x_linear/, pass 1, PENDING).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-sgd-clf": "SGDClassifier",
    "x-sgd-reg": "SGDRegressor",
    "x-glm-poisson": "PoissonRegressor",
    "x-glm-gamma": "GammaRegressor",
    "x-glm-tweedie": "TweedieRegressor",
    "x-huber": "HuberRegressor",
    "x-bayes-ridge": "BayesianRidge",
    "x-ard": "ARDRegression",
    "x-lars": "Lars",
    "x-lasso-lars": "LassoLars",
    "x-quantile": "QuantileRegressor",
    "x-perceptron": "Perceptron",
    "x-pa-clf": "PassiveAggressiveClassifier",
    "x-pa-reg": "PassiveAggressiveRegressor",
    "x-sgd-ocsvm": "SGDOneClassSVM",
    "x-ridge-clf": "RidgeClassifier",
    "x-ridge-cv": "RidgeCV",
    "x-lasso-cv": "LassoCV",
    "x-enet-cv": "ElasticNetCV",
    "x-logistic-cv": "LogisticRegressionCV",
}
PUBLIC_PENDING_LANES = {
    "x-sgd-clf": "no reference",
    "x-sgd-reg": "no reference",
    "x-glm-poisson": "no reference",
    "x-glm-gamma": "no reference",
    "x-glm-tweedie": "no reference",
    "x-huber": "no reference",
    "x-bayes-ridge": "no reference",
    "x-ard": "no reference",
    "x-lars": "no reference",
    "x-lasso-lars": "no reference",
    "x-quantile": "no reference",
    "x-perceptron": "no reference",
    "x-pa-clf": "no reference",
    "x-pa-reg": "no reference",
    "x-sgd-ocsvm": "no reference",
    "x-ridge-clf": "no reference",
    "x-ridge-cv": "no reference",
    "x-lasso-cv": "no reference",
    "x-enet-cv": "no reference",
    "x-logistic-cv": "no reference",
}
