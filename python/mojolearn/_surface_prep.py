"""THE PREP LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `prep` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_prep",) once bindings/build_x_prep.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_prep", binding="_mojolearn_x_prep_host",
                        routes="_mojolearn_x_prep", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_prep",)
FAMILIES = (
    dict(
        family="x_prep",
        binding="_mojolearn_x_prep_host",
        routes="_mojolearn_x_prep",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "x-prep-robust-scaler", "x-prep-maxabs-scaler",
            "x-prep-ordinal-encoder", "x-prep-onehot-encoder",
            "x-prep-target-encoder", "x-prep-simple-imputer",
            "x-prep-kbins",
            "x-prep-gaussian-nb", "x-prep-multinomial-nb", "x-prep-bernoulli-nb",
            "x-prep-lda", "x-prep-qda",
            "x-prep-quantile-transformer", "x-prep-power-transformer",
            "x-prep-normalizer",
            "x-prep-polynomial-features",
            "x-prep-spline-transformer",
            "x-prep-binarizer",
            "x-prep-label-encoder",
            "x-prep-label-binarizer",
            "x-prep-multilabel-binarizer",
            "x-prep-iterative-imputer",
            "x-prep-variance-threshold",
            "x-prep-select-kbest",
            "x-prep-mutual-info",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "RobustScaler", "MaxAbsScaler",
            "OrdinalEncoder", "OneHotEncoder",
            "TargetEncoder", "SimpleImputer", "KBinsDiscretizer",
            "GaussianNB", "MultinomialNB", "BernoulliNB",
            "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis",
            "QuantileTransformer", "PowerTransformer",
            "Normalizer",
            "PolynomialFeatures",
            "SplineTransformer",
            "Binarizer",
            "LabelEncoder",
            "LabelBinarizer",
            "MultiLabelBinarizer",
            "IterativeImputer",
            "VarianceThreshold",
            "SelectKBest",
        ),
        display="preprocessing additions, naive Bayes and discriminant analysis",
        host_modules=(
            "x_prep/host/program.mojo", "x_prep/common.mojo", "x_prep/prims.mojo", "x_prep/eigh.mojo",
            "x_prep/units.mojo", "x_prep/target.mojo", "x_prep/kbins.mojo",
            "naive_bayes/nb.mojo", "naive_bayes/da.mojo",
            "x_prep/transform.mojo", "x_prep/spline.mojo", "x_prep/iterative.mojo", "x_prep/stats.mojo", "x_prep/mutual_info.mojo",
        ),
        exports=(
            "x_prep_host_numeric_mode", "x_prep_host_vendor", "x_prep_host_column", "x_prep_host_sabotage",
            "x_prep_run", "x_prep_numeric_mode", "x_prep_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the prep lane's CPU route (preprocessing additions, naive Bayes, discriminant analysis).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-prep-robust-scaler": "RobustScaler",
    "x-prep-maxabs-scaler": "MaxAbsScaler",
    "x-prep-ordinal-encoder": "OrdinalEncoder",
    "x-prep-onehot-encoder": "OneHotEncoder",
    "x-prep-target-encoder": "TargetEncoder",
    "x-prep-simple-imputer": "SimpleImputer",
    "x-prep-kbins": "KBinsDiscretizer",
    "x-prep-gaussian-nb": "GaussianNB",
    "x-prep-multinomial-nb": "MultinomialNB",
    "x-prep-bernoulli-nb": "BernoulliNB",
    "x-prep-lda": "LinearDiscriminantAnalysis",
    "x-prep-qda": "QuadraticDiscriminantAnalysis",
    "x-prep-quantile-transformer": "QuantileTransformer",
    "x-prep-power-transformer": "PowerTransformer",
    "x-prep-normalizer": "Normalizer",
    "x-prep-polynomial-features": "PolynomialFeatures",
    "x-prep-spline-transformer": "SplineTransformer",
    "x-prep-binarizer": "Binarizer",
    "x-prep-label-encoder": "LabelEncoder",
    "x-prep-label-binarizer": "LabelBinarizer",
    "x-prep-multilabel-binarizer": "MultiLabelBinarizer",
    "x-prep-iterative-imputer": "IterativeImputer",
    "x-prep-variance-threshold": "VarianceThreshold",
    "x-prep-select-kbest": "SelectKBest",
    "x-prep-mutual-info": "mutual_info_classif / mutual_info_regression",
}
PUBLIC_PENDING_LANES = {
    "x-prep-robust-scaler": "no reference",
    "x-prep-maxabs-scaler": "no reference",
    "x-prep-ordinal-encoder": "no reference",
    "x-prep-onehot-encoder": "no reference",
    "x-prep-target-encoder": "no reference",
    "x-prep-simple-imputer": "no reference",
    "x-prep-kbins": "no reference",
    "x-prep-gaussian-nb": "no reference",
    "x-prep-multinomial-nb": "no reference",
    "x-prep-bernoulli-nb": "no reference",
    "x-prep-lda": "no reference",
    "x-prep-qda": "no reference",
    "x-prep-quantile-transformer": "no reference",
    "x-prep-power-transformer": "no reference",
    "x-prep-normalizer": "no reference",
    "x-prep-polynomial-features": "no reference",
    "x-prep-spline-transformer": "no reference",
    "x-prep-binarizer": "no reference",
    "x-prep-label-encoder": "no reference",
    "x-prep-label-binarizer": "no reference",
    "x-prep-multilabel-binarizer": "no reference",
    "x-prep-iterative-imputer": "no reference",
    "x-prep-variance-threshold": "no reference",
    "x-prep-select-kbest": "no reference",
    "x-prep-mutual-info": "no reference",
}
