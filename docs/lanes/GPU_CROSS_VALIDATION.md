# Bounded serial cross-validation

Implemented 2026-09-10: `mojolearn.model_selection.cross_val_score`.
This is a D2 orchestration slice using optional sklearn, not completion of
D1 native GPU splitting. No new CPU learner is introduced.

```python
from sklearn.pipeline import Pipeline
from mojolearn import StandardScaler, GradientBoostingClassifier
from mojolearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scale', StandardScaler(numeric_mode='identical')),
    ('tree', GradientBoostingClassifier(numeric_mode='identical')),
])
# X: dense finite Float32 array; y: binary integer or string labels.
scores = cross_val_score(pipeline, X, y, cv=5)
```

Only dense 2-D NumPy X and 1-D NumPy y are accepted; dtypes are preserved.
Estimator validation controls numeric/label support. sklearn `check_cv`
selects unshuffled five-fold splitting by default, stratified for classification,
or accepts an explicit splitter or iterable. Groups are passed only to the
splitter. Host split metadata and array slicing are orchestration; this is not
a device-resident pipeline or a native Mojo RNG implementation.

Each fold has nonempty, unique integer indices, in range and with no train/test
overlap. All folds are materialized and checked before training starts; memory
therefore scales with the total index count. Test rows may repeat across folds
for repeated CV. Input index arrays are copied. Splitter and scorer code is
caller supplied and responsible for its own behavior.

A fresh sklearn clone fits each training fold. Include unfitted preprocessing
inside the pipeline so learned statistics and quantization belong to that fold;
this API cannot detect preprocessing performed before it is called. It leaves
the original estimator untouched and returns one Float64 host score per fold,
without averaging, selecting a model or refitting. Each fold clone is released
before the next fit; native cleanup remains owned by the existing estimators.

Default scoring calls `estimator.score`; the bounded MojoLearn tree adapters
use GPU accuracy or R². Callable scorers receive `(estimator, X_test, y_test)`
and must return one real scalar. Explicitly negate losses when higher scores
should be preferred, and pass the mode to custom GPU metrics. Named sklearn
scorers are not exposed by this bounded API. No identity guarantee is made for
arbitrary scorers, estimators or pipelines. Explicit modes on every step are
required for a future qualified pipeline.

Only `n_jobs=1` and `error_score='raise'` are accepted. Errors propagate;
nonfinite scalar scores are preserved, as opposed to silently replaced.
There is no fit metadata, sample weighting, eval_set forwarding, metadata
routing, sparse/precomputed-kernel contract or parallel execution. Using sklearn
GridSearchCV directly remains the existing serial search route.

Behavioral source: installed sklearn 1.8.0,
`model_selection/_validation.py:320,375` (split selection and clone per fold),
`:540-690` (`cross_val_score`) and `:820-865` (`_fit_and_score`). We inspected
those implementations. CV-1 bounds this to serial dense supervised execution
with propagated errors; CV-2 adds pre-fit overlap and duplicate checks. This
is host orchestration around existing Mojo GPU implementations, not an implementation of
a sklearn training kernel.

[Local evidence](../../bench/results/serial_cv_2026-09-10/RESULTS.md) covers
fold isolation and six small M4 GPU pipeline fits across all three modes.
Native splitters, weighted fitting/scoring, metadata routing, classifier-specific
CV fixtures, cancellation during native execution, large-CV memory behavior and
cross-vendor intermediate/final identity remain queued. No timing claim.
