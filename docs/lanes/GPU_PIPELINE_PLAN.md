# GPU pipeline and estimator compatibility plan

Decision and source audit: 2026-09-10, on
`lane/trees-gpu-growth-rf-hist-2026-09-09`. This extends the
[decision-tree roadmap](DECISION_TREE_ROADMAP.md) to the surrounding library.
The bounded A1 regression-error APIs and B1 RF/ET sklearn protocol are now
implemented; see [regression metrics](GPU_REGRESSION_METRICS.md) and
[forest compatibility](FOREST_SKLEARN_PROTOCOL.md) for contracts and qualification.
The remaining phases are planned. A complete cross-vendor pipeline has not
been qualified. Metrics and estimator compatibility take priority over
the longer-tail tree features.

## Decision: keep DETERMINISTIC for now

The three modes share source and compile-time policy; they are not three
separate learners. FAST permits the existing fast arithmetic choices.
DETERMINISTIC targets repeatability on a given device/configuration.
IDENTICAL additionally pins cross-vendor arithmetic for qualified profiles.
`checks/numerics.mojo` distinguishes `PIN_DETERMINISM` from
`PIN_CROSS_VENDOR`; portable transcendental/FTZ choices and histogram policy
are not all the same between the two pinned tiers.

Do not deprecate or silently alias DETERMINISTIC now. Its runtime advantage
over IDENTICAL is unmeasured in this audit, but its contract is distinct.
The tangible maintenance burden is an extra build/artifact/test column:
the source audit found explicit third-tier handling in 15 binding build
scripts, with deterministic mentions in 11 Python and 19 native test files.
These counts are not build-time or disk-size measurements.

Measure before reconsidering: compile wall time, artifact bytes, test time,
and matched FAST/DETERMINISTIC/IDENTICAL fit/predict/metric times across
representative estimators and vendors. If the middle mode has no meaningful
benefit, write a versioned migration proposal with saved-model loading,
warnings and compatibility tests. Existing models persist numeric-mode
metadata; changing a label must not silently select different arithmetic.
Until then, new pipeline operations use shared kernels with explicit mode
policies and all three modes receive appropriate correctness checks.

## Initial audit (before A1/B1 implementation)

| Surface | Present | Missing or incomplete |
| --- | --- | --- |
| Metrics | Fourteen public functions including accuracy, R² and clustering metrics | MSE/MAE/RMSE, log loss, confusion matrix, precision/recall/F1, ROC-AUC and PR curve |
| Reusable metric primitives | GPU contingency counts and SSE/reduction pieces used by existing metrics | Public contracts, weighting/label semantics and complete metric outputs |
| Preprocessing | Internal column means/centering and related numeric primitives | Public StandardScaler/MinMaxScaler, one-hot and target encoders |
| Model selection | Native seeded permutation/bootstrap/index primitives in `resample/` | Public train/test split, KFold/StratifiedKFold and cross-validation orchestration |
| Estimator protocol | Several non-tree estimators already implement `score`; tree fit methods return self and expose useful learned metadata | RF/ET/GBDT lack the complete get/set parameters, score and tags/clone contract |

Sources: `python/mojolearn/_metrics_impl.py`, `metrics/`, `core/`,
`resample/estimator.mojo`, `python/mojolearn/{_mode,randomforest,extratrees,ensemble}.py`.
Missing F1/precision/recall functions are not all named in `_NOT_PORTED`.
Some RF metric helpers are host loops; their existence is not a GPU metric API.
Native resampling infrastructure is useful but is not already KFold. Its
current permutation-statistic ranking has a 1,024-row limit and quadratic
work; it cannot simply become the general-purpose splitter. Existing GPU
float-key sorting also needs score/index association and tie grouping for AUC.

The absence of a matching cuML C++ kernel is not a technical reason to
refuse a standard metric or transformer. Use the mathematical definition,
scikit-learn behavior and appropriate upstream primitives as references;
implement and independently validate the Mojo GPU path. Update old absence
messages as each feature lands instead of advertising unimplemented names.

## Delivery sequence and implementation approach

| Phase | Concrete work and reference | Gates before completion |
| --- | --- | --- |
| A1: regression metrics | Implement public MSE, RMSE and MAE over GPU residual kernels and the existing pinned reduction infrastructure. Start with finite 1-D Float32 inputs, then weights/multioutput. Reuse audited SSE work; define result precision and finite/empty input behavior. | Hand-computed cases (weighted when enabled), independent high-precision reference, cancellation/extreme-value cases, mode and vendor checks, installed public calls. |
| A2: classification metrics | Expose integer confusion counts using contingency infrastructure; derive precision/recall/F1 with explicit binary/micro/macro/weighted conventions. Implement clipped-probability log loss using the pinned log path in IDENTICAL. | Label ordering, absent classes, zero division, weights, binary/multiclass and normalization oracles; integer count identity and final scalar bits. |
| A3: ranking curves | Implement binary ROC-AUC and PR curve from stable score ordering, grouped ties and prefix counts; share one ordered-count primitive. Assess existing GPU sort limits first, then add scalable stable sorting if needed. Define integration, endpoints and degenerate-class behavior. | Tied scores, shuffled equal-score rows, all-equal scores, weighted cases, independent rank/AUC and curve oracles, large inputs; extend multiclass/multilabel separately. |
| B1: sklearn protocol pilot | Start with RF/ET, a shared explicit parameter registry and raw constructor parameter storage. Ensure validated `_cfg` is rebuilt when parameters change; preserve mode in parameter discovery. Add get/set parameters, fitted-state checks, classifier/regressor tags and default accuracy/R² score through mode-aware metrics. Reuse sklearn public protocol where appropriate without requiring its training backend. | `clone` retains every parameter including numeric_mode, does not copy fitted state, nested Pipeline updates change native parameters, invalid combinations still refuse, fit returns self, serial GridSearchCV/refit works. |
| B2: GBDT classifier/regressor contract | Provide bounded sklearn classifier/regressor adapters around existing GBDT semantics before changing its raw `predict` API. Classification adapters use class predictions/probabilities; regression adapters use numeric predictions. Add accuracy/R² default score using the mode-aware metrics. | sklearn scoring uses the correct response method, labels/classes/tags agree, mode survives cloning/refit/save-load, explicit loss support. Preserve existing raw approximation prediction behavior. |
| C1: scalers | Mirror StandardScaler population-variance and MinMaxScaler contracts in Mojo: fixed reduction schedule, defined precision, zero-variance/range handling, then elementwise transform/inverse transform. Own learned statistics and propagate numeric_mode. | Independent moments, constant columns, weights where supported, NaN policy, overflow/cancellation, fit-transform/inverse behavior and round trips; transformed bytes compared across devices. |
| C2: encoders | Stable LabelEncoder/OrdinalEncoder vocabulary first, then one-hot: category order, unknown-category policy, serialized mappings and bounded dense output; sparse output follows a real sparse contract. Target encoding later with out-of-fold training values, smoothing and explicit leakage prevention. | Train/test category mismatch, deterministic codes, heldout leakage tests, fold ownership and category-map round trips. Do not expose a target encoder that fits on evaluation labels. |
| D1: splitting and folds | Build train_test_split and KFold on explicit index permutations and a stable integer RNG mapping; add StratifiedKFold with deterministic class allocation/ties. Reuse resample mechanisms where their algorithm and scale fit. | Exact index coverage/disjointness, stable seed vectors, size rounding, class imbalance and tiny classes; publish any RNG difference from sklearn rather than imply same-seed index equality. |
| D2: cross-validation/search | Implement serial cross_val_score orchestration around clone/fit/predict/metric; use sklearn GridSearchCV through compatibility first rather than build another search engine. Fit transformations and quantizers on training folds only. | No preprocessing/quantization/target leakage, explicit score direction, stable fold aggregation and best-parameter ties, reproducible refit, errors/cancellation and bounded device memory. |
| E: pipeline qualification | Compose a bounded numeric preprocessing → training → prediction → scoring workflow with owned/device-aware data and explicit mode at every stage. Save the transforms, model and schema together. | Cross-vendor intermediate and final identity using installed artifacts, complete provenance, independent accuracy checks and end-to-end time/memory measurements. |

A1/A2 and B1 can run as independent source lanes. A3 needs a sort design
audit, not just another reduction. C1 can reuse the metric reduction contract.
B2 depends on scoring semantics; D2 depends on cloning and fold generation.
Scalers and encoders also implement the shared parameter/clone/tags protocol
and fit/transform (with fit_transform delegation where appropriate), so
Pipeline can clone and tune every step, not only the final estimator.
Run device/build qualification serially on each shared device.

## Why sklearn compatibility is not just three methods

`NumericModeMixin` currently injects `numeric_mode` through a wrapped
constructor. Ordinary signature inspection follows the wrapped original and
can miss that argument, so a naive clone can revert to the process default.
RF/ET keep normalized parameters in `_cfg`; a generic `setattr` would leave
the actual fit configuration stale. GBDT normalizes constructor values and
initializes some learned attributes to None, requiring deliberate clone and
fitted-state handling. Keep raw user parameters separate from validated
native configuration and test real sklearn orchestration, not just getters.

Use the [scikit-learn estimator protocol](https://scikit-learn.org/stable/developers/develop.html)
as the compatibility specification: cloning, constructor parameters, tags,
learned attributes, response methods and metadata routing matter. Qualify an
explicit sklearn version range. Start with serial search; GPU process/thread
concurrency and per-worker context ownership are separate work.

## The first end-to-end IDENTICAL claim

Start with dense numeric binary classification and regression, one heldout
split, no missing values, explicit seeds, and a fixed documented dtype.
Use a GBDT or RF profile already qualified on the target devices. For
classification certify log loss and accuracy first, then add AUC; for
regression certify MSE/MAE/R². Include optional StandardScaler to test the
complete transformation contract, but do not make scaling mandatory for
trees or claim it improves their performance. Scaling is a more central
workflow for linear, distance and kernel estimators; qualify those separately.
[StandardScaler contract](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[ROC-AUC contract](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

Freeze input bytes/schema, split indices, preprocessing order, category
mappings, seeds and quantization configuration. Read back the compiled mode
and vendor for every native operation; estimator mode alone does not control
external preprocessing or a process-global metric. Add explicit metric mode
selection or a tested scoped pipeline context so mixed-mode processes cannot
silently score with another tier.

Compare learned scaling statistics, transformed bytes, model bytes,
predictions/probabilities and metric outputs on Metal/CUDA/HIP. For later CV,
also compare fold indices, per-fold state, aggregated scores, selected
parameters and final refit. Record source, wheel hashes, compiler, driver,
device and dataset hash. Run independent numerical references alongside bit
comparisons: equal bits alone do not establish correct statistics.

The current metric boundary uploads host arrays and creates a context per
call. Explicit reusable device buffers are a later performance slice; initial
GPU metrics must not be described as a fully resident pipeline.

Python orchestration and small host metadata operations remain allowed;
there is no new CPU tree-training backend. GPU-resident throughput is a
separate measurement from numerical identity. An external sklearn scaler or
scorer can make an interoperable pipeline useful immediately, but does not
qualify that entire pipeline as cross-vendor IDENTICAL.

Do not claim arbitrary sklearn pipelines are identical, or that no competitor
can offer a similar guarantee. Publish the precise certified workflow and
its intermediate evidence instead. This plan remains the implementation and qualification queue. A1 unweighted
Float32 errors and B1 forest compatibility are the first implemented slices;
weights, multiple outputs and broader protocol support remain pending.
