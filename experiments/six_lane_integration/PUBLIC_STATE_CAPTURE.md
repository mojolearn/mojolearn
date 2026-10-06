# Public fitted-state capture contracts

The future worker can capture complete typed fitted-state exports for PCA and
KMeans through their existing public `save` methods. This changes evidence glue
only (`bench_board_state.py` and `six_lane_evidence.py`). It does not change model
arithmetic, public model implementations, benchmark settings, or compiled code.

The source review used numerical freeze
`4fc63c9d5486261f27fdd55eb31d1900e2cc0e8f` and integration base
`f689732258552bdfb33aa5c48369fb5ff7ef9a85`. No model was fitted, loaded, predicted,
transformed or assessed during this review. Capture remains unverified with real
models until a later authorized scored execution. Existing completed receipts
are unchanged; an old untyped save hash cannot be upgraded retrospectively to a
complete typed-state comparison.

## Source coverage

| Estimator | Retained fitted state in its public save | Contract decision |
| --- | --- | --- |
| PCA | Components, mean, singular values, explained variance and ratio, noise variance, fitted component/feature/sample counts, whitening, solver and numeric mode | `mojolearn.public-fitted-state/pca-1` |
| KMeans | Centers, fitted labels, inertia, iteration count, fixed-point sum/weight scales, fitted dimensions/row count, metric/init names and codes, fit controls and numeric mode | `mojolearn.public-fitted-state/kmeans-1` |
| LinearRegression / OLS | Coefficients, intercept, feature count, fit-intercept flag and numeric mode; **retained training means are omitted** | `UNAVAILABLE` for complete fitted-state capture; public-save hash remains partial |

PCA's evidence is in `python/mojolearn/decomposition.py`, particularly
`PCA.fit`, `_fit_randomized`, `save` and `load` (save starts at line 516 at the
reviewed source). Both fit paths retain their fitted numerical arrays in the
Python owner; transform/inverse-transform consume those arrays, the fitted
counts and whitening state. The public archive contains those arrays and the
additional fitted variance diagnostics. There is no retained optimizer or
incremental-fit state in this class.

KMeans's evidence is in `python/mojolearn/cluster.py`, `KMeans.fit`, `save` and
`load` (save starts at line 429). The binding returns fitted arrays and scalar
diagnostics to the owner. The archive includes labels and diagnostics in
addition to inference centers. The nine `<i8` metadata values are k, features,
fit rows, iterations, metric code, init code, max iterations, restarts and seed.
The five `<f8` real values are inertia, sum scale, weight scale, tolerance and
oversampling factor.

OLS's `python/mojolearn/linear_model.py:_save_linear` and
`LinearRegression.save/load` explicitly preserve what predict reads and omit
training-time means. Fit retains `_x_mean` and `_y_mean` (including the centered
TSQR path). The capture helper names both omissions and does not inspect private
arrays, reconstruct their values, or call fit/predict to fill the gap. A reviewed
complete public exporter is still needed before claiming complete OLS state
identity. Coefficient/intercept identity alone remains a narrower scope.

These contracts cover retained **fitted numerical state**, not an arbitrary
Python object dump or a refit checkpoint. Input-copy diagnostics and binding
module/device caches are execution metadata. PCA constructor settings are pinned
separately by the worker's estimator-settings receipt. KMeans `init_centroids`
is an original refit input, explicitly excluded by the public save contract;
its provenance belongs to the workload/settings, while learned centers are
captured. These exclusions are retained in capture provenance. No scope claim
extends to a new estimator or a new save-format version automatically.

## Future worker use

`capture_model` continues to prefer a public `state_dict` when present. Otherwise
it accepts only the explicitly reviewed PCA/KMeans owner types and versioned NPZ
schemas. It invokes `save` once after clocks stop and reads with
`allow_pickle=False`. It checks the exact field set, dtype, shape, fitted counts,
saved owner, mode and relevant name/code consistency. Missing or additional
fields, unexpected dtype/shape, unknown schema, or export failure produce an
explicit unavailable result. Other save-capable models remain partial.

For recipe generation, import the metadata-only helper:

```python
from bench_board_state import public_fitted_state_paths

workload['model_state_paths'] = public_fitted_state_paths('PCA')
# Or public_fitted_state_paths('KMeans') for that family.
```

The PCA paths are `$.components`, `$.estimator`, `$.explained_variance`,
`$.explained_variance_ratio`, `$.format`, `$.mean`, `$.meta`,
`$.noise_variance`, `$.numeric_mode`, `$.singular_values`, `$.svd_solver`.

The KMeans paths are `$.centers`, `$.estimator`, `$.format`, `$.init`, `$.labels`,
`$.meta`, `$.metric`, `$.numeric_mode`, `$.reals`.

An omitted recipe path list uses the helper's reviewed complete contract. An
explicit partial or incompatible list produces `scope_not_qualified`, with
missing and unexpected paths retained. The later comparison input must pin the
same full path set. A capture is not itself an identity assessment, a quality
gate pass, or promotion evidence.

`HostPCA` and `HostKMeans` inherit these public save schemas. Their saved class
name is normalized to the logical family name for comparison; numerical arrays
and dtypes are unchanged. The unmodified complete export hash and exact original
class tag remain in `model_state.provenance`, alongside the contract and source
locator. Host inference-export support does not establish full host fit coverage
(for example, `HostPCA._dense_binding` explicitly refuses dense CPU fits).

Add the new capture helper only to a subsequent authorized harness freeze. Do
not modify an active frozen worker or rerun completed cells merely to obtain a
new capture. Old missing-state cells remain visibly incomplete.

## Validation limits

The new checks use invented NPZ/schema fixtures with fake save-only owners and
fit/predict/transform methods that refuse execution. They cover complete schemas,
unknown/missing fields, type/shape errors, host class normalization, partial
recipe declarations, OLS omissions and export failures. The existing receipt
fixtures check compatibility of historical receipt formatting and hashing.
Neither suite imports a product binding, runs an estimator, compares actual
candidate state, measures performance, or establishes cross-vendor identity.
