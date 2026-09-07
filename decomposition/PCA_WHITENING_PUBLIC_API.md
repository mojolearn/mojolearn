# PCA whitening binding completion

The source now connects the existing whitening kernels and host functions to
the Python PCA API through two additive `_mojolearn_estimators` exports:

- `pca_whiten_transform(x, mean, components, singular_values, out, params)`
- `pca_whiten_inverse_transform(scores, components, singular_values, mean, out, params)`

Array arguments are borrowed FP32 addresses. `params` is
`[query_rows, features, components, fit_rows]`. The fitted row count is retained
across transform calls. Existing unwhitened exports and their arithmetic are
unchanged. The native extension must be rebuilt with
`bindings/build_estimators.sh`; older binaries receive an explicit refusal
when either whitening export is absent.

For component row `c` with singular value `s`, forward scaling performs:

```
r = Float32(sqrt(Float64(fit_rows - 1)))
v = ftz(identical_mul(component[c, j], r))
component_forward[c, j] = ftz(identical_div(v, s))
```

For ordinary nonzero modes, this is the real-arithmetic factor
`1 / sqrt(explained_variance[c])`. Its pinned multiplication/division order is
retained instead of substituting a differently rounded variance-based
expression. Inverse scaling uses `Float32(1 / sqrt(Float64(fit_rows - 1)))`,
then multiplies by `s` with the same pinned seams. The copied components feed
the existing projection GEMM and inverse mean addition.

The existing cuML-derived skip-zero contract is explicit: when
`s < Float32(1e-10)`, skip the singular division or multiplication and retain
the sample-count scaling. This includes zero and subnormal singular values.
The threshold is strict; equality takes the ordinary scaling path. This is
not scikit-learn's epsilon clamp. Degenerate fitted columns do not promise
unit variance. Nonfinite input/model arrays, negative singular values or
explained variances, and nonfinite outputs are refused on the public
whitening path. Native input validation also checks dimensions, pointer spans
and output overlap before opening the device context.

The Python source tests in `python/mojolearn/tests/test_pca_whitening_surface.py`
cover ABI routing, older partial bindings, fitted row retention, the unchanged
unwhitened calls and finite-state refusals with host sentinels. Three opt-in
remote tests cover exact planted zero scaling/round-trip, unit sample variance
after fitting, and the strict skip threshold. Existing lower-level
`decomposition/checks/pca_check.mojo::check_whiten_edges` covers signed zeros,
subnormals, threshold neighbors and deliberate incorrect arithmetic controls.

Only the root agent may execute validation, under its remote CUDA/HIP resource
guard. A bounded public gate after rebuilding is:

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_RUN_PCA_WHITEN_GPU=1 \
  python -m pytest -q python/mojolearn/tests/test_pca_whitening_surface.py
```

This change is authored and unqualified. No tests, builds, models or
measurements were executed by the authoring subagent. Existing kernel code and
historical cards do not establish new Python-path, installed-wheel, or
cross-vendor whitening qualification; those need retained root-run evidence.
