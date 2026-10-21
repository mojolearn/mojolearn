# PCA native owner completion

This is an internal implementation and qualification record. The public PCA
route has not changed, and this source slice has not been compiled or executed.
It is not evidence of public native compliance or release qualification.

## Existing interface

`PCA` accepts `n_components`, `copy`, `whiten`, `svd_solver`, `tol`,
`iterated_power`, `n_oversamples`, `power_iteration_normalizer`, `random_state`
and the inherited `numeric_mode`. Its operations are `fit`, `fit_transform`,
`transform`, `inverse_transform`, `save`, `load`, parameter access/reset,
fitted-state inspection, sklearn tags and actual vendor/mode reporting. There
is no existing `score`, `score_samples`, `get_covariance` or `get_precision`
method to migrate.

Exact fitting supports covariance aliases `auto`, `covariance_eigh`, `jacobi`,
and full SVD, including the existing identical-mode wide route and one-feature
covariance route. Component selection accepts integer/None, a variance fraction
or Minka MLE. `randomized` is a separate supported algorithm, never substituted
by an exact solver. Whitening and externally replaced supported fitted buffers
must retain their behavior. The format remains `mojolearn-pca-1`, with its five
float32 arrays, float64 noise, four int64 metadata words and estimator/mode/solver
strings. `HostPCA` has separate CPU admission and saved-model semantics.

## First private source slice

- `bindings/pca_native.mojo` provides `pca_c`, registered as `pca_native_call` by
  the shared tokenizer binding owner. ABI: `(model, operation, args, context)`.
- `_native_pca_bridge.connect(PCA)` connects exact types and verified C entries
  during test/setup only. Every operation uses the resulting cached entry and
  context; it never reconnects or calls a Python preparation helper.
- Exact/full fitting, fractional/MLE selection, whitening and transform/inverse
  orchestration call the existing numerical entries. MLE uses the existing
  resident device operation on GPU and the declared host entry on CPU. No
  numerical algorithm, fold, route policy or performance default is changed.
- Native descriptors carry validated dimensions. Exposed arrays still use
  native-owned ndarray/buffer storage and mutable Python API metadata. This
  does **not** claim a persistent all-struct model representation.
- Constructor/get/set/fitted metadata and native archive read/write preserve the
  existing scalar and file schema. Invalid updates validate before clearing
  fitted state. Native input and model-class admission refuses Python hooks.
- The private tests compare full fit/transform/inverse state bits with the
  existing route, check an independent float64 SVD subspace/spectrum, consume
  outputs, preserve complete archive bytes and audit whole operations. They
  include corrupt state/archive and poisoned Python helper/protocol cases.

## Required before public activation

1. Implement the randomized owner, preserving centered input, seeded normal
   draws, power iterations, orthonormalization/rank deficiency, thin SVD, sign
   choices, total variance/noise and all supported options. Reuse a native
   matrix/handle owner with cleanup for every allocation; never call `_Kit`,
   `_M` or `_rsvd_core` Python drivers.
2. Preserve sparse CSR/CSC/COO/DOK/LIL/DIA/BSR ingestion (matrix and array,
   32/64-bit indices), duplicate accumulation semantics and all supported input
   shapes through native conversion. Custom Python `toarray` callbacks are
   refused under the owner's native-only contract, without removing standard
   sparse formats.
3. Complete inherited validation: ignored `y` input/length checks, pandas
   feature-name storage/checks/warnings, copied-input reporting for pandas and
   Torch, sklearn transformer tags and public NumPy/Array output behavior.
   The private dense slice deliberately does not certify these paths.
4. Preserve the CPU training authorization gate and HostPCA mode, fit refusal,
   archive class compatibility, inference and model digest interfaces. The
   private owner currently admits exact PCA, not HostPCA or arbitrary subclasses.
5. Connect metadata once at package integration setup, then replace public
   methods with one cached native binding call. Audit the **first public call**
   and the transitive operation, including mode guards, replacement arrays,
   serializer failures and cleanup; private entry tests do not establish that.
6. Compile the changed tokenizer closure centrally and run CPU/Metal tests;
   qualify NVIDIA/AMD, installed-package and same-version identity coverage
   separately. Existing upstream PythonObject lifetime findings remain open;
   no refcount compensation or unsupported compiler workaround is introduced.

## Qualification commands (after registration and a matching frozen build)

Use the orchestrator's frozen package/native-closure setup and existing CPU
reference training authorization, then run:

```sh
python -m pytest python/mojolearn/tests/test_pca_exact_native_owner.py -q \
  --junitxml=/absolute/evidence/pca-exact-native.xml \
  > /absolute/evidence/pca-exact-native.log 2>&1
```

Record vendor, numeric mode, exact source/native hashes, process exit code,
test/pass/fail/skip counts and retained full logs. All runtime results remain
pending until that command executes against the matching compiled slice.
