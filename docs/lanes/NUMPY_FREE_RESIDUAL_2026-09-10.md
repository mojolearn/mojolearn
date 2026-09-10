# NumPy-free integration and remaining work

User direction, September 10: remove NumPy from the shipped runtime and
optimize native operations that lag it. Keep NumPy as an optional benchmark
and correctness oracle. GPU learners only; compiled host buffer operations
are part of the GPU product. Publication of 0.8.0 remains paused pending
installed-wheel qualification.

## Integrated source

The preserved `numpy-free-0.7` work is merged with current main, resolving
conflicts while retaining parallel_groves, class weights, GBDT adapters,
metrics/scalers/CV, wide IDENTICAL PCA, kNN batching, generalized byte LM,
resident training, schedules/accumulation, Samba and transformer ring caches.
All estimators use the shared Array/buffer layer. Runtime metadata no longer
requires NumPy. Built-in CV cloning and fold metadata use the standard
library; sklearn pipelines and splitters remain optional interoperability. Optional verification references may still require the test
extra. This changes array returns from ndarray to Array; classes are Python
lists. NumPy callers can obtain zero-copy views through `np.asarray`.

Shared native helpers perform casts, transposes, finite validation, column
means, centering, row scaling, probability validation/packing and byte row
gathering. Raw allocation replaces anonymous mmap destinations for native
conversions. Views pin the allocation; pickle retains data, shape, layout
and read-only state, never borrowed addresses or GPU handles.

## Measured host conversions

`tools/bench_numpy_free_conversion.py` runs alternating arms in one process,
two warmups and ten measured samples. Inputs are 2,000,000 × 20, read-only;
the NumPy baseline includes the former tiled C-to-F path. The native base
extension was built at the release's apple-m1 CPU baseline. Full output
bytes match. These are host conversion measurements on one Mac, not GPU
training speedups or NVIDIA performance evidence.

| Conversion | NumPy median ms | Native median ms |
|---|---:|---:|
| float64 C → float32 C | 5.69 | 5.73 |
| float64 C → float32 F | 21.21 | 16.51 |
| float32 C → float32 F | 22.13 | 11.59 |
| float64 F → float32 C | 13.77 | 11.94 |
| float32 F → float32 C | 12.49 | 11.82 |

The flat cast is approximately parity; layout conversions improve in this
run. The allocator comparison that motivated the change measured the old
mmap flat cast at 13.10 ms and raw allocation at 5.92 ms, with NumPy at
6.11 ms. Do not attribute these gains to IDENTICAL arithmetic.

Retained integrated samples, path names, output hashes and binary hash:
[conversion evidence](../../bench/results/numpy-free/2026-09-10-host-conversion.json).

## Remaining priorities and release gates

1. Build and qualify installed wheels on CUDA and HIP, plus supported macOS
   Python versions, in environments without NumPy. Run real fit/predict,
   metrics, scaler, checkpoint and resident training paths; import-only
   checks do not establish end-to-end independence.
2. Requalify changed numeric boundaries: OLS/ridge centering now uses defined
   sequential Float64 column sums rather than NumPy's blocked reduction.
   FAST/DETERMINISTIC GBDT sigmoid uses the existing native helper. Existing
   certification does not automatically cover these changes. Preserve
   IDENTICAL GPU reduction schedules and verify cross-vendor output bytes.
3. Measure whole-fit and repeated-fit costs on representative large NVIDIA
   datasets in IDENTICAL mode, and decision-tree FAST on Apple. Compare
   competitors there; these host timings do not establish learner parity.
4. Profile remaining integer widening/casting, label encoding, weighting,
   array gathers and checkpoint packing. Move expensive elementwise host
   loops to shared native helpers; reuse output/storage where lifetime and
   concurrency allow it. Avoid creating per-estimator converters.
5. Extend large conversion cases to wide, strided, non-native-endian and
   integer buffers, testing exact dtype/range/ownership semantics. Big-endian
   inputs are normalized by compiled `array.byteswap`; zero-copy views
   continue to require native endian.
6. Retire the superseded release candidate tag deliberately after source and
   artifacts are frozen. Rebuild native files; do not overlay new Python
   calls onto binaries missing the helpers. Publish only the qualified set.

## Local validation and corrected fixture pins

The follow-up full source suite passes 943 tests and 89 subtests, with 53
optional/inapplicable skips. Rebuilding the IDENTICAL metrics extension did
not restore the old UMAP pins: both current layouts instead match every
word of the previously retained Apple AND NVIDIA H100 native captures in
`bench/results/umap_portable_host_math_2026-09-10/`. The Python tests still
used earlier captures predating the device optimizer. Their literal pins
now reference those independently recorded current captures (16 and 48
layout words). No tolerance was relaxed and no algorithm was changed.
The earlier pre-merge-wrapper comparison remains historical evidence in
[the comparison](../../bench/results/numpy-free/2026-09-10-umap-head-comparison.log).
This resolves the two local failures; fresh installed release artifacts
still require qualification.

`tools/check_numpy_free_runtime.py` blocks NumPy imports while running real
GPU RF/ET fit/predict, scoring and both scalers. The local source-tree check
passed. Run it from an installed-wheel environment without PYTHONPATH for
release qualification; the source run is not a substitute. Base native
helpers were rebuilt in all three modes; each mode passed 139 host helper
and conversion checks (32 inapplicable no-conversion combinations skipped).
Additional NumPy-free CV tests pass with both NumPy and sklearn imports
blocked. The real GPU runtime smoke, including CV, passed on local CPython
3.10, 3.11, 3.12, 3.13 and 3.14; these are source-tree checks.

## Dependency audit and release checks

| Dependency | Role and action |
|---|---|
| NumPy | Removed from required Python runtime metadata. Keep optional conversion/serialization/numerical test oracles and explicit transformer diagnostic references. |
| SciPy | Removed the spectral affinity boundary's import. Caller-supplied COO/CSR/CSC objects use their `tocoo()` protocol; dense inputs need no SciPy. |
| scikit-learn | Optional external Pipeline/tags/exceptions interoperability and test oracle. Built-in CV and estimators do not require installation. |
| PyTorch | Optional neural correctness oracle; not a runtime requirement. |
| setuptools and wheel | Python wheel build dependencies, not installed runtime requirements. Retain the working packaging toolchain. |
| Mojo and MAX | Build toolchain and compiled runtime support. Native runtime libraries are bundled by release packaging; removing them requires replacing the execution backend. |
| GPU driver/runtime | Metal/CUDA/HIP platform requirements remain. This is a GPU library. |

Both macOS and Linux release qualification now run the installed package's
NumPy-blocked check BEFORE installing NumPy for reference smoke tests. The
check verifies the package comes from the venv, runtime metadata has no
required distributions, and NumPy is absent. macOS `--no-gpu` only checks
import/Array and keeps its explicit device-not-tested label. The full
macOS interpreter matrix still runs all requested numeric modes using the
reference harness afterward. Linux records runtime and test dependencies
separately. Ten shell-harness tests pass, including a sabotage proving a
failed dependency-free check cannot be hidden by successful oracle tests.

These changes repair the qualification harness; they do not constitute a
new installed-wheel release run. No 0.8.0 package has been published by this
work.
