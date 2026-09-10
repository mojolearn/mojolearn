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

## Local validation and known artifact issue

The merged source suite ran 930 passing tests, 88 passing subtests and
53 skips (including optional PyTorch reference checks). Two UMAP pinned
fixture checks fail against the existing local metrics binary. Running the
pre-merge NumPy wrapper and merged wrapper against that same IDENTICAL
binary gives identical output bits for both fixtures; the discrepancy is
pre-existing. Expected fixture bits were not changed. See
[the comparison](../../bench/results/numpy-free/2026-09-10-umap-head-comparison.log).
This still needs artifact/source qualification before publication.

`tools/check_numpy_free_runtime.py` blocks NumPy imports while running real
GPU RF/ET fit/predict, scoring and both scalers. The local source-tree check
passed. Run it from an installed-wheel environment without PYTHONPATH for
release qualification; the source run is not a substitute. Base native
helpers were rebuilt in all three modes; each mode passed 139 host helper
and conversion checks (32 inapplicable no-conversion combinations skipped).
Additional NumPy-free CV tests pass with both NumPy and sklearn imports
blocked. The real GPU runtime smoke, including CV, passed on local CPython
3.10, 3.11, 3.12, 3.13 and 3.14; these are source-tree checks.
