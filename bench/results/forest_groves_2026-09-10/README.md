# Parallel-groves inference: residency, vector traversal and borrowed buffers

This campaign improves prediction for the same trained RF/ET forest. It changes
neither training nor the fixed 32-grove reduction graph. `parallel_groves` remains
an opt-in public engine; `sequential` remains the public default.

Three changes are implemented and selected inside the GPU engine:

- Validate/upload an owned model once and retain its GPU buffers/context.
- Traverse each tree once for 2–8 outputs, following nvForest's vector-leaf loop.
- Borrow contiguous host input/output during each synchronous native call,
  removing intermediate Lists and pinned output staging. Device I/O allocation,
  input upload and output readback still occur per call.

The [engine contract](../../../docs/FOREST_INFERENCE_ENGINES.md),
[source audit](../../../docs/lanes/GPU_FOREST_INFERENCE_NEXT.md) and
[measurement instructions](../../../docs/lanes/FOREST_INFERENCE_MEASUREMENT.md)
explain numerical semantics, ownership and remaining work. No CPU learner was
added. RF/ET share the kernel, ownership, binding helpers and Python cache.

## NVIDIA IDENTICAL public throughput

All models have 100 trees and maximum depth 16. Public calls include input/output
handling, validation and transfers; fitting is excluded. The three MojoLearn
arms use the same trained forest and require exact complete prediction bits.
"Transient" uploads the model every call; "resident" retains it but stages I/O
through Lists; "borrowed" retains it and borrows host I/O directly.

Initial single-call measurements were noisy and remain in `cuda/*-higgs.json`,
`et-year.json` and `et-covtype.json`. One bounded throughput follow-up used eight
consecutive public calls per timed block, eight blocks per arm, rotated order.
All eight outputs are retained and checked outside the timer; each cold first
call is recorded separately. Values below are median block time divided by eight,
**not a single-call latency certification**. Stability requires every compared
arm's maximum/minimum sample time to be at most 1.10. No outliers were removed.

| Workload | Transient ms/call | Resident ms/call | Borrowed ms/call | Qualification |
| --- | ---: | ---: | ---: | --- |
| [ET HIGGS](cuda/et-higgs-batched.json), 1M fit / 500k predict × 28, 2 outputs | 76.459 | 35.141 | 21.148 | All arms stable; 3.62× throughput versus transient |
| [ET Year](cuda/et-year-batched.json), 463,811 fit / 51,534 predict × 90, scalar | 61.561 | 7.231 | 3.866 | All arms stable; 15.92× throughput versus transient |
| [RF HIGGS](cuda/rf-higgs-batched.json), 1M fit / 500k predict × 28, 2 outputs | 112.944 | 35.131 | 21.548 | Transient and borrowed unstable; no qualified full ratio |
| [ET Covtype](cuda/et-covtype-batched.json), 522,911 fit / 58,101 predict × 54, 7 outputs | 32.948 | 6.473 | 3.269 | Borrowed unstable; no qualified full ratio |

RF's separately trained cuML 26.08 cached nvForest comparator measured 10.695
ms/call in the throughput follow-up and passed its stability gate. MojoLearn's
borrowed RF arm did not, so **RF/cuML parity is not established**. ET has no
claimed equivalent cuML comparator in this campaign. These are inference
measurements, not training speedups or measurements of identity overhead.

HIGGS preserves the original first 1M training rows and last 500k held-out rows;
the uploaded cache was cropped to those ranges and verified by raw array hashes.
Year and Covtype are complete real-data companion workloads, not substitutes for
the million-row HIGGS target. Scale reminders remain visible, including warnings
for smaller row counts. The four forests contain approximately 0.93M–4.02M nodes.

## Apple FAST public calls

[RF HIGGS](metal/rf-higgs-resident-borrowed.json) passed all bit checks, but its
transient/resident timing arms were unstable. No speed ratio is certified.

On [ET Year](metal/et-year-resident-borrowed.json), resident and borrowed arms
were both stable: median single-call time decreased from 43.817 to 42.464 ms
(about 3.1%). The transient arm was unstable, so the entire three-arm comparison
is not qualified. The Year loader initially failed because it imported pandas
on a cache hit; that failure is retained, and the import now occurs only when
CSV decoding is needed. No numerical or dataset-split logic changed.

## Isolated vector traversal

The synthetic resident-kernel fixture contains one million rows × 28 features
and 100 distinct complete depth-16 trees (13,107,100 nodes), with two or seven
outputs. Timers include enqueue/drain but exclude all uploads and readback.
Scalar/vector launches alternate within one process; every paired output bit is
checked. These figures isolate traversal reuse and are not public-call speedups.

| Device / mode / outputs | Scalar median ms | Vector median ms | Qualification |
| --- | ---: | ---: | --- |
| H100 IDENTICAL / 2 | 26.522 | 25.564 | Stable; 3.61% lower kernel time |
| H100 IDENTICAL / 7 | 71.141 | 37.999 | Stable; 46.59% lower kernel time |
| Apple M4 FAST / 2 | 1969.782 | 1537.635 | Stable; 21.94% lower kernel time |
| Apple M4 FAST / 7 | 4361.943 | 1925.879 | Scalar spread 1.10183; unqualified |

See [CUDA samples](cuda/kernel-summary.json), [Metal samples](metal/resident-fast-summary.json)
and the [vector implementation report](../../../docs/lanes/FOREST_VECTOR_GROVES.md).
Vector reuse is now the default for 2–8 outputs; `MOJOLEARN_FOREST_SCALAR_GROVES`
retains the reference for diagnosis. Outputs one and above eight use scalar
traversal. Default and force-scalar paths both have named checks.

## Correctness, provenance and limits

- [67 host tests](host-tests.log) passed, including cache invalidation, immutable
  snapshots, stale binding refusal, pickling and release on refit/destruction.
- [CUDA/Metal IDENTICAL](cross-vendor-identical.json) matched all 1,440 recorded
  output rows across scalar, vector and final-default routes. The fixtures cover
  tree tails, 1/2/3/7/8/9 outputs, signed zero, subnormals and cancellation.
- Native ownership checks cover borrowed/List/transient equality, real split
  addressing, stale handles, malformed graphs, non-finite input and output,
  cleanup and destruction. Public final-default checks passed all four
  estimators on [CUDA IDENTICAL](cuda/public-final-default-identical.run.log)
  and [Metal FAST](metal/public-final-default-fast.run.log), with graph oracle,
  repeat calls, pickle and versioned archive checks.
- The [artifact manifest](artifact-manifest.json) records final CUDA bindings,
  baseline artifacts and intermediate source overlays in the local evidence
  store outside this repository, following `tools/hooks/pre-commit`. Readable
  executed source snapshots are committed under `source_snapshots/`.
  The intermediate pre-default borrowed binary itself was not archived. JSON files carry the actual measured binary/source hashes.
  The pre-batching driver is also retained. CUDA build/timing scripts and logs
  are included; final builds use no vector-enable define.
- Metal binary/source hashes are in the resident provenance JSON files. Public
  FAST bindings were built with the then-required vector define; the final
  default selector was checked separately in native FAST/IDENTICAL builds.
- Build logs retain deprecation warnings and the initial native global-type
  compilation error. Setup logs retain unused image Torch dependency conflicts
  and tcmalloc NUMA-binding warnings. No causal timing claim is inferred from
  those warnings. GPU telemetry is retained; no clocks were changed.

HIP and broader large-model cross-vendor qualification remain pending. Next
performance work is device I/O allocation reuse, GPU-array input/output ownership
and measured nvForest-style packed node layout, retaining the same reduction
graph. RF/cuML needs a stable matched timing cell before a parity claim.

The dedicated H100 rental was deleted after all artifacts were retrieved and
hash-verified; API absence was verified. Estimated compute was **$1.68**, excluding
storage, not an invoice. The protected training pod was untouched. See
[rental metadata](rental.json). `SHA256SUMS` covers retained evidence files.
