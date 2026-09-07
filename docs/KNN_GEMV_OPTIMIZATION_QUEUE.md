# kNN and GEMV optimization queue

Status: source/evidence review, not a new qualification. Root alone runs tests,
builds, measurements, model code and rentals. Subagents must never run them.
No Apple testing is requested. Preserve the published ratios and all retained
historical artifacts. Do not promote a candidate from this document alone.

## What the evidence establishes

The paper's approximately 25x kNN and 21x GEMV AMD ratios compare existing
IDENTICAL and FAST implementations. They do not isolate an unavoidable cost
of the identity contract. kNN's pinned distance calculation lacks the operand
reuse of a fast matrix-product implementation. That source observation does
not establish that operand reuse explains GEMV's gap, or how much of either
end-to-end gap each mechanism contributes.

The [transposed index candidate](../neighbors/checks/transposed_index_distance_candidate.mojo)
keeps each distance's ascending feature FMA/FTZ chain, original row-major norm
calculation, clamp and rooted-distance operation. Its changed index layout
makes adjacent lanes' index loads contiguous. This is coalescing; it does not
introduce a tiled matrix-product reuse scheme. The added transpose and scratch
must be counted. Small-k selection is a separate change, requiring separate
attribution through the existing four-arm experiment.

The [four-arm record](../bench/results/e1g/2026-09-05_163958-nvidia-mamba/knn-layout-summary.json)
retains baseline, selector, transpose and combined activation witnesses,
complete output checks and request timings. The
[experiment notes](../neighbors/checks/SMALLK_DISPATCH_EXPERIMENT.md) report
NVIDIA RTX 4090's 1,000-query median changing from 17.241 ms selector-only to
9.273 ms combined. The later
[AMD campaign](../bench/results/e1/2026-09-05_215006-mojolearn-e2-amd/README.md)
and [cross-vendor comparison](../bench/results/resume/2026-09-05-next-certification/amd-nvidia-knn-layout.json)
retain matching output bits; the latter reports `cross_vendor_bits: PASS` and
explicitly does not perform a cross-vendor speed comparison. These are bounded
fixture results, not coverage of arbitrary data distributions or current wheels.

The earlier [public-request selector record](../bench/results/e1g/2026-09-05_103918-nvidia-mamba/knn-public-price-summary.json)
is a different experiment. Its speed improvements must not be attributed to
the subsequently introduced transpose. Existing Apple records may be cited as
historical evidence, but this queue authorizes no new Apple runs.

The [GEMV candidate](../core/gemv_serial_layout_candidate.mojo) transposes the
matrix from `[M,K]` to `[K,M]` while retaining each output's ascending FMA/FTZ
chain and final `+0` operation. It adds `M*K` scratch floats and a transpose;
the serial dependency remains. Its header still calls it unqualified, but a
retained [NVIDIA candidate log](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/gemv-candidate.log)
does end in `GEMV SERIAL LAYOUT QUALIFICATION PASS` and contains separate
legacy, transpose-plus-product and prepared-product samples. That supports
the named harness run, not production activation, a full causal decomposition,
or current-source cross-vendor qualification. Preserve its source/status/binary
provenance when using it; do not treat a log footer alone as a new certificate.

## Missing integration and next source changes

- [ ] kNN: inspect existing opt-in dispatch in
  [knn_brute_force.mojo](../neighbors/impl/neighbors/detail/knn_brute_force.mojo).
  Integration already exists behind `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL`
  and `MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL`; do not add a duplicate
  dispatch. Missing work is bounded shape/data qualification, installed-binary
  activation witnesses and a justified default policy. Keep selector-only,
  transpose-only and combined routes separately observable.
- [ ] Keep original norms, per-distance FP32 operation order, tie/composite-key
  selection and output offsets unchanged. First extend host dispatch metadata
  with selected flags, query tile, metric and transpose/preparation scope.
  Preserve unsupported-k/metric fallbacks and all FAST routes.
- [ ] GEMV: add an explicit opt-in route to the existing layout helper at the
  actual pinned GEMV entry, plus a route witness. No production dispatcher
  currently calls `serial_layout_gemv_*`. Start with the full single-request
  transpose-plus-product path. A prepared-matrix API needs explicit ownership,
  invalidation and unchanged-input semantics before pricing reuse.
- [ ] Do not replace a serial FP32 chain with warp reductions, reassociation,
  vendor GEMM or altered FTZ seams while claiming the same profile. First
  optimize address layout and launch/workspace policy without changing math.

## Root-only NVIDIA comparison protocol

Use one external peer per lane: cuML brute-force kNN, and one explicit CUDA BLAS
FP32 GEMV arm. Compare our FAST and IDENTICAL paths against that same peer on
the same rented NVIDIA GPU. Existing
[public comparison code](../tools/nvidia_public_compare.py) supplies useful
host-request structure, but its GEMV peer is currently PyTorch `A @ b.T`,
labelled `torch-cublas-fp32-host-request`; it does not prove `cublasSgemv` was
called. For an explicitly named BLAS GEMV result, add and witness that call,
including the row-major/column-major transpose mapping, alpha=1 and beta=0.

Record before execution: frozen source inventory/commit and binary hashes;
GPU UUID/model/driver; runtime/peer versions and loaded CUDA libraries; native
vendor/mode/dispatch witnesses; exact FP32 input files and hashes; dimensions,
strides, metric, k, tie policy and output dtypes; math/precision policy including
TF32 exclusion; warmup and rotating paired-round schedule; deadline, peak
workspace allowance and synchronization boundary; predeclared numerical
tolerances and refusal rules. Retain every output and raw timing sample.

For kNN use identical index/query bytes and Euclidean brute-force distances.
State whether fit/index upload is included on every request. For GEMV use the
same matrix/vector bytes and output shape. Report host upload-through-download
timing separately from any prepared/device-resident scope. Count transpose,
scratch allocation and preparation consistently. Existing
[public GEMV results](../bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/campaign/public-gemv/results.json)
are historical host-array measurements, not kernel-only or candidate timings.

The two candidate kernels launch GPU work in source; neither is a CPU fallback.
That does not prove an installed public wrapper selects them. Refuse a timing
without the actual loaded binding/vendor/mode and candidate-route witnesses.
External tolerance agreement is separate from our bitwise identity claim.

## Acceptance before promotion

- [ ] Root runs bounded adversarial gates before timing: cancellation,
  subnormals/FTZ, signed zero, duplicate/equidistant neighbors, rooted/unrooted
  metrics, ragged dimensions, tile-boundary counts and nonzero output offsets.
  Reuse [kNN layout checks](../bench/knn_layout_dispatch_check.mojo),
  [adversarial checks](../bench/knn_layout_adversarial_check.mojo) and
  [GEMV layout checks](../bench/gemv_serial_layout_main.mojo).
- [ ] Compare every FP32 output bit and kNN index against baseline, across
  repeat runs, launch/query-tile changes and separate NVIDIA/AMD columns built
  from the same frozen numerical sources. Preserve input bytes and every
  output, including intermediate distance checks where applicable.
- [ ] Prove the candidate route was reached and sabotage changed arithmetic
  or ordering is rejected. Do not accept aggregate hashes without matching
  shape/record counts and complete success status.
- [ ] Run installed public-surface checks and the bounded NVIDIA peer protocol.
  Keep release promotion separate until root reviews identity, regression,
  workspace and performance evidence. Use the serial vendor guards, two CPU
  threads, explicit RSS/VRAM/deadline bounds and no overlapping GPU jobs.
