# Reuse vector-leaf traversal in fixed 32-grove GPU inference

The vector kernel walks each row/tree pair once and updates all of that leaf's
outputs. The scalar-output reference walks the same tree separately for every
output. Both use the existing fixed 32-grove sum for each output: grove `g`
adds trees `g, g+32, ...`, followed by the halving steps `16, 8, 4, 2, 1` and
one final division. Reusing traversal does not change that addition graph.

The source is nvForest v26.08.00, commit
`cef3a50da0f74b0015876b9d6d424c86141898dc`,
`cpp/include/nvforest/detail/infer_kernel/gpu.cuh:139–158`: `evaluate_tree`
produces a leaf ID once, then the vector-output loop accumulates each class.
The local pinned checkout is `upstream/nvforest-v26.08.00` alongside this repo.
The earlier prototype already explicitly deviated from nvForest's variable
grove count and hardware shuffle by fixing 32 logical groves and a shared-memory
halving tree. This change retains those decisions; it is not a claim of
bit-identical nvForest output.

## Implementation and scope

`core/forest_inference.mojo:forest_vector_grove32_kernel` uses one logical
32-thread group per row, four rows per 128-thread block. Capacities 2, 4 and 8
cover 2–8 outputs. Per-output shared-memory planes preserve the existing sum
order; the maximum scratch is 4 KiB per block. A single output and more than
eight outputs retain the scalar-output kernel. No rows×trees temporary is
allocated. `launch_forest_inference` is the shared enqueue dispatcher used by
both the synchronous helper and the separate resident model owner.

Vector traversal is now the default for 2–8 outputs inside `parallel_groves`.
`-D MOJOLEARN_FOREST_SCALAR_GROVES=1` forces the scalar-output diagnostic
reference. No enable flag is required; the former vector-enable definition is
unnecessary and has no additional effect. The public GPU engine itself remains
opt-in, and `sequential` remains the existing public default. This kernel choice
is separate from retaining a GPU model snapshot between calls.

The comparison and arithmetic seams are shared with the reference: RF flushes
only input features in IDENTICAL; ET compares finite subnormal values by integer
keys, treating signed zeros as equal. IDENTICAL adds flush operands/results and
uses portable division. This candidate must match the scalar-grove reference's
bits; neither grove route promises equality with the legacy ordered tree sum.
The cancellation check deliberately proves that distinction.

## Local correctness and resident timing

Apple M4 runs passed FAST and IDENTICAL, each with a forced scalar build and a
vector-enabled build. In each mode, all **52 complete output fingerprints**
matched between builds: RF/ET, 1/31/32/33 trees, 1/2/3/7/8/9 outputs, row tails,
threshold equality, signed zeros, positive/negative subnormal input and threshold
comparisons, and four cancellation fixtures. Independent handcrafted leaf
selection and fixed-topology host oracles also passed. Outputs 1 and 9 explicitly
exercise fallback. [Correctness logs](../../bench/results/forest_groves_2026-09-10/metal/fast-vector.log)
and their scalar/IDENTICAL counterparts are retained in that directory.

`bench/speed/forest_grove_kernel.mojo` benchmarks both resident kernels in one
process, with alternating AB/BA pairs, two paired warmups and six measured samples
per arm. Every pair compares all output bits and verifies a repeated checksum.
The fixture has **1,000,000 rows × 28 features and 100 distinct complete depth-16
trees: 13,107,100 nodes**. Features, thresholds and leaf channels vary. This is
substantial synthetic model-memory pressure, not a model trained on real data.
A conservative host/device array estimate stays below 2 GiB for both cells.

| Apple M4 FAST resident kernel | 2 outputs | 7 outputs |
| --- | ---: | ---: |
| Scalar median | 1969.782 ms | 4361.943 ms |
| Vector median | 1537.635 ms | 1925.879 ms |
| Scalar max/min spread | 1.0283 | 1.10183 |
| Vector max/min spread | 1.0132 | 1.03562 |
| Observed scalar/vector ratio | 1.2810× | 2.2649× |
| Timing gate | Passed | **Failed: scalar spread >1.10** |

The two-output cell measured **21.94% lower median resident-kernel time**.
The seven-output ratio is unqualified despite its large observed difference;
it must not be promoted to a certified speedup. Both cells retained exact
scalar/vector output agreement across all eight pairs. The gate requires at
least six measured samples per arm and max/min sample spread at most 1.10.
[Raw samples and explicit qualification](../../bench/results/forest_groves_2026-09-10/metal/resident-fast-summary.json)
include `timing_stable` and a null `qualified_speedup` for seven outputs.

The timer includes kernel enqueue and context drain, while excluding fixture
creation, model/input upload, output readback and hashing. Setup/validation/upload
time is separately printed. These are resident synthetic kernel results, not
full public prediction latencies or a general RF/ET performance guarantee.
Build and benchmark locks serialized these runs against other local work.

## Reproduction and remaining qualification

```sh
tools/with_build_lock.sh pixi run mojo run -I . \
  checks/forest_inference_gpu.mojo
tools/with_build_lock.sh pixi run mojo run -I . \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  checks/forest_inference_gpu.mojo
# Repeat both with -D MOJOLEARN_FOREST_SCALAR_GROVES=1; compare output records.
tools/with_build_lock.sh pixi run mojo build -I . \
  bench/speed/forest_grove_kernel.mojo -o /tmp/forest-grove-kernel-fast
tools/with_build_lock.sh /tmp/forest-grove-kernel-fast 1000000 100 16 2 6
tools/with_build_lock.sh /tmp/forest-grove-kernel-fast 1000000 100 16 7 6
```

The resident benchmark explicitly launches both kernels regardless of the
default selector. Add `-D MOJOLEARN_NUMERIC_IDENTICAL=1` when building
its IDENTICAL executable. CUDA runs must use the existing device/toolchain's
correct target and retain compiled mode/vendor metadata. No script here provisions
hardware or extends leases.

## H100 IDENTICAL qualification and default selection

The same resident fixture and paired schedule passed on H100 CUDA in IDENTICAL.
Each arm retained six measured samples and matched every output bit in all eight
pairs, including warmups.

| H100 IDENTICAL resident kernel | 2 outputs | 7 outputs |
| --- | ---: | ---: |
| Scalar median | 26.522029 ms | 71.140734 ms |
| Vector median | 25.563924 ms | 37.998586 ms |
| Scalar max/min spread | 1.00252 | 1.00231 |
| Vector max/min spread | 1.01161 | 1.00426 |
| Scalar/vector ratio | 1.03748× | 1.87219× |
| Lower median kernel time | 3.61% | 46.59% |
| Timing gate | Passed | Passed |

[Two-output raw log](../../bench/results/forest_groves_2026-09-10/cuda/kernel-1m-depth16-output2.log)
and [seven-output raw log](../../bench/results/forest_groves_2026-09-10/cuda/kernel-1m-depth16-output7.log)
retain all samples and checksums. The speed differences compare scalar and
vector traversal **within IDENTICAL**; they are not acceleration caused by
selecting IDENTICAL mode.

All [1,440 handcrafted IDENTICAL output-bit records](../../bench/results/forest_groves_2026-09-10/cross-vendor-identical.json)
match across CUDA scalar/vector and Metal vector routes. The
[public CUDA check](../../bench/results/forest_groves_2026-09-10/cuda/public-identical.run.log)
also passed all four RF/ET classifier/regressor entrypoints, graph-oracle checks,
resident reuse/release, pickle and versioned archives. These checks do not imply
sequential-sum bit identity.

This measured improvement with preserved grove arithmetic supports default
vector reuse for the bounded 2–8-output specialization. One output and more than
eight retain scalar traversal, and force-scalar remains available for diagnosis.
The unstable seven-output Metal timing is retained as unqualified; the CUDA
result does not retroactively qualify it.

The public engine stays experimental and opt-in. The kernel numbers above
exclude transfers and ownership costs. A separate [public-call campaign](../../bench/results/forest_groves_2026-09-10/README.md)
validated ET HIGGS/Year throughput in eight-call blocks; RF and several
single-call cells remain noisy. Those results concern model residency and
borrowed buffers, not an isolated vector-kernel speed ratio. HIP and broader
large-model cross-vendor coverage remain open.


After promotion, fresh local FAST and IDENTICAL builds each passed the default
vector selector against force-scalar: all 1,440 output-bit records and 52
fingerprints matched per mode. The selector readout confirmed vector for
2/3/7/8 outputs and scalar fallback for 1/9; force-scalar disabled vector for
all tested widths. [Final default check](../../bench/results/forest_groves_2026-09-10/metal/identical-default-final.log)
and corresponding force-scalar/FAST logs retain the exact commands and exits.
