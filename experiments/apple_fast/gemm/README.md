# Apple GEMM throughput experiments — NOT COMPILED / NOT RUN

These are source-only hypotheses. No correctness execution, compiler invocation,
benchmark, throughput measurement, or peak comparison was performed. No speedup
or correctness certification is claimed. Production remains unchanged unless the
explicit master flag is defined on an Apple FAST build.

## Source findings

`core/gemm.mojo` normally delegates FAST NT products to MAX `linalg.matmul`.
The locally available Modular checkout's
`max/kernels/src/linalg/matmul/gpu/apple/matmul_8x8.mojo` uses 64x64 output
blocks, four simdgroups, 8x8 FP32 MMA fragments, and direct global operand
loads. Its `BLOCK_K` parameter does not control staging: there is no shared
operand tile in that kernel. Merely changing that parameter would not test
threadgroup reuse. This lane therefore implements an independent kernel using
the same intrinsic and fragment ABI, with actual cooperative staging and
barriers. The checkout was read as API reference; its file was not copied.
The installed SDK may differ: private intrinsic compatibility is unverified.

## Activation and arms

The master compile definition is `MOJOLEARN_APPLE_GEMM_EXPERIMENT`.
Definitions use presence semantics (`is_defined`), so omit them to disable;
setting a definition to zero still enables it. No command was executed.
The master flag hooks `core.gemm.gemm_nt` and `gemm_nt_gram` only on Apple
with `NUMERIC_FAST`. IDENTICAL and other devices keep their existing routes.
`n == 1` retains the existing GEMV route.

All named arms below require the master flag. Modifier names are prefixed
`MOJOLEARN_APPLE_GEMM_`.

| Arm | Modifiers | Block M×N×K | Shared bytes | Question |
|---|---|---:|---:|---|
| G1 direct control | DIRECT | 64×64×16 | 4 | Independent MMA baseline against SDK dispatch |
| G2 small direct | DIRECT, SMALL | 32×32×16 | 4 | Lower register pressure vs fewer products per operand |
| G3 wide direct | DIRECT, WIDE | 64×128×16 | 4 | More N reuse vs doubled accumulators |
| G4 tall direct | DIRECT, TALL | 128×64×16 | 4 | More M reuse for opposite aspect ratios |
| G5 shared | none | 64×64×16 | 8,192 | Reuse loaded panels across the four simdgroups |
| G6 deep shared | DEEP | 64×64×32 | 16,384 | Amortize two barriers over twice as many K steps |
| G7 padded shared | PADDED | 64×64×16 | 10,240 | Alter shared row-bank stride for transposed fragments |
| G8 wide shared | WIDE, DEEP | 64×128×32 | 24,576 | Share larger B panels within the 32 KiB budget |
| G9 tall shared | TALL, DEEP | 128×64×32 | 24,576 | Share larger A panels within the 32 KiB budget |
| G10 padded deep | DEEP, PADDED | 64×64×32 | 18,432 | Combine barrier amortization and stride padding |

Shared byte counts are algebraic allocation sizes, not measurements. WIDE,
TALL and SMALL are mutually exclusive. Other modifier combinations are allowed
and statically limited to 32 KiB. PADDED has no effect on direct arms. DEEP
on direct arms only groups iterations; it does not prefetch or double-buffer.

For standalone resident buffers, call
`apple_gemm_experiment[BM, BN, BK, STAGED, PAD, TRANSPOSE_B]` from `mma.mojo`.
That entry also supports NN (`TRANSPOSE_B=False`) with coalesced B loading
followed by transposed shared writes. It requires Apple FAST and FP32 storage;
no FP16/BF16 conversion or lossy operand compression occurs. There is one
launch, no buffer creation and no synchronization. Output cannot overlap
inputs; the inputs can alias one another for Gram products. Caller owns all
buffer lifetimes. Extents and capacities are checked before enqueue.

## Static design review and limitations

Each simdgroup owns a disjoint rectangle; each lane writes two columns per
8x8 output fragment. Shared pages contain A and physically transposed B, so
both NT and NN feed the same fragment mapping. All 128 threads participate
in both staging barriers, including edge blocks. M/N edges and K tails are
zero-filled; K=0 writes zeros. Padded K does extra zero MMAs, which is allowed
only as a FAST experiment and is not a promise of signed-zero or NaN identity.
No split-K or atomics changes the reduction schedule. Accumulation uses the
Apple FP32 MMA intrinsic, whose rounding is not the IDENTICAL scalar contract.

Only call sites reaching `core.gemm` are affected. In particular,
the source contains direct NT calls in `decomposition/estimator.mojo` (PCA
transform/inverse transform), `decomposition/impl/linalg/detail/pca.mojo`,
`kernel_methods/checks/kernel_matrix.mojo`, and
`neighbors/impl/detail/knn_brute_force.mojo`. These are source reachability
examples, not evidence that every estimator's selected shape takes that arm.
`cholesky/checks/potrf.mojo` normally calls `identical_gemm_into`; its
`CHOL_SAB_VENDOR_MATMUL` comparison arm reaches `gemm_nt`. This lane does not
activate sabotage flags or silently replace that profile. PCA's small Gram
shapes may be handled by `gram_splitk` before this hook. Other LU, LLE, KPCA,
MinCovDet, SVGP and randomized-SVD paths must be traced at their actual shapes
before attributing a whole-row gain to these experiments.

## Deferred validation, only after new authorization

First compile each isolated arm and check the installed intrinsic ABI. Compare
resident NN/NT outputs with an independent FP64 oracle on rectangular, small,
ragged, K=0, K-tail and aliased-input Gram cases; include cancellation-heavy
and large-dynamic-range inputs. Verify unchanged IDENTICAL dispatch separately.
Then isolate GEMM from packing, allocation and panel solves at representative
shapes: large square LU trailing updates, narrow K panels, Gram matrices, and
SVGP rectangles. Only then collect standalone device time and compute
`2*m*n*k / seconds`; report which FP32/MMA peak definition and exact M3 Ultra
configuration are used. Inspect register spills, occupancy, barriers and shared
traffic before selecting a winning tile. Algorithm residuals and forecast or
model quality remain separate gates. None of these deferred steps ran here.
