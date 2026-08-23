# svm: C-SVC (binary, dense FP32), from cuML's SMO solver and cuVS's kernel matrices

Rung 1. **COPY, DO NOT IMPROVE.** One estimator, the binary `SVC`
(`svcFit` / `svcPredict` of `cpp/src/svm/svc_impl.cuh`): the two-level
decomposition SMO of `smosolver.cuh`, the block solver of
`smoblocksolve.cuh`, the working-set selection of `workingset.cuh`, the
results extraction of `results.cuh`, and the LINEAR and RBF kernel matrices
of cuVS `distance/detail/kernels/kernel_matrices.cu`. DEVIATIONS 630-649
are this lane's.

## Status

**CONSTRUCTION plus one Apple device's gates; no second vendor has run
this.** Rung 1 is complete and gated on one M4, 2026-08-23: binary C-SVC,
dense FP32, LINEAR and RBF, `C`, `gamma`, `tol`, `max_iter`,
`max_outer_iter`, `nochange_steps` (their convergence rule transcribed),
`predict` and `decision_function`, support vectors + dual coefficients +
intercept out. 16 of 16 gates pass under `NUMERIC_IDENTICAL` (device ==
host oracle BITWISE on all seven fixtures: working-set sequence, alpha and
`f` after every outer iteration, `b`, dual coefs, support indices, decision
function on 277 queries; launch-invariant over 7 arms x 3 fixtures), 16 of
16 under `NUMERIC_FAST` where the device-vs-oracle and launch-invariance
gates are REPORTS (see "Commands and outputs" at the foot). No timing was
measured and none is published.

REFUSED BY NAME (each raises with the parameter's name;
`check_refusals` gates all of them): `svmType != C_SVC` (SVR is rung 2),
`epsilon != 0`, `cache_size != 0` (the LRU cache; see "the cache
decision"), kernels `POLYNOMIAL` / `TANH` / `PRECOMPUTED`, `sample_weight`
(and class_weight, which is sample_weight upstream), more than two classes,
sparse input. `probability` is not in their C++ surface. Full table:
`svm/UNPORTED.tsv`.

Files: `svm/ported/svm/{svm_parameter,smo_sets,ws_util,workingset,
kernelcache,smoblocksolve,smosolver,results,svc_impl}.mojo`,
`svm/ported/distance/kernel_matrices.mojo`, `svm/mojo_only/
{device_select,pinned_argreduce,smo_oracle,svc_check}.mojo`,
`svm/svc_main.mojo`, `svm/PORTED_MAP.tsv`, `svm/UNPORTED.tsv`.

## THE IDENTITY CONTENT, written before the code

The claim rung 1 is built to support: **for the same `(X, y, params)` bits,
the sequence of working sets, the alphas, the `f` vector after every outer
iteration, `b`, the dual coefficients, the support indices and the decision
function are a pure function of the input bits, and launch-invariant.**
Under `NUMERIC_IDENTICAL`. Under `NUMERIC_FAST` the same gates are REPORTS.

Their solver has five places where a vendor, a library fold shape, a block
count or an unstated tie-break can reach the bits. Each one, and what ours
does:

### 1. The kernel-matrix rows (cuVS `GramMatrixBase::linear`, `rbf_kernel_expanded`, `matrixRowNormL2`)

Theirs: LINEAR is a cuBLAS GEMM. RBF is the SAME GEMM followed by the
expansion `exp(-1.0 * gain * (norm_x[i] + norm_y[j] - dot * 2))`
(`kernel_matrices.cu`, `rbf_kernel_expanded`), where the norms are
`raft::linalg::rowNorm<L2Norm>` (the SQUARED norm, a block fold) and, for
`math_t = float`, the `-1.0 * gain` literal promotes the argument to DOUBLE
and the `exp` is the double one, rounded back to float on store. There is no
clamp at zero: the cancellation `||x||^2 + ||y||^2 - 2 x.y` can go negative
and `K(x, x)` can come out a hair above 1.

Ours, ONE spelling, its rounding sequence stated:

    dot     = GEMM (OP_NT)      FAST: core/gemm.mojo::gemm_nt (MAX matmul, the
                                cuBLAS stand-in); IDENTICAL: gemm/mojo_only/
                                gemm_identical.mojo::identical_gemm, profile
                                mojolearn.identical.gemm.fp32.v1 (serial chain
                                for k <= 128, leaves + the fixed balanced tree
                                above that; gemm/IDENTICAL_FP32_CONTRACT.md)
    norm_i  = per-row SERIAL ascending chain  acc = ftz(fma(x, x, acc))
                                (one thread per row, both modes; theirs is a
                                block fold whose shape is cub's)
    linear  K = dot
    rbf     s = ftz( ftz(norm_x + norm_y) - ftz(2 * dot) )     <- THEIR expansion,
            e = ftz( (-gamma) * s )                               same association
            K = ftz( identical_exp(e) )                           (DEVIATION 630)

DEVIATION 630: the RBF exponential is taken in FLOAT32 through
`identical_exp` (`portable_expf`), not in double. There is no float64 on the
Apple GPU (memory: `mojolearn-hardware-limits`), so their double `exp` cannot
be mirrored on this column at all; and the device `exp` differs per vendor
(IDENTITY_PATHS row 12). The measurement is `check_rbf_float_vs_double_reference`
in `svc_check.mojo`: the max ULP distance of our K against the host Float64
reference of THEIR spelling. The pinned-tile spelling
(`neighbors/mojo_only/pinned_distance_tile.mojo`, `fma(-2, dot, nx+ny)`
clamped at zero) is NOT used: it is a different association and a clamp
they do not have, and mirroring their rounding sequence is the point.

### 2. Working-set selection (`workingset.cuh::SimpleSelect`, `GatherAvailable`)

Theirs: `cub::DeviceRadixSort::SortPairs(f, f_idx)` over all `n_train` with
`f_idx = 0..n-1`, then `set_upper`/`set_lower` flags, a permutation of the
flags into sorted order, `cub::DeviceSelect::Flagged` (order-preserving
compaction), and a copy of the FRONT `n_needed/2` (upper set, smallest `f`)
then the BACK `n_ws - n_selected` (lower set, largest `f`), then a fill from
whatever is left. `cub`'s radix sort is STABLE and its float key transform
is the monotone bit twiddle, so their order is in fact `(f twiddled bits,
then index)`; it is a property of the library, documented nowhere in their
file. Duplicate training rows produce exactly equal `f`, so the tie order IS
which rows enter the working set.

Ours: the same sequence, with the total order SPELLED: keys are the
twiddled `f` bits (`bits & 0x80000000 ? ~bits : bits | 0x80000000`), values
are the indices, the sort is the repository's stable one-bit-per-pass LSD
radix (`gbdt/gpu_util/kernel/radix_sort.mojo::launch_radix_sort_bins`, READ
ONLY, imported), and the compaction is a device scan + scatter. **DEVIATION
631**: the selection is a pure function of `(f bits, index)` and nothing
else; gated by the duplicated-rows fixture (`check_ws_sequence_is_pure_in_f_and_index`)
against the host oracle's `(twiddle(f) << 32 | idx)` integer sort, and by
the sabotage that reverses the index half of the key (must FAIL on that
fixture).

### 3. The block solve (`smoblocksolve.cuh::SmoBlockSolve`)

Theirs: one block of `n_ws` threads; each inner iteration does
`cub::BlockReduce<KVPair>::Reduce(pair, cuda::minimum{})` for `u`, a float
`Reduce(max)` for `f_max`, and a `KVPair` `Reduce(maximum)` for `l`.
`KVPair::operator<` compares `val` ONLY (`kselection.cuh:69`, "///@todo:
should we also consider the key when values are the same?"), so on equal
`f` the winner is whichever pair cub's fold shape happens to keep, a
function of the block reduction topology and therefore of the vendor. The
per-thread update `f += q * (Kui - Kli)` is a multiply-add the CUDA compiler
contracts.

Ours: the three reductions are halving trees through threadgroup memory
with NO lane primitive (the shape of `core/pinned_reduce.mojo`), and the two
arg-reductions carry an explicit tie-break: **DEVIATION 633**, equal values
resolve to the LOWER TRAINING INDEX (`ws_idx[tid]`, not `tid`), so the
winner is a pure function of the working SET and not of the order the
cache permuted it into. `f += q*(Kui - Kli)` is `identical_mul_add(q,
ftz(Kui - Kli), f)`; `(f_u - f)^2 / eta` is `ftz(ftz(d*d) / eta)` with `d =
ftz(f_u - f)`; `eta = max(ftz(ftz(Kd_i + Kd_u) - ftz(2*Kui)), 1e-12)`. The
block size is a parameter (1024 as theirs, or the smallest power of two
`>= n_ws`); a selection over a total order cannot move with it, and the
launch-invariance gate runs both.

### 4. The f update (`smosolver.cuh::UpdateF`, `cublasgemv`)

Theirs: `f = 1 * K[n_rows x n_nz] . delta_alpha + 1 * f` through cuBLAS,
whose fold over the working-set axis is its own business, over the
nonzero-delta columns in the order the KERNEL CACHE left them.

Ours: **DEVIATION 634**: a hand GEMV, one thread per training row, `acc =
ftz(fma(K[j, i], da_j, acc))` over the nonzero deltas in ASCENDING TRAINING
INDEX (a host-computed rank permutation of the `<= 1024` nonzero indices,
uploaded per outer iteration), then `f_i = ftz(f_i + acc)`, which is
`alpha * acc + beta * y` at `alpha = beta = 1`. Ascending training index
rather than working-set position so that the fold order, like the
tie-break above, cannot see the cache permutation: with both pinned this
way, NOTHING numeric in the solver depends on `cache_size`, and a future
port of the LRU cache is bit-inert by construction. Gated by the
rotate-by-block sabotage (start the fold at `block_idx % nnz`; must FAIL).

### 5. The intercept (`results.cuh::CalcB`)

Theirs: `b = -cub::DeviceReduce::Sum(f over free SVs) / n_free`, a library
fold; or `-Sum(f) / n_train` when there is no support vector; or `-(min f
over upper + max f over lower) / 2` when every SV is bound (two selections,
exact).

Ours: **DEVIATION 632**: the sum is a SERIAL ASCENDING chain over the
order-preserving compaction of the free SVs (one thread, `acc = ftz(acc +
f_i)`), then `ftz(-acc / Float32(n_free))`. The selections are halving
trees. Gated bitwise against the oracle.

### What is NOT numeric and is left free

Grid shapes, the threads per block of every elementwise kernel, the batch
size of the full-tile computation (`kernel_tile_byte_limit`), the predict
batch (`buffer_size`), buffer padding and poison. The launch-invariance gate
moves each of them and asserts the bytes do not.

### The cache decision (`kernelcache.cuh`, `raft/util/cache.cuh`)

Their `KernelCache` wraps a 32-way set-associative LRU (`raft::cache::Cache`,
`hash = key % n_cache_sets`, time stamps from a call counter, eviction rank
by `cub::BlockRadixSort` over the stamps), and it REORDERS the working set
(`PreparePartitionedIdxOrder`: cached keys first in working-set order, then
uncached keys sorted by cache set with `cub::DevicePartition::Flagged`'s
reversed tail as the tie order). **The eviction order and the permutation
are a pure function of the sequence of working sets and of `cache_size`
(MiB) and `n_rows`**: every input to them is a counter, a modulus, or a
stable sort. They depend on no block count, no clock and no vendor. Under
DEVIATIONS 633-634 neither the permutation nor which columns come from the
cache can reach a bit (a cached row equals a recomputed row bit for bit
under IDENTICAL because every cell is a pure function of its pair).

Rung 1 PORTS THE STRUCTURE (`KernelCache` with `InitWorkingSet`,
`getSquareTileWithoutCaching`, `InitFullTileBatching`,
`getNextBatchKernel`, the `n_rows` batching) and takes THEIR `cache_size ==
0` path exactly (`raft::cache::Cache` with `n_cache_sets = 0`: every
`if (batch_cache.GetSize() > 0)` is skipped and `ws_idx_mod == ws_idx`).
`cache_size > 0` is REFUSED BY NAME and recorded in `UNPORTED.tsv`; it is
not a silent substitute.

## Commands and outputs (2026-08-23, one M4)

    pixi run mojo run -I . svm/svc_main.mojo -- oracle                # host only
    tools/with_build_lock.sh     pixi run mojo run -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/svm.card \
        tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo -- card
    python3 tools/identity_trace_diff.py /tmp/svm_mac.card /tmp/svm_other.card

Sabotage arms (each must behave as the table below says):

    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_WS_TIE=1      -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_FOLD_ROTATE=1 -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_STD_EXP=1     -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_NO_FTZ=1      -I . svm/svc_main.mojo

The fixtures (all hashed, 13-bit significands; `svc_check.mojo` header):
F1 blobs n=300 k=8 linear; F2 xor n=240 k=2 rbf gamma=0.5 C=10; F3 dup
n=1280 k=8 linear, 640 rows EACH TWICE (equal `f` forever, and n > 1024 so
the working-set sort and FIFO run); F4/F5 = F2 at C=0.05 / C=100; F6 big
n=1500 k=8 linear overlapping (266 support vectors, selection + FIFO); F7
wide_k n=96 k=200 (the v1 GEMM leaf + tree path).

IDENTICAL, head of the output (the full run prints every fixture's
`n_support`, `b` as decimal/hex, KKT gap and accuracy):

    == svm/svc_main.mojo [IDENTICAL] all ==
    CONSTRUCTION plus one Apple device's gates; no second vendor has run this.
      F1.blobs [IDENTICAL] oracle f32: n_support=5 outer=4 inner=11 b=0.09479656/0x3dc224b3 kkt_gap=0.00087659806/0x3a65cb80 acc=1.0
    PASS oracle_kkt_and_accuracy F1.blobs ... F7.wide_k            (7 gates)
    PASS oracle_objective_decreases F1.blobs, F2.xor                (worst relative rise 0.0)
      F1.blobs f32 vs f64 reference: n_sv 5 vs 5 (only32=0, only64=0) b 0.09479656 vs 0.09453905 max|decision diff|=0.00143
    PASS oracle_f32_matches_f64_reference F1.blobs
      DEVIATION 630 measurement: rbf float32 vs their double-exp spelling: cells=57600 differing=5294 max_ulp=1
    PASS rbf_float_vs_double_reference (DEVIATION 630)
      refusals by name: 8 (cache_size, svmType, epsilon, POLYNOMIAL, TANH, PRECOMPUTED, sample_weight, multiclass)
    PASS refusals_by_name
      F1.blobs [IDENTICAL] device vs oracle: IDENTICAL  outer=4 n_support=5 b=0.09479656/0x3dc224b3
      F2.xor   [IDENTICAL] device vs oracle: IDENTICAL  outer=5 n_support=16 b=-0.006553972/0xbbd6c2b4
      F3.dup   [IDENTICAL] device vs oracle: IDENTICAL  outer=6 n_support=54 b=0.112782255/0x3de6fa62
      F4.c_small [IDENTICAL] device vs oracle: IDENTICAL  outer=5 n_support=196 b=0.04327399/0x3d314011
      F5.c_large [IDENTICAL] device vs oracle: IDENTICAL  outer=4 n_support=7 b=-0.007706914/0xbbfc8a48
      F6.big   [IDENTICAL] device vs oracle: IDENTICAL  outer=6 n_support=266 b=0.017735064/0x3c914920
      F7.wide_k [IDENTICAL] device vs oracle: IDENTICAL  outer=5 n_support=36 b=0.012158405/0x3c47340c
    PASS device_matches_oracle (7 fixtures)
      F3.dup ws sequence: 6 outer iterations, first divergence at none; equal-f duplicate pairs in final f: 640 of 640
    PASS ws_sequence_is_pure_in_f_and_index (DEVIATION 631, F3.dup)
      F2.xor / F6.big / F7.wide_k [IDENTICAL] launch invariance over 7 arms: INVARIANT
    PASS device_is_launch_invariant (F2, F6, F7 x 7 arms)
      card: two traced fits (different launches) first divergence: none
    PASS card_is_emitted
    == svm/svc_main.mojo [IDENTICAL] 16/16 gates passed ==

The 7 launch arms: block-solve block size 1024 (theirs) vs the smallest
power of two >= n_ws; full-tile batches of n_rows vs 256 vs 100 rows
(`kernel_tile_byte_limit`); predict batch of n_rows vs 1 (`buffer_size`
0.001 MiB); scratch padding 0 / 37 / 1029 floats with poisons -7.25e20 /
3.0e-39; and all of them at once. Compared: every per-iteration alpha
hash, f hash and working set, `b`, dual coefs, support indices, decision
function and class on 277 queries.

FAST, the same binary without the define (the differences are the FAST
arms, as designed): the oracle gates pass identically; device-vs-oracle is
a REPORT -- F1, F3, F6 (linear, k=8) happen to agree with the oracle
bitwise on this column, F2/F4/F5 (RBF: the Metal `exp` is not the host
`exp`) and F7 (k=200: the vendor matmul's fold) diverge from
`iter000`; launch invariance is a REPORT -- F6 moves at the predict batch
(the vendor matmul routes `n == 1` to a GEMV); 16/16 with those two
reported. The two cards (IDENTICAL vs FAST, F2) diverge FIRST at
`svm.iter000.alpha` -- the instrument has an address.

## Sabotage table (all under IDENTICAL; restored after each)

| sabotage (define) | what it breaks | predicted | observed |
|---|---|---|---|
| `SAB_WS_TIE` (`workingset.mojo`): the (key, index) pairs reach the stable sort in reversed position, so equal f sorts by DESCENDING index | DEVIATION 631 | FAIL on F3.dup only; F1/F2/F4/F5/F7 have n <= n_ws (no selection), F6 has no equal f | `F3.dup [IDENTICAL] device vs oracle: DIVERGES (28: outer iteration count device 5 vs oracle 6)`, `FAIL ws_sequence_is_pure_in_f_and_index ... diverges from the (f bits, index) order at iteration 0`; every other fixture IDENTICAL; 14/16 |
| `SAB_FOLD_ROTATE` (`smosolver.mojo`): the f-update fold starts at `block_idx % nnz` | DEVIATION 634 | FAIL device-vs-oracle from the first `f`; FAIL launch invariance on the batch arms | `F1.blobs [IDENTICAL] device vs oracle: DIVERGES (10: f hash differs at outer iteration 0)`; `F6.big launch invariance ... MOVES: batch256:f hash at outer iteration 0 batch100:... all:...`; 14/16 |
| `SAB_STD_EXP` (`kernel_matrices.mojo`): stdlib `exp` instead of `identical_exp` in the RBF arm | DEVIATION 630 / row 12 | FAIL on the three RBF fixtures only | `F2.xor [IDENTICAL] device vs oracle: DIVERGES (21: alpha hash differs at outer iteration 0)`; F1/F3/F6/F7 IDENTICAL; 15/16 |
| `SAB_NO_FTZ` (`smosolver.mojo`): the f update stores without `ftz` | row 10 | NO failure on Apple: the hardware flushes, so the software flush is bit-inert here; a failure is only visible on a denormal-honoring column | 16/16, REPORTED: `F1..F7 device vs oracle: IDENTICAL`. This is the expected Apple result and is NOT evidence the seam is unreached; the seam's value is on CUDA/HIP |

## ROW TEXT FOR THE IDENTITY LANE

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| 40 | `svm/` C-SVC (`svcFit`/`svcPredict`): SMO decomposition, block solve, working-set selection, kernel matrices, results | (1) kernel rows: cuBLAS GEMM + `rowNorm` block fold + RBF `exp` promoted to DOUBLE for float; (2) working-set order from cub radix-sort STABILITY, unstated; (3) `cub::BlockReduce<KVPair>` compares `val` only -- the winner among equal f is the fold shape's (their own `@todo`); (4) `UpdateF` is `cublasgemv`, fold unspecified, over cache-permuted columns; (5) `CalcB`'s mean is `cub::DeviceReduce::Sum` | (1) v1 identical GEMM + per-row serial norm + `identical_exp` in float32 (DEVIATION 630, measured 1 ulp from their double spelling); (2) total order `(twiddled f bits, index)` spelled, stable one-bit radix + device scan compaction (DEVIATION 631); (3) halving-tree argmin/argmax, ties to the smaller TRAINING INDEX (DEVIATION 633); (4) hand GEMV, fold in ascending training index (DEVIATION 634), so nothing numeric sees the cache permutation; (5) serial ascending chain (DEVIATION 632). Host oracle in Float32 through the same helpers; device == oracle bitwise on 7 fixtures; launch-invariant over 7 arms; 4 sabotages recorded | Apple M4 only, 16/16 IDENTICAL; cache_size > 0 refused (LRU unported, proven bit-inert by construction); no second vendor |

## HAND-OFF TO THE IDENTITY LANE

- IDENTITY_PATHS.md: append the row above (row number is the ledger's to
  assign; 40 is a guess at the next free one).
- `mojo_only/numerics.mojo`: no change needed for rung 1. A `TANH` kernel
  port would need an `identical_tanh`; not requested.
- `core/pinned_reduce.mojo`: `svm/mojo_only/pinned_argreduce.mojo` is the
  argmin/argmax-with-index sibling of `pinned_block_max/min` (same halving
  shape, explicit key tie-break). If the identity lane wants ONE home for
  it, move it beside them; no other caller yet.
- pixi tasks (pixi.toml is not mine):

      check-svm          = "tools/with_build_lock.sh pixi run mojo run -I . svm/svc_main.mojo"
      check-svm-identity = "tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo"
      check-svm-oracle   = "pixi run mojo run -I . svm/svc_main.mojo -- oracle"

- SVC surface (for a future `python/mojolearn/svm.py` + bindings; NOT
  written in this rung): `svc_fit(ctx, x_row_major, labels, n_rows, n_cols,
  SvmParameter, KernelParams, card, trace) -> SvmModel` and
  `svc_predict(ctx, model, x, n_rows, n_cols, KernelParams, buffer_size_mib,
  predict_class, card) -> List[Float32]`, in `svm/ported/svm/svc_impl.mojo`.
  sklearn-shaped defaults a wrapper should map: `C=1.0, kernel='rbf',
  gamma='scale' -> 1/(n_features * X.var()) computed on the host before the
  call (their python does the same, `svm_base.pyx:305`), `tol=1e-3,
  cache_size -> 0 in rung 1 (refused otherwise), max_iter=-1,
  class_weight/probability/decision_function_shape -> refused by name`.
  Labels may be any two floats; the larger is the +1 class, exactly as
  `getOvrlabels(idx = 1)`.
- No second vendor has run `svm/`. The E-series leg for it is: build both
  modes on the H100/MI325X and run `svc_main.mojo` plus the `card` form,
  then `tools/identity_trace_diff.py` against `/tmp/svm_mac.card`.
