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
intercept out. 18 of 18 gates pass under `NUMERIC_IDENTICAL` (device ==
host oracle BITWISE on all seven fixtures: working-set sequence, alpha and
`f` after every outer iteration, `b`, dual coefs, support indices, decision
function on 277 queries; launch-invariant over 7 arms x 3 fixtures; the
row-39 signed-zero fixture; the NaN audit), 18 of 18 under `NUMERIC_FAST`
where the device-vs-oracle, launch-invariance and two-card gates are
REPORTS (see "Commands and outputs" at the foot). No timing was measured
and none is published. The ROW 39 AUDIT section below (2026-08-23) is the
signed-zero / NaN / FAST-assertion audit; DEVIATIONS 635-637 came out of it.

**SVR, 2026-08-31: PORTED, UNGATED, AND STILL REFUSING.** All six pieces
`NOT_IMPLEMENTED.tsv` rung 2 listed are in the tree: the scope check, the
`n_train = 2 * n_rows` domain in `SmoSolver` and `Results`, `SvrInit`
(`f = +-epsilon - y` and the `+-1` label vector), `UpdateF`'s second gemv on
`f + n_rows`, `CombineCoefs`' fold of the two alpha halves, and
`KernelCache`'s `ws_idx_mod_svr` with `GetVecIndices`. `solve` raises anyway,
and the message says why: **no fixture, no sabotage arm and no
eps-insensitive oracle exist**, so no run has ever checked that the six
pieces agree with each other. A learner nobody has measured is not a learner.

That last item is the load-bearing one and it is not paperwork. The device
arm and `smo_oracle.mojo` are two implementations, but they are written from
one reading of cuML, so a shared misreading of `SvrInit`'s signs or of
`CombineCoefs`' direction would pass a device-versus-oracle gate silently.
The two gates in this lane that test a PROPERTY rather than an agreement,
the global KKT gap and the monotone dual objective, are both
classification-shaped and have no SVR form. Either an independent arm on
`tools/knn_sklearn_oracle.py`'s model, or the eps-insensitive dual and tube
asserted directly, has to exist before the refusal comes out.

Two more things a regression fixture has to be built to catch.
`fold_order_for`'s docstring asserts that working-set indices are distinct
within a set, which is TRUE for C_SVC and FALSE under SVR the moment row `i`
and its twin `i + n_rows` are both selected and `ws_idx % n_rows` collides.
And `f = +-epsilon - y` puts `+0.0` and `-0.0` into the SAME vector whenever
`y == epsilon`, from ORDINARY input, where the signed-zero section below
reached `f = -0.0` only through a planted negative subnormal.

REFUSED BY NAME (each raises with the parameter's name;
`check_refusals` gates all of them): `svmType` NU_SVC and NU_SVR, which are
unported upstream too; **EPSILON_SVR, which is now a different kind of
refusal**, see below; `epsilon != 0` ON A CLASSIFIER, where upstream ignores
it (on a regressor it must be finite and non-negative instead);
`cache_size != 0` (the LRU cache; see "the cache
decision"), kernels `POLYNOMIAL` / `TANH` / `PRECOMPUTED`, `sample_weight`
(and class_weight, which is sample_weight upstream), more than two classes,
sparse input; and, DEVIATION 636, non-finite `C`, `tol`, RBF `gamma` (or a
negative one), and any non-finite cell of `X`, `labels` or the predict `X`
(`check_nan_never_recorded` gates all of them). `probability` is not in
their C++ surface. Full table: `svm/NOT_IMPLEMENTED.tsv`.

Files: `svm/derived/svm/{svm_parameter,smo_sets,ws_util,workingset,
kernelcache,smoblocksolve,smosolver,results,svc_impl}.mojo`,
`svm/derived/distance/kernel_matrices.mojo`, `svm/original/
{device_select,pinned_argreduce,smo_oracle,svc_check}.mojo`,
`svm/svc_main.mojo`, `svm/DERIVATION_MAP.tsv`, `svm/NOT_IMPLEMENTED.tsv`.

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
                                cuBLAS stand-in); IDENTICAL: gemm/original/
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
(`neighbors/original/pinned_distance_tile.mojo`, `fma(-2, dot, nx+ny)`
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
with NO lane primitive (the shape of `core/pinned_reduce.mojo`), and ALL
THREE carry an explicit tie-break: **DEVIATION 633** (the two
arg-reductions) and **DEVIATION 635** (`f_max`, row 39), equal values
resolve to the LOWER TRAINING INDEX (`ws_idx[tid]`, not `tid`), so the
winner is a pure function of the working SET and not of the order the
cache permuted it into, and the only equal values a float `max` can tell
apart, `+0.0` and `-0.0`, are decided by the key and not by a vendor's
`max`. `f += q*(Kui - Kli)` is `identical_mul_add(q, ftz(Kui - Kli), f)`;
`(f_u - f)^2 / eta` is `ftz(ftz(d*d) / eta)` with `d = ftz(f_u - f)`; `eta
= ftz(ftz(Kd_i + Kd_u) - ftz(2*Kui))` floored at `1e-12` by the compare `if
eta < 1e-12: eta = 1e-12` (not a hardware `max`). The block size is a
parameter (1024 as theirs, or the smallest power of two `>= n_ws`); a
selection over a total order cannot move with it, and the launch-invariance
gate runs both.

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
f_i)`), then `ftz(-acc / Float32(n_free))`. The two selections are
one-thread serial scans with strict `<` / `>` (`serial_min/max_f32_kernel`),
so the FIRST index in training order wins a `+0.0`/`-0.0` tie, as the
oracle's scan does. Gated bitwise against the oracle.

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
`cache_size > 0` is REFUSED BY NAME and recorded in `NOT_IMPLEMENTED.tsv`; it is
not a silent substitute.

## Commands and outputs (2026-08-23, one M4)

    pixi run mojo run -I . svm/svc_main.mojo -- oracle                # host only
    tools/with_build_lock.sh     pixi run mojo run -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo
    pixi run check-svm-oracle                                         # the pixi tasks
    tools/with_build_lock.sh     pixi run check-svm                   # FAST
    tools/with_identical_mode.sh pixi run check-svm                   # IDENTICAL
    MOJOLEARN_IDENTITY_TRACE=/tmp/svm.card \
        tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo -- card
    python3 tools/identity_trace_diff.py /tmp/svm_mac.card /tmp/svm_other.card

Sabotage arms (each must behave as the table below says):

    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_WS_TIE=1      -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_FOLD_ROTATE=1 -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_STD_EXP=1     -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_NO_FTZ=1      -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_ARG_TIE_HIGH=1    -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_FMAX_NOKEY=1      -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX=1      -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX_SWAP=1 -I . svm/svc_main.mojo

The fixtures (all hashed, 13-bit significands; `svc_check.mojo` header):
F1 blobs n=300 k=8 linear; F2 xor n=240 k=2 rbf gamma=0.5 C=10; F3 dup
n=1280 k=8 linear, 640 rows EACH TWICE (equal `f` forever, and n > 1024 so
the working-set sort and FIFO run); F4/F5 = F2 at C=0.05 / C=100; F6 big
n=1500 k=8 linear overlapping (266 support vectors, selection + FIFO); F7
wide_k n=96 k=200 (the v1 GEMM leaf + tree path).

IDENTICAL, head of the output (the full run prints every fixture's
`n_support`, `b` as decimal/hex, KKT gap and accuracy):

    == svm/svc_main.mojo [IDENTICAL] all ==
    CERTIFIED Apple M4 <-> NVIDIA H100 <-> AMD MI325X at leg 11 both halves (commit 144aa5b, judged by tools/e3_round_judge.sh section 7 on 2026-08-23): the IDENTICAL card is bit-identical across the three vendors, 32 stages; the FAST cards differ, recorded, the shipped arm makes no cross-vendor claim; AMD MI325X is OWED (that leg was not run).
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
      Z.order_A [IDENTICAL] block=32 device diff=-0.0/0x80000000 n_iter=0 | oracle diff=-0.0/0x80000000 n_iter=0 | expected diff bits 0x80000000 OK
      Z.order_A [IDENTICAL] block=1024 ... OK;  Z.order_B [IDENTICAL] block=32 / 1024 device diff=0.0/0x0 ... OK
    PASS block_solve_signed_zero_tie (row 39, DEVIATIONS 633/635, Z x 2 orders x 2 blocks)
      NaN refusals by name (DEVIATION 636): 8 (X NaN, X inf, labels, C, tol, gamma inf, gamma < 0, predict X)
      overflow fixture (linear, |x| ~ 1.5e19, K = +inf everywhere) [IDENTICAL]: RAISED: SMO error: NaN found during fitting. ... (DEVIATION 637: NaN in alpha or f after outer iteration 0, before its record)
    PASS nan_never_recorded (row 39 FACT 2, DEVIATIONS 636/637)
    == svm/svc_main.mojo [IDENTICAL] 18/18 gates passed ==

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
(the vendor matmul routes `n == 1` to a GEMV); the two-card gate is a
REPORT too (its two launches differ in batch shape, which the vendor
matmul may fold differently; on this M4 the cards agree); the signed-zero
gate and the NaN audit ASSERT in both modes (no arithmetic, no library
call on their paths; the overflow trap arm is RECORDED under FAST); 18/18
with those three reported. The two cards (IDENTICAL vs FAST, F2) diverge
FIRST at `svm.iter000.alpha` -- the instrument has an address.

## Sabotage table (all under IDENTICAL; restored after each)

| sabotage (define) | what it breaks | predicted | observed |
|---|---|---|---|
| `SAB_WS_TIE` (`workingset.mojo`): the (key, index) pairs reach the stable sort in reversed position, so equal f sorts by DESCENDING index | DEVIATION 631 | FAIL on F3.dup only; F1/F2/F4/F5/F7 have n <= n_ws (no selection), F6 has no equal f | `F3.dup [IDENTICAL] device vs oracle: DIVERGES (28: outer iteration count device 5 vs oracle 6)`, `FAIL ws_sequence_is_pure_in_f_and_index ... diverges from the (f bits, index) order at iteration 0`; every other fixture IDENTICAL; 14/16 |
| `SAB_FOLD_ROTATE` (`smosolver.mojo`): the f-update fold starts at `block_idx % nnz` | DEVIATION 634 | FAIL device-vs-oracle from the first `f`; FAIL launch invariance on the batch arms | `F1.blobs [IDENTICAL] device vs oracle: DIVERGES (10: f hash differs at outer iteration 0)`; `F6.big launch invariance ... MOVES: batch256:f hash at outer iteration 0 batch100:... all:...`; 14/16 |
| `SAB_STD_EXP` (`kernel_matrices.mojo`): stdlib `exp` instead of `identical_exp` in the RBF arm | DEVIATION 630 / row 12 | FAIL on the three RBF fixtures only | `F2.xor [IDENTICAL] device vs oracle: DIVERGES (21: alpha hash differs at outer iteration 0)`; F1/F3/F6/F7 IDENTICAL; 15/16 |
| `SAB_NO_FTZ` (`smosolver.mojo`): the f update stores without `ftz` | row 10 | NO failure on Apple: the hardware flushes, so the software flush is bit-inert here; a failure is only visible on a denormal-honoring column | 18/18, REPORTED: `F1..F7 device vs oracle: IDENTICAL`. This is the expected Apple result and is NOT evidence the seam is unreached; the seam's value is on CUDA/HIP |
| `SAB_ARG_TIE_HIGH` (`pinned_argreduce.mojo`): both arg-reductions tie to the HIGHER key | DEVIATIONS 633/635, row 39 | FAIL on every fixture with equal f (F1's first iteration has f = -y), on F3.dup, and on Z order A | `F1.blobs [IDENTICAL] device vs oracle: DIVERGES (16: alpha hash differs at outer iteration 0)`; `FAIL ws_sequence_is_pure_in_f_and_index ... at iteration 1`; `Z.order_A block=32 device diff=0.0/0x0 ... oracle -0.0/0x80000000 MISMATCH`; 15/18 |
| `SAB_FMAX_NOKEY` (`smoblocksolve.mojo`): `f_max` through the pre-635 strict-`>` tree, no key | DEVIATION 635, row 39 | FAIL on Z order A only (the tree's survivor among the lower-set zeros `[-inf,+0,-inf,-0,+0,-inf,-0,-0]` is +0.0; the smallest key holds -0.0); order B inert (both rules land on -0.0 there, diff = +0.0 either way) | `Z.order_A [IDENTICAL] block=32 device diff=0.0/0x0 n_iter=0 \| oracle diff=-0.0/0x80000000 ... MISMATCH` (block 1024 the same), `Z.order_B ... OK`; `FAIL block_solve_signed_zero_tie`; 17/18. Vendor-invariant but position-decided: the pre-audit defect |
| `SAB_FMAX_HWMAX` (`smoblocksolve.mojo`): `f_max` through the HARDWARE `max(mine, other)` tree | row 39 FACT 1 | **APPLE-INERT**: Apple's `max(+0,-0)` returns the SECOND operand, and on order A the tree's second operands land on -0.0, the key answer, so nothing moves; on NVIDIA/AMD `max(+0,-0)` = +0.0 (IEEE-2019 maximum), so f_max = +0.0, diff = +0.0 and order A FAILS there. Order B: +0.0 on every vendor = the key answer, inert everywhere | `18/18`, every `Z.order_*` line `OK`. THIS IS THE HAZARD ROW 39 NAMES: a one-device gate cannot see a hardware-max spelling that happens to agree with the key rule on that device, which is exactly why the fold is a key compare and not a `max` |
| `SAB_FMAX_HWMAX_SWAP` (`smoblocksolve.mojo`): the same tree with the operands swapped, `max(other, mine)` | row 39 FACT 1 | FAIL on order A on Apple (the second operand is now the lower position, the tree lands on +0.0); FAIL on NVIDIA/AMD too (+0.0) | `Z.order_A [IDENTICAL] block=32 device diff=0.0/0x0 ... oracle -0.0/0x80000000 MISMATCH`; `FAIL block_solve_signed_zero_tie`; 17/18. The swap MOVES the Apple answer, which proves the hardware max was deciding the zero's sign by operand order |

## ROW 39 AUDIT (2026-08-23, one M4): signed zero, NaN payloads, FAST assertions

The three facts (IDENTITY_PATHS row 39, measured on Apple M4 / NVIDIA H100 /
AMD MI325X): `max(+0.0, -0.0)` is -0.0 on Apple (second operand) and +0.0
on NVIDIA and AMD; a computed NaN's payload is the vendor's; a FAST-pass
assertion of anything vendor-shaped is fatal on the other two. What the
audit found in `svm/`, site by site.

### Sites reviewed

| site | what it is | can +-0 / NaN reach it | verdict |
|---|---|---|---|
| `original/pinned_argreduce.mojo` `pinned_block_argmin` / `pinned_block_argmax` | halving-tree arg-reductions, `ov < mv or (ov == mv and ok < mk)` | YES both zeros (f = +0.0 on the margin; f = -0.0 from a negative subnormal flushed at the f seam, row 10); NaN no (DEVIATIONS 636/637) | KEY-DECIDED: `==` is TRUE on (+0,-0) on every vendor, so the training index decides and the returned value is that sample's own bits; no hardware min/max. Kept, commented, gated (Z) |
| `original/pinned_argreduce.mojo` `pinned_block_max_all` (pre-audit `f_max`) | strict-`>` halving tree, no key | YES both zeros | DEFECT, fixed: on a (+0,-0) tie nothing updates, so the survivor was the tree POSITION's (not "lowest index": with masked lanes the survivor can be a higher position), and the oracle's serial scan keeps the FIRST index -- device diff = +0.0 vs oracle diff = -0.0 with f_u = +0.0 (`svm.iterNNN.diff` is recorded). **DEVIATION 635**: `f_max` is now the key-tied argmax; the oracle's scan carries the same tie. The old fold survives only as the `SAB_FMAX_NOKEY` sabotage arm |
| `derived/svm/smoblocksolve.mojo` `if eta < ETA_EPS: eta = ETA_EPS` (x2) | the `max(eta, 1e-12)` floor of theirs | -0.0 eta YES (`Kd_t + Kd_u == 2*Kui` exactly gives +0.0; -0.0 only via a flushed negative subnormal difference) | a COMPARE, not a hardware max: `-0.0 < 1e-12` is TRUE on every vendor, so ETA_EPS results regardless of operand orientation; NaN eta needs a NaN kernel cell (overflow), raised by DEVIATION 637 before any record. Commented |
| `derived/svm/smoblocksolve.mojo` `tmp_l if tmp_l < q_l else q_l`, `tmp_u if tmp_u < tmp_l2 else tmp_l2` | the alpha-step clip (their `min`) | NO: `alpha` starts at +0.0 and every `a +- q*y` with `0 <= q <= min(tmp_u, tmp_l)` stays in `[+0.0, C]` (exact zero is +0.0 in RN, a positive subnormal flushes to +0.0), so `a`, `C - a`, `q_l = (f - f_u)/eta` (f_u < f) are `>= +0.0` with the sign bit clear | PROVEN UNREACHABLE (comment in the file header) |
| `derived/svm/smoblocksolve.mojo` `diff_end = eps if eps > d10 else d10` | their `max(eps, 0.1*diff)` | d10 = -0.0 YES (diff = -0.0, see Z order A) | compare-select; `eps > -0.0` TRUE (eps > 0 refused otherwise) -> eps on every vendor |
| `derived/svm/smosolver.mojo` `check_stopping_condition` (`abs`, `<`, `>` on diff) | host Float64 rule | diff = -0.0 YES | host arithmetic, sign-insensitive (`abs`, `-0.0 < tol` TRUE); the raw diff bits are recorded as `svm.iterNNN.diff` and are the same on device and oracle (Z gate) |
| `derived/svm/workingset.mojo` + `original/device_select.mojo::twiddle_keys_kernel` | the working-set order | both zeros YES | INTEGER KEYS: `-0.0` twiddles to 0x7FFFFFFF, `+0.0` to 0x80000000, so -0.0 sorts strictly BEFORE +0.0 (cub's own order), deliberately; ties on index. No float compare. Gated by F3.dup |
| `original/device_select.mojo` `serial_min_f32_kernel` / `serial_max_f32_kernel` (`CalcB` bound-only arm) | one-thread strict `<` / `>` scans | both zeros YES (f) | FIRST INDEX in training order wins (the compaction preserves index order); the oracle's `_results` scan has the same rule; a NaN at index 0 would persist but none reaches here. Commented both sides |
| `derived/svm/results.mojo` `b = ftz(-s / n)`, `-ftz(b_up + b_low)/2` | the intercept | `b = -0.0` YES (all free f = +0.0 -> s = +0.0 -> b = -0.0), recorded as `svm.b` | negation and division are IEEE-defined, no `nsz` rewrite on any of the three (row 39 table); device = oracle bitwise |
| `derived/svm/svc_impl.mojo` `decision_kernel` `val = ftz(acc + b)`, `label0 if val < 0 else label1` | predict | `b = -0.0` YES; `val = +0.0` | `(+0) + (-0) = +0`; `-0.0 < 0` FALSE = `+0.0 < 0` FALSE, so the class cannot depend on the zero's sign; the host `n_support == 0` arm spells the same `0.0 + b` |
| `derived/distance/kernel_matrices.mojo` `rbf_kernel_expanded_kernel` | `s = (nx + ny) - 2 dot`, `e = -gamma * s`, `exp(e)` | `s = -0.0` NO (norms are `>= +0.0` chains from +0.0; `a - b` is +0.0 when equal); `s < 0` from cancellation YES -> `exp(+small) > 1`, finite, as theirs; `e = -0.0` YES (`-gamma * (+0.0)`) -> `identical_exp(-0.0) = 1.0` | no max/min/sqrt on the path; NaN only from `gamma = inf` (refused, DEVIATION 636) or an overflowed norm (DEVIATION 637) |
| `derived/svm/kernelcache.mojo` | `if b < 1`, `w1 if w1 > w2` ... | integers (tile sizing) | not numeric |
| `original/svc_check.mojo`, `original/smo_oracle.mojo` (`abs`, `<`, `>` in the host gates) | tolerance gates and the oracle's scans | host only | oracle scans now carry the DEVIATION 635 tie (`_block_solve`), the CalcB scans commented |

No site in `svm/` calls `std.math.max/min` on floats, `SIMD.reduce_max/min`,
`.clamp()`, `Atomic.min/max`, `copysign`, or `core/pinned_reduce.mojo`'s
`pinned_block_max/min` (`grep` of the directory: the only `max(` on a float
is the sabotage arm `sabotage_block_max_hw`, which the port never calls).

### The -0.0 fixture and its sabotages

`check_block_solve_signed_zero_tie` (gate 17): a working set of 8 planted
at the BLOCK-SOLVE ENTRY (the real `launch_block_solve`, block 32 and 1024,
against the real oracle `_block_solve`), keys a permutation, the upper set
(y = +1, alpha = 0) and the lower set (y = -1, alpha = 0) disjoint, EVERY f
a zero: order A `f by key = [-0,+0,-0,+0,-0,+0,-0,-0]` so the smallest
upper key holds +0.0 (f_u) and the smallest lower key holds -0.0 (f_max),
`diff = -0.0 = 0x80000000`; order B is every sign flipped, `diff = +0.0`.
Asserted bitwise (diff, n_iter, alpha, delta_alpha) AND against the
rule's predicted bits, in BOTH modes (no arithmetic runs at n_iter = 0;
`(+-0) - (+-0)` is IEEE-exact; integer keys and IEEE compares only). Why
planted: `f = -0.0` is reachable in a fit only through a negative
subnormal flushed at the f seam, which no O(1)-scale 13-bit-significand
input can produce, and `f = +0.0` needs a sample exactly on the margin;
the entry is the only place both can be put in one set.

| sabotage | Apple result | predicted NVIDIA/AMD | why it matters |
|---|---|---|---|
| `SAB_FMAX_NOKEY` (the pre-635 tree) | FAIL order A (`device diff=0.0/0x0 vs oracle -0.0/0x80000000`), B inert; 17/18 | FAIL order A | the defect the audit fixed: position-decided |
| `SAB_FMAX_HWMAX` (`max(mine, other)` tree) | **INERT**, 18/18 (Apple's `max` returns the SECOND operand and the tree's second operands land on -0.0 = the key answer) | FAIL order A (`max(+0,-0) = +0` there) | the Apple-inert hardware-max spelling: one device cannot see it, which is exactly why the fold must be a key compare and not a `max` |
| `SAB_FMAX_HWMAX_SWAP` (`max(other, mine)`) | FAIL order A; 17/18 | FAIL order A | the swap MOVES the Apple answer: the hardware max was deciding the zero's sign by operand order |
| `SAB_ARG_TIE_HIGH` (both arg-reductions tie to the higher key) | FAIL F1 (`alpha hash differs at outer iteration 0`), F3.dup ws order at iteration 1, Z order A; 15/18 | the same | the key direction is load-bearing on every fit's first inner iteration (f = -y has two values) |

The sabotages are compile-time defines; the clean source was run in both
modes before and the FAST/IDENTICAL runs above are the restored state.

### NaN audit, per recorded stage (FACT 2)

Stages: `svm.input.x`, `svm.input.y`, `svm.init.f`, per outer iteration
`svm.iterNNN.{ws_idx,diff,alpha,f}`, `svm.b`, `svm.dual_coefs`,
`svm.support_idx`, `svm.n_iter`, `svm.n_outer_iter`,
`svm.predict.{class,decision}.{x,out}`.

| NaN source | where it is stopped |
|---|---|
| NaN / inf in `X`, `labels`, predict `X` | **DEVIATION 636** (`svm_parameter.mojo::check_finite_list`, called from `svc_fit` / `svc_predict` before any stage): refused naming the array and the flat index |
| `C = inf` (alpha unbounded -> `inf - inf`), `tol = NaN` (passes `tol > 0`, never stops), RBF `gamma = inf` (`-inf * 0` on the diagonal) or `gamma < 0` (exp overflow) | **DEVIATION 636** (`check_rung1_scope`): refused by name |
| float OVERFLOW of a finite input (`|x| ~ 1e19`: every K = +inf, `eta = inf - inf = NaN`, `q = NaN`, NaN alphas; or `inf - inf` in `f += q*(Kui - Kli)`, or an overflowed RBF norm) | **DEVIATION 637** (`smosolver.mojo`): `flag_nan_f32_kernel` over `alpha` and `f` once per outer iteration, read back beside their `host_return_buff`; raises their "SMO error: NaN found during fitting" sentence BEFORE the iteration's record. Theirs catches only a NaN `diff`, and `diff` is the FIRST inner iteration's, so a NaN born later in the block reaches alpha and f unseen; the overflow fixture in `check_nan_never_recorded` shows exactly that (the raise carries the "(DEVIATION 637 ...)" suffix, not the diff path) |
| `svm.b` = `-(b_up + b_low)/2` with `b_up = +inf`, `b_low = -inf` | DEVIATION 637: `isnan(model.b)` raises before `svm.b` is recorded |
| predict `sum K * dual + b` overflowing to `inf + -inf` | DEVIATION 637: the host scan of `preds` raises before `svm.predict.*.out` is recorded |
| fixtures' scratch poisons (`-7.25e20`, `3.0e-39`) | padding beyond `n_ws` in `delta_alpha`; `record_device` reads `n_ws` / `n_train` counts only, and the block solve's padding lanes pass `+-inf`, never the poison |
| `inf` (not NaN) in `f` or `alpha` | NOT trapped, deliberately: an inf's bits are the same on every vendor; it is recorded as inf and the NaN it turns into raises the next iteration |

Verdict: no recorded stage can hold a computed NaN; no NaN canonicalization
is needed and none was added.

### FAST demotions (FACT 3)

`check_card_is_emitted`: its two traced fits differ in full-tile batch,
predict batch and padding; under FAST the kernel rows are the vendor
matmul, which may fold per batch shape, so the "two cards agree" assertion
is vendor-shaped there. Now `if IDENTICAL: raise else: RECORDED [FAST]`.
Every other FAST-mode raise in `svc_check.mojo` was already either
IDENTICAL-gated (`device_matches_oracle`, `launch_invariant`, the FAST arm
of `ws_sequence` asserts only the first, arithmetic-free working set),
host-only (the oracle gates, the DEVIATION 630 ULP measurement,
`check_finite_list`), refusal-by-name, or the new signed-zero gate whose
path has no arithmetic and no library call (asserted in both modes, stated
above). The overflow-trap arm of `check_nan_never_recorded` is RECORDED
under FAST (a fast-math build may fold `x != x`).

### Commands and results of the audit (one M4)

    tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo      -> 18/18 gates passed [IDENTICAL]
    tools/with_build_lock.sh     pixi run mojo run -I . svm/svc_main.mojo      -> 18/18 gates passed [FAST]
    MOJOLEARN_IDENTITY_TRACE=... tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo -- card
                                                                                -> card written, 35 lines, n_support=16
    sabotages: see the table above (NOKEY 17/18, HWMAX 18/18 inert, HWMAX_SWAP 17/18, ARG_TIE_HIGH 15/18)

## ROW TEXT FOR THE IDENTITY LANE

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| 44 | `svm/` C-SVC (`svcFit`/`svcPredict`): SMO decomposition, block solve, working-set selection, kernel matrices, results | (1) kernel rows: cuBLAS GEMM + `rowNorm` block fold + RBF `exp` promoted to DOUBLE for float; (2) working-set order from cub radix-sort STABILITY, unstated; (3) `cub::BlockReduce<KVPair>` compares `val` only -- the winner among equal f is the fold shape's (their own `@todo`); (4) `UpdateF` is `cublasgemv`, fold unspecified, over cache-permuted columns; (5) `CalcB`'s mean is `cub::DeviceReduce::Sum` | (1) v1 identical GEMM + per-row serial norm + `identical_exp` in float32 (DEVIATION 630, measured 1 ulp from their double spelling); (2) total order `(twiddled f bits, index)` spelled, stable one-bit radix + device scan compaction (DEVIATION 631); (3) halving-tree argmin/argmax, all three reductions tie to the smaller TRAINING INDEX (DEVIATIONS 633/635; a +0.0/-0.0 tie is decided by the key, never a hardware max, row 39); (4) hand GEMV, fold in ascending training index (DEVIATION 634), so nothing numeric sees the cache permutation; (5) serial ascending chain (DEVIATION 632). Host oracle in Float32 through the same helpers; device == oracle bitwise on 7 fixtures; launch-invariant over 7 arms; non-finite inputs refused by name and a NaN in alpha/f raises before any record (DEVIATIONS 636/637); 8 sabotages recorded | Apple M4 only, 18/18 IDENTICAL; cache_size > 0 refused (LRU unported, proven bit-inert by construction); no second vendor |

## HAND-OFF TO THE IDENTITY LANE

- IDENTITY_PATHS.md: row 44 is this lane's (assigned at `633a562`); the
  row text above is its current state after the row-39 audit.
- `original/numerics.mojo`: no change needed for rung 1. A `TANH` kernel
  port would need an `identical_tanh`; not requested.
- `core/pinned_reduce.mojo`: `svm/original/pinned_argreduce.mojo` is the
  argmin/argmax-with-index sibling of `pinned_block_max/min` (same halving
  shape, explicit key tie-break). If the identity lane wants ONE home for
  it, move it beside them; no other caller yet.
- pixi tasks exist: `check-svm` (FAST via `tools/with_build_lock.sh pixi
  run check-svm`; IDENTICAL via `tools/with_identical_mode.sh pixi run
  check-svm`) and `check-svm-oracle`; no `*-identity` task by design.

- SVC surface (for a future `python/mojolearn/svm.py` + bindings; NOT
  written in this rung): `svc_fit(ctx, x_row_major, labels, n_rows, n_cols,
  SvmParameter, KernelParams, card, trace) -> SvmModel` and
  `svc_predict(ctx, model, x, n_rows, n_cols, KernelParams, buffer_size_mib,
  predict_class, card) -> List[Float32]`, in `svm/derived/svm/svc_impl.mojo`.
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
