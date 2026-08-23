# gemm: the cross-vendor bit-identical FP32 GEMM lane

Opened 2026-08-23. The lane's brief is the `LANE CHARTER` section of
`IDENTICAL_GEMM_PLAN.md`. DEVIATIONS 530-539 are this lane's.

**Status: PHASE 0 AND PHASE 1 ONLY.** The contract is written and the oracle
and its adversarial fixtures run green in both modes. There is NO scalable
kernel (Phase 2), NO invariance gate (Phase 3) and NO benchmark (Phase 4) —
deliberately, because the charter says those must be built on a contract that
has been reviewed and building ahead of the contract is what it forbids.

    pixi run mojo run -I . gemm/mojo_only/gemm_oracle_check.mojo

    # both modes, the way the other identity gates do it:
    tools/with_build_lock.sh     pixi run mojo run -I . gemm/mojo_only/gemm_oracle_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . gemm/mojo_only/gemm_oracle_check.mojo

No pixi task is registered; the orchestrator registers it. Host only — no GPU
is required to run any of this.

## WHY THIS DIRECTORY IS NOT CALLED `linalg`

The charter says `linalg/`. **MAX ships a package called `linalg`, and it
wins.** `core/gemm.mojo` imports `from linalg.matmul import matmul` and `from
linalg.gemv import gemv_gpu` out of it. With a repo-local `linalg/` on the
`-I .` path, `from linalg.mojo_only.gemm_oracle import ...` resolves into
MAX's package and fails:

    error: unable to locate module 'mojo_only'
    error: 'mojo_only' does not refer to a nested package

Measured directly, 2026-08-23, on a two-line probe. `gemm`, `blas`,
`contraction` and `linalg2` were all probed and are all free; `gemm` was
chosen because the charter's own name for the deliverable is a GEMM. **The
one thing that does NOT happen is a shadowing of MAX's package** — the
resolution goes the other way, so `core/gemm.mojo`'s imports are untouched,
which was checked before the rename.

`gemm/ported/linalg/contractions.mojo` still mirrors the upstream path
exactly, because the collision is at the TOP level only.

## What is here

| file | what |
|---|---|
| `IDENTICAL_FP32_CONTRACT.md` | **the deliverable.** Twelve sections: dtypes, layout, the three orientations, the multiply-add policy, the flush policy and its seven seams, the logical k partition, the evaluation order, ragged k, NaN / infinity / signed zero, the exclusions, what is NOT promised, and a clause-to-code index. |
| `mojo_only/gemm_oracle.mojo` | the contract in code. Host, scalar, single-threaded, built from `identical_mul_add` and `ftz` so that it IS the contract rather than an opinion about it. The partition count is a PARAMETER, which is what makes the fixtures provable. |
| `mojo_only/gemm_oracle_check.mojo` | the adversarial fixtures and their separation proofs. Every one refuses to pass unless the two alternatives produce different bits. |
| `ported/linalg/contractions.mojo` | `raft/linalg/contractions.cuh`'s `KernelPolicy`, `ColKernelPolicy` and the three float policy families, parameterized. COPY, DO NOT IMPROVE. |
| `PORTED_MAP.tsv`, `UNPORTED.tsv` | upstream -> ours, and what was not ported and why. |

## Upstream

cuVS `94c2819`, cuML `00094f7`, RAFT `661a3b8` (`PORTING_RULES.md` 0a).

**RAFT'S STANDALONE MATRIX PRODUCT HAS NO SOURCE TO PORT.**
`raft/linalg/gemm.hpp` and `raft/linalg/detail/gemm.hpp` are wrappers over
`cublaslt_wrappers.hpp`, i.e. over cuBLASLt. `PORTING_RULES.md` 0b-i's narrow
exception applies verbatim: *"where the path their dispatch actually takes
calls a CLOSED library we cannot read or port — cuBLAS, cuSOLVER — call the
MAX equivalent, because there is nothing to port."* So there is no upstream
reference GEMM anywhere in cuML, cuVS or RAFT to check ours against, and the
oracle is what stands in for one.

**WHAT IS OPEN, AND IS PORTED:** `raft/linalg/contractions.cuh`'s policy, and
the ascending-k ORDER of the contraction that policy parameterizes
(`raft/linalg/detail/contractions.cuh` for the loader,
`raft/distance/detail/pairwise_distance_base.cuh:139-149` and `:223-241` for
the loop). That order is the reason contract section 7.1 says "ascending, one
leaf at a time": upstream walks `kidx` from 0 to `k` ascending in steps of
`Kblk` and `ki` from 0 to `Kblk` ascending in steps of `Veclen`, with ONE
block owning the entire `k` range of its output tile — no split-K, no
cross-block combination. The contract mirrors it rather than inventing a rule.

**PyTorch was NOT reached for.** The charter's fallback ("when cuML and RAFT
are exhausted for a routine, switch to PyTorch's algorithms and mirror those,
recording the switch") did not fire in Phase 0 or Phase 1: the policy came
from RAFT and the oracle has no upstream by construction. If Phase 2's
scalable kernel needs a split-K structure RAFT does not have, that is the
first place the switch would be recorded, and it goes in `PORTED_MAP.tsv`.

---

# THE INVENTORY: every contraction arm in this repository

Phase 0's first deliverable. Ten arms plus their callers, swept across
`core/`, `cluster/`, `neighbors/`, `dbscan/`, `glm/`, `decomposition/`,
`ensemble/`, `extratrees/`, `gbdt/`, `mojo_only/`, `bench/` and `bindings/`.
Line numbers are as of 2026-08-23; several of these files are under live edit
by other lanes.

## Standalone products (UNFUSED — the product matrix is materialized)

| # | arm | file:line | operation | shape it serves | ours / vendor | in scope |
|---|---|---|---|---|---|---|
| 1 | `gemm_nt` (dispatcher) | `core/gemm.mojo:211` | `C = A·Bᵀ` | general; `n == 1` rerouted to `gemv_n` first | dispatcher | **YES — `OP_NT`** |
| 1a | ↳ `matmul[transpose_b=True]` | `core/gemm.mojo:274` | `C = A·Bᵀ` | the FAST arm | **VENDOR** (MAX `linalg.matmul`) | stays under FAST; IDENTICAL must never fall through to it |
| 1b | ↳ `pinned_gemm_nt_kernel` | `core/gemm.mojo:122` | `C = A·Bᵀ`, one thread per cell, k ascending | any | OURS | **YES** — it is the contract at `P == 1`, and see the defect below |
| 2 | `gemv_n` (dispatcher) | `core/gemm.mojo:455` | `z = A·y` | OLS step 6 at 128x128; OLS predict at `n_rows x n_features` | dispatcher | **YES — `OP_NT` at `n == 1`** |
| 2a | ↳ `gemv_gpu[transpose_b=False]` | `core/gemm.mojo:505` | `z = A·y` | the FAST arm | **VENDOR** (MAX `linalg.gemv`) | stays under FAST |
| 2b | ↳ `pinned_gemv_n_kernel` | `core/gemm.mojo:157` | `z = A·y`, one thread per element | any | OURS | **YES** — same defect |
| 3 | `gemm_tn` (dispatcher) | `core/gemm.mojo:277` | `C = Aᵀ·B` (Gram) | m = n = n_features ≤ 128, k = n_rows in the millions | dispatcher; RAISES under IDENTICAL over capacity | **YES — `OP_TN`** |
| 3a | ↳ `gemm_tn_via_transpose` | `core/gemm.mojo:362` | two `transpose_kernel` launches then `gemm_nt` | outputs with enough tiles to fill the device | **VENDOR** behind two of our transposes | in scope as the thing `OP_TN` replaces |
| 3b | ↳ split-K Gram pair | `core/gram_splitk.mojo:423`, `:679`, `:693`, `:711` | `C = AᵀA`, k-chunked partials + a serial ascending fold | tiny square output, enormous k | OURS | **YES** — it is Phase 2's architectural seed |
| 4 | `pinned_distance_tile_kernel` | `neighbors/mojo_only/pinned_distance_tile.mojo:61` | `C = ‖q‖²+‖y‖²−2·q·yᵀ` in one kernel | query tile x n_index x n_features | OURS | **CONSUMER, not a shape.** Its product is an `OP_NT`; its epilogue is an epilogue, which contract section 10 excludes |
| 5 | `xty_kernel` | `core/column_stats.mojo:205` | `out = Aᵀb` | one block per feature, striding rows | OURS | **NO** — already pinned (`identical_mul_add` + `pinned_block_sum`); it is a gemv-transpose with its own fold, not a GEMM shape |
| 6 | `row_norm_kernel` | `core/row_norms.mojo:68` | `‖aᵢ‖²`, the diagonal of `A·Aᵀ` | one block per row | OURS | **NO** — a vector reduction; every fused arm consumes it |
| 7 | `transpose_kernel` | `core/column_stats.mojo:347` | `dst = srcᵀ`, no arithmetic | tile 32 | OURS | **NO** — but native `OP_NN`/`OP_TN` delete three of its call sites |
| 8 | `expand_distances_kernel` | `core/expand_distances.mojo:21` | the distance epilogue over a materialized product | k-NN FAST arm | OURS | **NO** — an epilogue (section 10) |

## FUSED arms — OUT OF SCOPE, and the charter says why

*"Do not replace an existing specialized FUSED contraction merely to make the
API look general."* Each of these accumulates a product in registers and
consumes it in the same kernel; the `m x n` matrix is never written.

| # | arm | file:line | operation | shape | ours / vendor | why out of scope |
|---|---|---|---|---|---|---|
| 9 | `fused_distance_nn_kernel` | `cluster/ported/distance/fused_distance_nn/simt_kernel.mojo:245` | `A·Bᵀ` ⊗ expanded-L2 ⊗ per-row `(value,key)` argmin | m = n_samples (2·10⁵), n = n_clusters (16), k = n_features | OURS (cuVS port) | the materialized product at k-means' shipped shape is 3.2 M floats it never writes |
| 10 | `fused_l2_knn_kernel` | `neighbors/ported/neighbors/detail/fused_l2_knn.mojo:297` | `A·Bᵀ` ⊗ expanded-L2 ⊗ register `WarpSelect` top-k | m = n_queries, n = n_index (4·10⁵), k = 32 | OURS (cuVS port) | its own header prices the alternative: a 409.6 MB tile, ~23 GB of traffic for 51.2 GFLOP |
| 11 | `eps_unexp_l2_sq_neigh_kernel` | `dbscan/ported/neighbors/epsilon_neighborhood.mojo:132` | `Σ(x−y)²` UNEXPANDED ⊗ `≤ eps²` ⊗ degree sum | m = batch, n = n_rows, k = n_features | OURS (RAFT port) | **it is not a matrix product.** Reaching it through a GEMM needs the expanded identity, a DIFFERENT arithmetic with different cancellation — and here one ULP flips an adjacency BIT (IDENTITY_PATHS row 19) |
| 12 | `eps_dist_sq` | `neighbors/ported/neighbors/ball_cover/common.mojo:58` | per-pair `Σ(a−b)²` functor | nine call sites inside ball-cover kernels | OURS (RAFT port) | there is no matrix to produce; the value feeds an eps compare or a 1-NN argmin immediately |
| 13 | `std_dev_partials_kernel` | `gbdt/methods/random_score_helper.mojo:86` | `Σ wᵢ·(wtᵢ/wᵢ)²` | per-leaf | OURS (CatBoost port, DEVIATION 137) | DEVIATION 137 fused `DivideVector` + `DotProduct` precisely so `tmp` is never materialized |

## Probed and NOT wired

| thing | where recorded | verdict |
|---|---|---|
| `linalg.bmm.batched_matmul` | `mojo_only/vendor_correctness_check.mojo:1820`; rejected for the Gram shape at `core/gram_splitk.mojo:28-33` | **no production caller anywhere.** Batched GEMM is DEFERRED — see the contract, section 0.3 |
| `matmul[transpose_a=True]` | `core/gemm.mojo:437-441` | refused at compile time by MAX (`matmul/__init__.mojo:110`). This is why `OP_TN` is spelled as two transposes today |
| `linalg.gemv.gemv` (host) | `core/gemm.mojo:466`, `glm/.../lstsq.mojo:38-52` | host-only, no ctx, no target |
| `linalg.transpose` | `core/column_stats.mojo:355` | compiles and SIGNALS on device buffers |

## Host Float64 contractions — outside the device profile

`gbdt/lapack/linear_system.mojo:58` (Cholesky + triangular solves, a `dposv`
port) and `gbdt/methods/leaves_estimation/step_estimator.mojo:59` (the Armijo
direction·gradient dot). Both host, both Float64, neither in this profile.

## Which operations are actually required, and the batched recommendation

Full evidence in the contract, section 0.2 and 0.3. In brief:

- **`OP_NT`: seven production callers.** k-means unfused assignment, k-means++,
  the tiled k-NN distance step, `pca_transform`, `tsvd_transform_host`, OLS
  step 5, and `gemm_tn_via_transpose`'s own tail.
- **`OP_TN`: three production callers** (OLS step 1, `compute_covariance`,
  `tsvd_fit`) and today all three go through **two materialized transposes**.
- **`OP_NN`: one production caller** —
  `decomposition/estimator.mojo:135-140`, `inverse_transform_host` — and it
  too is spelled as a transpose plus an `OP_NT`.
- **Batched: DEFERRED.** No production caller passes a batch dimension. And
  under contract sections 6 and 7 a batched GEMM CANNOT produce different bits
  from a loop of unbatched ones, because the arithmetic for a cell depends on
  `k` and the profile alone. It is an execution-plan feature: latency, not
  numerics. Deferring it costs nothing in the contract and adds nothing to it
  later.

Recommendation: implement all three orientations natively in Phase 2, because
under contract section 3 they are ONE kernel with two `if`s in the addressing,
and each one deletes a materialized transpose from a shipped path.

---

# The six adversarial distinctions, and the bits that prove they separate

Every fixture below is BUILT TO SEPARATE and the check REFUSES to pass unless
the two alternatives produce different bits. `mojo_only/ieee_arith_check.mojo`
scoring a contracting backend as UNFUSED on 2^20 hashed patterns — of which
ZERO separate the two spellings — is why this is the standard here.
Bit patterns below are from the IDENTICAL run; the FAST run prints the same
ones, because the adversaries are written with explicit spellings rather than
with the mode-gated helpers.

| # | distinction | fixture | alternative A | alternative B |
|---|---|---|---|---|
| F1 | serial vs split-k | `A = [2^24, 1 x 1023]`, `B = 1`, `k = 1024`. `2^24` has ulp 2, so `2^24 + 1` is the halfway case round-half-to-even discards: the serial chain swallows all 1023 additions. Partitioned, each later leaf sums its own ones to an exact 128 and the fold adds them where they ARE representable | serial `P = 1` → `16777216.0 / 0x4b800000` | contract `P = 8` → `16778112.0 / 0x4b8001c0` |
| F2 | one partition count from another | same | `L = 128, P = 8` → `0x4b8001c0` | `L = 64, P = 16` → `16778176.0 / 0x4b8001e0` (and a third point, `L = 256, P = 4` → `0x4b800180`) |
| F3 | fused vs unfused multiply-add | `mojo_only/ieee_arith_check.mojo` ARM 6's construction inside a `k = 2` GEMM: `a0, b0` half-width mantissas, `A = [1, a0]`, `B = [−fl(a0b0), b0]`. Unfused is `q − q = +0.0` EXACTLY; fused is the rounding error. 1,621 of 4,096 patterns separate | fused → `5.9604645e-08 / 0x33800000` | unfused → `0.0 / 0x00000000` |
| F4a | FTZ vs gradual underflow, RESULT seam | `A = [1.5·2⁻¹²⁶, 2⁻¹²⁶]`, `B = [1, −1]`. **Both operands NORMAL**, their difference `2⁻¹²⁷` SUBNORMAL, and it is the answer | FTZ → `0.0 / 0x00000000` | gradual → `5.877472e-39 / 0x00400000` |
| F4b | FTZ vs gradual underflow, OPERAND seam | `A = [2⁻¹²⁷]` (subnormal) `x B = [2²⁰]`. The unflushed product is an ordinary NORMAL number, so the divergence is not confined to the subnormal range | FTZ → `0.0 / 0x00000000` | gradual → `6.162976e-33 / 0x0a000000` |
| F5 | balanced fold vs ascending serial fold | partition held at the contract's `L = 128`; a single `2^24` in leaf 0 and a single `1.0` in each of the other seven, zeros elsewhere (`fma(0,1,acc) == acc` exactly). Serial swallows all seven ones; the halving tree pairs them into 2s and 4s first | serial → `16777216.0 / 0x4b800000` | halving → `16777222.0 / 0x4b800003` |
| F6a | signed zero — the FOLD SEED | every one of the 8 partials is exactly `−0.0`, produced by `ftz` of a negative subnormal placed at the END of each leaf (at the start, the trailing zero products erase the sign — `fma(0,1,−0.0)` is `+0.0`) | `+0.0` seed (contract) → `0.0 / 0x00000000` | first-partial seed → `−0.0 / 0x80000000` |
| F6a-P1 | signed zero — the UNCONDITIONAL FOLD | the same at `k = 128`, where `P == 1` and "seeded with the first partial" IS "skip the fold, one leaf needs no folding" | fold runs (contract) → `0.0 / 0x00000000` | fold bypassed → `−0.0 / 0x80000000` |
| F6b | cancellation | `[2^24, 1, …, −2^24, −1, …]`, exact sum ZERO, at the contract's `L = 128` | serial `P = 1` → `−1.0 / 0xbf800000` | contract `P = 8` → `0.0 / 0x00000000` |

**Every distinction the charter named has a separating fixture. None had to be
recorded as unbuildable.**

Three more checks sit beside them and are not fixtures in the same sense:
`check_ieee_zero_assumptions` (the three IEEE facts section 9.2 rests on,
asserted through a `List` so a constant folder cannot answer for the
hardware), `check_leaf_partition_is_a_pure_function_of_k` (the partition table
at nine `k`, plus the no-empty-leaf invariant), `check_ported_policy_matches_
upstream` (RAFT's policy arithmetic, cross-checked against `core/gemm.mojo`'s
independent flattening of the same table), `check_oracle_matches_the_contract_
spelling` (the reach proof for `identical_mul_add` and `ftz`) and
`check_orientations_agree` (NN == NT == TN bit for bit, with a wrong-layout
sabotage proving the check can see a difference at all).

## Two fixtures that were WRONG on the first try, recorded because that is the point

Both were caught by the checks' own refusals, not by inspection.

1. **F3's "unfused" spelling was fused.** The first version wrote `var prod =
   a * b; return prod + c` — the spelling `ieee_arith_check.mojo`'s ARM 6
   relies on, and the spelling that is supposed to give a compiler nothing to
   contract. It refused itself on **1,621 of 4,096 patterns**: Mojo's HOST
   codegen contracted across the two statements. The replacement builds the
   unfused answer out of `fma(a, b, +0.0) + c`, where no multiply appears
   syntactically and no compiler may rewrite it, since `fma(a,b,0)+c` and
   `fma(a,b,c)` are different values. Had the check only reported a
   separation, it would have reported a real difference and attributed it to
   the wrong axis.
2. **The helper reach proof tested nothing.** Its first fixture put the
   subnormal in the accumulator and then added `1.0`, which swallows it: both
   spellings returned `1.0` and the FAST arm cheerfully reported "agree". It
   now uses F4a's fixture and calls `_separates` on the two arms FIRST, so a
   fixture that stops separating fails loudly instead of passing quietly.

---

# Reported, not fixed — defects in files this lane does not own

## 1. `core/gemm.mojo`: the two pinned kernels do not run the fold

`pinned_gemm_nt_kernel:154` stores `ftz(acc)` and `pinned_gemv_n_kernel:183`
stores `ftz(acc)`. Contract section 9.2(b) requires the one-term fold, whose
only arithmetic effect is to turn a `−0.0` accumulator into `+0.0`.

Today both kernels always run at `P == 1`, so nothing disagrees with them and
no shipped bit moves. **Phase 2's partitioned arm WILL disagree**, on exactly
the negative-subnormal cell, and then the same product has two signs depending
on which arm ran. Fixture F6a-P1 is that case.

The exact change, one expression each:

    core/gemm.mojo:154   z.unsafe_store(cell, ftz(acc))
                      -> z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))

    core/gemm.mojo:183   z.unsafe_store(i, ftz(acc))
                      -> z.unsafe_store(i, ftz(Float32(0.0) + ftz(acc)))

`core/gram_splitk.mojo::gram_splitk_reduce_kernel` already complies — its fold
is `+0.0`-seeded and runs unconditionally.

## 2. `core/gram_splitk.mojo` pins the COUNT where the contract pins the SIZE

`PINNED_GRAM_SPLITK_CHUNKS = 128` (`:209`) is a chunk COUNT, so the chunk
LENGTH is `ceil(k / 128)` and grows without bound: at `k = 4,000,000` each
chunk is a 31,250-term serial fp32 chain. The contract fixes the LEAF SIZE
instead and derives the count, so the same `k` gives 1,024 leaves of 3,907.

Two consequences Phase 2 has to reconcile, neither of which is a bug today:

- **Conditioning.** 31,250 sequential roundings per partial against 3,907.
- **Empty chunks.** Pinning the count first means `k < 128` produces empty
  chunks; the kernel handles them explicitly (*"A chunk whose slice starts at
  or past k writes an all-zero partial"*). The contract's derivation order
  makes an empty leaf impossible, so that handling has nothing to handle —
  which is a simplification, not a fix.

Not changed here: `gram_splitk` is another lane's file and its constant is
load-bearing for the workspace sizing and for `gram_splitk_scratch_covers`,
which read the same function.

## 3. Mojo's HOST codegen contracts across statements

Measured 2026-08-23 by fixture F3 refusing itself: `var prod = a * b; return
prod + c` returned the FUSED answer on 1,621 of 4,096 patterns.

This is the mojotrees plateau-tie incident's class, reproduced on the host,
and it is a warning for any future check that writes an "unfused" adversary
that way. It is **not** a refutation of
`mojo_only/ieee_arith_check.mojo`'s ARM 6, which relies on the DEVICE
compiler, not the host one, and whose verdict (Metal FUSED on 1,629 of 1,629)
is unaffected either way — a host that contracts and a device that contracts
agree. The suggested change is defensive, not corrective: ARM 6's `var prod =
av2 * bv2` on `ieee_arith_check.mojo:365` computes the fixture's `c` and would
be more robust as `fma(av2, bv2, Float32(0.0))`, which is the same value
except at a zero product and cannot be contracted into anything.

## 4. No new kernel-matrix or hardware-matrix row is needed

Checked, because the charter requires the request to be made rather than the
edit. Nothing in Phase 0 or Phase 1 has a vendor-divergent constant: the two
profile constants are pure numbers, and the oracle is host-only. Phase 2 will
need one only if the scalable kernel's EXECUTION plan wants a per-vendor tile
size — and by contract section 6.1 that row must not be readable from anything
the numerical plan touches.
