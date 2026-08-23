# gemm: the cross-vendor bit-identical FP32 GEMM lane

Opened 2026-08-23. The lane's brief is the `LANE CHARTER` section of
`IDENTICAL_GEMM_PLAN.md`. DEVIATIONS 530-539 are this lane's.

**The profile is `mojolearn.identical.gemm.fp32.v1`.** Named 2026-08-23 by
Andrew's Phase 2 contract call, which also replaced the fold ACROSS leaves
with a fixed balanced tree. Changing the leaf rule or the fold topology from
here creates a v2; it does not amend v1.

**Status: PHASES 0, 1, 2a, 2b, 3 AND 4 ARE LANDED, ON ONE DEVICE.** The
contract is written and versioned, the two oracles are split and named, the
fixtures — covering all seven of clause 5's distinctions — run green in both
modes, the device kernel matches the oracle bit-for-bit at 62 shapes across
its eight execution plans with six sabotages all shown to fail, the
invariance gates (launch, batch, and batch COMPOSITION) run, and the price
harness is wired.

**LEG 11 (commit 144aa5b, 2026-08-23): the v1 device card is bit-identical Apple M4 <-> NVIDIA H100, 60 stages, judged by `tools/e3_round_judge.sh` section 7; the FAST cards happen to agree too. AMD MI325X is OWED (that leg was not run), so the three-vendor completion sentence is not yet earned.** Everything above that is not the H100 column is construction plus one Apple device's gates. `tools/gemm_column_invariance.sh` compiles three
vendor COLUMNS onto ONE backend, which catches a source that reads its
vendor's constants and cannot catch contraction, denormal policy or `sqrt`
rounding — those are the BACKEND's and not the column's. The standing proof
that this distinction is not pedantic is NVIDIA's `sqrt`, which is not
correctly rounded on 180,714 of 2^20 patterns, 176,577 of them on normals: a
defect only its own silicon shows. Until `gemm/E1G_RUNBOOK.md`'s leg runs,
the completion sentence in the charter has NOT been earned.

This ordering was deliberate and is worth keeping in view when reading the
file table below: clause 5 says the distinguishing gates come BEFORE
optimizing, and a kernel written against an unreviewed oracle is what the
charter forbids. So 1 and 2a were finished and green before a line of 2b was
written.

    # Phases 0-1, host only, no GPU:
    pixi run mojo run -I . gemm/mojo_only/gemm_oracle_check.mojo

    # both modes, the way the other identity gates do it:
    tools/with_build_lock.sh     pixi run mojo run -I . gemm/mojo_only/gemm_oracle_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . gemm/mojo_only/gemm_oracle_check.mojo

    # Phase 2b, THE DEVICE GATES — needs a GPU and takes the build lock:
    tools/with_identical_mode.sh pixi run mojo run -I . gemm/mojo_only/gemm_device_check.mojo

    # Phase 3, the identity card and the three-column sweep:
    tools/gemm_card.sh oracle                             # the reference card
    MOJOLEARN_GEMM_CARD_HOST_CAP=1 tools/gemm_card.sh device   # the kernel's, same m/n cap
    /usr/bin/python3 tools/identity_trace_diff.py <oracle.card> <device.card>
    tools/gemm_card.sh compare                            # tree vs chain: must DIVERGE at P > 1
    tools/gemm_column_invariance.sh

    # The fold-ladder card (DEVIATION 533), one hash per tree LEVEL:
    tools/gemm_ladder.sh emit
    tools/gemm_ladder.sh sabotage FOLD_STRIDE

    # Phase 4, the price harness. WIRING, not a published number:
    tools/gemm_price.sh

    # The three-vendor leg. UNRUN; read gemm/E1G_RUNBOOK.md first:
    tools/gemm_remote_leg.sh

No pixi task is registered; the orchestrator registers it.

Last verified on this tree, 2026-08-23 (Apple M4, IDENTICAL): the oracle and
device cards are byte-identical at 60 stages, 0 shapes skipped, and
`tools/identity_trace_diff.py` reports `IDENTICAL` on them
(`bench/results/gemm_card/2026-08-23_101058_verify/`); the fold ladder is
GREEN, 57 records over 5 shapes, and its card diffs clean against the
previous day's card at every level
(`bench/results/gemm_ladder/2026-08-23_101159_ec944d8-dirty/`); the price
harness's device arms build and run in both modes with `_plans_agree`
holding at every swept `P` (no number kept); `gemm_device_check.mojo` is
all green, 6 gates, in IDENTICAL and in FAST, with the `LEAF_ROTATE`
sabotage build failing 3 of 6; `sh -n` passes on `tools/gemm_ladder.sh` and
`tools/gemm_remote_leg.sh`. The ladder's `.bin` sidecars are not committed
(none are, anywhere in `bench/results/`); `MOJOLEARN_GEMM_LADDER_DUMP=fold`
regenerates them.

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
| `IDENTICAL_FP32_CONTRACT.md` | **the deliverable.** Thirteen sections: the v1 version rules, dtypes and the FP32 accumulation requirement, layout, the three orientations, the multiply-add policy, the flush policy and its seven seams, the logical k partition, the evaluation order and the fold tree's node addressing, ragged k and the degenerate `P`, NaN / infinity / signed zero, the exclusions, what is NOT promised, a clause-to-code index, and the clause-6 performance ANALYSIS. |
| `mojo_only/gemm_oracle.mojo` | the contract in code, and BOTH references. `gemm_oracle` (leaves + the fixed balanced tree) is NORMATIVE; `gemm_oracle_serial` (the whole-K ascending chain) is DIAGNOSTIC. Host, scalar, single-threaded, built from `identical_mul_add` and `ftz` so that it IS the contract rather than an opinion about it. The partition count is a PARAMETER, which is what makes the fixtures provable, and the fold tree's structure is a pure function of `P` that Phase 2b addresses from the device. |
| `mojo_only/gemm_oracle_check.mojo` | the adversarial fixtures and their separation proofs. Every one refuses to pass unless the two alternatives produce different bits. |
| `mojo_only/gemm_identical.mojo` | **PHASE 2b: the device kernel.** Eight execution plans over ONE numerical plan -- a flat thread-per-cell arm, five tiled arms that give one block an output tile and all of its `k` leaves (contract 13.5's second workspace escape: no global scratch at any shape), and two split-K arms that materialize named partials into a predetermined workspace. `contract_partition` is the only producer of `(L, P)` in the file and it takes `k`. Host entry point `identical_gemm(ctx, c, a, b, m, n, k, op)`. DEVIATIONS 530-532. |
| `mojo_only/gemm_device_check.mojo` | the device gates: per-cell bits against `gemm_oracle` over 62 shapes, launch invariance over all eight plans, batch invariance across a dispatch boundary AND across batch composition (the same cell inside a 256-cell launch and a 262,144-cell launch; four independent GEMMs through one dirty shared workspace), and the host proof that the kernel's register fold IS `fold_balanced_tree` at every `P` in 1..2049. Six build-define sabotages, each shown to fail. |
| `ported/linalg/contractions.mojo` | `raft/linalg/contractions.cuh`'s `KernelPolicy`, `ColKernelPolicy` and the three float policy families, parameterized. COPY, DO NOT IMPROVE. |
| `PORTED_MAP.tsv`, `UNPORTED.tsv` | upstream -> ours, and what was not ported and why. |
| `E1G_RUNBOOK.md` | the operator's document for the three-vendor leg (`tools/gemm_remote_leg.sh`): preconditions, the dry run, the guard, the result layout, and what a divergence means. **Says at its top that the leg HAS NOT RUN.** DEVIATION 536. |

The lane's drivers live beside the other benches and tools, because the
charter gives this lane `bench/gemm_*` and `tools/gemm_*` and nothing else
outside `gemm/`:

| file | what | deviation |
|---|---|---|
| `bench/gemm_shapes.mojo` | the ONE shape table, every row with provenance, tagged by NAME. Every card, ladder and price run reads it; none may add a row to go green. | -- |
| `bench/gemm_card_main.mojo` | the `gemm.fp32.v1` identity card: per-shape input hashes and one output hash, three arms (`oracle`, `serial`, `device`), shapes capped in `m`/`n` and never in `k`. The device arm records the product straight off the device buffer. | DEVIATION 534 (the device arm) |
| `tools/gemm_card.sh` | drives the card under the right mode, READS THE MODE BACK, and turns two cards into a verdict; `compare` asserts the tree and the chain DISAGREE wherever `P > 1`. | -- |
| `tools/gemm_column_invariance.sh` | compiles the APPLE, NVIDIA and AMD vendor COLUMNS onto one backend and diffs the cards. Catches a source that reads its vendor's constants; cannot catch contraction, denormals or `sqrt` rounding, which are the backend's. | -- |
| `bench/gemm_ladder_main.mojo` | **the fold-ladder card**: one hash per LEVEL of the reduction tree per shape, on the staged split-K plan (the only plan that materializes every node), plus two proofs per shape (top level == emitted output; device == `gemm_oracle` per cell). Turns "`.out` differs" into "level 1 moved, level 0 agrees". **GREEN on Apple**, 57 records over 5 shapes, `bench/results/gemm_ladder/`. | DEVIATION 533 |
| `tools/gemm_ladder.sh` | `emit`, `diff` (localizes a divergence to a level), `sabotage NAME` (clean vs broken, localized; exit 3 when the ladder cannot see the defect). Reads the mode AND the sabotage witness back from the run. | DEVIATION 533 |
| `bench/gemm_price_main.mojo` | the Phase 4 price harness over contract 13.6: fold in isolation, staged vs fused, end to end over the shape table, the leaf-loop seams, workspace, limiter inputs. Host arms and device arms; `_plans_agree` is ASSERTED under IDENTICAL. WIRING, not a published number. | DEVIATION 535 (the device arms) |
| `tools/gemm_price.sh` | runs the harness in both modes for N rounds, reads the mode back, discards a mislabelled leg, reports medians with the A5 contamination banner. | -- |
| `tools/gemm_remote_leg.sh` | **the guarded RunPod leg**: `tools/runpod_guard.sh arm` FIRST, one-hour hard cap, clean-tree refusal, same-commit cross-check, the remote card diffed against the Apple reference. **UNRUN. `sh -n` only.** | DEVIATION 536 |

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

# The seven clause-5 distinctions, and the bits that prove they separate

Clause 5 of Andrew's Phase 2 contract call names seven distinctions the
fixtures must be able to make, *before optimizing*. Each has one. Every
fixture is BUILT TO SEPARATE and the check REFUSES to pass unless the two
alternatives produce different bits — `mojo_only/ieee_arith_check.mojo`
scoring a contracting backend as UNFUSED on 2^20 hashed patterns, of which
ZERO separate the two spellings, is why that is the standard here.

Bit patterns below are from the IDENTICAL run; the FAST run prints the same
ones, because the adversaries are written with explicit spellings rather than
with the mode-gated helpers.

| clause-5 distinction | # | fixture | alternative A | alternative B |
|---|---|---|---|---|
| whole-K serial vs leaf-partitioned balanced | F1 | `A = [2^24, 1 x 1023]`, `B = 1`, `k = 1024`. `2^24` has ulp 2, so `2^24 + 1` is the halfway case round-half-to-even discards: the whole-K chain swallows all 1023 additions. Partitioned, each later leaf sums its own ones to an exact 128 and the tree adds them where they ARE representable | serial `P = 1` → `16777216.0 / 0x4b800000` | contract `P = 8` → `16778112.0 / 0x4b8001c0` |
| (same, through the two named references) | clause 4 | the same fixture through `gemm_oracle` and `gemm_oracle_serial` themselves, in `check_serial_oracle_is_the_one_leaf_case` | `gemm_oracle` (NORMATIVE) → `0x4b8001c0` | `gemm_oracle_serial` (diagnostic) → `0x4b800000` |
| balanced vs ascending serial leaf fold | F5 | partition held at the contract's `L = 128`; a single `2^24` in leaf 0 and a single `1.0` in each of the other seven, zeros elsewhere (`fma(0,1,acc) == acc` exactly). The serial fold swallows all seven ones; the tree pairs them into 2s and 4s first | serial ascending, SUPERSEDED → `16777216.0 / 0x4b800000` | **balanced tree, contract** → `16777222.0 / 0x4b800003` |
| **odd-leaf carry vs zero padding** | **F7** | `k = 300` → `L = 128`, `P = 3` **(odd, and the last leaf is RAGGED at 44 elements)**. Every partial is exactly `-0.0`, produced by `ftz` of a negative subnormal placed at the END of each leaf. The odd tail is carried, or it is paired with a `+0.0` pad | carry, contract → `-0.0 / 0x80000000` | `+0.0` padding → `0.0 / 0x00000000` |
| ↳ and the coincidence that makes it necessary | F7b | the same tree with a NORMAL value in the odd tail. `x + (+0.0) == x` on every finite value, every infinity and every NaN | carry → `16777220.0 / 0x4b800002` | `+0.0` padding → `16777220.0 / 0x4b800002` — **they COINCIDE**, asserted |
| ↳ and the second coincidence | F7c | the `-0.0` fixture with a `-0.0` pad. `x + (-0.0) == x` on EVERY float32, `-0.0` included | carry → `0x80000000` | `-0.0` padding → `0x80000000` — **bitwise equal, asserted**; forbidden as a spelling, not as a value |
| **one balanced topology vs an alternate pairing** | **F8** | `k = 512` → `L = 128, P = 4`, partials `[2^24, 1.0, +0.0, -2^24]`. Adjacent pairs `(0,1),(2,3)`; the stride form pairs `(0,2),(1,3)`, which is `pinned_block_sum`'s shape. They differ only by swapping `c1` and `c2`, so the fixture had to be built with `c1 != c2` | adjacent, contract → `0.0 / 0x00000000` | stride, `pinned_block_sum` → `1.0 / 0x3f800000` |
| fused vs unfused leaf accumulation | F3 | `mojo_only/ieee_arith_check.mojo` ARM 6's construction inside a `k = 2` GEMM: `a0, b0` half-width mantissas, `A = [1, a0]`, `B = [−fl(a0b0), b0]`. Unfused is `q − q = +0.0` EXACTLY; fused is the rounding error. 1,621 of 4,096 patterns separate | fused → `5.9604645e-08 / 0x33800000` | unfused → `0.0 / 0x00000000` |
| FTZ vs gradual underflow, RESULT seam | F4a | `A = [1.5·2⁻¹²⁶, 2⁻¹²⁶]`, `B = [1, −1]`. **Both operands NORMAL**, their difference `2⁻¹²⁷` SUBNORMAL, and it is the answer | FTZ → `0.0 / 0x00000000` | gradual → `5.877472e-39 / 0x00400000` |
| FTZ vs gradual underflow, OPERAND seam | F4b | `A = [2⁻¹²⁷]` (subnormal) `x B = [2²⁰]`. The unflushed product is an ordinary NORMAL number, so the divergence is not confined to the subnormal range | FTZ → `0.0 / 0x00000000` | gradual → `6.162976e-33 / 0x0a000000` |
| **different physical geometries, one logical tree** | **F9** | four unrelated evaluation ORDERS over the same node addresses — level-major, reverse-order multi-pass, coprime-stride multi-pass, and depth-first with an explicit stack — compared at EVERY node address, not only at the root. Run at `P = 4` (F8's fixture), `P = 3` (one carry, F7's `-0.0` partials) and `P = 5` (**two** carries; `P = 7` looks like the second odd case and carries only once) | all four schedules, adjacent tree → `0.0 / 0x00000000` | stride pairing → `1.0 / 0x3f800000` — the SENSITIVITY half, without which four schedules agreeing would prove nothing |

**All seven clause-5 distinctions have a separating fixture. None had to be
recorded as unbuildable.** F9's is the host analogue of a launch geometry —
Phase 2a has no device — and the device version with real grids is Phase 3's.

Beside them, still green and still separating:

| # | distinction | alternative A | alternative B |
|---|---|---|---|
| F2 | one partition count from another (three points, none of them the serial one) | `L = 128, P = 8` → `0x4b8001c0` | `L = 64, P = 16` → `0x4b8001e0`; `L = 256, P = 4` → `0x4b800180` |
| F6a | **the signed-zero bit the v1 fold MOVED.** All 8 partials exactly `-0.0`; the seedless tree preserves the sign, the superseded `+0.0`-seeded serial fold laundered it | balanced tree, no seed (v1) → `-0.0 / 0x80000000` | `+0.0`-seeded serial (SUPERSEDED) → `0.0 / 0x00000000` |
| F6a-P1 | the same at `k = 128`, where `P == 1` and clause 3 says the leaf reaches the output unchanged | no fold addition (v1) → `-0.0 / 0x80000000` | one-term `+0.0` fold (SUPERSEDED) → `0.0 / 0x00000000` |
| F6b | cancellation. `[2^24, 1, …, −2^24, −1, …]`, exact sum ZERO, at the contract's `L = 128` | whole-K serial → `−1.0 / 0xbf800000` | contract `P = 8` → `0.0 / 0x00000000` |

And the checks that are not fixtures in the same sense:
`check_ieee_zero_assumptions` (the IEEE facts section 9.2 rests on, asserted
through a `List` so a constant folder cannot answer for the hardware),
`check_leaf_partition_is_a_pure_function_of_k` (the partition table at ten
`k`, plus the no-empty-leaf invariant), **`check_fold_tree_addressing`** (the
tree's shape at eight `P`: widths against a level-by-level halving, exactly
`P - 1` arithmetic nodes, at most one carry per level, dense addresses),
`check_ported_policy_matches_upstream` (RAFT's policy arithmetic against
`core/gemm.mojo`'s independent flattening), `check_oracle_matches_the_
contract_spelling` (the reach proof for `identical_mul_add` and `ftz`, **in
two arms** — see below), **`check_serial_oracle_is_the_one_leaf_case`** (the
two named references agree at `P == 1` and separate above it), and
`check_orientations_agree` (NN == NT == TN bit for bit, with a wrong-layout
sabotage proving the check can see a difference at all).

## The tree's node addressing, which Phase 2b depends on

    level 0     the P leaf partials, ascending by LOGICAL leaf index
    level d     N_d = ceil(P / 2^d) nodes,  d = 1 .. D
    D           smallest d with N_d == 1;  D = 0 when P == 1

    node(d,q), 2q+1 <  N_{d-1}   ARITHMETIC: ftz(ftz(node(d-1,2q)) + ftz(node(d-1,2q+1)))
    node(d,q), 2q+1 == N_{d-1}   CARRY:      node(d-1,2q), bit for bit
    output = ftz(node(D,0))

`(d, q)` is the normative address. The flat form
`fold_node_addr(P,d,q) = fold_level_base(P,d) + q`, with cell `(i,j)`'s block
at `(i*n+j) * fold_node_total(P)`, is one legal realization. All six
functions are host-computable pure functions of `P` in
`mojo_only/gemm_oracle.mojo`. Measured shapes:

| P | levels | depth D | adds | carries | nodes |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 0 | 1 |
| 2 | 2 | 1 | 1 | 0 | 3 |
| 3 | 3 | 2 | 2 | 1 | 6 |
| 4 | 3 | 2 | 3 | 0 | 7 |
| 5 | 4 | 3 | 4 | **2** | 11 |
| 7 | 4 | 3 | 6 | 1 | 14 |
| 8 | 4 | 3 | 7 | 0 | 15 |
| 1024 | 11 | 10 | 1023 | 0 | 2047 |

## Four fixtures that were WRONG on the first try, recorded because that is the point

All four were caught by the checks' own refusals, not by inspection.

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
3. **The same reach proof was REACHED BUT INERT for the whole of Phase 2a's
   subject.** It ran at `k = 2`, which is `P == 1`, so it exercised the leaf's
   `fma` and `ftz` seams and NOT ONE NODE OF THE FOLD. A gated oracle that
   folded serially would have passed it. Found by asking what a sabotage would
   have to break, not by a failing run — and then confirmed by breaking it: a
   serial `fold_balanced_tree` slips past the `k = 2` arm and is caught by the
   `k = 1024` arm added beside it (`gated oracle = 0x4b800000` against
   `contract = 0x4b800003`). The new arm calls `_separates` on the balanced
   and serial spellings before asserting the agreement.
4. **F9's second odd case was the wrong integer.** The schedule-invariance
   fixture claimed `P = 7` carries twice, at levels 1 and 2. It carries ONCE:
   widths go 7 → 4 → 2 → 1 and only the first predecessor is odd. The smallest
   `P` that carries twice is **5** (5 → 3 → 2 → 1). The check refused itself
   on the count, which is the fixture doing its job on its own author.

---

# Reported, not fixed — defects in files this lane does not own

## 1. `core/gemm.mojo`'s two pinned kernels LAUNDER a `-0.0` at `P == 1`

**AMENDED 2026-08-23 BY PHASE 2b, AND THE SENTENCE THAT WAS WRONG IS DELETED
RATHER THAN LEFT BESIDE ITS CORRECTION.** This entry used to read *"RETRACTED:
`core/gemm.mojo`'s two pinned kernels are RIGHT at `P == 1`"*, and it stated
as fact that `pinned_gemm_nt_kernel` and `pinned_gemv_n_kernel` store
`ftz(acc)` with no fold. **They do not.** As of this checkout they store

    z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))    core/gemm.mojo:164
    z.unsafe_store(i,    ftz(Float32(0.0) + ftz(acc)))    core/gemm.mojo:196

which is the `+0.0`-seeded one-term fold the Phase 1 report PROPOSED, under
the superseded section 9.2(b). Somebody applied it. So the retraction was
written against a file that no longer said what it described, and a reader
following this section would have concluded that no edit was needed when one
is.

**MEASURED, not read.** `OP_NT`, `m = n = 4`, `k = 128` (so `P == 1`), with
`A`'s rows built by `gemm_device_check.mojo::_minus_zero_leaves` so the single
leaf partial is exactly `-0.0`, and `B` all ones:

    gemm_oracle          cell (0,0) = 0x80000000   (-0.0)
    core/gemm.mojo::gemm_nt          = 0x00000000   (+0.0)
    gemm/mojo_only/gemm_identical    = 0x80000000   (-0.0)

**Why that is a divergence from v1.** Contract section 7.3: *"`P == 1`
performs NO fold addition. The single leaf partial reaches the output through
the declared output seam (5g) and through nothing else."* Section 9.2(f)
states the same conclusion in the opposite direction: *"Under v1 that report
is wrong and the proposed change would be the defect."* `x + (+0.0)` is the
identity on every float32 except `-0.0`, where it is `+0.0` — so the
`Float32(0.0) +` is bitwise inert everywhere except on exactly the value
section 9.2 is about, and there it destroys the sign the v1 fold is defined to
preserve.

**THE EXACT CHANGE REQUESTED, and this lane does NOT make it** —
`core/gemm.mojo` is the identity / E2 lane's file (`IDENTICAL_GEMM_PLAN.md`,
LANE BOUNDARY item 1) and rows 27 and 28 are certified on its current bits:

    core/gemm.mojo:164   ftz(Float32(0.0) + ftz(acc))  ->  ftz(acc)
    core/gemm.mojo:196   ftz(Float32(0.0) + ftz(acc))  ->  ftz(acc)

and the comment blocks above each store (`:152-163` and `:193-195`) cite
section 9.2(b) as *"seeding at `+0.0` and always folding is what keeps the
SIGN OF A ZERO from being a function of the partition count"*. **That sentence
is now false and should be deleted, not amended**: v1's tree has no seed, the
sign of a zero is a pure function of the input bits, `k` and the profile
(section 9.2(c)), and `P == 1` is the case where the rule and the
"optimization" coincide.

It is a BIT-MOVING change on a certified path — it moves exactly one value,
`-0.0` to `+0.0`, and only where a leaf accumulator flushes negative subnormal
— so it needs the identity lane's sign-off before a line is edited.

What is unchanged and is a larger statement than this one, contract section
7.6 rather than a defect report: both kernels compute `gemm_oracle_serial` —
the DIAGNOSTIC reference — whenever `k > K_LEAF_MIN`, because they do not
partition `k` at all. Reconciling that is a bit-moving migration of certified
behaviour, it needs the identity lane's agreement, and it is the last step of
Phase 2, not an incidental one.

## 2. `core/gram_splitk.mojo` pins the COUNT where the contract pins the SIZE

`PINNED_GRAM_SPLITK_CHUNKS = 128` (`:209`) is a chunk COUNT, so the chunk
LENGTH is `ceil(k / 128)` and grows without bound: at `k = 4,000,000` each
chunk is a 31,250-term serial fp32 chain. The contract fixes the LEAF SIZE
instead and derives the count, so the same `k` gives 1,024 leaves of 3,907.

Three consequences Phase 2 has to reconcile, none of which is a bug today:

- **Conditioning.** 31,250 sequential roundings per partial against 3,907.
- **Empty chunks.** Pinning the count first means `k < 128` produces empty
  chunks; the kernel handles them explicitly (*"A chunk whose slice starts at
  or past k writes an all-zero partial"*). The contract's derivation order
  makes an empty leaf impossible, so that handling has nothing to handle —
  which is a simplification, not a fix.
- **NEW with the v1 fold: its reduce kernel folds SERIALLY, and the v1
  contract folds with a fixed balanced tree.** Those are different arithmetic
  — fixture F5 is the difference measured — and IDENTITY_PATHS row 27 is
  closed on the serial spelling. So `gram_splitk` is either a SEPARATE PROFILE
  with its own name and certificate, or it migrates to
  `mojolearn.identical.gemm.fp32.v1` and its committed bits move. Contract
  section 7.6 names it; `IDENTICAL_GEMM_PLAN.md`'s LANE BOUNDARY item 2 is the
  instruction to name it now rather than discover it after v1 is published.

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

---

# What Phase 2b inherits, and what it must not do

Phase 2a is host-only by design. The device kernel is a separate brief because
clause 5 says the distinguishing gates come BEFORE optimizing.

**Inherited, ready to use:**

- the versioned contract, `mojolearn.identical.gemm.fp32.v1`;
- `gemm_oracle` (normative) and `gemm_oracle_serial` (diagnostic), named per
  clause 4;
- the fold tree's node addressing as six host-computable pure functions of
  `P`, with its shape asserted at eight `P`;
- eleven separating fixtures covering all seven clause-5 distinctions, each
  with its two bit patterns, each refusing to pass if it stops separating;
- four sabotages already run against them (a serial gated fold, a `+0.0` pad
  for the carry, a stride pairing for the adjacent one, and one broken
  schedule) — all four caught, by the fixture the amendment was written for.

**Forbidden, and each has a fixture that would catch it:**

- sub-partitioning a leaf into per-register or per-lane accumulators and
  adding them at the end (a hidden tree inside one leaf — contract 7.1);
- letting a thread, warp or block index stand in for a logical leaf index
  (the stride pairing — F8);
- padding an odd level to a power of two (F7);
- padding a ragged leaf instead of masking it (contract 8);
- **any dispatch that selects a different fold at a performance boundary.**
  Clause 6: *"The numerical tree cannot change at a performance dispatch
  boundary."* No `if P < 32 use the serial fold`.

**Permitted, and expected:** any tiling, any block and thread count, any
staging, any vectorization, output tiling to bound the workspace, and fusing
several LOGICAL tree levels into one physical block — provided the pairings
and the bits are the oracle's. Contract section 13.4 argues that at
`MAX_LEAVES = 1024` the whole fold fits in one threadgroup on every target, so
the multi-launch cost clause 6 warns about need not arise at any legal `k`.
That is an argument, not a measurement; section 13.6 lists what to measure.

---

# PHASE 2b LANDED, 2026-08-23: the device kernel and its gates

`gemm/mojo_only/gemm_identical.mojo` and `gemm/mojo_only/gemm_device_check.mojo`.
DEVIATIONS 530 (the register-stack realization of the fold tree), 531 (the
fused one-block-owns-all-k arm as the default and the workspace escape it is),
532 (the split-K arms and the level-wise fold over the normative `(d, q)`
addressing).

## The workspace escape, contract 13.5

**Chosen: the SECOND escape, "let one block own an output tile and ALL of its
`k` leaves".** The default arm is a `TM x TN` tiled kernel in which one thread
owns one output cell, walks every one of its `k` products, and folds its own
`P - 1` nodes in registers. `identical_gemm_workspace_floats` returns **0** for
it at every `m`, `n` and `k` -- the 2 GB at `4096 x 4096 x 4096` and the 64 GB
at `k = 4,000,000` that 13.5 tabulates do not arise, because no partial is ever
materialized anywhere.

The split-K arms are kept for 13.5's opposite shape -- small `m n`, enormous
`k`, which is the shipped Gram aspect -- where the fused arm has only `m * n`
threads of parallelism. `choose_gemm_plan` selects between them on `m`, `n` and
`k`, which contract 6.1 explicitly permits ("the EXECUTION plan may look at
`m`, `n`, the device, the occupancy and anything else it likes"). It returns a
plan id and nothing else; every plan then calls `contract_partition(k)`.

## Why launch geometry cannot reach the arithmetic

Three structural facts, not three promises:

1. **One producer of `(L, P)`.** `contract_partition(k)` takes `k`. Every
   kernel receives `L` and `P` as arguments and has no other route to a leaf
   boundary; `block_dim`, `grid_dim`, `block_idx`, `thread_idx`, `m` and `n`
   appear in no expression that reaches `_leaf_bounds` or a tree level.
2. **No float crosses a thread boundary** in the FLAT and TILE plans.
   Threadgroup memory carries OPERANDS -- a bit-exact copy of `A` and `B` --
   never partial sums. There is no cross-thread combination for a block size to
   reorder.
3. **The fold's merge rule is a pure function of the leaf index.** The register
   stack merges when a level is occupied, which is exactly when the contract's
   tree pairs; an unpaired leaf never merges, which is the CARRY realized as no
   instruction at all.

## The gate / sabotage matrix, IDENTICAL, 2026-08-23

Two of the five defects are INVISIBLE to both invariance gates, and that is the
point of the table: a launch-invariant kernel is not the same thing as a
contract-conforming one.

| sabotage | oracle | launch-inv | batch-inv | host fold |
|---|---|---|---|---|
| `LEAF_READS_LAUNCH` (`L` scaled by the block size) | FAIL 50/62 | FAIL 13 | FAIL 6 | ok |
| `FOLD_STRIDE` (`pinned_block_sum`'s pairing) | FAIL 49/62 | FAIL 8 | FAIL 6 | **FAIL** |
| `PAD_PLUS_ZERO` (`+0.0` instead of the carry) | **FAIL 3/62** | ok | ok | ok |
| `FOLD_SERIAL` (the superseded ascending fold) | **FAIL 46/62** | ok | ok | ok |
| `NODE_ORDER` (partial addressed by `block_idx`) | FAIL 26/62 | FAIL 8 | FAIL 8 | ok |
| `LEAF_ROTATE` (leaf start rotated by `block_idx`, every plan; added 2026-08-23 for the composition arms) | FAIL 32/62 | FAIL | **FAIL 12** (all five width arms at both `k`; in the shared-workspace loop the two GEMMs whose shape differs from the solo launch, 144/256 and 213/256 cells) | ok |

`PAD_PLUS_ZERO` failing on only three shapes is the whole reason those three
exist. `x + (+0.0) == x` on every finite float, every infinity and every NaN --
it differs on exactly ONE value, `-0.0` -- so the first version of the oracle
gate ran 42 shapes against this sabotage and **exited 0**. The fixtures that
catch it build every leaf partial to be exactly `-0.0` through `ftz` of a
negative subnormal, which is the only route contract 9.2(a) leaves open, and
`_minus_zero_case` REFUSES ITSELF if the oracle does not come back
`0x80000000`.

## The `ftz` pin, reached on a device

Under FAST, six of the 62 shapes -- the four `-0.0` fixtures and two whose
every product is subnormal -- differ between the host oracle and the device.
Under IDENTICAL all 62 agree. That is a SEPARATING measurement for the flush
pin on Metal: the host does not flush and the hardware does, and `ftz` is what
brings them to one array of bits. It is not the tie that
`check-ieee-arith`'s first `fma` verdict was built on.

---

# BATCH COMPOSITION, 2026-08-23: the gate's two new arms and their sabotage

The charter's launch-invariance bullet names "two batch compositions
(compute the same cell inside a small launch and a large one)". GATE 3
already varied `n` across 4, 64 and 256 and crossed a dispatch boundary; it
did not reach a LARGE launch and it never composed several products. Two
arms were added to `check_device_is_batch_invariant` rather than a new
check, so one gate owns the property:

| arm | what | clean, IDENTICAL | clean, FAST |
|---|---|---|---|
| `n: 4 vs 4096` | cell `(i, j)`, `i < 64, j < 4`, computed inside a `64 x 4` launch (256 cells, SPLITK) and inside a `64 x 4096` launch (262,144 cells, TILE 16x16); every cell of the overlap sits in a different block in the two launches | OK, 256/256 at `k = 4096` and `k = 4097` | OK |
| shared-workspace loop | four independent GEMMs (`64x4`, `64x64`, `130x64`, `64x4`) enqueued back to back with NO synchronize between them through ONE workspace sized for the largest, so it is DIRTY for every launch after the first; each compared per cell to a solo `64 x 4` launch that had a clean workspace | OK, 4 of 4 at both `k` | OK |

**The sabotage, `-D MOJOLEARN_GEMM_SABOTAGE_LEAF_ROTATE=1`** (a sixth
switch in `gemm_identical.mojo`, compiled out in every build that does not
name it): fold position `t` visits logical leaf `(t + block_idx.x) mod P` in
every plan. The leaf arithmetic and the tree are untouched; only WHICH leaf
stands at which tree position depends on the block a cell was scheduled in,
which is exactly what a change of batch composition moves. Built and run,
IDENTICAL:

    FAIL n: 4 vs 4096: 144 of 256 overlapping cells CHANGED BITS with the launch width.
         First at cell (1, 2): 64x4 gave 531.37036/0x4404d7b4   64x4096 gave 531.37054/0x4404d7b7
    OK   loop[0] 64x4   -> SPLITK ... 256 overlapping cells identical to the solo launch
    FAIL loop[1] 64x64  -> SPLITK ... 144 of 256 overlapping cells CHANGED BITS against the solo launch
    FAIL loop[2] 130x64 -> TILE 16x16 ... 144 of 256 overlapping cells CHANGED BITS against the solo launch
    OK   loop[3] 64x4   -> SPLITK ... 256 overlapping cells identical to the solo launch
    check_device_is_batch_invariant [IDENTICAL]: 12 arms found a cell whose BITS depend on how many other cells shared the launch
    gemm_device_check [IDENTICAL]: 3 of 6 gates FAILED (sabotage: LEAF_ROTATE)

`loop[0]` and `loop[3]` PASS under the sabotage and that is the honest
reading, not a gap: they are the solo launch's own shape, a rotation that is
a pure function of the block index reproduces itself at the same geometry,
and it is why the loop carries more than one shape. The oracle gate fails
32 of 62 shapes under it (every `P > 1` shape whose cells reach a second
block) and the launch-invariance gate fails as well.

**Restored** by building without the define: all six gates green in
IDENTICAL and in FAST, and the device card emitted after the kernel edit is
byte-identical to the oracle card
(`bench/results/gemm_card/2026-08-23_101058_verify/device_after_leaf_rotate_edit.card`
against `oracle.card`), so the default build's bits did not move. **No
deviation number was spent**: the kernel's shipped arithmetic is unchanged
and the switch is dead code unless named.

## The fold ladder's own sabotage, run the same day

`tools/gemm_ladder.sh sabotage FOLD_STRIDE`
(`bench/results/gemm_ladder/2026-08-23_101329_5c6932b-dirty/`): the clean
card and the sabotaged card agree at `dims`, at the inputs and at `fold.L00`
on every shape, and diverge from `fold.L01` upward on the four shapes with
`P > 1`:

    llama8b.qkv.t1       levels L00:= L01:X L02:X L03:X L04:X L05:X   out DIFFER
    verdict  LEVEL 01 MOVED AND LEVELS 00..00 ARE IDENTICAL.
    pca.transform.8192x4x4   levels L00:=   out same   (P = 1: no fold addition, contract 7.3)
    5 shape(s) compared; DIVERGENCE LOCALIZED ABOVE

That is the instrument doing the one thing the input/output card cannot:
naming the LEVEL, so a cross-vendor divergence reads "the leaves agree and
the fold moved" rather than "`.out` differs".

---

# ROW TEXT FOR THE IDENTITY LANE

One row, in the ledger's table format, for `IDENTITY_PATHS.md`. This lane
does not edit the ledger; the identity lane assigns the number.

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| (next) | `gemm.fp32.v1`: `C = op(A) . op(B)`, FP32, NN/NT/TN, contiguous row-major, profile `mojolearn.identical.gemm.fp32.v1` (`gemm/IDENTICAL_FP32_CONTRACT.md`); entry `gemm/mojo_only/gemm_identical.mojo::identical_gemm` | RAFT's standalone GEMM is cuBLASLt (`raft/linalg/detail/gemm.hpp`), a closed library: its k-split, its summation order, its FMA contraction and its denormal handling are all the vendor's and change with the GPU, the block count and the batch. Our own FAST arm is MAX `linalg.matmul`, the same class of dependence. | CONSTRUCTION plus Apple's gates (DEVIATIONS 530-536). One numerical plan -- leaves of `L = contract_leaf_size(k)`, `P = contract_leaf_count(k)` from `k` alone, ascending `identical_mul_add` with `ftz` at seven seams inside a leaf, a fixed balanced adjacent-pair tree with bit-for-bit carries across leaves, no seed, `P == 1` folds nothing -- under EIGHT execution plans (FLAT; TILE 16x16/32, 8x32/32, 32x8/16-reversed, 16x16/8-transposed, 4x4/32-transposed; SPLITK; SPLITK_STAGED). Host oracle `gemm_oracle` (normative) and `gemm_oracle_serial` (diagnostic), eleven separating fixtures covering all seven clause-5 distinctions. Device gates: per-cell bits against the oracle at 62 shapes (`k` from 0 to 4,000,000, `P` from 0 to 1024, ragged and odd `P`, `-0.0` leaf partials); launch invariance over all eight plans; batch invariance across a dispatch boundary and across composition (`n` 4 vs 4096; four GEMMs through one dirty shared workspace). SIX sabotages each shown to fail (`LEAF_READS_LAUNCH`, `FOLD_STRIDE`, `PAD_PLUS_ZERO`, `FOLD_SERIAL`, `NODE_ORDER`, `LEAF_ROTATE`). The fold-ladder card (DEVIATION 533) hashes every tree level per shape and localizes `FOLD_STRIDE` to level 01 with level 00 identical. The identity card's oracle and device arms are byte-identical at 60 stages on Apple. Price harness wired, no number published. | **CLOSED ON APPLE ONLY.** The three-vendor run is OWED: `gemm/E1G_RUNBOOK.md` and `tools/gemm_remote_leg.sh` (DEVIATION 536) exist and have not run. "Bit-identical across GPUs" is not a measured sentence for this path until it does. |

