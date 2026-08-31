# cluster: a port of cuVS k-means

**Same strategy as `gbdt/`, different upstream.** This section mirrors
[rapidsai/cuvs](https://github.com/rapidsai/cuvs) at commit `94c2819`
(branch-25.08, checked out at `~/CascadeProjects/upstream/cuvs`), file
for file, so that "did we port this?" is answered by `ls` rather than by
reading.

## The upstream is cuVS, not RAFT, and that was not always true

k-means and brute-force k-NN used to live in RAFT. They do not any more.
RAFT 26.10 (`9aa17e5`) ships `cpp/include/raft/` with `linalg`, `matrix`,
`random`, `sparse`, `stats`, `solver`, `spectral` and `util`, and **no
`cluster/` and no `neighbors/`**; `fusedL2NN` survives there only in a test
shim. The algorithms moved to cuVS and the primitives they call stayed in
RAFT, which is a clean two-layer split and is the split this section
mirrors:

| layer | upstream | where it lands here |
|---|---|---|
| algorithm | cuVS `cpp/src/cluster/`, `cpp/src/distance/` | `gbdt/` |
| primitive | RAFT `linalg`, `matrix` | `checks/` |

A RAFT call is not itself a `gbdt/` file, because RAFT is a general library
this tree does not mirror file for file. **It is still readable, so it is a
PORT candidate and not a substitution candidate**: RAFT, CUB and Thrust are
open source and their kernels get transliterated. `checks/` files name the
RAFT call they came from and cite the RAFT kernel they were read from --
`reduce_by_key.mojo` cites `raft/linalg/detail/reduce_rows_by_key.cuh`, not
just `raft::linalg::reduce_rows_by_key`.

Only a CLOSED library with nothing to read (cuBLAS, cuSOLVER) is a reason to
call a MAX equivalent instead. In this section that applies to exactly one
thing: the candidate-cost matrix product in k-means++, and their own dispatch
materializes that matrix too (`detail/kmeans.cuh:195-196`), so nothing is
being unfused by the substitution. **The Lloyd assignment step calls no vendor
primitive at all** -- it is a transliteration of their SIMT fused kernel, and
that is what took this arm from 450 ms to 175 ms.

## Path mirroring

cuVS has two roots and both are kept, with their constant prefixes dropped
the same way `catboost/cuda/` is dropped on the boosting side:

    cpp/src/            ->  gbdt/
    cpp/include/cuvs/   ->  gbdt/
    python/cuvs/cuvs/   ->  gbdt/python/

So `cpp/src/cluster/detail/kmeans.cuh` is `gbdt/cluster/detail/kmeans.mojo`
and can be diffed against it side by side.

## The one rule still applies here

**COPY. DO NOT IMPROVE.** Same rule, same reason: if this comes out slow we
have to be able to say it is cuVS's design that is slow on this hardware and
not our interpretation of it. Every deviation is numbered in the root
`PORTING.md` (items 14 and up belong to this section) and every unported file
is named in `NOT_IMPLEMENTED.tsv`.

## What runs and what does not

`gbdt/cluster/kmeans.mojo::fit` is wired end to end: norms, the fused
distance-and-argmin kernel, fixed-point cluster accumulation, centroid
finalize with the empty-cluster rule, the centroid-shift reduction, the
host-side stopping rule, and the post-loop inertia pass.

**The default fit runs ONE convergence test, not two.** `inertia_check` is
`false` by default in cuVS (`cpp/include/cuvs/cluster/kmeans.hpp:120`), and
with it off their loop never reduces the cluster cost and never applies the
`delta > 1 - tol` ratio test (`detail/kmeans.cuh:468-489` is entirely inside
that `if`). The centroid shift is the whole default rule. Setting
`inertia_check = True` turns the second test on and costs a second full-length
reduction and a second drain per iteration, which is why theirs is off.

**The test runs on the HOST, in the same iteration.** `detail/kmeans.cuh:491`
is the loop body's one `sync_stream`, `:492` is the test, `:494-497` breaks.
There is no device-side convergence kernel in cuVS and no flag read one
iteration late; an earlier version of this port had both and attributed them
to cuVS with citations that pointed at a function signature.

**cuVS's DEFAULT initialization runs.** `oversampling_factor = 2.0` selects
scalable k-means|| (`initScalableKMeansPlusPlus`,
`detail/kmeans.cuh:568-785`), ported in `gbdt/cluster/detail/kmeans.mojo::
init_scalable_kmeans_plus_plus` with its vendor calls named kernel by kernel
in `checks/scalable_init.mojo` (PORTING.md 47 and 48 price the
randomness and selection mechanisms). Until 2026-08-20 this arm raised
rather than silently substituting classic k-means++; `oversampling_factor =
0` still switches to the classic sequential variant, exactly as their
dispatch at `:910-915` does. `check_scalable_sampling_selection` holds the
sampling round to an exact host replay of the same predicate and sabotages
the candidate costs; `check_scalable_kmeans_plus_plus_init` runs the default
config end to end, run-twice bitwise; `check_scalable_supplement_branch`
starves the rounds to reach the fewer-than-k random-supplement arm.

**It has now been LAUNCHED and it passes.** `cluster/kmeans_main.mojo` builds
and runs `cluster/checks/kmeans_check.mojo`:

    check_reach_by_sabotage OK: centroid_norm moved 384/512 labels;
      x_norm moved 512 distances and 0 labels, which is the predicted shape
    check_kmeans_fit OK: 4/4 centroids matched as a permutation,
      0/512 rows misassigned, inertia 170.703125 vs expected 171.19362
      (rel 0.0029), 2 iterations

The reach evidence is a SABOTAGE and not a digest, and the two sabotages
predict DIFFERENT movements on purpose. Corrupting the centroid norms must
change which centroid wins; perturbing the sample norms must change the
distance and NOT the labels, because a per-row constant cannot move an
argmin. A no-op passes neither. Something that merely "moves" under any
corruption passes the first and fails the second.

**The first version of the second sabotage failed, and the kernel was
right.** It replaced the sample norms instead of offsetting them, which drove
every expanded distance negative; their clamp then flattened all four to
exactly 0.0 and the tie-break handed every row to centroid 0. That is a real
property of their kernel worth knowing: the clamp is safe for the round-off
it exists for and is not order-preserving in general.

`check_kmeans_fit` initializes with `INIT_ARRAY` deliberately, because a
check that also depends on the draw cannot say which half failed;
`check_kmeans_plus_plus_init` covers the k-means++ draw separately and also
passes.

## Validation

Against scikit-learn, on the same data, never against our own previous
output. `cluster/tools/sklearn_reference.py` writes the fixture and the
expected values.

**Compare inertia, not centroid identity.** cuVS keeps the old centroid for a
cluster that goes empty; scikit-learn relocates it onto the furthest point.
Two implementations that disagree there diverge permanently from the same
seed while both remaining correct.

## The finding worth reading before anything else here

cuVS's float32 distance GEMMs do not run in plain float32 on NVIDIA tensor
cores. Their CUTLASS specializations select
`cutlass::arch::OpMultiplyAddFastF32`
(`cpp/src/distance/detail/fused_distance_nn/gemm.h:122` and
`cpp/src/distance/detail/pairwise_distance_gemm.h:75`), which is CUTLASS's
**3xTF32** emulation: three TF32 products summed, not one. Their own
commented-out alternative on the next line says what the other choice would
have been -- `OpMultiplyAdd`, "this runs only 1xTF32 for float inputs" -- and
they did not take it.

Two things follow, and the second is the one that matters:

1. **The precision claim is 3xTF32, not TF32.** 3xTF32 recovers most of
   float32's mantissa; a plain 10-bit TF32 claim overstates the gap and this
   file used to make it, citing a `CUBLAS_COMPUTE_32F_FAST_TF32` constant and
   a file (`unfused_distance_nn.cuh`) that does not exist in cuVS.
2. **Their float32 distance arithmetic is architecture-dependent by
   construction**, because these specializations are `ArchTag = Sm80` tensor-op
   paths and their SIMT kernel is not. No implementation on other hardware
   reproduces it bit for bit, and neither do two NVIDIA generations
   necessarily. See `checks/gemm.mojo`, and the `bitwise-gbdt` tree, where
   this is a second incumbent with a device-dependent number system.

The kernel this tree actually ports is their SIMT one
(`cpp/src/distance/detail/fused_distance_nn/simt_kernel.cuh`), which has no
tensor-op path in it at all.

### The same finding from the other side: OUR vendor products on NVIDIA are 1xTF32 (DEVIATION 529, 2026-08-23)

The H100 leg of E2 failed `check_assignment_arm_dispatch` under FAST with
"fused and unfused min_dist diverge, worst relative 51968000000.0" after
every fused-arm check before it had passed and with the labels agreeing.
The number decodes: the old comparison was `|a - b| / max(|a|, 1e-6)` with
`a` the FUSED value, so 5.2e10 means the fused arm returned ~0 (correct:
the fixture's points sit 100x the jitter from their own centroid and the
expanded form at magnitude 1.3e9 clamps to zero) and the UNFUSED arm
returned ~51968, which is 4e-5 of the magnitude, 400 ulps -- not an fp32
accumulation of 32 terms, a TF32 tensor-core product. The unfused arm is
`core/gemm.mojo::gemm_nt` = MAX 26.5.0's `linalg.matmul`, and on NVIDIA that
is **1xTF32 by default with no compilable opt-out before Blackwell**
(`use_tf32=True` is the default, the cuBLAS fallback hard-codes
`CUBLAS_TF32_TENSOR_OP_MATH`, `use_tf32=False` asserts at compile time
except on SM100; `checks/kernel_matrix.mojo::column_vendor_fp32_matmul_is_tf32`
carries the source lines).

Three consequences, in order of how much they matter:

1. **The fused arm -- the arm the fit ships -- was RIGHT.** It is the SIMT
   kernel above, fp32 on every column, and the diff that failed was
   against the lossy arm. The fix is the judge: both arms are now held to
   a Float64 oracle with a MAGNITUDE-relative budget (fp32 for the fused
   arm everywhere; fp32 on an exact column and the TF32 bound on a lossy
   one for the unfused arm, the line printed naming which), and their
   outputs are poisoned before every launch so an unwritten row reads back
   as poison. `check_assignment_arms_match_oracle` is the well-conditioned
   twin where the distances are resolvable (fp32 arms land ~1e-7 of the
   magnitude; a TF32-rounded self-sabotage lands ~1e-4, rejected by the
   fp32 budget and admitted by the TF32 bound, on this Mac).
2. **Under FAST on NVIDIA, the parts of k-means that DO run through
   `gemm_nt` are TF32-accuracy:** the k-means++ candidate costs
   (`detail/kmeans.mojo`, `gemm_nt` into `candidate_cost_kernel`) and the
   unfused assignment arm if anything ever selects it. The Lloyd assignment
   itself is not -- so on an H100 our k-means distance arithmetic is MORE
   precise than cuVS's 3xTF32 for the assignment and LESS for the seeding.
   Under IDENTICAL neither is reachable: `gemm_nt` is the pinned fp32
   kernel (DEVIATION 526).
3. **The column sims found two more NVIDIA failures waiting in this file
   behind the one the leg hit.** `check_fused_policy_dispatch`'s fixture
   was a constant 8256 rows sized to the M4's 120-block grid cap; on a
   132-SM H100 the cap is larger, the fixture fits under it, and the check
   refused itself with "fixture too small". The row count is now derived
   from the launcher's own cap. And its "fused == unfused" label clause is
   now excused per row where the oracle gap sits inside the lossy arm's
   budget. `checks/gram_splitk_check.mojo` had the same class of
   Apple-only assertion (DEVIATION 540).
