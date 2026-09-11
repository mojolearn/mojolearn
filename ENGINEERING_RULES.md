# Engineering rules

These are binding. They are the rules this repository learned the expensive
way, and every one of them has a measured failure behind it.

This file was called `ENGINEERING_RULES.md` until 2026-09-10. It opened on
2026-08-19 with a bootstrapping charter that said the exercise was to take an
incumbent library's code and implementation it to Mojo, and that charter is retired. It
was written the day before the reference checkouts were first cloned, before
the numeric contract existed, before the Metal backend, and before any of the
work that the library is actually for. What survives is the discipline. The
rule numbers are unchanged so that the citations throughout the tree still
resolve.

## 0a. The reference checkouts

Several rules below say "check it against theirs". These are the checkouts
that answer that, and they exist for one purpose: a published implementation
of a well-studied algorithm is a cheap oracle for whether our answer is right.
Clone them if the directory is missing.

| reference | checkout | pin | sections it informs |
|---|---|---|---|
| CatBoost | `/private/tmp/catboost-src` | `54a8143a` | boosting |
| cuVS | `/Users/andrewhendel/CascadeProjects/upstream/cuvs` | `94c2819` | `cluster/`, `neighbors/` |
| cuML | `/Users/andrewhendel/CascadeProjects/upstream/cuml` | `00094f7` | `dbscan/`, `decomposition/`, `glm/` |
| RAFT | `/Users/andrewhendel/CascadeProjects/upstream/raft` | `661a3b8` | primitives under all of the above |

Clone recipe, blobless and shallow:

    git clone --filter=blob:none --depth 1 --single-branch \
      --branch branch-25.08 https://github.com/rapidsai/<repo>.git <repo>

Nothing from these trees is copied into this repository, and nothing in this
repository is generated from them. They are read.

## 0b. Do not invent where a settled answer exists

The invention budget belongs to the numeric contract, the Metal backend, the
identity ladder and the performance work. It does not belong to re-deriving a
histogram layout or a k-means initialization at two in the morning, and the
record is unambiguous that inventing one mid-task produces something worse
than the settled formulation it replaced.

So when a question is a solved algorithm question, answer it from the
literature or from a reference implementation that a lot of people have
measured, and move on. When a question is a contract question, a portability
question, or a performance question, that is our question and the reference
has no opinion worth having, because none of those libraries was ever asked to
run anywhere but CUDA.

The corollary is not deference. Once a design is understood, beating it is
ordinary work here and it happens. Rule 8's last clause is the discipline that
governs it: a measured, bit-identical win flips the default in the same
session.

## 0b-i. Follow the dispatch that the parameters actually take

When checking our behavior against a reference, check against the path their
dispatch takes **for the parameters in question**. Not a neighboring function.
Not the one that is easier to read. Not the general case, when their dispatch
sends these parameters somewhere else.

The measured case. Our k-NN used `linalg.matmul` for the distance step and
`nn.topk` for the selection. The distance matrix therefore had to be
materialized, so the selector had to read it back, so ~23 GB of traffic moved
to perform 51.2 GFLOP: a job with a ~13 ms compute floor took 306 ms. cuVS's
dispatch for those exact parameters (k<=64, row-major, L2,
`knn_brute_force.cuh:443`) does not go to `tiled_brute_force_knn` at all. It
goes to `fusedL2Knn`, which keeps the selection queue in registers and never
writes a distance. We had checked ourselves against their fallback and the
file's header said so for a month.

**A device-wide vendor call cannot be fused.** It reads its input from memory
and writes its output to memory, by construction. Standing one in for a step
that belongs inside a kernel freezes the unfused structure permanently and
there is no way back from it. Where a closed library (cuBLAS, cuSOLVER) is the
only thing on the other side, there is nothing to compare against, and the MAX
equivalent is the fallback. CUB and Thrust are open and readable, so a
question about them has an answer.

`max.gpu.primitives.block` and `std.gpu.primitives.warp` are not covered by
any of this. They are the Mojo spelling of `__syncthreads` and `__shfl_*_sync`.
Use them freely.

## 0b-ii. GPU, plus the host the GPU path needs. No CPU path.

**There is no CPU-only implementation of anything here and none is wanted.**
The product is the GPU path plus whatever host control-plane work that path
requires.

Two things this does not mean:

- **Host work is not a "CPU path" and is not something to eliminate.** The
  control plane runs on the host and reads scalars back. That is correct.
- **A host reference used to CHECK a device answer is not a CPU path.** The
  Float64 host Jacobi and the host-computed k-NN truth are oracles. They stay.

A file in this tree is exactly one of two things: an implementation, or a
`checks/` file that gates one.

## 0b-iii. A tier we will not benchmark is a tier we do not ship

DEVIATION 2490, 2026-09-10. Three tiers used to mean three tiers everywhere.
They no longer do, and the rule is one sentence:

**Fast and deterministic ship for the three tree lanes, `gbdt`, `rf` and
`trees`. Every other binding ships IDENTICAL only.**

Cross-vendor bitwise identity is the product. It is the default tier, it is
the thing no other library sells on any platform, and it needs no speed
argument. A FAST binary is a different kind of thing: it is a claim, and a
claim we cannot publish is a liability. So a fast tier ships only where it has
a measured win over the opponent's own CPU, and today that is trees on Apple
silicon. Tree fitting calls no BLAS anywhere (histogram building and split
finding are scatter-gather over integers), so the opponent gets nothing from
Accelerate's AMX coprocessor, and ExtraTrees measured 1.25-1.61x scikit-learn
on ALL TEN cores at covtype 581k.

Nothing else has that argument. Measured on an M4, 4 performance cores,
10-core GPU, 120 GB/s: Accelerate reaches 1438 GFLOP/s of fp32 GEMM on four
cores because macOS >= 14 arm64 NumPy links it, so scikit-learn gets AMX free
on every BLAS call, and one CPU thread already takes 88 of the 120 GB/s the
GPU must share with it. The GPU's whole margin over its own CPU is about 2.5x
on compute and 1.0x on bandwidth. k-means, k-NN, PCA, SVD, OLS, UMAP, GP and
ARIMA are precisely the families whose inner loop IS a BLAS call, so that
margin is spent against AMX and there is nothing left to win. `SVC` and `SVR`
could beat libsvm's single thread, and they still ship identical only: two
families with a fast tier that are not "trees" is a rule a user has to look
up, and one rule beats two wins. The neural lanes gate every fused kernel on
the identical contract, so their lower tiers were SLOWER than the default
(DEVIATION 2300 is the cost, a `k_last` failure that lived only in a tier
nobody ran).

So the rule, stated so it binds the next family too:

- **Adding a tier is adding a claim.** Ship `fast` for a family only when we
  intend to measure it against that family's real opponent and publish the
  result, AND the rule stays one sentence a user can hold. If we would not
  run the comparison, we do not build the binary.
- **It is an allowlist.** `_backend.py`'s `_TIERED` names the three tree
  bindings; `_IDENTICAL_ONLY` is everything else in `_MODULES`. A binding
  added tomorrow is identical only until someone measures a win and adds it,
  which is the safe direction to be wrong in. The pack and build lists
  (`packaging/linux/pack_wheel.py`, `packaging/linux/build_sets.sh`,
  `packaging/macos/build_release_wheel.sh`) carry the same pair and
  `packaging/check_ext_lists.py` holds them to it.
- **Withdrawing a tier is a refusal, never a silent absence.** Rule 8 is why:
  a spelling that survives with nothing exercising it is an unchecked path.
  Every identical-only `bindings/build_*.sh` exits 2 with the reason,
  `_backend.binding()` raises a sentence a caller can act on, and both
  release smokes (`packaging/macos/smoke.py`, `packaging/linux/smoke.py`)
  assert BOTH arms per binding: it must launch under identical and must
  refuse BY NAME under fast and deterministic. One that answers there is the
  failure.
- **Shared host helpers resolve from the identical binary, whatever tier the
  caller runs.** `_buffer.py`'s casts and finiteness check live in the base
  `_mojolearn` binding, which now exists in one tier; resolving them through
  the running tier refused every fast tree fit at its first input conversion
  the day the rule landed. Anything a tree lane shares with an identical-only
  lane is loaded the same way.

- **The only number is OUR IDENTICAL arm against THE OPPONENT'S FAST arm**,
  with exactly one exception: the three tree lanes, where `fast` is a
  shipped tier and MAY be timed on Apple silicon
  (`bench/speed/forest_speed_arm.py`). Everywhere else, timing a `fast` or
  `deterministic` arm is timing a binary that does not ship -- see 0b-iii.
  Do not build one to benchmark it.
- **Never compare our fast arm to our identical arm and call it a result.**
  That ratio is the COST OF IDENTITY, an internal number for deciding what a
  pin is worth. It says nothing about whether we beat anyone, and it must
  never reach a claim, a paper table or a default decision.
- **Name the opponent's threading and its BLAS before quoting a ratio.**
  `SVC`, `SVR`, `KernelDensity` and `GaussianProcessRegressor` have no
  `n_jobs` at all; `KMeans` and `PCA` have none either but get threaded BLAS
  underneath. On Apple silicon that BLAS is Accelerate, and Accelerate is
  AMX. A ratio against a single-threaded opponent and a ratio against AMX are
  not the same measurement and must not sit in one column.

## 8. A non-default path is an unchecked path

Rule 3 says a file no caller reaches is not done. **This is the case rule 3
misses**: the file has a caller, it is not in `archive/plans/UNWIRED.md`, and
the suite is green, because every check runs the DEFAULT side of the switch
and nothing runs the other.

The measured case. `ball_cover` shipped opt-in behind `eps_nn_method`. It
passed set equality against a host brute force at five configurations, with
two sabotages proving both prunes were reached. It was ALSO passing the whole
dataset as the query on every batch instead of the batch's rows. 412 of 612
labels were wrong at five batches. `check_dbscan_batching_agrees` already
existed and was already green, because with RBC opt-in it exercised brute
force. **Flipping the default is what ran the check, and the check failed on
the first try.**

So:

- **Every switch is exercised on BOTH sides, by a named check per side**, with
  the switch set explicitly inside the check. "The suite covers it" is not
  coverage. A parameter that selects a kernel is a parameter the checks
  enumerate.
- **A number taken on a non-default path is provisional until a check has run
  that path.** The first RBC sweep was measured, written up, and re-run,
  because a number taken on a defect is not a number. Its 50,000-row anomaly
  was mostly the defect, not the hardware (0.90x -> 1.06x, and the impossible
  sublinearity 231.7 -> 323.1 for twice the data became 196.3 -> 316.9 ->
  632.7), and it had already been given two plausible hardware explanations
  before anyone ran the check. **The explanations were fluent and both wrong.**
  What identified the bug was noticing that a curve did something no hardware
  does, not reasoning about which hardware effect it was.
- **The benchmark prints which path it took, beside the timing.** A harness
  that cannot name the kernel it ran can publish a number about a different
  one.
- **A switch that outlives its measurement is a defect, not untidiness.** Once
  one side is measured better and provably identical in output, it becomes the
  default in the SAME session. Leaving it opt-in cannot protect a user, since
  the outputs match, and it does keep one side of itself unchecked, which is
  the whole failure above.

Rule 7's sabotage requirement composes with this: sabotaging the default path
proves nothing about the other one. **Reach is per-branch.**

## 9. Two datasets, different in kind, before a number is a result

Andrew, 2026-09-11, after the RandomForest board read 0.33 of cuML's time on
one dataset. A ratio measured on one dataset is a fact about that dataset. A
change whose win comes from the shape of the data (deep trees on a noisy
binary target make many tiny pure leaves, DEVIATION 2502) will read
differently on data of another shape, and a board that never shows the other
shape cannot tell an optimization from a fit to the fixture.

**Trees (gbdt, rf, trees): every speed or quality claim runs on TWO large
datasets that differ in KIND, one low-feature and one high-feature, both at
or above 1,000,000 rows.** Today those are HIGGS (28 features, binary; the
low-feature set) and a high-feature set at or above 1,000,000 rows to be
fixed in `bench/OPPONENT_REFERENCE.md` when its opponent rows are first
measured (Bosch, 1.18M x 968, is the real candidate in the gbm-bench family;
the harness's `synthclf` at 800k x 100 is a fallback, not the rule). Two
datasets, not more: several sizes of one dataset are one dataset (the 1M/2M/5M
rungs of HIGGS are the SAME kind), and a third kind buys less than it costs.
A win that shows on one of the two and not the other is reported as exactly
that, and is not flipped as a default until it is understood.

**Neural (byte-LM, Mamba, transformer): every training or inference claim
runs on TWO different kinds of data**, two corpora or two generating
distributions, not two seeds of one corpus and not two lengths of one file.
Same reason, same shape of the rule.

The 1,000,000-row floor for tree timing (2026-09-01) stands underneath this
rule; this one adds the second kind, and removes the size sweep as a
substitute for it.

**Non-tree classical (kNN, kmeans, ols, pca, iforest and the rest): the same
rule at the lane's large shape.** Andrew, 2026-09-11, on the kNN selection
default: "use 2 different reasonably large datasets since the large case is
the one we should be optimizing for, but not too large because I don't want
to waste that much time, and 2 of them that are different so they
generalize." Reasonably large means the shape at which the kernel, not the
launch or the transfer, is the cost (kNN today: 400,000 x 32 index with
4,000 queries; kmeans, ols and pca: 4,000,000 x 32), and no larger: a bigger
shape buys rental time, not evidence. Different in kind means one real
dataset beside one generator, or two real sets of different structure. Two
generators of the same shape are ONE kind: the kNN gate's `dyadic` and
`large` fixtures are both 400,000 x 32 synthetic blocks, so a selection win
timed on both has been timed on one kind. A win timed on one kind is
provisional; it may stay flipped for the vendor it was measured on while
the second kind is owed, and the brief says so.

**Opponents under this rule.** An opponent row is measured ONCE per (GPU,
driver, opponent version, dataset) and cached in `bench/OPPONENT_REFERENCE.md`;
later rounds run ours alone against the cached row. A second dataset kind is
a new tuple, so its opponent row is measured once on its first leg, and
never again after that.
