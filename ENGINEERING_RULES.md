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

## 0c. Assume our code is broken

When our code and a well-tested reference disagree, start from the assumption
that ours is wrong. When a measurement disagrees with a design that a lot of
people measured, suspect the measurement. When our code looks like it is
already doing the right thing, check it anyway.

This is not deference. It is the record of what checking actually found:

| what we believed | what checking found |
|---|---|
| `build_necessary_histograms` was correct | its state machine was exactly inverted |
| the histogram writeback was fine | it used the looked-up leaf id where the dense one is required |
| replication was tuned | `replicas_for` was invented; it follows from occupancy |
| leaf values were a correct Newton step | the sign was inverted against the `der` convention |
| the tree grew to `max_depth` | growth stops when a split repeats |
| the histogram loop was right | it loaded 1 element per thread where 4 is the shape |
| a threadgroup barrier was the only option | a warp sync is enough, and `syncwarp` exists |
| `knn_brute_force.mojo` matched the brute-force k-NN | it matched the fallback; `knn_brute_force.cuh:443` sends k<=64 + row-major + L2 to `fusedL2Knn` |
| the k-NN distance step wanted a vendor GEMM | the fused path calls no vendor primitive and keeps the top-k in registers, so no distance is ever written |
| DBSCAN's neighborhood was "the shape the runner depends on" | `EpsUnexpL2SqNeighborhood` is a fused kernel using unexpanded L2 that materializes no distances at all |
| k-means tested convergence on device, citing `detail/kmeans.cuh:817-825` | that line range is a function signature. The loop syncs at `:491` and tests on the host at `:492`. The kernel was invented and the citation supported nothing |

Eleven, across two sessions. Every one found by checking against something,
none by reasoning about our own code. Every optimization invented in that
window (`replicas_for`, the widened barrier, the on-device convergence test)
was worse than the thing it replaced.

**The last four all have the same shape and it is worth naming: a function was
checked faithfully, and nobody checked whether the dispatch sends our
parameters to that function at all.** A check against the wrong path is
invisible. It compiles, it passes, its docstring cites real line numbers, and
it cost a measured 20x. See `0b-i`.

## 1. Read the source, not our notes

Our notes have been wrong about our own code four times in one day and our
instruments have failed three times. If a claim in our docs is falsified by
what a file says, **delete the false sentence in the same commit.** Do not
annotate it.

## 2. The control plane is code too

`gbdt/gpu_lib/` is 57 headers of scheduler and it is as much a part of the
learner as the histogram kernels.

**Where a decision belongs on the GPU, it stays on the GPU.** If a decision is
kept on the device so the host never learns it, keep it there. If a value is a
kernel argument, pass it as a kernel argument. The host/device split is part of
the algorithm, not an implementation detail to re-decide casually.

Learned the expensive way: every place the driver did host arithmetic that
belongs on the device cost a round trip. Nine drains per level became two by
deleting inventions, not by optimizing them.

## 3. A missing file is visible; a wrong file is not

`build_necessary_histograms` sat in this tree fully written, commented, tested
by a probe, and **with its state machine exactly backwards**. Nothing caught
it because nothing called it.

So:

- **Transcribe a state machine branch by branch, in order.** Do not paraphrase
  it from the comments. The comments describe intent; the branches are the
  algorithm.
- **A file that no caller reaches is not done.** Track it in
  `archive/plans/UNWIRED.md` and treat wiring it as part of the work.
- **Cite the line range of any loop checked against a reference** so a reviewer
  can diff it.
- The other failure mode is a capability that is silently absent. A named
  refusal beats a missing symbol. Record it in `NOT_IMPLEMENTED.tsv`.

## 4. Work around the toolchain, never around the algorithm

Mojo and Metal will refuse things CUDA allows. Known so far:

| wall | workaround |
|---|---|
| no dynamic trait objects | tagged union, which is what a worker switches on anyway |
| `ctx.stream()` raises on Metal | one queue; handles still handed out, over-ordering is safe |
| whole-struct load in a kernel kills the Metal compiler | read the fields through the pointer |
| kernel cannot write an `enqueue_create_host_buffer` | explicit copy; `map_to_host` measured 2x slower |
| `enqueue_function` refuses derived pointers as aliasing | pass one struct pointer |

Every one of these changes HOW something is said, never WHAT is said. If a
workaround would change the algorithm, it is not a workaround, it is a fork,
and it needs Andrew.

Each one goes in a `DEVIATION BLOCK` banner in the file, with the measurement
that established it.

## 5. Deviations are declared, in the file, with a number

A `DEVIATION BLOCK` states what the alternative does, what ours does, and the
measured reason. "Slower" and "faster" without a number are not reasons. An
undocumented departure is a bug even when it works.

## 6. Names are a diff surface

Where a check compares against a reference implementation, keeping that
implementation's symbol names makes the comparison greppable and is worth more
than a prettier name. `TCudaManager`, `TPointsSubsets`, `TLeaf`, `TCFeature`
and `TSplitPointsContext` are here for that reason. Rename freely where the
name is a lie: `cuda_lib` became `gpu_lib`, because none of it is CUDA.

## 7. Measurement rules

- Andrew, 2026-09-10: **Optimize for large datasets.** Training-speed claims,
  kernel selection and performance-driven defaults need representative
  large-data measurements. Small fixtures are for correctness, smoke checks and
  diagnosis; they are not the optimization target. Include rows, features,
  classes, tree depth and memory pressure when judging scale. Benchmark
  reminders should warn about small workloads without blocking useful small
  checks.

- Only arms interleaved inside ONE process compare. This box drifts 2-3x
  across time windows.
- A digest cannot tell a working change from a no-op. Sabotage the path and
  watch the check move before trusting a bit-identical change.
- A check whose expected value is the same in every cell verifies the total
  and nothing about placement. Plant scattered values, compare per cell.

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
