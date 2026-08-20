# Porting rules

Andrew, 2026-08-19. These are binding. The exercise is **take an incumbent
library's code and its algorithms and port them to Mojo.** Nothing else.

## 0a. Where their source is

Every rule below says "read their file". These are the files. Clone them if
the directory is missing; a session without them is a session guessing.

| upstream | checkout | pin | sections it governs |
|---|---|---|---|
| CatBoost | `/private/tmp/catboost-src` | `54a8143a` | root (boosting) |
| cuVS | `/Users/andrewhendel/CascadeProjects/upstream/cuvs` | `94c2819` | `cluster/`, `neighbors/` |
| cuML | `/Users/andrewhendel/CascadeProjects/upstream/cuml` | `00094f7` | `dbscan/`, `decomposition/`, `glm/` |
| RAFT | `/Users/andrewhendel/CascadeProjects/upstream/raft` | `661a3b8` | primitives under all of the above |

The three RAPIDS checkouts were cloned on 2026-08-19 and **did not exist before
that date.** Every port written before then was written from a recollection of
their code rather than from their code, and the DBSCAN neighborhood step is the
proof: it was recorded as "ported the SHAPE their runner depends on" when their
actual kernel is fused, unexpanded, and materializes nothing. Treat any port
predating the clone as unverified against its own upstream.

Clone recipe (blobless, shallow, minutes not hours):

    git clone --filter=blob:none --depth 1 --single-branch \
      --branch branch-25.08 https://github.com/rapidsai/<repo>.git <repo>

## 0b. The charter

**COPY. DO NOT IMPROVE.**

We are not designing these algorithms. CatBoost, cuML and cuVS already did,
on GPUs, with people who measured. Every design question has an answer already
sitting in one of the checkouts above and the answer is whatever they wrote.

## 0b-i. FOLLOW THEIR DISPATCH

Andrew, 2026-08-19 (evening). **Port the path their library ACTUALLY TAKES for
the parameters in question.** Not a neighbouring function. Not the one that is
easier to reach. Not the general case, when their dispatch sends these
parameters somewhere else.

This replaces the old two-rule scheme, whose second rule ("where the incumbent
calls a VENDOR primitive -- cuBLAS, cuSOLVER, CUB, Thrust -- call OURS") is
**DELETED**. Do not cite it again.

What survives is one narrow exception: where the path their dispatch actually
takes calls a **CLOSED** library we cannot read or port -- cuBLAS, cuSOLVER --
call the MAX equivalent, because there is nothing to port. **CUB and Thrust are
OPEN.** Their kernels are readable, so they are PORT candidates.

**Why the old rule died: A DEVICE-WIDE VENDOR CALL CANNOT BE FUSED.** It reads
its input from memory and writes its output to memory, by construction. Standing
one in for a step that belongs INSIDE a kernel freezes the unfused structure
permanently, and there is no way back from it.

The measured case. Our k-NN substituted `linalg.matmul` for the distance step
and `nn.topk` for the selection, both correctly under the old rule. The distance
matrix therefore had to be materialized, so the selector had to read it back,
so ~23 GB of traffic is moved to perform 51.2 GFLOP: a job with a ~13 ms compute
floor takes 306 ms. Meanwhile cuVS's dispatch for those exact parameters
(k<=64, row-major, L2 -- `knn_brute_force.cuh:443`) does not call
`tiled_brute_force_knn` at all. It calls `fusedL2Knn`, which uses no vendor
primitive: their own Contractions tile kernel with a register-resident
`faiss_select::WarpSelect` queue that never writes a distance. We ported their
FALLBACK and the file's header said so for a month.

**When they fuse, they hand-write. Follow that.**

Before reaching for a MAX primitive, answer in writing:

1. Does their dispatch take this path for these parameters?
2. Is this step standalone in THEIR code too, or does it live inside a kernel?

If either answer is no, port their kernel.

`max.gpu.primitives.block` and `std.gpu.primitives.warp` are NOT covered by any
of this. They are the Mojo spelling of `__syncthreads` and `__shfl_*_sync` --
language-level counterparts to CUDA intrinsics, not a library standing in for an
algorithm. Use them freely.

## 0b-ii. GPU, PLUS THE HOST THE GPU PATH NEEDS. NO CPU PATH.

Andrew, 2026-08-19. **There is no CPU-only implementation of anything here, and
none is wanted.** The product is the GPU path plus whatever host control-plane
work that path requires -- which is exactly what cuVS, cuML and RAFT are: their
C++ is GPU-only, and cuML's CPU story is refusing the work and handing back to
scikit-learn (`dbscan.pyx:261` raises `UnsupportedOnGPU`).

Two things this does NOT mean:

- **Host work is not a "CPU path" and is not something to eliminate.** Their
  control plane runs on the host and reads scalars back; mirroring that is
  correct, and `0c` below applies to it like everything else.
- **A host reference used to CHECK a device answer is not a CPU path.** The
  Float64 host Jacobi and the host-computed k-NN truth are oracles. They stay.

A file in this tree is exactly one of two things:

- `gbdt/` -- a port of a real file of theirs
- `mojo_only/` -- something they never needed

There is no third category of "good idea worth adopting."

## 0c. ASSUME OUR CODE IS BROKEN

Andrew, 2026-08-19. When our code and theirs disagree, **theirs is right.**
When a measurement of ours disagrees with their design, suspect the
measurement. When our code looks like it is already doing what they do, read
their file anyway.

This is not deference for its own sake. It is the record:

| what we believed | what reading their source found |
|---|---|
| `build_necessary_histograms` was ported | its state machine was exactly inverted |
| the histogram writeback was fine | it used the looked-up leaf id, theirs uses the dense one |
| replication was tuned | `replicas_for` was invented; they compute it from occupancy |
| leaf values were a ported Newton step | the sign was inverted against their `der` convention |
| the tree grew to `max_depth` | they STOP when a split repeats |
| the histogram loop was transliterated | they load 4 elements per thread, we loaded 1 |
| a threadgroup barrier was the only option | they sync a warp, and `syncwarp` exists |
| `knn_brute_force.mojo` ported cuVS's brute-force k-NN | it ported their FALLBACK; `knn_brute_force.cuh:443` dispatches k<=64 + row-major + L2 to `fusedL2Knn`, which we do not have |
| the k-NN distance step wanted a vendor GEMM | their default path calls no vendor primitive; it fuses, and keeps the top-k in registers so no distance is ever written |
| DBSCAN's neighborhood was "the SHAPE their runner depends on" | RAFT's `EpsUnexpL2SqNeighborhood` is a fused Contractions kernel using UNEXPANDED L2 that materializes no distances at all |
| k-means tested convergence "on device, exactly as theirs", citing `detail/kmeans.cuh:817-825` | that line range is the SIGNATURE of `kmeans_fit`. Their loop syncs at `:491` and tests on the HOST at `:492`. The kernel was invented and the citation supports nothing |

Eleven now, across two sessions. Every one found by reading their checkout,
none by reasoning about our own code. Every "optimisation" we invented
(`replicas_for`, the widened barrier, the on-device convergence test) was worse
than the thing it replaced.

**The last four all have the same shape and it is worth naming: we read a
function of theirs, ported it faithfully, and never checked whether their
DISPATCH sends our parameters to that function at all.** A faithful port of the
wrong path is invisible -- it compiles, it passes, its docstring cites real line
numbers -- and it cost a measured 20x. See `0b-i`.

**So the default is: our version is wrong until their file says otherwise.**
The same applies to cuML for `cluster/`. If you cannot find their file, that
is the work -- not a licence to design.

## 1. Read their source, not our notes

Our notes have been wrong about our own code four times in one day, and our
instruments failed three times. Their source has been wrong zero times.

Before changing anything, open the file in their tree. Cite it by path and
line in the commit and in the code. If a claim in our docs is falsified by
what you read, **delete the false sentence in the same commit** -- do not
annotate it.

## 2. The control plane is code too, and it gets ported like everything else

`catboost/cuda/cuda_lib/` is 57 headers of scheduler and it is as much a part
of CatBoost as the histogram kernels. It is ported into `gbdt/gpu_lib/`.

**If they do something on the GPU in the control plane, we do it on the GPU.**
If they keep a decision on the device so the host never learns it, we keep it
on the device. If they pass a value as a kernel argument, we pass it as a
kernel argument. The host/device split is part of the algorithm, not an
implementation detail we get to re-decide.

Corollary, learned the expensive way: every place our driver did host
arithmetic that theirs does on the device cost a round trip. Nine drains per
level became two by DELETING our inventions, not by optimizing them.

## 3. An unported file is visible; a MIS-ported file is not

`build_necessary_histograms` sat in this tree fully written, commented,
tested by a probe, and **with its state machine exactly backwards**. It
treated `PreviousPath` as "needs nothing" and paired up `Zeroes` leaves,
where theirs does the precise opposite. Nothing caught it because nothing
called it.

So:

- **Port the state machine by transcribing their branches in their order.**
  Do not paraphrase it from the comments. The comments describe intent; the
  branches are the algorithm.
- **A ported file that no caller reaches is not done.** Track it in
  `UNWIRED.md` and treat wiring it as part of the port, not a follow-up.
- **Cite the line range of the loop you transcribed** so a reviewer can diff
  branch for branch.

## 4. Work around the toolchain, never around the algorithm

Mojo and Metal will refuse things CUDA allows. Known so far:

| wall | workaround |
|---|---|
| no dynamic trait objects | tagged union, which is what their worker switches on anyway |
| `ctx.stream()` raises on Metal | one queue; handles still handed out, over-ordering is safe |
| whole-struct load in a kernel kills the Metal compiler | read the fields through the pointer |
| kernel cannot write an `enqueue_create_host_buffer` | explicit copy; `map_to_host` measured 2x slower |
| `enqueue_function` refuses derived pointers as aliasing | match their signature, which passes one struct pointer |

Every one of these changes HOW we say it, never WHAT is said. If a workaround
would change the algorithm, it is not a workaround, it is a fork, and it needs
Andrew.

Each one goes in a `DEVIATION BLOCK` banner in the file, with the measurement
that established it.

## 5. Deviations are declared, in the file, with a number

A `DEVIATION BLOCK` states what theirs does, what ours does, and the measured
reason. "Slower" and "faster" without a number are not reasons. An
undocumented departure is a bug even when it works.

## 6. Their names

Classes keep their CatBoost names: `TCudaManager`, `TPointsSubsets`, `TLeaf`,
`TCFeature`, `TSplitPointsContext`. Directories may be renamed when the name
is a lie (`cuda_lib` -> `gpu_lib`, because none of it is CUDA), and the
rename is recorded in `PORTED_MAP.tsv`.

The symbol is the diff surface. Keep it greppable in their tree.

## 7. Measurement rules that survive from before

- Only arms interleaved inside ONE process compare. This box drifts 2-3x
  across time windows.
- A digest cannot tell a working change from a no-op. Sabotage the path and
  watch the check move before trusting a bit-identical change.
- A check whose expected value is the same in every cell verifies the total
  and nothing about placement. Plant scattered values, compare per cell.

## 8. A NON-DEFAULT PATH IS AN UNCHECKED PATH

Andrew, 2026-08-19 (evening). Rule 3 says a ported file no caller reaches is
not done. **This is the case rule 3 misses**: the file HAS a caller, it is not
in `UNWIRED.md`, and the suite is green -- because every check runs the DEFAULT
side of the switch and nothing runs the other.

The measured case. `ball_cover` shipped opt-in behind `eps_nn_method`, mirroring
cuML's `BRUTE_FORCE` default (`dbscan.hpp:74`). It passed set equality against a
host brute force at five configurations, with two sabotages proving both prunes
were reached. It was ALSO passing the whole dataset as the query on every batch
instead of the batch's rows, where `algo.cuh:131` and `:146` both build the
query view as `data.x + start_vertex_id * k`. 412 of 612 labels were wrong at
five batches. `check_dbscan_batching_agrees` already existed and was already
green, because with RBC opt-in it exercised brute force. **Flipping the default
is what ran the check, and the check failed on the first try.**

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
  632.7) -- and it had already been given two plausible hardware explanations
  before anyone ran the check. **The explanations were fluent and both wrong.**
  What identified the bug was noticing that a curve did something no hardware
  does, not reasoning about which hardware effect it was.
- **The benchmark prints which path it took, beside the timing.** A harness
  that cannot name the kernel it ran can publish a number about a different
  one. That is how every k-NN figure before `0b-i` was taken.
- **A switch that outlives its measurement is a defect, not untidiness.** Once
  one side is measured better and provably identical in output, it becomes the
  default in the SAME session. Leaving it opt-in cannot protect a user -- the
  outputs match -- and it does keep one side of itself unchecked, which is the
  whole failure above.

Rule 7's sabotage requirement composes with this: sabotaging the default path
proves nothing about the other one. **Reach is per-branch.**
