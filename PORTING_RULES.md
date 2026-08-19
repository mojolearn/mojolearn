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

A file in this tree is exactly one of two things:

- `ported/` -- a port of a real file of theirs
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

Seven, in one session. Every one found by reading their checkout, none by
reasoning about our own code. The two "optimisations" we invented
(`replicas_for`, the widened barrier) were both worse than the thing they
replaced.

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
of CatBoost as the histogram kernels. It is ported into `ported/gpu_lib/`.

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
