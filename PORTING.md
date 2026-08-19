# Where CUDA does not reach Metal, and what was written instead

Every entry is a place the port could not be literal. The rule is to write
the closest thing and record it here, never to substitute a better idea, so
that if this tree ends up slow we can tell their design from our
interpretation of it.

## 1. Threadgroup memory: 48 KB wanted, 32 KB available

Every CatBoost histogram kernel sizes its shared buffer to **49,152 bytes**:

- one-byte: `BlockSize * 32` floats at `BlockSize = 384`
  (`hist_one_byte.cu:22-24`, `:421`)
- half-byte and binary: `BlockSize * 16` floats at `BlockSize = 768`
  (`point_hist_half_byte_template.cuh:18-20`, `hist_half_byte.cu:72`,
  `hist_binary.cu:86`)

Apple silicon caps threadgroup memory at 32 KB. The buffer size is
`BlockSize * 16` floats, so the block size is what has to give:
**`BlockSize = 512` gives exactly 32,768 bytes** and **`BlockSize = 256`
gives 16 KB** with room for anything else the kernel needs.

Ported at **`BLOCK_SIZE = 512`**, which asks for exactly 32,768 bytes: the
largest block that fits. This is a real deviation and it costs replication:
CatBoost gets `BlockSize / 32 = 24` per-warp copies of the histogram to
reduce contention, and 512 gives 16.

**And it nearly introduced a silent wrong answer.** `Reduce()` writes the
literal 512 in its first stage (`if (threadIdx.x < 512)`,
`point_hist_half_byte_template.cuh:123-133`), which is safe at their
BlockSize of 768 and is NOT safe below it. At a first-cut `BLOCK_SIZE = 256`
stage 1 would have written only slots 0-255 while stage 2 goes on to read up
to `32 * 15 + 7 + 24 = 511`, so half the histogram would have been read as
whatever the scratch happened to hold. The port carries `REDUCE_WIDTH =
min(BLOCK_SIZE, 512)` and an outer slot loop so the stage covers all 512
slots with whatever block size is configured. At 512 it runs once and is
their loop verbatim.

This is the first hazard the port has surfaced that is invisible in the
original: a constant that is only correct in conjunction with a block size we
cannot use.

## 2. There are no warp-level primitives in Mojo 1.0

CatBoost's accumulator is conflict-free by construction rather than by
atomics. `SliceOffset()` hands each 32-lane warp its own 512-float copy and
each group of 8 lanes its own sub-copy, and `AddPoint` rotates which feature
a lane handles with `(threadIdx.x + i) & 7`, so within one iteration no two
lanes of the tile touch the same slot. Between iterations it calls
`tiled_partition<8>(this_thread_block()).sync()` -- an **8-lane barrier**.

Mojo 1.0 exposes only `barrier()`, which is threadgroup-wide. So the port
widens an 8-lane sync to a 256-thread sync. **This is correct but strictly
more expensive**, and it is the single largest known deviation in the port.
Marked at every site as `DEVIATION: tile sync widened to block barrier`.

The one true warp SHUFFLE in the whole path is in the bin prefix scan
(`cub::WarpScan<double>` plus `cub::ShuffleIndex<32>`,
`histogram_utils.cu:381`, `:413`, `:423`). That one is substitutable without
loss: a threadgroup scan computes the same values.

## 3. `float` accumulation, not fixed point

CatBoost accumulates in `float` in shared memory and flushes with a
non-deterministic global `atomicAdd` guarded by `abs(val) > 1e-20f`. Copied
as-is. This makes the port's histograms non-deterministic across runs, which
is THEIR behavior; do not "fix" it to fixed point here.

## 4. `cub::DeviceRadixSort::SortPairs` per leaf

`split_points.cu:658-689` sorts each leaf's index range on the host loop, one
CUB call per leaf, 255 of them for a depth-8 tree. CatBoost's own comments
call this out: `//TODO(noxoomo): cub sucks for this, write proper segmented
version` (`:657`) and `//TODO(noxoomo): for oblivious trees we have overhead
for launching kernel per leaf` (`split_points.cpp:53`). There is no CUB in
Mojo. Port writes a stable 1-bit partition per leaf range, which is what the
sort is being used for.


## 5. Vector loads not ported

CatBoost instantiates a different `TComputeHistogramImpl` per load width and
picks 2 or 4 elements by arch (`point_hist_half_byte_template.cuh:34-41`).
The port takes the `OneElement` specialization only. Scheduling, not numeric:
the same values are added in the same order, fewer at a time.

## 6. `AlignMemoryAccess` peel omitted

It exists to align the vector loads of item 5. At one element per load there
is nothing to align. Required the moment the load width moves above 1.

## 7. THE ONE THAT IS NOT OURS TO CHOOSE: Metal has no float atomic add

CatBoost flushes every histogram with `atomicAdd(dst + fold, val)` on
`float`. **Metal has no floating-point atomic add at all.** Not slower, not
discouraged: the instruction does not exist.

This is the only deviation in this file that a mode cannot express, because
there is no faster alternative to fall back from. `NUMERIC_FAST` is defined
by the float atomic flush, and on Apple that mode's defining row is
unavailable, so `spec_for` FORCES `deterministic_flush` on the apple column
and records `flush_forced_by_vendor` beside it.

Three consequences worth stating plainly:

1. **The port cannot be literal here even in FAST mode on our primary
   target.** Apple accumulates fixed-point `Int32`.
2. **Apple is reproducible by default and identity costs it nothing**, since
   integer addition is associative and Apple's lane width already equals the
   pinned 32. The bit-identical column is paid for by NVIDIA and AMD.
3. Fixed point needs a scale, and the scale must bound every partial sum.
   That is a real piece of work CatBoost never has to do, and it has to be
   ported from nothing.


## 8. The bin prefix scan: no `cub::WarpScan`

`ScanHistogramsImpl` scans with `cub::WarpScan<double>` and
`cub::ShuffleIndex<32>` (`histogram_utils.cu:381`, `:413`, `:423`). Those are
the **only warp shuffles in the entire oblivious path**, so this is the one
place the missing warp primitives bite the algorithm rather than just the
barrier width.

Substituted with a serial scan, one thread per (feature, leaf, stat). Safe
for identity, not free for speed: a prefix sum is order-defined, so a serial
scan and a correct parallel scan agree exactly in exact arithmetic and NOT in
floating point, which is why the scan is a numeric row pinned to one shape
across vendors rather than tuned per vendor.

Cheap here because folds per feature is small (1 for the binary features that
dominate covtype, 255 at the very most) while features are many, and the leaf
and stat axes fill the machine anyway. It would need revisiting on a dataset
of few, very high-cardinality features.

## 9. Mojo kernel signature rules, learned the hard way

Every "it compiles" claim before the first launch probe was
`mojo build --emit=object`, which targets the HOST. A kernel body that
typechecks for the host proves nothing about whether it can be launched, and
three files reported as compiling did not compile at a real use site. The
rules, each one found by a failure:

1. **Scalar kernel parameters must be `Int32`, not `Int`.** An `Int`
   parameter fails `enqueue_function` instantiation with no message beyond
   "function instantiation failed". Take `n: Int32` and widen inside.
2. **Pointers are `MutPointer[T, MutAnyOrigin]`**, not
   `UnsafePointer[Scalar[DType.T]]`, which cannot infer its origin at a call
   site even though it compiles standalone.
3. **Index with `unsafe_load` / `unsafe_store` / `[unsafe_offset=i]`**;
   positional `p[i]` is deprecated.
4. Shared memory is `stack_allocation[N, Scalar[DType.f32],
   address_space = AddressSpace.SHARED]()`, and THAT one does keep
   `UnsafePointer` with the origin unbound as `_`.

**Rule for this tree: a kernel is not ported until it has been ENQUEUED.**
Compiling is not evidence. `src/launch_probe.mojo` is the smallest harness
that produces the evidence and every new kernel gets added to it.

## 11. CORRECTION to item 2: widening the tile sync is NOT merely expensive

Item 2 says CatBoost's 8-lane `addToHistTile.sync()` becomes a threadgroup
`barrier()` and calls that "correct and strictly more expensive". **That was
wrong, and a known-answer check caught it.**

A threadgroup barrier that some warps reach and others skip is undefined
behavior. CatBoost never hits it because its sync is lane-local, so warps
with different iteration counts never wait on each other. Widened to the
threadgroup, they do.

It is not an edge case: a 64-row partition over a 512-thread block gives warp
0 one iteration and warps 1 to 15 zero. The measured symptom was every
feature's histogram reading 0.0.

The fix is a matrix row, `requires_uniform_iteration_for`, not a local
workaround: every thread of a block runs the same iteration count and threads
with no rows contribute a 0.0 stat, which keeps every lane inside every
barrier. Adding 0.0 changes no sum, so it is a scheduling change.

**The limitation is MOJO's, not Apple's**, which is the opposite of the
natural assumption. `max.gpu.primitives` exposes `block` only, for every
vendor; NVIDIA and AMD hardware both have lane primitives and CatBoost uses
them heavily. So the row is `SYNC_BLOCK` on all four columns today, and the
kernel refuses rather than running an unwritten lane-sync path if it ever
says otherwise.

## 12. Async copies: one host staging buffer per copy

Not a port issue, a harness one, recorded because it cost an hour and
presented as a broken kernel. `enqueue_copy` is asynchronous. Reusing one
host buffer for three copies and overwriting it between them let the last
value land in the first copy: `part_ids[0]` took the row count instead of 0,
the kernel indexed a one-element partition array out of bounds,
`active_block_count` resolved to 0, and every thread took the early return.
Every histogram cell read 0.0 with nothing wrong in the kernel.

One staging buffer per copy, or synchronize between them.

## 13. Derive-by-copy does not propagate fixes. It already bit once.

`hist_half_byte.mojo` was derived from `hist_binary.mojo` by textual copy,
because the two differ in only `GroupSize` and the writeback. Then item 11's
divergent-barrier bug was fixed in the binary file, and the half-byte file
kept the broken loop: a grep for the fix found seven references in one and
ZERO in the other. It would have produced silently wrong histograms for every
feature with 2 to 15 folds.

**CatBoost does not have this problem and the reason is instructive.** Their
three histogram kernels share ONE loop, `ComputeSplitPropertiesDirectLoadsImpl
<THist, blockSize, GroupSize>`, instantiated per policy. The loop exists once;
only the accumulator type and the writeback differ. A fix lands in one place.

We cannot do that yet, and the blocker is item 10: Mojo cannot pass a
shared-memory pointer across a function boundary without a concrete origin,
so the loop had to be inlined into each kernel. The duplication is a
CONSEQUENCE of that language limit, not a choice, and it is the second cost
that limit has imposed after the accumulator split.

Until it can be unified, treat the two files as one: any change to the loop
in either must be applied to both in the same commit, and the launch probe
must exercise both. The right fix is a comptime-parameterized loop shared the
way CatBoost's template shares it, once shared pointers can cross a boundary.

---

# Deviations in the `cluster/` section (cuVS)

Items 14 and up belong to the k-means port. Same numbering, same file, on
purpose: these are all one tree and a reader should not have to know which
upstream a constraint came from to find it.

## 14. `cub::BlockReduce` over a key-value pair

`unfused_distance_nn.cuh:99` reduces the per-thread nearest centroid with
`cub::BlockReduce<KVType, TPB>` and a custom comparator. Mojo 1.0 has no CUB,
so it becomes the shared-memory tree reduction this tree already uses in
`compute_scores.mojo`.

**Unlike item 8 this one costs nothing**, and the difference is worth being
precise about. The histogram scan lost fidelity because a scan is a SUM and
float addition is not associative, so a different tree gives a different
answer. This reducer is a MIN over a total order (`value`, then `key`
ascending), so every reduction tree returns the same pair. Tiling the
centroids does not change it either, because the merge arm uses the same
total order.

The one thing that would break it is a partial order. If ties were left
unbroken, block shape would decide the winner and the assignment would become
backend-dependent. Their comparator breaks ties on the lower key and that is
load-bearing, not tidiness.

## 15. Convergence: was a deviation, is now RESTORED

**This item used to describe a deviation. The deviation is gone and the
entry is rewritten rather than annotated**, per the standing rule that a
document a result falsifies is part of the result.

cuVS evaluates both stopping tests in a device lambda and reads only the
resulting FLAG back, checking it at the TOP of the NEXT iteration
(`detail/kmeans.cuh:817-825`, `:920-930`). The first version of this port
summed the cost and the shift to the host and tested there, which cost **six
drains and two full transfers per Lloyd iteration** and made a converged fit
report one fewer iteration than cuVS reports on the same data.

That was OUR artifact, not their design, and `HOST_AND_DEVICE.md` is explicit
that a wait we have and they do not is a fidelity defect rather than an
optimization opportunity. So it is fixed:

- `mojo_only/reduce_by_key.mojo::finish_sum_kernel` folds the block partials
  into a device scalar, so no sum reaches the host.
- `detail/kmeans_common.mojo::check_convergence_kernel` is their
  `check_convergence` as the device function it is in their source, advancing
  `prior_clustering_cost` in the same call because splitting it would
  reintroduce an ordering hazard.
- The loop reads the flag at the top of the next iteration and decrements
  `n_current_iter` when it fires, which is theirs, decrement included.

**Now one drain per iteration and one four-byte flag**, with inertia read
once per restart instead of once per iteration. The host-side
`check_convergence` is kept beside the kernel because it documents the three
easy-to-"fix" details in prose and is what a bring-up harness wants.

The init path had the same defect in a worse form and is fixed in the same
commit. `kmeans_plus_plus` used to copy all `n_samples` distances to the host
to draw its candidates, ONCE PER ACCEPTED CENTROID, which is O(rows) host
traffic and breaks `HOST_AND_DEVICE.md`'s first rule outright. cuVS draws on
device (`raft::random::discrete`, `detail/kmeans.cuh:187`), so again this was
ours and not theirs. It is now a two-level device search in
`mojo_only/plus_plus.mojo`: contiguous chunk sums, pick the chunk over at
most 256 totals, then scan inside that chunk, plus a one-launch
`gather_rows_kernel` where there used to be one copy per trial.

**What deliberately did NOT change is the host still deciding.** It draws the
`n_trials` uniforms and takes the greedy argmin over `n_trials` costs, both
O(candidates), and cuVS also brings `bestCandidateIdx` to the host every
accepted centroid (`detail/kmeans.cuh:224`).

The whole fit now moves nothing that scales with rows across the bus. Every
remaining device-to-host read is one flag, one scalar, or `n_trials` floats.

## 16. The k-means++ candidate cost is FUSED

`detail/kmeans.cuh:196-215` writes an `n_trials x n_samples` matrix with
`matrix_vector_op(min_op)` and then reduces it with `reduce<ALONG_ROWS>`.
`cluster/mojo_only/plus_plus.mojo` does both in one kernel and never
materializes the matrix.

Arithmetically identical, and still a deviation, because it changes the
summation ORDER over samples. Two candidates that tie to the last bit would
be resolved differently and the fit would diverge from there. Recorded rather
than waved off.

## 17. Host RNG: `std::mt19937` has no counterpart

cuVS picks the first k-means++ centroid and each restart's seed with
`std::mt19937` on the host, and draws k-means|| candidates with
`raft::random::discrete` on the device. Neither stream is reproducible in
Mojo, so matching cuVS draw for draw is impossible.

`HostRng` is splitmix64, chosen because it is short enough to audit in place.
What survives is the property validation needs: same seed, same fit. What
does not survive is any hope of comparing our trajectory to theirs
iteration by iteration, which is why `cluster/tools/sklearn_reference.py`
compares INERTIA OVER SEVERAL SEEDS and prints the seed-to-seed spread of the
oracle itself before any tolerance is chosen.

## 18. cuBLAS is not a file we can read

`cublasGemmEx` (`unfused_distance_nn.cuh:205`) is closed source, so
`cluster/mojo_only/gemm.mojo` is a plain tiled product and is not pretending
to be a port.

**Their call carries the most interesting finding in the cuVS source**, and it
is a fact about their numerics rather than about our port: for `float` input
the compute type is `CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits. cuVS's
shipped float32 k-means does not compute float32 distances on NVIDIA. See
`cluster/mojo_only/gemm.mojo` and the `bitwise-gbdt` tree.

# Deviations and hazards in the `neighbors/` section (cuVS + RAFT)

## 19. A pointer conditional expression picked the WRONG BRANCH

This one cost a debugging session and is the most important entry in this
file for anyone writing Mojo kernels here.

In `select_radix.mojo` the last-filter stage chose its input with

    var lf_ptr = out_ptr if writes_buffer else in_base

`writes_buffer` was False, verified by writing it to a debug buffer from
inside the same block, and `in_base` held the right data, verified by reading
`in_base.unsafe_load(0)` from the same block. **`lf_ptr` still pointed at
`out_ptr`.** Every top-k came back as `0.0` because `out_ptr` was a zeroed
scratch buffer.

Rewriting it as an explicit `if` fixed it outright and all four size and
buffer configurations went to zero errors.

**Be precise about the claim.** A minimal probe of `a if flag else b` on two
`MutPointer`s in a kernel did NOT reproduce it, so this is not "Mojo pointer
ternaries are broken". What is established is that in this kernel, with the
operands being `var`s reassigned across several branches inside a loop, the
conditional expression selected the wrong operand while a plain `if` did not.

**The rule for this tree: do not select a pointer with a conditional
expression. Use `if`.** It costs three lines and this failure mode is
invisible, because the wrong pointer is still a valid pointer to zeros.

## 20. A reach sabotage has a WINDOW, at both ends

Three checks in this session failed on sabotage magnitude while the kernel
under test was correct. The pattern is worth naming because a failed reach
check reads exactly like a real bug.

- **Too large, replacing:** the k-means x_norm sabotage REPLACED the sample
  norms with small values, which drove the expanded distances negative, and
  the clamp at `unfused_distance_nn.cuh:81` flattened all of them to `0.0` so
  the tie-break sent every row to centroid 0. 384 of 512 labels moved.
- **Too large, offsetting:** the k-NN query-norm sabotage added 5000 where
  distances are of order 2.7 and neighbor gaps of order 0.01. At 5000 the
  float32 ulp is 0.03 and the ranking dissolved. 438 slots moved.
- **Correct:** offset by 0.1 per query. Distances visibly move, ranking is
  preserved to well inside the gaps, and the assertion "a per-query constant
  cannot change a ranking" is actually being tested.

So a sabotage must be **large enough that the result must visibly move, and
small enough that it does not destroy the property being asserted**. When a
sabotage fails, check its magnitude against the precision of the quantity
before believing the kernel is wrong.

## 21. `L2Expanded` in float32 cannot rank collinear points, at any scale

Not a deviation, a property of the metric cuVS defaults to, found by a
fixture that assumed otherwise.

The expanded identity computes `||x||^2 + ||y||^2 - 2 x.y`. For N points on a
line the closest-pair squared distance is about `(range/N)^2` and the norm is
about `range^2`, so their ratio is about `1/N^2` INDEPENDENT OF SCALE. At
N=4096 that is 6e-8 against float32's 1.2e-7 relative precision, so the
subtraction has no significant bits left. Rescaling the data does not help.

The first k-NN fixture hit this exactly: norms about 1e10, distances about
1e3, ulp about 1024, true distance 900 returned as 0.0. The direct formula
has no such problem and is far slower, which is the whole reason the expanded
form is used.

Consequence for benchmarking: any speed comparison against a CPU
implementation using the DIRECT formula is also an accuracy comparison, and
this is where our answer differs from theirs.
