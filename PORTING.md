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

## 2. WRONG, CORRECTED 2026-08-19: Mojo DOES have warp primitives

**This item claimed Mojo 1.0 has no warp-level primitives, only `block` and
`barrier()`. That is false and it has been load-bearing since the first
commit.** It justified the scan substitution in item 8, it is why RAFT's
`select_warpsort.cuh` was ruled untranslatable, and it is repeated in the
memory notes.

The namespace is `std.gpu.primitives.warp`. Probed AVAILABLE in this
toolchain: `shuffle_down`, `shuffle_idx`, `shuffle_xor`, `lane_id`,
`prefix_sum`, `reduce`, `sum`, `max`, `broadcast`. Also `syncwarp` from
`max.gpu.sync`, and block collectives from `max.gpu.primitives.block`.

The earlier searches looked under `std.gpu`, `max.gpu`, `std.gpu.block` and
`max.gpu.block` and missed the `primitives` level in all four. Block
primitives live under MAX and warp primitives under STD, which is why one
search could not find both.

**Deleted rather than annotated**, per the standing rule, except for this
note recording that the claim existed and what it cost. What follows from the
correction is re-derived in `VENDOR_LIBRARIES.md` and is NOT yet done:
`cub::WarpScan` and `cub::ShuffleIndex` may port directly, and
`select_warpsort.cuh` may be translatable, which matters because RAFT's own
dispatch prefers it for every k a k-NN user asks for.

**Metal's missing float `atomicAdd` is a HARDWARE limit and is untouched by
this.** `checks/fixed_point.mojo` and its overflow proof stand.

## 3. `float` accumulation, not fixed point -- NOW PER COLUMN

CatBoost accumulates in `float` in shared memory and flushes with a
non-deterministic global `atomicAdd` guarded by `abs(val) > 1e-20f`. Copied
as-is on the NVIDIA and AMD columns, whose shared-memory budgets hold their
design. On the APPLE column (and everywhere under `NUMERIC_IDENTICAL`) the
hist_2 family's shared accumulation is Int32 fixed point instead, by
measurement, per item 41 below; the sentence that stood here forbidding
exactly that was written before Metal's 32 KB ceiling was priced and is
deleted rather than annotated.

## 4. `cub::DeviceRadixSort::SortPairs` per leaf

`split_points.cu:658-689` sorts each leaf's index range on the host loop, one
CUB call per leaf, 255 of them for a depth-8 tree. CatBoost's own comments
call this out: `//TODO(noxoomo): cub sucks for this, write proper segmented
version` (`:657`) and `//TODO(noxoomo): for oblivious trees we have overhead
for launching kernel per leaf` (`split_points.cpp:53`). There is no CUB in
Mojo. Port writes a stable 1-bit partition per leaf range, which is what the
sort is being used for.


## 5. Vector loads: PORTED (this entry was stale). The one residue is the
## alignment CLAIM, measured a wash 2026-08-21

This entry used to say the port takes the `OneElement` specialization only.
That was falsified by the code long before it was falsified in this file:
every histogram family now carries the `FourElements` loop -- `LOAD_SIZE =
4`, the warp-stripe layout, the `Unroll` batches, `ldg[width=4]` -- ported
from `compute_hist_loop_two_stats.cuh` / `_one_stat.cuh`, with the item-6
peel. What their arch pick chooses (`point_hist_half_byte_template.cuh:34-41`:
FourElements on modern arch) is what the port runs.

What did NOT come across is their 16-byte alignment GUARANTEE. Their column
stride is `AlignedColumnSize()`, so `(uint4*)` casts are legal at any
partition offset; our columns stride at raw `n_rows`
(`greedy_search_helper.mojo`), so the loads are stated `alignment=4`.
MEASURED 2026-08-21, isolated stripe-read at the hist loop's exact access
shape, 128 MB, arms alternated in one process: `alignment=4` ~52 GB/s
median against `alignment=16` ~56 GB/s, ranges overlapping heavily
(35-62 vs 50-64). Even reading the 8% edge as real, deviation 60's
calibration (5.9x isolated became 2.7% end-to-end) bounds it far under 1%
of a tree. **Porting `AlignedColumnSize()` padding to buy the claim is
therefore DECLINED, priced here.** The width was the money; the claim is
noise on this box.

## 6. `AlignMemoryAccess` peel: PORTED with the loads of item 5

The original entry recorded the peel as omitted because at one element per
load there was nothing to align. The `FourElements` port brought it across:
block 0 walks the unaligned head and tail through scalar `AddPoint` and the
striped loop sees the aligned middle only, which is what makes the 4-wide
load legal (no per-element bounds test in the body). Their two-macro shape
(`ALIGN_MEMORY` / `ALIGN_MEMORY_GATHER`) is mirrored in each kernel file.

## 7. THE FLOAT ATOMIC FLUSH (THE PREMISE WAS WRONG)

CatBoost flushes every histogram with `atomicAdd(dst + fold, val)` on
`float`. This section used to assert that Metal has no floating-point atomic
add at all, that the instruction does not exist, and that the port therefore
could not be literal here on our primary target.

**That was false, and it was the load-bearing claim under
`checks/fixed_point.mojo`.** Probed 2026-08-19 on the M4: 1024 threads
each adding 1.0 through `Atomic.fetch_add` return exactly 1024.0. The
instruction is there. Nothing forced the substitution.

So this is no longer a deviation a vendor imposes on us; it is a choice, and
the ladder now expresses it. `FAST` takes CatBoost's float atomic on every
vendor, which is what they ship. `IDENTICAL` pins every vendor to the integer
accumulator, where integer addition is associative and the sum does not
depend on which block lands first. No column is forced.

Two things survive the correction:

1. Fixed point needs a scale, and the scale must bound every partial sum.
   That is a real piece of work CatBoost never has to do, and it had to be
   written from nothing. It is also where the 20x loss regression came from,
   so the range contract in `HANDOFF.md` is the part of this section worth
   keeping.
2. Reproducibility is still cheaper on Apple than elsewhere, because Apple's
   lane width already equals the pinned 32. But it is no longer FREE, since
   the integer path is no longer what Apple runs anyway. The cost is
   unrecorded.


## 8. The bin prefix scan: `cub::WarpScan` (SEE ITEM 2, THE PREMISE WAS WRONG)

**The justification below rests on item 2's claim that Mojo has no warp
primitives. That claim was false.** `std.gpu.primitives.warp.prefix_sum` and
`shuffle_idx` are both available, so this substitution may be unnecessary and
the fidelity loss it describes may be avoidable. Not yet re-measured; the
block scan is still what runs. Tracked in `VENDOR_LIBRARIES.md`.

### The original entry, kept because the deviation is still in the code

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

### CORRECTED 2026-08-21: there is no language limit, and there never was

This entry used to say the loop could not be shared because "Mojo cannot pass
a shared-memory pointer across a function boundary without a concrete origin"
and cited item 10. **Both halves are wrong.** There is no item 10 in this
file -- the numbering goes 9, 11 -- so the citation pointed at nothing, and
the claim it stood in for is false.

A shared-memory pointer crosses a function boundary today. The whole of it is
that `stack_allocation` yields `MutUntrackedOrigin`, and a callee written as

    def f(smem: MutPointer[Float32, MutAnyOrigin, address_space = ...SHARED])

rejects it -- not because the pointer cannot cross, but because
`MutAnyOrigin` is a DIFFERENT origin, not a wildcard. Parameterize instead
and it compiles and runs:

    def f[origin: MutOrigin, //](
        smem: MutPointer[Float32, origin, address_space = AddressSpace.SHARED]
    )

Measured, not argued: `checks/shared_pointer_probe.mojo` allocates a
threadgroup buffer, accumulates into it through exactly such a callee, and
reads back the 64 expected values. `pixi run check-shared-pointer`.

This is the fourth "Mojo cannot" in this file to fail on inspection (item 2's
warp primitives, item 8's scan, item 14's block reduce, and now this one),
and the third to fail because an ANNOTATION was mistaken for a capability.
The standing lesson is the one `mojotrees-code-not-source-of-truth` records:
a limit this repository asserts about its own toolchain is a measurement that
has expired, not a fact.

### What still stands, and what it now costs

Everything above the correction is unchanged: the derive-by-copy bug was
real, it shipped, and a grep found seven references to the fix in one file
and zero in the other. What changes is the remedy. CatBoost's shape -- one
loop, instantiated per accumulator -- is available to this port and was
available all along.

The NEW pointwise family (`gbdt/methods/kernel/`, rung 1 of `NEXT_TWO.md`) is
being written that way from the start, which is most of why its line estimate
is below the greedy-subsets family's 1.2x expansion over the CUDA it mirrors.

Unifying the EXISTING greedy-subsets loop is a separate, larger change to
files another session is working in, and it is NOT done here. Until it is,
the original rule stands for those files and only those: treat
`hist_binary.mojo` and `hist_half_byte.mojo` as one file, apply any loop
change to both in the same commit, and exercise both in the launch probe.

---

# Deviations in the `cluster/` section (cuVS)

Items 14 and up belong to the k-means port. Same numbering, same file, on
purpose: these are all one tree and a reader should not have to know which
upstream a constraint came from to find it.

## 14. SUPERSEDED 2026-08-19: the key-value block reduce is now warp-based

**The justification below rested on item 2's false claim that Mojo has no
warp primitives.** `std.gpu.primitives.warp.shuffle_xor` exists, and both
distance kernels now do a shuffle butterfly on the (value, key) pair followed
by a small block merge, which is the shape CUB itself defaults to
(`BLOCK_REDUCE_WARP_REDUCTIONS`). The old whole-block shared tree is gone.

The conclusion of the original entry survives unchanged and is the reason the
substitution was safe: the reducer is a MIN over a total order, so every
reduction shape returns the same pair. That is why this could be re-derived
without re-validating the numerics.

### The original entry, kept for the argument about total orders

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

- `checks/reduce_by_key.mojo::finish_sum_kernel` folds the block partials
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
`checks/plus_plus.mojo`: contiguous chunk sums, pick the chunk over at
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
`cluster/checks/plus_plus.mojo` does both in one kernel and never
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
`cluster/checks/gemm.mojo` is a plain tiled product and is not pretending
to be a port.

**Their call carries the most interesting finding in the cuVS source**, and it
is a fact about their numerics rather than about our port: for `float` input
the compute type is `CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits. cuVS's
shipped float32 k-means does not compute float32 distances on NVIDIA. See
`cluster/checks/gemm.mojo` and the `bitwise-gbdt` tree.

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

# Deviations and hazards in the `decomposition/` section (RAFT)

## 22. `x ** 0.5` is not `sqrt(x)` here, and two other Mojo import traps

`jacobi_eigh` failed to converge and the first suspect was the rotation
formula. It was `(1.0 + theta * theta) ** 0.5`. Replaced with
`std.math.sqrt` everywhere and recorded rather than quietly fixed, because
the expression compiles, looks right, and is wrong.

Two smaller ones from the same family, both of which cost a build:

- `from math import ...` does not resolve. It is `from std.math import ...`.
- `out` is reserved and cannot name a kernel parameter.

Also in this section: the eigen step runs on the HOST. That is inside
`HOST_AND_DEVICE.md`'s rule, which forbids host work that is O(rows), not
host work as such. This is `n_cols^3` on an `n_cols x n_cols` matrix, and the
covariance that produced it is the only part of PCA that touches rows.
cuSOLVER runs it on device because it already has a tuned batched kernel;
we do not, and the honest first version says so. The condition that would
change it is `n_cols` approaching `n_rows`, which no real PCA is near.

## 23. A TRANSPOSED CONTRACTION SHOWS UP AS NON-CONVERGENCE, NOT AS A WRONG NUMBER

The most useful debugging lesson in this section.

`covariance_kernel` loaded its two shared tiles with the row index and the
feature index the wrong way round. The output was not garbage and was not
obviously wrong. It was **plausible and non-symmetric**, and the symptom was
that the Jacobi eigensolver ran to its sweep limit and raised, because Jacobi
converges only on symmetric input.

So the failure surfaced two files away from its cause, as a solver problem
rather than a data problem, and the first instinct was to blame the solver.
What settled it was testing `jacobi_eigh` standalone on a 2x2 with known
eigenvalues, where it was immediately correct.

**Rule: when a solver will not converge, check the SYMMETRY of what it was
handed before touching the solver.** A cheap assertion that
`cov[i][j] == cov[j][i]` would have found this in one run, and any future
kernel producing a matrix with a mathematical structure should be checked for
that structure and not only for plausible magnitudes.

# `dbscan/` (cuML + RAFT)

## 24. Mojo refuses the same buffer as two mutable kernel arguments

DBSCAN's distance matrix is the dataset against ITSELF, so both GEMM operands
are `x` and both norm operands are `x_norm`. `enqueue_function` rejects that:

    error: aliasing values passed mutably to 'args' argument and passed
    mutably to 'args' argument

It checks the underlying value, not the expression, so binding through two
locals does not help either.

The operands are read-only in the kernel and cuVS declares them `const`, so
the right fix is an immutable pointer type on the kernel signature, which
would also be MORE faithful to their declarations rather than less. Until
then `runner.mojo` makes one aliased copy of `x` and of `x_norm`. That is
`n_rows * n_features` floats against an `n_rows^2` distance matrix that
already exists, so the cost is small, but it is a real allocation that their
code does not make.

## 25. Read their file. Do not describe it.

I wrote a core-point filter into DBSCAN's scan and compaction and documented
it as "not an optimization, it is the definition". Their
`thrust::exclusive_scan` runs over the whole degree array with no mask, and
`adj_to_csr` emits every non-zero. The core restriction is applied one step
later, in `weak_cc`'s `filter_op`.

The invented version would have produced a graph missing exactly the border
edges the labeler needs, which is the classic wrong DBSCAN, and it was
justified in a docstring with a confident paragraph about what "their
version" does. **Nothing in the docstring was read from their source.**

This is the same failure the repository's founding note describes: our code,
and our description of their code, are not evidence about their code. Read
the file.

## 26. `stack_allocation` with no address space is MEMORY, not registers

The register-tiling port of `core/gemm.mojo` was SLOWER than the naive kernel
it replaced, on every arm.

RAFT's contraction holds its accumulators in `DataT acc[AccRowsPerTh]
[AccColsPerTh]` and its operands in `regx[]` / `regy[]`, plain C arrays that
nvcc keeps in registers. The obvious Mojo transliteration is
`stack_allocation[N, Scalar[DType.float32]]()` with no `address_space`, and
that is thread-local MEMORY. Every `acc[i] += ...` became a load and a store,
which is precisely the traffic register tiling exists to remove.

Rewriting the accumulators as `SIMD[DType.float32, 4]` values recovered it:
k-NN went 30.9 ms to 22.3 ms and from INDISTINGUISHABLE to a 1.41x win.

**Rule: in a kernel, an array meant to live in registers must be a `SIMD`
value or named scalars.** `stack_allocation` is for shared memory, where the
address space is passed explicitly, and using it for thread-private data
silently spills what the whole optimization was about.

# Deviations added 2026-08-19, late round

## 27-29. Never issued

The deviation counter jumped from 26 to 30 during the 2026-08-19 lane
fan-out; no code, lane file, or doc references 27, 28 or 29. Recorded so
nobody hunts for them.

## 30. `logicalWarpReduce<P::AccThCols>` is a comptime-width shuffle group

Lives at `dbscan/impl/neighbors/epsilon_neighborhood.mojo` (the
`updateVertexDegree` section): the sub-warp reduction's width is a comptime
constant and every lane reaches every shuffle unconditionally, because a lane
that skips a full-mask shuffle hangs the lanes that reach it. Its block-size
sweep (cited by LANE_rbc-maxk: 142.10 against 129.08 ms at 200k) is the bar
any K_LIB wiring of this kernel has to clear.

## 31. `vd` is memset once because the kernel ACCUMULATES

Lives at `dbscan/impl/neighbors/epsilon_neighborhood.mojo`. Their contract
is `cudaMemsetAsync(vd, 0, (m + 1) * sizeof(IdxT))` before
`epsUnexpL2SqNeighborhood`, which adds into `vd` rather than writing it; ours
is `ctx.enqueue_memset` in the same position. Dropping the zero looks fine on
the first batch and corrupts every later one.

## 32. Their device-wide scan is three launches here

`adjgraph/algo.cuh:65` runs `thrust::exclusive_scan` (CUB decoupled
lookback, single pass). One threadgroup cannot do that shape on Metal, and
the first port ran `grid_dim=(1,1,1)` -- one block scanning the whole array
serially, twice per fit. Now a three-launch scan-then-propagate at
`dbscan/impl/dbscan/adjgraph/algo.mojo`, verified exact at 2,000,000
entries across 977 blocks (LANE_dbscan-brute D3;
`check_exclusive_scan_beyond_the_old_cap`).

## 33. `make_monotonic`'s unique-value step is replaced

Lives at `dbscan/impl/label/classlabels.mojo`, which also records why the
relabel is NOT optional (cuML runs `final_relabel` + `relabelForSkl` on every
fit -- `runner.cuh:412` -- so label VALUES are API, not just the partition).
The header says "do not improve"; read it before touching the file.

## 34. `adj_to_csr`: the warp-aggregated atomic and multi-block rows are unported, priced

Lives at `dbscan/impl/dbscan/adjgraph/algo.mojo` (module docstring). The
shared per-row cursor, chunked 16-bool loads and unordered output are theirs;
the warp aggregation of the cursor atomic (needs `coalesced_threads()`) and
the multi-block-per-row grid are not, and the docstring prices both. Label
propagation converges to the same fixed point either way, so this moves a
wait, never an answer.

## 35. DBSCAN defaults to the ball cover; cuML defaults to brute force

Lives in full at `dbscan/impl/dbscan/runner.mojo`, above `EPS_NN_BRUTE_FORCE`.
Short form: RBC beats our own brute force 2.70x to 27.53x from 16,000 points
up and loses at no measured size, with labels identical point for point.

**The mirroring argument for it was withdrawn.** An earlier version of that
note claimed cuML's restrictions "cost us nothing"; the opposite is true.
`runner.cuh:143-150` downgrades RBC to BRUTE_FORCE whenever the label type is
`int32_t`, and `:235` builds the index only under `float && int64_t`. This
port is int32-label, so **cuML's dispatch would never hand our parameters to
RBC at all.** The default rests on the measurement alone.

The inherited consequence is real and is now guarded: their int64 requirement
exists because the CSR is indexed by EDGE count, which passes 2^31 at dense
neighbourhoods long before it runs out of memory. `runner.mojo` raises on that
overflow rather than wrapping, mirroring their brute-arm assertion at
`:180-184`. Without it the offsets wrap, the CSR is garbage, and `weak_cc`
still returns a plausible labelling -- silently.

## 36. k-NN defaults to fused-iff-`grid_x == 1`; cuVS defaults to fused everywhere

Lives in full at `neighbors/impl/neighbors/detail/knn_brute_force.mojo`,
above `KNN_METHOD_FUSED`. This is entry 35 pointed the other way, and it is
the more instructive of the two, because the port was CORRECT and the
expectation was still wrong.

`fusedL2Knn` is what their dispatch takes at `k <= 64` + row-major + L2
(`knn_brute_force.cuh:443`). It was ported down to `faiss_select::WarpSelect`
in registers, verified register-resident in the emitted Metal IR, and it
matches a host Float64 oracle slot for slot and in order. It is also 1.15x to
1.18x SLOWER than the materialized tiled path at every size from 20,000 to
400,000 index rows, arms interleaved inside the repeat loop and the whole
sweep re-run with the arms in the opposite order.

**The estimate going in was 6x to 10x faster.** It was arithmetic over memory
traffic: the fused kernel never writes the ~23 GB distance matrix, so it
"must" win. What that model left out is how many blocks the kernel can field.
At `gridDim.x == 1` -- the only configuration ported at the time the numbers
were taken -- the block count was `ceil(n_queries / Mblk)` and did not depend
on the index size at all. Holding the index at 200,000 and
raising the query count shrinks the deficit monotonically (0.66x at 500
queries, 0.86x at 2,000, 0.92x at 8,000, 0.93x at 32,000) and never crosses.

**Rule: a traffic model that does not count blocks can predict the wrong
SIGN, not merely the wrong magnitude.** Their `gridDim.x > 1` split LANDED on
2026-08-19 (entry 40) and both arms were RE-TIMED the same evening, arms
interleaved, both orders pooled. The sign flips with the grid shape the
ported `launchConfigGenerator` picks:

- `grid_x == 1` (2,000+ queries at the bench shapes): fused is at worst a tie
  and at 32,000 queries a clean non-overlapping 1.25x win; medians favor it
  at every index size from 50,000 up (1.07x-1.19x).
- x-split engaged (below ~1,905 queries): fused LOSES, and at 500-1,000
  queries catastrophically (0.19x-0.22x; worst sample 2.1 s against tiled's
  43 ms). The mutex merge is correct on Metal (entry 40) but serialized
  per-row merges cost far more than the occupancy the split buys.

So the REVISED default is `KNN_METHOD_AUTO`: consult `fused_l2_knn_grid` --
the exact computation the launch itself uses -- and take fused iff it says
`grid_x == 1`, tiled otherwise. `KNN_METHOD_FUSED` still restores cuVS's
dispatch exactly, `KNN_METHOD_TILED` the 2026-08-19-morning default. The full
tables live above `KNN_METHOD_FUSED` in `knn_brute_force.mojo`;
`check_dispatch_takes_fused` asserts the AUTO default on both sides of the
boundary by which output buffer comes back written.

## 40. The cross-block merge's `__threadfence` is SPELLED as acquire/release, because Apple legalizes nothing else

`fusedL2kNN`'s `gridDim.x > 1` arm serializes per-row merges with a mutex
array: producer blocks hand their per-row top-k to consumer block 0 through
the output buffer, guarded by `atomicCAS` / `atomicExch` spins with
`__threadfence` on both sides (`fused_l2_knn.cuh:241-338`). Porting it needed
a device-scope fence, and the repo's standing note said that was an OPEN
question on Metal. It is now CLOSED, in three compiler-verified parts and one
probe:

- `std.gpu.intrinsics.threadfence` is comptime-asserted
  `"only implemented on NVIDIA GPUs"` (Mojo 1.0,
  `stdlib/std/gpu/intrinsics.mojo:790-792`). No standalone device fence
  exists for the Apple target.
- The Metal backend rejects a STRONG compare-exchange ("Apple GPU only
  supports `weak` compare-exchange; AIR exposes no strong compare-exchange
  primitive") and rejects acquire/acq_rel orderings on RMW ops ("Apple GPU
  does not support `acquire` atomic ordering").
- It DOES legalize `Atomic.load[Ordering.ACQUIRE]` and
  `Atomic.store[Ordering.RELEASE]` (SEQUENTIAL too), verified by enqueue.

So the port spells their protocol as a test-and-test-and-set: spin on an
ACQUIRE load, claim with a weak RELAXED compare-exchange, release with a
RELEASE store. Same protocol in the C++11 model (CUDA defines
`__threadfence()` as `atomic_thread_fence(seq_cst, thread_scope_device)`);
no ABA hides in the relaxed claim because each mutex state has one writer
role. `neighbors/mutex_probe_main.mojo` is the evidence: 650 in-envelope
contended handoff launches bit-exact against a host oracle over HASHED
payloads with a POISONED exchange buffer, plus two sabotage arms -- a
producer that skips half its words and a producer that releases BEFORE
writing -- each caught on EVERY iteration (25/25 and 200/200), so the probe
demonstrably sees violations, and a 2.1x-oversubscribed grid still made
progress. The protocol's real precondition is CO-RESIDENCY: a spinning
producer terminates only if its consumer runs, which is exactly why their
`launchConfigGenerator` caps the grid at `numSMs * blocksPerSM`
(`pairwise_distance_base.cuh:295-322`). That computation is ported with M4
inputs in `neighbors/impl/distance/detail/pairwise_distance_base.mojo`:
10 cores, thread-slot occupancy (3072 / 256 = 12 blocks per core), and the
shared-memory term as a 32 KB validity wall rather than a divisor, because
family-9 threadgroup memory is dynamically cached -- the measured query
sweep (entry 36) is only possible with many 18.5 KB blocks per core. The
grid shape and every hardware input live in that ONE file.

## 41. hist_2's Apple accumulation: 2-warp-shared Int32 slices, dither-quantized

CatBoost's `TPointHist2OneByte` gives every warp a PRIVATE 1024-float
shared-memory slice, 32 floats per thread (`hist_2_one_byte_base.cuh:20-22`).
At Metal's 32 KB threadgroup ceiling that caps a core at 256 resident
threads -- 12.5% occupancy, a wall built into the layout. The matrix row
`hist_smem_mode_for` (NUMERIC: integer fixed point is a different arithmetic
than float adds) keeps their design on NVIDIA/AMD `FAST` and, on Apple and
under `NUMERIC_IDENTICAL`, shares each 1024-slot slice between TWO warps
with LOCAL Int32 atomics -- Metal has no local float atomics -- halving
shared bytes per thread so the block doubles to 512 at the same 32 KB.

Measured: 46.5 -> 90.2 G updates/s on the accumulate probe (1.94x);
43.5-48.9 -> 32.3-35.9 ms/tree whole-tree at 800k x 100 x 128 folds
(1.33-1.51x), interleaved, identical trees both arms. 512 threads/32 KB and
256 threads/16 KB (two resident blocks) measured indistinguishable -- both
put 512 threads on a core, which is what pays.

Pass structure, bin decode and slot arithmetic are theirs; what changed is
WHERE the add lands, and the quantizer, which took three attempts and two
measured failures to get right (`hist2_quantize`): plain truncation and
round-to-nearest both accumulate a value-correlated bias LINEAR in the row
count, which `compute_scores`'s exact `partStat - sumLeft` right side reads
as phantom mass past every bin -- trees locked onto (feature, last bin) and
stopped at depth 1 on three different 300k-650k datasets. The shipped rule
is dithered floor keyed on the document position: exactly unbiased for any
value distribution, deterministic run to run and vendor to vendor, and
exact on integer-valued stats, which is what keeps `hist2_check` an exact
`!=` compare. A side effect worth naming: integer accumulation is
associative, so the Apple arm's histogram is deterministic run to run,
which CatBoost's own float path is not.

## 37. A tiny DBSCAN batch budget raises here; theirs wraps `size_t` into a full batch

`dbscan.cuh:66` computes `max_mbytes_per_batch * 1000000 - est_mem_fixed` in
`size_t`. Hand it a nonzero budget smaller than the fixed cost and the
subtraction WRAPS to ~2^64; the `std::min` at `:69` then quietly turns the
wrap into `batch_size = n_owned_rows` -- the tiniest budget buys the LARGEST
batch. And a budget between the fixed cost and one row's cost yields
`batch_size = 0`, which reaches `raft::ceildiv` at `runner.cuh:131` and
divides by zero. Neither behavior is a design; both are what unsigned
arithmetic does when nobody expected the input.

`compute_batch_size` in `dbscan/impl/dbscan/dbscan.mojo` raises on both:
budget at or under the fixed cost, and a computed batch under one row.
Copying the wrap would make `max_mbytes_per_batch = 1` MEAN "unbatched", and
`check_dbscan_tiny_budget_agrees` uses exactly that value to force many
batches, so the wrap is not merely unhelpful, it is untestable.

## 38. DBSCAN per-phase timing prints where cuML has nvtx ranges

cuML wraps every DBSCAN phase in an nvtx range (`runner.cuh:255`, `:299`,
`:330`, `:355`, `:373`, `:397`, `:411`) and logs per-batch progress through
the `verbosity` parameter (`dbscan.cuh:114`). Metal has no nvtx consumer, so
`dbscan_fit` takes `phase_timing: Bool = False` and prints each range as a
machine-parseable wall-clock line instead:

    PHASE <loop>.<phase> batch <i>/<n> <ms>

with `passes <p>` appended on `label.weak_cc`, the count their `WeakCC`
range cannot show either. The full format and the range-for-range mapping
are documented on `dbscan_fit` in `dbscan/impl/dbscan/runner.mojo`;
`dbscan/phase_main.mojo` is the dedicated main. Every phase already ends on
a `ctx.synchronize()` the port performs anyway, so the flag adds no
synchronization, and off (the default) it prints nothing. Timestamps are
host wall clock, which on one queue with a sync at each boundary is the
phase's device time plus its enqueue overhead -- the same thing their nvtx
range brackets.

## 39. Batch 0's RBC fill runs after the CSR buffer is sized, not inside loop 1

`dbscan/impl/dbscan/runner.mojo`. cuML's first batch loop fills batch 0's
CSR columns as it counts it (`need_ja_compute`, `runner.cuh:257`, taking the
two-pass arm of `vertexdeg/algo.cuh:137-163` which resizes `adj_graph` to
batch 0's own edge count at `:150`), and `runner.cuh:317` then GROWS that
buffer to `maxadjlen` -- legal because `rmm::device_uvector::resize`
preserves contents when growing. `DeviceBuffer` has no growing resize, so
ours sizes `col_ind` at `maxadjlen` first and runs batch 0's fill immediately
after, against the `ex_scan` and `vd` that loop 1's last, reversed iteration
left resident. Same single fill of batch 0 per fit, and the device state at
loop 2's entry is identical byte for byte --
`check_ball_cover_max_k_wiring` reads batch 0's resident CSR at the top of
its loop two and byte-compares it against a fresh two-pass answer, and a
sabotage that skips the fill fails `check_dbscan` outright ("blobs 0 and 1
were merged").

The dispatch this placement serves is a PORT, not a deviation: loop 1
measures `maxklen[i]` (`:289`), loop 2 skips batch 0 (`:327`) and sends every
other batch down the ONE-PASS `max_k` arm whenever `algo.cuh:119`'s spare
guard admits it (`rbc_take_one_pass`), falling back to count + fill when it
does not. Two walks over the dataset per RBC fit where this port previously
did three.

# Deviations added 2026-08-20 (fused L2-NN policy diff)

## 44. The fused L2-NN kernel is SINGLE-buffered: Apple's 32 KB wall

`Contractions_NT`'s smem budget is `P::SmemSize = 2 * SmemPage`
(`raft/linalg/contractions.cuh:104`): TWO page pairs, written and read in
alternation (`pageWr`/`pageRd`, `detail/contractions.cuh:157-186`), so
`PairwiseDistances::run()`'s main loop needs ONE `__syncthreads()` per
k-tile (`pairwise_distance_base.cuh:141-149`) and overlaps the next tile's
global loads with the current tile's arithmetic. At `Policy4x4<float>` that
is 36,864 bytes plus the norm rows, and Apple caps a threadgroup at 32,768,
so `cluster/impl/distance/fused_distance_nn/simt_kernel.mojo` runs ONE
page pair (18,432 B + 512 B at veclen 4) with TWO barriers per k-tile.

Priced: the deviation costs overlap only when `k > Kblk`, i.e. above 32
features on the normal policy. Every shipped k-means shape has `k <= Kblk`,
where their main loop body executes zero times and the two structures are
the same kernel. Everything else in the load machinery is theirs at the
matching line: `ldgX`/`ldgY`'s `Veclen`-wide loads with zero-fill
(`detail/contractions.cuh:189-259`), `sts`/`lds` vector smem traffic
(`:262-299`), `srowid/scolid/accrowid/acccolid` thread partitions
(`:96-102`), and the STRIDED row/column ownership (`accrowid + i *
AccThRows`). The pre-rewrite kernel read scalars where their `ldg` reads
`float4` and owned columns in blocked runs; both were transcription errors,
not decisions, and are gone.

The launcher (`min_cluster_distance_compute.mojo::_launch_fused`) feeds
their `launchConfigGenerator` port the kernel's ACTUAL single-buffered
footprint where theirs passes `P::SmemSize` (`fused_l2_nn.cuh:135-136`) --
each call describes the kernel it launches. `grid.x` is pinned to 1 there;
that is the PRE-existing `updateReducedVal` replacement (`DERIVATION_MAP.tsv`
`replaced` row), not part of this entry.

## 45. `sqrt` at the row write, not per accumulator cell

`l2_exp_distance_op::epilog` takes the square root of every accumulator
cell before the min reduce when `sqrt` is requested
(`distance_ops/l2_exp.cuh:137-145`). The port takes it once per row at the
final write. `sqrt` is monotone and injective on `[0, inf)` under IEEE
round-to-nearest, so the argmin, the tie set (and therefore the key
tie-break), and the written value `sqrt(min v) == min(sqrt v)` are all
bit-identical; the cost drops from `AccRowsPerTh * AccColsPerTh` sqrts per
thread per column tile to `AccRowsPerTh` per row. The rest of the epilog is
theirs verbatim, including the self-neighbor round-off guard
(`val * (val > 0) * !((val * val < 1e-6) * (xn == yn))`, `:127-135`), which
the pre-rewrite kernel had dropped.

## Hazard: `linalg.matmul[transpose_b=True]` at `n == 1` does not write

Not a deviation from an upstream, a defect in a vendor primitive we call, and
it is recorded here because it was live on five paths at once.

Measured through `core/gemm.mojo::gemm_nt` itself: m=64, n=1, k=32, output
poisoned before the call, **63 of 64 rows still held the poison afterwards.**
Nothing written wrong; almost nothing written at all. `VENDOR_LIBRARIES.md`
had said "returns zeros for some outputs", which is why this looked benign for
as long as it did -- zeros are visible, stale buffer contents are not.

The fault is `transpose_b=True` and not the shape: the same product with
`transpose_b=False` is correct at the identical shape. `n == 1` reaches it
from ordinary parameters, including `n_clusters % centroid_batch == 1` at
`min_cluster_distance_compute.mojo:197`, which is a REMAINDER. `gemm_nt` now
routes `n == 1` to `gemv_n`, which is what RAFT does and which tests correct
at every m from 1 to 100,003.

**Rule: a vendor primitive is UNCHECKED until a poisoned output has survived
it.** A signature proves reach; only a run proves the answer.

## Hazard: a `col_major` TileTensor view is honored by SOME matmul arms and silently ignored by OTHERS

Probed 2026-08-19 (LANE covariance-unblock), chasing the zero-copy T-N Gram
shape: a `col_major(m, k)` view over X's row-major buffer IS `X^T`, which is
exactly how cuBLAS serves `CUBLAS_OP_T`, and `matmul[transpose_b=False]`
accepts it and is CORRECT at 32x32x100003, 33x17x255, 8x8x8, 129x127x513 —
per-cell against a Float64 oracle, bit-identical to `gemm_tn`'s materialized
route at 32x32x10007.

**And it is WRONG at EVERY CELL across the whole m=n in {4..64} x
k in {64..2048} band**, at n=1 with m>1, and in `gemv_gpu` at every output.
Written wrong, never unwritten: the losing arms index the operand as raw
row-major memory and produce PLAUSIBLE, NON-SYMMETRIC numbers. The ok/wrong
boundary zigzags (33x17x255 ok, 33x33x512 wrong; 8x8x8 ok, 8x8x64 wrong) and
matches no predicate worth trusting across a toolchain update, so the view is
UNWIREABLE. `gemm_tn`'s vendor arm stays on `transpose_kernel` + `gemm_nt`;
its default arm for the shipped small-output Gram shapes is the split-K
kernel (`core/gram_splitk.mojo`, LANE gram-splitk), which needs no view and
no transpose.

Two lessons beyond the verdict. **A probe battery that passes is evidence
about ITS shapes only** — the first battery (eight shapes, all correct except
n=1) missed the wrong band entirely because none of its shapes landed in it;
the checks that caught it were PCA's `check_covariance_is_symmetric`
(bitwise) and OLS's Jacobi convergence gate, on the first wired run, exactly
as PORTING.md 23 predicts a transposed/garbled contraction surfaces. And
**dispatch-arm-dependent correctness is disqualifying by itself**: a
primitive right at the shape you tuned for and wrong at the shape a check
uses cannot be guarded, because the guard would encode a closed
implementation's internals. `check_matmul_colmajor` keeps four sentinel
shapes in the vendor table so a toolchain that fixes this announces itself.

## 42. PCA's covariance implements RAFT's `stable=false` @todo: the centering is FUSED into the split-K Gram read

`raft::stats::detail::cov` has two arms
(`raft/cpp/include/raft/stats/detail/cov.cuh`): the shipped `stable=true`
path at `:58-66` -- `meanCenter` IN PLACE (`:61`), then the cuBLAS GEMM
(`:65-66`), with cuML adding the mean back after (`pca.cuh:138`) -- and a
`stable=false` arm at `:67-69` that is nothing but
`///@todo: implement this using cutlass + customized epilogue!` over
`ASSERT(false, "cov: Implement stable=false case!")`. Fusing the centering
into the contraction's read is THEIR declared design; they never shipped it
because cuBLAS exposes no epilogue hook. Our Gram kernel is hand-written
(`core/gram_splitk.mojo`, the Apple column's arm), so the epilogue exists:
`gram_splitk_partial_centered_kernel` reads every element as `x - mu[col]`
into the staging tile -- both operand reads see centered values, X is NEVER
written -- and `compute_covariance`'s split-K arm drops the center AND
restore launches entirely (~100 ms of PCA's remaining 166 at the bench
shape was those two passes).

The arms split on ONE predicate, `gram_splitk_applies`, the same call
`gemm_tn` makes -- so `compute_covariance` fuses exactly when `gemm_tn`
would have taken split-K, and non-Apple targets (where the predicate is
False at every shape) run the shipped `stable=true` path verbatim: center,
`linalg.matmul` via transpose, restore.

Bit-identity is proven by `check_gram_centered_fused`
(`checks/gram_splitk_check.mojo`), not argued: the shipped pipeline
(`column_mean_kernel` -> `shift_columns_kernel(-1)` -> `gemm_tn_splitk`)
against the fused arm on the same device mu and hashed, offset-mean data,
every cell compared with `!=` -- the fused tile load performs the identical
fp32 subtraction the center pass stores (`x + (-1.0) * mu` is bitwise
`x - mu`), and products of bit-identical fp32 inputs in the same order are
bit-identical. The same check asserts X is bit-identical after the fused
call, and `check_covariance_fused_and_fallback_restore`
(`decomposition/checks/pca_check.mojo`) holds the wired arm to it while
proving the fallback arm's center + restore pair still runs by sentinel
(center must MOVE x when the restore is withheld). The fusion is OPT-IN:
`gemm_tn` never dispatches to the centered entry, so OLS and tSVD are
bit-for-bit untouched (ols_main 4/4 unchanged; plain-arm bit dump identical
across the change).

## 46. The k-means accumulate reads X `veclen`-wide: upstream's scalar-read premise is NVIDIA's, not Apple's

RAFT's `reduce_rows_by_key` accumulate reads ONE element per thread:
`SumsT val = d_A[j + lda * i];`
(`raft/cpp/include/raft/linalg/detail/reduce_rows_by_key.cuh:285`, in
`sum_rows_by_key_large_nkeys_kernel_rowmajor`; the colmajor arm `:213-227`
and `reduce_cols_by_key.cuh` are scalar likewise -- no `TxN_t`, no `ldg`
anywhere in either file). That is not an omission on their part: on NVIDIA
a warp's 32 adjacent scalar reads coalesce into full memory transactions,
so scalar-per-thread IS the full-bandwidth pattern and a veclen buys them
nothing. The LANE_kmeans-kernel audit (2026-08-20, table row 6) confirmed
our port faithful to those scalar reads.

On this Apple M4 the premise does not hold, and that is measured, not
argued: the assignment kernel's re-port to upstream's `Veclen` `ldg`
pattern -- the same scalar-to-vector swap and nothing else on the read
path -- took assignment from 63 to 21 ms/iter (re-verdict 2026-08-20,
commit ef0c4ba), while the faithful-scalar accumulate sat at 54 ms/iter =
~9.5 GB/s effective on a ~120 GB/s device at 4M x 32 float32. So the
privatized centroid-sum kernel
(`cluster/checks/reduce_by_key.mojo::accumulate_centroid_sums_
privatized_kernel`) now takes a comptime `veclen` and reads X as
`SIMD[float32, veclen]`, one chunk of `veclen` consecutive cells per
thread step -- a DELIBERATE DEVIATION BEYOND upstream, priced here, not a
fidelity fix.

The selection is not new machinery: the dispatch
(`launch_accumulate_centroid_sums`) routes on `fused_veclen_for` -- the
SAME 16/8/1-byte ladder upstream selects their assignment veclen with
(`fused_distance_nn-inl.cuh:107-110`) and the assignment port already
transcribed -- fed x's base address for both pointer terms because this
kernel reads one matrix. The ladder's `4 * n_features % 16 == 0` term is
the load-safety contract, exactly as it is upstream: `n_features` a
multiple of `veclen` means a chunk never straddles a row (so ONE label and
ONE weight read serve the whole chunk -- the scalar body re-read both per
CELL) and an aligned load that starts in bounds ends in bounds. Rows that
fail the ladder take `veclen = 1`, which is the old scalar body verbatim.

Bit-identity is untouched, by construction and then by check: a vector
load returns the same bits as `veclen` scalar loads; each lane's
quantization `Int32(x * w * scale)` is the identical scalar fp32
expression on identical bits in the identical order; and everything after
the quantization is Int32 addition, associative and commutative, so chunk
order cannot move a total. Proven, not argued:
`check_privatized_accumulate` pins the veclen=4 arm at the fit's d=32 on
real device buffers and holds it bitwise (`!=`, no tolerance) to the
scalar direct scatter-add oracle on hashed, scattered, skewed data, plus
run-twice equality and the dropped-flush sabotage;
`check_accumulate_veclen_dispatch` proves the scalar arm
reached-and-correct at d=33 (ladder rejects both widths) and the 2-wide
arm at d=34, each against the same oracle with a nonzero-oracle guard so a
silently-launched-nothing arm cannot pass.

Deliberately NOT widened: the weight-per-cluster kernel reads no X (8
bytes per row against the sums kernel's 128 at d=32, 1/16 the traffic)
and stays scalar; the direct scatter-add fallback arm is the
upstream-faithful kernel and stays verbatim -- it is also the oracle the
checks compare against, which it can only be while it stays scalar.

## 47. k-means||'s per-sample uniforms are a splitmix64 counter hash, not Philox, and its probability is formed in f32

`initScalableKMeansPlusPlus` draws one uniform PER SAMPLE per round with
`raft::random::uniform` (`cuvs/src/cluster/detail/kmeans.cuh:689-690`) -- a
counter-based device generator whose stream nothing in Mojo reproduces
(deviation 17 already prices the host-side half of the RNG story). Two
constraints pin the replacement: `HOST_AND_DEVICE.md` forbids the host
manufacturing O(rows) randomness, and validation needs the draw to be a pure
function of (seed, index). `cluster/checks/scalable_init.mojo::
scalable_uniform` is splitmix64 -- the same finalizer `HostRng` uses, short
enough to audit -- as a counter hash of `(round_seed, sample_index)`, with
the host contributing ONE 64-bit seed per round. Same mechanism class as
Philox (counter-based, stateless per element), different stream, so cuVS and
this port sample DIFFERENT candidate sets from the same seed; comparisons
are over inertia across seeds, never draw for draw, which deviation 17
already made the rule.

Second half: `SamplingOp` forms `prob_x = (oversampling_factor * n_clusters
* a.value) / cluster_cost` with the `double` factor promoting the arithmetic
before truncating to DataT (`kmeans_common.cuh:73-81`). Apple silicon has no
device Float64, so `scalable_keep` computes in Float32 with
`oversampling_factor * n_clusters` pre-multiplied in Float64 on the host.
The difference is at most an ulp at the probability threshold, i.e. it can
flip a draw only when the uniform lands within one ulp of the probability --
and the check does not depend on that never happening, because
`check_scalable_sampling_selection`'s host replay runs the SAME f32
expression, not their double one.

## 48. `cub::DeviceSelect::If` becomes flags + the f32 scan + a rank scatter, exact below 2^24 rows

`sampleCentroids` compacts the selected samples with `cub::DeviceSelect::If`
over an `ArgIndexInputIterator` and reads back the count
(`kmeans_common.cuh:228-269`), then marks the selection flags with
`thrust::for_each_n` (`:270-276`). Neither ships here. The replacement is
the operation spelled out: `sample_flags_kernel` writes {0,1} flags, the
THREE-STAGE INCLUSIVE SCAN k-means++ already ships (and
`check_device_inclusive_scan` already holds to a host scan element by
element) ranks them, and `select_scatter_kernel` writes each selected index
at `csum[i] - 1`, folding the thrust flag update into the same pass.
Stability -- selected samples land in index order -- is `DeviceSelect::If`'s
contract and survives by construction.

The price: the scan carries counts in Float32, which is exact only while
every partial count fits 24 bits, so `init_scalable_kmeans_plus_plus`
RAISES at `n_samples >= 2^24` rather than mis-scattering silently. The same
bound covers `count_labels_kernel` (their `countLabels` /
`cub::DeviceHistogram::HistogramEven`, `:95-135`, float counters as theirs):
atomic adds of exact integer-valued floats below 2^24 commute exactly,
which is why the step-7 weights are deterministic with no ordering
guarantee. Raising the row bound means an integer scan, not a tolerance.

One more mechanism swap inside the same function: the step-8 Lloyd pass
over the weighted candidates needs its own fixed-point scales (the
candidate matrix weighted by counts up to n_samples is a different
magnitude bound than the caller's), and they are chosen by `choose_scale`
over an O(candidates) host readback of the candidates and weights -- host
traffic `HOST_AND_DEVICE.md` permits, where upstream's float atomics need
no scale at all (deviation: theirs never quantizes; ours inherits the
fixed-point scheme every fit here uses).

# Deviations added 2026-08-20 (the CTR block's two device primitives)

## 49. The segmented scan's BLOCK-level half is hand-written: `prefix_sum` takes no operator

`gbdt/gpu_util/kernel/segmented_scan.mojo`. Theirs is
`cub::DeviceScan::InclusiveScan` under a CUSTOM ASSOCIATIVE OPERATOR --
`TSegmentedSum` and `TNonNegativeSegmentedSum`,
`cuda_util/kernel/segmented_scan_helpers.cuh:11,24` -- reached from
`SegmentedScanVector` (`cuda_util/segmented_scan.h:8`) and
`SegmentedScanAndScatterNonNegativeVector` (`cuda_util/kernel/scan.cu:47`).
The combine is `newFlag = left.flag | right.flag` and
`newValue = right.flag ? right.value : left.value + right.value`.

`reorder_one_bit.mojo` calls `max.gpu.primitives.block.prefix_sum` for its
block half and this file cannot: `prefix_sum` is ADDITION ONLY, with no
operator parameter, so there is no shipped counterpart to stand in for a
segmented combine. The in-block half is therefore a Hillis-Steele over
their operator in threadgroup memory, which is what `cub::BlockScan` runs
on their side. The three-launch device-wide decoupling around it is the
same gap `reorder_one_bit.mojo` already records (VENDOR_LIBS.md 3b/3c),
and phase 2 scans the per-block aggregates with one thread where CUB uses
a decoupled lookback.

**The library-call alternative was refused on arithmetic, not on taste.** A
segmented scan can be faked from two UNSEGMENTED ones as
`cum[i] - cum[start(i) - 1]`, which would let `prefix_sum` do all the work.
It is a subtraction of two nearly equal floats: at 800k rows of unit
weights `cum` reaches 8e5 while a segment sum is order 1, so float32 leaves
roughly 0.06 of absolute error on a quantity of size 1, and CatBoost's
`Eps` is 1e-12. Their operator accumulates FROM the segment start and
carries no such term.

Two of their behaviours are transliterated rather than tidied, and both are
easy to mistake for bugs:

* Neither entry point runs an exclusive scan. Both run an INCLUSIVE scan
  written one slot to the right (`segmented_scan_helpers.cuh:206-209`,
  `:68-73`), followed by `ZeroSegmentStartsImpl` (`segmented_scan.cu:11`)
  or by a pre-fill (`scan.cu:59`).
* `SegmentedScanVector` exclusive therefore NEVER WRITES SLOT 0. In their
  code it is covered only because `UpdateBordersMaskImpl` flags `i == 0`
  unconditionally (`ctrs/kernel/ctr_calcers.cu:139`). Ours leaves it alone
  the same way.

Gate: `pixi run check-segscan`, four arms
({vector, scatter} x {inclusive, exclusive}) against a host tally cell by
cell, on a fixture that forces a segment at index 0, at a scan-block
boundary, one past one, at an emit-block boundary (a DIFFERENT block size,
256 against 768), three starts in a row, and a `-0.0f` start. Six
sabotages, all red: flags ignored in phase 1 (3807/4001), the phase-2 carry
never resetting (13/4001), the phase-3 carry added unconditionally
(3070/4001), the exclusive shift dropped (3524/4001), the scatter reading
`indices[i]` instead of `indices[i + 1]` (3524/4001), and `raw < 0` in
place of their `ExtractSignBit` bit test (253/4001 -- the `-0.0f` rows,
which is why that helper is a bit test).

**No kernel-matrix row was needed.** The block comes from
`column_shared_limit` at 8 bytes a thread, which is 4096/6144/8192 on
Apple/NVIDIA/AMD, and CatBoost's own `GetScanBlockSize()` of 768
(`cuda_util/kernel/scan.cuh:7`) is below all three -- so the cap binds
everywhere and the geometry is identical across the three columns.

## 50. `ReorderBins` is their own one-bit reorder looped LSD, not CUB's multi-bit radix

`gbdt/gpu_util/kernel/radix_sort.mojo`. The CTR bin builder sorts with
`ReorderBins(Bins, Indices, 0, newBits, Tmp, DecompressedTempBins)`
(`ctrs/ctr_bins_builder.h:223`), which is `TRadixSortKernel<ui32, ui32>` ->
`cub::DeviceRadixSort::SortPairs` (`cuda_util/kernel/sort_templ.cuh:26`).
CUB is OPEN and a port candidate under `PORTING_RULES.md` 0b-i, and MAX
ships no device sort, so there is nothing to call.

What is written is not a fresh design: it is CatBoost's OWN
`NKernel::ReorderOneBit<ui32, ui32>` (`cuda_util/kernel/reorder_one_bit.cu
:11`, instantiated at `:61`, exposed as `ReorderOneBit<TMapping>` in
`cuda_util/reorder_bins.cpp:74`) driven over the bit range. Three named
differences from their CUB call:

1. ONE BIT PER PASS where CUB does a radix DIGIT (4-8 bits). Same answer --
   LSD over stable one-bit passes is stable and total -- at `bits` passes
   rather than `bits/5`. Priced, not measured; nothing calls this yet.
2. PING-PONG rather than their single-bit wrapper's two `cudaMemcpyAsync`
   per pass (`reorder_one_bit.cu:21-22`), which is what CUB does too
   (`cub::DoubleBuffer`, `sort_templ.cuh:10`, with the conditional
   copy-back at `:53` mirrored here). Mojo will not let two device pointers
   of different origins swap in one variable, so the parity selects the
   buffers instead of the pointers being exchanged.
3. NO DESCENDING ARM. `TRadixSortContext::Descending` exists
   (`cuda_util/kernel/sort.cuh:25`) and `ReorderBinsImpl` passes `false`
   (`sort.cpp:558`). Unported rather than written blind.

Gate: `pixi run check-radixsort`, three arms (6 passes even, 7 odd, and a
`[2, 7)` window for their `offset` argument) against a host STABLE counting
sort, comparing keys AND the value pairing cell by cell, plus conservation.
Sortedness is computed and printed SEPARATELY, because it is the check that
cannot see the failure that matters: under the value-routing sabotage the
run prints `keys out of order: 0` and `keys wrong: 0` beside
`stable pairing wrong: 4001 of 4001`.

**A sabotage found a hole in the check itself and it is worth recording.**
The odd-parity arm was first written as 37 keys over 7 bits, where bit 6 is
zero in every key: the seventh pass was the identity, the sorted answer was
already in the caller's buffer after six passes, and DELETING THE COPY-BACK
CHANGED NOTHING -- green on a branch it never reached. The arm now uses 97
keys, and `check_radix_sort` refuses any arm whose top bit is constant.
With that fixed the copy-back sabotage prints 685 descending steps and
3966/4001 keys wrong on the odd arms while the even arm stays green, which
is reach per branch (`PORTING_RULES.md:8`) shown rather than claimed.

## 51. Model serialization is OURS, in plain text, because theirs is flatbuffers

CatBoost persists a model with flatbuffers
(`catboost/libs/model/flatbuffers/model.fbs`, written and read by
`libs/model/model.cpp`) and exports six more formats from
`libs/model/model_export/` -- JSON, CoreML, ONNX, PMML, C++ and Python.
`gbdt/models/model_text.mojo` ports NONE of that. It writes a format of our
own.

What is taken from them is the CONTENT list, not the encoding. Their
`TModelTrees` carries `TreeSplits`, `TreeSizes`, `LeafValues`, `LeafWeights`
and `FloatFeatures` -- borders included -- because an applied model has to
quantize raw floats against the grid it was trained on. `TrainedModel`
carries the same set and so does the file.

WHY NOT PORT THE FLATBUFFERS FORMAT. Three reasons, in order of weight.
Their `.fbs` compiles through `flatc` into generated C++ that this toolchain
has no counterpart for, so a port is a hand-written encoder for a schema we
do not control. Their `TModelTrees` is a superset of everything this port
can build (`NonSymmetricStepNodes`, `TextFeatures`, `EstimatedFeatures`,
`EmbeddingFeatures`, `MultiBias`, `CtrFeatures`), so a faithful writer
mostly writes empty vectors and a faithful READER has to refuse most files
it is handed. And a binary format cannot be read in a failure, which is the
opposite of what every other fixture in this tree is for.

WHY TEXT WITH THE BITS SPELLED OUT. `String(x)` in this toolchain does NOT
round-trip: measured over 200,000 random bit patterns per width, 917 of
199,223 float32 values and 212 of 199,907 float64 values come back one ULP
wrong, and `String(Float32(1.4e-45))` is `"0.0"`. So each float is written
`<decimal>/<IEEE-754 bits in hex>`, the bits are what loads, and the decimal
is checked against them to one ULP so a hand edit cannot drift unseen. For
the record of what a correct decimal writer looks like, CatBoost's own JSON
export pins `FloatNDigits = 9` and `DoubleNDigits = 17`
(`libs/helpers/json_helpers.h:22-32`), which are the round-trip digit
counts; theirs honors them and ours does not.

Gated by `pixi run check-model-io`: bit-identical predictions through both
the host apply and the device evaluator, field-by-field struct equality, and
five sabotages that each turn it red. The float-precision sabotage -- rewrite
every token's bits from its own decimal half, which is exactly what a
decimal-only format stores -- moves 150 of 512 predictions, which is the
measurement that pays for the format.

# Deviations added 2026-08-20 (the CTR block)

## 52. The FeatureFreq calcer runs on the HOST (the rest of this landed on the device 2026-08-21)

**MOSTLY RETIRED.** This entry used to cover three things: the CTR bin
ordering, the segmented scan, and the frequency calcer. The first two are
now on the device and the sentences claiming otherwise are deleted rather
than annotated.

WHAT LANDED ON THE DEVICE, 2026-08-21:

* `TCtrBinBuilderGpu` (`gbdt/ctrs/ctr_bins_builder.mojo`) is
  `ProceedNewBins` (`ctrs/ctr_bins_builder.h:212-222`) launch for launch:
  `ComputeCurrentBins` (`ExtractMask` + `ScanVector` + `ScatterWithMask`),
  `GatherWithMask`, `ReorderBins` -- their LSD radix sort,
  `gpu_util/kernel/radix_sort.mojo` -- and `UpdateBordersMask`.
* `THistoryBasedCtrCalcerGpu` (`gbdt/ctrs/ctr_calcers.mojo`) is
  `Reset` (`ctr_calcers.h:85-99`) and `VisitCatFeatureCtr` (`:121-153`)
  launch for launch, including both calls to
  `SegmentedScanAndScatterNonNegativeVector` and `GetGatheredBinSample`'s
  `ui8` gather.
* Two `cuda_util` files the above needed and nothing had ported:
  `gpu_util/kernel/transform.mojo` (`GatherWithMask`, `ScatterWithMask`,
  `transform.cu:214-276`) and `gpu_util/kernel/scan.mojo`
  (`ScanVector<ui32>`, `scan.cu:10-19`).

WHAT WAS STILL HOST SIDE LANDED ON THE DEVICE 2026-08-21, closing this
deviation's port. `TWeightedBinFreqCalcerGpu`
(`gbdt/ctrs/ctr_calcers.mojo`) is `VisitEqualUpToPriorFreqCtrs`
(`ctrs/ctr_calcers.h:307-341`) launch for launch, and the two device
pieces it needed are ported beside it: `UpdatePartitionOffsets`
(`gpu_util/kernel/partitions.mojo`, from
`cuda_util/kernel/partitions.cu:81-107` + `:155-176`, both dispatcher
arms) and the Sum arm of `SegmentedReduceVector`
(`gpu_util/kernel/segmented_reduce.mojo` -- their `cub::
DeviceSegmentedReduce` hand-written portably; the vendor check is
recorded in that file's header: MAX ships no segmented reduce). Gated
BIT-EQUAL against the host driver plus an independent tally by
`pixi run check-freq-ctr-device`, across the packing-policy cardinality
boundaries, both priors, the `partCount == size` dispatcher arm, a 100k
multi-block run, and a sabotage that must land exactly on the affected
categories.

`TCtrBinBuilder` and `TWeightedBinFreqCalcer` (no suffix) stay as the
gated host references (`PORTING_RULES.md` 0b-ii). WHAT REMAINS OF THIS
ENTRY IS WIRING, not porting: `train()`'s permutation-independent call
still runs the host driver, and the one-line swap to
`compute_simple_ctrs_device(ctx, ...)` belongs to the lane that owns
`gbdt/train.mojo` -- the handoff note is in that driver's docstring, and
UNWIRED.md carries the entry until the line lands. The
`counter_calc_method == Full` arm keeps its host implementation: opt-in,
reached by no benchmark, and the same number on a fit with no test pool.

`THistoryBasedCtrCalcer` (no suffix) is likewise kept, as the reference
`checks/ctr_device_check.mojo` compares the device calcer against cell by
cell. A host reference used to CHECK a device answer is not a CPU path
(`PORTING_RULES.md` 0b-ii).

Two consequences that were stated here in advance and both held:

* `_stable_sort_by_bin` had to stay STABLE when it became a radix sort. The
  order of rows within a category IS the CTR estimation permutation, which
  is the history an ordered target statistic reads. An unstable sort is
  invisible to any FeatureFreq check (counts do not care about order) and
  silently randomizes every `Borders` value. `check-ctr-device` gates
  stability SEPARATELY from sortedness for exactly that reason, and the
  distinction earned its keep: under the skip-the-sort sabotage the run
  prints both, and under an off-by-one bit count it prints 60 descending
  steps at k=17 against 2003 wrong index words.
* `checks/ctr_check.mojo` gating the host arithmetic against planted
  counts and an independent O(n^2) tally is what gave the device version a
  reference to match rather than a claim to inherit.

## 53. `model_size_reg` STOPPED being a no-op the day CTR columns appeared

`UpdateFeatureWeightsForBestSplits` fills the feature-weight vector with
1.0 and RETURNS EARLY when `GetCtrsCount() == 0`
(`cuda/methods/update_feature_weights.cpp:20-22`). That early return is why
`model_size_reg = 0.5` cost nothing here for as long as CTRs were unported,
and `catboost_options.mojo` said so.

With CTR columns present the early return is not taken, and every CTR
column that is not yet USED gets
`pow(1 + maxCtrUniqueValues / maxUniqueValues, -modelSizeReg)` instead of
1.0 (`:27-44`), which the score kernel multiplies into every gain
(`compute_scores.cu:136-137`). This port does not implement that function,
so its CTR candidates are scored HIGHER than stock CatBoost's by exactly
that factor.

Live divergence, not a dormant one. The docstring in
`gbdt/options/catboost_options.mojo` was rewritten rather than annotated.

## 54. Mojo CONTRACTS the MinEntropy penalty into an FMA; clang does not

Found while gating CTR target binarization, and it is the same defect class
as the `std.math.log` one already recorded in
`gbdt/grid_creator/binarization.mojo`: arithmetic noise re-deciding a
tie-break on a dynamic-programming plateau.

`_penalty_min_entropy(w)` is `w * log(w + 1e-8)` and the DP adds it to a
precomputed term, `prev_error[i] + _penalty_min_entropy(total -
sweights[i])`. **Mojo contracts that multiply-then-add into an FMA across
the inlined penalty**, evaluating `fma(w, log(w), prev_error[i])` with the
product kept at full width. C++ at clang's default `-ffp-contract=on`
contracts only within one SOURCE expression, and `Penalty<type>(...)` is a
separate call, so CatBoost adds the ROUNDED product.

On a column of distinct values every weight is 1, so
`Penalty(sweights[j] - sweights[i])` depends only on `j - i` and the
objective is exactly symmetric. At 4001 distinct values and budget 1 the
optimum sits at cut 1999 AND cut 2000, mathematically equal; under the FMA
they differ by ONE ULP -- measured 30412.210990606218 against
30412.210990606214 -- and the last match's `<` tie-break, which takes the
FIRST index, takes the second one instead.

PROVED BY CONSTRUCTION, not inferred: recomputing the same scan with the
penalty marked `@no_inline` makes every symmetric pair bit-identical again
(i = 1997/2002 and 1998/2001 collapse to one value each, 1999/2000 to
another) and the first index wins as theirs does.

**The one-line fix is `@no_inline` on `_penalty_min_entropy`, and this lane
does not own that file.** Reported rather than applied. Four of the fifteen
cases in `bench/ctr_target_oracle.txt` diverge by exactly one adjacent,
equally optimal cut; `checks/ctr_check.mojo` names them, still requires
the divergence to be exactly one ADJACENT border, and FAILS if one of them
ever matches exactly, so the allowance is deleted when the fix lands rather
than left standing.

It survived until now because `bench/minentropy_oracle.txt` never runs
budget 1, where `bins == 2` makes the per-level loops execute zero times
and the last match is the only comparison in the algorithm.

# Deviations added 2026-08-20 (the CTR apply seam)

## 55. The GPU evaluator has a ONE-HOT arm. Theirs REFUSES the case entirely

`TGPURepackedBin` carries three members and their kernel reads two.
`FeatureXorMask` (`libs/model/cuda/evaluator.cuh:14`) is never touched by
`CalcIndexesUnwrapped` or `CalcIndexesBase` (`evaluator.cu:128-156`),
because their GPU evaluator declines a categorical model at construction:

    CB_ENSURE(!model.HasCategoricalFeatures(),
              "Model contains categorical features, GPU evaluation impossible")
                                            (`libs/model/cuda/evaluator.cpp:22`)

So the mask is not a hole in their kernel. It is a member their CPU
evaluator uses and the GPU repack copies through unread.

**This port takes the case they decline**, because `train(cat_features=...)`
produces one-hot splits and the evaluator is our inference path. The
predicate is therefore TRANSCRIBED FROM THEIR CPU EVALUATOR rather than
invented. Their repack, `libs/model/model.cpp:566-572`:

    if (feature.Type != ESplitType::OneHotFeature) {
        rb.SplitIdx = featureIndex.SplitIdx;
    } else {
        rb.XorMask = ((~featureIndex.SplitIdx) & 0xff);
        rb.SplitIdx = 0xff;
    }

and their compare, `libs/model/cpu/evaluator_impl.cpp:38`:

    indexesVec[docId] |= ((binFeaturePtr[docId] ^ xorMask) >= borderVal)
                         << depth;

`(b ^ (~v & 0xff)) >= 0xff` holds exactly when `b == v` for a byte, so one
branch-free compare covers both predicates and a float split is unchanged
under `xorMask == 0`. Their CPU templates the mask away when no split needs
it (`NeedXorMask`, `:16`, `:257`); `pack_model_for_evaluator` sets the same
flag from the model and `launch_eval` dispatches on it, so a FLOAT-ONLY
model runs the same kernel it ran before this arm existed.

NO KERNEL-MATRIX ROW. The arm is one `xor` on a value already in a
register: no warp width assumed, no wavefront primitive, no shared memory,
no atomic beyond the one the file already had. Nothing about it is vendor
specific.

REACH IS PROVED, not asserted. `checks/ctr_apply_check.mojo` flips every
one-hot split's type to `TakeGreater` in a LOADED model and re-runs the
evaluator, which has no layout to cross-check against: 405 of 512
predictions move. A no-op arm would have been bit-identical and green.

## 56. The CTR value table is keyed by the DENSE CODE, and `ctr_borders` is not written

Their `TCtrValueTable` is keyed by a 64-bit hash of the raw categorical
value and read through `TDenseIndexHashView::GetIndex` with a
`NotFoundIndex` sentinel (`libs/model/static_ctr_provider.cpp:48-50`,
`:66-70`). `train()` in this port takes DENSE CODES `0..k-1` and raises on a
negative one, so there is no hash step to port: the code indexes the count
array directly and out-of-range is their `NotFoundIndex`, taking the same
`emptyVal = Calc(0, denominator)` branch. Same job, smaller structure.

This is the same place the hash will have to arrive: `RECON_CTRS.md` step 6
already records that a feature COMBINATION bin is formed from their hash,
so tree CTRs force the port that FeatureFreq lets us skip. The seam is one
function, `TCtrValueTable.value_for`.

The rest of their decomposition IS ported rather than simplified: the table
stores the raw statistic per category and the model carries `PriorNum`,
`PriorDenom`, `Shift`, `Scale` and `CounterDenominator` beside it, with the
value formed at apply time by `TModelCtr::Calc` (`online_ctr.h:289-292`).
Storing the divided value instead would have been one number rather than
five and would have made `Shift` and `Scale` unrepresentable -- their CPU
learner fills both from `CalcNormalization` (`private/libs/algo/split.cpp:73-81`)
and their GPU learner leaves them at `0` and `1`, which is what ours writes.

`ctr_borders`, which `model_text.mojo`'s seam block planned, is NOT
written. Their `TCtrFeature::Borders` exists because a CTR feature is not a
`TFloatFeature` in their model and carries its own binarization; here a CTR
column IS a column of the feature table and its `feature` record already
carries exactly those borders, which `train()` computes from the CTR's own
grid (`batch_binarized_ctr_calcer.cpp:57-63`). A second copy would be the
same numbers twice and a second thing to corrupt. The plan sentence was
deleted rather than annotated.

MEASURED, because the format's own rule says a float-only model's file must
not move: an 8-tree, 5-feature float-only model writes 8022 bytes with SHA-256
`932fddfa04a970b3400244219cbe1b51d6f24b9a1162675e854aac0395534035` before and
after this round. The `ctr_columns`, `ctr_table` and `ctr_entry` records are
written only when non-empty and `split_type take_bin` is a TRAILING token on
one-hot splits alone, which is what buys that.

# Deviations added 2026-08-21 (the CTR estimation permutation)

## 57. The CTR estimation permutation, which is never their id 0

`gbdt/data/permutation.mojo` ports `TDataPermutation`
(`cuda/data/permutation.{h,cpp}`) and everything under it -- their
`Shuffle` (`cuda/data/data_utils.h:21-47`), `TRandom`
(`libs/helpers/cpu_random.h`) and MT19937-64
(`util/random/mersenne64.{h,cpp}`) -- bit for bit. `pixi run
check-permutation` compares the raw 64-bit stream, `Uniform(t)`, and whole
orders against CatBoost's own generator compiled by
`tools/permutation_oracle/`. That part is not a deviation.

**Two things around it are.**

### 55a. `permutation_count` datasets, and the loop that reads them

Their builder loops (`gpu_data/doc_parallel_dataset_builder.cpp:251-262`):

    for (permutationId = 0; permutationId < permutationCount; ++permutationId) {
        ds.GetCtrsEstimationPermutation().WriteOrder(ctrEstimationOrder);
        writeCtrs(useTest, ds.PermutationDependentFeatures, ...);
    }

so `permutation_count` (4, `boosting_options.cpp:14`, and NOT collapsed to 1
here because a Borders CTR is a permutation feature,
`cuda/train_lib/train.cpp:102-107`) separate sets of `Borders` columns are
written, one per permutation dataset. Their boosting loop then picks a learn
permutation per iteration and estimates leaves on
`GetEstimationPermutation()`, which is `PermutationsCount() - 1`
(`methods/doc_parallel_boosting.h:101-103`, `:344-352`).

**Half of this landed 2026-08-21 and the paragraph that said otherwise is
deleted rather than annotated.** `train()` now builds all
`permutation_count` sets, one compressed index each (DEVIATION 89), with
`permutation_count` resolved exactly as `UpdateGpuSpecificDefaults` resolves
it -- 4 by default, ASSIGNED down to 1 with no CTR-bearing categorical
feature, an assignment that discards an explicit 4
(`cuda/train_lib/train.cpp:99-108`). All of them are binarized against
PERMUTATION 0'S borders, because their border builder caches by feature id
and permutation 0 fills the cache first (`gpu_binarization_helpers.cpp:
31-54`, `doc_parallel_dataset_builder.cpp:250`). `pixi run
check-permutation-count` gates it on an identity: `permutation_count=1` and
`permutation_count=4` read at permutation 0 must produce the same model bit
for bit, with permutation 1 as the control that must differ.

**AND THE LOOP LANDED THE SAME DAY.** `fit_with_test` keeps one cursor per
permutation, draws the structure permutation exactly as `:349-351` draws it,
estimates leaf values separately on every permutation (`:371-385`), and
exports the estimation permutation's ensemble (`:526-528`).

**Their modulus is `learnPermutationCount - 1`**, so at four permutations
the structure comes from permutation 0 or 1 and permutation 2 is never
searched on. That reads like an off-by-one in their code and it is
transcribed rather than corrected; the same expression appears in their
feature-parallel learner (`dynamic_boosting.h:286-289`), which is evidence
it is old if not deliberate.

Two of theirs remain absent and neither is read: their per-permutation
ENSEMBLES (`TVector<TResultModel>`), whose only consumers are snapshot
restore and a bootstrap `GetL1LeavesSum()` this port does not take, and the
`AppendEnsembles` replay, which needs snapshotting. Only the estimation
permutation's ensemble is accumulated here.

DEVIATION 64's shortcut -- RMSE at Newton-1 taking the searcher's leaves
instead of the estimator's -- is now conditioned on ONE permutation. It
rests on the searcher's leaf being the number a Newton step from zero gives
FOR THE ROWS THE SEARCHER PARTITIONED, and the other permutations partition
the same tree differently. The equality it claims is re-measured every run
of `check-permutation-count`: gate 1 compares a one-permutation RMSE fit,
which takes the shortcut, against a four-permutation fit read at
permutation 0, which does not, and 4000 of 4000 predictions match bit for
bit.

It is NOT ordered boosting, which is a different learner entirely; see 88.

### 55b. Their permutation 0 is the identity and this port must not use it

`FillOrder` returns `std::iota` when `Index == IdentityPermutationId()`,
which is 0 (`permutation.cpp:14-17`, `permutation.h:81-83`). So one of their
four CTR column sets IS computed in row order.

That is safe on their side and not on ours, and the reason is a stage
upstream of everything in `catboost/cuda`: `ShuffleLearnDataIfNeeded`
shuffles the whole learn pool at load whenever the data has any categorical
feature and `has_time` is false (`private/libs/algo/preprocess.cpp:161-181`,
`:183-199`). By the time their GPU builder sees a row order, it is already
random. `train()` here is handed `x_colmajor` in whatever order the caller
has, which may be sorted by target.

MEASURED, on a 4001-row target-sorted fixture with a category that carries
no information about the target
(`checks/ctr_device_check.mojo` section 3):

    leak = mean(ctr | target bin 1) - mean(ctr | target bin 0)

    row order (their permutation id 0):   0.303
    the shipped permutation (id 3):      -0.0027

and at the `train()` level, on the same fixture with the categorical column
as the only feature, 30 trees at depth 4:

    train loss under the permutation:  0.1234
    train loss in row order:           0.0606

Row order fits a feature that contains nothing twice as well, because the
statistic is reading the label it is supposed to be estimating without.
That is what "a different and worse estimator, not a slower one" means, and
it is why `ctr_estimation_permutation_id` defaults to 3 rather than to their
0. The pool shuffle itself is not ported: it is not a `catboost/cuda` file,
and reordering the caller's rows would change what `predict` returns.

# Deviation added 2026-08-20 (the Borders apply-time table)

## 58. One `TCtrValueTable` per model COLUMN, so three priors carry three copies of one histogram

Their `ctr_data` is a map keyed by `TModelCtrBase`
(`libs/model/ctr_provider.h`, walked by `TStaticCtrProvider::CalcCtrs`
through `ctr.Base`), and the three priors of their default `Borders`
fan-out are three `TModelCtr` values SHARING one `TCtrValueTable`. This
format keys a table by the model COLUMN it feeds -- `ctr_table <column>`
in `gbdt/models/model_text.mojo` -- because a column is what this model
applies, and `column_plan` reconstructs the raw-input map by walking
columns. So a categorical feature under their default writes the same
`uniqueCategories * TargetClassesCount` histogram THREE TIMES, once per
prior.

What it costs: bytes in the file, and nothing else. A 255-category feature
at two target classes is 510 ints stored three times instead of once,
1530 integers, on a model that already carries a border list per column.

What it does NOT cost: work. `build_ctr_tables` computes the histogram
ONCE per categorical feature and hands the same `List[Int]` to every
`Borders` config; only the prior differs, and only inside `Calc`. That is
their decomposition, and `checks/ctr_apply_check.mojo` asserts the
three tables are bit-identical blobs producing three different columns,
because a build that recomputed the histogram per prior would be doing
three passes for one answer and nothing about the ANSWER would say so.

Deviation 56 already records the other half of this table's shape: it is
keyed by the DENSE CODE rather than by their 64-bit category hash. The two
depart from the same place, which is that this port has no hash step.

### 58a. What the Borders arm CLOSED, rather than added

`train()`'s implicit fallback was `TCatFeatureParams.feature_freq_only()`
for one round, which was itself a departure from their GPU `simple_ctr`
default. The reason was sequencing: `Borders` trained and could not score,
because the table builder had no target-class arm, so the default would
have shipped a model that fits and refuses. That reason is gone and the
fallback is `TCatFeatureParams.default()` -- three `Borders` priors plus
one `FeatureFreq`, four columns per categorical feature, CatBoost's own.
`feature_freq_only()` survives as an opt-in surface and both sides are
exercised by name (`PORTING_RULES.md` 8).

### 58b. A `Borders` model's apply does NOT reproduce its own fit, and that is theirs

Worth writing down beside the deviation because it looks like one and is
not. `FeatureFreq` is permutation-independent (`ctr_type.cpp:44-58`), so
the apply-time table reproduces the learn column bit for bit and
`checks/ctr_train_check.mojo` gates exactly that. `Borders` has no such
identity in CatBoost either: the column a fit trains on is the ORDERED
statistic over the estimation permutation, and the column an applied model
carries is the FULL-LEARN-SET histogram their `CalcFinalCtrsImpl` builds
(`private/libs/algo/online_ctr.cpp:909-930`). MEASURED on
`ctr_train_check`'s 19913-row fixture:

    feature_freq_only()   fit 7.207e-09   applied 7.207e-09   identical
    default()             fit 6.353e-09   applied 6.648e-09   +4.6%

So the Borders gate is agreement with an independently computed full-set
tally PER ROW, never agreement with the fit's loss. Pointing the
bit-identity gate at Borders would have been gating a property the
estimator does not have.

### 58c. The `empty_value` dispatch is reached, and a sabotage had to prove it

`empty_value` branches on `ctr_type` because their provider computes
`emptyVal` inside each arm: `Calc(0, denominator)` for Counter/FeatureFreq
(`static_ctr_provider.cpp:65`) and `Calc(0, 0)` -- the prior alone -- for
Borders (`:95`). At their {1, 1} prior on a 4096-row pool those are 1.0
and 2.4e-4, and an unseen category is what a held-out set is made of.

**The first version of that gate could not fail.** Replacing the Borders
`emptyVal` with FeatureFreq's, in the source, ran the whole
`check-ctr-apply` file green with stdout bit-identical to the clean run,
because every Borders table this port writes carries
`CounterDenominator = 0` (their builder sets it only on the other arm,
`online_ctr.cpp:934-939`) and `Calc(0, 0)` and `Calc(0, denominator)` are
then the same number. The dispatch was correct and unreachable at once. It
is now exercised on a table carrying a NONZERO denominator, where the two
formulas separate, and the same sabotage turns that red.

The same round of sabotage found `compare_models` blind to the new axis: a
writer that dropped `TargetClassesCount` produced a file whose `counts`
list was the same LENGTH (a two-class histogram read as twice as many
one-count categories), so field-by-field equality passed. Both fields are
compared now.

## 59. The tree planes belong to the FIT, not the tree: a pool of one

Their `TCudaManager` hands every device buffer out of a per-device memory
pool (`cuda_lib/memory_pool.h`), so `CreateInitialSubsets`
(`split_properties_helper.cpp:1040-1080`) gets tree 2's histograms from
the memory tree 1 released. This port dropped the pool layer
(`gbdt/gpu_lib/NOT_PORTED.md`) and called `enqueue_create_buffer`
directly, which made every tree allocate its own planes.

MEASURED, 50k rows x 100 features x 254 borders, depth 6, on the M4:
`run_tree_layout`'s setup before the first level cost **4.69 ms of a
12.3 ms tree**, and allocating the three large planes -- `hist` and
`acc_i32` at 13 MB each plus the block scratch -- was **1.9 ms** of that.
The whole level loop was 6.08 ms, so the tree spent nearly as long
getting ready as growing.

`TTreeWorkspace` is a pool of one, which is all a single-device
single-stream port needs. `fit` holds a `List[TTreeWorkspace]` across the
boosting loop; a tree reuses the planes when they are large enough and
rebuilds them when they are not. An EMPTY list means no pool and restores
allocate-per-call exactly, which is what the `checks/` callers pass.

The `FillBuffer` semantics of `CreateInitialSubsets` are unchanged: both
pooled planes are still memset at tree start, because a pooled plane
carries the previous tree's cells where a fresh allocation carried
whatever the driver had. That memset is 0.5 ms and is real work, not
overhead.

MEASURED RESULT, interleaved against CatBoost CPU across five row counts:
fixed per-tree cost 12.56 -> 9.43 ms at 254 borders (25% off), row slope
unchanged at 18.5 us per 1000 rows, and the row count where we overtake
their CPU moves from 616k to 436k. Train mse is BIT-IDENTICAL to the
allocate-per-tree build at every size and both border counts, which is
the reach proof that no tree is reading a stale plane -- staleness would
change tree 2 onward. Sabotage: forcing the rebuild branch on every tree
puts 12.06 ms back on the clock.

## 60. Histogram copy and subtract move 16 bytes per thread, not 4

Their `CopyHistogramsImpl` and `SubstractHistogramsImpl`
(`greedy_subsets_searcher/kernel/histogram_utils.cu:15-34` and the
subtract beside it) move ONE float per thread, and this port
transliterated that. On NVIDIA it costs nothing: `__ldg` plus
`WriteThrough` already move a full sector per warp.

On this Metal box it is not free. MEASURED in isolation at a depth-6
level's shape (100 features x 254 folds x 2 stats, 32 pairs):

    one float per thread    11.0 GB/s
    four floats per thread  65.2 GB/s

Same bytes, same grid shape, 5.9x. `copy_histograms_vec4_kernel` and
`substract_histograms_vec4_kernel` are the 4-wide arms; the launcher
dispatches to them only when the plane base is 16-byte aligned
(`hist_cells_per_leaf * stat_count % 4 == 0` and
`hist_cells_per_leaf % 4 == 0` respectively) and falls back to their
1-per-thread kernel otherwise, so no dataset silently takes an unaligned
access. The `max(., 0)` guard on stat 0 is applied per lane, not per
vector: widening the access must not widen the predicate.

GPU-AGNOSTIC: 4-wide float is the one vector width every row of the
kernel matrix has. No lane width, no shared memory, no intrinsic.

End to end this is worth 2.7% at 50k rows, measured by alternating the
two builds in one window -- small, and worth stating plainly beside the
isolated 5.9x, because the copy is not the tree's bottleneck.

## 61. NOT a deviation: the serial scan is not costing us

`PORTING.md 8` records that `scan_histograms_kernel` replaced CatBoost's
`cub::WarpScan<double>` with one thread per feature scanning serially,
because the port believed Mojo had no warp primitives. That belief is
stale -- warp primitives are `std.gpu.primitives.warp` -- so the
substitution was re-examined for SPEED and measured against their shape:
32 lanes per feature, 8 features per 256-thread block, carry through the
last lane, with shared memory in place of the shuffle.

    leaves      1      2      4      8     16     32   (per level, ms)
    ours     .073   .086   .155   .100   .113   .153
    theirs   .035   .057   .070   .094   .171   .292

Their shape wins at shallow levels and LOSES at deep ones, and over a
whole depth-6 tree the two are a wash (0.68 ms vs 0.72 ms). The serial
scan stays. Recorded so the next reader does not spend the afternoon
this cost.

Their accumulator is `double` and ours is `float32`, which is a real
fidelity gap and is NOT closable here: this target has no float64.

## 62. Tweedie's `variance_power` reaches our kernel; on their GPU it reaches nothing

`TPointwiseTargetsImpl::Init` reads the parameter into the member
`VariancePower` (`pointwise_target_impl.h:288-291`) and **nothing ever reads
that member again.** The only float handed to the target kernel is
`GetAlpha()` (`:151-166`, `:346-356`), and `Init`'s Tweedie case never sets
`Alpha`, so it stays at its declared `0` (`:364`). Their GPU Tweedie therefore
trains at `variancePower = 0` whatever the user asked for, which is not
Tweedie: at p = 0 the loss degenerates to `-y*e^f + e^{2f}/2`.

Verified by grep against the pinned checkout: `GetTweedieParam` has exactly
two callers in `catboost/`, one of which is its own declaration
(`loss_description.h:144`), and the other two live sites are
`pointwise_target_impl.h:290` (the dead store) and
`private/libs/algo/tensor_search_helpers.cpp:308` — **their CPU**, which uses
it correctly.

This port threads it. Two reasons, and both have to hold:

1. **The arm we are measured against is their CPU.** Their GPU cannot run on
   Apple silicon at all — that is the entire thesis — so every Tweedie number
   this repository ever quotes is against a CatBoost CPU fit, and their CPU
   honours `variance_power`. Porting the GPU's dead store would make the two
   arms run different objectives and MSE parity would never close.
2. A loss whose defining parameter is ignored is not that loss.

**The arithmetic is unchanged.** `target_score` / `target_der` /
`target_der2` are `TTweedieTarget`'s three methods transcribed term for term
(`pointwise_targets.cu:34-57`). Only the VALUE of the parameter differs, and
it differs toward their own CPU.

This is the one deviation in this round that changes a number CatBoost's GPU
would produce. **IT IS NO LONGER AN OPEN ITEM**: section "62 (CLOSED)" below
carries the measurement -- 0 of 48 splits wrong against their CPU, and their
GPU's own `variance_power = 0` refusing to train at all.

## 63. The objective is a comptime parameter, not a host template switch

`PointwiseTargetKernel` (`pointwise_targets.cu:447-519`) is a host switch on
`ELossFunction` that constructs one of nine objective structs and passes it BY
VALUE into `PointwiseTargetImpl<TTarget, BLOCK_SIZE>`. Mojo has no zero-cost
struct-by-value kernel argument of that shape, so the switch became a comptime
parameter and the three methods became three comptime-dispatched functions
(`target_score`, `target_der`, `target_der2`).

Same specialization, moved one step earlier: theirs picks the template
instantiation on the host at run time, ours picks it at compile time. The host
switch itself is still ported, as `launch_pointwise_target_kernel`, in their
case order, so a reader can diff it arm for arm. No arithmetic difference.

## 64. RMSE at Newton-1 skips the estimator their code runs

Their `NeedEstimation()` is `LeavesEstimationMethod != Simple`
(`greedy_subsets_searcher.h:67-69`), which is TRUE for RMSE, so their boosting
loop runs the leaves estimator and OVERWRITES the leaf value the searcher
wrote (`doc_parallel_leaves_estimator.cpp:39`). This port does not.

The argument: for RMSE the two answers are the same number. The searcher
writes `stats / (w + L2Reg)` (`greedy_search_helper.cpp:646-647`); one Newton
step from zero writes `Gradient / (Hessian + 1e-20)` where
`Hessian = sum(weight * Der2) + L2Reg` and `TRmseTarget::Der2` returns `1.0f`
(`pointwise_targets.cu:188-190`), so `Hessian = w + L2Reg` and the two differ
only by the `1e-20` in the denominator.

**BEFORE 2026-08-21 THE SKIP WAS CONDITIONED ON THE OBJECTIVE ALONE**, which
was wrong the moment the estimation method became configurable: a caller
asking RMSE for Gradient leaves, or for more than one iteration, would have
silently got neither. It is now conditioned on the objective AND `Newton` AND
`iterations == 1`, which is the exact set the argument covers.

**MEASURED 2026-08-21, and the argument holds.** 8,192 rows x 6 features,
20 trees at depth 5, hashed fixture, RMSE. The skip condition was disabled in
the source and the identical configuration re-fitted with the estimator
running at Newton x 1:

    bit-identical predictions   8192 of 8192
    worst relative difference   0.0

So at the configuration this deviation is conditioned on, it is not an
approximation, it is the same arithmetic reached by a shorter route.

**AND THE CONJUNCTS ARE LOAD-BEARING, which the same probe measured.** The
first attempt at this compared the skip against the estimator at TWO
iterations, on the reasoning that their walker's second step "only moves if
the first did not already land". That was wrong: 0 of 8192 predictions
matched and the worst relative difference was 6.0%, with the two-iteration
arm reaching a slightly LOWER loss (15.7158 against 15.7422). Their walker at
`Iterations > 1` does not take the `:151-156` shortcut at all -- it
backtracks and re-evaluates -- so it is a different estimator, not the same
one run longer. **That is exactly why the skip is conditioned on
`iterations == 1` and not on the objective alone**, and before 2026-08-21 it
was conditioned on the objective alone.

## 65. `SegmentedRadixSort` is CatBoost's own one-bit reorder, batched over segments

`ComputeWeightedQuantile` calls
`SegmentedRadixSort(orderedTargets, orderedWeights, tmpTargets, tmpWeights,
binsOffsets, binCount, 10, 32)` (`leaves_estimation_helper.h:110-112`), which
is `cub::DeviceSegmentedRadixSort::SortPairs`. CUB is OPEN and therefore a
port candidate under PORTING_RULES 0b-i, and MAX ships no device sort
(VENDOR_LIBS.md, re-checked 2026-08-20), so there is nothing to call.

`gbdt/gpu_util/kernel/segmented_sort.mojo` is the same construction
`radix_sort.mojo` already carries — their `NKernel::ReorderOneBit`
(`reorder_one_bit.cu:11`) looped over a bit range — with a segment axis on
`block_idx.y`. **The segment axis is a shape their code already has**: their
own driver runs that reorder once per leaf (`SortWithoutCub`,
`split_points.cu:692-700`).

WHY BATCHED RATHER THAN THEIR PER-LEAF LOOP: their shape costs
`4 * bits * segments` launches. At depth 6 and their bit range that is
4 x 22 x 64 = 5,632 launches per tree, against a whole per-tree fixed-cost
budget of 9.43 ms (`PERF_2026-08-20_fixed-cost.md`). Batched it is 88,
independent of depth. **That is launch arithmetic, not a measurement** — the
5,632-launch variant was never built and timed. **OPEN ITEM**, though a cheap
one to close if anyone doubts it.

Two details of theirs that are copied and that a reader will want flagged:

* **Bits [10, 32), not the whole key.** Their call drops the bottom ten
  mantissa bits, so the exact leaf value is the weighted quantile of the
  targets rounded to about four significant decimal digits.
* CUB's float key decode (`NumericTraits<float>::Digit`) is applied
  explicitly here as `float_to_sortable`, because passing a float key to CUB
  applies it invisibly and their call site says nothing about it.

## 66. `ComputeNeedWeights`' early return moved below its barrier

Theirs (`exact_estimation.cu:51-73`):

    const ui32 begin = beginOffsets[blockIdx.x] + threadIdx.x;
    const ui32 end   = endOffsets[blockIdx.x];
    __shared__ float localBuffer[BLOCK_SIZE];
    localBuffer[threadIdx.x] = 0;
    if (begin >= end) { return; }            // BEFORE the barrier
    ...
    __syncthreads();

A leaf with fewer rows than the block width makes `begin >= end` true for some
threads and false for others, so the survivors reach `__syncthreads()` with
part of the block already retired. CUDA calls that undefined; NVIDIA's
hardware survives it. Metal promises nothing, and a barrier some threads never
reach is the one Metal failure mode that HANGS rather than returning a wrong
number.

Ours keeps every thread alive to the reduce and lets the out-of-range ones
contribute zero — the shape `pointwise_target_kernel` already uses for its own
tail, and their own idiom in their own file (`pointwise_targets.cu:255-257`).
**The sum is unchanged**: the threads that returned early in theirs had nothing
to add.

## 67. The quantile search is bounded by the BIN count, not their object count

`ComputeWeightedQuantileWithBinarySearchImpl` guards with

    const ui32 i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= objectsCount) { return; }

and then indexes `beginOffsets[i]`, `endOffsets[i]` and `point[i]`, **all of
which are per-BIN**. It is launched at `CeilDivide(binCount, 256)` blocks
(`:148`) and passed `Targets.Size()` as `objectsCount`
(`exact_estimation.h:113`). With `binCount` far below `objectsCount` the guard
never fires, and the threads of the last block read `beginOffsets` past
`binCount + 1` and WRITE `point` past `binCount`. On a 64-leaf tree that is
192 slots of overrun.

It cannot be copied: `point` here is the leaf-value buffer, so the overrun
would be into live data. Ours bounds by `bin_count`. Every in-range thread
computes exactly their arithmetic.

## 68. Never issued

Withdrawn before it landed. It claimed their GPU MAPE exact estimator divides
by `max(1, |residual|)` where `TMAPETarget::Der` divides by
`max(1, |raw target|)` (`pointwise_targets.cu:151-154`), and proposed passing
the raw column. Reading their CPU killed it: `CalcExactLeafDeltas` fills
`leafSamples[i]` with `targets[i] - approxes[i]`
(`private/libs/algo/approx_calcer.cpp:693`) and hands it to
`CalcOneDimensionalOptimumConstApprox`, whose MAPE arm is
`weightsWithTarget[idx] /= Max(1.0f, Abs(target[idx]))`
(`libs/metrics/optimal_const_for_loss.h:112-114`) — the residual again. **Both
of their arms agree**, so ours does too, and the number is retired rather than
reused.

## 69. Bernoulli and Poisson do not filter their zero-weight rows

`AreZeroWeightsAfterBootstrap` is true for exactly those two
(`enum_helpers.cpp:849-856`), so their `BootstrapAndFilter` runs
`FilterZeroEntries`, gathers the der / weight / index columns down to the
survivors, and returns `isContinuousIndices = false`
(`gpu_data/bootstrap.h:126-153`, `weak_objective_impl.h:30-45`). Their
searcher then works on a SMALLER row set through a non-contiguous index list.

This port multiplies and stops. **The model is the same model**: a row whose
bootstrap weight is zero contributes `0` to its weight plane and `0` to its
gradient plane, so it adds nothing to any histogram cell, nothing to any leaf
sum, and nothing to either fixed-point magnitude (`|0| == 0`, so the scale is
unchanged too). Filtering is a way of not paying for those rows, not a way of
getting a different answer.

Pinned analytically, 2026-08-21, on a 4,096 x 4 hashed fixture, 6 trees:

    subsample = 1.0   `u < 1.0` true for every draw in [0,1)
                      -> 0 of 4096 predictions differ from bootstrap OFF
    subsample = 0.0   `u < 0.0` false for every draw
                      -> 0 of 4096 predictions are nonzero, i.e. every
                         weight AND every gradient was zeroed, so no split
                         scored
    subsample = 0.5   -> 4096 of 4096 predictions differ from OFF

The first two are exact identities forced by their own kernel's arithmetic,
so they are reach proofs rather than a tally of ours. The third only shows the
draw scatters.

**THE COST, unmeasured:** at `subsample = 0.66` we stream 100% of the rows
through the histogram where they stream 66%. The sampling buys accuracy here
and buys accuracy AND about a third of the histogram time there. **OPEN
ITEM.**

**THE ONE PLACE THE ANSWER COULD DIVERGE, and it cannot today:** a score-side
test that counts ROWS rather than summing WEIGHTS would see the filtered count
on their side and the full count on ours. `min_data_in_leaf` is that test and
it is NOT WIRED in this port's searcher — grep `greedy_search_helper.mojo` and
there is nothing. **If `min_data_in_leaf` is ever wired, this deviation stops
being output-identical and the filter becomes required.**

## 70. FIXED: the extension emitted no Metal kernels, and the cause was not what this section said twice

**THIS SECTION HAS BEEN WRONG TWICE AND IS REWRITTEN, NOT ANNOTATED.** The
first version blamed GBDT's kernel count and named five dead hypotheses. The
second blamed THE BASENAME OF THE ENTRY FILE, on the strength of
byte-identical sources under different names producing 113 / 29 / 0 compiled
Metal functions, and built a whole apparatus around it: a stem loop, eleven
alternate names, and a "size cliff" the module was said to be outgrowing.

**The basename is innocent. So is the size cliff. Both were reading a poisoned
compiler cache.** `GradientBoosting` fits and predicts from Python; the stem
loop is gone from `bindings/build.sh` and was never added to
`bindings/build_gbdt.sh`.

### Cause 1: `--target-accelerator`, at ANY value, suppresses AOT Metal compilation

With `--target-cpu apple-m1` held fixed on one source, counting compiled Metal
functions (the mangled name with an `air` suffix):

    no --target-accelerator            every kernel loads
    --target-accelerator metal:1         0 AIR blobs   nothing loads
    --target-accelerator apple-m1-metal4 0 AIR blobs   nothing loads

So `metal:1` being an unrecognised string was the SMALLER half. Fixing it to a
real target does not help; REMOVING it does. Still true, independently
reproduced, and still the reason `_mojolearn_estimators.so` is broken (below).

### Cause 2: `MACOSX_DEPLOYMENT_TARGET` in the ENVIRONMENT does the same thing

Measured 2026-08-21 on `bindings/_mojolearn_gbdt.mojo`, one variable at a
time, **with the compiler cache cleared before each build** -- the step every
earlier experiment omitted, and the reason they all read wrong:

    MACOSX_DEPLOYMENT_TARGET   --target-cpu    AIR blobs
    11.0                       apple-m1            0
    12.0                       apple-m1            0
    (unset)                    apple-m1          141
    11.0                       (host)              0
    (unset)                    (host)            141

`--target-cpu apple-m1` is innocent -- it was only ever guilty by proximity,
sitting in the same flag string as `--target-accelerator`. The VALUE of the
deployment target is innocent too: 12.0 fails exactly as 11.0 does. **SETTING
THE VARIABLE AT ALL is what does it.** With it set, `mojo build` writes an
EMPTY 134-byte metallib for every kernel and embeds nothing; the extension
imports cleanly and dies at the first launch with "Failed to create Metal
function".

Four different basenames built cold with the variable set all give 0. The same
four built cold with it unset all give 141. **The basename does nothing.**

### Why every earlier measurement disagreed: the cache

`$MODULAR_HOME/cache/.mojo_cache` is content-addressed and **its key does not
include the macOS deployment target**. An empty metallib produced by a build
with the variable set is therefore served to every later build that hashes to
the same key, whatever ITS flags are. When this was found that cache held
40,772 files, of which **20,682 were 134-byte empty metallibs**, the oldest
stamped 05:58 that morning.

That one fact explains the entire history:

* A "good" basename was one whose kernels happened to hash to keys still
  holding REAL metallibs from an older, working configuration. A "bad" one
  hashed to poisoned keys. Deterministic on repeat, because a cache is.
* The famous decline -- 113 blobs, then 101, then 86, then 84 across four
  points in the history -- was **cache attrition, not a size cliff**. Each
  source change invalidated more of the surviving real entries, each rebuild
  replaced them with empties, and nothing could refill them because every
  build set the variable. Those counts were archaeology, not compilation.
* `mojo run` was unaffected and every check passed, because JIT compilation
  goes down a different path and writes REAL metallibs. That is also where
  the good cache entries came from in the first place.

**Clear the cache before any build whose kernel count you intend to believe,
or the number is fiction.** With a warm cache, `MACOSX_DEPLOYMENT_TARGET=12.0`
measured 141 and looked like the fix; it was reading back the `(unset)` build
from one minute earlier. That near-miss is the reason this section now leads
with the cache.

### The fix, which keeps both properties

The deployment target still has to be low -- `mojo build` takes it from the
host SDK, which means `minos 26.0` and a wheel installable on almost nothing.
Pass it to the LINKER instead:

    -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker <sdk>

`ld` stamps LC_BUILD_VERSION exactly as the environment variable would, and
the Metal compile step never sees it. Measured on a cold cache: **141 AIR
blobs AND `minos 11.0`, together.** Both `bindings/build.sh` and
`bindings/build_gbdt.sh` do this, both read the Mach-O header back rather than
assuming, and both fail with a message that names the cache first.

### What the split was, and was not, for

GBDT moved to `bindings/_mojolearn_gbdt.mojo` on 2026-08-21. It was
commissioned as a way under a size cliff that does not exist, so **that is not
why it is worth keeping.** It is worth keeping for the reason
`bindings/_mojolearn_estimators.mojo` gives: an independently changing binding
should not be a merge point. Every parameter added to `GbdtFitParams` used to
have to be unpacked in two files that could silently disagree about the order
of a flat list, which is a wrong answer rather than a failure. Now there is
one.

### Still broken the same way, and not ours to fix

`bindings/build_estimators.sh:8,11` does BOTH things at once -- it exports
`MACOSX_DEPLOYMENT_TARGET="11.0"` and passes `--target-accelerator metal:1` --
and `python/mojolearn/_mojolearn_estimators.so` has **0 AIR blobs**. Verified
2026-08-21 from a clean-venv install of the built wheel: DBSCAN, PCA and
LinearRegression all fail with "Failed to create Metal function" while
GradientBoosting, KMeans and NearestNeighbors all pass. It has been that way
since the artifact was built, independent of any cache. The fix is two lines,
both above. That file belongs to another session; flagged, not touched.

## 71. `multilogit`'s `functionValue` is per-block partials

Same substitution `pointwise_targets.mojo` and `bootstrap.mojo` record, for
the same reason: theirs ends in a block reduce plus a float `atomicAdd`
(`multilogit.cu:96-99`), which makes the sum depend on block arrival order.
Their `FillBuffer(functionValue, 0.0f, 1, stream)` prologue (`:186-188`)
exists only because that atomic accumulates and is not ported -- each block
writes its own slot.

## 72. `ElementsPerThread` is comptime, and both of their launchers pass 1

Theirs is a template parameter (`multilogit.cu:9`, `:103`) and both
`MultiLogitValueAndDer` and `MultiLogitSecondDer` instantiate it at 1
(`:181`, `:205`). The per-element arrays are `InlineArray`, which is
registers at that size. The unrolled shape is kept rather than collapsed to
a scalar because their `#pragma unroll` loops are the file's structure and a
reader diffing against `:59-92` has to find them.

## 73. `__ldg` is a plain load; the align sizes are still arguments

Mojo 1.0 ships no read-only-cache or non-temporal load hint, the deviation
`transform.mojo` and `fill.mojo` already record.
`predictionsAlignSize` / `derAlignSize` / `der2AlignSize` are passed exactly
as theirs are, so a caller that pads its planes still works; every caller in
this port passes `size`.

## 74. `SolveLinearSystemCholesky` is transcribed, not called

Theirs is LAPACK's `dposv_` (`private/libs/lapack/linear_system.cpp:46-47`)
through the clapack vendored at `contrib/libs/clapack`. clapack is OPEN, so
under PORTING_RULES 0b-i it is a port candidate rather than a call to make --
the "call the platform's equivalent" exception is for CLOSED libraries
(cuBLAS, cuSOLVER) where there is nothing to read. The shape rules a vendor
call out anyway: this runs on the HOST, once per leaf, on a
`(numClasses - 1 + 1) x (numClasses - 1 + 1)` matrix -- seven by seven for
covtype. A device linalg call would be a round trip per leaf to solve a 7x7.

`dposv` is `dpotrf` (Cholesky) followed by `dpotrs` (two triangular solves);
both are transcribed in the textbook form LAPACK's own reference
implementation uses.

**THEIR FAILURE BEHAVIOUR IS COPIED AND IT IS NOT WHAT IT LOOKS LIKE.**
`dposv` returns `info > 0` when the leading minor of order `info` is not
positive definite, stops factorizing, and **leaves the right-hand side
UNMODIFIED**. Their check is

    CB_ENSURE(info >= 0, "LAPACK dposv_ failed with status " << info);

which PASSES for every positive `info`. So when the Hessian is not positive
definite CatBoost does not raise and does not fall back -- it proceeds with
the right-hand side still holding the raw GRADIENT, and that leaf takes a
gradient step instead of a Newton one. This port does the same and returns
the `info` so a caller that wants to count it can.

**MEASURED 2026-08-21, AND IT IS NOT OPEN ANY MORE.** The walker now counts
the blocks whose factorization failed and the boosting loop reports the
total once per fit. A 4,096 x 5 hashed fixture, 7 classes, 25 trees at
depth 5, learning rate 0.5:

    l2_leaf_reg = 3.0 (their default)   0 blocks       final loss 0.0440
    l2_leaf_reg = 0.0                   772 blocks     final loss 7.09e+18

So the fallback is **unreachable at the shipped default and catastrophic
without the L2 term**, which is exactly what the theory predicted:
`diag(p) - p p^T` is positive SEMI-definite, and only the `+ lambda` on the
diagonal (`pointwise_oracle.cpp:178`) makes it definite. At lambda 0 the
Hessian is singular by construction -- the softmax's shift invariance is a
null direction -- Cholesky stops, 772 leaves take the raw gradient as their
step, and the fit diverges.

**That is a reason to treat `l2_leaf_reg = 0` as unsupported for MultiClass
rather than as a tuning choice**, and it is CatBoost's position too: their
`CB_ENSURE(info >= 0)` passing on failure means their own fit would diverge
the same way, silently. Ours at least says so.

## 75. The blocked Hessian is reduced one row at a time, into its own buffer

Theirs reduces every Hessian row into disjoint SLICES of one
`reducedHessianGpu` and reads the whole thing back once
(`pointwise_oracle.cpp:135-157`), with an `offset` accumulator threading
`columnCount * blockCount` per row. That bookkeeping exists because
`ReadReduce` is one call over one buffer.

Ours reads each row's reduce as it is produced: `numClasses` copies of
`binCount * (row + 1)` floats instead of one copy of
`binCount * lowTriangleMatrixSize`. **Same numbers, same order within a
row.** It costs `numClasses - 1` extra device-to-host copies per estimation
iteration -- at their Logloss default of ten iterations and seven classes
that is 60 extra copies per tree, each a few hundred bytes -- and buys not
reproducing the `offset` arithmetic, which is the part of their function most
likely to be transcribed wrong.

**UNMEASURED.** The extra copies have not been timed. **OPEN ITEM**, and a
cheap one to close: the alternative is their single-buffer form, which is
maybe thirty lines.

## 76. `add_model_value_kernel` gained an approx-dimension axis

Their `AddBinModelValues` takes a `TCudaBuffer` whose column count carries
the approx dimension; ours takes `dim_count` and `cursor_stride` and puts the
dimension on `block_idx.z`. `dim_count == 1, cursor_stride == 0` is the
single-dimensional path and produces exactly the arithmetic the kernel had
before the axis existed.

THE TWO LAYOUTS DIFFER AND THAT IS THEIRS. The shift is BIN-MAJOR --
`newPoint[bin * cursorDim + dim]` (`pointwise_oracle.cpp:41-47`) -- and the
cursor is PLANE-MAJOR, one contiguous column per class. Reading both as the
same layout would train a model with the classes rotated and nothing would
assert.

## 77-78. Reserved for the parallel lanes

77 is the CPython shared-lib loader lane, 78 the CatBoost-oracle lane. Both
were assigned before those lanes started, per PORTING_RULES 3: a five-lane
round once produced a three-way number collision because they were not.

## 79. MultiClass's search magnitude is ONE bound for all class planes

NO CATBOOST COUNTERPART, like the fixed-point accumulator it feeds: their
histograms flush with a float `atomicAdd` and need no bound at all.

`multilogit_val_and_first_der_kernel[.., search=True]` reports
`sum over rows of max over classes |w * der_k|` as its single gradient
magnitude, rather than one sum per plane.

WHY IT IS VALID FOR EVERY PLANE: for any class `k`,
`sum_rows |der_k| <= sum_rows max_j |der_j|`, so the bound holds for all
planes at once and overflow stays impossible -- which is the only property
`checks/fixed_point.mojo` requires of it.

WHY IT IS ONE NUMBER: `choose_scale` takes ONE scale for the whole
histogram, and `greedy_search_helper` already maxes the weight and gradient
magnitudes before calling it, so per-plane sums would be collapsed to their
max anyway.

WHAT IT COSTS, **MEASURED 2026-08-21** rather than bounded. The bound is
`sum_rows max_k |der_k|`; the tightest valid per-plane bound is
`max_k sum_rows |der_k|`. Their ratio IS the resolution given up, and
`checks/multilogit_check.mojo` prints it on every run. On 2,053 hashed
rows:

    numClasses  2    1.0000x     worst case 1x    0.00 bits
    numClasses  3    1.366x      worst case 2x    0.45 bits
    numClasses  7    3.084x      worst case 6x    1.62 bits

So it runs at about HALF the theoretical worst case, and at seven classes
costs 1.62 bits out of the margin `choose_scale` documents as millionfold
(~20 bits). Two classes is exactly tight, as it must be: with one free
plane the max over planes IS that plane.

**CLOSED.** The per-plane alternative would need `numClasses` reduction
lanes where the deterministic fold is comptime-fixed at two, and it would
buy back under two bits of a twenty-bit margin.

`checks/multilogit_check.mojo` verifies the bound actually bounds: every
class plane's own sum of absolute values is checked against the reported
magnitude, at 2, 3 and 7 classes.

## 80. NOT a deviation: the walker projects on the way out, and this port did not

Recorded because it was a real defect for a day and the shape of it will
recur.

`TNewtonLikeWalker::Estimate` returns `Oracle.MakeEstimationResult(...)` at
BOTH of its exits -- the `Iterations == 1` shortcut (`descent_helpers.cpp
:153`) and the loop (`:204`). For every single-dimensional loss that is the
identity, so the port returned the raw point and nothing noticed. For
MultiClass it is the GAUGE PROJECTION from the walker's `numClasses`-wide
point down to the cursor's `numClasses - 1`, and returning the raw point
hands the caller a vector one component too wide in a gauge the cursor
cannot read.

**The identity is why it survived**: a projection that is the identity on
every path a test exercises is invisible until the day it is not. It was
found by reading their file for a different reason, which is rule 1 doing
what it is for.

## 81. `add_bin_values` gained an approx-dimension axis, and the two layouts differ

Their `AddObliviousTree` takes a `TCudaBuffer` cursor whose COLUMN COUNT
carries the dimension and adds every column (`models/add_bin_values.h`,
`add_model_value.cu`). Ours takes a pointer plus a stride, so the dimension
is `block_idx.y` -- the same axis `add_model_value_kernel` grew, for the same
reason.

THE TWO LAYOUTS DIFFER AND THAT IS THEIRS. `leaf_values` is BIN-MAJOR,
`[leaf * dim + d]`, which is what `MakeEstimationResult` produces and what
the model stores; the cursor is PLANE-MAJOR, one contiguous column per class.
A port that read both the same way would predict with the classes rotated
and nothing would assert. `check-multiclass-train`'s cross-check between the
two apply paths is what would catch it.

THE LEAF INDEX IS COMPUTED ONCE PER ROW and shared by every dimension,
because an oblivious tree's structure does not depend on the approx: all
`dim` values of a row come out of the same leaf.

## 82. The text format wrote `n_leaves` values per tree, not `n_leaves * dim`

Found by `check-multiclass-train`'s round-trip gate on the day MultiClass
first trained. `model_text` already carried `dim` in its `tree` record --
the format was written for this -- but both the writer's loop and the
loader's count validation used `1 << depth`, so **a MultiClass model would
have lost every dimension but the first, silently, on save**, and reloaded
as a model whose leaf-value count disagreed with its own declared `dim`.

Both sides count `(1 << depth) * dim` now. `leaf_weights` still counts
`1 << depth`, because a leaf has ONE weight and `dim` values -- their
`CB_ENSURE(task.Model->BinCount() == weights.size())`
(`doc_parallel_leaves_estimator.cpp:21-23`) says exactly that.

A one-dimensional model's bytes are unchanged: the record index is the flat
`leaf * dim + d`, which is `leaf` when `dim == 1`. `check-model-io` stayed
green through the change, which is the evidence for that sentence.

## 83. THE MAPE DEFECT: their estimator's alpha is a DIFFERENT FLOAT from their kernel's

Found 2026-08-21 by `check-loss-oracle`, against CatBoost's own CPU output.
It is the most expensive kind of defect this port can have -- a wrong number
produced by correct kernels -- and it survived every gate that existed.

**CatBoost carries two alphas and this port carried one.**

The KERNEL's alpha is `GetAlpha()`, the single float
`PointwiseTargetKernel` receives (`pointwise_targets.cu:451`). For MAPE,
`TPointwiseTargetsImpl::Init`'s case is a bare `break`
(`pointwise_target_impl.h:261-266`), so it is the member's declared `0`
(`:364`) -- harmless, because `TMAPETarget` never reads it.

The ESTIMATOR's alpha is read somewhere else entirely
(`leaves_estimation_helper.h:72-74`):

    const auto &params = lossDescription.GetLossParamsMap();
    auto it = params.find("alpha");
    float alpha = it == params.end() ? 0.5 : FromString<float>(it->second);

**out of the loss params map, defaulting to 0.5.** For MAE and Quantile the
two coincide, so nothing showed. For MAPE the estimator wants 0.5 and the
kernel's is 0.

This port fed the kernel's float to `ComputeWeightedQuantile`. At alpha 0,
`ComputeNeedWeights` produces zero and their binary search collapses to the
segment start, so **every MAPE leaf value was that leaf's MINIMUM residual
instead of its MAPE-weighted median.** Measured before the fix: 37 of 48
splits wrong from tree 2 on, prediction relative RMS 0.64, objective
relative gap 1.55. After: **0 of 48 splits wrong.**

WHY NOTHING CAUGHT IT, and this is the part worth keeping:

* `check-pointwise-target` gates the target KERNEL, which was correct.
* `check-exact-estimation` gates `compute_weighted_quantile`, which was
  correct -- and it passes alpha EXPLICITLY, so it could not see a wrong
  alpha arriving. It also ran `use_mape_weights=False` on every arm, so
  **the MAPE branch had no caller in any check**, which is PORTING_RULES 8
  verbatim.
* Both were right about their own piece. The defect lived in the WIRING
  between them, which is the seam an analytic gate cannot reach and only
  the incumbent's own output can.

`BinOptimizedOracle` now carries `estimator_alpha` beside `alpha`, fed from
`TLossDescription.get_alpha()` -- which IS their `GetAlpha(lossDescription)`
(`loss_description.cpp:95-102`), the same accessor with the same 0.5
default.

## 62 (CLOSED). Tweedie's `variance_power`, now measured both ways

The entry above records the argument. `check-loss-oracle` measured it,
2026-08-21, our GPU against their CPU on a 3,000 x 8 constructed fixture,
12 trees, identical configuration on both arms:

    variance_power threaded (ours)   0 of 48 splits wrong
                                     prediction relative RMS  8.2e-08
                                     objective relative gap   1.7e-11

the tightest agreement of any of the ten arms.

**The counterfactual is stronger than the agreement.** Training our arm at
`variance_power = 0` -- their GPU's actual behaviour, since `Init` reads the
parameter into a member nothing ever reads again -- does not fit a different
Tweedie. It **refuses to train**: "All splits have infinite score ... [level
0, live leaves 1]". At p = 0 the loss degenerates to `-y*e^f + e^(2f)/2`,
whose second derivative grows as `2*e^(2f)`, and on a positive target every
candidate overflows at level 0.

So their GPU's dropped parameter is not a variant of Tweedie; on this
fixture it is not a model at all. Threading it is what makes a Tweedie fit
exist. **DEVIATION 62 IS CLOSED.**

## 84. Their Tweedie iteration count differs BETWEEN THEIR OWN ARMS, and we take the GPU's

`GetEstimationMethodDefaults` keys Tweedie on task type
(`catboost_options.cpp:221-231`): the CPU takes **1** Newton iteration, the
GPU takes **20**. `set_leaves_estimation_default` takes the GPU's 20, which
is faithful -- we are the GPU.

But **the arm we are measured against is their CPU**, so a DEFAULTED Tweedie
fit here does not run the same number of estimation iterations as a
defaulted CatBoost CPU fit, and a like-for-like comparison has to set the
count explicitly on both sides. `check-loss-oracle` does exactly that, which
is why deviation 62's numbers above are not confounded by it.

Recorded here because it is the second time Tweedie's defaults have differed
between their arms in a way that only shows up in a comparison, and because
STANDING_ORDERS rule 5 -- same everything except the device -- is what makes
it matter.

## 85. The device evaluator refuses a multi-output model, because THEIRS does

`gbdt/models/cuda/evaluator.mojo` was one-dimensional and would have
silently predicted the FIRST class's approxes from a MultiClass model,
because its tree walk reads `leaf_values[leaf]` where such a model stores
`n_leaves * dim`. Nothing routed a MultiClass model to it, which is exactly
the condition PORTING_RULES 8 names.

The fix was NOT to make it multi-dimensional. Their own GPU evaluator
refuses the case (`libs/model/cuda/evaluator.cpp:28`):

    CB_ENSURE(ModelTrees->GetDimensionsCount() == 1,
              "Model is not one-dimensional, GPU evaluation is not
               supported yet");

and their kernel agrees structurally: `EvalObliviousTrees` advances
`leafValues += (1 << curTreeDepth)` per tree (`evaluator.cu:222`) -- one
value per leaf, no dimension stride -- and accumulates into a scalar per
document. `ApproxDimension` appears in that file only inside
`ProcessResults`, the prediction-type post-processing, and never in the
tree walk.

So there is no multi-dimensional path of theirs to port here, and building
one would be inventing. `pack_model_for_evaluator` now refuses with their
message and a citation to their line.

MULTICLASS PREDICTION HAS ITS OWN APPLY and always did:
`predict_multi_floats` goes through `compute_bins_and_add_kernel`, which IS
multi-dimensional (DEVIATION 81) and is gated by `check-multiclass-train`.

NOTE WHICH MODELS ARE ACTUALLY REFUSED. `dim` is `numClasses - 1`, so a
TWO-class MultiClass fit has `dim == 1` and is an ordinary one-dimensional
model that the evaluator packs normally. Only three classes and up are
refused. The check gates both sides, and the first version of that gate --
which expected every MultiClass model to be refused -- failed at two
classes, correctly.


## 86. MultiClassOneVsAll: the same chassis, four differences that all matter

PORT OF `MultiClassOneVsAllValAndFirstDerImpl` (`multilogit.cu:613-673`) and
`MultiClassOneVsAllSecondDerImpl` (`:675-704`), plus the `StochasticDer` arm
that reaches them (`multiclass_targets.cpp:46-49`).

It rides everything MultiClass built -- the multi-plane cursor, the
multi-dimensional model and text format, `predict_multi_floats` -- and
differs in four places, none of which is optional:

1. **NO PINNED CLASS.** `GetDim()` returns `NumClasses`, not
   `NumClasses - 1` (`multiclass_targets.h:129-134`). So `SingleBinDim()
   == cursorDim`, the gradient is `(*gradient) = *DerAtPoint` with no
   reconstructed last component, `MakeEstimationResult` is the identity,
   and there is NO GAUGE TO FIX. A check asserting shift-invariance here
   would be asserting something false.
2. **DIAGONAL HESSIAN.** `GetHessianType()` is `Diagonal`
   (`:118-123`), so `HessianBlockSize()` is 1 and the walker takes its
   diagonal arm -- no Cholesky runs. Their generic blocked path reaches
   the same place at `blockCount == rowSize`, degenerating to one launch
   at row 0 producing `rowSize` columns, which IS the OneVsAll second-der
   kernel writing every plane at once.
3. **NO `MultiLogitOptimization`.** It is set only for MultiClass
   (`multiclass_targets.cpp:32-35`), so the score kernel needs no extra
   leaf contribution: the histogram already carries every class plane.
4. **NO MAX SUBTRACTION**, because each plane is its own sigmoid rather
   than a shared softmax. Overflow is handled by their `isfinite(expVal)`
   fallback, exactly as `CrossEntropyImpl` does.

**AND ONE CONSTANT THAT IS NOT THE ONE NEXT DOOR.** Its per-plane
arithmetic is `CrossEntropyImpl`'s term for term, so reusing that kernel is
tempting -- and would be wrong: their `ClipProb`
(`cuda_util/kernel/kernel_helpers.cuh:228-230`) clamps the probability at
**1e-7**, where `CrossEntropyImpl`'s inline clamp is **1e-40**
(`pointwise_targets.cu:354`). `checks/multilogit_check.mojo` gates both
halves of that: the two kernels must AGREE per plane on a fixture inside
+-6 where neither clamp bites, and must DISAGREE at an approx of -40, where
they read 1.0e-7 and 4.2e-18 -- ten orders of magnitude, which is the clamp
being real rather than assumed.

That anchor's first version read zeros from both kernels, because it filled
ONE host staging buffer three times between three `enqueue_copy` calls.
An enqueue is a promise, not a read; the copies raced the refills. Three
buffers now.

## 87. `use_best_model` without a test set: they warn, this refuses

`UpdateUseBestModel` (`libs/train_lib/options_helper.cpp:109-112`):

    if (!hasTest && outputFilesOptions->UseBestModel) {
        CATBOOST_WARNING_LOG << "You should provide test set for use best
            model. use_best_model parameter has been switched to false
            value." << Endl;
        outputFilesOptions->UseBestModel = false;
    }

`gbdt/train.mojo` raises there instead.

**The price**: a caller who passes `use_best_model=1` with no eval set gets
an exception where CatBoost gets a model. That is a real behavioural
difference and it is the reason this is numbered.

**Why it is worth paying.** Their warning goes to a console that a person
running `catboost fit` is watching. This is a library call whose result is a
model object: the switch-off leaves no trace on it, so the caller receives
the LAST-iteration ensemble while believing they asked for and received the
best-iteration one. There is nothing on the returned value to check -- no
field says "your option was ignored" -- and the two models differ in exactly
the way the caller was trying to avoid. A silent downgrade of an explicit
request is the one case where their leniency cannot survive the change of
medium.

**It applies only to the EXPLICIT arm.** Unset stays unset and resolves by
their own data-dependent rule (`:106-108`), so the default path is
bit-identical to theirs and nothing that does not name the option can hit
this.

`checks/overfitting_detector_check.mojo` gate 5 pins the refusal, beside
the same gate for `od_type` without an eval set, which this port already
refused for the same reason.

## 88. Ordered boosting is not in the learner this port mirrors, and their file says so

Scoping, 2026-08-21, before any code. **`boosting_type=Ordered` cannot be
added to `gbdt/methods/doc_parallel_boosting.mojo`, because CatBoost does not
have it there either.**

`catboost/cuda/train_lib/train_template_pointwise_greedy_subsets_searcher.h`
opens with their own comment:

    /*
    * New implementation of doc-parallel training with support for any type of
    * trees and multiclassification
    * But no ordered boosting
    */

and both of its entry points refuse it outright (`:34`, `:79`):

    CB_ENSURE(catBoostOptions.BoostingOptions->BoostingType ==
              EBoostingType::Plain, "Only plain boosting is supported in
              current mode");

`TGreedySubsetsSearcher` is the weak learner this repository ported. So the
learner we mirror is CatBoost's PLAIN learner, and adding ordered boosting to
it would not be porting CatBoost -- it would be inventing a variant they
decided not to build.

### Where theirs actually is

`ELossFunction::RMSE` on GPU with symmetric trees does not reach that file at
all. `pointwise.cpp` registers `TGpuTrainer<TPointwiseTargetsImpl>` from
`train_template_pointwise.h`, which dispatches on the DATA PARTITION
(`:27-48`):

    DataPartitionType == FeatureParallel  ->  TDynamicBoosting
                                              + TFeatureParallelPointwiseObliviousTree
    otherwise                             ->  TBoosting (doc_parallel_boosting.h)
                                              + TDocParallelObliviousTree

and `EDataPartitionType` defaults to **FeatureParallel**
(`boosting_options.cpp:25`). `UpdateDataPartitionType`
(`cuda/train_lib/train.cpp:73-84`) moves it to DocParallel only when boosting
is Plain. Boosting type in turn defaults to **Ordered** on GPU for every
non-multiclass loss (`catboost_options.cpp:803-807`).

**So CatBoost's shipped GPU default for RMSE is feature-parallel dynamic
boosting, and this port is their `boosting_type=Plain` arm.** That is worth
saying plainly: it is the arm you get from CatBoost by asking for Plain, by
training multiclass, or by choosing a non-symmetric grow policy -- not a
reduced version of their default, but a different one of their two.

### What ordered boosting is, in their code

`CreateFolds` (`dynamic_boosting.h:189-223`) cuts the already-permuted
documents into NESTED PREFIX folds that grow geometrically by
`fold_len_multiplier` (2.0) from `min_fold_size` (100):

    fold 0   estimate [0, m)              evaluate [m, m*r)
    fold i   estimate [0, prev.Right)     evaluate [prev.Right, prev.Right*r)

roughly `log2(n/m)` of them. Each fold and permutation gets its OWN cursor
(`TFoldAndPermutationStorage`, `:97-118`). The structure search then adds ONE
TASK PER FOLD (`:314-330`): gradients from the fold's prefix cursor, score on
the samples after it. That is the don't-look-ahead property -- a document's
gradient never comes from a model that has seen it.

The `Plain` arm of the same file collapses to a single fold covering
everything (`:205-209`), which is what this port already does.

### The price, counted

| what | lines |
|---|---|
| `dynamic_boosting.h` | 673 |
| `feature_parallel_pointwise_oblivious_tree.{h,cpp}` | ~150 |
| `oblivious_tree_structure_searcher.{h,cpp}` | 713 |
| `pointwise_optimization_subsets.{h,cpp}` | 184 |
| `pointwise_scores_calcer.h` + `pointwise_score_calcer.cpp` | 125 |
| `histograms_helper.{h,cpp}` | 519 |
| `helpers.{h,cpp}` | 277 |
| `pointwise_kernels.{h,cpp}` | 699 |
| `add_oblivious_tree_model_feature_parallel.{h,cpp}` | 133 |
| `feature_parallel_dataset{,_builder}.{h,cpp}` | 583 |

about **4,000 lines of host code**, plus the feature-parallel histogram
kernels under `methods/kernel/`, which are a SECOND histogram family --
`greedy_subsets_searcher/kernel/` is the one this repository ported, and
neither is a configuration of the other.

**The multi-task structure searcher is the part that has no counterpart
here.** Ours searches one (target, dataset) pair. Theirs takes N fold tasks
and sums their scores, which is why their greedy-subsets learner refuses
ordered boosting rather than approximating it.

### What is worth doing instead, and first

`permutation_count > 1` inside `doc_parallel_boosting.h`, which this port
already mirrors and already runs at one permutation. It is the same
machinery -- per-permutation cursors, per-permutation ensembles, structure
searched on a permutation that is not the estimation one -- without the fold
prefixes, and CatBoost reaches it in PLAIN mode as soon as the data has a
categorical feature (`cuda/train_lib/train.cpp:100-108`). Since CTRs landed
here, that is now a live divergence on every categorical fit, and it is
recorded as such in `doc_parallel_boosting.mojo`'s audit list.

Ordered boosting is not refused here yet; nothing in this port accepts a
`boosting_type` at all. It should stay that way until either the option
exists and refuses Ordered by name, or the learner behind it does.

## 89. A permutation costs a whole compressed index here, not just its CTR columns

Their builder splits the features by permutation dependence and gives each
permutation a compressed-index dataset holding ONLY the dependent ones
(`doc_parallel_dataset_builder.cpp:104-124`):

    permutationIndependentCompressedDataSetId   float, one-hot, FeatureFreq
                                                -- built once, shared
    dataSet.PermutationDependentFeatures        Borders CTRs -- one per
                                                permutation

This port packs every column of a fit into ONE compressed index and the
histogram kernels read it as one buffer, so there is nowhere to put a
per-permutation slice. `train` therefore builds `permutation_count` COMPLETE
indices.

**The price is memory, and only on a categorical fit**, because
`permutation_count` collapses to 1 without a CTR-bearing feature
(`cuda/train_lib/train.cpp:99-108`). At their default of 4 it is 4x the
compressed index where theirs is 1x plus four copies of the dependent
columns alone. On a pool whose CTR columns are a small share of the width --
which is the common case, since a categorical feature becomes 4 CTR columns
among however many numeric ones -- the difference is close to the whole 4x.

It is not a correctness difference: the four indices hold exactly the values
their four datasets hold.

**What would fix it** is a compressed index that can be assembled from two
buffers, one shared and one per-permutation, which means teaching the
histogram kernels a second base pointer and a per-feature selector. That is
a change to the hottest kernel family in the repository, so it wants a
measurement showing the memory actually binds before it is made -- and the
measurement is available: a categorical fit that fits at
`permutation_count=1` and OOMs at 4 is the whole argument.

## 90. The per-permutation leaf partition is built on the HOST

Their leaves estimator hands the oracle a `bins` buffer -- one leaf index
per row, computed on the device by `ComputeBinsForModel`
(`doc_parallel_leaves_estimator.cpp:45-49`) -- and the oracle partitions off
it, on the device, itself.

This port's oracle takes rows ALREADY GROUPED BY LEAF, with per-leaf offsets
and sizes, because that is what the searcher produces as a by-product of
growing the tree and the gather kernels were written against that shape. So
for a dataset the tree was NOT grown on there is nothing to inherit, and
`partition_from_bins`
(`gbdt/methods/leaves_estimation/doc_parallel_leaves_estimator.mojo`) builds
the grouping with a **stable counting sort on the host**: bins back, count
into `2^depth` buckets, row order out.

**The price**, per permutation per tree: two host passes over `n_rows` and
one round trip each way. On unified memory the copies are memcpy rather than
bus traffic, so the host passes dominate; it lands on the fixed-cost side of
`ms/tree = a + b*rows`, which is the side this repository is already losing
on (`RESUME.md`).

It is STABLE deliberately. Rows keep ascending order within a leaf, so the
oracle's per-leaf float sums are formed in a fixed order. An unstable sort
would make a fit's leaf values depend on scatter order, and a model that
changes between two runs of the same command is not a model this repository
can gate.

**The learn permutation does not go through it at all**: it keeps the
searcher's own partition, so a one-permutation fit is unchanged to the bit.
That is not an optimization either -- routing it through a second grouping
would re-order the sums and move every number in the fit for nothing.

**What would remove it** is a device radix sort on a `depth`-bit key. The
pieces are in this repository already (`gbdt/gpu_util/kernel/segmented_sort.mojo`
and the LSD radix sort behind the searcher), so this is a wiring job, not a
new kernel -- held until a measurement shows the host sort costing something
that matters.

## 91. The device-count answer: at one device the two layouts ARE the same, and the searcher this port mirrors is CatBoost's MULTICLASS symmetric learner

Scoping, 2026-08-21, no code. `NEXT_TWO.md` staged item 1 asked one question --
how much of `TFeatureParallelDataSet` survives at device count 1 -- and said to
write the answer here whichever way it came out. It came out four ways, and two
of them correct things this repository has been saying.

### A. The two layouts produce a BIT-IDENTICAL compressed index at one device

`feature_layout_common.h:176-189` defines the whole difference between them as
a mapping table, and only one row differs:

    TFeatureParallelLayout          TDocParallelLayout
      Features  Stripe                Features  Stripe
      Samples   MIRROR                Samples   STRIPE     <- the only difference
      CIndex    Stripe                CIndex    Stripe
      Weights   Mirror                Weights   Mirror

and at `GetDeviceCount() == 1` that row does not differ either.
`TStripeMapping::SplitBetweenDevices(n)` and `TStripeMapping::RepeatOnAllDevices(n)`
(`cuda_lib/mapping.h:256-280`) both return the single slice `[{0, n}]`, which
is what `TMirrorMapping(n)` is.

Walking `TCudaFeaturesLayoutHelper` on both sides
(`feature_layout_feature_parallel.h:19-105`, `feature_layout_doc_parallel.h:26-104`),
every step coincides at one device:

| step | feature-parallel | doc-parallel | at 1 device |
|---|---|---|---|
| `CreateLayout` | `SplitBetweenDevices(featureCount)` | `RepeatOnAllDevices(featureCount)` | `[{0,F}]` both |
| `CreateDocLayout` | `TMirrorMapping(docCount)` | `SplitBetweenDevices(docCount)` | `[{0,N}]` both |
| feature order | `Shuffle(TRandom(0))` then sort WITHIN each device slice | `Shuffle(TRandom(0))` then sort the WHOLE vector | one device slice IS the whole vector -- same `std::sort`, same comparator, same input |
| `AddDeviceFeatures` | `(devSlice, cindexOffset, FULL docCount)` | `([0,F), cindexOffset, deviceSlice docCount)` | the same three arguments |
| `CudaFeaturesHost[i]` | `allFeatures[i]` | `allFeatures[dev * F + i]` | `dev == 0`, so `allFeatures[i]` |
| `FoldsHistogram` | `ComputeFoldsHistogram(devSlice)` | `ComputeFoldsHistogram()` | `feature_layout.cpp:34-36` defines the no-arg form AS `ComputeFoldsHistogram(TSlice(0, FeatureIds.size()))`, which is `devSlice` |
| `CudaFeaturesDevice` | `CreateFromSizes(trainFeatureSlicesSizes)` | `CreateLayout(features.size() / devCount)` | `[{0, features.size()}]` both |
| `BinFeatures` | `BuildBinaryFeatures([0, allFeatures.size()))` | `BuildBinaryFeatures([0, F))` | FP pushes each feature once, DP pushes `devCount * F` = `F` -- same slice |
| `HistogramsMapping` | `CreateMapping<TStripeMapping>(BinFeatureCount)` | `RepeatOnAllDevices(BinFeatures.size())` | same single slice |

**So `feature_layout_feature_parallel.h` costs zero. This port already builds
the compressed index it would build.** The 583-line dataset estimate in
`NEXT_TWO.md` was priced against multi-device bookkeeping that does not exist
on one GPU, and what remains of `TFeatureParallelDataSet` beyond that is
`InverseIndices`, the CTR targets, the cat-feature dataset and the samples
grouping -- all of which this repository already has or does not use.

The one thing worth carrying is the TYPE distinction, because it records what
CatBoost meant: doc-parallel splits ROWS, feature-parallel splits FEATURES, and
on two devices those are different programs. A port that erases the distinction
is correct today and unportable to a second device tomorrow.

### B. There are THREE GPU searchers, not two, and `NEXT_TWO.md` had it wrong

The note said "CatBoost has two GPU learners". It has three, and the third is
the one this repository ported:

| searcher | file | mapping | folds | tree CTRs |
|---|---|---|---|---|
| `TFeatureParallelObliviousTreeSearcher` | `oblivious_tree_structure_searcher.{h,cpp}` (713) | Mirror | YES | YES |
| `TDocParallelObliviousTreeSearcher` | `oblivious_tree_doc_parallel_structure_searcher.{h,cpp}` (295) | Stripe | no | no |
| `TGreedySubsetsSearcher` | `greedy_subsets_searcher/` | Stripe | no | no |

The first two SHARE their SCORING stack -- `pointwise_scores_calcer.h`,
`histograms_helper.{h,cpp}`, `pointwise_kernels.{h,cpp}`, and the
`pointwise_hist2*` kernel family.

**CORRECTED 2026-08-21 BY PORTING IT. This paragraph used to say the two were
"one implementation templated on the mapping, with two `TSubsetsHelper`
specializations whose `CreateSubsets` differ in exactly three lines", and it
was wrong in three ways -- see DEVIATION 120 for the source citations.
`TSubsetsHelper<TMirrorMapping>` has no `CreateSubsets` at all; the two
`Split`s are different code calling different kernels; and the feature-parallel
arm needs `docBins`, a per-document bit-packed array with no counterpart on the
doc-parallel path, which cost 1,067 lines and two kernels to port.** The
sentence is deleted rather than annotated because `PORTING.md` 119 and
`NEXT_TWO.md` both priced rung 2 off it.

What IS three lines is the initial subsets state, and it is worth keeping
because it is still the whole of ordered boosting at this level
(`pointwise_optimization_subsets.cpp:12-14` vs
`oblivious_tree_structure_searcher.cpp:36-37`):

    Stripe:  FoldCount = 0;               FoldBits = 0;                   bins all zero
    Mirror:  FoldCount = initParts.size(); FoldBits = IntLog2(FoldCount);  bins from WriteFoldBasedInitialBins

And `FoldCount` is not even read: their two `Fit`s pass `subsets.FoldCount`
(`:93`) and the literal `1` (`:40`) into the same score-calcer argument, and at
one task those are the same number. `FoldBits`, which IS read everywhere, is 0
on both.

**That is ordered boosting, in full, at the subsets level: the fold id occupies
the LOW bits of the bin and the depth bits sit above it**, which is why every
call downstream reads `CurrentDepth + FoldBits` rather than `CurrentDepth`.

The third searcher -- ours -- has its own histogram family
(`greedy_subsets_searcher/kernel/`) and shares none of that.

### C. `TDocParallelObliviousTree` is a WEAK-LEARNER SWAP into the loop we have

`train_template_pointwise_greedy_subsets_searcher.h:35-36` and
`train_template_pointwise.h:49-50` build the same thing around different weak
learners:

    TBoosting<TTargetTemplate, TGreedySubsetsSearcher<TModel>>      <- ported
    TBoosting<TTargetTemplate, TDocParallelObliviousTree>           <- not ported

Same `TBoosting` (`doc_parallel_boosting.h`), same `TDocParallelDataSet`, same
`TObliviousTreeLeavesEstimator`, same `TAddModelDocParallel`. The weak learner
itself is 85 lines of glue
(`doc_parallel_pointwise_oblivious_tree.h`). So the boosting loop, the dataset,
the estimator and the apply are all already here; what is missing is the
searcher and the histogram family under it.

### D. `pointwise_hist1.cu` is DEAD IN THE UPSTREAM -- 935 lines that cost nothing

`ComputeHist1` is registered as a kernel (`pointwise_kernels.cpp:8`) and
dispatched (`:95-120`), and **nothing in the CatBoost tree calls it.** A search
for the symbol across every `.cpp`, `.h` and `.cuh` in the repository returns
only `pointwise_hist1.cu` itself and its own wrapper. It goes in
`gbdt/NOT_IMPLEMENTED.tsv` as unreachable upstream, not as a gap.

### E. What this does to the price

`NEXT_TWO.md` priced objective 1 at ~4,000 host lines and ~3,700 kernel lines
as a single indivisible prerequisite. The correct shape is a ladder with a
gate on every rung, and the first rung is most of the value:

| rung | what | new lines | why it is gateable alone |
|---|---|---|---|
| 1 | the pointwise stack + `TDocParallelObliviousTree` | ~1,700 host + ~2,500 kernel | its histograms must agree cell for cell with the greedy-subsets histograms this repo already has, on the same rows and the same compressed index |
| 2 | `TFeatureParallelObliviousTreeSearcher` at ONE fold | ~400 | must reproduce rung 1 to the bit, since at `FoldBits == 0` and one device it is the same program |
| 3 | folds + `TDynamicBoosting` = ordered boosting | ~700 | fold boundaries are closed-form in `n`, `min_fold_size`, `fold_len_multiplier` |
| 4 | tree CTRs | 1,640 | the tensor hash against a `tools/` oracle, before any fit |

Rung 2 is the one the layout finding collapses: with the compressed index
proven identical at one device, the feature-parallel searcher differs from the
doc-parallel one by the fold bits and the tree-CTR block, and nothing else.

### F. THE CORRECTION THAT MATTERS MOST: which learner CatBoost runs for symmetric trees

This is a claim this repository has been making loosely and it needs pinning
down. `GetTrainerFactoryKey(loss)` defaults its second argument to
`EGrowPolicy::SymmetricTree` (`train_lib/train.h:86`), so the registration
tables read:

| loss | grow policy | boosting | GPU searcher |
|---|---|---|---|
| RMSE, Logloss, MAE, ... | SymmetricTree | Ordered (their default) | `TFeatureParallelPointwiseObliviousTree` |
| RMSE, Logloss, MAE, ... | SymmetricTree | Plain | `TDocParallelObliviousTree` |
| RMSE, Logloss, MAE, ... | Lossguide / Depthwise | Plain only | `TGreedySubsetsSearcher<TNonSymmetricTree>` |
| **MultiClass, MultiClassOneVsAll, MultiRMSE, MultiLogloss, MultiCrossEntropy, RMSEWithUncertainty** | **SymmetricTree** | **Plain only** | **`TGreedySubsetsSearcher<TObliviousTreeModel>`** |

The last row is `multiclass.cpp:5-14`, which includes
`train_template_pointwise_greedy_subsets_searcher.h` and registers at the
default grow policy with the default `TModel = TObliviousTreeModel`. And
`pointwise_non_symmetric.cpp:36` carries the registration that would send
single-target symmetric trees down the same path, **commented out**:

    //    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRmseOT(GetTrainerFactoryKey(ELossFunction::RMSE, EGrowPolicy::SymmetricTree));

So the searcher this repository ported is a live, intended, supported CatBoost
GPU symmetric-tree path -- **it is the one CatBoost uses for MULTICLASS
symmetric trees** -- and it is NOT the one CatBoost reaches for single-target
RMSE symmetric trees, in either boosting mode.

**What that does and does not mean.** It does not mean our trees are wrong:
`tools/catboost_oracle.py` compares split-for-split against CatBoost's own
dumped decisions and matches 48/48 at three border budgets. But that oracle
passes no `task_type`, so it runs CatBoost's **CPU** learner. The honest
statement is therefore:

* our split selection reproduces CatBoost's CPU oblivious learner, 48/48;
* our searcher mirrors CatBoost's GPU multiclass symmetric learner;
* **we have never compared against either of CatBoost's GPU single-target
  oblivious searchers, because neither is ported.**

Rung 1 above is what closes that, and it is the reason to do it.


## 92. A divergent barrier is benign until the barrier is load-bearing, and then it silently drops points

Measured 2026-08-21 while porting the pointwise histogram loop, and it puts a
question mark over `PORTING.md` 11 without answering it.

### What was expected

Item 11 is why two histogram families here run ONE iteration count for the
whole block instead of CatBoost's per-thread counts. CatBoost syncs a
`tiled_partition<8>` inside `AddPoint`
(`pointwise_hist2_one_byte_5bit.cu:79`, `:108`, `:147`); Mojo exposes only a
threadgroup `barrier()`; so a thread that runs out of points early walks past
a barrier its neighbours are still waiting on. Item 11 states this was
OBSERVED:

    "It is not an edge case: a 64-row partition over a 512-thread block gives
     warp 0 one iteration and warps 1 to 15 zero. The measured symptom was
     every feature's histogram reading 0.0."

`compute_point_hist2_loop.mojo` therefore got the same treatment: a per-block
`max_iters`, with threads past their own count contributing `(bin 0, 0.0,
0.0)`. Adding 0.0 changes no sum, so it is a scheduling change.

### What actually happened

**The gate written to catch a divergent barrier cannot catch one, because a
divergent barrier does not fail on this device.**

`checks/pointwise_loop_check.mojo` L6 runs the whole sweep through an
accumulator that barriers inside `add_point`. Reverting `compute_histogram_2`
to CatBoost's per-thread count -- which genuinely diverges, 8 iterations on
the thread at `i == 0` against 7 at `i == 254` -- leaves **all 160 cases
exact**.

`checks/divergent_barrier_probe.mojo` then tested it directly, at item
11's own shape: a 512-thread block, a barrier inside a loop whose count is
`(n - tid + block - 1) / block`, so warp 0 runs one iteration and warps 1-15
run zero. All 512 slots correct, at n = 64, 100, 511, 512, 513, 1000, 2000.
No hang, no zeros. `pixi run check-divergent-barrier`.

### What that does and does not mean

It does NOT mean divergent barriers are safe. CUDA's `__syncthreads` and
Metal's `threadgroup_barrier` both require uniform execution, and "undefined
happens to work on one M4 today" is not something a port builds on. **The
uniform path stays**, justified by the specification and priced at one
predicate per point.

It does NOT mean item 11's observation was invented. But it is no longer
supported by anything reproducible, and there is a better-fitting suspect.
**Twice while writing these checks, a RACING per-thread tally produced
exactly the reported symptom** -- cells reading zero or a fraction of what
they should, with nothing wrong in the loop -- and the second time it turned
every gate red against a loop that was correct. `PORTING.md` 12 records a
third instance of the same symptom from a completely different cause (a
reused async staging buffer). "Every histogram cell read 0.0" has now been
produced by three mechanisms in this repository and by a divergent barrier
zero times.

That is a hypothesis about item 11, not a finding, and it is written down so
someone can test it rather than inherit it. What is a finding: **item 11's
stated mechanism is currently unreproducible, and the checks that appear to
cover it do not.**

### AND THEN IT REPRODUCED, three hours later, in the 5-bit accumulator

Everything above stands as written and its CONCLUSION was wrong. Landing
`TPointHist<0,0,BlockSize>` produced the failure on the first run:

    scalar n=1   256 cells exact
    scalar n=4   256 cells exact
    uint2        115 cells WRONG, every one of them LOW
    uint4         73 cells WRONG, every one of them LOW

The cause was a divergent barrier, and it was in code this entry had already
looked at and passed over. When `compute_histogram`'s BODY was converged, the
peel loops were left as CatBoost writes them:

    for (; colId < 128; colId += blockDim.x / HIST_BLOCK_COUNT)

At `BLOCK_SIZE = 256` that loop is entered by threads 0-127 and skipped
entirely by 128-255, and `AddPoint` takes eight threadgroup barriers. So half
the block ran eight barriers the other half never reached, and the two halves
came out of the peel eight barriers out of phase. Points went missing.
Converging the peel -- every thread runs `ceil(span / col_step)` iterations
and contributes a zero point when its column is out of range -- makes all
four entry points exact and identical.

Coverage was never wrong. Both versions read columns 0-127 exactly once.

### The refined mechanism, which is what should have been written first

A divergent threadgroup barrier is benign when nothing depends on the
synchronisation, and corrupts when the barrier is LOAD-BEARING for ordering
shared writes.

    divergent_barrier_probe   each thread owns its slot; the barrier orders
                              nothing; 512/512 correct at seven sizes
    PointHist5                eight threads share an inner histogram copy and
                              the barrier is the ONLY thing holding their two
                              half-writes apart; points vanish

Both observations are real and neither generalises without the other. Item
11's mechanism is correct. This entry's earlier conclusion -- that it "will
not reproduce" -- was drawn from a probe too simple to contain the thing it
was probing for, which is the same mistake as gating a histogram with uniform
data.

### The rule this leaves behind, and it is not the one drafted first

**Gate a kernel against a REAL accumulator, not a convenient one.**
`checks/pointwise_loop_check.mojo` gives every thread a private tally, so
it measures coverage, and coverage was correct in both the broken and the
fixed version -- all 160 cases, all six gates, at block 128 AND at block 256.
It could not have found this. `checks/pointwise_hist2_5bit_check.mojo`
found it on its first run, because eight of its threads share a slot and the
barrier is what keeps them apart.

A private-slot tally is the histogram equivalent of uniform test data
([[uniform-test-data-hides-permutation]]): it verifies the sum and nothing
about the contention that the code exists to manage.

The secondary rule still holds: a gate whose sabotage does not move it is not
coverage, and the check should say so rather than let a green tick imply
otherwise.

## 93. Metal has no THREADGROUP float atomics, and the 8-bit accumulator is the file that needs them

Probed 2026-08-21, porting `TPointHist<2,1,BlockSize>`.

`Atomic.fetch_add` on a `Float32` in `AddressSpace.SHARED` does not compile:

    Failed to create compute pipeline state (GPU machine code generation):
    Unsupported local float atomic operation for given target.

**This is narrower than "Metal has no float atomics" and the difference
matters.** Global float atomics work. Warp primitives work. It is
specifically the threadgroup float case, which is the one every shared
histogram wants. `metal-hardware-gaps` already carried this for the
greedy-subsets family; it is restated here because a reader arriving at
`pointwise_hist2_one_byte_8bit.mojo` will not have read that.

### Why only the 8-bit accumulator cares

The other three never need an atomic, and their reason is structural. A warp
slice is 1024 floats, so wider bins buy fewer private copies:

    bits   copies   threads per copy   how they avoid collisions
      5       4             8          they cannot collide: (f, flag) is
                                       distinct across the 8
      6       2            16          `writeFirstFlag`, two turns
      7       1            32          `writeTime`, four turns
      8       2            64          NEITHER -- CatBoost calls atomicAdd

At 8 bits `OUTER_HIST_BITS_COUNT = 2` makes the slice 4096 wide, so a
256-thread block has 2 slices for 8 warps and 64 threads land on each copy.
Turn-taking would need 64 turns.

### The substitution

Int32 fixed point through `histogram_utils.hist2_quantize` and
`hist2_dither` -- the SAME quantizer the greedy-subsets family uses, not a
second one. There must be exactly one dithered quantizer in this tree or the
two families will drift apart in the last bits and nobody will know which is
right.

`PointHist2.add_point` grew a `row` argument to carry this, and **it is the
document id, not the position.** A document's position in the index array is
reordered at every level; its id is not. Key the dither on the position and
the same row quantizes differently at different depths, which breaks
`parent == child + sibling` -- and that identity is what lets the partial
pass compute one child and subtract for the other. The 5/6/7 accumulators
ignore the argument.

### What it costs and what it buys

COSTS quantization error, which the dither makes zero-mean and
O(sqrt(rows)) rather than O(rows). Measured in
`checks/pointwise_hist2_8bit_check.mojo` B3: signed error **-0.0105 per
row-stat** over 28,000 of them, and 680 of 1,224 non-exact cells erring
negative. Truncation, sabotaged in, gives **1,224 of 1,224 negative**.

BUYS reproducibility CatBoost does not have. Integer addition is
associative, so the shared accumulation is order-independent. The check
proves it the hard way: the four loop entry points assign DIFFERENT points
to different threads, so their deferred runs differ completely, and every
one of the 2,048 cells still lands on the same total.

### The gate that would not have worked

Scale 1.0 with integer stats makes the quantizer exact -- `floor(x)` plus a
fraction compare cannot move an integer -- so B1 compares all 2,048 cells
with no numeric slack. That is what catches placement. It does NOT catch a
biased quantizer, because at scale 1.0 truncation and dither agree. B3 is
the other half, and its sharp gate is the SIGN SPLIT rather than the
magnitude: a dither rounds up about half the time, truncation never does,
and that test is independent of how many rows are in a cell.

Sabotages run, each caught by a different gate:

    Reduce's pending flush deleted     B1, 48 cells, short by 209,152
    flush attributes to the NEW bin    B1, 1,224 cells, SHORT BY ZERO
    quantizer truncates                B3 sign split, 1224/1224
    dither constant across rows        B3 sign split, 1224/1224

The second is the one to remember. Net zero: the mass moved rather than
vanished, and every total in the histogram is correct. A check that summed
would have passed it.

## 100. The pointwise host launchers use the kernel matrix's block sizes, not CatBoost's literals

THEIRS: `const int blockSize = 384;` for the one-byte family
(`pointwise_hist2_one_byte_templ.cuh:238`) and `const int blockSize = 768;`
for both small-bin families (`pointwise_hist2_binary.cu:141`,
`pointwise_hist2_half_byte.cu:145`).

OURS: `PW_HIST2_BLOCK` (256) and `PW_HB_BLOCK` (512), imported from the kernel
files rather than restated in `gbdt/methods/pointwise_kernels.mojo`, so a
launcher cannot drift from the kernel it launches.

MEASURED REASON: the accumulators are sized per thread -- 32 floats each for
the one-byte family, 16 for the small-bin one. At 768 the small-bin
accumulator wants 16 x 768 x 4 = 49,152 bytes of threadgroup memory against
Apple's 32,768; at 384 the one-byte accumulator wants 32 x 384 x 4 = 49,152
against the same limit. The matrix resolves both, and 512 is a FLOOR as well
as a budget: `pointwise_hist2_half_byte_template.mojo` carries a
`comptime assert` because that family's `Reduce` folds its warp slices under
`if (threadIdx.x < 512)`.

WHAT IT DOES AND DOES NOT CHANGE. It does not change any grid: every
`numBlocks` expression is in FEATURES and PARTS, never in threads. It does not
change the multiplier -- `EstimateBlockPerFeatureMultiplier` counts blocks. It
DOES change how many blocks the scan launch needs
(`ceil(featureCount / blockSize)`), and the number of warp slices each
accumulator folds, which is a float summation order already recorded at the
accumulators.

## 101. `exit(1)` and `CB_ENSURE_INTERNAL` become raised errors

THEIRS: the multiplier ladder ends `} else { exit(1); }`
(`pointwise_hist2_one_byte_templ.cuh:266`, `_binary.cu:174`,
`_half_byte.cu:175`) -- a bare process abort with no message. The `histCount`
guards end `CB_ENSURE_INTERNAL(false, ...)` (`pointwise_hist2.cu:99`, `:129`).

OURS: both raise, and the multiplier one names the offending value.

REASON: not a choice about behaviour. Mojo has no `exit` inside a `def` that
already `raises`, and a library that kills the process instead of returning an
error cannot be gated. Both are unreachable by construction --
`EstimateBlockPerFeatureMultiplier` only doubles from 1 and is clamped to 64,
so it is always a power of two in [1, 64]; `histCount` is 2 at the only
caller. `checks/pointwise_dispatch_check.mojo` F7 sweeps 245
configurations, finds 12 that return 128 BEFORE the clamp, and asserts every
clamped value is one of the seven -- so the clamp is live and the raise is not.

## 102. `TComputeHist2Kernel` becomes a function

THEIRS: `TComputeHist2Kernel : TStatelessKernel` holds thirteen members,
declares `Y_SAVELOAD_DEFINE` over all of them, registers in a global table as
`REGISTER_KERNEL(0x420000, ...)`, and is dispatched by
`LaunchKernels<TKernel>(targets.NonEmptyDevices(), ...)`.

OURS: `compute_hist2(...)`, taking the same thirteen values as arguments, on
one device.

REASON: `Y_SAVELOAD_DEFINE` and `REGISTER_KERNEL` serialize a kernel
invocation so it can be sent to another PROCESS -- CatBoost's multi-host path.
There is no such path here and porting the table without it would be porting a
name. `TCudaBufferPtr<T>` carries a pointer and a size; ours are separate
arguments (PORTING.md 9). `NonEmptyDevices()` is the multi-device fan-out,
settled by PORTING.md 91 A: at device count 1 the layouts coincide.

TWO CONSEQUENCES, both real ports of theirs:

* `TFoldsHistogram` (`gpu_data/folds_histogram.h`) is ported into
  `gbdt/methods/pointwise_kernels.mojo`, not `gbdt/gpu_data/`, because the
  one-byte fan-out cannot be written without it and that lane owned two files.
  27 lines; move it to `gbdt/gpu_data/folds_histogram.mojo` the moment
  anything else needs it.
* `TComputeHist1Kernel` is NOT ported: `pointwise_hist1.cu` is dead in the
  upstream (PORTING.md 91 D, `gbdt/NOT_IMPLEMENTED.tsv`).

INHERITED, NOT NEW: the scan launch's `numBlocks.x` loses their `* 32`,
because PORTING.md 8 replaced their warp-per-feature scan with one thread per
feature. The grid is the last place that substitution surfaces. And the 8-bit
path takes a `fixed_scale` their kernels have no parameter for -- DEVIATION 93;
this layer only threads it through.

### 102a. THE HOST ECHO OF THE `15`, verified from both sides

`pointwise_kernels.cpp:57-60` dispatches the one-byte family as

    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 4, 5)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 6, 6)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 7, 7)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 8, 8)

**The 5-bit kernel's feature count spans bits FOUR AND FIVE**, and the ranges
are not uniform. That is the same fact as the device-side
`lowerBound = BITS > 5 ? upperBound / 2 : 15` recorded at the driver, seen
from the other end: a feature with 16 folds has `IntLog2 == 4` and belongs to
the 5-bit kernel. Reading the range as `5,5` under-counts, the multiplier
comes out too small, and the 16-fold features go unsplit.

The two halves were found independently -- the device bound while porting the
driver, the host range while porting the launcher -- and they corroborate.
PORTING.md 91 F's fixture group 2 exists for the device half;
`pointwise_dispatch_check.mojo` F6 exists for this one.

## 94. No float64 anywhere in the pointwise scorer; theirs accumulates in double

`gbdt/methods/kernel/pointwise_scores.mojo`, port of
`catboost/cuda/methods/kernel/pointwise_scores.cu` + `score_calcers.cuh`.

Metal has no fp64 at any cost, so every `double` in those two files is
`Float32` here. Theirs, by citation:

* `struct TPartitionStatistics { double Weight; double Sum; double Count; }`
  (`gpu_data/gpu_structures.h:113-116`) -- the per-partition totals every
  kernel subtracts the left child from.
* `void AddLeaf(double sum, double weight)` in all five calcers and
  `double Score; double DenumSqr;` in `TCosineScoreCalcer`
  (`score_calcers.cuh:22, 53, 83, 114, 152, 182-183`).
* `double scoreBeforeSplit`, `double l2`, `double scoreStdDev` as kernel
  arguments (`pointwise_scores.cu:50`, `:325-326`).
* `__shared__ volatile double localBuffer[BLOCK_SIZE]` and
  `Reduce<double, BLOCK_SIZE>` in `PartitionUpdateImpl` (`:637`, `:644`),
  which is where the partition totals are MADE. This one compounds: a
  Float32 total then feeds the Float32 subtraction above.

WHAT IT COSTS. The right child is derived as `part.Sum - sumLeft`
(`:263`, `:369`), the cancellation step of the whole scorer. Theirs cancels
in double and narrows afterwards; ours cancels in Float32 throughout. So a
split CHOICE can differ from CatBoost's whenever two candidates sit within
Float32 epsilon after cancellation, and the gap widens with `pCount` and
with `foldCount`. Same deviation the greedy-subsets scorer already carries;
recorded again because it is a different file with a different cancellation
depth.

NOT FAKED: no Float64 appears pretending to be theirs. The gate's host
reference is Float32 for the same reason, and its exactness (S1 and L1
report a worst relative discrepancy of 0.0) is measured, not assumed.

### 94a. THE TWO SCORERS IN THIS TREE HAVE OPPOSITE SIGNS

Verified at both sites while merging, and it is the single most dangerous
thing to assume about this file.

    pointwise_scores.mojo (this one)   FLT_MAX sentinel, `gain < bestGain`,
                                       LOWER IS BETTER -- CatBoost's own
    greedy_subsets_searcher/kernel/    larger is better, every comparison
      compute_scores.mojo              FLIPPED, one negation folded into the
                                       host (its docstring says so at :36-42)

Both are correct for their own callers. Neither is wrong. But a searcher
that reads a best-split record from one and compares it with the other's
convention picks the WORST split at every level and still returns a
well-formed tree.

## 95. The pointwise scorer's struct pointers become flat typed arrays

A Mojo kernel argument cannot be a pointer to a non-trivial struct, and
PORTING_RULES 4 already records that `enqueue_function` refuses derived
pointers as aliasing. The four struct arguments of `pointwise_scores.cu` are
passed as the flat arrays their C++ memory image already is:

| theirs | ours |
|---|---|
| `const TCBinFeature* bf` | `MutPointer[UInt32]`, 3 words: `[3b]=FeatureId`, `[3b+1]=BinId`, `[3b+2]=SkipInScoreCount`. `{ui32; ui32; bool;}` is 12 bytes with the bool in the low byte of word 3, so this is byte-identical, not a re-encoding |
| `const TPartitionStatistics* parts` | `MutPointer[Float32]`, 3 per entry: Weight, Sum, Count (Float32 by 94) |
| `const TDataPartition* parts` | `MutPointer[UInt32]`, 2 per entry: Offset, Size -- the same encoding `split_properties_helpers.mojo` already uses |
| `TBestSplitProperties* result` | TWO pointers: `result_ids` (UInt32, 2/block) and `result_scores` (Float32, 2/block). `result += blockIdx.x` becomes `2 * blockIdx.x` into each |

Their `TScoreCalcer calcer` is also a by-value kernel argument (`:250`,
`:481`). Mojo has no by-value user struct across the launch boundary, so the
calcer's CONFIGURATION crosses as scalars (`lambda_l2`, `meta_exponent`,
`normalize`, `score_std_dev`, `global_seed`) and the calcer is constructed
inside the kernel. Every decision that was host-side upstream stays
host-side, in particular the `MetaExponent` coin flip (`:507`), which is
`meta_exponent_draw` and is gated on both arms.

The five calcer CLASSES become one comptime-tagged `ScoreCalcer[
score_function]` -- PORTING_RULES 4's "tagged union, which is what their
worker switches on anyway", and their host does switch on it at `:483`.

## 96. `StreamLoad` and `__ldg` have no portable spelling

`ComputeSum` loads through `NKernel::StreamLoad` (`pointwise_scores.cu:27`),
which is `cub::ThreadLoad<cub::LOAD_CS>` -- PTX `ld.global.cs`, a
cache-streaming hint that evicts the line early so a one-pass sum does not
displace the working set. `LdgWithFallback` is
`cub::ThreadLoad<cub::LOAD_LDG>`, the read-only data cache path. Both are
NVIDIA cache-policy PTX with no counterpart on Metal.

* `LdgWithFallback` / `__ldg` -> `std.gpu.intrinsics.ldg`. Kept: it is the
  Mojo spelling of the same intrinsic and lowers to a plain load where the
  target has none, so it is language-level, not a library standing in for an
  algorithm.
* `StreamLoad` -> a plain `unsafe_load`. A HINT ONLY: `LOAD_CS` and a normal
  load return the same bytes, so the arithmetic is unchanged and only cache
  residency differs. Recorded rather than silently dropped.

The 16-wide manual unroll around it IS ported (`:23-32`) as a `comptime for`,
because the unroll fixes the ORDER of a float summation and collapsing it
into the tail loop would give a different sum. Their own comment attributes
it to nvcc 11.4+ refusing `#pragma unroll 16`; the reason it is ported is the
summation order, not the compiler. Gate R1 uses a 20000-row partition, which
exceeds `15 * 1024` and reaches the unrolled loop at their block size; a
500-row partition reaches only the tail. Sabotaging the unrolled stride moves
4 cells.

### 96a. Four things in `pointwise_scores.cu` that are theirs and look wrong

Transcribed as written, all verified against the source while merging.

1. **`GatherHistogramByLeaves` cannot reach leaves past 1023.** The launcher
   sets `numBlocks.z = ceil(leafCount / 1024)` (`:600`) and the kernel's only
   z term is `threadIdx.z * BLOCK_SIZE` (`:565`) -- `threadIdx`, not
   `blockIdx` -- in a one-dimensional block, so it is always 0. The extra z
   blocks recompute leaves 0-1023 and leaves 1024+ are never written.
   Unreachable in a default fit; the dead term is transcribed, not "fixed".
2. **`FindOptimalSplitSolarImpl` does not compute `TSolarScoreCalcer`'s
   formula.** The calcer is a per-leaf `-sum^2 (1 + 2 log(w+1)) / w`; the
   dynamic kernel is a held-out estimate `-2 mu sumTest + wTest mu^2` scaled
   by `(1 + 2 log(totalTestWeight + 1))` only when that weight exceeds 2.
   Same name, different objective, selected purely by `foldCount == 1`.
3. **`denumSqr` is seeded `1e-20f` in the dynamic cosine kernel (`:344`) and
   `1e-10f` in `TCosineScoreCalcer::NextFeature` (`score_calcers.cuh:149`)**,
   under the same `> 1e-15f` guard -- so the dynamic path falls through to
   `FLT_MAX` on an all-empty feature and the single-fold path never does.
4. **The `ScoreStdDev` noise enters at a different point in the two cosine
   paths**: dynamic does `score *= catWeight` then adds noise; single-fold
   adds noise inside `GetScore()` then multiplies. With a cat-feature weight
   other than 1 those are different distributions.

Also: `FindOptimalSplitDynamic` supports only 3 of the 7 score functions
(SolarL2, Cosine, NewtonCosine). `L2`, `NewtonL2`, `SatL2` and `LOOL2` throw
the moment `foldCount > 1`, and `find_optimal_split_dynamic` raises in the
same place.

## 97. `TOptimizationSubsets` state layout

`gbdt/methods/pointwise_optimization_subsets.mojo`.

**Theirs.** Six device buffers, two of them arrays of structs:
`TBuffer<ui32> Bins/Indices`, `TBuffer<TDataPartition> Partitions`
(`{ui32 Offset, Size}`), `TBuffer<TPartitionStatistics> PartitionStats`
(`{double Weight, Sum, Count}`), and `TL2Target`'s two separate
`TCudaBuffer<float>` columns (`pointwise_optimization_subsets.h:14-24`,
`weak_target_helpers.h:11-14`, `gpu_data/gpu_structures.h:113-116`).

**Ours**, five departures, none arithmetic:

1. `Partitions` is ONE `UInt32` buffer, `partitions[2p] = Offset`,
   `[2p + 1] = Size` -- `TDataPartition[]` reinterpreted, in their field
   order. An earlier draft used two parallel arrays, arguing from the greedy
   searcher's convention; that was OVERRULED and the reversal is recorded
   rather than dropped. It mirrors `TDataPartition` exactly (the tie-breaker
   under COPY-DO-NOT-IMPROVE), and it is already the GATED CONTRACT of two
   layers that landed first -- `kernel/split_properties_helpers.mojo:193,196`
   and `kernel/pointwise_hist2_one_byte_templ.mojo:291-292`, with
   `checks/pointwise_dispatch_check.mojo` green on 3,686 and 12,360 cells.
   **Cost, named:** the offsets kernel is PORTED, not reused. In a
   parallel-array layout `TPartitionOffsetWriter::Write` and
   `TVecOffsetWriter::Write` collapse to one store; interleaved they do not,
   which is why CatBoost instantiates the template twice.
2. `WeightedTarget`/`Weights` are two columns of ONE buffer. The shape is
   CatBoost's own (`TOptimizationTarget::StatsToAggregate`,
   `greedy_subsets_searcher/split_properties_helper.h:41`).
3. Source plane 0 is the WEIGHT, plane 1 the weighted target -- the reverse
   of `TL2Target`'s declaration order, matching
   `TPartitionStatistics{Weight, Sum}` and `greedy_search_helper.mojo:244`.
4. `PartitionStats` is Float32, not `double` (Metal has no float64).
   **COST UNPRICED**: no measurement against a Float64 host reduction of a
   real fixture. The gate sidesteps it with integer plants under 2^24.
5. `PartitionStats` is STRIDE 3 and **plane 2 is dead weight, stored anyway.**
   Their `Count` is never reduced on this path (`counts == nullptr`,
   `pointwise_kernels.h:240`, so `PartitionUpdateImpl` takes
   `else { tmp = size; }` at `methods/kernel/pointwise_scores.cu:668-676`),
   AND no scorer ever reads it -- theirs touches `.Weight`/`.Sum` at
   `pointwise_scores.cu:89,92,260,263,359,362` and `.Count` nowhere; the
   ported scorer indexes `3*off + 0` and `3*off + 1` only
   (`kernel/pointwise_scores.mojo:700-703`, `:884-885`, `:1011-1014`). The
   plane carries no information into any consumer. It is three wide because
   `TPartitionStatistics` is three doubles wide and the ported scorer's
   reader is compiled against that stride; a stride-2 record would be
   arithmetically complete and would silently misalign every read in that
   file. It starts carrying a reduction when the PAIRWISE family lands.

Two adapters exist for this and are labelled not-a-port:
`deinterleave_partitions_kernel` splits the record for
`compute_partition_stats` (whose signature is the greedy searcher's live
contract), and `pack_partition_stats_kernel` widens stride 2 to 3 and writes
the Count arm. Both run at most `max_part_count` threads: 64 at depth 6.

Gated per cell by `checks/pointwise_subsets_check.mojo`, whose
`check_layout_contract` pins the record with `comptime assert` against
literals taken from the call sites above -- because the first version of that
check spelled the layout with this file's own constants, and a swap of
`PART_OFFSET`/`PART_SIZE` moved ZERO cells. It was verifying
self-consistency, which a swap preserves perfectly. The contract now fails at
BUILD time and names the call site it contradicts.

## 98. Which partition reducer `UpdateSubsetsStats` calls

CatBoost has two similarly-named, different kernels:

| symbol | file | shape |
|---|---|---|
| `UpdatePartitionProps` | `methods/kernel/pointwise_scores.cu:681` | `<<<partsCount, 1024>>>`, ONE BLOCK PER PARTITION, three sequential `double` reductions |
| `UpdatePartitionsProps` | `cuda_util/kernel/update_part_props.cu:197` | grid-strided over (chunk, part, stat) at 512, plus a `SaveResultsImpl` phase |

`UpdateSubsetsStats` (`pointwise_optimization_subsets.h:66`) dispatches the
FIRST. **We call the SECOND -- `gbdt/gpu_util/partitions_reduce
.compute_partition_stats` -- deliberately, and the faithful port of the first
exists in-tree and is not called.** `partition_update_kernel` /
`update_partition_props` landed in `gbdt/methods/kernel/pointwise_scores.mojo`
(`:1290`, `:1800`) with the three null arms and the `tmp = size` Count arm
intact; it is a correct port of the function their dispatch names.

Why we do not call it:

* their form puts the WHOLE DATASET through ONE THREADGROUP at depth 0, since
  the grid is `partsCount` and that is 1 there;
* this repo has built that shape twice and measured against it twice
  (`gbdt/gpu_util/partitions_reduce.mojo`);
* the arithmetic is the same either way, and the `double`-vs-Float32 gap
  (97.4) puts bit-parity with their kernel out of reach on this target
  regardless, so there is nothing to buy.

**PRICED, AND UNMEASURED ON THIS PATH.** The two cited measurements are on the
greedy path and are cited, not re-taken; nothing calls this file yet, so a
timing here would mean nothing. What IS counted is launches: their `Split` is
10, ours is 9 -- the two adapters are not bought at the cost of a launch
budget. If a measurement later favours the one-block form, the swap is one
call in `update_subsets_stats` and the kernel is already written.

Summation order differs from theirs and can differ across machines. That is
true of their own code too (`UpdatePartitionsProps` sizes its grid from
`TArchProps::SMCount()`), and `partition_stats_chunks` is pinned under
`NUMERIC_IDENTICAL` for exactly that reason.

## 99. `methods/helpers.mojo`, three non-arithmetic departures

1. **`GetBinsForModel` and `CacheBinsForModel` are NOT PORTED**
   (`helpers.cpp:3-58`). They need `TScopedCacheHolder`, `TTreeUpdater` and
   `TFeatureParallelDataSet`, all unported, and their four call sites are all
   in the FEATURE-PARALLEL learner. They belong with rung 2 of section 91 E.
2. **`TBinarizedFeaturesManager` is unwrapped into parameters.** `ToSplit`,
   both `SplitConditionToString` overloads, `PrintBestScore` and
   `HasPermutationDependentSplit` take the values they would have read.
   Arithmetic unchanged. One real consequence: `to_split` has NO
   feature-bundle arm (`helpers.cpp:159-161`) and RAISES on one;
   `PrintBestScore`'s CTR tensor tail is likewise absent.
3. **Float-to-text is Mojo's, not C++ `ostream`'s** -- six significant digits
   there, shortest round-tripping form here. LOG TEXT ONLY. The gate checks
   which border index and which comparator, not the digits.

Also transcribed but with NO caller in either tree: `ReverseBits`,
`GetOddBits`, `GetEvenBits`, `MergeBits` (`helpers.h:53-101`). Checked:
`ReverseBits` has no `catboost/cuda/` caller (the one at
`private/libs/data_types/groupid.h:14` is a different, one-argument
function); `GetOddBits`/`GetEvenBits` are called only from
`methods/ut/test_pairwise_tree_searcher.cpp:152-153`; `MergeBits` has none
(the `MergeBits` in `ctrs/ctr_kernels.h:170` is an unrelated device kernel).
Ported for completeness of the assigned file, gated against an independent
oracle, and their lack of a caller is stated rather than left to be found.

### 99a. Their two `ToSplit` clamps are asymmetric, and their comment does not say why

`helpers.cpp` clamps a categorical split to `GetBinCount(f)` and a float
split to `GetBorders(f).size() - 1`. Their comment explains why a clamp
exists at all, not why the two differ by one. Transcribed as written; gated
both ways.


## 97.2 (CORRECTED at the wiring step) -- the weak target is TWO buffers, and why that was learned late

An earlier draft merged `WeightedTarget` and `Weights` into one two-column
buffer, and this file defended the merge. **It is reversed.** Both are two
separate buffers on both sides now -- `TL2Target.weights`/`.weighted_target`
and `TOptimizationSubsets.gathered_weight`/`.gathered_target` -- which is what
`weak_target_helpers.h:11-14` declares.

**IT IS A WALL, NOT A PREFERENCE.** The pointwise histogram family takes the
two columns as two independent pointers -- `compute_hist2(..., target:
MutPointer[Float32, o5], weight: MutPointer[Float32, o6], ...)` -- which is
their signature too. Two views of ONE buffer cannot be handed to a kernel:

* `buf.unsafe_ptr()` with `buf.unsafe_ptr().unsafe_offset(n)` is refused,
  "aliasing values passed mutably to 'target' argument and passed mutably to
  'weight' argument";
* `unsafe_bitcast[Float32]()` does NOT launder the origin;
* the check fires at `enqueue_function` ITSELF, so no wrapper can hide it at
  any level.

`PORTING_RULES` 4. **Found at the WIRING step, which is the only place it
could have been found** -- every layer below had been gated for hours with
the merged buffer and none of them could see it, because none of them passed
both columns to one kernel.

### What the reversal costs, stated rather than absorbed

Two things, and the second was not obvious:

* the gather goes 1 launch -> 2, which is exactly their `GatherTarget` body,
  so this half is fidelity at no real cost;
* **the partition reduce goes 1 call -> 2.** `compute_partition_stats` reads
  its planes as `stats[stat * line_size + row]`, contiguous, so two separate
  allocations cannot be reduced in one call. It runs once per column at
  `n_stats = 1`: +2 launches, AND a different chunk count, because their grid
  formula is `CeilDivide(2 * SMCount, statCount)`
  (`update_part_props.cu:215`) and `statCount` is now 1 instead of 2. The
  float summation tree has a different SHAPE than before the split. Still
  deterministic, still pinned through `partition_stats_chunks`, and
  INVISIBLE to the gate because its plants are exact integers -- recorded
  because nothing else would record it.

### The launch budget, corrected on both halves

The earlier "theirs is 10, ours is 9" was wrong twice over and is deleted.
`ReorderBins` is not 4 launches: `launch_radix_sort_bins` at one bit is SIX
-- four in `_radix_pass` plus two copy-backs, since one pass is an odd count
and the `Current() != keys` arm fires every level.

    step                            theirs   ours
    UpdateBinsFromCompressedIndex      1        1
    ReorderBins (1 bit)               ~3        6
    UpdatePartitionDimensions          2        2
    de-interleave adapter              -        1
    GatherTarget                       2        2
    UpdatePartitionStats               1        4
    pack adapter                       -        1
    ----------------------------------------------
    total                             ~9       17

### 98a. OPEN, and the number that justified DEVIATION 98 has moved

DEVIATION 98 declined `update_partition_props` -- the reducer their dispatch
actually names, ported and sitting unused in
`gbdt/methods/kernel/pointwise_scores.mojo:1800` -- on the grounds that one
block per partition starves the device at depth 0.

After the interleaved record, the stride-3 stats and now the two-buffer
split, **that reducer wants exactly what this struct holds**: interleaved
`parts`, stride-3 `part_stats`, and target/weights as two separate float
pointers. It is a drop-in. Calling it replaces the de-interleave, both
reduce calls and the pack -- **six launches and two scratch buffers, for one
launch of the faithful function** -- taking `Split` from 17 to 12, and
removing the chunk-count change recorded above.

The trade was six launches to avoid one when the buffers were merged. It is
now six launches AND two scratch buffers to avoid one, against a reducer
that is also the faithful one. What is still bought is depth-0 occupancy,
and **that remains UNMEASURED**: at depth 0 there is exactly one partition,
so their form puts the whole dataset through a single threadgroup, a shape
this repo has lost to twice on the greedy path.

Not acted on, because the decision needs a measurement rather than an
argument and no benchmark is authorised. The swap is one call in
`update_subsets_stats` and the kernel is already written.

## 106. `fit`'s pointwise arm, and the three things that only wiring could find

`gbdt/methods/doc_parallel_boosting.fit` and `fit_with_test` take
`use_pointwise_searcher`, defaulting to False. True selects
`TDocParallelObliviousTreeSearcher` -- CatBoost's learner for SINGLE-TARGET
symmetric trees at `boosting_type=Plain` (`PORTING.md` 91 F) -- in place of
`TGreedySubsetsSearcher`, which is what they run for MULTICLASS symmetric
trees and what this repository has always used.

Gated by `pixi run check-fit-pointwise`: the same data both ways, and the
loss curves agree to the bit over twenty iterations. The two arms share the
compressed index, the weak target, the bootstrap, leaf estimation, the apply
and the loss; only the structure searcher differs.

### The arm does NOT reuse the searcher's own partition

Their `FitImpl` estimates leaves inside the searcher; ours returns the
structure only (DEVIATION 104) and re-derives the partition through
`compute_bins_for_model` + `partition_from_bins`, which is the SAME path the
non-estimation permutations already take. Deliberate: it means the pointwise
arm shares every line of leaf estimation with the greedy arm, so any
difference between them can only come from the structure.

### DEVIATION 64 does not apply to this arm

Item 64 skips leaf estimation entirely for RMSE + Newton at one iteration and
one permutation, because the GREEDY searcher's leaf already IS the Newton
step. The pointwise searcher produces no leaf, so the estimator must run
whatever item 64 says -- and `est_sm`, which item 64's branch leaves at -1,
has to be queried for this arm too.

### Three bugs the wiring found, none of them in a ported kernel

Every one of the fifteen gates below was green throughout.

1. **`IntLog2` is `ceil`, not floor.** A 100-fold feature was counted under
   bit 6, launching the 6-bit kernel, whose device bound `(32, 64]` rejected
   it -- while the 7-bit kernel never launched because
   `if (featureCountForBits)` saw zero. Every one-byte feature whose fold
   count is not a power of two went unhistogrammed, silently.
2. **`result_size` IS the scorer's grid.** Each block writes its own record
   at `2 * blockIdx.x` and `ReadOptimalSplit` folds them
   (`histograms_helper.h:205-208` sizes it
   `min(CeilDivide(count, 128), 32)`). Reading block 0 is an argmin over the
   first 128 bin features. The number named it: the searcher picked feature 7
   bin 27, and bin 27 is the last bin of feature 7 inside block 0.
3. **`BIN_SPLIT_TAKE_BIN` is 0 and `BIN_SPLIT_TAKE_GREATER` is 1**, which is
   the opposite of what the names suggest. Writing `1 if one_hot else 0`
   makes every ORDINARY feature an equality test. The tree still had the
   right depth and the right splits in the right order -- and partitioned the
   rows completely differently, 8 non-empty leaves instead of 12. Found by
   comparing tree 0's LEAF VALUES after its structure had already matched.

And one that was a wall rather than a bug: `TCFeature::Offset` is an ELEMENT
offset while this tree's layout stores a COLUMN index. The multiply by
`n_rows` was present in the score helper and absent in `split_subsets`, so
the split read column 0's bits with another feature's shift, every document
went one way, and `HasSplit` stopped the tree at depth 1.

## 107. `PolicyBlock.group_offset` is not a group index, and two gates passed while it was read as one

Found 2026-08-21 by a measurement, not by a check.

`PolicyScoreHelper` built `TCFeature::Offset` as

    (block.first_column + block.group_offset[i]) * n_rows

**`PolicyBlock.group_offset` is 0 for EVERY feature.** It is not a group
index. A feature's compressed-index column lives in
`layout.features[f].offset`, and the two agree only while a policy fits in
ONE word:

    4 one-byte features per UInt32, 8 half-byte, 32 binary

So any policy with more than that had every feature past the first group
reading the FIRST group's column -- correct bins for a different feature,
histogrammed into the right slot. The searcher then split on a feature whose
data it had never seen.

### It survived two gates, and the reason is the point

`check-pointwise-vs-greedy` and `check-fit-pointwise` both planted their
signal on features inside the first group of their policy. The wrong column
was therefore never the one carrying the answer: the tree still grew, still
matched the greedy searcher, and still drove the loss down.

It was caught by `checks/pointwise_default_probe.mojo` -- a TIMING probe
-- because that probe asserts the two arms end at the same loss before it
reports a ratio. The fixture it happened to use put the signal on the fifth
one-byte feature.

Localisation took three steps, each cheap: a depth-and-rows sweep showed the
disagreement at EVERY shape, which cleared scale; a threshold sweep cleared
split balance; and a per-bit-width sweep showed all ten widths failing, which
cleared the bit dispatch and left the fixture shape itself. Dumping the
layout then showed `group_offset` flat at 0 with the real column in
`cf.offset`.

### Both gates now carry a feature past the first group

`check-pointwise-vs-greedy` and `check-fit-pointwise` both use SIX one-byte
features and put their strongest signal on the sixth. Verified: restoring the
old expression makes the first fail on tree depth and the second on tree 0's
splits.

**The rule this earns**: a fixture must exercise the SECOND of anything the
code groups -- the second word, the second block, the second group -- because
the first is where every off-by-a-base-address is invisible. Same family as
[[reached-but-inert]].

## 98 (CLOSED 2026-08-21 by measurement): we now call THEIR reducer

DEVIATION 98 declined `UpdatePartitionProps` -- the reducer
`UpdateSubsetsStats` actually dispatches -- on occupancy grounds: at depth 0
there is one partition, so their grid is a single threadgroup for the whole
dataset, and this repository had lost to that shape twice on the greedy path.
It was recorded as PRICED AND UNMEASURED, and 98a said the number had moved.

**Measured** (`pixi run probe-partition-reducer`; 200k rows, 10 SMs, 5
interleaved reps, min of each):

    parts     theirs      ours (2 chunked calls)    theirs/ours
        1     0.225 ms        0.220 ms                 1.02
        2     0.219           0.233                    0.94
        4     0.201           0.398                    0.51
        8     0.230           0.292                    0.79
       16     0.182           0.276                    0.66
       32     0.225           0.398                    0.57
       64     0.197           0.904                    0.22

**A TIE at the one shape the objection was about, and up to 4.6x faster
everywhere else.** The occupancy fear does not materialise: at one partition
their single threadgroup and our chunked grid land within 2% of each other,
and from four partitions on their one launch beats our six.

Switched. `update_subsets_stats` now calls `update_partition_props`, and the
de-interleave adapter, the pack adapter and one of the two reduce calls are
gone with it -- six launches to one, two scratch buffers freed, and the
chunk-count numeric note that block used to carry no longer applies because
`statCount` is 2 again.

Verified by the gates that could have seen a change: `check-pointwise-subsets`
is still exact per cell, and **`check-fit-pointwise` still requires the two
`fit` arms to be BIT-IDENTICAL over twenty iterations and still passes** --
so the different summation order changed nothing observable at this fixture's
magnitudes.

## 106a. The `use_pointwise_searcher` default STAYS FALSE, and now for a priced reason

`PORTING.md` 106 shipped it defaulting to False and said the flip was a
measurement's job.

**Measured** (`pixi run probe-pointwise-default`; 60k rows x 16 features
spanning all three policies, depth 6, 25 iterations, 3 interleaved reps):

    greedy-subsets   221-308 ms
    pointwise        371-412 ms
    ratio            1.34x - 1.68x, pointwise SLOWER

with **identical final loss on every run**, which the probe asserts before it
reports a ratio.

So the decline is priced: the pointwise arm costs between a third and two
thirds more for output that cannot differ. The spread is wide because another
session is running GPU work on this machine -- the greedy baseline itself
moved 221 to 308 ms between runs -- so the direction is solid and the
magnitude is not. **A tighter number needs a quiet box.**

The known cost inside that gap is `split_stat_planes`: a host round trip of
`2 * n_rows` floats per tree, because this tree carries the weak target as
one two-plane buffer and the pointwise kernels cannot take two views of one
buffer (97.2). Removing it means the boosting loop carrying the weak target
as two buffers throughout, which `stats`'s other three readers make a real
change rather than a rename. That is the first thing to try before measuring
again.

### And the probe found a correctness bug, which is why it is kept

`probe-pointwise-default` asserts the two arms end at the SAME loss before it
reports a ratio. Its first run came back `0.128` against `107.04` -- and that
is how `PORTING.md` 107 was found, with two gates green. A timing probe that
does not check what it is timing measures two different computations and
reports their ratio as if it meant something.

## 108. The pointwise searcher against CATBOOST'S OWN TREES: 144 of 144, and the one that was not

Until 2026-08-21 every gate on the pointwise family compared it against a
HOST RECOMPUTATION of CatBoost's formula, or differentially against this
repository's other searcher. Neither is a comparison with CatBoost.

`oracle_main.mojo` now runs BOTH searchers against `bench/oracle*.txt` --
splits CatBoost itself produced and `tools/catboost_oracle.py` dumped:

    fixture          borders  policy reached        greedy   pointwise
    oracle.txt          15    half-byte              48/48     48/48
    oracle100.txt      100    one-byte, 7-bit        48/48     48/48
    oracle254.txt      254    one-byte, 8-bit        48/48     48/48

**The 254 fixture is the only one that reaches the 8-bit kernel at all**, and
it is the one that failed: 7 of 48 on the first run, first divergence at tree
0 depth 1.

### The bug, and DEVIATION 95 had already written its symptom down

`fit`'s pointwise arm passed a hardcoded `Float32(1.0)` as `fixed_scale`. The
8-bit accumulator holds Int32 fixed point (DEVIATION 93, because Metal has no
threadgroup float atomics), so at scale 1.0 every gradient below 1.0
quantizes to zero.

DEVIATION 95's block describes the failure mode exactly, for the same
accumulator on the greedy path: *"The tree still learns, because the leaf
VALUES come from `compute_partition_stats` and never touch the accumulator;
only the SPLITS go bad. That is exactly a model that stays monotone, still
beats the mean, and is several times worse than it should be."*

Which is why nothing else caught it. `check-fit-pointwise` requires the two
arms to be BIT-IDENTICAL and passed, because its fixture's widest feature has
127 folds and never reaches the 8-bit kernel. `check-pointwise-hist2-8bit`
gates that accumulator exactly -- but it is handed a scale, and gates the
arithmetic at whatever scale it is given.

Fixed by deriving the scale the way the greedy path does: `choose_scale` over
the larger of the two planes' sums of magnitudes, which `fit` already reduces
on the device into `mags`.

**PRICED**: the pointwise arm now drains once per tree to read those two
floats. DEVIATION 95 removed exactly that drain from the greedy path by
deriving the scale ON the device, and this arm cannot do the same yet because
`compute_hist2` takes `fixed_scale` as a host scalar. Making it a device
pointer is the fix and is not attempted here.

### What this changes about every earlier claim

Every "the pointwise family is gated" statement before this one meant gated
against our own arithmetic. This is the first that means gated against
CatBoost. The gap was named in `NEXT_TWO.md` rung 5 and in `PORTING.md` 91 F
the whole time; it is now closed for the CPU oracle, and `task_type="GPU"`
remains unrun.

## 109. CatBoost's GPU arm cannot run on this machine, and the oracle is not weaker for it

Probed 2026-08-21:

    catboost.CatBoostRegressor(task_type='GPU', devices='0').fit(X, y)
    -> catboost/libs/train_lib/trainer_env.cpp:9:
       Environment for task type [GPU] not found

CatBoost's GPU build is CUDA. `NEXT_TWO.md` had twice named "run the oracle
with `task_type='GPU'`" as the highest-value open item, and `PORTING.md`
91 F recorded it as a known gap. **It is not an item.** It needs different
hardware, which is what `tools/nvidia_bench.sh` and `tools/remote_gpu.sh`
exist for.

It is also the project's THESIS rather than a limitation of the port: their
GPU arms cannot run on Apple silicon, and that is the win condition.

### OUR GPU ARM AGAINST THEIR CPU ARM IS THE COMPARISON, not a fallback

This entry first framed the CPU oracle as "the right reference" and said what
remained unverified was whether CatBoost's GPU arm agrees with its CPU arm.
**That is not the point and the framing is deleted.**

The point is `mojolearn-plan`'s thesis: **GPU ACCESS, NOT TIER.** Their GPU
arm cannot run on this Mac at all, so the comparison that matters -- for
correctness AND for speed -- is

    OUR arm, on this Mac's GPU     against     THEIR arm, on this Mac's CPU

`PORTING.md` 108's 144 of 144 is exactly that comparison on the correctness
side, and it is the strongest evidence in this repository: our GPU searcher
reproducing, split for split, the trees their CPU learner chose on the same
data and the same grid.

The speed side of the same comparison is a BENCHMARK and is not run here.

Whether CatBoost's own two arms agree with each other is a question about
CatBoost, is unanswerable on this hardware, and is not this port's business.

## 119. RUNG 2 IS NOT A SECOND SEARCHER, and the fold layout is what it actually costs

`NEXT_TWO.md` priced rung 2 as porting `TFeatureParallelObliviousTreeSearcher`
-- 713 lines beside the doc-parallel searcher already here. Reading it says
otherwise.

**Their searcher is ONE object with TWO modes**
(`oblivious_tree_structure_searcher.h:88-100`):

    SetTarget(target)                  SingleTaskTarget   ONE task, PLAIN
    AddTask(learnTarget, testTarget)   FoldBasedTasks     N pairs, ORDERED

and `CreateSubsets` chooses between them with one ternary (`:30-31`):

    SingleTaskTarget == nullptr ? WriteFoldBasedInitialBins(subsets.Bins)
                                : WriteSingleTaskInitialBins(subsets.Bins)

Everything after that line -- the depth loop, the histograms, the scorer and
the `TakeBest` fold -- is the same code on both arms. **THE SPLIT IS NOT, and
this section used to claim it was.** It concluded, from 91 A and the version of
91 B since corrected, that "what rung 2 costs is the fold LAYOUT plus wiring,
not a second searcher". Rung 2 was then ported and that was false: the
feature-parallel `Split` reads `docBins`, a per-document bit-packed array with
no counterpart on the doc-parallel path, and building it is three kernels and
1,067 lines. See DEVIATION 120. The estimate is deleted rather than annotated
because two documents were priced off it.

What remains true, and is the reason rung 2 could be gated as an IDENTITY: at
`FoldBits == 0` the two chains produce bit-identical `subsets.Bins` at every
level, measured over 3 splits and 16,434 per-document leaf ids.

`gbdt/methods/oblivious_tree_fold_tasks.mojo` is the fold layout half of it.

### The encoding, and why the pairing is load-bearing

`WriteFoldBasedInitialBins` (`:338-364`) walks the tasks and per task `k`:

    learn slice -> bin 2k        parts.push_back({cursor, learn size})
    test  slice -> bin 2k + 1    parts.push_back({cursor, test  size})

So N tasks give **2N partitions**, alternating, over ONE concatenated
document array. `FoldCount` is 2N and `FoldBits` is `IntLog2(2N)` -- CEIL,
the same function `PORTING.md` 107 records costing a day when it was read as
floor.

**THE DYNAMIC SCORER ALREADY DEPENDS ON THIS.**
`find_optimal_split_solar_kernel` and the dynamic cosine one read folds
`(f, f + 1)` as `(estimate, test)` and step by two -- ported and gated before
this file existed. Fold `2k` is task `k`'s estimate half, `2k + 1` its test
half. The two halves of ordered boosting meet exactly here: this file lays
the pairs out, that kernel consumes them. Put the test half first and the
scorer evaluates every split against the wrong half of every fold, with
finite scores and a well-formed tree to show for it.

### DEVIATION 119: no streams

`ForeachOptimizationPartTask` runs the per-task fills on up to 8 CUDA streams
(`RunInStreams(tasks.size(), Min<ui32>(tasks.size(), 8), ...)`). There are no
streams on Metal, so the walk is sequential. It changes nothing: `cursor` and
`currentBin` are captured by reference and advanced in task order in their
code too, so the streams overlap the FILLS and never the layout.

### What remains of rung 2

The layout is in and gated (`pixi run check-fold-tasks`). What is left is
wiring it: a `create_subsets` that takes a `FoldLayout` instead of assuming
one task, and a searcher loop that carries `fold_count > 1` through to the
dynamic scorer. Both are small; neither is a second searcher.

## 110. `IQueriesGrouping` is a tagged union, not a virtual interface

`gbdt/methods/dynamic_boosting_folds.mojo`.

THEIRS: an abstract class with five pure virtuals and two implementations,
`TWithoutQueriesGrouping` and `TQueriesGrouping` (`gpu_data/samples_grouping.h
:13-58`, `:60-127`), passed to `CreateFolds` by `const&`.

OURS: one struct with a `kind` tag, dispatching on it. `PORTING_RULES` rule 4
names this workaround -- Mojo has no dynamic trait objects, and a tagged union
is what their worker switches on anyway. NOT ARITHMETIC: all five accessors
are transcribed branch for branch, including the two DIFFERENT out-of-range
answers (`GetQueryOffset` returns the DOC count, `GetQueryId` the GROUP count).

NOT BUILT, stated rather than left to be found: `TQueriesGrouping`'s pair
vectors, read only by the pairwise losses, and `TDataPermutation::FillGroupOrder`'s
shuffle. Ours takes group sizes already in their final order. A caller cannot
use this to PERMUTE groups, only to describe a grouping already permuted.

Both arms are gated separately (F3 ungrouped, F6 grouped); sabotaging the
grouped accessor leaves F3 green, which is rule 8's reach-is-per-branch in one
measurement.

## 111. `ui32`/`ui64` widths become `Int`, and `IntLog2` is reused

THEIRS: `ui32` throughout `MinEstimationSize` and `CreateFolds`, and one
narrowing cast -- `static_cast<ui32>(minEstimationSize * growthRate)`
(`dynamic_boosting.h:211`, `:217`), a `double` multiply truncated toward zero
and wrapped to 32 bits.

OURS: `Int`, matching `gbdt/gpu_lib/slice.mojo`. `Int(Float64)` truncates
toward zero, so the truncation is identical; the WRAP is not carried.
UNREACHABLE IN EITHER TREE: firing it needs `sampleCount * growthRate >= 2^32`
with `growthRate > 1`, which their own `ui32 sampleCount` cannot supply. NOT
GATED for that reason, and the check says so.

`NCB::IntLog2` is `(ui32)ceil(log2(x))` in `double`. This file CALLS
`gbdt/ctrs/ctr_bins_builder.int_log2`, already in this tree, which computes it
as the bit length of `x - 1` -- that identity holds for every `x >= 1`.
Re-verified over all of `[1, 1 << 20]` against a third spelling and at the four
powers of two `MinEstimationSize`'s `>= maxFolds` threshold lands on, because
that threshold is exactly where a float `ceil(log2)` lands a hair off. It is a
DIFFERENT function from the `1 << (ui32)ceil(log2((float)FoldCount))` inside
`PointwisePartOffsetsHelper`, which is a DEVICE expression in `float`.

## 112. `CreateFolds`' growth loop carries an iteration bound theirs does not

THEIRS: the loop (`dynamic_boosting.h:215-222`) has no bound. It terminates
because `NextQueryOffsetForLine` is strictly increasing below `sampleCount` in
both groupings.

OURS: the same loop with `max_iterations = sample_count + 2` and a `raise`.
Not a knob -- it is the longest sequence their own termination argument
permits, since the right edge is a strictly increasing integer in
`[1, sample_count]`.

MEASURED, AND THE ONLY REASON IT EXISTS IS THAT A GATE HAS TO BE ABLE TO FAIL
RATHER THAN HANG. Two of the check's ten sabotages turn the recurrence into a
fixed point -- dropping the `+ 1` from `NextQueryOffsetForLine` (at
`g = 1.05, r = 1`, `floor(r * g) == r`) and stepping from `min_estimation`
instead of the previous right edge. Without the guard both HANG; with it they
raise at 502 passes over 499 samples and 9 over 6, and the run goes red by
name in two seconds.

Unsabotaged it has never fired: 4,899 folds at `n = 5,000, g = 1.0000001`
against a bound of 5,002; 6 at the default `g = 2.0`.

### 112a. Three things in `CreateFolds` that look like something else

Verified against the source at merge.

1. **`NextQueryOffsetForLine` is the whole off-by-one.** With no groups it is
   `Min<ui32>(line + 1, DocCount)` (`samples_grouping.h:52-54`), so it does
   ARITHMETIC on the default path while looking like a group-snapping call.
   Every fold edge is one past the geometric value: `2 * (m0 + 1) - 1`, not
   `2 * m0`.
2. **`MinEstimationSize` has a CLIFF at 500, not a taper** (`:177-185`).
   `n = 499` estimates the first fold on 2 documents and produces 8 folds;
   `n = 500` estimates on 11 and produces 6.
3. **The `folds >= maxFolds` arm needs `n > 13,107,200`** at the default
   `min_fold_size` of 100, and it LOWERS the first fold size (100 -> 51),
   which is the opposite of what "cap the fold count" suggests.

And `CB_ENSURE(minEstimationSize, ...)` (`:201`) is UNREACHABLE: the
group-count ensure precedes it, every `MinEstimationSize` arm returns at least
1, and `NextQueryOffsetForLine(>= 1)` is at least 2. Transcribed; the check
states it is not gated.

### 112b. `learnPermutationCount - 1` in full

Permutation 2 of 4 has folds built, cursors allocated, leaf values estimated
on it (`dynamic_boosting.h:378-396`) and the model added back to it
(`:447-465`). It is only the STRUCTURE SEARCH that never sees it
(`:286-289`). Transcribed verbatim; a port using `% learnPermutationCount`
would train a different model on the default config. Gated: over 1,000 draws
the structure search reaches rows 0 and 1 and never 2.

## 113. The categorical oracle is ONE-HOT ONLY, and that is their CPU learner's limit rather than a fixture chosen to pass

`bench/oracle_cat.txt` puts three categorical columns (k = 3, 5, 8) beside
eight numeric ones so the searcher has to weigh EQUALITY candidates against
ordered ones inside one tree, and so the compressed index carries two grid
policies at once. It contains no CTR features. That is a restriction on what
CatBoost can be asked for on this machine, and it is worth writing down
precisely, because "we compared against the subset that agreed" is exactly
the failure mode [[no-dataset-cherry-picking]] names.

Two independent blockers, both in their source:

- **`FeatureFreq` is not a CPU CTR.** The default this port mirrors is their
  GPU `simple_ctr`, which is `Borders` plus `FeatureFreq`
  (`cat_feature_options.cpp:231`). `IsSupportedCtrType(CPU, FeatureFreq)`
  returns FALSE (`private/libs/options/restrictions.h:18-48`). Ask their CPU
  learner for the set we implement and it refuses to start.
- **`max_ctr_complexity` above 1 is refused on CPU** where their own default
  is 4, so feature combinations cannot be exercised either.

Their GPU arm has both. Their GPU arm cannot run on this machine
(DEVIATION 109). So the honest description is: the ONE-HOT half of our
categorical path is checked against CatBoost's own decisions, 21 of 21, and
the CTR half is checked against everything else in `checks/ctr_*` and
against no CatBoost output at all. The CTR half is not oracle-covered, and
DEVIATION 109's "their CPU arm is the comparison" does not repair that --
this is the one place where their CPU arm is not merely slower than their
GPU arm but cannot express the feature.

Recorded rather than worked around. A fixture that dropped the categorical
columns entirely would have been green too, and would have missed 114.

### 113a. A one-hot split is compared per TYPE, not only per (feature, bin)

`catsplit <tree> <flat_feature_index> <code>` is an equality level, written
interleaved with `split` in depth order. The comparison had to grow a third
axis to be worth running: `> code` and `== code` name the SAME (feature,
bin) pair and partition the rows differently. A checker that matched on
(feature, bin) alone would have called 114 a pass.

## 114. The scorer's one-hot flag array was a hardcoded constant, and both gates on that kernel handed the array in by hand

`scan_pointwise_histograms_kernel` skips the bin prefix scan when a feature
is one-hot, mirroring `split_properties_helpers.cuh:126`
(`if (!feature->OneHotFeature)`). A one-hot bin is an equality test, so a
running prefix across its bins is not a quantity that means anything.

`PolicyScoreHelper.__init__` built the flag array the kernel reads with a
hardcoded `oh.append(UInt8(0))` -- on the line AFTER the same loop read
`layout.features[block.feature_ids[i]].offset` off the layout, where
`.one_hot_feature` was sitting unread two fields away. Every one-hot feature
was therefore prefix-summed and its equality candidates scored as
thresholds.

Symptom, from `bench/oracle_cat.txt` the first time it ran: **0 of 18**
one-hot splits matching CatBoost, train mse 2.30 against their 0.147. Not a
subtle drift -- the tree was picking a different feature at nearly every
level. After the fix, 21 of 21, and 192 of 192 splits across all four
fixtures on both searchers, mse 0.14671102 against their 0.14671103.

### Why nothing already in the repository could see it

The skip IS gated, twice. `pointwise_offsets_check` plants
`[0, 0, 0, 1, 0, 0]`. `pointwise_dispatch_check` plants a one-hot feature
per policy. Both are correct and both still pass with the defect in place,
because **both construct the flag array themselves and hand it straight to
the kernel.** The kernel was right, the gates on the kernel were right, and
the array the PRODUCT passed was a constant.

This is [[mojotrees-verify-reach-not-output]] with a sharper edge: reach was
never the problem. The constructor ran, on every call, in every fixture. It
computed the wrong thing, and no gate had ever asked it to compute anything
-- they all went around it.

`PORTING_RULES` 3 and 8 at once: the file had a caller, it was not in
`UNWIRED.md`, the suite was green, and the branch the suite ran was not the
branch the product ran. Sixth instance of **"reached but inert"** in this
port.

## 115. A gate that builds the kernel's inputs cannot check the caller that normally builds them

The general shape behind 114, worth its own number because it recurs and
because the fix is mechanical.

A kernel gate has to construct inputs. That construction is a SECOND
implementation of whatever the product does to produce them, and it is
usually the better one -- written deliberately, by hand, with the values
chosen to be discriminating. So the gate proves the kernel correct with
respect to inputs the product never passes, and the real constructor sits
underneath, unexamined, for as long as its output is well-formed enough not
to crash.

`checks/one_hot_flags_check.mojo` is the countermeasure for this one:
it hands `PolicyScoreHelper` a LAYOUT and reads back off the device what the
constructor actually built.

  - **H1** per-feature flags for a layout with one-hot features in it.
  - **H2** all zeros for a layout with none, so a constructor that flags
    everything cannot pass.
  - **H3** the flags survive the POLICY SPLIT -- a flagged feature sits at
    its within-policy index, which is not its global feature id. That
    indexing is where a per-policy table goes wrong and it is invisible to
    any fixture with one policy.

Sabotaged: restoring the `UInt8(0)` fails H1 on exactly the three flagged
features, at the right positions, while H2 and H3 stay green -- so H1 is
carrying the gate and the other two are not passing it by accident
([[sabotage-when-required]]: the bound was ours and the path was new).

The rule to apply forward: when a check plants a kernel input BY HAND, ask
what builds that input in the product, and whether anything reads it back
from there. If nothing does, the constructor is unchecked no matter how
green the kernel is.

## 116. `TFeatureTensor` lives with the batch builder, and its splits are `Int32`

THEIRS: `TFeatureTensor` and `TBinarySplit` are both in
`catboost/cuda/data/feature.h`, and all three members of `TBinarySplit` are
`ui32`.

OURS: `TBinarySplit` landed earlier in `gbdt/models/oblivious_model.mojo`
with `Int32` fields, because the model is where this port first needed it.
`TFeatureTensor` is in `gbdt/methods/batch_feature_tensor_builder.mojo`
rather than a new `gbdt/data/feature.mojo`, because the lane that ported it
shipped two files and a third would have been a directory decision.

WHY IT MATTERS: `Int32` is the wrong signedness for BOTH the comparator and
the hash. `std::tie(FeatureId, BinIdx, SplitType)` orders `0x80000001` ABOVE
`0x7fffffff`; read as `Int32` it orders below, which reverses the canonical
form and therefore the hash and therefore the tree-CTR dataset cache key. So
every field read in that file goes through `_as_u32`, and
`checks/feature_tensor_check.mojo` fixtures 41 and 42 exist to fail if
one of those reads is ever dropped (sabotage: they turn gates 1, 2, 4 and 5
red). Moving `TBinarySplit` to `UInt32` is the real fix and belongs to
whoever next touches the model.

**A Mojo defect rode in on that fixture and is worth more than the deviation:
`UInt64(f(x))` SIGN-EXTENDS when `f`'s body is a same-width `Int32 -> UInt32`
cast** -- the cast is elided at the call site and the constructor binds
against the original `Int32`. `UInt64(f())` for an `f` returning a UInt32
LITERAL is correct, and so is binding the intermediate to a `var`. It
produced a wrong `TBinarySplit::GetHash()` for every split with a field at or
above 2^31 and MOVED NOTHING ELSE, because the canonical form, the
comparator, `IsSubset` and every scalar compare two values mangled
identically. Defended twice: `@no_inline` on `_as_u32`, and a `_widen_u32`
that masks to 32 bits so the answer is right whichever way the constructor
resolves. Third member of the family `PORTING.md` 17 and
`gbdt/models/hash.mojo` opened: assume Mojo's numeric conversions are
approximate until an external oracle says otherwise.

### 116a. `GetComplexity()` counts all the splits as ONE

`feature.h:89-188`. `GetComplexity()` is
`CatFeatures.size() + min(Splits.size(), 1)`, and `Size()` -- the plain sum
-- sits two lines away. A depth-6 tree's six splits plus one cat feature is
complexity **2**, not 7. That is what `max_ctr_complexity` is counting, so a
port that reached for `Size()` would refuse tree CTRs their config permits.
Sabotage: returning `Size()` turns gates 2, 5 and 7 red.

`IsSimple()` is likewise a SIZE-1 test, not a "one cat feature" test -- a
one-split, no-cat tensor is simple too, which is why `IsTreeCtr` is
`IsCtr && !IsSimple`.

### 116b. Their canonicalisation is order-INDEPENDENT, and the hash has three traps

Every mutator ends in `SortUniqueSplits()` / `SortUniqueCatFeatures()` =
`Sort` then `Unique`, so insertion order is ERASED. The check builds the same
tensor by six routes and demands one hash (260 rebuilds).

`GetHash` = `MultiHash(TVecHash<TBinarySplit>()(Splits), VecCityHash(CatFeatures))`,
and all three of these are load-bearing:

- `MultiHash` folds from the **tail**: split type hashed first, feature id
  XORed last.
- `TVecHash` accumulates in **ui32**, so each split's 64-bit hash is
  truncated, and it returns **`int`** -- a result at or above 2^31 enters
  sign-extended as `0xffffffff________`.
- `VecCityHash` hashes the **raw bytes** of the ui32 vector, so empty gives
  `k2`, not 0.

800 pairwise-different tensors, 0 colliding hashes. Sabotaging `GetHash` to a
constant produces **319,600** colliding pairs, which is the measurement that
says gate 6 is discriminating rather than lucky.

Two sabotages moved NOTHING and are recorded as honest zeroes: accumulating
`TVecHash` in ui64 and truncating once at the end (truncation commutes with a
`*`/`+` fold -- it is the same function), and merging the two inner loops of
`VisitCtrBinBuilders` (see 117).

## 117. `RequestStream` returns a batch width, not a stream

THEIRS: `TBatchFeatureTensorBuilder::RequestStream`
(`batch_feature_tensor_builder.cpp:67-77`) calls
`GetCudaManager().RequestStream()` once per new slot and hands each
`TCtrBinBuilder` its stream id, so the first inner loop of
`VisitCtrBinBuilders` submits `buildStreams` independent bin builds
concurrently and the second consumes them. Their `:24` comment forbids
merging the two loops for exactly that reason.

OURS: there are no streams (`ctx.stream()` raises on Metal, DEVIATION 119),
so `builder_streams[j]` holds the slot index `j` and the batch is serial. The
BATCH WIDTH and the two-loop structure are both kept, because `buildStreams`
decides which builder object serves which feature -- a grouping that is
observable to the visitor whether or not the passes overlap in time.

COST: none in output; `buildStreams` passes of latency instead of one.

MEASURED: merging the two inner loops was sabotaged and moved NO gate, which
is the honest statement -- at one queue it changes nothing any check can see,
and it is kept apart on their authority rather than on ours. What the width
DOES change is coverage: with 7 features at width 3 the loop runs three
groups and slot 0 serves features 11, 14 and 17. Three separate batch
sabotages (never re-running `SetIndices`, indexing the cat column by batch
position instead of feature id, advancing the outer loop by 1) are each
invisible in the first group and each caught in the second. That fixture
shape is the whole reason gate 8 has teeth.

`RequestStream` only ever GROWS the pool, so the returned width can be
smaller than the pool, and builders keep the previous batch's state until
`SetIndices` -- which is what makes the first of those three sabotages a
silent wrong answer rather than a crash.

## 118. Dense cat codes and no `currentBins` cache in the batch builder

THEIRS: `VisitCtrBinBuilders` reads `TCompressedCatFeatureDataSet` (packed
`ui64` blocks, GPU- or CPU-resident) and calls
`AddCompressedBinsWithCurrentBinsCache(currentBins, ...)`
(`ctr_bins_builder.h:113-125`) with a `currentBins` computed ONCE before the
loop.

OURS: `gbdt/ctrs/ctr_bins_builder.mojo` holds dense category codes rather
than packed blocks (that decompression deviation is already recorded there),
and its `add_cat_feature_bins` is their `ProceedNewBins(uniqueValues)` -- the
arm that recomputes `CurrentBins` from its own `Indices` first. The two
things the builder actually reads off their manager and dataset
(`GetFeatureCpu(id)`, `GetBinCount(id)`) are passed in directly, indexed BY
FEATURE ID as both of their accessors are.

WHY VALUE-IDENTICAL, not merely close: the loop resets the builder to
`baseTensorIndices` immediately before every add, so
`ComputeCurrentBins(Indices)` and `ComputeCurrentBins(baseTensorIndices)`
read the same array. The cache is a saved pass, not a different answer.

COST: one extra O(rows) pass per categorical feature instead of one per
batch. `_set_indices` is their `SetIndices` (`ctr_bins_builder.h:32-52`)
written as a free function so the lane touched no file in `gbdt/ctrs/`.

### 118a. `NCB::IsSubset` reads backwards at the call site

`NCB::IsSubset(Splits, other.Splits)` asks whether OURS is contained in
THEIRS -- two `std::includes`, subset first. Reversing it turns gates 4 and 5
red, and it is the kind of argument order a port gets wrong silently because
both arguments have the same type.

## 120. RUNG 2 IS A SECOND SEARCHER AFTER ALL: `PORTING.md` 91 B's "three lines" was wrong, and `docBins` is what it costs

Ported 2026-08-21, `gbdt/methods/oblivious_tree_structure_searcher.mojo` +
`gbdt/methods/oblivious_tree_bin_builder.mojo`, gated by
`pixi run check-feature-parallel-identity`.

**91 A HELD.** The two data layouts build a bit-identical compressed index at
device count 1; every step of both `TCudaFeaturesLayoutHelper` walks
coincides, and the feature-parallel searcher reads the index this repository
already builds. Nothing found here contradicts it.

**91 B DID NOT, IN THREE WAYS.** It said the two searchers are "one
implementation templated on the mapping, with two `TSubsetsHelper`
specializations whose `CreateSubsets` differ in exactly three lines", and
`PORTING.md` 119 and `NEXT_TWO.md` both priced rung 2 off that sentence. Both
have been corrected in place.

1. **`TSubsetsHelper<TMirrorMapping>` HAS NO `CreateSubsets`.**
   `pointwise_optimization_subsets.h:72-105` declares `Split` and two
   `CurrentPartsView` overloads and nothing else; only the Stripe
   specialization declares `CreateSubsets` (`:127-128`). The feature-parallel
   `CreateSubsets` is a METHOD ON THE SEARCHER
   (`oblivious_tree_structure_searcher.cpp:29`). There is no pair to differ.

2. **THE TWO `Split`s ARE DIFFERENT CODE CALLING DIFFERENT KERNELS.**

       Stripe  UpdateBinFromCompressedIndex(cindex, feature, bin,
                 docsForBins, depth, Bins)                   ONE kernel
               (`pointwise_optimization_subsets.cpp:35-40`)

       Mirror  UpdateBins(Bins, nextLevelDocBins, docMap,
                 CurrentDepth, FoldBits)                     reads docBins
               (`pointwise_optimization_subsets.h:82`)

   `docBins` does not exist on the doc-parallel path. It is one `ui32` per
   document in ORIGINAL document order whose bit `d` is the document's side
   of split `d`, and filling it is `TTreeUpdater::AddSplit` ->
   `WriteCompressedSplit` -> `UpdateBinFromCompressedBits`: **three kernels
   and a bit-packed intermediate where the doc-parallel arm runs one
   kernel.**

3. **THE CALL SITE DIFFERS, NOT ONLY THE BODY.** `treeUpdater.AddSplit(
   bestSplit)` (`:276`) precedes the subsets `Split` (`:283`), and the bit it
   writes is `BinarySplits.size()` -- exactly the `loadBit` the next
   statement reads. Two order-critical statements present on one arm only.

**WHY THEY DO IT THE LONG WAY.** The packed `TMirrorBuffer<ui64>` is what
`TScopedCacheHolder` CACHES, keyed by `(dataset scope, split)`
(`oblivious_tree_bin_builder.cpp:84-87`, `:124-132`). One bit per document is
1/32 of a `ui32` bin array, which is what makes caching every split in a tree
affordable, and `docBins` after the last level IS the model's leaf assignment
-- `Fit` ends `CacheBinsForModel(..., std::move(docBins))` (`:300-304`) and
the feature-parallel leaves estimator reads it. The doc-parallel searcher
caches nothing and estimates leaves inline from the partition stats it
already holds (`..._doc_parallel_...cpp:150-157`), so it needs neither.

**THE ANSWER IS UNCHANGED; THE PRICE IS NOT.** At `FoldBits == 0`, one
device, one task and no CTR columns the two chains produce bit-identical
`subsets.Bins` at every level. 1,067 new lines of `gbdt/` say so.

### Two further differences 91 B does not mention, both inert at one fold

* **The bootstraps differ.** Feature-parallel calls
  `Bootstrap.BootstrappedWeights` then `MultiplyVector` on both planes
  (`:61-72`); doc-parallel calls `BootstrapAndFilter`, which can DROP ROWS
  (`..._doc_parallel_...cpp:216`). At `bootstrap_type=No` both are no-ops,
  and neither port carries the bootstrap anyway (DEVIATION 104).
* **`subsets.FoldCount` is 1 on Mirror and 0 on Stripe.** Mirror computes
  `initParts.size()`; Stripe hardcodes 0
  (`pointwise_optimization_subsets.cpp:12`). Their two `Fit`s pass
  `subsets.FoldCount` (`:93`) and the literal `1` (`:40`) to the same score
  calcer argument, so the two are the same number and nothing else reads the
  field. `FoldBits`, which IS read everywhere, is 0 on both. **That plus the
  initial bin fill is the whole of the real "three lines".**

### DEVIATION 120: four things their `Fit` does that this function does not

`ComputeWeakTarget` (`:57`, `:377-467`), the bootstrap block (`:59-73`) and
the tree-CTR block (`:136-147`, `:201-255`) are not here, and the fourth is
`MakeDocIndices` (`:476-498`), which theirs calls INSIDE the depth loop
(`:124`) and which allocates a fresh buffer per level to copy an array that
does not change; it is hoisted out of the loop.

The first two are DEVIATION 104 exactly, for its reason: this repository's
boosting loop already owns the gradient path for the greedy learner, and
forking it is the one thing that must not differ between two learners being
compared. The searcher takes the weak target already computed and returns
the STRUCTURE plus `docBins`. The third is rung 4. The fourth is
`MaxDepth - 1` allocations, same values.

`MakeDocIndicesForSingleTask` (`:469-474`) copies
`SingleTaskTarget->GetTarget().GetIndices()`, which at permutation id 0 with
no shuffle is the identity -- DEVIATION 105's assumption holding on this arm
too. **The gather is still performed rather than collapsed**, because
`docBins` is indexed by DOCUMENT and only `subsets.Indices` may be collapsed.

### The gate, and what each half of it catches

`check-feature-parallel-identity` runs four gates at TWO row counts:

    1 IDENTITY     splits/order/type against the doc-parallel searcher
    2 LEAF IDS     docBins per document vs a HOST recomputation
    3 COMPRESSION  WriteCompressedSplit's readIndices arm, never taken at
                   this rung, against its nullptr arm (PORTING_RULES 8)
    4 CONTROL      a moved-signal fixture that MUST split differently

    4,000  rows   1 compression block  -- blockIdx.x is always 0
    16,434 rows   3 blocks, last holding 50 keys

Both green: 3 splits identical, 16,434 leaf ids exact, 8 of 16 leaves
occupied, control differs.

### THE SABOTAGE TABLE, and four of its rows are the point

R = red, `.` = still green, R* = red at 16,434 and GREEN at 4,000.

    #  planted defect                                      1  2  3  4
    -  --------------------------------------------------  -  -  -  -
    1  feature_offset left as the COLUMN index             R  R  .  .
    3  TBinUpdater's OR made a STORE                       .  R  .  .
    4  CompressBlock bit position `id` not `63 - id`       R  R  .  .
    5  add_split reads the depth AFTER the push            R  R  .  .
    6  block stride zeroed in both new kernels             R* R* .  .
    7  UpdateFoldBins loadBit given CurrentDepth + 1       R  .  .  .
    8  compressed_split_size sized ceil(n / 64)            CRASHES THE RUN
    9  readIndices arm reading indices[offset] with no     .  .  R* .
       block base
   10  gathered_target flattened before submit_compute     R  .  .  R
    2  split_subsets_mirror given subsets.indices as       REFUSED BY THE
       doc_map                                             COMPILER

* **Sabotage 3 does not change the tree.** A STORE leaves bit `CurrentDepth`
  of `docBins` correct and only clears its neighbours, and `UpdateFoldBins`
  reads exactly that bit. Every split is identical; 3 documents in 4 land in
  the wrong leaf. **Gate 2 is the only thing in the repository that sees it**,
  and it sees it only because it compares PER DOCUMENT rather than a total
  ([[uniform-test-data-hides-permutation]]).
* **Sabotage 7 does not move gate 2**, the mirror image: `docBins` is written
  correctly and only `subsets.Bins` reads the wrong bit. The two gates cover
  different halves of one chain and neither is redundant.
* **Sabotage 4 does NOT redden gate 3**, which an earlier draft of the
  check's docstring asserted it would. Gate 3 compares two arms that share
  `CompressBlock`, so a defect in the shared half cancels. Sabotage 9 is what
  makes gate 3 non-vacuous and it reddens gate 3 ALONE.
* **Sabotages 6 and 9 are invisible at 4,000 rows.** A compression block is
  8,192 documents, so at rung 1's fixture `blockIdx.x` is always 0 and every
  `+ KEYS_PER_COMPRESS_BLOCK * block` term is identically zero. At 16,434 the
  first wrong document is row 8,193 -- the first document of block 1. That is
  `PORTING.md` 107's rule verbatim, and the second row count is the only
  reason those two rows are in this table rather than in the paragraph below.
* **Sabotage 2 is unwritable.** Mojo refuses it:
  `aliasing values passed mutably to 'doc_map' argument and passed mutably to
  'subsets' argument`. The wall DEVIATION 97.2 records makes the most obvious
  way to get `docMap` wrong a compile error.

**AND ONE SABOTAGE MOVED NOTHING AND IS KEPT FOR IT.** The first attempt at
sabotage 10 memset `gathered_target` immediately AFTER `submit_compute`
rather than before it, and all four gates stayed green -- the histogram had
already been built. `PORTING.md` 20 ("a reach sabotage has a WINDOW, at both
ends"), live, at the cost of one run.

There is NO sabotage switch in either shipped file. `PORTING_RULES.md` 8
says a switch that outlives its measurement is a defect, and a permanently
wired defect selector is one. Every row above is an edit, a run, and a
revert.

### Four things in their source worth knowing before touching this file

* **The packed layout is INTERLEAVED, not contiguous.** `CompressBlock`
  (`compression_helper.cuh:93`) puts key `BLOCK_SIZE*id + tid` in word `tid`
  at bit `63 - id`. Consecutive documents are in consecutive WORDS; the 64
  sharing a word are 128 apart. Reading it as `bits[k/64] >> (k%64)` gives
  the right multiset in the wrong places -- and every total still balances.
* **`CompressedSize` is `numBlocks * 128`, not `ceil(n/64)`.** The last block
  is allocated in full (`compression_helpers_gpu.cpp:249-254`). Sizing it
  tight crashed the run; that is sabotage 8.
* **`UpdateFoldBins` reads bit `CurrentDepth` and writes bit
  `CurrentDepth + FoldBits`** -- two different positions, because `docBins`
  carries no fold id while `subsets.Bins` packs it in the low bits. They
  coincide only at `FoldBits == 0`, **which is exactly why rung 2 can be an
  identity and rung 3 cannot.**
* **`docBins` after the last level IS the model's leaf assignment**, which is
  why `Fit` ends in `CacheBinsForModel` and why the doc-parallel searcher
  never needs one.

## 121. DEVIATION: `TTreeUpdater` has no test set

`TTreeUpdater` takes `LinkedTest` and `TestBins`
(`gpu_data/oblivious_tree_bin_builder.h:103-104`) and `AddSplit` mirrors
every split into the second array (`:204-206`), so the held-out pool carries
the same leaf ids and the leaves estimator can score it. This repository's
boosting loop owns the test pool elsewhere and hands the searcher one set of
documents.

A test-bin array with no reader is a field that is never checked. It is not
arithmetic, it changes no split, and it costs one
`UpdateBinFromCompressedBits` per level whenever a test pool exists -- which
is what the wiring will have to pay when the feature-parallel boosting loop
lands.

## 122. DEVIATION: no `TScopedCacheHolder`, so nothing is cached, and within one tree that is free

`TSplitHelper::GetCompressedBits` caches the packed split bits keyed by
`(dataset scope, split)` (`gpu_data/oblivious_tree_bin_builder.cpp:84-87`,
`:124-132`), so a split proposed again by a later tree, or needed again by
the test set or a tree-CTR tensor tracker, is a lookup rather than a pass
over the compressed index. Ours recomputes.

**WITHIN ONE TREE THE CACHE CANNOT HIT.** `Fit` breaks on
`result.HasSplit(bestSplit)` (`:266-268`) BEFORE calling `AddSplit`, so no
split is ever added twice. The cost is therefore one `WriteCompressedSplit`
per REPEATED split across trees, and the rung-2 identity is unaffected.

Implementing a cache whose only caller can never hit it would be a path with
no reachable branch, which this repository has been bitten by four times in
one day ([[reached-but-inert]]). It becomes worth having when the test set
(121) or tree CTRs (rung 4) give it a second reader.

## 123. DEVIATION: `CompressBlock`'s four-register accumulator is one register

`TCompressionHelper::CompressBlock` (`compression_helper.cuh:78-103`) keeps
`TStorageType compressedEntries[4]`, fills them from a `#pragma unroll 4`
inner loop, and OR-folds them at `:100-103`. This port uses one accumulator.

The value is the OR of the same 64 terms in the same order-independent
operation; the register split is instruction-level parallelism on a
dependence chain and nothing else. It is declared because it is a visible
textual difference in a kernel this rung's identity depends on, not because
anything was measured -- **no benchmark was authorised for this rung and none
was run.**

The other half of that function is NOT a deviation and is transcribed:
`if (tid < srcSize) dst[tid] = ...` guards in KEYS, not words, and matches
`DecompressBlock`'s `tid < dstSize` (`:118`) exactly. Sabotage 8 above is
what happens when the sizing rule that guard depends on is loosened.

## 124. DEVIATION: three of their directories land in one file under `gbdt/methods/`

`PORTING_RULES.md` 6 keeps their names and records renames. This rung breaks
the DIRECTORY correspondence and the reason is lane ownership, not design:

| ours | theirs |
|---|---|
| `gbdt/methods/oblivious_tree_bin_builder.mojo` | `catboost/cuda/gpu_data/oblivious_tree_bin_builder.{h,cpp}` |
| ... same file | `catboost/cuda/gpu_data/splitter.h` |
| ... same file | `catboost/cuda/gpu_data/kernel/split.cu` |
| ... same file | `catboost/cuda/cuda_util/kernel/compression_helper.cuh` |
| `oblivious_tree_structure_searcher.split_subsets_mirror` | `catboost/cuda/methods/pointwise_optimization_subsets.h:74-93` |

The symbols keep their names and every function cites its upstream file and
line, so the diff surface is intact; what is lost is that a reader looking
for `WriteCompressedSplit` under `gbdt/gpu_data/` will not find it.

**The correct homes are `gbdt/gpu_data/oblivious_tree_bin_builder.mojo` (with
the split kernels), `gbdt/gpu_util/kernel/compression.mojo` (for
`TCompressionHelper`, which is generic in `BitsPerKey` upstream and is
specialised to 1 here), and `split_subsets_mirror` beside `split_subsets` in
`pointwise_optimization_subsets.mojo` where CatBoost puts the Mirror
specialization.** Moving them is a pure rename plus the `DERIVATION_MAP.tsv` rows
and is not attempted while three lanes are writing this directory. **This
paragraph is the debt; a later round that moves them deletes it.**

## 130. THEIR one-byte dispatch is in TWO places, and two of the four accumulators had no fixture

`PORTING.md` 108 shipped three fixtures and named the 8-bit accumulator as
"the only one that reaches it at all". The same sentence is a statement about
the other three, and it was not checked: at 15, 100 and 254 borders the 5-bit
and 6-bit accumulators were never entered by any differential against
CatBoost.

Their rule, read out of their source rather than inferred from ours:

HOST, `pointwise_kernels.cpp:57-60`. All four widths are launched
unconditionally, each with its own count:

    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 4, 5)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 6, 6)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 7, 7)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 8, 8)

`FeatureCountForBits` (`folds_histogram.h:16-24`) sums `Counts[bit]`, and
`Counts` is filled at `feature_layout.cpp:23-32` by
`Counts[NCB::IntLog2(foldCount)]++`. **`IntLog2` is CEIL**
(`libs/helpers/math_utils.h:14-16`), which is the same trap that already cost
this port a day on the fold count. The only host gate is
`if (featureCountForBits)` (`pointwise_hist2_one_byte_templ.cuh:226`).

DEVICE, `pointwise_hist2_one_byte_templ.cuh:179-183`. Each launched kernel
refuses the blocks that are not its own:

    constexpr ui32 upperBound = (1 << BITS);
    constexpr ui32 lowerBound = BITS > 5 ? upperBound / 2 : 15;
    if (maxBinCount <= lowerBound || maxBinCount > upperBound) return;

with `maxBinCount` the MAX `TCFeature::Folds` over the block's four
CONSECUTIVE features (`GetMaxBinCount`, `split_properties_helpers.cuh:25-45`;
`feature += (blockIdx.x / M) * 4`, `:169`). So the ranges, in FOLDS -- which
for a float feature is the border count, `GetFoldsCount` returning
`binCount - 1` (`feature_layout.cpp:49-58`):

    5 bit    16..32    bench/oracle24.txt     NEW
    6 bit    33..64    bench/oracle48.txt     NEW
    7 bit    65..128   bench/oracle100.txt
    8 bit   129..256   bench/oracle254.txt

NOTE THE 15 AND NOT 16 in `lowerBound`: a feature with exactly 16 folds
belongs to the 5-bit kernel. Folds at or below 15 never reach this family --
`SplitByPolicy` (`compressed_index_builder.h:66-70`) against `MaxFolds()`
(`grid_policy.h:62-64`) sends them to HalfByte.

The host and device rules agree: `ceil(log2(f))` is 4 or 5 exactly on 9..32,
6 on 33..64, 7 on 65..128, 8 on 129..256.

BOTH NEW FIXTURES MATCHED CATBOOST ON THE FIRST RUN, 48 of 48 on both
searchers. **The gate is now 288 of 288 across six fixtures.** That the 6-bit
accumulator was correct is not what the item bought; that nothing could have
said so is.

`oracle_main.mojo` now PRINTS the accumulator each fixture reaches beside its
result (`PORTING_RULES` 8), and refuses to continue if any one-byte
4-feature group falls outside every range -- a group nobody claims has no
histogram written, and every split below it is a decision taken on zeros.

## 131. A GREEN DIFFERENTIAL IS NOT A REACH PROOF, and the four widths compute the same answer

The fixture built for the 6-bit accumulator that silently lands on the 5-bit
one still matches CatBoost split for split, because **both accumulators
compute the same histogram**. The whole point of having four is collision
avoidance -- copies per thread, threads per group, how two threads writing
one bin are resolved -- and none of that changes a value. So 48 of 48 on
`bench/oracle48.txt` is not evidence that the 6-bit kernel ran, and a fixture
that landed one width low would have been indistinguishable from one that
did not.

`checks/onebyte_reach_check.mojo` is the observation
(`pixi run check-onebyte-reach`). DEVIATION 115's rule decides what it may
build: everything that DECIDES the dispatch comes from the product --
`build_layout` assigns the policy, and `PolicyScoreHelper.__init__` builds
both `d_folds` (the array `GetMaxBinCount` reduces) and `folds_hist` (the
host gate). What the check plants is only what is INERT to both gates: the
target, the weight, the document order, one whole-dataset partition.

It then runs the four widths ONE AT A TIME into a zeroed histogram.

    R1   bench/oracle24.txt    24 folds -> 5bit:768   6bit:0     7bit:0     8bit:0
         bench/oracle48.txt    48 folds -> 5bit:0     6bit:1536  7bit:0     8bit:0
         bench/oracle100.txt  100 folds -> 5bit:0     6bit:0     7bit:3200  8bit:0
         bench/oracle254.txt  254 folds -> 5bit:0     6bit:0     7bit:0     8bit:8128

Each count is `2 * 16 features * folds` exactly, so the claiming width wrote
the WHOLE histogram and the other three wrote nothing. Two non-empty would
mean our ranges overlap; four empty would mean the fixture reaches no
accumulator and its differential result is about a histogram of zeros.

R2 IS THE SABOTAGE, and R1 is a constant without it. One fixture's data --
`oracle24`, whose bins run 0..24 -- re-declared at four fold counts, RAISED
only so no planted bin leaves its slot:

    24 -> 5bit:768   40 -> 6bit:800   80 -> 7bit:800   200 -> 8bit:800

The claim walks 5 -> 6 -> 7 -> 8 as only the declared fold count moves. (800
is `2 * 16 * 25` occupied bins, so the arithmetic is right as well as the
dispatch.)

## 132. The depth and feature-count sweep, and why the sabotage runs at EVERY cell

`NEXT_TWO.md` rung 5's second item. `pixi run oracle` varies the border
budget and nothing else: one depth (4) and one feature count (16). Both are
structural. Depth decides how many partitions the histogram helper carries
and therefore when the subtraction trick replaces a full pass. Feature count
decides how a policy's features are cut into four-feature blocks and how many
blocks the multiplier spreads over the document axis. Neither had ever been
varied against CatBoost.

`tools/catboost_oracle.py` now reads `ORACLE_DEPTH`, `ORACLE_FEATS`,
`ORACLE_ROWS` and `ORACLE_TREES` from the environment, the same mechanism
`ORACLE_BORDERS` already used. **`bench/oracle_d4_f16.txt` is BYTE-IDENTICAL
to `bench/oracle.txt`**, which is the check that the plumbing changed nothing
at the defaults. `ORACLE_FEATS` below 8 is refused rather than silently
reshaped: the target is drawn from columns 0, 3 and 7, and a fixture whose
target changed with its width would make every cell a different question.

`checks/oracle_sweep_main.mojo` (`pixi run oracle-sweep`) runs five depths
(1, 2, 4, 6, 8) by three feature counts (8, 16, 32), both searchers.
**Thirty cells, 1512 splits, ZERO disagreeing** in the stable state, losses
matching CatBoost to 8-9 significant figures at every cell.

IT DOES NOT RAISE ON A DISAGREEING CELL. `pixi run oracle` is the gate; this
is a matrix, and a matrix that stops at its first red cell reports one number
instead of thirty. Every cell prints at full strength whether it agrees or
not ([[no-dataset-cherry-picking]]); the driver exits non-zero at the end.

THE SABOTAGE RUNS AT EVERY CELL RATHER THAN ONCE, and that is the entry's
point. Fifteen cells of 96/96 with nothing checking them is exactly the shape
of a check that went quiet at parameters nobody had run before
([[sabotage-when-required]]: the path is new). Moving the EXPECTED bin of
tree 0 depth 0 up by one must cost exactly one match and put the first
divergence at (0, 0), per cell. Every cell did. The sharp expectation is what
earns it: a comparison that is REACHED but not POSITIONED -- one that tallies
totals, or compares the tree as a SET of splits -- passes the green run and
passes a loose sabotage too.

## 133. The one-byte accumulators at DEPTH 8, the cell both sweeps left empty

The four bit-width fixtures (DEVIATION 130) are all at depth 4. The
depth-by-features matrix (DEVIATION 132) is all at 15 borders. So after both,
the HALF-BYTE kernel was still the only one that had ever run at a depth
other than 4, and the intersection was empty.

That intersection is not a formality. Depth is what decides when the
histogram helper stops doing full passes and starts recovering a sibling by
SUBTRACTION, and the 8-bit accumulator is the one holding Int32 fixed point
(DEVIATION 93). A subtraction of two quantized histograms is exactly where
that representation would show, and nothing had asked it to.

Four more fixtures, depth 8, 16 features, one per accumulator, in
`oracle-sweep`:

    5bit    24 borders   96/96 greedy, 96/96 pointwise   0.2671046853   vs 0.2671046536
    6bit    48 borders   96/96 greedy, 96/96 pointwise   0.2117507160   vs 0.2117507078
    7bit   100 borders   96/96 greedy, 96/96 pointwise   0.1910397112   vs 0.1910397046
    8bit   254 borders   96/96 greedy, 96/96 pointwise   0.1910049319   vs 0.1910049232

768 more splits, no divergence. The sweep stands at 38 cells and 2280 splits,
and with the gate's 576 the port reproduces CatBoost on 2856 splits across ten
fixtures, three grid policies and all four one-byte accumulators.

### 133a. What is still NOT covered: mixed-width feature groups

Their guard claims a block by the WIDEST of its four consecutive features, so
features of 3, 5, 40 and 200 folds all go to the 8-bit kernel together. Every
fixture here is uniform-width, so that per-group behaviour is printed but not
differentially checked. It needs an oracle whose columns get DIFFERENT border
budgets, which `pool.quantize(border_count=)` does not express directly.
Recorded as a hole, not as coverage.

## 134. OPEN, AND THE MOST IMPORTANT RESULT OF THIS ROUND: an intermittent, one run in ~100, on BOTH searchers at once

While building DEVIATION 132's matrix, one sweep run came back with its FIRST
cell wrong, and it has not come back since:

    depth 1, 8 features   |  2/12  first div t2d0  |  8/12  first div t8d0
    greedy mse    4.679346084594727
    pointwise mse 4.383606433868408
    CatBoost      4.133252577078921
    correct, every other run: 12/12 and 12/12, both arms 4.1332526206970215

Three things about it, in order of how much they matter.

1. **THE TWO SEARCHERS DISAGREED WITH EACH OTHER.** `check-fit-pointwise`
   requires them to be BIT-IDENTICAL over twenty iterations. On this run they
   were not, and they diverged from CatBoost at DIFFERENT trees.
2. The model was coherent and wrong, not garbage -- a real tree, monotone,
   still beating the mean, several percent worse than it should be. That is
   DEVIATION 95's failure signature word for word, and it is the signature
   that nothing but a differential can see.
3. It was the FIRST cell of the process, so it is not accumulated state.

NOT REPRODUCED in ~100 further executions of that exact cell: three more full
sweep runs, thirty warm iterations of `d1_f8` plus thirty of `d1_f16` and
thirty of `d2_f8` inside one process (all clean, and the two arms
bit-identical in all ninety), and six FORCED COLD-RECOMPILE single-cell runs.

WHAT IS ESTABLISHED ABOUT THE MECHANISM, and it is half an answer.

`GLOBAL_NUMERIC_MODE = NUMERIC_FAST` is the shipping default
(`checks/numerics.mojo:74`). Under FAST on a column that HAS float atomics
-- Apple does -- `deterministic_flush_for` (`checks/kernel_matrix.mojo:1003`)
returns False and the GREEDY family's multi-block histogram flush takes
CatBoost's float `atomicAdd` (`hist_half_byte.mojo:527-528`, same shape at
`hist_binary.mojo:524` and `hist_2_one_byte_base.mojo:484-521`). The file says
so itself: *"Order-nondeterministic, exactly as theirs is, and CatBoost ships
it that way."* And that branch is LIVE at exactly this cell:
`active_block_count = min(ceil(p_size / (4 * BLOCK_SIZE)), max_blocks_per_part)`
(`hist_half_byte.mojo:170-176`), so a 4096-row root partition at depth 1 puts
about four blocks into every float cell in arbitrary order. Depth 1 with 8
features is the smallest grid in the matrix and the maximum contention on that
flush.

**IT DOES NOT EXPLAIN THE POINTWISE ARM, AND THAT IS THE ALARMING HALF.**
The pointwise family's atomic is `comptime if m > 1`
(`pointwise_hist2_one_byte_templ.mojo:393-398`,
`pointwise_hist2_half_byte.mojo:162-165`), and `m` comes from
`estimate_block_per_feature_multiplier`, whose loop is gated on
`(dsSize / multiplier) > 10000` (`split_properties_helpers.mojo:228`). At 4096
rows that is false on the FIRST test, so `multiplier == 1` always and the
pointwise flush is a plain store. Sweeping `gbdt/` for every other float
atomic leaves `histogram_utils.mojo:136`, which is Int32 and therefore
order-independent, and `models/cuda/evaluator.mojo:435`, which is in the device
evaluator and not on `fit`'s apply path (`compute_bins_and_add_kernel`).

**SUPERSEDED BY 134c: there IS a mechanism for the pointwise arm, it is not
arithmetic at all, and it has been reproduced on demand.** This paragraph is
kept because the reasoning that ruled the atomic out for that arm was correct
and is what sent the hunt somewhere else.

`RESUME.md:509-518` records the determinism story as closed on 2026-08-21 --
"every rep's loss is BIT-IDENTICAL". This observation contradicts that as
written; the claim is at minimum scoped more narrowly than it reads. That file
belongs to another session and was flagged rather than edited
([[fix-docs-on-discovery]] wants it fixed by whoever owns it).

### 134a. MEASURED: 600 warm reps clean, which RULES OUT the warm-process arm

`pixi run soak-determinism`, 600 reps of the exact cell, both searchers, one
process:

    runs disagreeing with CatBoost   0
    runs where the ARMS disagreed    0
    distinct greedy losses:     4.1332526206970215  x 600
    distinct pointwise losses:  4.1332526206970215  x 600

**EXACTLY ONE distinct loss per arm, 600 times each.** If the event were an
independent per-fit event at the observed rate of about one in a hundred, a
clean run of 600 has probability 0.99^600, which is under a quarter of one
percent. So it is not a per-fit event inside a warm process, and the
order-nondeterministic float atomic -- which fires on EVERY fit, warm or cold
-- is now a poor fit for the greedy arm as well as for the pointwise one. A
float reduction that reordered would show up here as a second loss value
differing in the last few bits, and there is no second value.

That is consistent with the one thing the original sighting said about itself:
**it was the FIRST work its process did.** So the hunt moves to cold
processes, which is the configuration it was seen in and the one the earlier
~100 runs mostly were not (six of them were cold, which at this rate is nearly
uninformative).

This also raises the prior on a defect whose window is process startup or
first-allocation rather than arithmetic -- a device buffer freed at its last
use with the next allocation landing on it, say, which is intermittent,
allocation-order dependent, produces plausible-but-wrong values rather than
crashes, and would strike whichever arm happened to land on the stale block.
That last property is the only account so far of why BOTH arms were wrong and
wrong DIFFERENTLY. A confirmed instance of exactly that bug was found and
fixed in `write_fold_based_initial_bins` the same day (DEVIATION 125a).

### 134b. THE CONDITION NOBODY WROTE DOWN: the sighting happened under 20-30 concurrent GPU processes

Three lanes were writing this repository at once when that run happened, and
one of them reports the box carrying **20 to 33 concurrent `mojo run`
processes for most of the session**, with a single check taking 8-10 minutes
where it normally takes under one. Every subsequent attempt to reproduce --
the ~100 by the lane, and the 600 warm reps in 134a -- was made on a box that
was comparatively quiet.

**That is a difference between the sighting and every reproduction attempt,
and it was not controlled for.** It is also a plausible mechanism in its own
right rather than merely an excuse: under device-memory pressure an allocation
that would normally get fresh memory can get recycled memory, and a Metal
command buffer competing with thirty peers is exactly the window in which a
buffer freed at its last use (DEVIATION 125a's bug, confirmed in this
repository the same day) gets its block handed to somebody else before the
enqueued work runs.

It would also explain the property nothing else explains: **both arms wrong,
and wrong differently.** A pressure-triggered allocation defect strikes
whichever arm happens to land on the reused block, so two arms in one process
get two different wrong answers, which is precisely what was seen and precisely
what an arithmetic account cannot produce.

DO NOT READ THIS AS "THE MEASUREMENT ENVIRONMENT WAS BROKEN, SO THE RESULT IS
DISCOUNTED." A learner that silently returns a well-formed wrong model under
memory pressure is a worse defect than one that returns a wrong model always,
because it will do it on a user's loaded machine and nothing will say so. If
the mechanism is resource exhaustion, the required behaviour is a LOUD FAILURE,
not a plausible tree.

THE EXPERIMENT THIS IMPLIES, and it is cheap: run the cold soak while the box
carries a comparable synthetic GPU load. If the anomaly returns under load and
never returns quiet, the trigger is identified even before the site is.

### 134c. THE POINTWISE HALF IS FOUND, FIXED, AND REPRODUCED ON DEMAND

**A buffer freed at its last use. Not the float atomic.**

`_estimate_and_apply` (`gbdt/methods/doc_parallel_boosting.mojo:542`) stages the
leaf offsets and sizes in two host buffers and copies them to the device:

    ctx.enqueue_copy(dst_buf=d_p_off, src_ptr=h_po.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p_sz,  src_ptr=h_ps.unsafe_ptr())

Those two `enqueue_copy`s are `h_po`'s and `h_ps`' LAST TEXTUAL USE, so Mojo
may free them there -- the copies are enqueued, not run
([[mojo-buffer-freed-at-last-use]]). And the very next host allocation is
`h_leaves` inside `make_bin_optimized_oracle`
(`leaves_estimation/pointwise_oracle.mojo:788`): **same pool, same dtype, same
length**, and filled `h_leaves[i] = i` immediately. If it lands on the freed
block, `d_p_sz` receives `[0, 1, 2, ...]`.

HOW IT WAS CONFIRMED, and this is the part worth copying. The event is one run
in a hundred and would not reproduce -- 600 warm reps (134a) and 150 cold
processes, all clean, one distinct loss. So it was not caught; it was
RECONSTRUCTED. Forcing exactly that write, on ONE tree, and comparing the
result against the recorded observation:

                            splits    first divergence    pointwise mse
    OBSERVED, one run in ~100   8/12    tree 8             4.383606433868408
    FORCED at call 7            8/12    tree 8             4.383575439453125
    forced at call 8            9/12    tree 9             4.383563995361328
    correct                    12/12    none               4.1332526206970215

Same split count, same divergence tree, and a loss agreeing to five
significant figures -- from a corruption chosen before the comparison was run,
at a site predicted by an audit rather than found by a search. Corrupting
EVERY tree instead gives 15.14, worse than the mean baseline, which is why the
single-tree shape is the one that matters: the observation was coherent and
wrong, and only a single-tree corruption is coherent and wrong.

THE FIX, and why it is not a `synchronize()`. Holding the two buffers past the
`ctx.synchronize()` this function already runs before applying the estimate
costs NOTHING: the copies are certainly complete there, so the blocks cannot be
recycled in between. A `synchronize()` at the copy would also work and would
cost one full device drain PER TREE, which is the wrong price for a lifetime
bug (`mojotrees-speed-mandate`: blow up the control plane, not the kernels).

Verified after the fix: the differential holds at 288 of 288 across six
fixtures on both searchers, and `check-fit-pointwise` passes.

### 134d. STILL OPEN: the greedy half, and the load condition

Two things this does NOT close.

**The greedy arm's 2 of 12 is unexplained.** The site above is on the pointwise
path only -- at this configuration `need_estimation` is False for the greedy
arm, which applies its leaves inside `run_tree_layout` on-device and never
enters `_estimate_and_apply`. So the observed run had TWO defects firing at
once, or one defect with two reaches. The audit that found the pointwise site
ranked `TTreeWorkspace.__init__`
(`greedy_subsets_searcher/greedy_search_helper.mojo:2466-2557`) as the greedy
arm's candidate: seven dead staging buffers, thirteen intervening allocations,
one `synchronize()` only at the very end, and two exact size-and-type matches
among the followers. It is fit-scoped rather than per-tree, which fits a
divergence at tree 2 only if a single bin-feature's descriptor moved. UNTESTED
-- **and the positive-control technique above is how to test it**, not another
soak.

**Eight further sites carry the same pattern** on the training path, ranked in
that audit. None is a false positive: the allocator was measured
returning a freed block after ONE intervening allocation while a copy was
outstanding. They are hazards of the same class, and this class cannot be gated
by re-running -- only by not writing it.

**The load condition (134b) is still uncontrolled.** 750 clean runs on a quiet
box do not speak to a box carrying thirty concurrent GPU processes, which is
what the sighting had and what makes a deferred-by-one block recycle plausible
in the first place.

WHAT WOULD CLOSE IT, cheapest first: build once at
`GLOBAL_NUMERIC_MODE = NUMERIC_IDENTICAL` and soak `d1_f8` a few thousand
times. If the greedy arm goes silent and the pointwise arm still drifts, there
is a second defect and it is not the histogram flush. **Nothing was tuned,
dropped or deferred to make the cell green; it stays in the matrix at full
strength.**

**NO SPEED NUMBER AND NO PARITY CLAIM SHOULD BE QUOTED PAST THIS ENTRY UNTIL
IT IS CLOSED.** A learner that produces a different model one run in a hundred
does not have a loss to compare, and the two searchers disagreeing with each
other is the specific thing `check-fit-pointwise` exists to forbid.

## 125. `CreateSubsets`' fold arm pays ONE extra partition reduce, per TREE and not per level

`gbdt/methods/oblivious_tree_fold_tasks.create_fold_based_subsets`.

THEIRS (`oblivious_tree_structure_searcher.cpp:29-43`): the ternary picks
`WriteFoldBasedInitialBins` or `WriteSingleTaskInitialBins`, the bins are
written FIRST, and `UpdateSubsetsStats(src, &subsets)` runs ONCE at the end.

OURS: `create_subsets` (`pointwise_optimization_subsets.mojo:1252`) is shared
with the doc-parallel arm, so it seeds `FillBuffer(Bins, 0u)` and reduces; the
fold arm then overwrites the bins with `write_fold_based_initial_bins` and
reduces AGAIN.

PRICED: one `update_subsets_stats` over `1 << FoldBits` partitions per TREE --
16 partitions over 1,463 documents at the gate's fixture, once, not per level.
The alternative is a second entry point into `create_subsets` that skips the
seed, and forking the allocation of a file BOTH searchers share to save one
reduce per tree is the worse trade.

NOT an arithmetic difference: the final state is identical, and
`check-ordered-boosting` O3 compares the bin of every position, the offset and
size of every partition and both partition stats, per cell. Sabotaging the
second reduce away leaves O3 and O5 red.

### 125a. And the fill's staging buffers must outlive the enqueue

`write_fold_based_initial_bins` allocates one buffer per partition, memsets it
to the bin, and `enqueue_copy`s it into place. A `DeviceBuffer` handed to
`enqueue_copy` as `src_buf` is LAST USED at the enqueue
([[mojo-buffer-freed-at-last-use]]), so dropping it at the end of the loop body
lets the next iteration's allocation land on the same block and the next
memset overwrite a copy that has not run.

MEASURED, and it is intermittent: the same fixture read a correct partition
array on one run and all zeros on the next, with no source change between them.
The buffers are now held in a `List` until after `synchronize()`. **The
sabotage that removes the hold was GREEN on the run it was tried**, which is
the point -- this class of defect cannot be gated by re-running, only by not
writing it. Sibling of DEVIATION 134's open intermittent, and the reason that
one deserves a real hunt rather than another hundred green runs.

## 126. `PolicyScoreHelper` hard-codes `foldCount = 1` at three sites, and the searcher REFUSES rather than growing a tree a fold axis short

`TScoreHelper` takes `foldCount` and hands it to BOTH halves
(`histograms_helper.h:361-365`): the first sizing the histogram
`(1 << MaxDepth) * FoldCount * binFeatures * 2` (`:147-151`) and passing it to
`ComputeHistogram2` as `gridDim.z` (`:74`), the second passing it to
`FindOptimalSplit`, whose `foldCount == 1` test IS the dispatch between the
plain and the dynamic scorer (`pointwise_scores.cu:537`).

OURS hard-coded 1 at three sites in `pointwise_scores_calcer.mojo`.

**CLOSED the same day it was opened.** `PolicyScoreHelper` and
`ScoresCalcerOnCompressedDataSet` now take `fold_count` (defaulting to 1,
their argument order: `TScoreHelper(policy, dataSet, foldCount, maxDepth,
...)`), it is stored as a field, and it reaches all three sites --
`ComputeHistogramsHelper`, `compute_hist2`'s `gridDim.z`, and
`find_optimal_split`'s plain-versus-dynamic dispatch. Nothing else changed:
`histogram_alloc_size` already multiplied by `fold_count`, so `d_hist` grows
on its own.

Why it mattered enough to fix rather than leave guarded: **none of the three
crashes at `FoldCount > 1`.** The histogram would be allocated a
`FoldCount`-th of the size it needs, the launch grid would be one fold deep,
and the scorer would take the PLAIN arm over a partition array whose leaf
stride is `1 << FoldBits`. The tree comes out well formed. That is the same
failure mode as DEVIATION 114's hardcoded one-hot flag, in the same file, four
hours apart -- a constant standing in for a value the caller owns.

The guard stays. It is now a cross-check between two ported objects rather
than a refusal: the searcher asks the calcer what fold count it was built at
and raises if it disagrees with the layout, which costs one host comparison
per tree and means the first caller to pass folds gets a named error instead
of a plausible tree. Gated both ways by `check-ordered-boosting` O7: it fires
with folds and does NOT fire without them.

Verified after the change: the CatBoost differential holds at 288 of 288
across six fixtures on both searchers, and O7 still fires.

## 127. THE DOC-PARALLEL SEARCHER IS THE WRONG HOME FOR FOLDS, and its fold arm is a deviation scheduled for deletion

Upstream's doc-parallel `CreateSubsets` hard-codes `FoldCount = 0;
FoldBits = 0;` (`pointwise_optimization_subsets.cpp:12-14`) and nothing can
give it folds: `TDocParallelObliviousTreeSearcher` is constructed by
`TDocParallelObliviousTree`, which `TBoosting` (`doc_parallel_boosting.h`)
drives, and that is the PLAIN learner (`PORTING.md` 91 F). Ordered boosting
lives ONLY in `TFeatureParallelObliviousTreeSearcher`.

So `fit_oblivious_tree_structure`'s `folds` parameter has NO upstream
counterpart. It exists because rung 3's wiring was written in the same session
rung 2 landed, against the only searcher that existed when that lane started.
`PORTING_RULES` 0b-ii is explicit that there is no third category of file, so
this is DECLARED and scheduled for deletion, not kept.

What is NOT a deviation is everything under it -- `create_fold_based_subsets`,
`make_fold_doc_indices`, the fold stripe, the histogram fold axis and the
dynamic scorer -- because the two searchers share that whole stack and it is
ported from the feature-parallel side. Moving the arm is three lines in
`fit_feature_parallel_oblivious_tree_structure`, which now exists
(DEVIATION 120).

### 127a. The observation gather is NOT optional past depth 0

At one task `docIndices` is the identity and
`Gather(observationIndices, docIndices, subsets.Indices)` collapses to
`subsets.Indices` -- which is what DEVIATION 105 records. At N tasks it does
not, and a position in the concatenated array is not a document id.

MEASURED SYMPTOM, and it is not the one you would predict: skipping the gather
splits the level into QUARTERS instead of halves. The predicate is identical
at both levels, so the two split bits must agree and half the partitions must
be empty; reading the wrong document makes them disagree, and **every partition
offset still tiles the array perfectly.** The gate found it as 1,463 of 1,463
positions with the wrong (bin, index) pair while every size looked plausible.

## 128. Ordered boosting supports THREE of the seven score functions, and it does not touch the shipped default

`FindOptimalSplitDynamic` (`pointwise_scores.cu:443-473`) has two arms and a
`default: throw std::exception()`:

    SolarL2                     -> FindOptimalSplitSolarImpl
    Cosine, NewtonCosine        -> FindOptimalSplitCosineImpl
    L2, NewtonL2, SatL2, LOOL2  -> throw

`FindOptimalSplitPlain` (`:468-521`) covers all seven. So four score functions
that work perfectly at `boosting_type=Plain` cannot be asked for with Ordered,
and this is upstream's own limit rather than a gap in the port.

IT DOES NOT BLOCK THE SHIPPED CONFIGURATION. The GPU default score function is
`Cosine` (`oblivious_tree_options.cpp:22`), which is one of the three.

Refused in the searcher rather than at the launch, so the message names the
option the caller set instead of a kernel they never asked for, and so a tree
is never half-grown before it fires. Both halves gated: O6 runs all seven at
fold count 12 AND all four refused ones at fold count 1, because a score
function that was simply broken would pass the first half.

## 129. The ordered document array is LONGER than the dataset, and `size` stops being the row stride

`CreateFolds` builds fold `k`'s `EstimateSamples` as `[0, R_{k-1})` -- nested
prefixes, not a partition -- so `GetTotalIndicesSize()`
(`oblivious_tree_structure_searcher.cpp:321-336`) is `sum_k (L_k + |eval_k|)`,
not `n`. At n = 600, g = 2.0, min_fold_size = 100 it is **1,463 positions for
600 rows**, and every document appears in six folds, each carrying that fold's
own cursor value. That IS ordered boosting; it is not a bookkeeping artefact to
be normalised away.

The consequence for this port is a parameter that has been one number since the
compressed index was written and is now two:

    n_rows      the compressed index's ROW STRIDE -- `TCFeature::Offset * n_rows`
    doc_count   the length of the concatenated array -- `ComputeHistogram2`'s `size`

`PolicyScoreHelper` already takes them at two different places (the constructor
and `submit_compute`), so no signature changes; the searcher passes different
values to each. `TL2Target.line_size`, `create_subsets`' `doc_count` and the
`ReorderBins` length are all the SECOND one.

### 129a. Their scorer is never told the depth

`leavesCount = Parts.Size() >> foldBits` (`pointwise_kernels.h:339-341`). The
scorer INFERS the leaf count from the size of the partition-stats buffer, which
works only because `UpdateSubsetsStats` resizes `PartitionStats` every level
(`pointwise_optimization_subsets.h:58`). This port allocates once at
`max_part_count` and passes `1 << depth` explicitly -- the same number by a
different mechanism, and worth knowing before anyone "simplifies" that resize
into a fixed allocation on their side of the diff.

### 129b. A gate that sized its spans from the port's own answer

The sabotage that swaps LEARN and TEST in `plan_fold_layout` moved NOTHING on
its first run: O3, O4 and O5 all sized their spans from `lay.parts`, so the
defect moved both sides of every comparison. O1 now builds an INDEPENDENT
partition table from `create_folds`' output and everything else reads that;
the sabotage then turns five gates red.

**Second time in this port a gate has been found checking the port against
itself** ([[mojotrees-code-not-source-of-truth]]), and both times only a
sabotage could see it. Its sibling: with a FLOOR `IntLog2` the layout agreed
with itself and the run HUNG rather than failing, because the validator called
the same function it was validating. `create_fold_based_subsets` now carries a
shift-only `fold_count <= 1 << fold_bits` check and O1 asserts ceil-ness
without calling `int_log2_ceil`.

## 136. THE SMALLER-SIBLING TIE-BREAK WAS INVERTED, and "the subtraction is exact" was never true

At every level one sibling's histogram is COMPUTED and the other is derived as
`parent - sibling`. Which one gets computed is `BuildNecessaryHistograms`'
choice (`split_properties_helper.cpp:1318-1324`):

    if (firstLeaf.Size < secondLeaf.Size) { smallLeafId = ids[0]; bigLeafId = ids[1]; }
    else                                  { smallLeafId = ids[1]; bigLeafId = ids[0]; }

`ids` is pushed in ASCENDING leaf index (`:1300`) and `MakeSplit` gives the
LEFT child the existing (lower) id and the RIGHT child the appended, higher one
(`:861-862`, `:976-977`). So `ids[0]` is the LEFT child, the strict `<` is on
the LEFT, and **on an exact size tie the `else` branch fires and the RIGHT
child is the one computed.**

This port had it the other way round. Both sites started `small` at the LEFT
child and moved it only when the right was STRICTLY smaller, so a tie kept the
LEFT:

    var small = i            # left
    var big = half + i       # right
    if right_sz < left_sz:   # ours: strict `<` on the RIGHT
        small = half + i
        big = i

**Since when.** The host copy, `split_properties_helper.build_necessary_
histograms`, has been inverted since it was written -- `409a16c`, 2026-08-19,
whose message is "The level planner runs and gets every case right". DEVIATION
94 (`2bbe6af`, 2026-08-21) moved the choice onto the device as
`kernel/split_resolve.plan_level_kernel` and carried the inversion with it, so
the SHIPPED path has had it since that commit and the host twin for two days
longer. Both are fixed here; the host copy is off the shipped path (only
`probe_main` and `structure_searcher_template`'s schedule call it) and is fixed
anyway so the two cannot drift.

The pointwise family's own smaller-sibling choice was ALREADY right and is
untouched: `split_properties_helpers.mojo:200` is
`left if left_part_size < right_part_size else right`, which is their
`(leftPartSize < rightPartSize) ? leftPartOffset : rightPartOffset`
(`kernel/split_properties_helpers.cuh:102`) exactly. That is why the pointwise
arm does not move in any measurement below.

### 136a. THE DEFENCE THAT SAID IT COULD NOT MATTER, AND WHY IT IS WRONG

Two comments argued the choice was inert, both now deleted rather than
annotated (`greedy_search_helper.mojo:2846-2851`, applied 2026-08-21 by the
lane holding that file; `split_resolve.mojo:152-155`, applied):

> The Int32 fixed-point accumulator makes the subtraction EXACT ... so WHICH
> sibling is computed cannot change a bit of either histogram.

The accumulator IS exact. The subtraction is not, and it is not for a reason
the accumulator cannot reach: **the conversion out of fixed point happens
BEFORE the subtraction.** `write_reduces_from_fixed_kernel` writes each cell as

    Float32(Int(q)) / fixed_scale

and `substract_histograms_kernel` then works in float32 on those values.
`choose_scale` targets 2^30 (`SCALE_HEADROOM_BITS = 3`, or `2^30 - 1 -
row_count` with a row count), so `q` sits far above float32's exact-integer
range of 2^24 and `Float32(Int(q))` ROUNDS. Parent and computed child are
rounded SEPARATELY, so

    round24(parent_q) - round24(left_q)   need not equal   round24(right_q)

and the derived sibling differs from the sibling that would have been built.
The division by `fixed_scale` is exact (a power of two) and does not rescue it.

Note which plane escapes: with unit weights, stat 0's cells are `count *
fixed_scale`, an integer times a large power of two, exactly representable
while `count < 2^24`. The defect lives on the GRADIENT plane, whose quantized
cells are arbitrary integers.

### 136b. MEASURED, on this port's own kernels

`checks/sibling_tiebreak_check.mojo`, 2026-08-21, M4. Fixture: 4096 rows, a
full factorial of 8 binary columns over 256 cells x 16 replicate rows (so every
binary split halves every leaf EXACTLY) plus 4 columns at 254 folds that are a
function of the within-cell index alone, hence independent of every leaf those
binary columns can carve out. 1024 histogram cells per leaf per stat.

**TIES FORCED, counted rather than assumed** -- recomputed from the fit's own
chosen splits, at every level the planner actually plans for:

    sibling pairs the planner saw   744   (24 trees x depth 6)
    EXACT size ties                 632   (85%, 10 of them both-empty)

**HISTOGRAM CELLS THE CHOICE MOVES.** One parent and its two children built by
the shipped histogram path at the scale the boosting loop would choose, then
the shipped `substract_histograms_kernel` run BOTH ways and each derived
sibling compared bit-for-bit against the sibling actually built:

    fixed_scale 65536, largest |accumulator cell| 526,938,496  (2^24 = 16,777,216)

    computed LEFT  -> derived RIGHT : 19 of 2048 cells differ,  worst 2 ulp
    computed RIGHT -> derived LEFT  : 24 of 2048 cells differ,  worst 3 ulp
    ------------------------------------------------------------------------
    cells the tie-break choice moves: 43 of 4096 compared      (1.05%)

So the answer to "can it change a histogram bit" is YES, about one cell in a
hundred, by up to three ulp.

**AND IT CHANGES CHOSEN SPLITS.** The same 24-tree fit, greedy-subsets
searcher, before and after the flip -- two trees out of 24 pick a different
split at their last level:

    tree 12  ...(3>0)(10>77)   ->  ...(3>0)(8>135)
    tree 16  ...(11>33)(6>0)...  ->  ...(10>182)(6>0)...

The final loss is IDENTICAL to all 16 printed digits (0.011631562374532223) on
both arms and on both searchers, and the pointwise arm's 24 trees are
bit-identical across the flip, as 136 predicts. The two that moved are
knife-edge score ties that a one-ulp histogram difference re-decided; tree 16
in fact moved INTO agreement with the pointwise searcher and tree 12 out of it.
That is the cost: not accuracy, but WHICH of two near-tied splits is taken, and
it was being taken differently from CatBoost on 85% of the pairs in a fixture
built out of binary features.

**`pixi run oracle` is unchanged, exactly.** Six fixtures x two searchers, run
on a clean checkout of `3406e50` with only this change applied:

    before: 12 lines of "splits matching CatBoost exactly: 48 of 48"
    after:  12 lines of "splits matching CatBoost exactly: 48 of 48"

and every `our final mse` line is byte-identical too. The oracle fixtures are
4096 x 16 continuous columns at 15..254 borders, where an exact sibling size tie
is rare; that is why a real differential against CatBoost could sit green over
an inverted tie-break for two days. **A green differential is not a proof that
an untested branch is right** -- the same lesson as 131, on a different branch.

### 136c. THE GATE, AND WHAT MOVES IT

`checks/sibling_tiebreak_check.mojo`. Four crafted sibling pairs, two of
them exact ties, through the SHIPPED `plan_level_kernel`, against their `:1318`
rule worked by hand. The sabotage is `plan_level_inverted_kernel` in the same
file: the pre-136 body verbatim, run on the same input, required to disagree on
exactly the tied pairs.

    arm                                   pairs disagreeing with CatBoost
    shipped plan_level_kernel (post-136)   0 of 4       gate PASSES
    plan_level_inverted_kernel (pre-136)   2 of 4       gate FAILS, on the 2 ties
    shipped plan_level_kernel (pre-136)    2 of 4       gate FAILS  (run at 3406e50)

The third row is the check run against the tree as it stood before this entry:
the defect itself is the sabotage, and it turns the gate red.

The file also refuses a fixture that forced NO tie, because a tie-break gate on
a fixture with no ties is decorative in the way 131 describes.


## 135. The border subsample copied a real function on the WRONG CODE PATH: 100k with replacement per feature, where theirs is 200k without replacement shared

Found 2026-08-22 by a symbol-by-symbol read of their host quantizer against
ours, not by a failing check -- nothing here was red, because nothing here
compares our borders to theirs.

**What stood in `gbdt/train.mojo`.** Each float column drew its own sample of
100,000 rows WITH REPLACEMENT, seeded per `(random_seed, feature index)`. The
comment above it cited `GetSampleSizeForBorderSelectionType`
(`private/libs/quantization/utils.h:132-136`) and `SampleArray`
(`utils.cpp:14-24`). **Both citations are real functions and both are on the
wrong path.** `SampleArray` is reached from `NCB::BuildBorders`, whose callers
in their tree are a unit test (`external_columns_ut.cpp:47`) and the GPU CTR
border builder. The 100,000 is that helper's DEFAULT ARGUMENT, and the
training pipeline overrides it.

**What their quantizer actually does.** `GetSubsetForBuildBorders`
(`libs/data/quantization.cpp:118-141`) builds ONE `TArraySubsetIndexing` and
hands the same one to every feature, through `GetArraySubsetForBuildBorders`
(`private/libs/quantization/utils.cpp:25-51`) at
`options.MaxSubsetSizeForBuildBordersAlgorithms`, which is **200000**
(`libs/data/quantization.h:37`). When the pool is not already shuffled that
subset comes from `SampleIndices` (`libs/helpers/sample.h:20-43`), whose own
first line of documentation is *"Sample k element indices without
repetition"*. Three differences at once, then: SIZE, REPLACEMENT and SHARING.

**The cost, measured.** Sabotaging the fix back to a with-replacement draw and
running `pixi run check-sample-indices`:

    k = 200,000 drawn from n = 464,809 (covtype's row count)
      with replacement     162,500 distinct   37,500 repeats
      without replacement  200,000 distinct        0 repeats

162,500 is `n * (1 - (1 - 1/n)^k)` to the digit, which is what makes the
same arithmetic trustworthy at the configuration that actually shipped: at
k = 100,000 the old draw reached about **90,030 distinct rows**, so every
border of every float column was chosen from 90k rows of evidence where
CatBoost uses 200k. Not a tie-break difference -- a different estimator.

**What is fixed.** `sample_indices_for_borders` in `gbdt/train.mojo` ports
`SampleIndices` with both branches and their predicate
`k > 1 && k > (n / log2(k))`; the default is 200,000; the subset is drawn once
and every column gathers through it.

**What is NOT fixed, and stays a deviation.** Their engine is
`TRestorableFastRng64` and ours is `TRandom`, so at the same seed the SET
drawn is not theirs -- only the semantics match. Their `CB_ENSURE_INTERNAL(n
>= k)` becomes a `>=` branch here. Their rejection arm returns `THashSet`
iteration order where ours returns insertion order, which is order-equivalent
because the sample is sorted before borders are built.

**BLAST RADIUS, stated precisely so this is not read as a benchmark result.**
`bench/interleaved` does NOT reach any of this: it quantizes with CATBOOST'S
OWN quantizer and hands both arms identical pre-binned uint8
(`tools/interleaved_prep.py:1-10`). So no standing number moves. What moves is
every real `fit()` above 100,000 rows -- the user-facing path and the
end-to-end arm.

**The lesson, which is the reusable part.** The old comment was not vague; it
was precise, cited file and line, and was checked by whoever wrote it. It was
still wrong, because the function it named is not the function the pipeline
calls. A citation proves someone read A line. Only following the call chain
from the entry point proves they read THE line.

## 140. LOGLOSS'S TEN NEWTON ITERATIONS: neither arm runs ten, and the two stop at different places

**Found 2026-08-21 by `checks/logloss_leaf_oracle_check.mojo`, the first
comparison of this port's leaf ESTIMATOR against CatBoost's own leaf values.**
Until it existed, the ten-iteration Newton walker -- CatBoost's DEFAULT for
Logloss (`private/libs/options/catboost_options.cpp:157-164`, then
`:315-329`) -- had exactly one gate, `checks/logloss_estimator_check.mojo`,
which compares the device against a float64 host reimplementation written in
the same file. That gate has teeth and it is not a comparison with CatBoost:
if our reading of their walker were wrong, the reimplementation would encode
the same misreading and agree perfectly.

### What was measured

`bench/oracle_logloss_leaves.txt`, 3000 rows x 8 features, depth 3, 12 trees,
Logloss / Newton / AnyImprovement, every pin of `tools/catboost_arm.py:55-75`
(Plain, bootstrap No, rsm 1.0, has_time, no boost-from-average, no model
shrink, random_strength 0), two arms differing only in
`leaf_estimation_iterations`:

| gate | leaves outside band | worst \|dleaf\| | worst relative | splits |
| --- | --- | --- | --- | --- |
| L2, one iteration | 0 of 96 | 4.6e-08 | 7.1e-07 | 36/36 |
| L1, ten iterations | 3 of 96 | 3.5e-03 | 3.3e-03 | 36/36 |

Band is `1e-5 + 1e-3 * |theirs|`, derived from float32 accumulation and
stated before the run. The grid agrees cell for cell (the fixture is below
the border subsample threshold, so DEVIATION 135 does not touch it), and
every split of every tree still matches, so this is the ESTIMATOR alone.

### Where the three cells come from

Tree 0, leaf 2: 235 rows, nearly one class, the only leaf of the eight whose
Newton iterate is still moving after six steps. Seven of eight agree to
better than 1.2e-06. A float64 transliteration of both their walkers, run on
their own tree-0 partition at a zero cursor, gives leaf 2 / learning_rate as

    6 accepted steps   -3.47066423   = CATBOOST at leaf_estimation_iterations=10
    8 accepted steps   -3.48222831   = OURS      at leaf_estimation_iterations=10
    10 accepted steps  -3.48332208   = what ten steps actually give

**CatBoost accepts six of its ten. We accept eight. Neither runs ten.**

Three independent confirmations. Their leaf values are bit-identical at
`leaf_estimation_iterations` 6, 7, 8, 9, 10, 11, 12, 20 and 40 -- frozen, not
slow. At `leaf_estimation_backtracking="No"`, the arm with no acceptance test,
CatBoost does run all ten and returns -3.48332222, our number and the float64
number. And the mean Logloss decreases strictly at every one of the ten steps
in exact arithmetic (step 7 by 1.7e-07), so nothing has converged.

### Why theirs freezes

Their CPU accept test is `valueAfterStep < lossValue`, STRICT
(`private/libs/algo/approx_calcer/gradient_walker.h:92`), and the value is
the objective METRIC. Logloss stores approxes exponentiated, and
`TCrossEntropyMetric::EvalSingleThread`'s expApprox arm computes every row
through **`FastLogf`** (`catboost/libs/metrics/metric.cpp:270-275`), whose own
header states accuracy ~1e-05 (`library/cpp/fast_log/fast_log.h:79-86`). On
ARM64 the SSE3 vectorized arm is compiled out and hands back an empty holder
with `tailBegin = begin` (`catboost/libs/helpers/short_vector_ops.h:245-257`),
so EVERY row takes that scalar `FastLogf` path. Step 7's true improvement is
5.0e-04 in the summed loss, against 3000 per-row terms each carrying ~1e-05 of
approximation error. Once the improvement falls into that noise the strict
test fails -- and after one failure the halving converges to a no-op, which
can never strictly decrease anything, so the walk is frozen permanently.

### Why ours freezes later

Ours is their GPU walker: `FunctionValue <= nextFuncValue`, NON-strict
(`cuda/methods/leaves_estimation/step_estimator.cpp:8-66`), against the
target kernel's `functionValue` reduced in float32 on device. Same failure
mode, different precision and a different tie rule, so it stalls two steps
later.

### THE PRICE, and why nothing is being changed

This is not a defect in the port. The direction, the halving, the shared
iteration counter and the acceptance rule are their GPU's, and the
symbol-by-symbol audit of `descent_helpers.mojo` against
`descent_helpers.cpp:128-204` stands. It is a CPU-versus-GPU divergence
INSIDE CatBoost that any faithful GPU port inherits, and this machine cannot
run their GPU to see which side of it their own GPU lands on
(`PORTING.md` 109). COPY, DO NOT IMPROVE: making our walker match their CPU
would mean adopting a `FastLogf`-precision acceptance test, which is neither
their GPU's behaviour nor better arithmetic.

**Blast radius.** Logloss and CrossEntropy fits, at their default ten
iterations, on leaves that are still moving at step six -- extreme leaves,
which are exactly the ones a deep tree makes many of. On this fixture the
worst per-row prediction gap is 4.4e-03 against 4.5e-07 at one iteration.
Nothing else in the repository is affected: RMSE takes one iteration and
DEVIATION 64's skip, and the nine losses of `check-loss-oracle` compare tree
0 only.

**What is left OPEN.** Whether their GPU stalls where ours does. It is
answerable on an NVIDIA box in one run -- fit the same fixture at
`task_type="GPU"`, `leaf_estimation_iterations` 6 through 12, and look for
the freeze -- and it is not answerable here.

**The gate stays RED.** `check-logloss-leaf-oracle` reports three cells of
ninety-six and raises. It was not given an allowance for those three,
because an allowance sized to today's divergence also swallows tomorrow's
regression. Three, all on extreme leaves, is the known state; anything else
is new.

### ADDENDUM 2026-08-22: the fold's WIDTH was the unported half, and it is now theirs

The entry above says ours stalls later than their CPU because our
acceptance value is "the target kernel's `functionValue` reduced in
float32 on device" -- and that was only half true. The DEVICE half was
float32; the HOST fold of the per-block partials was Float64, so our
walker's acceptance test could resolve improvements BELOW one float32 ULP
of the total. Their walker cannot: `functionValue` is one FLOAT32 scalar
filled by `FastInBlockReduce<float>` + `atomicAdd(float*)`
(`pointwise_targets.cu:275-279`) and read through
`static_cast<float>(ReadReduce(valueGpu)[0])`
(`pointwise_oracle.cpp:106`). The extra resolution was OUR deviation and
is the mechanism behind "ours accepts eight where theirs accepts six".

FIXED in `pointwise_oracle.mojo` (both oracle arms): the partials fold in
Float32, one fixed host order -- the order substitution for their atomic
stays, the WIDTH no longer deviates. The diagonal walker's epsilon was
also matched to their float literal: `Hessian + 1e-20f`
(`descent_helpers.cpp:87`) promotes 9.99999968e-21, not the exact decimal
1e-20 this port added (`descent_helpers.mojo`).

MEASURED: `check-logloss-leaf-oracle` moves from 3 of 96 outside band to
FIVE of 96 (trees 0, 6, 7 x2, 10), same worst cell and value (tree 0 leaf
2, 3.5947e-03), splits still 36/36, L2 arm still 0 of 96 (worst 4.6e-08).
The gate compares against CatBoost's CPU, whose FastLogf noise floor is
not our target; where a float32-valued walk stalls is legitimately
different from where a FastLogf-valued walk stalls, and the entry above
already ruled that matching their CPU is not the goal. `oracle` 48/48 and
`check-loss-oracle` (nine objectives) unmoved. What this changes at higgs
scale -- the walk now stops where their GPU's value resolution stops --
is the orchestrator's timing/accuracy run to take.


## 141. `leaf_estimation_backtracking` is not an option here, it is a constant

`doc_parallel_boosting.mojo:601` calls `newton_like_walker_estimate` with a
literal `BACKTRACKING_ANY_IMPROVEMENT`. Their
`ELeavesEstimationStepBacktracking` has three values and all three are legal
on their GPU -- `No`, `AnyImprovement`, `Armijo` -- with `Armijo` supported on
GPU ONLY (`private/libs/options/catboost_options.cpp:664` refuses it on CPU).
`gbdt/methods/leaves_estimation/step_estimator.mojo` implements all three
faithfully, including their literal `C = 1e-5`; nothing can select two of
them.

The default is `AnyImprovement`, so the constant is the right VALUE and no
shipped fit is wrong. It is recorded because DEVIATION 140 turned the
acceptance rule into a load-bearing choice rather than a formality: `No` is
the setting under which CatBoost's own CPU arm reproduces our leaf values
exactly, and it is currently unreachable from `train()`. Closing this means
threading `leaf_estimation_backtracking` from `TCatBoostOptions` through
`fit_with_test` to that call, and it is under `gbdt/options/` and
`gbdt/methods/doc_parallel_boosting.mojo`, which the session that found it
did not hold.


## 137. `random_strength`, and CatBoost's TWO standard deviations that are not the same number

`random_strength` was refused by `CatBoostOptions.check()` and the kernel
path for it was already written and INERT: `pointwise_scores.mojo`'s
`ScoreCalcer.get_score` carried the noise term and
`pointwise_scores_calcer.mojo` threaded `score_std_dev` and `seed` down to
it, but both oblivious searchers defaulted the parameter to zero and no
caller ever overrode it. The greedy family had dropped the term outright,
with a comment in `compute_scores.mojo` saying so. This lane wires the
caller and deletes that comment.

PORTED, at their sites:

  * `gbdt/methods/random_score_helper.mojo` <- `random_score_helper.h`.
    `ComputeStdDev`, `CalcScoreModelLengthMult`, `ComputeScoreStdDev`.
  * `compute_target_variance_kernel` in
    `greedy_subsets_searcher/kernel/compute_scores.mojo` <-
    `ComputeTargetVarianceImpl` (`compute_scores.cu:226-283`), and the host
    `compute_target_std_dev` in `greedy_search_helper.mojo` <-
    `greedy_search_helper.cpp:369-378`.
  * the noise term itself in the greedy cosine calcer <-
    `score_calcers.cuh:159-167`, COSINE ONLY. `TL2ScoreCalcer` (`:40-69`),
    `TSolarScoreCalcer`, `TLOOL2ScoreCalcer` and `TSatL2ScoreCalcer` have
    no noise term and did not get one.
  * `CalcScoreModelLengthMult(objectCount, iteration * step)` per iteration
    in `doc_parallel_boosting.mojo` <- `doc_parallel_boosting.h:358-359`,
    multiplied into the greedy searcher's option exactly where
    `greedy_subsets_searcher.h:73-76` multiplies it and passed to the
    doc-parallel arm as their `ModelLengthMultiplier`.

### THE BRIEF SAID THE TWO ARMS COMPUTE THE SAME PRODUCT. THEY DO NOT.

Asked to verify that claim, and it is half wrong. Same: three factors, the
model-length multiplier, a standard deviation of the weak target, and the
option. Different: THE STANDARD DEVIATION, in three ways.

  1. THE DENOMINATOR. `ComputeStdDev` divides `sum wt^2/w` by the OBJECT
     COUNT (`random_score_helper.h:14-15`). `ComputeTargetStdDev` divides
     the same numerator by the TOTAL WEIGHT
     (`greedy_search_helper.cpp:376-377`). Equal only when every weight is
     1. Under Newton -- CatBoost's default leaf estimation -- the weight
     plane is the hessian, so they are never equal on a Logloss fit, and
     for Logloss (hessian <= 0.25) the greedy arm's std dev is at least
     twice the doc-parallel arm's on the same target.
  2. THE ZERO GUARD. Their greedy kernel skips rows with `w <= 1e-15`
     (`compute_scores.cu:239`); `ComputeStdDev` has none and divides
     through `DivideVector` with `skipZeroes` false, so a zero-weight row
     with a non-zero gradient poisons the whole reduction.
  3. WHERE IN THE ITERATION IT IS TAKEN. The doc-parallel arm computes it
     BETWEEN the gradient and the bootstrap
     (`oblivious_tree_doc_parallel_structure_searcher.cpp:200-218`), so it
     is the UNBOOTSTRAPPED target's. The greedy arm computes it inside
     `CreateInitialSubsets` from `ComputeTarget`'s output, which has
     already been through `StochasticDer(bootstrapConfig, ...)`
     (`greedy_search_helper.cpp:381-385`), so it is the BOOTSTRAPPED
     target's.

All three are transliterated rather than reconciled. The two arms of this
port therefore disagree about the noise magnitude exactly as far as
CatBoost's two arms disagree, and no further.

### DEVIATION, PRICED: their two generic device ops are one fused kernel

`ComputeStdDev` is `DivideVector(tmp, Weights)` then
`DotProduct(tmp, tmp, &Weights)` over a materialized temporary. Neither
`DivideVector` nor `DotProduct` is ported in this tree, and
`gbdt/gpu_util/kernel/transform.mojo` carries only the gather/scatter
family. `std_dev_partials_kernel` fuses the two and never materializes
`tmp`. Per-row arithmetic is theirs exactly, including the missing zero
guard. Price: one fewer allocation and one fewer pass than theirs, and a
kernel that has to be re-read against `random_score_helper.h` rather than
against two library calls.

Second half of the same deviation: the weak target reaches that kernel as
ONE two-plane buffer (`stats`, plane 0 the weight and plane 1 the
gradient), not as `TL2Target`'s two buffers. That is this repository's
convention, the one `split_stat_planes` bridges with a host round trip; the
kernel reads both planes through one pointer, so the round trip is not paid
here.

## 138. `ComputeTargetVariance` in Float32, and the lane their host comments out

Theirs reduces `(weightedSum, weightedSum2, totalWeight)` in
`cub::BlockReduce<double>` and ends in three `TAtomicAdd<double>`
(`compute_scores.cu:265-281`). Metal has no fp64 and this repository does
not accept an order-dependent float atomic on the fit path -- the whole
reason `deterministic_sum_lanes_kernel` exists. So each block STORES three
Float32 partials at its own slot and the fold is the deterministic 3-lane
one. Same three sums, wider accumulation error, same bits every run.

Their `FillBuffer(aggregatedStats, 0.0, 3, stream)` before the launch
(`:293`) has no counterpart, because nothing accumulates into the
destination any more: the fold WRITES all three slots.

Lane 0, their `weightedSum`, is computed by the kernel and never read --
their host has `//        double sum = l2StatsCpu[0];`
(`greedy_search_helper.cpp:374`). Ported and discarded on the same line,
because a lane dropped for being unread is a lane that silently changes
nothing until someone reads it.

## 139. One RNG stream per fit becomes one per tree, and a per-level seed that was not advancing

CatBoost has ONE `TGpuAwareRandom` for the whole fit and draws
`NextUniformL()` from it once per `ComputeOptimalSplits` call -- i.e. once
per LEVEL, across every tree
(`oblivious_tree_doc_parallel_structure_searcher.cpp:86`, `:104`;
`greedy_search_helper.cpp:468`, `:489`, `:510`). Here the boosting loop and
the level loop live in different functions and the searchers take value
parameters, so the stream is chunked: the boosting loop holds
`TRandom(random_seed)` and draws ONE seed per TREE, and each searcher makes
its own `TRandom(tree_seed)` and draws one per LEVEL.

Price: the sequence of `GlobalSeed` values differs from CatBoost's, so a
fit with `random_strength != 0` is not bit-comparable to CatBoost's at the
same seed. Everything the option is FOR is preserved -- a distinct draw per
level, reproducibility from one seed, no correlation between features -- and
at `random_strength = 0`, the default, no seed is consumed at all.

**AND IT FOUND A BUG.**
`oblivious_tree_doc_parallel_structure_searcher.mojo` was handing `seed`
ITSELF to every level, not a fresh draw. Every level of a tree would have
drawn the same per-feature normal, turning the noise from a per-level
redraw into a fixed per-feature offset for the whole tree. Nothing caught
it because nothing ever passed a non-zero `score_std_dev` -- the parameter
existed, was threaded correctly all the way to the kernel, and was dead.
That is PORTING_RULES 8 in its purest form: the wiring was checked, the
VALUE was never non-zero, and the one line that depended on the value being
non-zero was wrong.

## 142. THE NOISE CANCELS ON THE GREEDY ARM, and three of the eight sabotages move nothing

`checks/random_strength_check.mojo`, `pixi run check-random-strength`.
Six gates, all driven through a real `fit`; nothing hands a kernel its
inputs (PORTING.md 115).

The thing the gate exists to say: **`random_strength` is a model change on
one searcher and an exact no-op on the other, and that is CatBoost's
behaviour, not a gap here.**

    GREEDY (`compute_scores.cu`)     `TScoreCalcer beforeSplitCalcer =
                                     calcer` is copied AFTER `NextFeature`
                                     (`:85`), so both calcers carry the
                                     same `GlobalSeed` and the same
                                     `FeatureId`, both `GetScore()` calls
                                     add THE SAME DRAW, and
                                     `gain = score - scoreBefore` (`:134`)
                                     removes it. All three of their greedy
                                     kernels (`:131-134`, `:370-375`,
                                     `:459-464`).
    DOC-PARALLEL                     `gain = (noisyScore -
    (`pointwise_scores.cu:402`)      scoreBeforeSplit)` where
                                     `scoreBeforeSplit` is an UNNOISED HOST
                                     scalar carried from the previous
                                     level. Nothing cancels.

So the reach proof the brief asked for -- "at `random_strength = 1.0` the
splits must differ" -- is the RIGHT gate on the doc-parallel arm and the
WRONG one on the greedy arm, where the correct gate is the opposite
equality. Both are in the file, as R2 and R2G.

### Results, 3000 rows, 10 features, depth 4, 12 iterations, RMSE

    R1   random_strength=0.0 == the parameter unset, BOTH arms:
         48/48 splits, every leaf value and every loss identical
    R2   doc-parallel, 0.0 -> 1.0: 45 of 46 splits moved,
         12 of 12 losses moved
    R2G  greedy, 0.0 -> 1.0: 48 of 48 splits and all 12 losses IDENTICAL.
         The float32 residue of `fl(fl(s - n) + n)` flipped nothing at
         `random_strength = 1`; the cancellation is exact in real
         arithmetic and held to the bit here. It would NOT hold at a noise
         magnitude that absorbs the score -- see the R5 note below, where
         a deliberately leaked term at 1e6 moved every split by absorption
         alone.
    R3a  same seed twice: 46/46 splits identical
    R3b  seed 7 vs seed 99: 43 of 46 splits moved
    R5   score_function=L2, 0.0 -> 1e6: 48/48 splits identical
    R4   PLACEMENT. Eight features that are all THE SAME COLUMN, so every
         candidate's real score is identical across features and the argmin
         is decided purely by `noise(GlobalSeed + FeatureId)`. The winning
         FEATURE at all four levels of tree 0 was predicted on the host
         from the same seed arithmetic and matched: features 0, 4, 7, 1.
         Three of the four are not the tie-break's answer (which is always
         feature 0), so this is placement and not a total.

### The sabotage table, run

    mutation                                        moved
    ----------------------------------------------------------------
    greedy calcer: advance_seed_k(..,4) -> 3        NOTHING
    greedy calcer: seed = global_seed, no FeatureId NOTHING
    greedy: drop the before-calcer's `-= noise`     R2G, 44 of 48 splits
    run_tree_layout: force score_std_dev = 0        NOTHING
    pointwise calcer: advance_seed_k(..,4) -> 3     R4, all four levels
    pointwise calcer: seed without + FeatureId      R4, three of four
                                                    (depth 0 coincides with
                                                    the tie-break answer)
    pointwise: noise leaked into the L2 calcer      R5 -- but ONLY at 1e6
    boosting loop: force score_std_dev = 0          R2, R3b, R4

TWO GATES WERE DECORATIVE AND ONE STILL IS.

  * R5 was written at `random_strength = 1.0` and the L2 leak moved
    NOTHING, because the noise magnitude is calibrated off the target's
    standard deviation while an L2 score is `sum^2 / (w + lambda)` summed
    over thousands of rows -- the leak was orders of magnitude below the
    gap between candidates. FIXED by running R5 at 1e6:
    `TL2ScoreCalcer::GetScore()` returns `Score` unchanged at ANY
    `ScoreStdDev` (`score_calcers.cuh:57-60`), so the gate is entitled to
    pick a magnitude no leak survives. It now catches the leak in both
    forms tested (with and without `FeatureId` capture), 36-40 of 40
    splits.
  * THE GREEDY COPY OF THE SEED ARITHMETIC IS UNCHECKABLE THROUGH ANY
    MODEL, AND STAYS SO. Its `+ FeatureId` and its four advances feed a
    term that cancels, so no fit on that arm can distinguish them from any
    other seed. That is not fixable by a better gate; it is a property of
    their algorithm. What IS checked is that the term REACHES the kernel:
    deleting the before-calcer's share of the noise moves 44 of 48 splits.
    The arithmetic itself is checked by R4 on the pointwise copy of the
    same three lines, and by reading. OPEN, and recorded rather than
    papered over.

### Documents this falsified, fixed in the same change

  * `OPTIONS.md:100-102` said `random_strength` is "a NO-OP on their GPU
    symmetric path". True of the greedy searcher and FALSE of the
    doc-parallel one, which is the searcher CatBoost uses for single-target
    symmetric trees. Sentence replaced, not annotated.
  * `compute_scores.mojo` said the noise term "is NOT ported: it is
    `random_strength`, which `CatBoostOptions.check()` refuses". Deleted.
  * `compute_scores.mojo` said `beforeSplitCalcer` "is NOT ported". It is
    now, and it has to be: it is what carries the cancelling draw.
  * `doc_parallel_boosting.mojo`'s header said `CalcScoreModelLengthMult`
    is "NOT A DIVERGENCE HERE" because the option was unported. Rewritten.
  * `CatBoostOptions.random_strength`'s docstring said "no score noise is
    applied here". Rewritten to say what each arm does, with their line
    numbers.

### The refusal that was lifted, and the surface it was on

`CatBoostOptions.check()` refused any non-zero `random_strength`. That
refusal is now narrowed to two cases: a negative value, and a non-zero
value paired with `score_function` L2 or NewtonL2, where NO calcer of
theirs has a noise term at all. CatBoost itself accepts that pairing and
discards the value; this port refuses it, because an option that reads as
live and is not is worse than one absent.

**AND IT GATES NOTHING TODAY.** `CatBoostOptions` is constructed by exactly
one thing in this tree, `checks/options_check.mojo`. No fit path
builds one. `gbdt/train.mojo` is where real callers arrive and it takes
`random_strength` as a plain parameter with no validation of that pairing.
Closing that is a `train()`-signature lane, not this one; it is named here
so the next reader does not mistake `check()` for a guard.

## 144. `secondDerAsWeights` LANDS: the Newton score functions stop being their twins in name only

(143 is claimed in-source by the pointwise searcher's cross-tree pool --
`oblivious_tree_doc_parallel_structure_searcher.mojo:134` -- whose entry
belongs to that lane; this one takes the next number after it.)

`checks/second_der_weights_check.mojo`, `pixi run
check-second-der-weights`.

CatBoost decides what the histogram's WEIGHT plane holds FROM THE SCORE
FUNCTION: `secondDerAsWeights = IsSecondOrderScoreFunction(scoreFunction)`
(`greedy_search_helper.cpp:286-296`; `enum_helpers.cpp:830-846` says
exactly NewtonL2 and NewtonCosine). Under a second-order score function
their `StochasticDer` hands `&weightsView` -- COLUMN 0 of
`StatsToAggregate` -- to `Approximate` as the der2 output
(`pointwise_target_impl.h:193-201`), and the kernel fills it
`der2[i] = weight * target.Der2(relev, val)` (`pointwise_targets.cu:
268-269`; the cross-entropy kernel's `der2[idx] = weight[j] * scale[j]`,
`:373-375`) -- THE SAMPLE WEIGHT FOLDED IN. First-order keeps
`weightsView.Copy(weightsForIndices)` (`:203-213`). The doc-parallel
searcher keys the same choice as `NewtonAtZero` vs `GradientAtZero`
(`oblivious_tree_doc_parallel_structure_searcher.cpp:195-207`). Until now
this port always wrote the raw weight, so `NewtonCosine` fit a
bit-identical Cosine model and `NewtonL2` an L2 one; both were refused by
name at the Python surface for exactly that reason, and the refusal is
now lifted (`SCORE_FUNCTIONS` in `python/mojolearn/ensemble.py`).
SolarL2, LOOL2 and SatL2 STAY REFUSED: no calcer, and an option accepted
and ignored is worse than one absent.

THEIR CPU ARM CANNOT DISAGREE BECAUSE IT REFUSES: `MakePointwiseScoreCalcer`
is "Only Cosine and L2 score functions are supported for CPU"
(`algo/score_calcers.h:106-115`, and again `leafwise_scoring.cpp:530-550`).
`secondDerAsWeights` is GPU-only upstream, so the GPU arm is the only
semantics there is, and it is what this mirrors. (Their GPU LOSSGUIDE
default is NewtonL2, `catboost_options.cpp:980-991` -- so upstream this
flag is on by default on that policy; the symmetric-tree default stays
Cosine, which is why the shipped oracle never moved.)

### The port, and the three differences that are ours

- `second_der_as_weights` is a COMPTIME parameter on
  `pointwise_target_kernel` / `cross_entropy_kernel` and their two
  launchers, defaulted False, where theirs is a runtime bool deciding
  which buffer pointer the der2 write lands in. Same substitution
  DEVIATION 63 already records for the objective switch: the host branch
  (`fit_with_test`, the fits' ONE der launch, which BOTH searchers'
  planes come from) picks the instantiation. `estimation=True` plus the
  flag is refused by a `comptime assert`: their `ApproximateAt` has no
  such flag.
- Their `CB_ENSURE(!secondDerAsWeights, "MultiClass loss doesn't support
  second derivatives...")` (`multiclass_targets.cpp:27`) is raised at FIT
  ENTRY here rather than inside the first tree's der launch -- same first
  observable effect, no half-grown tree.
- The fixed-point magnitudes (ours, no CatBoost counterpart) bound plane
  0 AS STORED: `|weight * der2|` under the flag, `|weight|` otherwise.
  Anything else would hand `choose_scale` a bound for a plane that is not
  the one being accumulated.

The Python-side `random_strength` refusal widens from `'L2'` to
`('L2', 'NewtonL2')`, which `CatBoostOptions.validate` already said: the
noise term lives only in the Cosine calcer and NewtonL2 runs the L2 one.

### Gates and results (M4, 2026-08-21)

    S1  PER-CELL, hashed distinct weight/target/prediction per row (a
        total is satisfied by a permutation), 3000 rows, against a
        Float64 host recomputation. Logloss covers cross_entropy_kernel,
        POISSON covers pointwise_target_kernel -- both with non-constant
        der2, because for RMSE the flag is invisible by construction.
        First-order plane 0 must be the raw weight BIT-EXACTLY.
        Result: 0 bad cells in all 8 plane/arm combinations.
    S2  THE MODEL MOVES, Logloss, 4000 rows x 7 features, depth 4, 8
        trees: greedy NewtonCosine vs Cosine 2 of 32 splits and 48 of 128
        leaf values differ; NewtonL2 vs L2 29 of 37 splits, 48 leaves.
    S3  NEGATIVE CONTROL, RMSE: NewtonCosine vs Cosine 0 of 32 splits, 0
        of 128 leaf values -- bit-identical, forced by
        `TRmseTarget::Der2 == 1.0f`, NOT a tally of ours.
    S4  BOTH SEARCHERS: S2 and S3 identical numbers on the pointwise arm
        (2/32+48, 29/37+48, 0+0).

### Sabotage table, each run and reverted

    A   cross_entropy_kernel second-order arm stores `weight`
        -> S1 Logloss 2nd-order 3000/3000 bad; S2 all four pairs
        identical (5 gates). S3 unmoved, as it must be.
    A2  pointwise_target_kernel second-order arm stores `weight`
        -> ONLY S1 Poisson 2nd-order (3000/3000). Without the Poisson
        arm this branch is gated by NOTHING: S2 is Logloss (other
        kernel) and S3's RMSE coincides by construction. That is the
        reached-but-inert shape, and S1b is the one check with teeth on
        it.
    B   host passes second_order=False unconditionally
        -> S2 fails on BOTH searchers (4 gates); S1 does not move (it
        launches the kernel directly). S1 and S2 gate different layers.
    C   search-mode stores swap planes 0 and 1
        -> S1 Logloss both arms, both planes, 3000/3000 (placement, not
        a total).
    D   `is_second_order_score_function` returns True for Cosine too
        -> S2's NewtonCosine-vs-Cosine pairs collapse on both searchers
        (2 gates); the L2 pairs still differ; S3 STILL PASSES -- the
        RMSE identity holds even for a misclassified Cosine, which is
        the analytic point of the control.
    E   second-order arm stores raw `der2` WITHOUT the weight
        -> S1 Poisson 3000/3000 AND S3 fails on both searchers (7 of 32
        splits, 128 leaf values moved): the "did they fold the weight
        in" trap, and the sabotage that proves S3 has teeth.

Every gate is moved by at least one sabotage; none is decorative.

`pixi run oracle` 288 of 288 before and after (GPU symmetric default is
Cosine, first-order, so the shipped configuration must not and did not
move). One transient 14/40 pointwise-searcher failure during the
before-run reproduced as 288/288 and coincided with another lane actively
editing `pointwise_scores.mojo` / `pointwise_scores_calcer.mojo` /
`pointwise_split_resolve.mojo` in this shared checkout mid-compile; it is
that lane's in-flight state, not this change (mojotrees-shared-checkout
rule: the working tree is a moving target).


# =====================================================================
# DEVIATION NUMBERS 350-399 ARE THE DEPTHWISE LANE'S
# =====================================================================
#
# Assigned 2026-08-22 and agreed with the LOSSGUIDE lane (which took
# 300-349) BEFORE either lane wrote a numbered entry. The gbdt lane sits
# at 144, the forest lanes at 200-215 and 300-314 (`extratrees/
# DEVIATIONS.md`, `ensemble/PLAN.md`) -- so 300-349 collides with the
# ensemble block on paper and the lossguide lane knows it; 350-399 is
# clear of everything in the tree today.
#
# The rule this exists to enforce is PORTING_RULES.md 3, earned on
# 2026-08-20 when two lanes both claimed DEVIATION 42 concurrently and
# the entries had to be renumbered after the fact.

## 350. `TTreeNode`'s `ui16` fields narrow silently in theirs and raise here

`TTreeNode` is four `ui16` (`gpu_data/gpu_structures.h:167-171`), and their
flat model builder assigns a `ui32` `FeatureId` straight into it
(`model_builder.cpp:257`). A dataset with more than 65,535 features, or a
subtree wider than 65,535 leaves, wraps and produces a model that applies
cleanly and answers wrong.

`model_builder._to_ui16` raises instead. The type is UNCHANGED -- widening it
would make a model file theirs cannot read -- so this cannot change any model
their code would have built correctly. It is the cheapest class of deviation
there is: it converts an unrepresentable state into an error and has no
effect on any representable one.

## 351. `TBinFeatureTable`: their per-candidate walk becomes an O(1) lookup

`ToSplit(FeaturesManager, props)` (`methods/helpers.cpp:164-170`) resolves one
candidate at a time out of the features manager, and `resolve_split` in this
port is the same thing: an O(features) walk per call.

The symmetric arm resolves ONE candidate per level. The depthwise arm resolves
`argmaxBlockCount * leavesToVisit.size()` -- up to 64 x 64 -- because the host
reduce runs per leaf over per-block records (`greedy_search_helper.cpp:520-531`).
At 2000 features that is a walk of 8 million steps per level to answer a
question whose answer is a table.

So the table is built once per fit, indexed by bin-feature. **It is filled BY
`resolve_split`**, so it cannot disagree with the function it replaces --
and `checks/depthwise_check.mojo` claim 1 asserts that cell for cell
anyway, because "cannot disagree by construction" is a sentence this
repository has been wrong about before (`build_layout`'s two walks agreed only
because every fixture happened to be binary-first).

Bit-inert. HOST_AND_DEVICE.md rule one holds: the table is sized by the total
BIN count, which scales with features times borders and never with rows.

## 352. Partition stats are recomputed per level, not updated inside the split

Their `TSplitPointsKernel` updates `subsets->PartitionStats` as one of its five
steps ("Update part stats", `split_properties_helper.cpp:918`), so their
`ComputeOptimalSplits` finds the stats already correct for the new leaves and
`AllReduceThroughMaster` is only the multi-device fold.

This port has never ported that half. `run_tree_layout` calls
`compute_partition_stats` at the top of every level instead, and the depthwise
driver does the same rather than growing a second mechanism that would have to
be kept in step with the first.

**The cost is one extra reduction per level; the values are identical.** It is
the same reduction over the same rows with the same PINNED chunk count
(`IDENTITY_PATHS.md` row 7), so this cannot move a bit -- it can only cost
time. The depthwise arm pays it once more than the symmetric arm does, at
termination, because the final split moved the partitions after the last
`ComputeOptimalSplits` and the leaf values are read from those stats.

Priced and deferred: closing it means porting the stats update into the split
chain, which is a change to a file the symmetric lane owns, and it is worth
doing only once this lane has a number that says how much of a level it is.

## 353. `target_variance_blocks` is pinned under IDENTICAL (row-7 class; twin of 252)

Their `min(4 * TArchProps::SMCount(), CeilDivide(size, blockSize))`
(`compute_scores.cu:291`) sizes the greedy arm's target-variance reduce from
the machine's core count. The kernel strides by `gridDim.x * blockSize`, so
the block count decides which rows form each FLOAT partial; the partials feed
the score-noise std dev, and through it every noised score of the fit. Inert
at this port's default `random_strength = 0`; CatBoost's default is 1.0.

Pinned INSIDE `target_variance_blocks` (`kernel/compute_scores.mojo`) -- the
one place the formula lives, so the launch and the `partials` sizing in
`compute_target_std_dev` cannot disagree -- through the SAME
`kernel_matrix.partition_chunks_sm_for` row 7 uses: device count under FAST
(CatBoost's behavior bit for bit), 32 under IDENTICAL. DEVIATION 252 is the
doc-parallel twin (`random_score_helper.std_dev_blocks`), mirrored exactly.
The same commit routes `compute_target_variance_kernel`'s within-block fold
through `pinned_block_sum` (DEVIATION 251's family -- this kernel was a
producer site row 8's checklist had not listed).

## 354. Histogram replication is pinned under IDENTICAL (row-7 class)

`replication_for` (`greedy_search_helper.mojo`) is their
`CeilDivide(blocksPerSm * SMCount(), x*y*z)` (`hist_binary.cu:95` and twins),
and it LOOKS like pure occupancy. For the hist_2/one-byte families it is:
they quantize per row (`hist2_quantize`) and sum in Int32, so any partition
of rows into blocks gives the same bits. The binary and half-byte families
accumulate their shared histograms in FLOAT and the deterministic flush
quantizes the per-block PARTIAL (`Int32(val * fixed_scale)`), so
`active_block_count` -- derived from this replication -- decides which rows
form each rounded partial. A core count in that formula is a summation order.

Under IDENTICAL the formula is fed `partition_chunks_sm_for`'s pin (32
everywhere); under FAST the device's count, unchanged. One pin for all three
policies, because a pin only some policies read cannot be audited.

**KNOWN RESIDUE, not closed by this entry:** `kernel_matrix.block_size_for`
is not identical-gated, so the float families' BLOCK SIZE (hence
`min_docs_per_block`, replica count and `HIST_SIZE`) still follows the
vendor's shared-memory budget under IDENTICAL -- NVIDIA's 48 KB yields 768
where the identity floor's 32 KB yields 512. The comptime accessor needs the
same identical-gating `spec_for` (the runtime report) already has. That edit
is the kernel matrix owner's; reported 2026-08-22.


# =====================================================================
# DEVIATION NUMBERS 250-299 ARE THE SYMMETRIC LANE'S (250 reserved)
# =====================================================================
#
# Assigned by the orchestrator 2026-08-22. 250 is held for the parked
# partitions-reduce patch and is NOT claimed by either entry below.

## 251. `pinned_block_sum`: the within-block float fold does not follow the wavefront under IDENTICAL

IDENTITY_PATHS row 8's remainder. Every fv/magnitude producer
(`pointwise_target_kernel`, `cross_entropy_kernel`, both multilogit value
kernels, `std_dev_partials_kernel`) reduced its per-block partial with MAX's
`block.sum`, whose internal cross-lane fold runs at the HARDWARE'S warp
width -- 32 on Apple and NVIDIA, 64 on AMD's CDNA wavefront -- so the
partial's bits differed on the AMD column even with the partial counts pure
f(n_rows) and the cross-block combine already the fixed-order
`deterministic_sum_lanes_kernel`.

`pinned_block_sum[block_size]` (`gbdt/targets/kernel/pointwise_targets.mojo`)
is the REPLACE move: under `NUMERIC_IDENTICAL` a shared-memory halving tree
at exactly `block_size` lanes, no warp primitives -- the fold shape
`deterministic_sum_lanes_kernel` and `bootstrap.mojo`'s magnitude tail
already use -- one shape on every vendor. Under `NUMERIC_FAST` it IS the
`block.sum` call, bit for bit, so the shipped default cannot move.
CatBoost's own reduce at these sites is `FastInBlockReduce` -- a shared-
memory tree of their own -- ending in a float `atomicAdd` this port already
replaced (DEVIATION 71 and the pointwise determinism fix), so the IDENTICAL
arm is, if anything, closer to their in-block shape than the library call
was.

## 252. `std_dev_blocks` is pinned under IDENTICAL, the way row 7 pinned `partition_stats_chunks`

`min(4 * SMCount(), CeilDivide(size, blockSize))` (`compute_scores.cu:291`)
sizes the score-noise std-dev reduce (`random_score_helper.mojo`), and a
partial COUNT is a summation order: the machine's core count decided the
last bits of `random_strength`'s noise magnitude. Inert at this port's
default `random_strength = 0`; CatBoost's default is 1.0, so the pin lands
BEFORE anyone wires that default.

Pinned INSIDE `std_dev_blocks` -- the one place the formula lives, read by
the launch, the `partials` sizing and the fold count alike -- through the
SAME `kernel_matrix.partition_chunks_sm_for` row 7 uses (32 every vendor,
deliberately no real device's own number, mode-gated exactly as row 7 is:
device count under FAST). The greedy arm's twin, `target_variance_blocks`
(`compute_scores.mojo:663`), has the identical hazard and is the greedy
lane's file; reported, not touched.

## 257. `EnsureNewtonIsAvailable` was never ported; the Huber / Quantile+Newton / MAE+Newton hash coincidence is sha256 of zeros

Found 2026-08-23 explaining an E2-matrix oddity: `Huber delta=1.0`,
`Quantile leaf_estimation_method=Newton` and `MAE ... Newton` all hashed
`66adb9a97127bab9...` on the 20000x24 E2 fixture. That hash is the sha256
of 20,000 float32 ZEROS, not of three equal models. The chain, every link
theirs: the fixture's target sits in [3.74, 10.55] and the cursor starts
at 0 (Huber is not on `AdjustBoostFromAverage`'s list,
`train_lib/options_helper.cpp:362-368`), so on tree 0 every |residual|
exceeds delta and Huber's `Der2` is 0 on every row
(`pointwise_targets.cu:86-93`) exactly as Quantile's and MAE's always are;
Huber's default is Newton at one iteration (`catboost_options.cpp:
187-192`); the leaf Hessian is `0 + l2` (`pointwise_oracle.cpp:86-89`) so
the leaf is `sum(+-delta) / l2` (`descent_helpers.cpp:87`) -- 2812/3 and
17188/3 on tree 0 here; the Cosine score of a constant-sign gradient ties
every split at sqrt(n) (winners.scores = 141.42136 at all six depths), so
each tree is the same depth-1 split (`HasSplit` break,
`oblivious_tree_doc_parallel_structure_searcher.cpp:134-136`); the next
tree sees every residual saturated the OTHER way and its leaf is the exact
negation (`tree 0: +281.2/+1718.8`, `tree 1: -281.2/-1718.8`); after an
even tree count the sum is exactly 0. CatBoost CPU 1.2.10 on the same
fixture diverges the same way (Huber delta=1: predictions in [-786, 669],
MAE 79.8; delta=3: [-1910, 2276]; delta=30, where the quadratic arm is
live: MAE 0.24). THEIR SEMANTICS, KEPT: the Huber arm of
`pointwise_targets.mojo` is theirs symbol for symbol, `loss_delta` reaches
the kernel (delta 0.1 / 30 hash differently; 1 and 3 coincide because
both saturate every row), and `check-loss-oracle` holds Huber against
their CPU on a fixture whose residuals straddle delta (1474 of 3000 rows
inside, 1526 outside). A comment now sits on the Huber `Der2` arm saying
why the zero arm must not be "fixed".

WHAT WAS OURS: `EnsureNewtonIsAvailable` (`catboost_options.cpp:
588-601`, called from `Validate` on the RESOLVED method, `:740-742`)
refuses Newton for Quantile, MultiQuantile, MAE, LogLinQuantile, MAPE, the
two Stochastic* ranking losses and Lq with q < 2 -- catboost 1.2.10 says
"Newton leaves estimation method is not supported for Quantile loss
function" at fit time. This port accepted the override and ran a
zero-Hessian Newton step CatBoost cannot be configured to run. Ported as
`ensure_newton_is_available` in `gbdt/options/catboost_options.mojo`,
called at the end of `set_leaves_estimation_default` (the function that
resolves the method, the same value their `Validate` sees), GPU arm: the
CPU-only pairwise clause is false here and the Stochastic* losses are not
ported. `check-options` gains `check_newton_availability`: the five
refusals plus Lq q=1.5, AND the acceptances (Lq q=2.5, Huber, RMSE, and
Quantile's default resolving to Exact), sabotaged once (guard removed ->
"Newton on Quantile should have been refused"). The E2 matrix's
`gbdt_quantile_newton` cell now records a REFUSE by their message (the
tool classifies "not supported" as a refusal, a result), which is what
CatBoost itself does with that configuration.

Bit-inert for everything that still trains: Huber / Quantile / MAE /
Lq / RMSE hashes identical before and after on the E2 fixture, rebuilt
from a clean HEAD checkout to separate the rebuild from the change
(NOTE: the `_mojolearn_gbdt.so` that produced the brief's Huber numbers
was STALE against HEAD -- Huber Gradient `2cda0e6e` / Exact `91c9d0ca`
are not reproducible from HEAD's sources; HEAD-clean gives
`6b6c42839b1ab54c` / `0d5c5af4c718f02c`, and RMSE's `da34f396f968e546`
matches E1_RESULTS.md, so the rebuilt .so is the recorded one).
`check-loss-oracle` all nine pass, `check-pointwise-target` PASS,
`check-options` green.


## 259. `grow_policy` reaches `train()` and the Python surface: Depthwise and Lossguide boost non-symmetric trees

Landed 2026-08-23. Until this entry `doc_parallel_boosting.fit_with_test`
grew oblivious trees only and `catboost_options.check()` refused
`grow_policy != SymmetricTree` by name, honestly, because three things were
missing (UNWIRED.md's depthwise section named them): a policy on the
searcher call, their `TAddModelDocParallel<TNonSymmetricTree>` apply, and a
model / model-text that knows a second tree shape. All three are here.

WHAT IS THEIRS, AND WHERE. `TGpuTrainer<TPointwiseTargetsImpl,
TNonSymmetricTree>` is registered per (loss, policy) in
`cuda/train_lib/pointwise_non_symmetric.cpp:7-29` -- eleven losses x
{Lossguide, Depthwise} -- and its searcher is
`TGreedyTreeLikeStructureSearcher<TNonSymmetricTree>`
(`structure_searcher_template.h`), which this port already had as
`greedy_search_helper_depthwise.fit_non_symmetric_tree` (one driver, four
policy branches). The loop's per-tree step (`doc_parallel_boosting.h:
353-398`) does not change shape: target at the cursor, `optimizer.Fit`,
`NeedEstimation` -> `estimator.Estimate` per permutation (bins off the
model, `doc_parallel_leaves_estimator.cpp:43-56`), `Rescale`,
`AppendModels`. The non-symmetric arm in `fit_with_test` is exactly that:
`fit_non_symmetric_tree` -> `compute_non_symmetric_bins_for_model` per
permutation -> `partition_from_bins` -> `_estimate_and_apply` (the SAME
estimator the pointwise arm uses; it runs unconditionally, as the pointwise
arm's does, since the non-symmetric searcher never applies its own leaf --
DEVIATION 64's RMSE shortcut belongs to the greedy oblivious arm and its
numbers coincide here by 64's own derivation) -> `UpdateLeaves` +
`Rescale(step)` folded into the stored values -> `add_non_symmetric_model`.
The held-out arm and `predict` go through
`gbdt/models/add_non_symmetric_tree_doc_parallel.mojo`, their
`add_non_symmetric_tree_doc_parallel.cpp:182-216` (bins, then
`AddBinModelValues`, which is the `add_bin_model_value_kernel` the
estimator's `MoveTo` already had). `TAdditiveModel` carries
`non_symmetric_models` beside `weak_models` -- their trainer returns a
`std::variant` of the two `TAdditiveModel<T>` instantiations
(`train.cpp:436-455`) and a model is one shape or the other, enforced by
the two adders; `is_oblivious()` is their `TModelTrees::IsOblivious()`
(`model.h:271-273`, "no non-symmetric step nodes"). The model text gains
`ntree <t> nodes <n> dim <k> weights <0|1>` + one `node` record per
pre-order `TTreeNode` (+ `split_type take_bin` on a one-hot split) + `leaf`
per bin x dim + `weight` as a 64-bit token (their `LeafWeights` is
`TVector<double>`); an oblivious file's bytes do not move and an older
reader meets `ntree` as an unknown keyword and refuses. The GPU evaluator
refuses a non-oblivious model with their own message
(`libs/model/cuda/evaluator.cpp:25`).

WHAT IS REFUSED, EACH WHERE THEY REFUSE IT (Python, `train()`, and the
loop all say it by name):
- a loss with no non-symmetric trainer: `Lq` and `MultiClass` here --
  "Error: optimization scheme is not supported for GPU learning
  Loss=...;OptimizationScheme=..." (`train.cpp:279-280`, the registry
  `pointwise_non_symmetric.cpp:7-29`, `multiclass.cpp:5-14`);
- `use_pointwise_searcher=True` with a non-symmetric policy: that is
  `TDocParallelObliviousTreeSearcher`, oblivious by construction;
- `max_leaves` off `1 << depth` on any policy but Lossguide
  ("max_leaves option works only with lossguide tree growing",
  `catboost_options.cpp:993-1001`); over 65536 under Lossguide
  (`oblivious_tree_options.cpp:130-133`); depth over 16 on the full-binary
  policies (`:126-129`, `enum_helpers.cpp:870-875`);
- `min_data_in_leaf != 1` under SymmetricTree -- OURS, STRICTER: theirs
  discards it (`greedy_search_helper.cpp:685`), their docs say it is for
  Depthwise and Lossguide only, and an option accepted and dropped is
  PORTING_RULES.md 3's failure;
- `grow_policy='Region'`, no lane.
And one DEFAULT mirrored: CatBoost's GPU resolves an unset `score_function`
to NewtonL2 under Lossguide (`catboost_options.cpp:980-991`), Cosine
elsewhere, so `GradientBoosting(score_function=None)` does the same.
Ordered boosting / multi-host / PerTreeLevel sampling refusals of theirs
(`catboost_options.cpp:757-771`) have nothing to refuse here: this port is
Plain, one host, per-tree sampling.

MEASURED. Reach by contrast on the E2 fixture (20k x 24, FAST, this M4):
SymmetricTree `a1637c58ea8c9470` (UNCHANGED from the certified E2 hash,
IDENTICAL `da34f396f968e546` also unchanged), Depthwise `59593f3d1a15d560`,
Lossguide `213392e43fe6fc88`; Lossguide max_leaves 4 `d9ebebcea304e568`;
Lossguide min_data_in_leaf 500 `56e64daa9f829861`; Depthwise
min_data_in_leaf 500 `28c4dea293f4e01c`. Depthwise and Lossguide at a depth
the leaf budget does not bind (depth 4, max_leaves 16) COINCIDE bit for bit
on this fixture at either score function -- every leaf improves at every
level, so "split every improving leaf" and "split the best leaf until the
budget" close on the same full tree; they differ as soon as the budget
binds (depth 6 vs 31 leaves) or `min_data_in_leaf` does.
`checks/grow_policy_check.mojo` is the gate (seven claims: the
three-way contrast, the shape, the knobs, train/predict consistency, the
`ntree` round trip, five refusals, and a run-to-run control at the 8-bit
histogram shape); `check-depthwise`, `check-lossguide`,
`check-lossguide-policy`, `check-options` and the growth cards stay green;
`bindings/build_gbdt.sh`'s smoke gate fits all three policies, round-trips
the non-symmetric ones through save/load and checks the refusals. The E2
matrix gains `gbdt_rmse_depthwise`, `gbdt_logloss_depthwise`,
`gbdt_rmse_depthwise_minleaf200`, `gbdt_rmse_lossguide`,
`gbdt_rmse_lossguide_leaves8`, `gbdt_rmse_lossguide_cosine` (fits) and
`gbdt_rmse_depthwise_pointwise`, `gbdt_multiclass_lossguide` (REFUSED by
their message, the passing verdict); their cards carry `treeNNN.dN.*`
tags only (the non-symmetric driver took a `tag_prefix`, empty for the
single-tree gates so their cards are byte-identical).

## 260. The non-symmetric arm runs at the kernel matrix's `HIST2_SMEM_MODE`, and the first measurement that said it could not was a race

The lane gates (`check-depthwise`, `check-lossguide`, the E2 growth cards)
drive `fit_non_symmetric_tree` at accumulation mode 0 (CatBoost's
warp-private float); the boosting loop is the first caller to run it at
the matrix row -- shared-Int32 on Apple and under IDENTICAL. The first
run-to-run control at 20,000 x 24 x 128 borders said the Int32 arms were
wrong through this driver (64 borders: loss 41.898 vs mode 0's 38.323; 128
borders: 43.30 / 43.30 / 43.32 across three runs vs 38.299) while the
symmetric driver's two modes agreed bit for bit at every depth on the same
fixture. Bisected by depth: agreement through depth 3, divergence from
depth 4, the shape at which leaves first fit one histogram block. The
cause was DEVIATION 261, not the arms: after it the two modes agree bit for
bit through the non-symmetric driver too (`0x8cedc2e5a5fc1ae5` at depth 6,
128 borders, FAST; IDENTICAL deterministic at 38.299009). The matrix row
stands for this arm; pinning mode 0 here would have been an inline
vendor/arm fork hiding a race. Recorded because the wrong reading lasted
an hour and the table is the evidence either way.

## 261. The non-symmetric driver staged three id lists per level through ONE host buffer under queued copies

`TDepthwiseWorkspace` had one `h_ids`/`d_ids` pair, and a level wrote three
host-built lists into it before its first drain -- `plan.compute_ids` for
`zero_histograms_kernel`, `non_zero` for the histogram kernels and the
scan, `0..len(leaves)` for `compute_partition_stats` -- each followed by
`enqueue_copy(dst_buf=d_ids, src_ptr=h_ids)` and no wait. The device reads
the host buffer when the COPY EXECUTES, so list 2 or 3 could be what
kernel 1 received: the step-33 race class (`[[mojo-buffer-freed-at-last-
use]]`'s host-staging sibling; the symmetric lane's DEVIATION 134 is the
same mechanism on a different buffer). Invisible to every lane gate (4,096
rows, one tree) and to every traced run (a trace drains at every record);
found by the boosting loop's untraced run-to-run control at 20k x 24 x 128
(DEVIATION 260's table), where the faster Int32 histogram kernels let the
host reach the next write first and the float arms happened not to. Fixed
with one staging pair per list (`h_zero_ids`/`d_zero_ids`,
`h_all_ids`/`d_all_ids` beside the build pair); the two drains their
control plane already takes per level (`score.read`, `split.sizes`)
separate one level's writes from the next's. `check-depthwise` (7 claims),
`check-lossguide`, `check-lossguide-policy` and the growth cards are
unchanged and green; `check-grow-policy` claim 7 is the control that
fails without this fix.
