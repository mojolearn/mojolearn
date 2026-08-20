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
this.** `mojo_only/fixed_point.mojo` and its overflow proof stand.

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


## 5. Vector loads not ported

CatBoost instantiates a different `TComputeHistogramImpl` per load width and
picks 2 or 4 elements by arch (`point_hist_half_byte_template.cuh:34-41`).
The port takes the `OneElement` specialization only. Scheduling, not numeric:
the same values are added in the same order, fewer at a time.

## 6. `AlignMemoryAccess` peel omitted

It exists to align the vector loads of item 5. At one element per load there
is nothing to align. Required the moment the load width moves above 1.

## 7. THE FLOAT ATOMIC FLUSH (THE PREMISE WAS WRONG)

CatBoost flushes every histogram with `atomicAdd(dst + fold, val)` on
`float`. This section used to assert that Metal has no floating-point atomic
add at all, that the instruction does not exist, and that the port therefore
could not be literal here on our primary target.

**That was false, and it was the load-bearing claim under
`mojo_only/fixed_point.mojo`.** Probed 2026-08-19 on the M4: 1024 threads
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

Lives at `dbscan/gbdt/neighbors/epsilon_neighborhood.mojo` (the
`updateVertexDegree` section): the sub-warp reduction's width is a comptime
constant and every lane reaches every shuffle unconditionally, because a lane
that skips a full-mask shuffle hangs the lanes that reach it. Its block-size
sweep (cited by LANE_rbc-maxk: 142.10 against 129.08 ms at 200k) is the bar
any K_LIB wiring of this kernel has to clear.

## 31. `vd` is memset once because the kernel ACCUMULATES

Lives at `dbscan/gbdt/neighbors/epsilon_neighborhood.mojo`. Their contract
is `cudaMemsetAsync(vd, 0, (m + 1) * sizeof(IdxT))` before
`epsUnexpL2SqNeighborhood`, which adds into `vd` rather than writing it; ours
is `ctx.enqueue_memset` in the same position. Dropping the zero looks fine on
the first batch and corrupts every later one.

## 32. Their device-wide scan is three launches here

`adjgraph/algo.cuh:65` runs `thrust::exclusive_scan` (CUB decoupled
lookback, single pass). One threadgroup cannot do that shape on Metal, and
the first port ran `grid_dim=(1,1,1)` -- one block scanning the whole array
serially, twice per fit. Now a three-launch scan-then-propagate at
`dbscan/gbdt/dbscan/adjgraph/algo.mojo`, verified exact at 2,000,000
entries across 977 blocks (LANE_dbscan-brute D3;
`check_exclusive_scan_beyond_the_old_cap`).

## 33. `make_monotonic`'s unique-value step is replaced

Lives at `dbscan/gbdt/label/classlabels.mojo`, which also records why the
relabel is NOT optional (cuML runs `final_relabel` + `relabelForSkl` on every
fit -- `runner.cuh:412` -- so label VALUES are API, not just the partition).
The header says "do not improve"; read it before touching the file.

## 34. `adj_to_csr`: the warp-aggregated atomic and multi-block rows are unported, priced

Lives at `dbscan/gbdt/dbscan/adjgraph/algo.mojo` (module docstring). The
shared per-row cursor, chunked 16-bool loads and unordered output are theirs;
the warp aggregation of the cursor atomic (needs `coalesced_threads()`) and
the multi-block-per-row grid are not, and the docstring prices both. Label
propagation converges to the same fixed point either way, so this moves a
wait, never an answer.

## 35. DBSCAN defaults to the ball cover; cuML defaults to brute force

Lives in full at `dbscan/gbdt/dbscan/runner.mojo`, above `EPS_NN_BRUTE_FORCE`.
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

Lives in full at `neighbors/gbdt/neighbors/detail/knn_brute_force.mojo`,
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
inputs in `neighbors/gbdt/distance/detail/pairwise_distance_base.mojo`:
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

`compute_batch_size` in `dbscan/gbdt/dbscan/dbscan.mojo` raises on both:
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
are documented on `dbscan_fit` in `dbscan/gbdt/dbscan/runner.mojo`;
`dbscan/phase_main.mojo` is the dedicated main. Every phase already ends on
a `ctx.synchronize()` the port performs anyway, so the flag adds no
synchronization, and off (the default) it prints nothing. Timestamps are
host wall clock, which on one queue with a sync at each boundary is the
phase's device time plus its enqueue overhead -- the same thing their nvtx
range brackets.

## 39. Batch 0's RBC fill runs after the CSR buffer is sized, not inside loop 1

`dbscan/gbdt/dbscan/runner.mojo`. cuML's first batch loop fills batch 0's
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
so `cluster/gbdt/distance/fused_distance_nn/simt_kernel.mojo` runs ONE
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
that is the PRE-existing `updateReducedVal` replacement (`PORTED_MAP.tsv`
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
(`mojo_only/gram_splitk_check.mojo`), not argued: the shipped pipeline
(`column_mean_kernel` -> `shift_columns_kernel(-1)` -> `gemm_tn_splitk`)
against the fused arm on the same device mu and hashed, offset-mean data,
every cell compared with `!=` -- the fused tile load performs the identical
fp32 subtraction the center pass stores (`x + (-1.0) * mu` is bitwise
`x - mu`), and products of bit-identical fp32 inputs in the same order are
bit-identical. The same check asserts X is bit-identical after the fused
call, and `check_covariance_fused_and_fallback_restore`
(`decomposition/mojo_only/pca_check.mojo`) holds the wired arm to it while
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
(`cluster/mojo_only/reduce_by_key.mojo::accumulate_centroid_sums_
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
function of (seed, index). `cluster/mojo_only/scalable_init.mojo::
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

WHAT IS STILL HOST SIDE, and it is one class:
`TWeightedBinFreqCalcer::VisitEqualUpToPriorFreqCtrs`
(`ctrs/ctr_calcers.h:307-341`), which scans the segment ids, reduces per
segment and divides. It needs two device pieces the Borders path does not:
`UpdatePartitionOffsets` (`cuda_util/kernel/partitions.cu:81-107`) and
`SegmentedReduceVector` (a `cub::DeviceSegmentedReduce`). Neither is ported.
`TCtrBinBuilder` (no suffix) stays beside it as the host reference the
device builder is gated against, and `TCtrBinBuilderGpu.read_indices` is the
seam that feeds the host freq calcer a device-produced ordering.

`THistoryBasedCtrCalcer` (no suffix) is likewise kept, as the reference
`mojo_only/ctr_device_check.mojo` compares the device calcer against cell by
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
* `mojo_only/ctr_check.mojo` gating the host arithmetic against planted
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
equally optimal cut; `mojo_only/ctr_check.mojo` names them, still requires
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

REACH IS PROVED, not asserted. `mojo_only/ctr_apply_check.mojo` flips every
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

## 57. ONE CTR estimation permutation, not `permutation_count` of them, and it is never their id 0

`gbdt/data/permutation.mojo` ports `TDataPermutation`
(`cuda/data/permutation.{h,cpp}`) and everything under it -- their
`Shuffle` (`cuda/data/data_utils.h:21-47`), `TRandom`
(`libs/helpers/cpu_random.h`) and MT19937-64
(`util/random/mersenne64.{h,cpp}`) -- bit for bit. `pixi run
check-permutation` compares the raw 64-bit stream, `Uniform(t)`, and whole
orders against CatBoost's own generator compiled by
`tools/permutation_oracle/`. That part is not a deviation.

**Two things around it are.**

### 55a. One column set where they build `permutation_count` of them

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

**`gbdt/methods/doc_parallel_boosting.fit` holds ONE dataset and has no
permutation machinery**, so `train()` builds ONE set of columns, over
`ctr_estimation_permutation_id`, defaulting to `permutation_count - 1` --
their estimation permutation, the one whose model `Run()` exports
(`doc_parallel_boosting.h:526-528`). Porting the other three sets is
porting the ordered-boosting loop, which is a different piece of work.

What this costs, stated rather than hidden: their ordered boosting averages
the ordered-statistic noise over four independent orders and ours does not,
so a `Borders` fit here carries more of it. It is a QUALITY difference on
the same estimator, not a different estimator -- unlike substituting row
order, which is 55b.

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
(`mojo_only/ctr_device_check.mojo` section 3):

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
