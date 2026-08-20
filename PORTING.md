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

> **NUMBERING GAP, STATED RATHER THAN HIDDEN.** Entries 27 through 34 are
> referenced from code (`deviation 30`, `31`, `33`, `34`) and were written up
> in the lane files that produced them -- `LANE_dbscan-brute`,
> `LANE_vendor-correctness`, `LANE_rbc-build` -- but have not been folded into
> this file yet. The two below are here because nothing else records them.

## 35. DBSCAN defaults to the ball cover; cuML defaults to brute force

Lives in full at `dbscan/ported/dbscan/runner.mojo`, above `EPS_NN_BRUTE_FORCE`.
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

## 36. k-NN defaults to the TILED arm; cuVS defaults to the FUSED one

Lives in full at `neighbors/ported/neighbors/detail/knn_brute_force.mojo`,
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
At `gridDim.x == 1` -- their cross-block merge protocol is a mutex plus
`__threadfence` and is unported -- the block count is `ceil(n_queries / Mblk)`
and does not depend on the index size at all. Holding the index at 200,000 and
raising the query count shrinks the deficit monotonically (0.66x at 500
queries, 0.86x at 2,000, 0.92x at 8,000, 0.93x at 32,000) and never crosses.

**Rule: a traffic model that does not count blocks can predict the wrong
SIGN, not merely the wrong magnitude.** Porting their `gridDim.x > 1` split
is what would reverse this, and the arm stays reachable through `knn_method`
so that re-measuring it then costs one argument rather than a revert.

## 37. A tiny DBSCAN batch budget raises here; theirs wraps `size_t` into a full batch

`dbscan.cuh:66` computes `max_mbytes_per_batch * 1000000 - est_mem_fixed` in
`size_t`. Hand it a nonzero budget smaller than the fixed cost and the
subtraction WRAPS to ~2^64; the `std::min` at `:69` then quietly turns the
wrap into `batch_size = n_owned_rows` -- the tiniest budget buys the LARGEST
batch. And a budget between the fixed cost and one row's cost yields
`batch_size = 0`, which reaches `raft::ceildiv` at `runner.cuh:131` and
divides by zero. Neither behavior is a design; both are what unsigned
arithmetic does when nobody expected the input.

`compute_batch_size` in `dbscan/ported/dbscan/dbscan.mojo` raises on both:
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
are documented on `dbscan_fit` in `dbscan/ported/dbscan/runner.mojo`;
`dbscan/phase_main.mojo` is the dedicated main. Every phase already ends on
a `ctx.synchronize()` the port performs anyway, so the flag adds no
synchronization, and off (the default) it prints nothing. Timestamps are
host wall clock, which on one queue with a sync at each boundary is the
phase's device time plus its enqueue overhead -- the same thing their nvtx
range brackets.

## 39. Batch 0's RBC fill runs after the CSR buffer is sized, not inside loop 1

`dbscan/ported/dbscan/runner.mojo`. cuML's first batch loop fills batch 0's
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
