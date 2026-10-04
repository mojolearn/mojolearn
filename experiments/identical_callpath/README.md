# IDENTICAL shared callpath experiments (source only)

Requested as a separate Git worktree from the Apple FAST/GEMM/Kalman work.
The corrected scope is shared call overhead for **all algorithms and all GPU
vendors**, rather than an Apple-only or millisecond-row-only facility. The
branch is `experiment/identical-callpath-all-algorithms-20261004`; the source
API has no vendor restriction. CPU-only algorithms retain their existing host execution path.
Baseline: `c96137714`. This lane does not change production dispatch, kernels,
numeric helpers, or default behavior. Nothing here has been compiled, executed,
benchmarked, timed, or checked for bitwise equality. These are candidates,
not a verified identity or performance improvement.

The hypothesis is that persistent buffers and fewer host completion boundaries
can reduce fixed per-call costs without changing arithmetic. This can preserve
bitwise results **if** inputs, numerical mode, kernels, launch geometries,
initialization, dependencies, and model state remain identical. Changing a
floating-point reduction tree, RNG ordering, or a GEMM implementation is outside
this lane. Unchanged source is necessary evidence, not a compiled-code proof.

## Implemented variants

| Candidate | Entry | Intended difference from baseline |
| --- | --- | --- |
| I1: persistent single call | `ResidentIdenticalMinMax` or `ResidentIdenticalStandard`, one slot | Reuse device input/output, host staging, context and fitted model buffers; caller supplies result storage. |
| I2: independent batch with per-item waits | `transform_batch_into(..., group_waits=False)` | Reuse storage across multiple calls while retaining completion after each item; isolate allocation effects from wait grouping. |
| I3: independent batch with grouped wait | `transform_batch_into(..., group_waits=True)` | Same per-item uploads, kernels, and downloads in the same order; one completion boundary at batch end. |

I2 currently includes a redundant final completion call through `finish()`.
Keep that explicit when comparing the candidate variants in future authorized
work; it is a conservative lifetime boundary, not an asserted speed win.

`core/identical_callpath.mojo` implements `IdenticalCallSession`; `session.mojo`
re-exports it for the experiments. It is the algorithm-neutral typed workspace for
heterogeneous operations, independent input/output sizes and typed scratch.
It is the broad integration surface: an algorithm supplies its existing
unchanged enqueue sequence and explicitly identifies completion boundaries.
It is not an automatic wrapper that removes allocations hidden inside an
arbitrary estimator. The API being general does not establish that every
algorithm has been adapted, or that every vendor has passed identity checks.

The session has independent banks for Float16, BFloat16, Float32, Float64,
Int8/16/32/64 and UInt8/16/32/64. Banks allocate only when a slot is reserved;
a vendor need not support arithmetic on every storage type. Each slot has its
own logical extent, device allocation, and separate upload/readback mirrors.
Models, outputs, masks, sparse offsets, indices and scratch can coexist without
casting integer bits through floating point. Slots never alias each other.
Per-slot mirrors deliberately favor straightforward ownership over minimum
setup memory; device-only scratch allocation is a later refinement.

The universal sequence is reserve typed slots, stage host inputs, `begin`,
upload those inputs, enqueue the algorithm's existing operations, request
readbacks, `finish`, then collect into caller-owned lists. For example,
`reserve_f32`, `stage_f32`, `upload_f32`, `readback_f32` and `collect_f32` address
`session.f32.device[slot]`; the other banks use the corresponding suffix.
No input/output shape equality is required. Model slots can remain resident,
and successive algorithm stages can consume device results without readback.
Staging and reservation require an idle session. A slot permits one upload
and one readback per batch; use another slot for an independent call. `begin`
invalidates earlier readback readiness. Logical empty slots skip transfers.

Existing kernels receive `session.ctx` and the typed device buffers. The session
does not clear scratch: preserve every original initialization/zeroing operation
and do not assume recycled memory starts at zero. Preserve RNG states and model
versions explicitly. Refit or shape changes require new appropriately sized
slots while idle. No global cache, automatic eviction, graph capture, kernel
fusion, or algorithm dispatch replacement is introduced.

Always `finish` and collect before a required host decision or validation, then
`begin` the next phase. For multiple GPUs, use one session per existing device
selection (`IdenticalCallSession(device_id=rank)`, or the existing default when
omitted) and retain the original communication/collective order; this API does
not create a cross-device pool or change device selection. Host-only algorithms
do not benefit from GPU transfer batching and retain their original path.

If an externally enqueued operation raises, call `abort` before leaving its
scope. A failed submission, allocation or wait poisons the session; it cannot
be reused. Device-loss and partially constructed teardown remain unverified.
Slots and context are owned together, but internal fields are not an opaque
API: bypassing the methods can violate these contracts.

`families.mojo` adds unchanged-operation adapters for row norms, column means,
column shifts, core NT GEMM and the explicit identical-GEMM workspace entry.
These cover reusable stages of neighbors, clustering, linear models,
decomposition and solvers. They do not replace each family's own dispatch:
only use an adapter where the original path already called that primitive.
See [the family coverage map](COVERAGE.md) for implemented primitives,
remaining estimator wiring, and identity obligations across the repository.

`storage.mojo` is the fixed-shape Float32 allocation/staging scaffold. It owns the context,
so adapters cannot accidentally pass another queue to a pool operation.
`minmax.mojo` supplies a real consumer already present in this baseline:
`preprocessing.minmax.minmax_transform_into`. That production function still
launches `minmax_transform_kernel` with block size 256 and the unchanged grid,
arguments, Float32 seams, and inverse/clip behavior. No replacement kernel or
copy kernel is introduced. MinMax is a plumbing pilot, not MaxAbs.
`standard.mojo` additionally calls
`preprocessing.standard.standard_transform_into` with its existing 256-thread
geometry and arithmetic. Both-disabled flags retain the production kernel's
exact-copy behavior, including subnormal bits. Neither adapter refits models
or changes statistics reductions. These two concrete adapters use the small
fixed-shape scaffold; additional consumers can use the typed session directly.

Instantiation requires both compile definitions
`MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH` and `MOJOLEARN_NUMERIC_IDENTICAL`.
No production module imports the experiment. There is no target/vendor gate:
the same host-side design is eligible on Apple, NVIDIA and AMD where the
existing operation and DeviceContext backend are supported. Later checks must
record the selected device. Merely setting the opt-in does not route a
production API into it. Existing unsupported-operation refusals still apply.

## Ownership and semantic boundaries

Construct once with positive fixed rows, columns and slot capacity, and a
validated fitted scale/offset or mean/scale snapshot. Preallocate one result list per input.
Repeatedly call the synchronous batch method with one through `slots` inputs.
Refit or shape changes require a new session. Empty inputs are refused rather
than silently changing baseline empty-shape behavior.

Each slot owns separate input, output, upload staging and readback staging.
All staging is filled before submission, retained through completion, and
never reused while work is pending. All result sizes are validated first.
The host output lists are written only after successful batch completion.
The adapter privately uploads model copies during construction. Exceptions
poison the session, attempt a drain, and prevent subsequent reuse. Destructors
attempt to drain and release buffers before the context, following existing
repository patterns. A failed device synchronization is not an identity or
resource-safety guarantee; device-loss behavior still needs runtime review.
Constructor allocation failures and failed drains are not certified recovery
paths; partial construction and device loss need dedicated lifecycle review.

Use one session from one host thread at a time. Direct mutation of internal
storage/model fields is unsupported. The source scaffold is not an opaque
production API and has not had its Mojo borrow/lifetime rules compiler-checked.
Do not remove baseline finite-input, positive-scale, feature-range or output
validation in a binding that adopts this adapter. As with the production
low-level into function, this adapter is below those validators.

Grouped calls must be independent. There can be no host decision, model
mutation, intermediate validation or callback whose timing controls a later
call. The adapter deliberately returns the whole batch synchronously; it does
not preserve per-call early result/error observability. Baseline individual
calls remain appropriate when those boundaries are semantically meaningful.

## Extension candidates, not implemented adapters

| Algorithm family | Shared mechanism to reuse | Additional identity obligation |
| --- | --- | --- |
| GEMM, LU, Cholesky, PCA, kernel PCA, randomized SVD, LLE, MinCovDet, SVGP | Typed resident input/output and scratch; retain intermediate results on device | Keep original GEMM/solve dispatch, reduction order, pivots, numerical seams and initialization. A faster MMA kernel is a separate experiment. |
| Trees, forests, boosting, classification and regression | Resident models, bin/index buffers, outputs and batch completion | Keep tie policies, integer accumulation, histogram initialization, traversal and RNG streams. |
| Clustering, neighbors, manifold and sparse algorithms | Resident models/indices, typed scratch, independent readbacks | Keep index ordering, atomics policy, convergence checks, sparse layout and all host decisions. |
| Time series, Kalman, ARIMA and iterative optimizers | Reuse buffers between unchanged sequential steps | Preserve recurrence association, search order, convergence/host barriers; associative scan changes arithmetic and is outside this experiment. |
| VAR | Resident input/output and workspace, deferred independent readback | Keep every solve/GEMM dispatch, reduction order and autoregressive dependency. |
| knn-imputer | Resident model, masks and workspace | Preserve tie ordering, missing-value behavior, index initialization and donor selection. |
| random projections | Resident fitted projection matrix and output | Preserve exact matrix bits, sparse indices, RNG consumption and matmul dispatch. |
| additive-chi2 taxi | Resident input/output and host staging | Keep all transcendental/seam calls, feature ordering and validation. |
| maxabs-scaler | Resident fitted scales and buffers | Preserve extrema signed-zero/NaN rules and exact transform operation order. |

This is an integration map across the library, not a claim that every row is
wired. The listed row-specific implementations are not all present at this
baseline. No broad import from newer worktrees was made. Each integration needs
an explicit adapter using its existing kernels, including initialization of
integer/mask/scratch storage where required. The two fixed-shape Float32
adapters alone do not cover these families. The generic session supplies
storage and lifecycle mechanisms rather than algorithm arithmetic.

## Gates for later, separately authorized work

Do not execute these as part of this source-only task.

1. Compile the isolated candidate in IDENTICAL mode; confirm default/FAST
   refusal and inspect generated dispatch/geometry before any performance work.
2. Compare raw UInt32 result bits against repeated production MinMax and
   StandardScaler calls,
   including forward/inverse, clipping, signed zero, subnormals, near-constant
   fitted scales, finite extrema, and partial 256-thread tails. Retain public
   validators and their rejected-input behavior. For StandardScaler include
   all mean/std flag combinations, especially the exact-copy path. Every added
   consumer needs corresponding raw-bit and intermediate identity-card gates.
3. Repeat the same input through reused dirty slots, all slot occupancies,
   both wait modes, fresh sessions, and caller model/input mutations after
   construction. Verify no stale output, alias, or queue-lifetime effect.
4. Exercise wrong shapes, invalid flags, capacity overflow, concurrent misuse,
   submission failure, failed drain, and destruction. Confirm error paths do
   not return partially updated results or permit reuse of poisoned storage.
5. Compare each supported vendor to its unchanged baseline and compare
   cross-vendor IDENTICAL identity cards. Check Apple, NVIDIA and AMD separately
   wherever the operation is supported. One vendor's equality is insufficient
   for a cross-vendor claim; unsupported baseline cases remain refused.
6. Only after correctness approval, measure setup separately from warmed
   calls, and distinguish reuse from batching and result-materialization costs.
   Retain the baseline unless an actual bit-identical benefit is demonstrated.

## Integration compile and GPU gates (2026-10-04)

The later integration task authorizes compilation and NVIDIA/AMD runtime
validation; the source-only restrictions above describe the original task.
`compile_probe.mojo` instantiates every operation in all twelve typed banks,
the five primitive adapters, both scaler adapters and both wait modes. Build
it with `mojo build -j 1 -I . --target-accelerator sm_89` (NVIDIA) or `gfx942`
(AMD), the matching `MOJOLEARN_COLUMN_NVIDIA`/`MOJOLEARN_COLUMN_AMD` define,
plus `MOJOLEARN_NUMERIC_IDENTICAL` and
`MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH`. Always use the repository's compile
semaphore and selected pixi environment. This probe is compile-only.

`identity_gate.mojo` is for execution on authorized NVIDIA/AMD boxes only. It
compares raw bits against the unchanged production transform entry points:
48 scaler comparisons cover forward/inverse, clipping, all StandardScaler
flag combinations, both wait modes, reused dirty slots, partial blocks,
signed zero and subnormal inputs. It also verifies two exact UInt32 buffer
roundtrips and refusal to collect an earlier readback during a new batch.
Success prints `CALLPATH_GATE status=PASS comparisons=48` and a digest for
cross-vendor comparison. This is a correctness gate, not a timing harness.
It does not establish primitive-adapter runtime correctness, public validation
parity, device-loss recovery, or Apple/host identity.

The GEMM adapters require distinct left, right, output and workspace slots.
Bounds and distinctness checks precede untracked mutable handle borrows;
no slot allocation or list mutation occurs during those calls. A caller
needing the same matrix as both operands must reserve separate operand slots.
Kernel pointers use explicit `MutAnyOrigin` casts while storage remains owned
by the active session. These changes accommodate the current Mojo ownership
rules without changing kernels or dispatch.
