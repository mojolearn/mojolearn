# IDENTICAL shared callpath experiments (source only)

Requested as a separate Git worktree from the Apple FAST/GEMM/Kalman work.
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
| I1: persistent single call | `ResidentIdenticalMinMax`, one slot | Reuse device input/output, host staging, context and fitted model buffers; caller supplies result storage. |
| I2: independent batch with per-item waits | `transform_batch_into(..., group_waits=False)` | Reuse storage across multiple calls while retaining completion after each item; isolate allocation effects from wait grouping. |
| I3: independent batch with grouped wait | `transform_batch_into(..., group_waits=True)` | Same per-item uploads, kernels, and downloads in the same order; one completion boundary at batch end. |

I2 currently includes a redundant final completion call through `finish()`.
Keep that explicit when comparing the candidate variants in future authorized
work; it is a conservative lifetime boundary, not an asserted speed win.

`storage.mojo` is the shared allocation/staging scaffold. It owns the context,
so adapters cannot accidentally pass another queue to a pool operation.
`minmax.mojo` supplies a real consumer already present in this baseline:
`preprocessing.minmax.minmax_transform_into`. That production function still
launches `minmax_transform_kernel` with block size 256 and the unchanged grid,
arguments, Float32 seams, and inverse/clip behavior. No replacement kernel or
copy kernel is introduced. MinMax is a plumbing pilot, not MaxAbs.

Instantiation requires both compile definitions
`MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH` and `MOJOLEARN_NUMERIC_IDENTICAL`.
No production module imports the experiment. Construction also requires an
Apple GPU target through `has_apple_gpu_accelerator()`. Later checks must record
the actual selected device. Merely setting the opt-in does not route a
production API into it.

## Ownership and semantic boundaries

Construct once with positive fixed rows, columns and slot capacity, and a
validated fitted scale/offset snapshot. Preallocate one result list per input.
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

| Requested row | Shared mechanism to reuse | Additional identity obligation |
| --- | --- | --- |
| VAR | Resident input/output and workspace, deferred independent readback | Keep every solve/GEMM dispatch, reduction order and autoregressive dependency. |
| knn-imputer | Resident model, masks and workspace | Preserve tie ordering, missing-value behavior, index initialization and donor selection. |
| random projections | Resident fitted projection matrix and output | Preserve exact matrix bits, sparse indices, RNG consumption and matmul dispatch. |
| additive-chi2 taxi | Resident input/output and host staging | Keep all transcendental/seam calls, feature ordering and validation. |
| maxabs-scaler | Resident fitted scales and buffers | Preserve extrema signed-zero/NaN rules and exact transform operation order. |

Those row-specific implementations are not all present at this baseline.
No broad import from newer worktrees was made. They need explicit adapters
using their existing kernels, including persistent integer/mask/scratch storage
where required; this Float32 pilot alone does not cover them.

## Gates for later, separately authorized work

Do not execute these as part of this source-only task.

1. Compile the isolated candidate in IDENTICAL mode; confirm default/FAST
   refusal and inspect generated dispatch/geometry before any performance work.
2. Compare raw UInt32 result bits against repeated production MinMax calls,
   including forward/inverse, clipping, signed zero, subnormals, near-constant
   fitted scales, finite extrema, and partial 256-thread tails. Retain public
   validators and their rejected-input behavior.
3. Repeat the same input through reused dirty slots, all slot occupancies,
   both wait modes, fresh sessions, and caller model/input mutations after
   construction. Verify no stale output, alias, or queue-lifetime effect.
4. Exercise wrong shapes, invalid flags, capacity overflow, concurrent misuse,
   submission failure, failed drain, and destruction. Confirm error paths do
   not return partially updated results or permit reuse of poisoned storage.
5. Compare Apple results to existing cross-vendor IDENTICAL identity cards.
   Any future widening of the Apple guard requires checking each supported
   vendor; Apple-local equality is insufficient for that wider claim.
6. Only after correctness approval, measure setup separately from warmed
   calls, and distinguish reuse from batching and result-materialization costs.
   Retain the baseline unless an actual bit-identical benefit is demonstrated.
