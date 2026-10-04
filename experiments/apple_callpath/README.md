# Shared low-overhead call-path experiments

Status: **NOT COMPILED, NOT EXECUTED, NOT MEASURED.** Source proposals only,
as requested. No speedup or bitwise equivalence has been established.
Nothing imports this directory from a production dispatch path. All variants
require explicit imports and adapter calls; there is no default behavior change.
Baseline: this worktree starts at `c96137714`. Later Apple row-specific worktrees
may contain additional consumers; no commits from those lanes were imported.

The current shared `metrics/checks/device_io.mojo` helpers allocate a device
buffer and copy a host List before each upload, then synchronize. Downloads
allocate a host buffer and synchronize. These experiments target that plumbing
without changing arithmetic or launching replacement math kernels.

| Lane | Concrete implementation | Intended effect | Activation |
| --- | --- | --- | --- |
| C1 | `ResidentCallSlot` in `resident_slot.mojo` | Reuse device input/output, float and integer scratch, pinned upload and readback storage for fixed shapes | Construct once, `stage`, `begin`, enqueue existing kernels, `seal`, `wait`, `collect_into` |
| C2 | `wait_pair` in `resident_slot.mojo` | Share one host wait across two independent calls while keeping both copies alive | Stage and seal two distinct slots on one context, `wait_pair`, collect each |
| C3 | `PackedReadback` in `packed_readback.mojo` | Queue multiple existing result buffers into one retained pinned slab and wait once | `append` each result, retain returned offsets, `finish`, `collect_into`, `reset` |
| C4 | `enqueue_minmax_transform` in `minmax_adapter.mojo` | Example adapter with resident model buffers and unchanged scaler kernel/geometry | Explicit import; supply resident scale/offset and staged slot, then wait/collect |

C4 uses MinMax as a concrete existing-kernel wiring example; it does not
implement MaxAbs or silently substitute one scaler for another. The requested
VAR, KNN-imputer, random projections, additive-chi2 and MaxAbs names were not
found as named Mojo/Python implementations in this checkout. Their adapters
remain future wiring work. C1's integer scratch supports index-heavy clients
without forcing indices through Float32. C3 still submits one copy per result;
it experiments with shared storage and completion, not a combined DMA command.
C2 does not imply overlapping GPU execution on Metal.

## Lifetimes and usage contract

- Use the **same in-order DeviceContext** for construction, all kernels/copies,
  completion and cleanup. This fork verifies the original context address.
- Callers retain the slot, source/model buffers, readback object and context
  through completion. No thread sharing, cross-context sharing or nested reuse.
- Fixed capacities are deliberate: shape changes require a drained slot and a
  new allocation. Nothing grows or evicts a live device allocation.
- Staging is forbidden while in flight. `begin` requires staged input and
  claims the slot before enqueueing; errors leave it occupied for cleanup.
- Kernels must fully initialize every output element read back, and every
  scratch element they read. Reused scratch is not implicitly zeroed. Preserve
  any existing initialization kernels and launch dependencies in client adapters.
- `collect_into` requires a caller-owned List with the exact output length;
  it avoids a fresh result allocation, but still copies to that List.
- For exceptions after `begin`, call `slot.drain(ctx)` before destruction or
  reuse, then rethrow. For readback exceptions, call `readback.reset(ctx)`.
  If synchronization itself fails, these proposals provide no device recovery.
- The original `mark_completed` escape hatch is removed in this fork;
  `wait` and `wait_pair` advance readability after successful synchronization.
- A borrowed output or model buffer must not be overwritten by another queue
  before its queued copy/kernel consumes it. Packed readback keeps sub-buffer
  aliases alive until `reset`; callers also retain the source buffer.
- Destroy completed slots/readbacks before destroying the context, then drain
  the context after releasing buffers, matching existing repository allocator
  teardown conventions. No automatic context-owning destructor is introduced.

`PackedReadback` currently retains a List of sub-buffer aliases, so it does
not claim zero host allocations. C1 is the tighter warmed fixed-shape candidate.
The copies and unchanged scaler kernel are intended to preserve output bits,
and this fork adds an explicit FAST+Apple opt-in guard. Runtime validation
has not yet been performed. A separately guarded identical-mode lane can build on
these ownership observations without relying on an equivalence claim here.

## Deferred validation, not performed

Before enabling any consumer, compile it and compare raw output bits against
the original path using the same numeric mode; cover signed zero, subnormals,
tails, empty inputs, repeated calls with different values, output/scratch
initialization, exception cleanup and forbidden slot reuse. Independently
audit host visibility and allocator teardown. Only after separately authorized
correctness work should cold/warm standalone call latency be measured.

## Manager integration fork (2026-10-04)

Source copied from pinned `9ab2d3d3fb770498ef025db08f595a0149792bb7` into a
current-main experiment branch. The original worktree/catalog is untouched.
This fork requires `MOJOLEARN_APPLE_FAST_CALLPATH_CANDIDATES`, FAST mode and
Apple at construction. Every context-taking method checks the address of the
original borrowed context: retain that same context at a stable address until
all slots/readbacks are destroyed. Moving/copying a context is unsupported and
rejected. The public `mark_completed` escape hatch was removed. Slots still
require explicit draining on submission failures; these types do not own the
context or promise device recovery after a failed synchronization.

`bench/apple_callpath_quality.mojo` instantiates C1/C2/C3/C4. It checks raw
transfer words (signed zeros, subnormals, quiet NaN payload, infinities), 0/1/257/
4099-element shapes, three changing-input reuse cycles, ordinary result Lists,
float scratch, Int32 scratch beyond Float32 integer precision, foreign-context
rejection, occupied reuse, premature reads, capacity overflow, reset visibility
and drain/reuse. C4 compares the unchanged MinMax kernel/geometry against its
ordinary upload/download path for empty and 777-element inputs, forward,
clipped and inverse transforms. No speed or runtime pass is yet claimed.

M2 compile only:

```
mojo build -I . -D MOJOLEARN_APPLE_FAST_CALLPATH_CANDIDATES bench/apple_callpath_quality.mojo -o apple-callpath-quality
```

The manager stages this native executable with its source SHA and SHA256 to
M3. `tools/callpath_quality_run.py SOURCE BINARY_SHA256 BINARY_PATH` verifies
both, then executes the fixture. No custom Python binding, installed A.so,
M3 build, fit, opponent run, or timing is involved.

Later timing must cover stage + enqueue + wait + collect + full first read,
with separate cold/warm one-call scenarios (one scored run each). C2 requires
an equivalent two-call A/B contract. C3 submits one copy per result. Pinned
reads occur inside collection and must never be omitted from timed work.
C4 remains a wiring fixture: it makes no claim about still-slow algorithms.
