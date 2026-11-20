# Optimizer migration source audit — 2026-11-10

Snapshot: `8c5264994b`, compared with `e8aaef8d52^`; producer range
`e8aaef8d52` through `dbb3631cea`. Independent source inspection only: no
compile, execution, timing, vendor identity, or quality claim. Later producer
`bee32a8a30` is outside this frozen snapshot.

## Findings requiring action

1. **Constructor refusal order changed.**
   `bindings/optimizer_native_router.mojo:99–117` converts beta/lr/other scalar
   values before `OptimizerConstructorConfig.validate()` checks accumulation.
   Previously `optimizer_construct` checked the accumulation power-of-two
   requirement first, then maximize type and SGD Nesterov constraints, then
   betas (`e8aaef8d52^:bindings/optimizer_native_router.mojo:140–148`). For valid
   parameter buffers, `accumulation_steps=3, lr='bad'` formerly refused the
   accumulation setting; the new path reaches float conversion first. Invalid
   accumulation plus malformed Adam betas also reverses the first refusal.
   SGD Nesterov versus lr conversion ordering needs the same repair. This is
   an API error-order defect, not a changed optimizer formula. Concrete repair
   handed to `boundary_pilot`; root resumed that producer for the fix.

2. **New dispatch allocation cost remains unmeasured.**
   `training/optimizer_native_api.mojo:130–154` creates a one-item request List
   and one-item address List per step/io/clip/accumulate call.
   `core/native_descriptor_call.mojo:16–24` adds its required stable heap output
   owner. These replace Python envelopes and calls, but are still allocations;
   source inspection does not establish a net speedup. The existing supported
   span ABI could co-locate request/address/output in one owned call frame in a
   later measured optimization. No stack-address workaround is recommended:
   the shared helper records a real toolchain output-lifetime limitation.

3. **Strided preparation is a different copy implementation, not a measured
   improvement.** `training/optimizer_input_plan.mojo:96–142` uses the existing
   host worker policy and disjoint output elements; each element computes its
   coordinates with rank-many division/remainder operations. The prior
   `bindings/optimizer_public_door.mojo:429–466` used `PyBuffer_ToContiguous`.
   Both allocate one scratch buffer, zero-initialize it, then fill it once.
   Contiguous inputs return their original address in both versions. The new
   code removes a second buffer export and parallelizes large copies, but
   loses any contiguous-run optimization inside CPython's copy routine. Its
   end-to-end cost remains to be measured on the affected GPU operations; no
   claim of faster preparation or a confirmed regression follows from this
   review.

## Algorithm and kernel trace

- New host/GPU wrappers call the same `public_step`, `public_io`, `public_clip`,
  `public_accumulate`, resident upload/download/open/close entries:
  `training/optimizer_kernels_gpu.mojo:17–64`,
  `training/host/optimizer_kernels.mojo:17–52`, and
  `training/optimizer_native_api.mojo:80–117`. The neural-context key and
  numeric-mode selection match the old GPU wrapper.
- `_hyper` keeps the same SGD/Adam/AdamW values and conversions
  (`bindings/optimizer_public_door.mojo:473–487`, old:699–713). The step still
  passes tensor sizes/offsets, accumulation count, token divisor, schedule,
  maximize, clipping threshold, resident policy and device flags in PublicStep
  (`bindings/optimizer_public_door.mojo:606–709`). No replacement update,
  reduction tree, tuned switch, or generic fallback was found in this trace.
- Fourteen downstream source files are byte-unchanged, including public
  host/GPU execution, route/device plan, optimizer/clip/accumulation kernels,
  estimator, host math implementation, schedule and state-cast implementation.
  The exact before/after blob IDs and retained source bytes are in the evidence
  below. Unchanged source is not runtime correctness or identity evidence.

## Loops, copies and ownership

- Container, parameter-buffer, shape, keyword and result-owner loops in
  `optimizer_container_boundary.mojo:34–75` and the door are UNPACK/PACK
  metadata loops. They do not update individual tensor elements.
- The new strided gather's element loop is data work and is explicitly split
  among host workers (`optimizer_input_plan.mojo:121–139`). Inner shape-axis
  traversal computes one element's address. Numeric update/reduction work
  remains in the unchanged kernel entry points.
- `OptimizerBufferLease` keeps the original Py_buffer exporter reference until
  the native call returns (`optimizer_container_boundary.mojo:11–31`;
  `optimizer_public_door.mojo:247–309`). Array slots/memoryview and caller owners
  are also pinned. GIL release encloses native step/clip/accumulate and GPU
  transfers; pins are consumed only after those scopes end (door:625,707–711,
  768–771,824–827,843–846,977–981,1012–1030). No new Python destructor or
  callback inside these released scopes was found.
- Checkpoint conversion at this snapshot still uses the existing
  `optimizer_state_cast` once into its final owned destination
  (`optimizer_native_router.mojo:170–190`); its serial implementation and held
  GIL are pre-existing, not a newly introduced shortcut. Producer `bee32a8a30`
  reportedly changes this later and needs its own review/evidence.
- Shared output allocation replaces the old zeroed Python bytes temporary plus
  array copy (`old optimizer_public_door.mojo:220–241`) with
  `admitted_storage_native`/`native_array_zeros` (new door:154–156). Output
  ownership and initialization remain explicit; this is not an uninitialized
  or aliased-state shortcut.

## Removed mechanisms and remaining boundary state

Removed: internal Python configuration dictionary/constructor tuple staging,
Python tensor-list materialization between compiled doors, Python scalar
constructor calls where fixed scalar C APIs now suffice, duplicated output
shape/stride packing, and redundant strided-input re-export. Public parameter,
schedule and output objects still exist at UNPACK/PACK; the saved instance
mapping in `set_params` remains a rollback owner, not algorithm state. The old
code already avoided a temporary optimizer instance, so this migration must
not claim that removal as a new benefit.

Evidence: `~/mojolearn-evidence/python-mojo-migration-20261110/optimizer-audit/`:
`before/`, `after/`, `migration.diff`, `kernels.diff`,
`kernel-source-comparison.json` (14/14 downstream files unchanged).
All evidence is source material; runtime and performance qualification remain
with the orchestrator.
