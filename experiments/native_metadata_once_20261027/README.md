# Native metadata admission and reuse — 2026-10-27

This source repair addresses repeated stable metadata reads in tree/meta-estimator
orchestration. It is not a claim that the entire library now admits every input
only once, that CPython dispatch is gone, or that performance improved. Compiled
public runtime, within-release identity coverage and performance remain pending.

## Implemented scope

`TreesView` already carries native address, shape, strides, dtype and item size,
plus the retained storage owner. `TreesMatrix` already carries the float32 matrix
owner, address, row/column counts and layout. The repair reuses those types;
there is no global pointer cache, generic wrapper framework or new data copy.

- Stacking fit snapshots admitted X/y metadata and the optional weight address
  per invocation, binary/passthrough settings per invocation, and native method
  names/output widths before fold execution. Each fold snapshots its two index
  arrays once. Its member loop reuses those native values, including destination
  metadata. Stacking transform does the same for its input/destination/row map.
- Shared gather/scatter helpers consume retained native descriptors. Existing
  call sites without prepared descriptors snapshot trusted internal Arrays at
  the helper boundary. They no longer invoke Python `Array.__len__`.
- Probability column accumulation reads its column map once. Multiclass
  one-vs-rest captures the class-code descriptor before the class loop. Split
  count validation reads Array metadata without the Python length callback.
- Array/buffer admission captures the shape and stride tuples once before
  traversing rank metadata.

Kernel calls, data gathering, fit order, fold order, random settings, numerical
kernels and public signatures remain unchanged. A newly produced member output
still needs its own admission. Member fit/predict dispatch still crosses the
existing compiled estimator ABI and may admit the same input again; eliminating
that repeated admission is a separate integration step, not already accomplished
by the outer-loop snapshot.

## What is and is not a Python callback

`len(Array)` reaches the Python implementation documented in
`python/mojolearn/_array.py` and is an actual callback. Reading ordinary Array
slots or an exact memoryview's native metadata, and indexing exact Python tuples,
uses native CPython object machinery. It is still boundary overhead, but not by
itself evidence of Python bytecode. Calls through `py_call_typed` and
`prep_entry_call` must be classified by their actual target: the connected kernel
and estimator entries are compiled extensions. A native entry can still call
Python transitively, so public runtime audit remains necessary. Custom descriptor
or protocol hooks cannot be reclassified as metadata and silently exempted.

The existing documented Array length fallback registry is retained: other paths
still contain callbacks, and no fresh binary has yet proved removal on the
changed public operations. `inventory.json` records repaired and remaining sites.

## Refresh, mutation and lifetime contract

External X/y/weights are admitted anew for each fit/transform/predict call. These
snapshots never survive the call. Refit, `set_params`, replaced estimators and new
buffers therefore cannot reuse old shapes, dtypes, addresses or options. Fold
indices are private freshly generated/validated Arrays and are stable only
within that fold. The local owners and `TreesView.keep` retain backing storage;
prepared raw-address helpers retain their descriptors through the kernel call.
The source address of a float vector remains the caller's responsibility, as in
the existing ABI; stacking keeps its X/y/weight owners alive past those calls.

This does not freeze data values. Caller mutation between operations is observed
on the next operation. A provider must prevent resize/replacement while a borrowed
buffer is executing, or supply an owned snapshot through its existing admission
rules. Current ordinary Array storage has a stable address; general buffer
admission retains its memoryview export. Dynamic token counts, changing batch
views, outputs and optimizer state must be updated natively, not cached as
immutable metadata.

## Consolidating the provider boundary

The useful common contract is an admitted tensor/model description with explicit
ownership, not a directory containing unrelated Python calls. Existing pieces:

| Provider/path | Existing representation | Ownership and limitation |
| --- | --- | --- |
| Tree/meta input | `TreesView`, `TreesMatrix`, `TreesLabels` | retained Array/memoryview; rank/stride/dtype metadata native |
| Prep and model selection | `x_prep_native_inputs.mojo`, `msel_inputs.mojo`, arena row snapshots | native admission exists; several metadata dictionaries and re-admissions remain |
| Python sequential input | `sequential_inputs.mojo`, native buffer/storage helpers | preserves contiguous/strided and dtype behavior; some descriptors remain Python objects |
| Torch/MAX whole-model adapter | `adapters/native/abi.h` v1 tensor descriptors and whole-model program/options | caller retains storage through synchronous return; limited rank/kinds/layout/device contract |

A staged consolidation should:

1. Define the ownership-neutral tensor fields in a native boundary module:
   address, rank/extents, byte strides, dtype, device/address space, mutability
   and readiness. Keep provider-owned references/exports opaque and outside
   arithmetic. Reuse the established descriptors first; do not silently widen
   adapter ABI v1 or narrow Python inputs to its contiguous subset.
2. Give each supported provider an admission function returning those fields
   plus an ownership lease. Python admission retains exact admitted native
   buffers; Torch/MAX retain their framework owners and enforce their existing
   synchronization/device contract. Errors and unsupported protocols remain at
   admission. No global object-ID cache, unconditional host copy, or callbacks
   from the arithmetic path.
3. Add descriptor-taking *internal* estimator operations so a parent admitted
   X can flow into native members without another Python-shaped admission.
   Preserve existing public one-call wrappers and the whole-model-only adapter
   surface. Connect this with the separate native-dispatch lane rather than
   introducing a second member dispatch registry.
4. Persist native-owned model topology/options/weights with a model-state
   generation. Constructors, parameter setters, load/import and refit must
   explicitly rebuild/invalidate dependent state. Caller-owned mutable weights
   cannot be called immutable: either retain a checked versioned provider lease
   or admit anew. A file/folder move does not establish this contract.
5. Prove each migration on the actual public operation: callback counters and
   refusal controls, data/layout refresh, owner release/lifetime, changed model
   settings, output correctness and applicable device identity. Only then
   remove its fallback/debt record. Measure end-to-end full-workload A/B before
   any speedup/default claim; a metadata-only microbenchmark is insufficient.

This lane implements the tree/meta subset of steps 1–2 through existing records.
It does not add a universal ABI or persistent model cache. Generation execution
and direct dispatch are assigned to the other lane; its dynamic state must not
be frozen by this metadata design.

## Validation and integration

`tools/test_native_metadata_snapshot.py` guards the demonstrated source-level
regressions (five tests). `test_native_meta_metadata_snapshot.py` adds five public
runtime cases: strict native audit and a failing Array length sentinel for
stacking/OVR; changed rows, shape, dtype, layout and data; `set_params` and refit;
weighted unequal folds, temporary owners, output lifetime and input errors.
Those public tests are **not run in this lane**: they need the newly compiled
bindings. Existing public boundary tests remain applicable, including refusal of
unsupported caller protocols.

Integration must compile affected shared binding groups, run the new public
suite plus existing meta-boundary checks on the candidate, cover gather/scatter
callers (voting, multi-output, calibration and boosted/bagged trees), and retain
actual coverage. Reuse valid within-release identity evidence; record changed
source closures and run only invalidated/missing paths. No identity, GPU smoke,
release or performance claim follows from a source test passing.

Complete local logs and the validation summary are retained at
`~/mojolearn-evidence/native-metadata-once-20261027/`.

Source validation on this branch: 13 tests and 14 subtests passed; two existing
prep-plan tests failed. Both failures reproduce against retained base-commit
source (7d103e556): the meta fit plan/public operation-name assertion and the
keyword admission helper-name assertion. No dispatch repair is hidden in this
metadata change. See `validation.json` and the retained base-failure log. The
five new metadata source guards all passed; Python regression files parse.
