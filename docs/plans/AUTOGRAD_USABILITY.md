# Native model-building improvements, 2026-10-20

Numerical source freeze: `7939b679a2079051f191d707067a487a9741fcce`.
The work began from `f650239ac3ec583fe25860bbf8b09b87724a2230`.

## Implemented

- Sequential dropout is compiled into ordinary bindings. Training uses the
  existing Philox tape operation and step/layer counters; validation and
  prediction disable dropout. Checkpoint continuation retains its stream.
- `Sequential.fit(validation_data=..., restore_best_weights=True)` evaluates
  held-out loss in Mojo. Patience monitors validation when supplied. The best
  snapshot includes parameters, optimizer arrays, bias-correction scalars,
  step and epoch. Validation classes map into the training vocabulary in
  native kernels, including full-width integer labels. Per-call validation
  history never becomes a later training-only stopping baseline.
- `Program.fit` replays a captured graph on a recording tape and performs
  upload, seeded shuffle, minibatches, backward, optimizer updates and
  schedules inside one native call. A parameter subset freezes constants.
  One scalar captured objective supports custom losses (`loss="output"`).
- `Program.run(..., dynamic_shapes=True)` derives output dimensions by replay.
  Inputs keep their captured rank; operation and parameter compatibility
  still determines which dimensions can change. Fixed-shape replay remains
  the default. The serialized graph format is unchanged.
- LayerNorm has a recorded VJP built from tape primitives for higher-order
  differentiation. Its ordinary fused first-order backward is unchanged.
  Epsilon is retained in the custom node. Input, scale and bias Hessian-vector
  products have an independent float64 finite-difference oracle.

The generic native bridge retains NumPy allocation only for returned ABI
buffers; all numerical conversion, graph and training work runs in Mojo.
The runtime fence recognizes `DeviceExec` generic specialization as a device
route. Its regression fixture still rejects a host data loop inside that
helper and still classifies the corresponding `HostExec` export as host.
No baseline exemption was added.

## Scope and evidence

Qualification uses `tools/autograd_usability_check.sh`, nine suites, 106 tests
per full GPU run and 96 distinct digest keys. It covers numerical oracles,
dropout resume, validation vocabulary/stopping/restoration, failed replay
recovery, native graph training versus eager training, custom objectives,
frozen parameters, changing batch/sequence dimensions, and checkpoint resume.

| Executor | Passed | Skipped | Digest keys matching Apple |
|---|---:|---:|---:|
| Apple GPU | 106 | 0 | 96 |
| NVIDIA RTX 3090 | 106 | 0 | 96 |
| AMD MI325X | 106 | 0 | 96 |
| CPU | 104 | 2 GPU comparison cases | 96 |

All 21 runtime-fence tests passed. GPU runs explicitly read back the sequence
binding's vendor and native numeric-mode enum. The final NVIDIA and AMD runs
are `cuda-test-03` and `hip-test-03`; earlier Linux runs are not GPU evidence:
an incomplete binary layout selected the host binding, and the initial AMD
environment lacked Mojo's shared runtime. Those failures, selection records
and the corrected runs are retained. No numerical rebuild was needed for
these test-environment repairs. The checkpoint suite now probes the actual
sequence binding instead of a generic vendor query.

Full build/test logs, source snapshots and native binaries are retained under
`~/mojolearn-evidence/autograd-usability-20261020/`.
The initial failed compile, initial fence fixture failure and their repairs
remain there alongside the successful runs.

This is source-build functionality and identity qualification, not a release
or an end-to-end performance A/B. No performance default was promoted and no
speedup is claimed. Installed-wheel qualification remains a release task.

## Remaining boundaries

- Captured programs remain straight-line graphs; there is no new general
  data-dependent control flow or graph optimizer.
- `Program.fit` accepts one output. Multiple heads can contribute to a
  captured scalar objective. Built-in targets are matrices/vectors with a
  sample row per input example. Validation stopping is currently Sequential's
  API; custom programs retain their optimizer in the owning session.
- LayerNorm support does not make every custom operation higher-order
  differentiable. Convolution, pooling, embedding scatter and other custom
  backward rules still need individual recorded lowerings and independent
  derivative checks.
- Variable rank and automatically resizable parameter tensors are outside
  this change. Captured reshape dimensions remain explicit; `-1` can infer a
  changing dimension.

Earlier plans, especially `AUTOGRAD_HIGHER_ORDER.md`, describe the original
refusal set. This implementation supersedes their LayerNorm refusal and
their fixed-shape-only program status.

## Graph completion source round (2026-10-20, qualification pending)

The `finish/graph-20261020` round adds native `Program.run_if` and bounded
`Program.while_loop` over captured component programs. Conditions return one
finite scalar; only the selected branch executes. Loop state stays on the
executor, may change dimensions while keeping rank, and returns an explicit
`stopped` flag alongside the final values and iteration count. Loop state is
floating-point and execution is inference-only. Components save/load through
the existing program format; the control-flow composition itself is not yet a
single serialized graph. This does not add differentiation through control
flow, arbitrary Python callbacks, or symbolic tracing.

Fixed and dynamic array preparation now share one native binding boundary,
including shape checks, numeric list marshaling, allocation, upload and output
packing. Parameters and all input descriptors are checked before allocation;
partial-upload failures release already acquired input buffers. Node failures
name their index, operation and actual input shapes, with replay cleanup before
rethrowing.

`Program.inspect` exposes graph references, attributes and shape metadata.
`Program.profile` returns outputs plus per-node host enqueue and synchronized
wait time, logical transfer bytes, parameter storage and retained scratch
capacity. Synchronization serializes the queue; these are diagnostic host
intervals, not GPU timestamps or end-to-end performance evidence. Scratch
capacity includes retained buffers from earlier replays and is not a complete
process or allocator memory total.

`optimize=True` is an inference-only, opt-in candidate that shares exactly
identical deterministic nodes. Float attributes compare by their bits;
random, copy and detach operations are excluded. It does not reassociate
arithmetic, fuse reductions, alter the saved graph or change graph training.
The default remains false. Full-dataset NVIDIA+AMD A/B, identity qualification,
source-build tests and installed-package checks remain pending for this round.
No speedup or default promotion is claimed.

Qualification entry points for the new round are `python -m mojolearn
verify-graph --vendor {cpu,metal,cuda,hip}` and
`tools/graph_features_check.sh {source,installed} <vendor> <evidence-dir>`.
The installed mode clears `PYTHONPATH`, uses a neutral working directory and
requires the actual package path under the interpreter's site-packages.
Nine required cases retain 21 output digests, backend/mode readback and
per-call transitive Python traces. Deliberate changed-digest, missing-case,
failed-case and wrong-mode comparator controls must be rejected. Native
qualifications remain pending; the 18 source/controller tests passing here
validate the verifier and runtime-audit machinery, not a GPU implementation.
The native boundary keeps the ndarray return interface through its cached C
type and a native bytearray owner. Input/output pointer metadata bypasses
NumPy's Python `ctypes` helper. CPU host binding proxy attributes resolve once
at setup, before the audited graph operation.
