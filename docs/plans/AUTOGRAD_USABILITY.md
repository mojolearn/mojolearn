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
