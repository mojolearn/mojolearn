# Neural provider completion, 2026-10-27

This continuation is source work, not runtime or all-neural qualification.
Root integrates and compiles the frozen changes on the Mac. No measurements
start until the public-operation audit and focused Mac checks are complete.

## Delivered source tranche: Mamba block dispatch

The existing common `NeuralDescriptorProvider` now admits a named Mamba block
operation table for Mamba1/2/3 backward and Mamba3 fresh/session/owned prefill.
It calls the existing numerical owners, preserving their ordering, report
optionality, owned-weight generation checks, and session failure handling.
The native typed session admission entry exposes only our own session pointer;
the common provider retains its owner/image and reserves/releases under the GIL.
No Python extension layout is inspected. Backward finite-input validation calls
the same native predicate directly instead of using CPython function dispatch.

The existing `MOJOLEARN_NEURAL_NATIVE_DISPATCH=1` candidate switch selects these
paths for GPU identical bindings. CPU/legacy branches retain supported behavior;
they are not represented as migrated. Old binding doors remain available for
bounded A/B and negative-control probes. This does not promote a performance
default or claim unchanged outputs before the runtime probe executes.

Root's `mamba_block_probe.py` exercises each family's public backward operation,
session/ephemeral Mamba3 variants, exact gradient bytes, dispatch counts and a
nonfinite-gradient refusal followed by a valid call. Owned-weight toggle, Samba
prefill forwarding, concurrent session refusal and tape still need focused
runtime coverage. The source dependency guard is separate evidence.

## Remaining execution/ownership work

- Samba mixed-layer forward/decode still traverses Python-held layer/state
  objects. Move descriptors/counters into native operation-local state.
- Attention currently waits on the block's recursive Python lock per layer.
  Preserve blocking, reentrancy and ordering with a shared native owner; never
  pre-acquire all model locks or inspect a Python lock object's private layout.
- Transformer/Mamba immutable forward-tape creation, backward and close still
  include compiled-entry CPython dispatch at the provider boundary.
- Samba resident and nonresident training/loss/optimizer routes need complete
  descriptor and transitive ownership coverage, including optional routes.
- Generation distributed/packed/foreign-buffer routes and CPU owners are not
  implied qualified by ordinary GPU FP32 checks.

The numerical loop owners remain in Mojo. Moving a file or finding zero Python
frames alone does not prove metadata is admitted once or that all internal
CPython dispatch is gone. Shared provider lifetime, admission and errors are
reused; no per-model provider implementation is added.
