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
- Shared native operation locks are implemented for candidate construction,
  but shared mixed-layer orchestration and restored Transformer locks remain
  to migrate; preserve per-layer wait order and never pre-acquire all locks.
- Immutable Transformer/Mamba tape computation is now descriptor-driven on
  the candidate route; focused runtime consumption/concurrency coverage is pending.
- Samba resident and nonresident training/loss/optimizer routes need complete
  descriptor and transitive ownership coverage, including optional routes.
- Generation distributed/packed/foreign-buffer routes and CPU owners are not
  implied qualified by ordinary GPU FP32 checks.

The numerical loop owners remain in Mojo. Moving a file or finding zero Python
frames alone does not prove metadata is admitted once or that all internal
CPython dispatch is gone. Shared provider lifetime, admission and errors are
reused; no per-model provider implementation is added.

## Delivered source tranche: shared native operation lock

`core/operation_lock.mojo` owns a recursive host lock with stable native words,
thread identity, acquire/release atomics and a native contention wait. It keeps
per-layer acquisition order and never pre-acquires a model's set of locks.
`OperationLockLease` admits the owner once, retains it across the operation and
reuses its handle for acquire/release without further object metadata reads.
The Python-facing lock protocol consists of compiled instance descriptors;
there are no Python acquire/release/context-manager implementations.

Candidate identical Transformer construction creates this lock when the native
dispatch switch is enabled. Standalone fresh/carried forward, backward and tape
boundary calls use its lease directly. Legacy exact `_thread.RLock` instances
remain compatible under an explicitly separate legacy path; serialized objects
whose old `__setstate__` recreates those locks are not claimed migrated yet.

`native_lock_probe.py` is authored, not executed: recursion, contended waiting,
timeout, wrong-thread release, exception cleanup, public forward byte equality,
poisoned Python method bypass and unsupported-lock refusal. OS-signal behavior,
long contention and shared mixed-layer execution remain pending. The native
wait returns cancellation on EINTR rather than running application handlers in
native execution. No performance gain is claimed from the source change.

Immutable tapes (third source tranche): Transformer and Mamba3 forward snapshots
now use the shared block provider tables for forward, backward, and close. Exact
native owner allocation/handle admission is done once before forward; the tape
retains that provider image, boxed owner, handle, and family through consumption.
Replacing a block's later bindings cannot redirect an old tape to another image.
Reservations still occur under the provider GIL, numerical work releases it, and
cleanup preserves the previous consume-on-failed-backward contract. Close remains
idempotent. No new arithmetic or activation-retention default was introduced.

This applies only when the existing `MOJOLEARN_NEURAL_NATIVE_DISPATCH=1` candidate
and pre-existing `MOJOLEARN_IDN_ACT_RETAIN=3` compiled tape route are selected.
The default retained-session route, CPU route, and mixed Samba outer loops are
separate outstanding work. `samba_tape_probe.py` is authored but NOT executed:
it compares a public mixed-layer dropout training step's loss/parameter bytes
and requires table counters to prove the tape route was reached. Root owns
compilation and focused Mac runtime; tape consumption/failure/concurrency probes
and any unsupported compile branches remain pending, with no performance claim.

Owned Mamba3 snapshot tranche: the existing shared table also dispatches install,
export, carried-state forward/decode and close. Public `install_owned_weights`
now returns one native call; state publication happens after a successful native
upload. The installed snapshot retains its provider for export/close even if
later bindings change. Serialization drops the native provider references and
lazily restores owned snapshots as before. Arithmetic helpers, generation checks
and copy-on-install failure behavior are unchanged; the NN40 compile arm remains
opt-in. Source checks pass, but owned-weight runtime coverage remains pending.
