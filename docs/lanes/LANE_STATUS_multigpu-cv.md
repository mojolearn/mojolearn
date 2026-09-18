# Multi-GPU cross-validation implementation (WIP)

Worktree ~/mojolearn-wt/multigpu-cv, branch lane/multigpu-cv, based on main
0055bf441. Implements plan C5 while the existing release CPU matrix runs.
No new rental and no physical multi-GPU qualification yet. Keep this feature
branch off main until the actual GPU path is exercised on NVIDIA and AMD.

New explicit API: mojolearn.parallel_model_selection.cross_val_score(...,
devices=(0, 1)). Independent folds are assigned to persistent GPU workers in
bounded waves, one fold per selected device. Every fold is validated before
fitting. Parent-side cloning drops fitted state before serialization; each
fold gets independent constructor state. Only fold train/test rows are sent,
and scores return in fold order. Fit/scoring errors close the pool and raise;
there is no partial-result success or CPU parallel fit route. The serial API
uses the same validation and fit/scoring helpers with its existing contract.

DevicePool now rejects duplicate visible-device tokens (including whitespace
aliases) and invalid later indices before starting the first worker. This is
not physical UUID verification; mixed index/UUID aliases still need driver
witnesses. HIP's ROCR/HIP filter selection remains unchanged.

Validation checkpoint: 44 tests passed, 10 skipped (optional sklearn reference
tests unavailable in the pinned test environment). Tests use mocked workers
and byte gathers; they do NOT prove actual GPU execution. Covered fresh clones,
mutable parameters, uneven waves, fold/score order, scorer and fit errors,
pickling refusal, CPU/Metal refusal, device masks and HIP filter handling.
The initial invocation failed collection because the new worktree had no native
binding directory; rerunning with the existing release host directory resolved
that setup failure. Logs remain in mojolearn-evidence/next-wheel-coverage.

Next: physical UUID/PCI inventory per worker and duplicate-device refusal;
one-vs-two-device native gates on both vendors with prediction/model witnesses,
reversed device order, uneven folds, negative controls and timing kept separate.
Inventory alone must never be labeled evidence of kernel execution. Add an
actual-execution witness before admitting this as qualified multi-GPU coverage.
The module is not re-exported at the top-level API and no verifier lane is
promoted or release artifact changed by this feature branch.

## Driver inventory checkpoint

Added _gpu_witness.visible_gpu_inventory for CUDA Driver/HIP Runtime UUID,
PCI bus ID, visible ordinal, device name, worker PID and visibility-mask readback.
CV queries every worker before dispatching folds and requires exactly one visible
GPU per worker, unique UUIDs, unique PCI devices and unique worker processes.
This rejects mixed index/UUID aliases and two MIG instances on one PCI device.
Inventory is placement evidence only; it is not a kernel-execution trace.
The driver API signatures are linked in the source to NVIDIA and AMD references.

Updated contract tests: 66 passed, 10 optional sklearn tests skipped. These use
fake driver C calls and fake worker fits. Actual NVIDIA/AMD driver loading and
physical scheduling remain untested. Keep the feature off main. Next is the
one-vs-two-device numerical/placement runner with retained per-fold model and
prediction hashes; actual-execution tracing remains a separate admission gate.
