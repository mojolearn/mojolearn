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

## Prepared numerical/placement hardware runner

Added tools/parallel_cross_val_check.py and importable scorer
 tools/parallel_cv_witness.py. Two GBDT adapter families, five uneven folds,
one/two/reversed devices, two repeats. The scorer retains raw saved models,
full model/prediction/curve hashes, save/reload comparisons, score bits,
worker driver identity and native GBDT binding hashes. Dropped folds, changed
models/predictions and device aliases are explicit comparator controls.
These are comparator controls, not injected native computational faults.

The runner has NOT executed on hardware yet. A passing run reports
NUMERICS_AND_PLACEMENT_PASS with physical_execution_trace=OWED, never a full
multi-GPU certificate. Run from a clean checkout with matching GPU bindings,
inside the bounded rental/watchdog workflow, and use an output path outside
the checkout, e.g. on a provisioned Linux GPU host:

```
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$PWD/python"   timeout 1200s python tools/parallel_cross_val_check.py   --require-backend cuda --devices 0,1 --out /tmp/cv-cuda-unique-run
```

Repeat on HIP with --require-backend hip and a fresh output directory. Retain
all receipts/models, account for tool/wheel source provenance, and verify rental
deletion as usual. Do not start new rentals during the active CPU release matrix.
No rented resource was started by this branch.

Latest validation: 91 passed, 10 optional sklearn skips (software contracts).
Logs committed under bench/results/parallel_cv/2026-09-18-software-contracts.
The independent visibility-mask fix and its 15 tests landed on main 4e8ee54a1.
The CV implementation and driver inventory remain only on this feature branch.

## Expanded 0.8.7 follow-up

Merged current main into this branch, including the separately added forecast
worker operation. The hardware capture now accepts a guarded git archive with
a full commit witness and hashes the actual CV/scorer/worker/inventory source
files; clean checkouts remain required when .git exists. Previously the remote
runner could not start from the archive that every guarded cloud leg ships.

`tools/parallel_cv_remote_leg.sh` is a bounded two-GPU extra body: one GBDT
binding build and the one/two/reversed-device numerical/placement capture.
The outer controller must provision two devices and retain/delete them under
its watchdog. This does not manufacture an execution trace from inventory.
68 targeted software tests passed after integration. Hardware execution and
trace qualification are still owed, and this branch remains outside main.

## September 18 bounded NVIDIA physical leg preparation

Integrated current main plus classical distribution through 8be53d360. The
shared `_parallel_worker` merge retains CV inventory/folds and loaded-LM layer
operations. `parallel_cv_remote_leg.sh` now runs CV first, then builds ARIMA,
TSA, GP and IVF for a thirty-cell one/two/reversed-device capture. Completed
stages remain checkpointed; optional Nsight Systems captures/export and per-PID
nvidia-smi pmon samples retain execution evidence when available. Nine ordinary
GP/GPC/IVF/GBDT holds reuse these builds only if time remains.

Controller is unchanged `gemm_remote_leg.sh` with GPU_COUNT=2, a 55-minute armed
lease, 180-second readiness cap, 2700-second work polling cap, and its original
pre-create deadman, on-pod watchdog, source archive and verified deletion guards.
Body cap2300s includes all extra builds/captures. Existing generic GEMM gates run
before the extra body; their reused older Apple card does not qualify new GEMM
source and is not used as classical/CV evidence. No GEMM source changes.

Live inventory showed no two-4090 stock, and low two-L40S stock (~$1.58/h quoted,
$2.18/h secure list); root approved L40S under a total ~$8 cap. Dry run at
3805c5840 was GREEN. Evidence/logs are in
`/Users/andrewhendel/mojolearn-evidence/classical-distributed/nvidia-two-gpu-*`.
Actual rental/capture/deletion outcome will be appended after the guarded leg.
