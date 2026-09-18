# CPU kernel identity completion

Worktree `~/mojolearn-wt/cpu-kernel-identity`, branch `lane/cpu-kernel-identity`.
Implements plan A3: polynomial, sigmoid and Laplacian KernelRidge/Nystroem CPU
arithmetic and saved-model inference. Degree now travels through the existing
binding params contract (previously ignored on CPU), bounded 0..32. The host
restates the device's FMA/polynomial/tanh/L1-exp operations, with no GPU imports.
The Laplacian host sabotage arm reverses the ascending feature sum.

Six new harness lanes have held-out, save/reload and batch probes. They are
explicitly pending (`no reference`) and do not enter default public verification.
Their saved-model recording debt is named in SAVED_MODEL_INFERENCE_OWED rather
than hidden by the older registry being empty. The new kernels do not imply
precomputed-kernel support or public CPU training.

Validation so far: 164 source/manifest checks passed. Native clean/sabotage
builds are queued under mac_slot after the running Apple capture. External
logs/builds: `~/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity/`.
Build queue session 3015; 900-second execution cap, one compiler job. Pinned
compiler and identical pixi.lock from release-087-final; outputs are isolated,
and no frozen release file/binary is changed. Runtime tests and numerical
reference comparisons must pass before merging this implementation to main.

Still required: native compilation, new CPU runtime tests, full nine-fixture
repeated CPU/Apple comparisons, observed sabotage detection, and independent
NVIDIA/AMD records plus installed-wheel qualification before default promotion.
Plan and user direction also require merging completed work and removing only
finished, clean, merged worktrees with no active dependent jobs. Preserve the
release worktrees and unrelated owners' active work.

## Checkpoint after native builds

The initial direct compiler invocation failed before compilation because it
lacked pixi's module-search environment (`unable to locate module std`). The
full failed log is retained as `clean/bootstrap-failed.log`. Re-running inside
`pixi run --manifest-path ../release-087-final/pixi.toml` succeeded: both clean
and sabotage host bindings built in 13.24 seconds, one compiler worker. No
algorithm fallback or accuracy condition was changed.

Runtime job session **5999** is queued for a single slot; `runtime-tests.log`.
It uses the new clean binding, with unchanged support bindings from the frozen
release host directory. Twenty-four new route/runtime tests plus the existing
CPU family tests are selected. Do not report a skip as numerical validation.

Fresh Apple kernel-family compilation and the CPU/Apple/sabotage comparison
are queued as session **63052**, execution cap 3600 seconds. External scripts:
`build-apple-and-capture.sh` and `capture.py`; log `comparison-queue.log`.
The driver stages current Python source, freshly builds the relevant Apple
kernel binding, records all six new lanes on all nine fixtures twice on CPU
and Apple, then requires every sabotage training cell to change. It retains
all native hashes through the harness and explicitly labels reused unchanged
support bindings. This is a source comparison, not installed-wheel or full
cross-vendor qualification. On failure, inspect comparison-receipt.json and
the per-lane log; repair and rerun into fresh output paths rather than erasing
the failed evidence.

Andrew explicitly requested periodic WIP commits in dedicated worktrees.
Checkpoint commits may be unqualified; only tested implementation increments
are merged to main. This branch's native implementation is still not merged.

## Source comparison and saved-model dependency checkpoint

Session 63052 passed: six lanes × nine fixtures × two repeats; 216 applicable
CPU/Apple numerical properties matched, and all 54 sabotage training cells
changed. Raw records, receipt, capture scripts and initial runtime log are
committed under bench/results/identity_break/2026-09-18-cpu-kernel-variants.
No installed-wheel or NVIDIA/AMD qualification is claimed.

Session 5999 returned 15 passed, 16 failed. The public saved-model subclasses
use the estimators host binding, not the reference-training family binding.
The reused frozen estimators binary refused the new kernels. Its source also
needed degree transport at both oracle call sites; that fix is checkpointed
here and needs a fresh build plus runtime rerun before merging. Frozen release
files remain unchanged. Do not erase the initial failed log.

The saved-model rebuild/rerun is session 44239 under mac_slot, one compiler
worker and a 900-second execution cap. Recipe build-estimators-and-test.sh is
retained beside the witness records; external estimators-queue.log and
runtime-tests-fresh-estimators.log record its outcome. The script builds fresh
clean and sabotage estimators bindings, replaces only this evidence directory's
symlinks, then runs the 31 selected kernel/family tests. It never writes into
the frozen release. This checkpoint is WIP pending that job's actual result.
