# Bounded test iteration

Routine checks use the internal CPU oracle, one algorithm and the base fixture.
Apple is bypassed by default. No product routing, PyPI API, public CPU training
permission, numerical contract or frozen reference is changed by this runner.

```sh
# Inspect the exact jobs, coverage and limits without running anything.
pixi run -e test test-algo --lane transformer --plan

# Two fits plus inference/save/reload, on prebuilt CPU oracle bindings.
pixi run -e test test-algo --lane transformer --host-dir /path/to/host \
  --out /tmp/transformer-core

# Run just the additional batch or decode-consistency contract.
pixi run -e test test-algo --lane transformer --probe-group batch \
  --host-dir /path/to/host --out /tmp/transformer-batch
pixi run -e test test-algo --lane transformer --probe-group rlpair \
  --host-dir /path/to/host --out /tmp/transformer-decode

# All applicable groups, each in its own bounded process and record.
pixi run -e test test-algo --lane transformer --probe-group all --plan
# All nine hostile fixtures remain explicit, for the selected algorithm.
pixi run -e test test-algo --lane transformer --exhaustive --plan
```

`--host-dir` defaults to `MOJOLEARN_HOST_DIR`, then this checkout's host
binding directory. Missing CPU bindings fail immediately; the runner never
builds all families or falls back to Metal. It stages only the checkout's
Python sources into the output directory and uses the package's existing
CPU-only installation route. `identity_break --require-cpu` checks the actual
backend before fitting. The installed package is untouched.

Each job has a **60-second execution limit and 60-second queue limit**.
`--timeout` and `--wait-timeout` make longer checks explicit. A timeout is a
failure, never reduced coverage reported as success. Child process groups are
terminated before leases are released. Results use separate
`LANE--FIXTURE--GROUP.json` records; `--resume` preserves completed matching
jobs. Do not merge different groups into a release column: their enabled
protocols differ deliberately.

The `core` group includes training and inference/save/reload. `batch` and
`rlpair` each rerun that prerequisite, plus only their selected contract.
`all` expands to separate jobs; inapplicable batch and undeclared rlpair probes
are omitted from its plan, and requesting one explicitly refuses. Both fits and fixture sizes stay
unchanged. Skipped probes are recorded N/A with their skip reason.

Changed-path selection is also supported by `test-identity-changed --base REF`.
Naming a lane does not prove coverage of a shared numerical primitive. A
selector fallback refuses execution unless `--full-selection` is explicit.
`--mode metal` requires a release marker or `--metal-diagnostic`, and remains
subject to the broad-matrix guard;
`--mode run` reserves CPU capacity but uses the installation's chosen backend,
for explicit non-Metal GPU work. Neither is the routine default.

See [TEST_ORACLE_SCOPE.md](TEST_ORACLE_SCOPE.md) for what each comparison proves.

## Scheduler

`bash tools/mac_slot.sh metal COMMAND...` retains the existing shell interface.
The implementation is `tools/mac_slot.py`, shared across worktrees. It uses a
persistent `flock` guard for state changes and a monotonic ticket counter.
Tickets are not reused while waiters remain. A Metal job acquires its CPU slot
and GPU lease together; it never holds an idle GPU while waiting for a CPU slot.

```sh
bash tools/mac_slot.sh status
bash tools/mac_slot.sh --timeout 120 --wait-timeout 300 metal COMMAND...
```

CPU capacity remains `MAC_SLOTS` (default 5). CPU thread/build settings and
`nice -n 19` retain the prior behavior. `MOJOLEARN_MAC_SLOT_BASE`,
`MOJOLEARN_METAL_LOCK`, and `MOJOLEARN_METAL_QUEUE` allow isolated test instances;
all participants sharing a resource must share the same paths.

Lease directories carry an ownership token and child process group. Cleanup
cannot remove a replacement owner's lease, and a surviving child keeps a killed
launcher's lease live. An unpublished legacy directory receives a 60-second
grace period. The scheduler respects live legacy locks, but **old running shell
schedulers do not honor the new guard or FIFO protocol**: allow them to drain
when replacing the external `~/mojolearn-evidence/tools/mac_slot.sh` entrypoint.
Do not delete active locks. Commands must not daemonize outside their process
group. A generic shell command is not preempted/requeued: bounded chunks come
from `identity_iterate.py`, which releases between every cell.

## Checkpoints, timings, and resume

`identity_break.py --json record.json` writes an atomic checkpoint after every
completed fixture, with `complete: false` until the full requested run finishes.
Every fit and inference/save/reload, batch, extra and rlpair stage emits START
and DONE lines with elapsed time. Each cell records its stage timings separately
from numerical hashes. A killed run retains the preceding complete checkpoint;
an unfinished fixture is repeated in full, including both fits.

`--resume --json record.json` requires exactly the same protocol, fixture bytes,
commit, environment, platform, Python/NumPy versions, Python sources and native
binding files. It refuses older records without the resume signature and changed
installations, including external host bindings. Completed failures are retained;
use a fresh output directory to rerun after a fix. Resume is recovery, not a way
to reuse results from different code. Timing metadata does not affect identity
comparison or duplicate-cell agreement during merge.

The iteration runner passes `--fail-on-refused`. Existing release recording
commands retain their historical refusal/exit policy so their comparison stage
can handle missing capability explicitly.

## What still needs measurement

Recorded tiny-kernel probes establish that host/device synchronization is costly
on this M4. They do not establish a decomposition of every lane's runtime.
Use the stage report to choose an expensive operation, then profile its native
launches and waits. Remove a wait only after proving dependency and buffer lifetime
safety and checking the resulting bytes. This change removes scheduling waste,
repeated completed work, and unnecessary broad iteration sweeps; it does not
claim a GPU kernel speedup or change floating-point arithmetic.

Do not lower fixture floors to force a short run. The shrink-blindness audit
showed why a single training step can hide state/copy and scheduler defects.
`tools/fixture_floors.py` remains the authority for those floors.

## Applicability preflight

CPU and Metal iteration plans include an `inapplicable` map derived by
`lane_applicability.py`. Execution refuses those selections before staging a
package or acquiring a lease: for example, multi-device driver claims cannot
be tested on the CPU-only route, and a CPU-only lane does not test Metal.
No rejected lane is silently counted as a pass. `--mode run` leaves backend
selection to the package and does not infer a column for this preflight.

Batch groups require a callable batch probe. A declared `n/a` batch contract
no longer creates an extra two-fit job under `--probe-group all`.
Timeouts must be finite; NaN and infinity are rejected before scheduling.

Apple qualification runs for PyPI updates through the installed-wheel release
workflow. The full Apple identity matrix is no longer an additional release
requirement. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md#5b-apple-qualification-once-per-pypi-update).
