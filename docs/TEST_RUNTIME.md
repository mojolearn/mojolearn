# Fast iteration and fair Apple Metal testing

Use changed-lane checks while editing; reserve the complete cross-vendor matrix
for a release. A smaller selection is recorded as iteration coverage, never as
full qualification. The fixture sizes, repeat count and numerical contracts are
unchanged by this runner.

For one algorithm, name its lane explicitly. The default is the base fixture,
with two fits and the existing probes. `--exhaustive` selects all nine fixtures
for those lanes. `verify_lanes.py` uses the same base default.

```sh
pixi run -e test test-algo --lane transformer --plan
pixi run -e test test-algo --lane transformer --mode run --out /tmp/transformer-check
pixi run -e test test-algo --lane transformer --exhaustive --plan
```

The execution example requires a CPU host installation. Naming a lane tests
that algorithm only; use changed-path selection for shared source changes.

```sh
# Inspect the affected lanes without taking the GPU.
pixi run -e test test-identity-changed --plan --base HEAD^ cluster/host/kmeans_oracle.mojo

# One lane/fixture per Metal lease; two fits and all default probes per cell.
pixi run -e test test-identity-changed --base HEAD^ \
  --out /tmp/kmeans-check cluster/host/kmeans_oracle.mojo

# Additional hostile fixtures are explicit. Reuse completed cells after an interruption.
pixi run -e test test-identity-changed --base HEAD^ --fixtures base,odd,ties \
  --out /tmp/kmeans-hostile --resume cluster/host/kmeans_oracle.mojo

python3 tools/identity_timing.py /tmp/kmeans-hostile/*.json
```

The Python environment must have the intended installed MojoLearn bindings.
`--mode metal` is the Mac default; `--mode run` reserves only CPU capacity and
**does not switch a GPU installation to CPU execution**. Use a CPU installation
for host checks. The runner prints the selector's reasons. An unattributed
change refuses execution unless `--full-selection` is explicit. Splitting a
selection into processes does not bypass the existing Apple release guard.

Each cell has a 900-second execution limit by default (`--timeout` changes it).
Queue wait is separate. A timeout or refused stage stops the runner with a
nonzero exit. The child process group is terminated before releasing its lease.
The output directory carries the selection, per-cell records, and separate
scheduler timing records. Output streams live rather than waiting in a `tail`
pipeline. Completed records are not overwritten accidentally.

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
