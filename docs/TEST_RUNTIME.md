# Bounded test iteration

Routine checks use the internal CPU oracle, one algorithm and the base fixture.
Apple is bypassed by default. No product routing, PyPI API, public CPU training
permission, numerical contract or frozen reference is changed by this runner.

```sh
# Inspect the exact jobs, coverage and limits without running anything.
pixi run -e test test-algo --lane transformer --plan

# One fit plus inference/save/reload, on prebuilt CPU oracle bindings.
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

CPU/run invocations have a **five-minute budget** (`--budget 300`); Metal
diagnostics default to **one minute total** (`--budget 60`), including
planning, staging, queueing and execution. Each job also has a **60-second
execution limit and 60-second queue limit**. The scheduler receives one shared
monotonic deadline, so entering a new job or moving from queue to execution
does not reset the total budget. Process cleanup can add a short grace period.
Planning/staging are charged to the budget; no job starts if they exhaust it.

`run-summary.json` is updated atomically after each job and separates completed,
failed and pending jobs. Budget exhaustion exits 124 and never reports complete
coverage. Use `--resume` with the same selection/output to recover matching
records under a fresh budget; source/protocol provenance checks still apply.
Longer runs require an explicit larger `--budget`.

`--timeout` and `--wait-timeout` make longer checks explicit. A timeout is a
failure, never reduced coverage reported as success. Child process groups are
terminated before leases are released. Results use separate
`LANE--FIXTURE--GROUP.json` records; `--resume` preserves completed matching
jobs. Do not merge different groups into a release column: their enabled
protocols differ deliberately.

The `core` group includes training and inference/save/reload. `batch` and
`rlpair` each rerun that prerequisite, plus only their selected contract.
`all` expands to separate jobs; inapplicable batch and undeclared rlpair probes
are omitted from its plan, and requesting one explicitly refuses. Fixture sizes stay
unchanged. Skipped probes are recorded N/A with their skip reason.

Changed-path selection is also supported by `test-identity-changed --base REF`.
Naming a lane does not prove coverage of a shared numerical primitive. A
selector fallback refuses execution unless `--full-selection` is explicit.
`--mode metal` requires a release marker or `--metal-diagnostic`, and remains
subject to the broad-matrix guard;
`--mode run` reserves CPU capacity but uses the installation's chosen backend,
for legacy callers. Prefer explicit `--mode cuda` (NVIDIA) or `--mode hip`
(AMD), which reserve the GPU lease and check the loaded backend. None is the
routine default.

See [TEST_ORACLE_SCOPE.md](TEST_ORACLE_SCOPE.md) for what each comparison proves.
For a narrow native gradient-transfer check on CPU or GPU, see
[Transformer gradient readback](TRANSFORMER_GRADIENT_READBACK.md). It selects
one small case at a time and preserves complete bytewise output comparisons.
[Transformer weight setup](TRANSFORMER_WEIGHT_SETUP.md) documents batched
validation, separate output/refusal groups, and CPU/Metal/CUDA/HIP comparisons.

## One fit per cell

A check fits each cell ONCE (2026-09-19). A second fit on the same box only
separates "this box moves from run to run" from "this box differs from the
others", and a hash EQUAL to the reference a different machine produced already
rules out both. A repeat is informative only after a mismatch: rerun the
DIVERGENT cell with `--repeats 2` to classify it, never the selection.
`test-algo`, `tools/verify_lanes.py`, `python -m mojolearn verify` and its
cross-check all default to one. `tools/identity_break.py` itself still defaults
to two because its bytes are pinned by `verify_reference/table.json`
(`harness_sha256`); callers pass `--repeats 1`.

Measured on the M4 (`bench/results/identity_break/2026-09-18_installed-apple-properties`,
19 lanes, nine fixtures, two repeats, every probe): 615 s, of which train plus
inference/save/reload is 149 s (24%) and the batch, rlpair, batchgrad,
batchscale, ragged and stepfull probes are the other 76%. The end-model check
of the same cells at one repeat is therefore about 74 s, and about 8 s on the
base fixture alone.

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

CPU, Metal, CUDA and HIP iteration plans include an `inapplicable` map derived by
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

Changes to the iteration runner, scheduler and applicability audit are test
control changes, not numerical changes. The selector no longer widens those
paths into a full algorithm sweep. Tooling tests still apply; an import guard
checks that the package and identity harness do not depend on these tools.
Unknown paths and shared numerical dependencies retain conservative selection.

## Small Mac diagnostic rounds

```sh
pixi run -e test test-algo --lane transformer --mode metal \
  --metal-diagnostic --out /tmp/apple-transformer-core
```

The default runs one base fixture and one core group, with one fit, under a
60-second total deadline. To investigate batching or decoding, select just
`--probe-group batch` or `--probe-group rlpair`. An arbitrary single fixture
can be selected with `--fixtures NAME` without broadening the round.

The runner refuses multiple jobs unless `--metal-expanded` is explicit, even
with a release marker. Expanded jobs still share the one-minute budget unless
`--budget` is also changed. This does not change installed-wheel release gates.
No Apple runtime speedup is implied: this limits the amount tested per round.

## The same controls on CPU, AMD and NVIDIA

The routine default is CPU, base fixture, one fit and core probes on every
host. GPU work requires an explicit backend. The harness reads the loaded
backend before fitting, so a CUDA request cannot silently measure CPU or HIP.
All backends have per-job execution/queue limits and one shared run deadline.
GPU hosts keep their existing release qualification contracts; broad work is
explicit, not part of every edit.

```sh
# One selected NVIDIA or AMD lane on an already provisioned GPU host.
pixi run -e test test-algo --lane ridge --mode cuda --out /tmp/cuda-ridge
pixi run -e test test-algo --lane ridge --mode hip --out /tmp/hip-ridge

# Two CPU shards concurrently, sharing a five-minute total budget.
python tools/verify_lanes.py --lanes ridge,kmeans --backend cpu \
  --host-dir /path/to/prebuilt/host --shards 2 --jobs 2 --budget 300 \
  --out /tmp/cpu-round

# Inspect the CPU pod commands; this never rents or runs locally.
python tools/verify_lanes.py --lanes ridge,kmeans --runner pods --shards 2
```

`verify_lanes.py` now defaults to core probes and CPU-only source staging.
Use `--probe-group batch`, `rlpair`, or `all` deliberately. Fixture expansion,
backend selection and selector fallback remain explicit. `--resume` keeps
matching checkpoints; it never deletes earlier parts before running. A failed
shard stops further scheduling and cancels active peers through their
schedulers. The summary is complete only after merged records pass validation,
including sub-verdicts. A merge timeout also leaves the run incomplete.

CPU `--jobs` is explicit and cannot exceed shared capacity (`MAC_SLOTS`,
default 5); every child uses one compute thread. GPU `--jobs` stays one per
host. CUDA/HIP use the same exclusive GPU lease mechanism as Metal. Separate
CPU, NVIDIA, AMD and Apple hosts can run independently. Multiple GPUs in one
host are conservatively serialized: this change does not assign physical
devices or permit same-GPU concurrency. Old processes launched outside the
scheduler do not acquire its lease.

## Selected Python gates

`check-python-gates` no longer silently runs every module. It requires named
gates or an explicit `--all`. Only selecting `test_linalg_identity` can launch
the CPU oracle-card build, and that build shares the total budget.

```sh
pixi run check-python-gates --list
pixi run check-python-gates --gate test_linalg_identity --backend cpu \
  --host-dir /path/to/prebuilt/host --out /tmp/linalg-gate --plan
# Real-Apple broad gates are an explicit release task.
pixi run check-python-gates-release
```

Some public API gates require GPU training. CPU selection does not grant
public CPU training or silently reroute such a gate; choose its actual GPU
backend when needed. Gate logs and atomic summaries preserve failures and
unrun gates. The release task uses a ten-minute total budget, with each gate
still bounded to one minute. Other broad qualifications can explicitly set
`--all`, their backend and their budget.

## Recovery and exact coverage

The shard runner validates the requested **lane × fixture** set after merging.
Having one stable cell per lane is insufficient when several fixtures were
requested. Missing/extra cells, missing training verdicts, malformed JSON and
records not marked complete all produce an incomplete result. Failed merged
output is kept under `column.incomplete.json`, never `column.json`.

Before a resume changes manifests, logs or merged output, it compares the
commit, selected lanes, shard membership, fixtures, repeat count, backend and
probe group with the original manifest. A changed scope refuses and leaves the
existing evidence untouched. A fresh budget or a different CPU concurrency
limit is allowed; changing shard membership requires a new output directory.
The harness still checks source, binary and protocol provenance per checkpoint.

## Gate applicability before launching work

Gate modules may declare a literal `GATE_BACKENDS = ("cpu",)` (or a tuple of
supported backend names). The runner reads this declaration without importing
the gate or initializing a GPU. Undeclared scope remains conservative: the
module stays selectable on every backend. Invalid declarations refuse.

For `--all`, the plan and summary list inapplicable modules under `excluded`,
with reasons; those modules are not counted as completed checks. Explicitly
requesting an inapplicable `--gate` refuses before staging or scheduling.
CPU training gates now declare their CPU-only scope: their runtime checks
already skipped when a GPU set was loaded. Their source and host-runtime
checks remain available on CPU.

On 2026-09-17 this removes 19 CPU-training modules from each broad GPU gate
plan: 55 discovered modules become 36 selected modules. That is fewer modules
to launch, not a measured 35% reduction in elapsed time, and it does not change
the selected modules' assertions or public estimator behavior.
