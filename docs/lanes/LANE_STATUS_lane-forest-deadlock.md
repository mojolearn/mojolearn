# LANE STATUS: lane/forest-deadlock (2026-09-18)

A shipped deadlock on main, on the DEFAULT build, user-visible: releasing a
resident forest and then preparing another hangs the process. DEVIATION
3010 is the fix. Written for a session with no memory of this lane.

## The shape

`inference_engine="parallel_groves"` puts a forest's nodes on the device
once and keeps them there (`_ResidentForest`, `python/mojolearn/
_forest_protocol.py:22-32`; `ResidentForest`, `core/
forest_inference_model.mojo`). Each snapshot creates its OWN
`DeviceContext()`. The Python object's `weakref.finalize` calls
`native.forest_release_gpu`, so the snapshot is released when the estimator
is collected.

lane/forest-groves-row-schedule found the hang on the way past and reported
it without fixing it (`docs/lanes/LANE_STATUS_lane-forest-groves-row-
schedule.md`, "A deadlock on main, found on the way"):
`tools/forest_groves_identity.py large` finished `et-higgs` and then sat in
`futex_wait` on 195 threads, 0 percent CPU, GPU idle, for 30 minutes, on a
96-vCPU RTX 4090 pod. Every single model passed ALONE. Two live snapshots
passed. Only RELEASE THEN PREPARE hung, inside the second model's
`forest_prepare_gpu`.

## The cause: not new, and already written down

It is DEVIATION 2520, isolated by lane/byte-lm-lifetime on 2026-09-11 with
a native backtrace (`docs/lanes/BRIEF_byte_lm_lifetime_2026-09-10.md`,
"Run 6: the native stack, and the cause"). Releasing a device-owning object
enqueues its buffer frees on the context's stream; destroying the context
with those frees IN FLIGHT leaves the MAX runtime allocator's lock held,
and the next context's first `enqueueCreateBuffer` blocks in
`pthread_mutex_lock` for ever:

```
pthread_mutex_lock
libKGENCompilerRTShared.so (+0x8a1cc ...)
M::Driver::DeviceContext::enqueueCreateBuffer
AsyncRT_DeviceContext_createBuffer_async
```

The cure is to DRAIN between the releases and the context's death. The byte
LM and transformer bindings have carried that drain since 2026-09-11. The
resident structs never got it, and `ResidentForest.close` had its
`synchronize()` in the WRONG PLACE, which is easy to read as correct:

```
    def close(mut self) raises:
        self.pool = None
        if self.ctx:
            self.ctx.value().synchronize()   # drains the last PREDICTION
        self.output_workspace = None          # ... and then TEN releases
        ...                                   #     enqueue their frees
        self.offsets = None
        self.ctx = None                       # context dies, frees in flight
```

That is the deadlock, on main, on the default build.

## What DEVIATION 3010 changes

Four teardowns synchronize AFTER the releases and BEFORE the context goes:

| file | teardown |
|---|---|
| `core/forest_inference_model.mojo` | `ResidentForest.close`, `ResidentForest.__deinit__` |
| `core/forest_inference_pool.mojo` | `PooledForest.__deinit__` (multi-GPU) |
| `gbdt/resident_model.mojo` | the resident GBDT model's `__deinit__` |
| `kde/resident_fit.mojo` | `ResidentKdeFit.__deinit__` |

Host-side drain at teardown only. No kernel, no launch geometry, no
arithmetic, no reduction order: nothing a prediction's bits can see.

## Two sites of the same class are LEFT ALONE, and why

`training/byte_lm_model_pool.mojo:139` and
`training/byte_lm_offload.mojo:172` drop their buffers and then their
context with nothing between. Their explicit `close()` methods ALREADY
carry the 2520 drain, so only the drop-without-close path is exposed, and
both are neural-lane code this lane neither built nor gated. They are
reported, not touched (Andrew's "no more new lanes"). They are the last two
in the tree: a scan of every `.mojo` outside `bench/` for a context
destroyed after buffer releases with no `synchronize()` between returns
exactly these two once DEVIATION 3010 is applied.

## The box, and why the two trees differ in four files and nothing else

RunPod secure cloud pod `s14hskn0y3jorl`, one NVIDIA GeForce RTX 4090
(sm_89, driver 580.159.04), 64 vCPU, $0.74/h, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, Mojo from the
repository's pixi lock. The driver matters: the first pod rented for this
lane came up on 570.195.03, where Mojo refuses the module outright
(">= 580 required") and the system-ptxas fallback then died with
`CUDA_ERROR_INVALID_IMAGE`. It was terminated unused and a pod pinned to
`allowedCudaVersions: ["13.0"]` rented in its place, which is also the
driver family the original observation was made on.

`/root/ml-after` is this branch. `/root/ml-before` is the SAME tarball with
the drain patch reversed, so the two trees differ in exactly the four files
of the table above and in nothing else (`diff -rq` prints those four lines
and no others), and `ml-before`'s four files hash to `origin/main`'s blobs:

```
329e6f4a0bb500c0  core/forest_inference_model.mojo
9d8efcce5ec908b4  core/forest_inference_pool.mojo
363d126f7c897713  gbdt/resident_model.mojo
d4ba7e9bb3b8473b  kde/resident_fit.mojo
```

Five bindings built in each tree from its own sources into its own
directory (`_mojolearn`, `_mojolearn_rf`, `_mojolearn_trees`,
`_mojolearn_gbdt`, `_mojolearn_estimators`), so every file DEVIATION 3010
touches is COMPILED on the fixed side and no `.so` is shared between the
arms.

## Watched failing, then watched passing

`tools/forest_release_prepare_repro.py`, both trees, both modes:

| tree | mode | result |
|---|---|---|
| ml-before (origin/main) | `release` | **HANG**: entered phase 5 at 07:51:36Z and was still in it 11 minutes later; killed |
| ml-before (origin/main) | `keep` | DONE, A `ab11bdc3e35bda72`, B `1159155b9aee07b0` |
| ml-after (DEVIATION 3010) | `release` | DONE, A `ab11bdc3e35bda72`, B `1159155b9aee07b0` |
| ml-after (DEVIATION 3010) | `keep` | DONE, A `ab11bdc3e35bda72`, B `1159155b9aee07b0` |

The hang and the pass are the SAME PROGRAM on the SAME box minutes apart,
and the three arms that complete agree on both prediction hashes, so the
drain did not buy the pass by changing what is computed.

While the unfixed arm was hung: 132 threads, **130 in `futex_wait_queue`**,
0 percent CPU, GPU utilization 0 percent. Two SIGUSR2 samples two seconds
apart gave the SAME frames (`~/mojolearn-evidence/forest-deadlock/pod1/
HANG_before-release.nativestack.txt`):

```
pthread_mutex_lock
libKGENCompilerRTShared.so (+0x8a1cc, +0x8a0ad, +0x75fc8, +0x5d7f3,
                            +0x73ebc, +0x8a4a6, +0x94650)
M::Driver::DeviceContext::enqueueCreateBuffer
AsyncRT_DeviceContext_createBuffer_async
_mojolearn_rf.so   <- ResidentForest.__init__'s first enqueue_create_buffer
```

Those are the SAME seven `libKGENCompilerRTShared` offsets, in the same
order, that lane/byte-lm-lifetime recorded on 2026-09-11 for a hang in
`_mojolearn_byte_lm.so`. THE CAUSE IS NOT INFERRED FROM THE FIX WORKING: it
is the same lock, in the same allocator, reached from the same two runtime
frames, and the intervention that removes it is the one DEVIATION 2520
already named.

## The lock is PROCESS-WIDE, which is wider than "the next context"

Two measurements on the unfixed tree say the wedge is not confined to a
later context:

- `identity_break --lanes rf-clf-balanced-parallel --repeats 2` hangs at
  `base repeat=1/batch`. The batch part predicts with an estimator whose
  snapshot is ALIVE; what died before it is the save/reload estimator of
  the `model` part, collected at the end of that part.
- the same lane with `--no-batch` hangs at `base repeat=2/train`.

So after ANY context is destroyed with its frees in flight, the next
`enqueue_create_buffer` in the process blocks even on a context that is
still open. That is why "two live snapshots are fine" and "release then
prepare hangs" are both true, and it is why the exposure is any program
that lets one groves forest go.

## A watchdog inside the process cannot see this hang

The repro's own watchdog thread never ran: `--deadline 240` elapsed and the
process was still in phase 5 eleven minutes later. The forest binding holds
the GIL across `forest_prepare_gpu`, so once the main thread blocks inside
it no other Python thread is ever scheduled, and `faulthandler`'s own
timeout and a Python-level `SIGALRM` handler would have been just as
silent. The deadline has to come from outside (`timeout -k 30`), and the
native backtrace above was obtainable only because a SIGNAL HANDLER is not
a Python thread.

## The two `-parallel` identity_break lanes ARE unblocked

`rf-clf-balanced-parallel` and `et-reg-bootstrap-parallel` have been left
out of every CUDA column for weeks (`tools/infer_speed_trees_body.sh`,
`tools/forest_groves_body.sh`, both `SKIP_CUDA`). Measured on this box:

| tree | protocol | result |
|---|---|---|
| ml-before (main) | `--repeats 2`, rf lane alone | **HANG** at `base repeat=1/batch` |
| ml-before (main) | `--repeats 2 --no-batch`, rf lane alone | **HANG** at `base repeat=2/train` |
| ml-before (main) | `--repeats 1 --no-batch`, both lanes | **HANG (exit 124)** after ONE cell: rf completed, et hung at `repeat=1/train` |
| ml-after (3010) | `--repeats 1 --no-batch`, both lanes | 2 cells, 2 stable, 0 moved |
| ml-after (3010) | **full: `--repeats 2` WITH batch, both lanes** | 2 cells, 2 stable, 0 moved, `batch: stable=2` |

So on main these two lanes cannot produce a CUDA column at all -- not
together, not alone, and not even with the batch part removed. With the
drain they run the shipped protocol, batch part included, on one GPU. Both
`SKIP_CUDA` defaults are now empty and carry the reason.

The skip's stated cause was wrong and is corrected in both scripts: it was
read as "the parallel pool's batch protocol on this box". It is not the
batch protocol. The batch part predicts with an estimator whose snapshot is
ALIVE; what dies first is the `model` part's save/reload estimator, and the
lock it leaves held is process-wide.

## Bitwise: nothing moved

`identity_break --diff`, the SAME lane list and fixtures on both trees,
`--allow-separate-builds` because the two columns are deliberately
different builds:

| comparison | train | infer/model | batch |
|---|---|---|---|
| before vs after, 13 forest/GBDT/KDE lanes x 5 fixtures | IDENTICAL=65 | IDENTICAL=130 | IDENTICAL=65 |
| before vs after, `rf-score-weighted` x 5 fixtures | IDENTICAL=5 | n/a | n/a |
| before vs after, `rf-clf-balanced-parallel` (the one parallel cell main could produce) | IDENTICAL=1 | IDENTICAL=2 | n/a (--no-batch) |
| after: reduced protocol vs full protocol, both parallel lanes | IDENTICAL=2 | IDENTICAL=4 | n/a |

70 cells, 0 MOVED, 0 REFUSED. Lanes: `rf-clf`, `rf-reg`, `et-clf`,
`et-reg`, `rf-clf-entropy-log2-noboot`, `rf-reg-poisson`,
`rf-reg-gamma-ig`, `et-clf-entropy-bestfirst`, `rf-score-weighted`,
`gbdt-symmetric`, `gbdt-depthwise`, `gbdt-rmse`, `kde`, `kde-weighted`,
over `base,ties,odd,dupes,wide`, two repeats each. `rf-score-weighted`
first came back REFUSED on BOTH columns for a `_mojolearn_metrics.so` that
had not been built; the binding was built in both trees and the five cells
re-run, rather than leaving a symmetric refusal in the table.

The `et-reg-bootstrap-parallel` cell is ONE-COLUMN and says so: main cannot
produce it. A diff table that printed IDENTICAL there would be lying, and
identity_break labels the killed column INCOMPLETE in its own header.

## Evidence

`~/mojolearn-evidence/forest-deadlock/pod1/` (outside the repo): the
`fd_out` tree with every JSON, console and diff, the three driver scripts,
the setup log, and the two `HANG_*` captures with the native stack.
Summaries are committed under `bench/results/forest_deadlock_2026-09-18/`.

## Owed

- Apple and AMD: the drain compiles on every column (it is one
  `synchronize()` per teardown) but was RUN only on NVIDIA sm_89. The
  defect itself is sm_89-shaped in every observation on record: an L40S
  never showed it, and DEVIATION 2520's lane said the same. Nothing here
  claims the other columns were exercised.
- `training/byte_lm_model_pool.mojo:139` and
  `training/byte_lm_offload.mojo:172`, above: the same class, reported and
  not touched.
- The multi-GPU `PooledForest.__deinit__` drain is COMPILED and never RUN:
  `forest_device_count() > 1` needs a second device and this pod had one.
  Report it as unexercised, not as verified.
