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
## HANDOFF, for a session with no context (2026-09-18 09:0xZ)

**State: the forest hang is ISOLATED, FIXED, GATED and PUSHED. One of the
three reported manifestations is NOT explained and I could not reproduce
it.**

### What is settled

Releasing a resident forest and preparing another hung the process on main.
Cause: `ResidentForest.close` synchronized BEFORE its ten releases and
destroyed the `DeviceContext` immediately after them, so the context died
with their buffer frees in flight and the MAX runtime allocator's lock
stayed held for the whole process. That is DEVIATION 2520, already
isolated by lane/byte-lm-lifetime on 2026-09-11 with a native backtrace;
the resident structs never got its drain. Watched hanging on origin/main
(11 minutes, 130 of 132 threads in `futex_wait_queue`, GPU idle, native
stack identical to 2520's) and passing with the drain, same box, minutes
apart. 70 forest/GBDT/KDE cells bitwise IDENTICAL across the change, 0
moved, 0 refused.

### Manifestation 2: the two `-parallel` identity_break lanes ARE unblocked

Measured, above. On main they cannot produce a CUDA column at all; with the
drain they run the full protocol on one GPU. Both `SKIP_CUDA` defaults are
now empty.

### Manifestation 3: `verify --all` at `transformer-bf16w` is NOT EXPLAINED

lane/reference-regen's `verify --all` hung on NVIDIA at `transformer-bf16w`
after 47 clean lanes, 195 threads in `futex_wait`, 0 percent CPU, GPU idle
(`~/mojolearn-evidence/reference-regen/nvidia-hang/`). That is the same
SIGNATURE, and a signature is not a cause.

I found a real instance of the proven defect on that path and fixed it:
`bindings/_mojolearn_transformer.mojo`'s two STATELESS entries destroyed
their context right after destroying `w`, `kv`, `rope`, `stages` and `dx`,
and the backward one had the resident forest's exact misplaced
`synchronize()`. Both now drain. Bitwise gated: `transformer` alone
IDENTICAL=1, `transformer` + `transformer-bf16w` IDENTICAL=2.

**REPRODUCED ON THE SIXTH ATTEMPT (09:16Z to 09:26Z).** The prefix that
does it adds `mlp` and the three `byte-lm` lanes to the 29-lane list below,
run in ONE process on `/root/ml-before`:

```
rf-clf,rf-reg,et-clf,et-reg,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,
gbdt-rmse,kmeans,knn,knn-clf,knn-reg,dbscan,pca,pca-whiten,tsvd,ols,ridge,
logistic,kde,metrics,gbdt-ordered-rmse,gbdt-feature-freq,mlp,byte-lm,
byte-lm-host-infer,byte-lm-host-train,mamba1,mamba2,mamba3,transformer,
transformer-bf16w
```

279 cells completed, `transformer` finished all nine of its fixtures, and
the run then stopped dead at `# START transformer-bf16w/base repeat=1/train`
with **129 of 131 threads in `futex_wait_queue` and the GPU at 0 percent**
-- the reference-regen signature, at the reference-regen lane, in the
reference-regen order. (195 threads there against 131 here is the thread
pool tracking the box: 96 vCPU against 64.)

THE LANE LIST IS WHY THE FIRST FIVE ATTEMPTS PASSED, and it is a RACE, not
a threshold: the same `transformer` teardown runs in all of them. That is
what DEVIATION 2520's lane already said about the L40S ("its driver or
kernel drains faster than the release, which is a race, not a fix"). A
shorter prefix is not evidence of a cure, and no arm of this lane treats
it as one.

**THE DRAIN DOES NOT CURE IT. Measured, not assumed.** The identical lane
list on `/root/ml-after`, whose binding set is diff-equal to
`/root/ml-before`'s and which carries BOTH the resident-forest drain and
the transformer's two stateless drains, hangs in exactly the same place:

| tree | cells completed | stopped at | threads |
|---|---|---|---|
| ml-before (origin/main) | 279 | `transformer-bf16w/base repeat=1/train` | 129 of 131 in `futex_wait_queue`, GPU 0% |
| ml-after (DEVIATION 3010) | 279 | `transformer-bf16w/base repeat=1/train` | 129 of 131 in `futex_wait_queue`, GPU 0% |

Same cell count, same lane, same part, same signature. So:

**MANIFESTATIONS 1 AND 2 ARE ONE BUG AND ARE FIXED. MANIFESTATION 3 IS A
DIFFERENT DEFECT THAT SHARES THE SIGNATURE, AND THIS LANE DID NOT FIX IT.**

The transformer stateless drain is therefore INERT against this hang. It is
REACHED (the `transformer` lane runs all nine of its fixtures on the fixed
tree, through the patched binding) and it does not change the outcome. It
is kept because it removes a real instance of the proven defect -- the
backward entry had the resident forest's exact misplaced `synchronize()` --
and because it is bitwise inert on output (`transformer` alone IDENTICAL=1,
the pair IDENTICAL=2). It is NOT a fix for anything observed, and nothing
here should be read as one.

What this rules out for the next session, all by measurement:

- NOT the resident forest (`verify` and these lanes use the SEQUENTIAL
  engine and never construct a `ResidentForest`).
- NOT the transformer binding's four teardowns. All four now drain: the two
  stateless entries (patched here), `TransformerSession.__deinit__`/`clear`
  and `TransformerDecodeSession.release` (which already did). The hang
  survives all four.
- NOT the two-lane sequence: `transformer` then `transformer-bf16w` alone
  passes at one fixture, at nine fixtures with every part, and under
  `verify --all --lanes`.

### The backtrace, taken: it IS the same lock, and the caller is LINALG

Re-run with `tools/native_stack_dump.c` preloaded and `kill -USR2` from
outside while it was hung
(`bench/results/forest_deadlock_2026-09-18/HANG_transformer-bf16w.nativestack.txt`):

```
pthread_mutex_lock
libKGENCompilerRTShared.so (+0x8a1cc, +0x8a0ad, +0x75fc8, +0x5d7f3,
                            +0x73ebc, +0x8a4a6, +0x94650)
M::Driver::DeviceContext::enqueueCreateBuffer
AsyncRT_DeviceContext_createBuffer_async
_mojolearn_linalg.so   <- NOT the transformer binding
```

The SAME seven offsets in the same order as the forest hang and as
DEVIATION 2520's. So **all three manifestations block on the SAME LOCK in
the SAME allocator**: this is one defect CLASS, not three bugs. What
differs is the culprit teardown and the victim allocation, and DEVIATION
3010 covers only the forest's.

The victim here is `_mojolearn_linalg.so`'s low-bit GEMM entry
(`bindings/_mojolearn_linalg.mojo:288`), whose first
`enqueue_create_buffer` is the blocked call: `transformer-bf16w`'s train
goes through the bf16-weight GEMM. And the prefix change that MADE the hang
appear was adding `mlp`, which is the lane that pulls
`_mojolearn_linalg.so` in.

**THE SUSPECT WAS RIGHT, AND IT IS NOW CURED. Measured.** every GEMM entry in
`bindings/_mojolearn_linalg.mojo` creates `var ctx = DeviceContext()` inside
a `with GILReleased(...)` block and lets the context and its buffers die
together at the end of the block, with no drain and no explicit ordering
between them. `linalg_gemm_binding` (`:175`) is the barest: a
`DeviceContext()`, a call, and the end of the block. That is the shape
DEVIATION 2520 names, in the library the backtrace blames, reached by the
lane whose addition makes the hang appear. All seven entries now transfer their buffers
away and then `ctx.synchronize()` before the block ends.

THE A/B IS A SINGLE VARIABLE, because the previous arm had already ruled
the other two drains out:

| tree | linalg drain | cells | result |
|---|---|---|---|
| ml-before (origin/main) | no | 279 | **HANG** at `transformer-bf16w/base repeat=1/train` |
| ml-after, forest + transformer drains | no | 279 | **HANG**, same cell, same signature |
| ml-after, forest + transformer + LINALG drains | **yes** | **288** | **rc=0**, all nine `transformer-bf16w` cells complete |

288 is 279 plus the nine bf16w fixtures: the run finished. The ONLY change
between the second and third rows is `_mojolearn_linalg.so` rebuilt from
the patched source in the same tree.

Bitwise, before vs after over the whole reproducing prefix:
**IDENTICAL=252, 0 DIVERGENT, 0 MOVED** (450 infer/model, 243 batch, 36
rlpair). The 9 ONE-COLUMN cells are exactly the `transformer-bf16w` cells
main cannot produce, and the 27 REFUSED are the `byte-lm-host` lanes whose
host binding is absent on BOTH sides.

So all three manifestations are ONE DEFECT CLASS on ONE LOCK, and all
three are now fixed: the forest's resident teardown, and the linalg GEMM
entries'. The transformer stateless drain remains INERT against any
observed hang and is kept only as a correctness fix of the same class. `bindings/_mojolearn_mamba.mojo` has three undrained stateless
sites (`:380`, `:916`, `:1183`) and mamba1/2/3 run immediately before
`transformer`; `metrics/estimator.mojo` has nineteen; `mlp` pulls in
`training` and `linalg`, and adding `mlp` to the prefix is what made the
hang appear. It may also not be this class at all -- 129 threads in
`futex_wait` is the Mojo async runtime's pool, and only a native backtrace
of the hung thread will say which lock it is. THE NEXT STEP IS THAT
BACKTRACE, not another guess: build `tools/native_stack_dump.c`, run the
prefix with it preloaded and `MOJOLEARN_NATIVE_STACK_FILE` set, and
`kill -USR2` the hung pid from outside, exactly as this lane did for the
forest hang.
Four attempts, all on `/root/ml-before` (origin/main byte for byte) on the
RTX 4090 with driver 580.159.04, all PASSED:

1. `identity_break --lanes transformer,transformer-bf16w --repeats 1 --no-batch` -> 2 cells stable, 3 s
2. the same with all 9 fixtures and every part (`--batch-grad --batch-scale --ragged --step-full`) -> 18 cells stable
3. `python -m mojolearn verify --all --lanes transformer,transformer-bf16w` -> 3.2 s, and NOTE `transformer` itself was SKIPPED as "stale reference, not compared", so the suspected culprit lane never ran
4. a 29-lane prefix of the verify order (rf/trees/gbdt/core/estimators/metrics/mamba/transformer) in one process -> no hang

So the two-lane sequence is NOT sufficient. Whatever wedges the
reference-regen run needs something in the 47-lane prefix I have not
replicated. **Do not write "same bug" into anything until it reproduces.**

### THE EXACT NEXT COMMAND

A fifth attempt was RUNNING when this was written and its result is not in
this file. The three `byte-lm` lanes ran immediately before mamba/transformer
in the reference-regen log, byte-LM is where DEVIATION 2520 was originally
found, and `training/byte_lm_model_pool.mojo:139` and
`training/byte_lm_offload.mojo:172` are the two remaining undrained sites
of this class. So the next prefix includes them:

```sh
ssh -o StrictHostKeyChecking=no -p 52583 root@213.181.111.2
cd /root/ml-before
export PATH=$HOME/.pixi/bin:$PATH PYTHONPATH=python \
       MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 \
       MOJOLEARN_COMMIT=3e8dabc3776a8b22ccbc15c444b1dda4a27e6ca7
L="rf-clf,rf-reg,et-clf,et-reg,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,kmeans,knn,knn-clf,knn-reg,dbscan,pca,pca-whiten,tsvd,ols,ridge,logistic,kde,metrics,gbdt-ordered-rmse,gbdt-feature-freq,mlp,byte-lm,byte-lm-host-infer,byte-lm-host-train,mamba1,mamba2,mamba3,transformer,transformer-bf16w"
timeout -k 20 900 pixi run python3 tools/identity_break.py --lanes "$L" \
  --repeats 1 --vendor cuda-4090 --json /root/fd_out/before/before-prefix2.json \
  > /root/prefix2_before.log 2>&1; echo "rc=$?"
```

`rc=124` with the log stopping inside `transformer-bf16w` is the repro. Then
re-run the same command in `/root/ml-after` (which carries the drain) and
read the difference. If it does NOT hang, the next thing to add is the
families still missing from the prefix: `solver`, `svm`, `tsa`, `arima`,
`gp`, `preprocessing` (their bindings are NOT built on that pod yet), or
give up on the prefix and build all 24 bindings and run the real
`verify --all`.

### The class is much bigger than these fixes

A tree-wide scan for a `DeviceContext` destroyed after buffer releases with
no `synchronize()` between them returns **127 sites** outside `bench/`:
`bindings/_mojolearn_mamba.mojo` (3), `_mojolearn_rf.mojo`,
`_mojolearn_trees.mojo`, `_mojolearn_hdbscan.mojo`, `metrics/estimator.mojo`
(19), `mixture/estimator.mojo`, `gaussian_process/*`, `svm/estimator.mojo`
(5), `preprocessing/estimator.mojo` (4), `arima`, `tsa`, plus ~80 check
drivers. The scan is CRUDE and both over- and under-counts: it would have
MISSED the forest bug, because there the `synchronize()` was present but in
the wrong place. THIS IS NOT A LANE I OPENED (Andrew: no new lanes). It is
reported so the next person does not think DEVIATION 3010 closed the class.

### Pods

- `s14hskn0y3jorl`, RTX 4090 driver 580.159.04, $0.74/h. **TERMINATED
  after the work finished, DELETE 204 / GET 404.** While it was up it
  carried a `tools/runpod_guard.sh` lease, so it would have reaped itself
  had this session died. It held `/root/ml-before` (origin/main) and
  `/root/ml-after` (this branch) with 11 bindings each; everything under
  `/root/fd_out` was pulled before the reap.
- `uywhryt7b9vtz4` (the kNN pod) and `dgh8ghg5fu7iro` (a first forest pod
  whose driver was too old): both TERMINATED, DELETE 204 / GET 404.

### The kNN pod: done, filed, merged

Pod `uywhryt7b9vtz4` finished its own finish sequence at 07:30:25Z. Its
tip's identity, sabotage and k race are in
`docs/lanes/LANE_STATUS_knn-selector-speed.md` and
`bench/results/knn_selector_finish_2026-09-18/`, raw under
`~/mojolearn-evidence/knn-selector-speed/finish-2026-09-18/`. DEVIATIONs
3060 and 3061 move no bit (IDENTICAL=75/150/75 against the base AND against
the CPU column), both sabotage arms bite, the race is 0.38x to 0.95x with
digests equal. Five of its six diffs had died with `FileNotFoundError` and
printed an EMPTY summary that reads like a pass; they were re-run on the
pod with the right filenames before it was reaped. **lane/knn-selector-speed
is merged to main and pushed.** Nothing is owed there but the Apple and AMD
columns at the next release record.

### Evidence

`~/mojolearn-evidence/forest-deadlock/pod1/` -- the whole `fd_out` tree,
the driver scripts, the setup log, and the two `HANG_*` captures with the
native stack. Summaries committed under
`bench/results/forest_deadlock_2026-09-18/`.
