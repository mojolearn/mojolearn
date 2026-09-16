# LANE STATUS: lane/cpu-verifier-byte-lm-pool-offload (2026-09-16)

One question, answered no. **Can `par-byte-lm-model-pool` and
`par-byte-lm-offload` be honestly verified on a CPU column if the verifier
runs TWO logical workers instead of one?** Branch
`lane/cpu-verifier-byte-lm-pool-offload`, cut from origin/main bfb8f725a.
Wave 3's triage is `docs/lanes/LANE_STATUS_lane-cpu-training-par-wave3.md`,
section (c), which this lane rewrites with the permanent reason.

No code changed. No lane was covered. No box was rented, no Metal job was
taken, and nothing on `release/0.8.6` or db9047b9f was touched.

## The answer

No, and worker count was never the reason. Wave 3's sentence was right about
the conclusion and wrong about the mechanism, which mattered because the
mechanism it named ("with one CPU worker both sides are the same host
arithmetic") invited exactly this question. The real reasons are three, and
none of them moves when a second logical worker is added.

### 1. There are no workers on this path to give

Neither trainer touches `DevicePool`. Both open a NATIVE session on the
byte-LM binding.

    python/mojolearn/model_pool_training.py   lines 11-19, 50-73
    python/mojolearn/offload_training.py      lines 11-19, 50-72
    python/mojolearn/parallel_training.py     lines 51-87  (the base _open)

`_ModelPoolBinding` and `_OffloadBinding` rename `byte_lm_model_pool_*` and
`byte_lm_offload_*` onto the `byte_lm_parallel_*` transaction adapter, and
`_open` hands `list(self.devices)` to `byte_lm_parallel_open`. The pool's
admission list never sees the call. The decisive count:

    grep -c byte_lm python/mojolearn/_parallel_worker.py     # 0

The worker serves scaler, arima, holtwinters, neighbor, forest, mlp and samba
operations and no byte-LM operation at all. So
`_parallel_pool.CPU_OPERATIONS` and `CPU_SINGLE_DEVICE_COOPERATIVE`, the only
knobs a worker count could move, are not on this path in either direction.
Adding a name to them changes nothing for these two lanes.

### 2. `devices` indexes device contexts, not processes

`training/byte_lm_layer_pool.mojo`, in `ByteLayerPool.open`:

    var owner = (layer+Int(reserve_head_device)) * len(devices) // (shape.n_layers+Int(reserve_head_device))

`owner` subscripts `self.contexts`, a `List[DeviceContext]` filled by
`DeviceContext(device_id=devices[i])`, and `byte_lm_model_pool.mojo` uses it
the same way (`self.layers.contexts[owner]`). A CPU-only install builds no
`DeviceContext`, so there is no second owner to create by asking for one, at
any worker count. Two logical workers would be two processes; ownership here
is two contexts inside one process.

### 3. What a host could restate cannot fail, and what can fail is not host arithmetic

The pooled step folds each logical shard's gradient with `_ordered_add_kernel`
inside the owned range, then runs `byte_glue_update_launch` per chunk over
`len(chunk.p)` (`training/byte_lm_model_pool.mojo`, `step`). That update is
elementwise AdamW. `training/byte_lm.mojo` says the glue update refuses SGD
and clipping, the byte LM admits neither (`byte_validate_optimizer`), and
`python/mojolearn/_byte_lm_impl.py` line 166 fixes `max_norm=0.0`, so no
cross-tensor norm exists to be partitioned. Cutting an elementwise AdamW into
contiguous chunks cannot move a bit, whichever process runs it. A host
restatement of the ownership split would therefore be a check that cannot
fail, which is the thing this lane was told not to build.

What CAN fail in these lanes is the cross-context traffic, `transfer_bytes`
and `enqueue_copy_to` between contexts in `training/byte_lm_parallel.mojo` and
`training/byte_lm_layer_pool.mojo`. A one-process host route has nothing to
put in its place. This is the hazard class of
[[amd-mi300x-sriov-peer-copy-stale-read]], device transport, which only ever
reproduces on a device.

### And the reference side is single-worker by construction

Both lane bodies in `tools/identity_break.py` pin the reference to one device:

    ParallelByteLanguageModelTrainer(state0, devices=_par_devices()[:1],
                                     logical_shards=2, pool_optimizer=False)

A second worker could never reach it. It has no CPU route of its own either;
that is wave 3's b2 item, `par-byte-lm`.

## Seen, not assumed

A genuine CPU-only install was made without building anything and without a
GPU binary, by pointing `MOJOLEARN_HOST_DIR` at the already built host set
from a worktree that has no `.so` of its own. `select()` then falls to
`_select_cpu_only`, `vendor()` reads `cpu` and `_backend._CPU_ONLY` is set.

    cd <worktree>
    export PYTHONPATH=$PWD/python MOJOLEARN_NUMERIC_MODE=identical \
           MOJOLEARN_HOST_DIR=/Users/andrewhendel/CascadeProjects/mojolearn/python/mojolearn/host
    nice -n 19 python3 tools/identity_break.py \
      --lanes byte-lm,par-byte-lm,par-byte-lm-model-pool,par-byte-lm-offload \
      --fixtures base --repeats 1 --json <out>.json

Train column, vendor `cpu-apple-m4`:

| cell | verdict | by-name refusal |
|---|---|---|
| byte-lm/base | **STABLE** | the control: the CPU byte-LM route does produce a hash |
| par-byte-lm-model-pool/base | REFUSED | `rebuild bindings/build_byte_lm.sh for model pooling` |
| par-byte-lm-offload/base | REFUSED | `rebuild bindings/build_byte_lm.sh for offloaded replay` |
| par-byte-lm/base | REFUSED | `rebuild bindings/build_byte_lm.sh for parallel training` |

The refusal is `_byte_lm_trainer_host._HostTrainerBinding.__getattr__`, which
makes every `byte_lm_model_pool_*`, `byte_lm_offload_*` and
`byte_lm_parallel_*` entry read absent to `getattr(..., None)`:

    mojolearn: no CPU implementation of _mojolearn_byte_lm.byte_lm_model_pool_create;
    the CPU trainer (python/mojolearn/_byte_lm_trainer_host.py) serves the
    single-device entries only

`bindings/_mojolearn_byte_lm_host.mojo` exports eleven single-device
`byte_lm_host_*` entries and no session of any kind, which is the same fact
from the Mojo side.

**The probe was made to fail before it was believed.** The same two adapters
built over a stub binding that DOES carry those names construct without
refusing, so a REFUSED cell is the absent entries and not a broken probe. The
`byte-lm/base` STABLE cell is the second control, on the live route.

    <scratch>/probe_refusal.py      sections A (CPU binding, both refuse),
                                    B (positive control, both construct),
                                    C (the by-name AttributeError text)

## What would actually be needed, if it is ever wanted

A host restatement of the LAYER SCHEDULE, not of the ownership split. That is
a host `ByteLayerPool` forward and backward beside wave 3's b2 host parallel
session, new host arithmetic and a lane of its own. Even then it would check
the schedule and never the device transport the GPU columns check, so the
honest place for these two lanes stays the GPU columns.

The record already carries them there. All six committed 166-lane columns hold
**nine STABLE cells each** for both lanes (`cells` keyed `<lane>/<fixture>`,
verified with a control lane name that returns zero):

    bench/results/identity_break/2026-09-14_166-lanes/{apple-m4,
      nvidia-h100-sm_90a, amd-mi325x-gfx942,
      nvidia-2xh100-sm_90a.par-devices-0-1,
      amd-2xmi300x-gfx942.par-devices-0-1}.json

`apple-m4.batch-sabotage.json` carries no cell for either, which is correct:
both lanes declare `PAR_BYTE_LM_BATCH_NA` and take no batch part.

## Owed, not done here

- `_backend.host_binding_path()` and `forest_host_binding_path()` build their
  path from `_pkg_dir()` directly, while `host_module_path()` goes through
  `host_dir()` and honors `MOJOLEARN_HOST_DIR`. So under that override the
  byte-LM INFERENCE door looks in the wrong directory, which is why the
  `byte-lm` infer, batch and rlpair parts read REFUSED in the run above while
  the train cell was STABLE. It did not affect this lane's answer and was left
  alone as outside these two lanes. Worth one line to whoever owns the host
  path helpers.

## Done on this branch

- The answer above, and wave 3's section (c) rewritten with it.
- No code change, no lane covered, no record touched.

## Resume steps for a session with none of this context

**1. Take a worktree and sync.**

    cd /Users/andrewhendel/CascadeProjects/mojolearn   # SHARED: never build or commit here
    git worktree add -b lane/cpu-verifier-byte-lm-pool-offload <scratch>/wt-bytelm origin/main
    cd <scratch>/wt-bytelm && git fetch origin main && git merge origin/main

**2. Reproduce the refusal (no build, no GPU, one core).**

    export PYTHONPATH=$PWD/python MOJOLEARN_NUMERIC_MODE=identical \
           MOJOLEARN_HOST_DIR=/Users/andrewhendel/CascadeProjects/mojolearn/python/mojolearn/host
    nice -n 19 python3 tools/identity_break.py \
      --lanes byte-lm,par-byte-lm,par-byte-lm-model-pool,par-byte-lm-offload \
      --fixtures base --repeats 1 --json /tmp/cpu-bytelm.json

Expect `byte-lm/base` STABLE and the three par lanes REFUSED by name. If a par
lane ever reads anything but REFUSED, that is a routing bug and
`tools/cpu_identity_gate_check.py column --covered <list>` already fails on
it, because an uncovered lane must read REFUSED.

**3. Do not reopen this without new code.** The answer changes only when a
host `ByteLayerPool` forward and backward exists. Worker count, pool
admission names and `host_surface.py` lane lists cannot change it.

## Gates run before push

    python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins .
    python3 packaging/wheel_ci.py inventory python/mojolearn

## Pods

None. Nothing was rented on this branch.
