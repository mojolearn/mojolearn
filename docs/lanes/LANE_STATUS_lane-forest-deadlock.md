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

