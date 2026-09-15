# lane/metal-queue-leak

Question: does mojolearn, or the Mojo runtime under it, leak Metal command
queues on macOS? On Sep 15 2026 the M4 ran GBDT Metal fits about 20x slow and
the kernel logged about 147,000 IOGPUFamily lines, "Too many queues (N)
created, possibly leaking?" and "The number of queues (5041) exceeding limit
(512), failing IOGPUCommandQueue creation", during heavy Metal identity runs.

Evidence from before this lane: /Users/andrewhendel/mojolearn-evidence/gbdt-metal-slowdown-2026-09-15/
Evidence from this lane (outside the repo, survives a restart): /Users/andrewhendel/mojolearn-evidence/metal-queue-leak-2026-09-15/

## Status

Phase 1 (code reading, read-only counters, script and reproduction) is done.
Phase 2 (measurement on a quiet GPU) is NOT started, and waits for the
coordinator. The machine has NOT restarted, and on the evidence below it may
not need to.

## How to count queues (two counters, one trap)

- `ioclasscount AGXCommandQueue` is the kernel's live queue count, all
  processes. This is the real total.
- The `AppUsage` entries on each AGXDeviceUserClient, keyed by the client's
  `IOUserClientCreator` pid, give the per-process share. Checked at both ends
  of the range: while one process gained queues, its AppUsage entries and the
  kernel total both rose by exactly 36 in 20 s (557 to 593, 579 to 615); at the
  quiet floor the AppUsage entries summed to 33 against a total of 35.
- THE TRAP: `ioreg -l -c AGXCommandQueue` does NOT list command queues. Queues
  are not registry entries, so that command prints zero AGXCommandQueue nodes,
  and the `IOUserClientCreator` lines in its output belong to unrelated classes
  (IOHIDEventServiceUserClient 140, RootDomainUserClient 117,
  AppleKeyStoreUserClient 79, AGXDeviceUserClient 48). Counting those lines
  produces a table that looks like queue attribution and is not one. An earlier
  reading of this lane, and the "about 300 of 6754 queues name a live process,
  the other 6400 name none" figure, came from that command and do not show that
  queues outlive their creator.

## What the counters actually showed (Sep 15, read only, no GPU work)

| time  | AGXCommandQueue | note |
|---|---:|---|
| 18:25 | 471 to 615 | climbing with NO mojolearn GPU process alive |
| 18:27 | 654 to 692 | `VTDecoderXPCService` (pid 88848) held 632 to 670 of them, gaining about 1.8 a second |
| later | 6591 to 6754 | coordinator, still climbing |
| after `killall VTDecoderXPCService` | 6591, 6631 at +5 s, 6754 at +60 s | no immediate drop |
| 19:27 | 127, then 35 | back to a normal floor, with NO reboot (uptime 9 days, 11:49; boot Sep 6) |

Reading: the queues were reclaimed after the holding process died, but LAZILY,
minutes later, not at exit. The coordinator's samples at 5 s and 60 s were too
early to see it. The machine returned to a 35 queue floor on its own.

So the large pileup tracked one Apple system service
(`VTDecoderXPCService`, VideoToolbox decode, started 17:29), which is not
mojolearn and which nothing in this repo drives. The 5041 kernel message could
not be tied to a pid: `log show` over 12 h no longer returns those lines.

## Where a command queue is created in our code

- The Mojo runtime creates ONE Metal command queue per `DeviceContext`.
  `libKGENCompilerRTShared.dylib` carries `MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.cpp`,
  `newCommandQueue` and the error "Failed to create Metal command queue for
  context." No source ships, so whether dropping a context releases its queue
  can only be measured.
- Every Python binding call builds a fresh `DeviceContext` inside `GILReleased`
  and drops it when the call returns: `bindings/_mojolearn_gbdt.mojo` at 405
  (fit), 455 (predict), 501 (predict multi), 545 and 601. That file's docstring
  says "A `DeviceContext` is constructed per call". Constructions per binding
  file: training 14, estimators 13, `_mojolearn` 10, gbdt 5, byte_lm 4,
  trees/solver/rf/mamba 3, transformer/metrics/hdbscan/embedding 2, linalg/ivf 1.
- Inside a fit: `gbdt/binary_prediction.mojo:49`, and one per shard in
  `gbdt/methods/pointwise_multi_gpu.mojo:90` and
  `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:6001`. Long
  lived holders: `core/forest_inference_model.mojo:88`,
  `core/forest_inference_pool.mojo:117`, the `training/byte_lm_*` pools.
- No Python module builds a context; Python reaches the GPU only through the
  bindings. `tools/identity_break.py` runs every lane, fixture and repeat in
  ONE process, so one identity run makes hundreds to thousands of contexts.

## Hypotheses going into phase 2

H1, ours: a dropped `DeviceContext` does not release its queue, so a long-lived
process gains a queue per binding call. Untested. The `inproc` arm and the
reproduction measure it directly. Nothing so far either supports or refutes it,
because no mojolearn process was sampled while fitting.

H2, not ours: the Sep 15 pileup belonged to `VTDecoderXPCService`. Supported by
the table above, and by the count returning to 35 once that process was gone.

H3, kernel reclaim is lazy: confirmed. Queues counted minutes after their
process died were freed later without a reboot. Phase 2 must therefore wait
generously after a process exits before calling anything a leak.

## Phase 2 plan (only when the coordinator says go)

1. Record the floor first: `ioclasscount AGXCommandQueue` with no GPU job
   running.
2. Build `checks/device_context_queue_repro.mojo` in this worktree (never the
   shared checkout) and run `ITERS=200` under
   `tools/diag/metal_queue_leak.py --watch <pid>`, then the `-D ONE_CTX=1`
   build the same way. Three outcomes: flat (the runtime reuses one queue),
   grows then falls back after exit (a per-process leak that context reuse
   bounds), grows and stays for a long time after exit (kernel side).
3. Then the estimator arms alone on the GPU via `mac_slot.sh metal`, or
   `nice -n 19` once `pgrep -fl mojo` shows no other GPU job:
   `--arm inproc --fits 200 --every 20`, `--arm ctxonly --fits 200`,
   `--arm subproc --fits 20 --every 5`.
4. If our bindings add queues per call, reuse one context per process (a module
   level context in each binding, `max.gpu.host` only, GPU agnostic) and
   re-measure. Say plainly whether that stops the growth or only slows it.
5. If the runtime leaks regardless of our shape, write the report for Modular
   from the reproduction's numbers. Do not contact anyone.
6. Prove any fix: queues flat over 200 fits, per-fit seconds stable, and a
   base-fixture `identity_break.py` spot check of a few GBDT lanes against the
   committed Apple column. Then `python3 tools/docs_facts.py --check`,
   `python3 packaging/wheel_ci.py pins .`, merge and push HEAD:main.
