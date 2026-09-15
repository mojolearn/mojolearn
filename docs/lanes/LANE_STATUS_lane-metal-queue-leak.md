# lane/metal-queue-leak

Question: does mojolearn, or the Mojo runtime under it, leak Metal command
queues on macOS? On Sep 15 2026 the M4 ran GBDT Metal fits about 20x slow and
the kernel logged about 147,000 IOGPUFamily lines, "Too many queues (N)
created, possibly leaking?" and "The number of queues (5041) exceeding limit
(512), failing IOGPUCommandQueue creation", during heavy Metal identity runs.

Evidence from before this lane: /Users/andrewhendel/mojolearn-evidence/gbdt-metal-slowdown-2026-09-15/
Evidence from this lane (outside the repo, survives a restart): /Users/andrewhendel/mojolearn-evidence/metal-queue-leak-2026-09-15/

## Status

Phase 1 (code reading, read-only counters, script) is done. Phase 2 (GPU
measurement on the clean GPU after the Mac restart) is NOT started.

## Phase 1 findings

### Where a command queue is created

- The Mojo runtime creates ONE Metal command queue per `DeviceContext`.
  `libKGENCompilerRTShared.dylib` carries `MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.cpp`,
  `newCommandQueue` and the error "Failed to create Metal command queue for
  context." No source is shipped; whether the context's destructor releases the
  queue cannot be read, only measured.
- Every Python binding call builds a fresh `DeviceContext` inside
  `GILReleased` and lets it go out of scope at the end of the call. GBDT:
  `bindings/_mojolearn_gbdt.mojo` `gbdt_fit_binding` (405),
  `gbdt_predict_binding` (455), `gbdt_predict_multi_binding` (501),
  `gbdt_fit_two_level_feature_freq_binding` (545), and
  `gbdt_fit_ordered_rmse_binding` (601, `with DeviceContext() as ctx`). The
  file's own docstring says "A `DeviceContext` is constructed per call". The
  same shape holds for every binding (constructions per file: training 14,
  estimators 13, `_mojolearn` 10, gbdt 5, byte_lm 4, trees/solver/rf/mamba 3,
  transformer/metrics/hdbscan/embedding 2, linalg/ivf 1).
- Contexts built inside a fit: `gbdt/binary_prediction.mojo:49`, and one per
  shard in `gbdt/methods/pointwise_multi_gpu.mojo:90` and
  `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:6001`
  (`DeviceContext(device_id=rank)`; on the Mac only rank 0, and only when the
  multi-GPU path is taken). Long-lived holders: `core/forest_inference_model.mojo:88`,
  `core/forest_inference_pool.mojo:117`, the `training/byte_lm_*` pools.
- No Python module constructs a context; Python reaches the GPU only through the
  bindings.
- `tools/identity_break.py` runs every lane, fixture and repeat (fit, infer,
  model reload, batch probes) in ONE Python process, so one identity run is
  hundreds to thousands of binding calls, hence contexts, in one process. That
  is the long-lived-process shape a per-context leak would punish.

### Leak hypothesis

H1 (ours or the runtime's): a `DeviceContext` going out of scope does not
release its `MTLCommandQueue`, so a long-lived process gains at least one
queue per binding call and, past about 512, queue creation fails or slows.

H2 (not ours): another process holds the queues. At 18:25 ET on Sep 15, with
NO mojolearn GPU process alive, `ioclasscount AGXCommandQueue` read 471, then
543, 579, 615 over a few minutes. Almost all of them (557, then 593) sit on
the AGXDeviceUserClient created by pid 88848 `VTDecoderXPCService` (Apple's
VideoToolbox decode service, started 17:29:18), growing about 1.8 queues per
second, in lockstep with the kernel total. Samples in `queue-samples.txt` in
the evidence directory. Its client was not identified (its own log is empty).
A system video decode service that is leaking queues is a live candidate for
the Sep 15 kernel messages and the slowdown, independent of mojolearn.

The kernel log lines themselves were no longer retrievable (`log show` over 12h
returned none), so the 5041 count cannot be attributed to a pid from here.

Neither hypothesis excludes the other; phase 2 decides H1.

### Measurement: `tools/diag/metal_queue_leak.py`

Two counters, both read without touching the GPU:
- system: `ioclasscount AGXCommandQueue` (all processes; noisy, see H2);
- process: the count of `AppUsage` entries on the AGXDeviceUserClient whose
  creator is the measured pid. On Sep 15 this matched the kernel total step for
  step (+36 and +36 in 20 s), so it is the per-process queue count, immune to
  other processes.

Arms: `inproc` (one child, N sequential GBDT fit plus predict, sampled every
`--every` fits, then system count after exit), `ctxonly` (one fitted model,
N small predicts), `subproc` (N processes, one fit each). Each prints a table
and a `VERDICT` line: process and system queues per fit, first and last fit
seconds, whether the system count returned to baseline.

## Phase 2 plan (after the restart)

1. Before anything: `ioclasscount AGXCommandQueue` and the per-pid table
   (script's `process_queues`), with no GPU job running. If VTDecoderXPCService
   is climbing again, note which app is playing or capturing video, and quit it.
2. Build nothing in the shared checkout. Use a built package (the shared
   checkout's `python/` if its GBDT binding is current, otherwise build in this
   worktree with `bindings/build_gbdt.sh`).
3. Run alone on the GPU:
   `bash $SP/mac_slot.sh metal .pixi/envs/test/bin/python tools/diag/metal_queue_leak.py --pkg <pkg> --arm inproc --fits 200 --every 20`,
   then `--arm ctxonly --fits 200`, then `--arm subproc --fits 20 --every 5`.
   If the scratchpad helper is gone: `nice -n 19` one process, after
   `pgrep -fl mojo` shows no other GPU job.
4. If process queues per fit is about 0 and the system count returns to
   baseline: mojolearn does not leak; the Sep 15 queues were another process
   (H2). Record and close.
5. If process queues grow per fit: write a minimal Mojo reproduction (a loop
   creating and dropping `DeviceContext()` with one `synchronize`, counting
   AppUsage entries), to tell the runtime from our bindings. If the runtime
   leaks, write a report for Modular (do not send) and add a workaround: one
   process-lifetime context per binding module, reused across calls (GPU
   agnostic, `max.gpu.host` only).
6. Prove any fix: process queues flat over 200 fits, per-fit seconds stable,
   and a base-fixture `identity_break.py` spot check of a few GBDT lanes against
   the committed Apple column. Then `python3 tools/docs_facts.py --check`,
   `python3 packaging/wheel_ci.py pins .`, merge, push HEAD:main.
