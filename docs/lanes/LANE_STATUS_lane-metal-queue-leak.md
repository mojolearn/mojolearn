# lane/metal-queue-leak

Question: does mojolearn, or the Mojo runtime under it, leak Metal command
queues on macOS? On Sep 15 2026 the M4 ran GBDT Metal fits about 20x slow and
the kernel logged "Too many queues (N) created, possibly leaking?" and "The
number of queues (5041) exceeding limit (512), failing IOGPUCommandQueue
creation" during heavy Metal identity runs.

**Answer: no leak in our code was ever reproduced, and the accumulation we did
catch belonged to Apple system processes. This lane is CLOSED as a negative
result.** No fix was written, because nothing reproducible was found to fix.

Evidence outside the repo, survives a scratchpad wipe:
/Users/andrewhendel/mojolearn-evidence/metal-queue-leak-2026-09-15/queue-samples.txt
and gbdt-metal-health.log. Prior evidence:
/Users/andrewhendel/mojolearn-evidence/gbdt-metal-slowdown-2026-09-15/

## What was measured, and what held the queues

`ioclasscount AGXCommandQueue` is the kernel's live queue count. The per
process share is the count of `AppUsage` entries on the AGXDeviceUserClient
whose `IOUserClientCreator` names that pid.

| when | workload | kernel total | held by our process |
|---|---|---:|---:|
| Sep 15 19:29 | another lane's `mojo` process (pid 78701) | 1243, rising to 2491 | 1211 |
| Sep 15 19:39 | 5 GBDT fits, 2000 rows, one process | 34 flat | 34 flat, no growth |
| Sep 15 19:41 | 6 GBDT fits, 20000 rows, one process | 34 flat | 34 flat, no growth |
| Sep 16 05:52 to 05:59 | four consecutive lane jobs on the Metal slot | 22 to 26 | 0 |
| Sep 16 06:02 to 06:05 | `identity_break.py`, 7 GP lanes, repeats 2 (pid 66051) | 2105 rising to 4663 | **0 to 1** |

In that last row the climb was about 12 to 18 queues per second and
**4642 of 4663 queues belonged to pid 37288, `DockHelper`**
(`/System/Library/CoreServices/Dock.app/Contents/XPCServices/DockHelper.xpc`,
started 05:31:26, parent launchd). The seven-lane verification process running
on the GPU at the same moment held one queue, then zero.

The same shape appeared the night before: `VTDecoderXPCService` (VideoToolbox
decode, an Apple service) held 632 to 670 queues and gained about 1.8 a second
while no mojolearn GPU process was alive at all.

## What the 512 limit actually does

Measured on Sep 16 at 4663 queues, more than nine times the stated limit,
while a real GP identity run was in flight:

- kernel lines matching "too many", "exceed" or "fail" in the preceding four
  minutes: **0**
- refusal, error, traceback or NaN markers in that run's own log: **0**
- the run kept progressing normally (29.6% CPU, log still being written)

So the 512 figure is not a hard ceiling that silently corrupts cells, and there
is no evidence that a large identity run fails or fabricates results because of
queue pressure. The `Context leak detected, CoreAnalytics returned false` lines
that fill Metal logs are an Apple diagnostic present in committed evidence
since Sep 10 (`bench/results/classification_metrics_2026-09-10/`), unrelated to
queue counts.

## The one unreproduced sighting

Sep 15 19:29, pid 78701 named `mojo`, 1211 of 1243 queues, still climbing at
2491 a minute later, and down to 34 within about 15 seconds of that process
exiting. That observation is real and is recorded here, but nothing in a day of
measurement has reproduced it, and the two accumulations that were caught in
the act since both belonged to Apple system processes.

An object-lifetime explanation was offered for it during this lane: contexts
held as fields on model and pool objects (`core/forest_inference_model.mojo:88`
`self.ctx = DeviceContext()`, `core/forest_inference_pool.mojo:117`, the
`training/byte_lm_*` pools), with a long multi-lane process keeping many alive
at once. **The Sep 16 measurement contradicts it**: a seven-lane
`identity_break.py` process with repeats 2 held 0 to 1 queues, not hundreds.
That explanation should not be repeated as fact; if anyone wants to test it,
hold N fitted `parallel_groves` forests alive (the resident path in
`python/mojolearn/_forest_protocol.py` keeps one snapshot per estimator via
`forest_prepare_gpu`) and see whether queues track N.

## Where contexts are created in our code, for the next investigator

- The runtime creates one Metal command queue per `DeviceContext`
  (`libKGENCompilerRTShared.dylib` carries
  `MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.cpp` and "Failed to
  create Metal command queue for context"). No source ships.
- Every Python binding call builds a `DeviceContext` inside `GILReleased` and
  drops it at return, for example `bindings/_mojolearn_gbdt.mojo` at 405, 455,
  501, 545 and 601. Measured: this does NOT accumulate queues.
- Contexts held as object fields, listed above, are the only shape that could
  hold many at once. Untested against N.

## The counting trap

`ioreg -l -c AGXCommandQueue` does NOT list command queues. Queues are not
registry entries: it prints zero nodes of that class, and the
`IOUserClientCreator` lines in its output belong to other classes it walks
(IOHIDEventServiceUserClient 140, RootDomainUserClient 117, AGXDeviceUserClient
48). Counting those lines and subtracting from `ioclasscount` produces a table
that looks like queue attribution and is not one. An "about 6400 of 6754 queues
have no live creator, so queues outlive their process" conclusion came from
exactly that, and is false: queues are released when their process exits
(2491 down to 34 in about 15 seconds, no reboot, uptime 9 days).

## Operational conclusion

1. There is nothing to fix in mojolearn on this evidence. Do not write a
   context-reuse change against the per-call binding path; it was measured flat.
2. The Apple column's cost is not explained by queue pressure. A claimed
   decomposition attributing about 3.6 hours of it to this leak could not be
   found in any committed document across 227 remote branches, and no
   measurement here supports it.
3. Long recordings are still better run as several short processes, which is
   what the 0.8.6 Apple work ended up doing. Not because of queues, but because
   a short process is easier to retry, schedule against the Metal lock, and
   attribute when something goes wrong.
4. A no-reboot recovery exists and was observed twice: queues return to a floor
   of roughly 22 to 40 once the accumulating process exits. If the GPU feels
   slow, find the holder with the AppUsage attribution above before blaming our
   code, and expect it to be an Apple service.

## Tools left on this branch

- `tools/diag/metal_queue_leak.py`: samples both counters around N fits in one
  process, N predicts, N processes, or an already running pid (`--watch`).
- `checks/device_context_queue_repro.mojo`: creates, uses and drops one
  `DeviceContext` per iteration, `-D ONE_CTX=1` for the reuse arm. Built and
  runnable; never run to completion, because the premise collapsed first.
- `bench/results/classical_host/2026-09-15-apple-m4-gbdt-metal-baseline/`: the
  first GBDT Metal timings in this repo, taken on a known good machine.
