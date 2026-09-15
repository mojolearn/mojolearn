# lane/metal-queue-leak

Question: does mojolearn, or the Mojo runtime under it, leak Metal command
queues on macOS? On Sep 15 2026 the M4 ran GBDT Metal fits about 20x slow and
the kernel logged about 147,000 IOGPUFamily lines, "Too many queues (N)
created, possibly leaking?" and "The number of queues (5041) exceeding limit
(512), failing IOGPUCommandQueue creation", during heavy Metal identity runs.

Evidence from before this lane: /Users/andrewhendel/mojolearn-evidence/gbdt-metal-slowdown-2026-09-15/
Evidence from this lane (outside the repo, survives a scratchpad wipe): /Users/andrewhendel/mojolearn-evidence/metal-queue-leak-2026-09-15/queue-samples.txt

## Status

Phase 1 is done, and it answered the question without this lane running a
single GPU job: a live `mojo` process was caught holding 1211 of the machine's
1243 command queues, and the count fell to 34 within about 15 seconds of that
process exiting. A health check then measured GBDT Metal fits at about 2.7x
faster than the degraded afternoon numbers, with the queue count flat at 34
across eleven fits, so the degraded state is gone and no reboot is needed.
Phase 2 (finding which workload shape does accumulate queues, and the fix) has
not started.

## The finding

At 19:29 on Sep 15, with this lane doing nothing on the GPU,
`ioclasscount AGXCommandQueue` read 1243 and the AppUsage entries showed
**1211 of them held by one live process, pid 78701, named `mojo`** (another
agent's lane run). A minute later the total was 2491, still climbing. When that
process exited, the count fell to 34 within about 15 seconds and stayed near 35
to 40.

1. **A single Mojo process accumulates command queues as it runs**, over 1200
   in one process against the kernel's stated limit of 512. That is the
   mechanism behind the Sep 15 slowdown: a long-lived process
   (`tools/identity_break.py` runs every lane, fixture and repeat in ONE
   process, and every binding call builds a `DeviceContext`) crosses the limit
   inside its own lifetime, and queue creation begins to fail or crawl.
2. **The queues are not leaked past process exit.** They come back when the
   process dies. This first suggested that reusing one context per process
   would be a real fix. The health check below then ruled that out for the
   binding path: eleven GBDT fits in two processes left the count flat at 34,
   so the per-call `DeviceContext` is not what accumulates.

## How to count queues, and the trap that produced the opposite conclusion

- `ioclasscount AGXCommandQueue` is the kernel's live queue count, all
  processes. This is the real total.
- The `AppUsage` entries on each AGXDeviceUserClient, keyed by the client's
  `IOUserClientCreator` pid, give the per-process share. Cross-checked at both
  ends of the range: a climbing process moved its own entries and the kernel
  total by exactly 36 in 20 s; at the quiet floor the entries summed to 38
  against a total of 40; and the 1211 above sat inside a total of 1243.
- THE TRAP: **`ioreg -l -c AGXCommandQueue` does not list command queues.**
  Measured here on Sep 15: it prints **0** nodes of class AGXCommandQueue,
  because queues are not IORegistry entries. The `IOUserClientCreator` lines in
  its output belong to unrelated classes that the command also walks
  (IOHIDEventServiceUserClient 140, RootDomainUserClient 117,
  AppleKeyStoreUserClient 79, AGXDeviceUserClient 48). Counting those lines and
  subtracting from `ioclasscount` produces an "attribution" in which most
  queues appear to name no live creator, and so appear to outlive their
  process. That is an artifact of comparing two different populations. The
  direct observation above (2491 down to 34 at process exit, no reboot)
  contradicts it, so this lane does not carry the outliving-processes claim.

## Pre-restart baseline (Sep 15, read only, no GPU work by this lane)

| reading | time (EDT) | AGXCommandQueue | attributed to live clients | live mojo GPU processes |
|---|---|---:|---:|---|
| 1 | 19:33:06 | 40 | 38 | none |
| 2 | 19:37:22 | 41 | 39 | none |

With no mojolearn GPU process running, the machine sits at a floor of roughly
35 to 41 queues and is not climbing: one queue in 4 minutes 16 seconds, which
is ordinary desktop churn, against the 1.8 per second seen earlier while a
process was accumulating them. Uptime at the baseline was 9 days, 11:53
(boot Sep 6), so the collapse from 6754 to this floor happened with NO reboot.

Earlier points, same day, same counter: 471 to 615 at 18:25 (no mojolearn
process alive); 654 to 692 at 18:27, of which `VTDecoderXPCService` held 632 to
670, gaining about 1.8 a second; 6591 to 6754 later (coordinator), where
`killall VTDecoderXPCService` gave no immediate drop but the count was back to
127 and then 35 by 19:27.

## Health check, Sep 15 19:39 to 19:42 EDT, alone under the Metal lock

Two runs, identical mode, Metal, one core, `nice -n 19`, the shared checkout's
built bindings.

Small fixture, 2000 x 8, 20 trees, depth 6, five fits: 7186, 7067, 6747, 7000
and 7160 ms.

Like for like against the degraded evidence, the same script
(`/Users/andrewhendel/mojolearn-evidence/gbdt-metal-slowdown-2026-09-15/gbdt_direct.py`)
at 20000 rows:

| policy | now | degraded, Sep 15 afternoon |
|---|---|---|
| SymmetricTree | 8.289, 7.735, 8.288 s | 21.5 to 23.4 s |
| Depthwise | 11.848, 14.082, 13.373 s | 27.0 to 30.4 s |

**The degraded state is gone**: about 2.7x faster on SymmetricTree and about
2.2x on Depthwise, and nothing resembling the 20x slow fits. What cannot be
claimed from here is a clean bill of health against "about 1 s per fit": no
GBDT Metal timing from before the slowdown is committed anywhere in this repo,
so that figure has no anchor and 7 to 8 s is neither proven healthy nor proven
slow. Establishing a real baseline for this shape is separate work.

**Queues stayed at 34 throughout**: before each run, after every fit, the
moment each process exited, and 20 to 30 s later. Eleven fits across two
processes added none. So the per-call `DeviceContext` in the GBDT binding path
does NOT accumulate queues in this build, and it is not what filled pid 78701.
Whatever that process was doing (long-lived contexts held concurrently, a
check binary, or another lane's shape) is what phase 2 has to find.

## Where a command queue is created in our code

- The Mojo runtime creates ONE Metal command queue per `DeviceContext`.
  `libKGENCompilerRTShared.dylib` carries
  `MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.cpp`, `newCommandQueue`
  and "Failed to create Metal command queue for context." No source ships, so
  whether dropping a context releases its queue can only be measured.
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
  bindings.

## What phase 2 would have to measure, and why. NOT started

This lane is CLOSED at the finding above. Andrew has said no new lanes, so
nothing below has been run and no binding was touched. It is written down so
whoever picks this up does not repeat the two dead ends this lane already
paid for.

**The obvious fix is ruled out by measurement.** Reusing one `DeviceContext`
per process, instead of the per-call one every binding builds, would fix
nothing measurable: the queue count sat flat at 34 across eleven GBDT fits in
two processes, before, during and after each. The per-call path in the GBDT
binding does not accumulate queues in this build. Do not start there.

**The open question is which workload shape does accumulate.** pid 78701, a
`mojo` process, held 1211 of 1243 queues and was still climbing at 2491. Its
shape is unknown; GBDT fits through the Python binding are excluded. The
candidates worth measuring first, in order:

1. Contexts held CONCURRENTLY rather than created and dropped in sequence. A
   per-shard `DeviceContext(device_id=rank)` inside a fit, or a pool holding
   several at once, keeps N queues alive where the binding path keeps one.
   `core/forest_inference_pool.mojo:117`, `training/byte_lm_*` pools and the
   two multi-GPU shard sites listed above are the places to instrument.
2. A native `mojo run` or check binary rather than the Python bindings, since
   the process caught accumulating was named `mojo`, not `python`.
3. Long single processes that do thousands of calls, such as
   `tools/identity_break.py` over every lane, fixture and repeat, which is
   what was running when the kernel first complained.

The reproduction `checks/device_context_queue_repro.mojo` stays on this branch
for that work: it creates, uses and drops one `DeviceContext` per iteration,
with `-D ONE_CTX=1` for the reuse arm, so it separates the runtime's behavior
from any of our call shapes. `tools/diag/metal_queue_leak.py` samples both
counters around any of it, and its docstring carries the counting trap.

The runbook below is kept only as reference for that future work. `$REPO` is
the shared checkout `/Users/andrewhendel/CascadeProjects/mojolearn`; never
build, commit or switch branches there. Work in a worktree of this branch,
`lane/metal-queue-leak`. One GPU job at a time: wrap every GPU command in
`bash $SP/mac_slot.sh metal <command>` where `$SP` is the session scratchpad,
or, if that helper is gone, run one `nice -n 19` process at a time after
`pgrep -fl mojo` shows no other GPU job.

**Step 0, floor (optional, no reboot required).** `ioclasscount AGXCommandQueue`
with no GPU job running. Expect roughly 35 to 41. Write it down; every later
number is a delta from it. A reboot is NOT a precondition for any step here:
queues are released when their process exits, so the floor returns on its own
once no accumulating process is alive.

**Step 1, health check (do this first, it is also the 20x regression test).**

    cd <worktree> && MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$REPO/python \
      $REPO/.pixi/envs/test/bin/python -c "
    import time, numpy as np, mojolearn as ml, mojolearn._backend as b
    print(b.vendor(), b.numeric_mode())
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 8)).astype(np.float32)
    y = (X[:, 3] + 0.5 * X[:, 4] > 0).astype(np.int32)
    for r in range(5):
        t = time.perf_counter()
        m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss='Logloss').fit(X, y)
        print('fit', r, round(time.perf_counter() - t, 3), 's', flush=True)
    "

Compare against the measured baseline in
`bench/results/classical_host/2026-09-15-apple-m4-gbdt-metal-baseline/gbdt_metal_fit_times.txt`:
6747 to 7186 ms per fit on this 2000 row fixture, taken once the machine was
known good. That is a FIRST baseline, not a proven healthy figure. The "about
1 s per fit" figure that circulated on Sep 15 is UNVERIFIED and must not be
quoted; no GBDT Metal timing from before the slowdown exists anywhere in this
repo. The degraded state that started this lane read 21.5 to 23.4 s per fit on
the 20000 row fixture, against 7.7 to 8.3 s in that baseline.

**Step 2, one fit, queues created and released.** In one terminal run the
health check again; in another, before, during and 30 s after it:

    ioclasscount AGXCommandQueue

Record the floor, the peak during the run, and the value 30 s after the process
exits. Queues created by the fit equal peak minus floor; queues released at
exit equal peak minus the after value. The kernel reclaims lazily, so wait at
least 30 s before calling anything retained.

**Step 3, N fits in ONE process, the real question.**

    bash $SP/mac_slot.sh metal $REPO/.pixi/envs/test/bin/python \
      tools/diag/metal_queue_leak.py --pkg $REPO/python --arm inproc --fits 200 --every 20

This samples the kernel total and the child process's own share every 20 fits,
then again after the child exits, and prints a VERDICT line with queues per fit
and what was left behind. `--arm ctxonly --fits 200` repeats it with predicts
instead of fits; `--arm subproc --fits 20 --every 5` uses one process per fit,
which should stay flat if queues are released at exit. The counter helpers and
the trap above are documented in that file's docstring.

**Step 4, runtime or bindings.** Build the minimal reproduction, which contains
no estimator, only a loop creating, using and dropping one `DeviceContext`:

    mojo build -I . checks/device_context_queue_repro.mojo -o /tmp/qrepro
    ITERS=200 /tmp/qrepro            # sample with --watch <pid> from the script
    mojo build -I . -D ONE_CTX=1 checks/device_context_queue_repro.mojo -o /tmp/qrepro_one

Per-iteration contexts growing while `ONE_CTX=1` stays flat would mean the
runtime allocates a queue per context and does not release it inside the
process. Note what this lane already measured: through the GBDT binding, which
builds a context per call, the count did not grow at all. So if this arm shows
growth, the difference between it and the binding path is the thing to explain,
and reuse is a candidate for the accumulating shape only, not for the binding
layer.

**Step 5, a fix, only once a shape is caught accumulating.** Whatever that
shape turns out to be, the change belongs where the contexts are held, not in
the per-call binding path this lane cleared. Any candidate fix must be proved
with queues flat over the accumulating workload, per-fit seconds no worse than
the baseline file, and a base-fixture `tools/identity_break.py` spot check of a
few GBDT lanes against the committed Apple column, then
`python3 tools/docs_facts.py --check` and `python3 packaging/wheel_ci.py pins .`.

**Step 6, if the runtime accumulates a queue per context regardless of our
shape**, write the report for Modular from the reproduction's numbers. Do not
contact anyone.

What NOT to carry forward: the claim that a process making thousands of binding
calls will cross the 512 queue limit inside its own lifetime. This lane
measured eleven GBDT fits, each with several binding calls, and the count never
moved off 34. Splitting long Metal runs into smaller processes is therefore not
a justified workaround on this evidence.
