# Proposed macOS tiny-job supervisor

`tools/macos_serial_guard.py` has passed nine root-run mocked checks; no
supervised native build or Metal model has run. September 7 read-only OS
telemetry was readable, but the required launch reserve was absent.
[Retained preflight](../bench/results/resume/2026-09-07-root-metal-preparation/initial-preflight.json).
It is reserved for root's separately authorized tiny Metal job after the
NVIDIA/AMD prerequisites pass. It does not authorize local testing by agents.

The supervisor requires Darwin and acquires the shared root/NVIDIA/Metal locks
plus `/tmp/cbsym-build.lock`, coordinating with `tools/with_build_lock.sh`.
It never trusts an inherited `MOJOLEARN_BUILD_LOCK_HELD` flag to skip a lock;
only after acquisition does it set that flag for child build wrappers, avoiding
self-deadlock. It runs one child session/process group. Deadline is at most 180 seconds;
RSS cap is at most 4 GiB, default 2 GiB. All numerical-library and compiler
thread environment variables are set to two. Signal, monitoring failure,
deadline or resource refusal triggers process-group TERM then KILL. A separate
wall-timer thread signals the original group at the deadline and after a
two-second grace, without waiting for memory/process telemetry. It also tries
to kill observed escaped descendants after checking their current identities.
This is a same-process watchdog, not protection against supervisor SIGKILL,
interpreter failure or an OS that cannot schedule it.

Process snapshots retain observed ancestry and PID/start-time identities across
reparenting or PGID changes. Cleanup verifies the leader is reaped and no
observed live descendants remain; a leader exiting with live descendants is
not a successful job. Zombies are not treated as executing processes. If live
children persist or cleanup telemetry is unavailable, the guard keeps all
coordination locks held in a **cleanup-only quarantine**, retries KILL and
verification, and emits `CLEANUP_UNVERIFIED`. The payload deadline remains
bounded, but this supervisory wait can require operator intervention. Do not
kill that supervisor and assume its children were cleaned up. Optional JSON
output is a new exclusive file, finalized only after cleanup verification,
and records any quarantine/cleanup failures as a failed run.

Before launch it requires readable process, VM-page, pressure and swap counters,
normal memory pressure, and at least 4 GiB of free plus speculative memory.
During execution it refuses pressure above normal, reserve below 2 GiB,
RSS over the configured cap, swap growth over 128 MiB, or compressor growth
over 256 MiB. Free/speculative memory is a conservative reserve estimate;
inactive/compressed memory is not counted as available. Machines lacking
`kern.memorystatus_vm_pressure_level` fail closed. Root must inspect actual
telemetry availability before treating this as a usable local guard.

One `ps` process-table read and two memory queries are sampled roughly every
two seconds, each with a 1.5-second subprocess timeout and a one-MiB retained
output bound. There are no per-PID subprocess loops, GPU probes, package/model
imports or `system_profiler` calls. Aggregate CPU-time deltas divided by elapsed
wall time estimate cores used by the observed descendant set. More than three sampled
cores sustained for four seconds stops the job.

**No hard two-core affinity is claimed.** macOS has no `taskset` equivalent
used here; environment variables are requests. Sampling can miss brief CPU
bursts and CPU time from descendants that exit between snapshots. Group RSS
can double-count shared pages, and it is not a Metal allocation ledger. System
pressure/reserve also reflects other applications. These checks cannot prevent
instantaneous unified-memory allocations or exclude unrelated GPU users.
A child that escapes and is reparented before any snapshot can remain unknown;
short-lived workers can evade both ancestry discovery and CPU accounting. `ps`
start times distinguish observed PID reuse only to one-second precision, and
checking then signaling a PID is not an atomic kernel identity operation.
There is no claim to contain arbitrary daemonizing commands. Restrict launches
to reviewed, non-daemonizing tiny workloads; a killed supervisor still needs an
independent root cleanup procedure. Use a bounded single-step shape, not a compiler
stress test or broad campaign, and keep the root user's actual resource limits
stricter than this tool's maxima when appropriate.

No numerical qualification, successful local run, or immunity from host crashes
follows from this source. Root reviews and validates the supervisor before use.

Authored file-only mock checks are in `tools/test_macos_serial_guard.py`.
They cover observed session escape, PID reuse, lingering-child/telemetry cleanup
refusal, timer signal ordering and lock compatibility. No tests were executed
by subagents; root's earlier nine mocked checks passed. Mocks do not qualify
actual child supervision or Metal behavior. The newly added receipt policy
also requires explicit policy limits, clean verified cleanup, a nonexpired
watchdog and retained memory/CPU samples; old logs lacking those fields do
not gain retrospective admission. Root must test these additions separately.
