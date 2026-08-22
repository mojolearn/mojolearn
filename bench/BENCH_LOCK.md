# THE BENCH LOCK -- the one canonical description

Codified 2026-08-22 night by the orchestrator from the mechanics the
ensemble and depthwise lanes built and amended that day. The deeper
rationale lives in `ensemble/bench/quiet_window.py`'s docstrings and is
CITED here, not duplicated. Questions and window requests route to the
ORCHESTRATOR session (charter in its persistent memory); this file is the
mechanics.

## Substrate

`tools/bench_lock.sh` over `/tmp/cbsym-bench.lock`:

    bench_lock.sh acquire "<session>" "<what>" "<eta>"
    bench_lock.sh release
    bench_lock.sh status

Metadata (session / what / eta / pid) is advisory text written at
acquisition. `BENCH_LOCK_PID` must name a process that OUTLIVES the
window: a per-call shell dies in milliseconds and leaves a lock reporting
STALE-PID forever. **STALE-PID is NOT evidence the box is free** -- a
human overriding on that reading caused three retracted results on
Aug 21.

## Gating: `ensemble/bench/quiet_window.py run -- <cmd>`

1. Acquires the lock with its own pid, releases at exit;
   `--ignore-lock` for a caller that already holds it.
2. PRE-OPEN scan: refuses to open if any process is >=50% CPU
   (`BUSY_PCPU=50`; `NOTE_PCPU=20` is report-only). A refusal costs
   seconds and burns no window.
3. In-window sampler every 2s, excluding its own process tree
   (recomputed per sample). VOID if any process hits >=50% in >=2
   SEPARATE samples (`BUSY_MIN_SAMPLES=2`: `ps` %cpu spikes; one
   sighting is not competition).
4. **THE CANARY IS LOAD-BEARING.** A fixed GPU fit brackets every arm
   (pre/mid/post, plus a discarded `warmup` tag absorbing the
   idle-to-active clock ramp). Ticks group per `binary:` tag -- an A/B's
   two builds are DIFFERENT canaries. The node-count self-check must be
   constant per group. The floor is the WORST group's spread, and it is
   not a pass mark: it is the window's RESOLUTION -- any ratio inside it
   is indistinguishable. Spread >1.5x is gross contamination: VOID.
5. Voided or refused ARM numbers are NEVER quoted. Consistency across
   contaminated windows may be RECORDED as consistency, never as
   measurement.

## Etiquette (agreed by all lanes, 2026-08-22)

* Timing windows take the lock. A window-hunting LOOP holds the lock
  across the ENTIRE hunt (the day's amendment: per-window acquire reads
  FREE between attempts, and legal bursts land mid-hunt).
* Every lane checks `status` before a compile or run. Compiles batch
  into ANNOUNCED bursts -- `mojo build` pegs cores for a minute-plus --
  never trickle.
* Correctness gates and accuracy-only runs are exempt (their verdicts do
  not care about load) EXCEPT while the lock is held by someone else.
* Releases are explicit done-pings, never silent timeouts (two
  fifteen-minute-assumption failures on Aug 22).
* Scheduled/overnight windows pre-register with the orchestrator: start
  time + worst-case duration.
* Known standing contender: `aa_floor.core` (Andrew's backblaze job,
  respawning). No lane touches it; schedule around it or escalate to
  Andrew.

## Routing

Quiet-window requests, grants, and done-pings go through the
orchestrator by SendMessage; the lock is the machine-readable half, the
ping is the human-auditable half, and a window without both is a window
someone else will land in the middle of.
