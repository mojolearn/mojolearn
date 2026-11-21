# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Small rows are the median of consecutive calls inside the one scored run (AGENTS.md "Measurement",
Andrew, 2026-11-15).

A row whose FIRST SCORED call of OUR arm (the first call after the excluded warmup) took under
SMALL_ROW_MS is scored as the median of that call and SMALL_ROW_CALLS - 1 more consecutive calls of the
same operation on the same inputs, made inside that one scored run. The warmup call does not decide: it
carries CUDA/HIP start-up and kernel compilation (neural/mamba3-forward: warmup 1051 ms, steady call
5.7 ms, so a warmup rule never fired; harness-small-row, 2026-11-16). The cell records which rule applied
(`scoring`: SCORING_MEDIAN or SCORING_ONE), the number of calls and their spread (`calls`, `min_ms`,
`max_ms`); `ms` (and the board's `median_ms`) is that median, so every reader of `ms` is unchanged. Rows
whose first scored call is at or above SMALL_ROW_MS keep that one call. Every arm of a compare (the
candidate and the prior wheels re-timed on the same box) runs this same code, so each is scored by the
same rule; tools/paper_compare.py flags a row whose two arms landed on different rules (SCORING-MIXED,
a first call on either side of SMALL_ROW_MS; the flag replaces that column's REGRESSION / FASTER, because
the first scored call carries the first-touch faults the median does not). Opponent arms are never repeated (their stored cells
stay). One scored run per arm is unchanged and nothing is rerun. Reason: a single call of a millisecond-scale row is dominated by
first-touch page faults and the allocator (program-run 3.4 ms once vs 0.76 ms steady).

The identity hash, the saved outputs and the state receipt of a repeated run come from the FIRST call's
output (the same bits every call); the receipt is still taken after the clock.

Benchmark glue only, never product runtime. The N calls are serial by nature: they are the timed
repetitions of one operation on one device, and running them concurrently would time contention.
"""
import math
import os
import statistics
import sys

#: a first scored call of our arm under this many milliseconds makes its scored run a median of calls
SMALL_ROW_MS = 10.0
#: the calls of one scored run of a small row (the first scored call included)
SMALL_ROW_CALLS = 20
#: the rule a scored run was taken under, recorded on the cell (`scoring`)
SCORING_MEDIAN = "median-of-calls"
SCORING_ONE = "one-call"


#: ATTRIBUTION RUNS ONLY (tools/ops/attribution_job.sh, 2026-11-16): our arm's scored run makes exactly this many
#: calls whatever its first call took (1 is allowed: the job's differential profile subtracts a one-call run from a
#: 20-call run), on the phase FORCE_CALLS_PHASE_ENV names (`fit`, `infer`, or `infer/<batch>`; unset: every phase).
#: A board never sets it.
FORCE_CALLS_ENV = "MOJOLEARN_BENCH_FORCE_CALLS"
FORCE_CALLS_PHASE_ENV = "MOJOLEARN_BENCH_FORCE_CALLS_PHASE"
#: BOARD ORDER (lane fx-harness, 2026-11-18): the forced phase `fit+infer` forces BOTH phases to the same count, so
#: the window holds pairs in the board's order: a fit, then the FIRST predict right after it (the worker interleaves
#: them). Consecutive forced predicts cannot show a cost only the first predict after a large fit pays (mlp istella
#: infer: the executor pool sits at its 3 GiB bound after the fit; ideas-unexplained-rows-RESULT.md section 2).
BOARD_ORDER = "fit+infer"


def forced_calls(ours, phase=None, environ=None):
    """The forced call count of an attribution run (FORCE_CALLS_ENV), or None when it does not apply: not our arm,
    unset, not a positive integer, or another phase than FORCE_CALLS_PHASE_ENV names (`infer` matches every
    `infer/<batch>`; a site that passes no phase matches only an unset phase filter)."""
    env = os.environ if environ is None else environ
    raw = (env.get(FORCE_CALLS_ENV) or "").strip()
    if not ours or not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n < 1:
        return None
    want = (env.get(FORCE_CALLS_PHASE_ENV) or "").strip()
    if want == BOARD_ORDER:
        # board order: the fit, every inference batch and the pair itself
        return n if phase is not None and (phase in ("fit", BOARD_ORDER) or phase.startswith("infer")) else None
    if want and (phase is None or (phase != want and not phase.startswith(want + "/"))):
        return None
    return n


def paired_calls(fit_ms, infer_ms, ours, environ=None):
    """BOARD ORDER: the calls of each phase of a forced `fit+infer` run (the same count for both, sized by the time of
    the first scored pair, fit plus inference), or None when the run is not one."""
    env = os.environ if environ is None else environ
    if (env.get(FORCE_CALLS_PHASE_ENV) or "").strip() != BOARD_ORDER:
        return None
    forced = forced_calls(ours, BOARD_ORDER, env)
    if forced is None:
        return None
    try:
        pair = float(fit_ms or 0.0) + float(infer_ms or 0.0)
    except (TypeError, ValueError):
        pair = 0.0
    return profile_calls(forced, pair if pair > 0 else None, env)


#: ATTRIBUTION PROFILE RUNS ONLY: the NVTX range name the job's `nsys profile --capture-range=nvtx` waits for, and a
#: `pip --target` directory holding the `nvtx` package (the measured venvs stay untouched). Unset: no-ops.
NVTX_RANGE_ENV = "MOJOLEARN_BENCH_NVTX_RANGE"
NVTX_PATH_ENV = "MOJOLEARN_BENCH_NVTX_PATH"
#: CALL PROFILE RUNS ONLY (tools/ops/call_profile_job.sh, 2026-11-17; fb15-profiler 2026-11-18). The same window
#: (calls 2..n of a forced scored run) is also: a JSON line file the window's open and close are appended to (wall
#: clock, calls, pid, the shim's counters); a perf control FIFO pair `<ctl>,<ack>` (`perf record -D -1 --control
#: fifo:<ctl>,<ack>`: perf samples only inside the window); a failed `stat` of a marker path at open and close, which
#: `strace -ttt` records, so the trace is cut to the window; the LD_PRELOAD timing shim (tools/ops/call_profile_shim.c:
#: CUDA driver entries, libc allocations, mmap, big copies; counts, ns and bytes) read at open and close; py-spy
#: attached to this process for the window only (`<py-spy>,<out>,<rate>`: spawned at the open, before the clock
#: starts, stopped with SIGINT after it stops; the process first allows its child to trace it, prctl
#: PR_SET_PTRACER_ANY, for a yama ptrace_scope of 1); and the callgrind client requests (tools/ops/call_profile_cg.c:
#: instrumentation and collection on for the window only). PROFILE_SECONDS sizes the forced run by time: the forced
#: phase makes at least ceil(seconds / first scored call) calls (capped by PROFILE_MAX_CALLS), and the phase that is
#: NOT forced makes one call, so the window holds the forced phase alone (bench_board_algos interleaves fit and
#: inference calls). A board never sets any of them.
PROFILE_WINDOW_ENV = "MOJOLEARN_BENCH_PROFILE_WINDOW"
PERF_CTL_ENV = "MOJOLEARN_BENCH_PERF_CTL"
SHIM_ENV = "MOJOLEARN_BENCH_CALLPROF_SHIM"
PYSPY_ENV = "MOJOLEARN_BENCH_PYSPY"
CALLGRIND_ENV = "MOJOLEARN_BENCH_CALLGRIND"
PROFILE_SECONDS_ENV = "MOJOLEARN_BENCH_PROFILE_SECONDS"
PROFILE_MAX_CALLS_ENV = "MOJOLEARN_BENCH_PROFILE_MAX_CALLS"
#: ONE-CALL PROFILE (lane fx-profile-big-rows, 2026-11-20; CALL PROFILE RUNS ONLY, a board never sets it): for rows
#: whose one call takes minutes (lle, isomap, dbscan istella: 250-1,000 s), `1` makes the forced phase's run ONE call
#: in all, with the window around that FIRST scored call (range_open_one), and no warmup round (first_round,
#: fit_warmup_skipped): one call per tool instead of warmup + first call + window. The call carries what a first call
#: carries (module load, driver JIT cache, pool growth); the report sets its wall beside the stored board time. The
#: window's open is also written at the open (`open-mark`, with the wall clock), so a run the job kills at its cap
#: still says when the window opened. Fit or infer phases of the algos, classical and tree drivers.
PROFILE_ONE_CALL_ENV = "MOJOLEARN_BENCH_PROFILE_ONE_CALL"
#: the marker paths a window's open and close `stat` (they never exist; strace shows the failed call)
STRACE_MARK_OPEN = "/callprof-window-open"
STRACE_MARK_CLOSE = "/callprof-window-close"
#: py-spy prints this once it samples; the window's clock starts only after it (or after PYSPY_READY_S)
PYSPY_READY = "Sampling process"
PYSPY_READY_S = 20.0
_RANGE = []      # the open window's parts (one window at most: the extra calls of one scored run)


_LIBS = {}       # the call profile's ctypes libraries, loaded once at the window's open (never a dlopen in the window)


def _lib(path):
    import ctypes
    if path not in _LIBS:
        _LIBS[path] = ctypes.CDLL(path)
    return _LIBS[path]


def _shim_counts(env):
    """{entry: [calls, ns, bytes]} of the preloaded timing shim, or None when it is not loaded (no env, no library)."""
    path = (env.get(SHIM_ENV) or "").strip()
    if not path:
        return None
    try:
        import ctypes
        lib = _lib(path)
        lib.callprof_shim_name.restype = ctypes.c_char_p
        n = int(lib.callprof_shim_n())
        buf = (ctypes.c_uint64 * (3 * n))()
        lib.callprof_shim_read(buf, n)
        return {lib.callprof_shim_name(i).decode(): [int(buf[3 * i]), int(buf[3 * i + 1]), int(buf[3 * i + 2])]
                for i in range(n)}
    except Exception as exc:                      # noqa: BLE001 -- the report says the shim is missing
        print("CALLPROF-SHIM unavailable: %s: %s" % (exc.__class__.__name__, exc), file=sys.stderr, flush=True)
        return None


def _callgrind(env, which):
    """Call callprof_cg_<which> of the callgrind window library; True when this process runs under valgrind."""
    path = (env.get(CALLGRIND_ENV) or "").strip()
    if not path:
        return None
    try:
        lib = _lib(path)
        _ = (lib.callprof_cg_open, lib.callprof_cg_close, lib.callprof_cg_running)     # looked up before the window
        getattr(lib, "callprof_cg_" + which)()
        return bool(lib.callprof_cg_running())
    except Exception as exc:                      # noqa: BLE001 -- the report says the callgrind window is missing
        print("CALLPROF-CALLGRIND %s failed: %s: %s" % (which, exc.__class__.__name__, exc), file=sys.stderr, flush=True)
        return False


def _pyspy_start(env):
    """Attach `py-spy record --native` to this process for the window: (Popen, error). Waits until py-spy samples."""
    spec = (env.get(PYSPY_ENV) or "").strip()
    if not spec:
        return None, None
    parts = spec.split(",")
    if len(parts) != 3:
        return None, "%s wants <py-spy>,<out>,<rate>: %r" % (PYSPY_ENV, spec)
    exe, out, rate = parts
    import subprocess
    import time
    try:
        import ctypes
        # yama ptrace_scope 1 lets only an ancestor trace a process; py-spy is our child, so allow any tracer
        ctypes.CDLL(None, use_errno=True).prctl(0x59616d61, ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF), 0, 0, 0)
    except Exception:                             # noqa: BLE001 -- not Linux, or no yama: nothing to allow
        pass
    log = out + ".log"
    try:
        with open(log, "w") as fh:
            p = subprocess.Popen([exe, "record", "--pid", str(os.getpid()), "--native", "--threads", "--rate", rate,
                                  "--format", "raw", "--output", out],
                                 stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        return None, "py-spy did not start: %s" % exc
    t_end = time.monotonic() + PYSPY_READY_S
    while time.monotonic() < t_end:
        if p.poll() is not None:
            with open(log, errors="replace") as fh:
                return None, "py-spy exited %s: %s" % (p.returncode, fh.read()[-300:].replace("\n", " "))
        try:
            with open(log, errors="replace") as fh:
                if PYSPY_READY in fh.read():
                    return p, None
        except OSError:
            pass
        time.sleep(0.02)
    return p, "py-spy not sampling after %.0f s (kept attached)" % PYSPY_READY_S


def _pyspy_stop(p):
    """SIGINT py-spy (it writes its file and exits); its exit code, or an error string."""
    if p is None:
        return None
    import signal
    import subprocess
    try:
        p.send_signal(signal.SIGINT)
        return p.wait(timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        p.kill()
        return "py-spy stop: %s" % exc


def _perf_ctl(env, word):
    """Send `enable`/`disable` to perf's control FIFO and wait (5 s at most) for its ack; True when acked."""
    spec = (env.get(PERF_CTL_ENV) or "").strip()
    if not spec or "," not in spec:
        return False
    ctl, ack = spec.split(",", 1)
    try:
        import select
        import time
        # the ack's read end is open BEFORE the command goes out: a FIFO drops what was written to it once no end
        # is open, so a perf that acks and closes before we open would lose the ack (a flaky False)
        afd = os.open(ack, os.O_RDONLY | os.O_NONBLOCK)
        try:
            fd = os.open(ctl, os.O_WRONLY | os.O_NONBLOCK)     # ENXIO when perf is not reading it: never blocks
            try:
                os.write(fd, (word + "\n").encode())
            finally:
                os.close(fd)
            t_end = time.monotonic() + 5.0
            while time.monotonic() < t_end:
                ready, _, _ = select.select([afd], [], [], max(t_end - time.monotonic(), 0.0))
                data = os.read(afd, 64) if ready else b""
                if b"ack" in data:
                    return True
                if ready and not data:      # no writer yet (or between writers): wait for the ack
                    time.sleep(0.01)
            return False
        finally:
            os.close(afd)
    except OSError as exc:
        print("CALLPROF-PERF %s failed: %s" % (word, exc), file=sys.stderr, flush=True)
        return False


def _mark(path):
    try:
        os.stat(path)
    except OSError:
        pass


def _window_line(env, rec):
    path = (env.get(PROFILE_WINDOW_ENV) or "").strip()
    if path:
        import json
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def range_open(n_calls, phase=None, environ=None):
    """Open the attribution profile's window around the extra calls of a FORCED scored run (calls 2..n), so the
    profiler captures exactly those n - 1 calls and nothing of the data load, the fit before an inference, the
    warmup or the first call's hashing: the NVTX range (NVTX_RANGE_ENV) and the call profile's window
    (PROFILE_WINDOW_ENV, PERF_CTL_ENV, the strace markers, SHIM_ENV, PYSPY_ENV, CALLGRIND_ENV). Only when one of them is set,
    n_calls > 1 and forced_calls applies to this phase; True when a window opened. Outside every clock (the callers
    open it before the next call's t0)."""
    env = os.environ if environ is None else environ
    name = (env.get(NVTX_RANGE_ENV) or "").strip()
    window = (env.get(PROFILE_WINDOW_ENV) or "").strip()
    if not (name or window) or _RANGE or int(n_calls or 1) <= 1 or forced_calls(True, phase, env) is None:
        return False
    parts = {"nvtx": None, "window": None}
    if name:
        try:
            path = env.get(NVTX_PATH_ENV)
            if path and path not in sys.path:
                sys.path.insert(0, path)
            import nvtx
        except Exception as exc:                      # noqa: BLE001 -- the profile run reports a missing capture
            print("ATTRIBUTION-NVTX unavailable: %s: %s" % (exc.__class__.__name__, exc), file=sys.stderr, flush=True)
            if not window:
                return False
        else:
            parts["nvtx"] = nvtx
    if window:
        import time
        spy, spy_err = _pyspy_start(env)          # before the clock: attaching takes a moment
        perf = _perf_ctl(env, "enable")
        _mark(STRACE_MARK_OPEN)
        rec = {"event": "open", "phase": phase, "calls": int(n_calls) - 1, "pid": os.getpid(), "perf": perf,
               "pyspy": spy is not None, "pyspy_error": spy_err, "shim": _shim_counts(env)}
        if one_call(env):     # a capped one-call run is killed inside its call: the open must already be on disk
            import time as _t
            _window_line(env, {"event": "open-mark", "phase": phase, "calls": rec["calls"], "pid": rec["pid"],
                               "t_unix": _t.time(), "pyspy": rec["pyspy"], "pyspy_error": spy_err})
        rec["callgrind"] = _callgrind(env, "open")          # last before the clock: nothing of the above collected
        parts["window"] = (env, rec, spy, time.perf_counter_ns())
    if parts["nvtx"] is not None:
        parts["nvtx"].push_range(message=name)
    _RANGE.append(parts)
    return True


def range_close():
    """Close the window range_open opened (a no-op when none is open)."""
    if not _RANGE:
        return
    parts = _RANGE.pop()
    if parts["nvtx"] is not None:
        parts["nvtx"].pop_range()
    if parts["window"] is not None:
        import time
        env, rec, spy, t0 = parts["window"]
        wall_ns = time.perf_counter_ns() - t0
        cg = _callgrind(env, "close")                       # first after the clock
        shim = _shim_counts(env)
        _mark(STRACE_MARK_CLOSE)
        perf = _perf_ctl(env, "disable")
        stopped = _pyspy_stop(spy)
        end = {"event": "close", "phase": rec["phase"], "calls": rec["calls"], "pid": rec["pid"], "wall_ns": wall_ns,
               "perf": perf, "shim": shim, "callgrind": cg, "pyspy": spy is not None, "pyspy_exit": stopped}
        _window_line(env, rec)
        _window_line(env, end)


def one_call(environ=None):
    """True in a ONE-CALL call profile run (PROFILE_ONE_CALL_ENV `1`, with a forced count and a window or an NVTX
    range to open). Never on a board (the variables are unset)."""
    env = os.environ if environ is None else environ
    return (env.get(PROFILE_ONE_CALL_ENV) or "").strip() == "1" and \
        bool((env.get(FORCE_CALLS_ENV) or "").strip()) and \
        bool((env.get(PROFILE_WINDOW_ENV) or "").strip() or (env.get(NVTX_RANGE_ENV) or "").strip())


def first_round(environ=None):
    """The first round a race driver sends: 1 in a one-call run (no warmup round), else 0 (the excluded warmup)."""
    return 1 if one_call(environ) else 0


def range_open_one(phase, environ=None):
    """ONE-CALL run only: open the window around the FIRST scored call of `phase` when it is the forced phase (the
    callers call it before that call's clock starts and range_close after it stops); False otherwise (no-op)."""
    env = os.environ if environ is None else environ
    if not one_call(env):
        return False
    return range_open(2, phase, env)      # the window counts n - 1 calls: this one


def fit_warmup_skipped(environ=None):
    """True in an attribution run whose forced phase is an inference (FORCE_CALLS_PHASE_ENV `infer...`): the fit
    only builds the model the timed inference calls use, so its excluded warmup fit is skipped (a 15-minute fit
    would otherwise run twice per run); and in a one-call run (one_call: no warmup at all). Never on a board (the
    variables are unset)."""
    env = os.environ if environ is None else environ
    return (bool((env.get(FORCE_CALLS_ENV) or "").strip()) and
            (env.get(FORCE_CALLS_PHASE_ENV) or "").strip().startswith("infer")) or one_call(env)


def profile_calls(forced, first_ms, environ=None):
    """A forced run's call count: `forced`, raised in a call profile run (PROFILE_SECONDS_ENV) to the calls that fill
    that many seconds at the first scored call's time, capped by PROFILE_MAX_CALLS_ENV (default 100000)."""
    env = os.environ if environ is None else environ
    if one_call(env):
        return 1          # one call in all: the window holds the first scored call (range_open_one)
    raw = (env.get(PROFILE_SECONDS_ENV) or "").strip()
    try:
        secs = float(raw) if raw else 0.0
        ms = float(first_ms) if first_ms is not None and not isinstance(first_ms, bool) else 0.0
        cap = int((env.get(PROFILE_MAX_CALLS_ENV) or "100000").strip())
    except ValueError:
        return forced
    if not (secs > 0 and math.isfinite(secs) and ms > 0 and math.isfinite(ms)):
        return forced
    return max(forced, min(cap, int(math.ceil(secs * 1000.0 / ms)) + 1))


def calls_for(first_ms, ours, phase=None):
    """The calls the one scored run makes in all, given its FIRST scored call's time (taken after the
    excluded warmup): SMALL_ROW_CALLS for OUR arm when that call took under SMALL_ROW_MS, else 1 (an
    opponent, a big row, or no time to decide on). The caller makes calls_for(...) - 1 more calls.
    `phase` (`fit`, `infer/<batch>`) only matters to an attribution run's forced count (forced_calls)."""
    forced = forced_calls(ours, phase)
    if forced is not None:
        return profile_calls(forced, first_ms)
    if ours and (os.environ.get(PROFILE_WINDOW_ENV) or "").strip() and (os.environ.get(FORCE_CALLS_ENV) or "").strip():
        return 1        # a call profile run: the phase it does not profile makes one call (the window holds one phase)
    if not ours or first_ms is None or isinstance(first_ms, bool):
        return 1
    try:
        w = float(first_ms)
    except (TypeError, ValueError):
        return 1
    if not math.isfinite(w) or w < 0:
        return 1
    return SMALL_ROW_CALLS if w < SMALL_ROW_MS else 1


def spread(samples):
    """{ms: median, calls, min_ms, max_ms, call_ms}: the record of one scored run's calls."""
    xs = [float(x) for x in samples]
    if not xs:
        raise ValueError("a scored run has at least one call")
    return {"ms": float(statistics.median(xs)), "calls": len(xs), "min_ms": min(xs), "max_ms": max(xs),
            "call_ms": xs}


def line_suffix(rec):
    """The words a result line (FSPEED, ALGOS-ROUND, ...) carries after its fields: ` calls=1` for a scored
    run of one call (so a line-parsed cell records SCORING_ONE too), the calls and spread for more; empty
    without a record (the warmup, an arm that recorded none)."""
    if not rec:
        return ""
    if int(rec.get("calls") or 1) <= 1:
        return " calls=1"
    return " calls=%d min_ms=%.6f max_ms=%.6f" % (rec["calls"], rec["min_ms"], rec["max_ms"])


def cell_fields(scored_calls):
    """Board cell fields from a race arm's `scored_calls` list (one record per scored round): `scoring` (the
    rule: SCORING_MEDIAN when any round made more than one call, else SCORING_ONE), `calls` (the calls of
    each scored round) and, for a median, `min_ms`/`max_ms` over every call of every scored round (the
    spread the board's `min..max ms` column shows). {} when the race did not record calls (opponents'
    stored cells, races from before 2026-11-15; scoring_of reads those as one call)."""
    recs = [r for r in (scored_calls or []) if isinstance(r, dict) and r.get("calls")]
    if not recs:
        return {}
    out = {"calls": max(int(r["calls"]) for r in recs)}
    out["scoring"] = SCORING_MEDIAN if out["calls"] > 1 else SCORING_ONE
    if out["calls"] > 1:
        mins = [r["min_ms"] for r in recs if r.get("min_ms") is not None]
        maxs = [r["max_ms"] for r in recs if r.get("max_ms") is not None]
        if mins and maxs:
            out.update(min_ms=float(min(mins)), max_ms=float(max(maxs)))
    return out


def parse_suffix(fields):
    """A scored_calls record from a parsed key=value line (FSPEED / FSPEED-INFER), None without calls=."""
    try:
        calls = int(fields.get("calls")) if fields.get("calls") is not None else None
    except (TypeError, ValueError):
        return None
    if not calls:
        return None
    rec = {"calls": calls}
    for k in ("min_ms", "max_ms"):
        try:
            rec[k] = float(fields[k])
        except (KeyError, TypeError, ValueError):
            rec[k] = None
    return rec


def scoring_of(cell):
    """The rule a cell's time was taken under: its `scoring`, else from its `calls` (a cell without either
    was one call: every cell before 2026-11-15 and every stored opponent cell)."""
    cell = cell or {}
    if cell.get("scoring") in (SCORING_MEDIAN, SCORING_ONE):
        return cell["scoring"]
    try:
        return SCORING_MEDIAN if int(cell.get("calls") or 1) > 1 else SCORING_ONE
    except (TypeError, ValueError):
        return SCORING_ONE


def text(cell):
    """The compare's words beside a time: ' (median of 20 calls, 0.70..0.95 ms)', '' for one call."""
    calls = (cell or {}).get("calls")
    try:
        calls = int(calls)
    except (TypeError, ValueError):
        return ""
    if calls <= 1:
        return ""
    lo, hi = cell.get("min_ms"), cell.get("max_ms")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return " (median of %d calls, %.3g..%.3g ms)" % (calls, lo, hi)
    return " (median of %d calls)" % calls
