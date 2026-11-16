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
a first call on either side of SMALL_ROW_MS). Opponent arms are never repeated (their stored cells
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
    if want and (phase is None or (phase != want and not phase.startswith(want + "/"))):
        return None
    return n


#: ATTRIBUTION PROFILE RUNS ONLY: the NVTX range name the job's `nsys profile --capture-range=nvtx` waits for, and a
#: `pip --target` directory holding the `nvtx` package (the measured venvs stay untouched). Unset: no-ops.
NVTX_RANGE_ENV = "MOJOLEARN_BENCH_NVTX_RANGE"
NVTX_PATH_ENV = "MOJOLEARN_BENCH_NVTX_PATH"
_RANGE = []      # the open range's nvtx module (one at most: the extra calls of one scored run)


def range_open(n_calls, phase=None, environ=None):
    """Open the attribution profile's NVTX range around the extra calls of a FORCED scored run (calls 2..n), so the
    profiler captures exactly those n - 1 calls and nothing of the data load, the fit before an inference, the
    warmup or the first call's hashing. Only when NVTX_RANGE_ENV is set, n_calls > 1 and forced_calls applies to this
    phase; True when a range opened. Outside every clock (the callers open it before the next call's t0)."""
    env = os.environ if environ is None else environ
    name = (env.get(NVTX_RANGE_ENV) or "").strip()
    if not name or _RANGE or int(n_calls or 1) <= 1 or forced_calls(True, phase, env) is None:
        return False
    try:
        path = env.get(NVTX_PATH_ENV)
        if path and path not in sys.path:
            sys.path.insert(0, path)
        import nvtx
        nvtx.push_range(message=name)
    except Exception as exc:                      # noqa: BLE001 -- the profile run reports a missing capture
        print("ATTRIBUTION-NVTX unavailable: %s: %s" % (exc.__class__.__name__, exc), file=sys.stderr, flush=True)
        return False
    _RANGE.append(nvtx)
    return True


def range_close():
    """Close the range range_open opened (a no-op when none is open)."""
    if _RANGE:
        _RANGE.pop().pop_range()


def fit_warmup_skipped(environ=None):
    """True in an attribution run whose forced phase is an inference (FORCE_CALLS_PHASE_ENV `infer...`): the fit
    only builds the model the timed inference calls use, so its excluded warmup fit is skipped (a 15-minute fit
    would otherwise run twice per run). Never on a board (the variables are unset)."""
    env = os.environ if environ is None else environ
    return bool((env.get(FORCE_CALLS_ENV) or "").strip()) and \
        (env.get(FORCE_CALLS_PHASE_ENV) or "").strip().startswith("infer")


def calls_for(first_ms, ours, phase=None):
    """The calls the one scored run makes in all, given its FIRST scored call's time (taken after the
    excluded warmup): SMALL_ROW_CALLS for OUR arm when that call took under SMALL_ROW_MS, else 1 (an
    opponent, a big row, or no time to decide on). The caller makes calls_for(...) - 1 more calls.
    `phase` (`fit`, `infer/<batch>`) only matters to an attribution run's forced count (forced_calls)."""
    forced = forced_calls(ours, phase)
    if forced is not None:
        return forced
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
