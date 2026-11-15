# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Small rows are the median of consecutive calls inside the one scored run (AGENTS.md "Measurement",
Andrew, 2026-11-15).

A row whose excluded warmup call of OUR arm took under SMALL_ROW_MS is scored as the median of
SMALL_ROW_CALLS consecutive calls of the same operation on the same inputs, made inside that one scored
run. The cell records the number of calls and their spread (`calls`, `min_ms`, `max_ms`); `ms` (and the
board's `median_ms`) is that median, so every reader of `ms` is unchanged. Rows at or above SMALL_ROW_MS
keep one call. Opponent arms are never repeated (their stored cells stay). One scored run per arm is
unchanged and nothing is rerun. Reason: a single call of a millisecond-scale row is dominated by
first-touch page faults and the allocator (program-run 3.4 ms once vs 0.76 ms steady).

The identity hash, the saved outputs and the state receipt of a repeated run come from the FIRST call's
output (the same bits every call); the receipt is still taken after the clock.

Benchmark glue only, never product runtime. The N calls are serial by nature: they are the timed
repetitions of one operation on one device, and running them concurrently would time contention.
"""
import math
import statistics

#: a warmup call of our arm under this many milliseconds makes its scored run a median of calls
SMALL_ROW_MS = 10.0
#: the calls of one scored run of a small row
SMALL_ROW_CALLS = 20


def calls_for(warmup_ms, ours):
    """The calls the one scored run makes: SMALL_ROW_CALLS for OUR arm when its excluded warmup call took
    under SMALL_ROW_MS, else 1 (an opponent, a big row, or no warmup time to decide on)."""
    if not ours or warmup_ms is None or isinstance(warmup_ms, bool):
        return 1
    try:
        w = float(warmup_ms)
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
    """The words a result line (FSPEED, ALGOS-ROUND, ...) carries after its fields: empty for one call, so a
    big row's line is byte-for-byte what it was."""
    if not rec or int(rec.get("calls") or 1) <= 1:
        return ""
    return " calls=%d min_ms=%.6f max_ms=%.6f" % (rec["calls"], rec["min_ms"], rec["max_ms"])


def cell_fields(scored_calls):
    """Board cell fields from a race arm's `scored_calls` list (one record per scored round): `calls` (the
    calls of each scored round) and, when any round made more than one call, `min_ms`/`max_ms` over every
    call of every scored round (the spread the board's `min..max ms` column shows). {} when the race did not
    record calls (opponents' stored cells, races from before 2026-11-15)."""
    recs = [r for r in (scored_calls or []) if isinstance(r, dict) and r.get("calls")]
    if not recs:
        return {}
    out = {"calls": max(int(r["calls"]) for r in recs)}
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
