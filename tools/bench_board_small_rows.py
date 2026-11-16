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
import statistics

#: a first scored call of our arm under this many milliseconds makes its scored run a median of calls
SMALL_ROW_MS = 10.0
#: the calls of one scored run of a small row (the first scored call included)
SMALL_ROW_CALLS = 20
#: the rule a scored run was taken under, recorded on the cell (`scoring`)
SCORING_MEDIAN = "median-of-calls"
SCORING_ONE = "one-call"


def calls_for(first_ms, ours):
    """The calls the one scored run makes in all, given its FIRST scored call's time (taken after the
    excluded warmup): SMALL_ROW_CALLS for OUR arm when that call took under SMALL_ROW_MS, else 1 (an
    opponent, a big row, or no time to decide on). The caller makes calls_for(...) - 1 more calls."""
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
