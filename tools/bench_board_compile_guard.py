"""Compiles inside a timed run: the torch.compile counters around every scored call, and the warm-up
that runs the exact timed call until nothing compiles (Andrew 2026-10-14).

A torch.compile arm compiles in its untimed warm-up, but a guard that fails on a later call (a step
count, a state dict that changed, a new tensor object) recompiles INSIDE a scored round: the
torch-compile-fp32 Adafactor cell took 281 ms against 16.6 ms eager. Each worker reads
torch._dynamo's counters before and after the scored call and records `recompiles_in_timed_run`; a
cell with any is shown with status "recompiled-in-clock" (flagged, never hidden).

The warm-up is the exact timed call (round 0: the same shapes, step count and state transitions).
Where every call does the same work from the same state (the algos lanes: each fit resets its
optimizer, reloads its initial weights or reruns a stateless layer), a compile arm repeats that call
untimed until one call compiles nothing (`settle`). A cuML/CuPy arm's first-call JIT is covered by the
same round-0 call: one identical untimed call before the clock.

Standard library only; torch is read only when the worker already imported it.
"""
import sys

STATUS = "recompiled-in-clock"
#: the most extra untimed warm-up calls `settle` makes before giving up (it records what it saw)
MAX_EXTRA_WARMUP_CALLS = 3


def is_compile_arm(arm):
    """A torch.compile arm (torch-compile-*, torch-cpu-compile-*)."""
    return "-compile-" in str(arm)


def read():
    """torch._dynamo's compile counters now, or None when torch is not loaded in this worker."""
    if "torch" not in sys.modules:
        return None
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001  (a torch without dynamo: nothing compiles)
        return None
    frames, stats = counters.get("frames") or {}, counters.get("stats") or {}
    return {"frames_ok": int(frames.get("ok", 0)), "unique_graphs": int(stats.get("unique_graphs", 0)),
            "recompiles": int(sum((counters.get("recompiles") or {}).values()))}


def compiles(before, after):
    """Compiles between two readings (a compiled frame or a new graph), or None when unreadable."""
    if before is None or after is None:
        return None
    return max(0, max(after.get(k, 0) - before.get(k, 0) for k in ("frames_ok", "unique_graphs")))


def settle(call, reader=read, max_extra=MAX_EXTRA_WARMUP_CALLS):
    """After the warm-up call: repeat the exact call, untimed, until one compiles nothing."""
    seen = []
    for _ in range(max_extra):
        before = reader()
        if before is None:
            break
        call()
        n = compiles(before, reader())
        seen.append(n)
        if not n:
            break
    return {"extra_warmup_calls": len(seen), "compiles_per_extra_call": seen,
            "settled": bool(seen) and seen[-1] == 0}


def summarize(per_round):
    """The scored rounds' compile counts -> the arm's `recompiles_in_timed_run` (None when no
    round could read the counters)."""
    got = [n for n in per_round if n is not None]
    return sum(got) if got else None


def flag_cell(cell, recompiles):
    """Record the count on a cell; a cell timed with a compile inside its clock is shown with
    status "recompiled-in-clock" (only an otherwise complete timing is relabelled)."""
    if recompiles is None:
        return cell
    cell["recompiles_in_timed_run"] = recompiles
    if recompiles > 0 and cell.get("status") in ("ok", "not-scored"):
        if cell.get("status") == "not-scored":
            cell["also_not_scored"] = True
        cell["status"] = STATUS
    return cell
