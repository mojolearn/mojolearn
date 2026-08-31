# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 1946 in 60 lines: does a buffer outliving its DeviceContext
wedge the process?

Written 2026-08-29 after leg 2 of `tools/diag/rtx4090_hang.sh`
(`bench/results/e1/2026-08-29_165644-runpod-nvidia/diag/verdicts.txt`) left
one question standing: through the Python bindings the FIRST call of the rf
or iforest binding returns and the NEXT GPU call in the process never does,
while `ensemble/original/rf_ctx_probe.mojo` -- billed as "the binding's
shape without Python" -- passes two fits in one process.

It is not the GIL. The probe's `one_fit` takes `ctx` as a BORROWED argument,
so `main` owns it and every buffer is freed while it is still alive. The
bindings create the context INSIDE the call, and Mojo destroys a value at its
LAST USE, so `ctx` died at `synchronize()` and `_ = dx^ ... _ = hy^` then
freed five buffers -- two of them PINNED HOST allocations -- against a
context that had already gone (`bindings/_mojolearn_rf.mojo`), exactly as
`iforest_run_host` freed its model's eight buffers one line after its own
context's last use (`isolation_forest/estimator.mojo`). That is DEVIATION
1944's class -- a buffer freed against a context that is not the live one --
one call later.

This file asks that and nothing else. No forest, no fit, no Python: two
sequential calls, each creating its own context, in the two orders.

    -D ORDER_BAD=1     the pre-1946 binding: buffers freed after ctx's last
                       use, with ctx's death INFERRED from Mojo's ASAP rule
    -D ORDER_GOOD=1    the fix: `_ = ctx^` last, so the context dies last
    -D ORDER_FORCED=1  the same hazard with ctx's death STRUCTURAL rather
                       than inferred -- see "WHY A THIRD ARM" below

    mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_BAD=1 \\
        ensemble/original/rf_ctx_order_probe.mojo -o /tmp/order_bad && /tmp/order_bad
    mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_GOOD=1 \\
        ensemble/original/rf_ctx_order_probe.mojo -o /tmp/order_good && /tmp/order_good

WHY A THIRD ARM, AND WHY A TWO-ARM BOTH-PASS COULD NOT BE READ
===============================================================
With BAD and GOOD alone, "both print DONE" has TWO readings and the file
cannot tell them apart:

  1. the ordering is innocent on this box, or
  2. THE BAD ARM NEVER BUILT THE BAD ORDERING.

Reading 2 is not hypothetical here. `ORDER_BAD` does not destroy the context
itself; it relies on `ctx`'s last SYNTACTIC use being the `synchronize()`
above and on Mojo destroying a value at its last use. If `DeviceBuffer` holds
a reference to the context that created it -- which is the ordinary way such
a type is built, and which nothing in this repository has checked -- then `d`
and `h` keep the context alive no matter where the local handle dies, the BAD
arm is the GOOD arm with extra steps, and both arms pass on every box
including one where the bindings are hanging.

That is the same trap `rf_ctx_probe.mojo` fell into: it was written to
reproduce an ordering and accidentally fixed it. Writing a second probe that
falls into it one file later would be the more expensive mistake, because
this one would be read as EXONERATING the ordering.

`ORDER_FORCED` removes the inference. The context is created inside a helper
and the buffers ESCAPE it by transfer, so the context handle is structurally
gone before the caller releases them -- no ASAP-rule reasoning required. Then:

    BAD hangs                     the ordering is the poison; 1946 is the cure
    BAD passes, FORCED hangs      the hazard is real and `ORDER_BAD` was too
                                  weak to build it. 1946 is still the cure,
                                  and every OTHER conclusion drawn from an
                                  `ORDER_BAD` pass is void
    BAD and FORCED both pass,
    bindings still hang           the ordering is NOT the poison on this box.
                                  1946 is a correctness fix and not the cure,
                                  and the hunt goes back to GILReleased
    FORCED will not compile       a buffer cannot outlive its context in this
                                  language version, so the hazard cannot
                                  exist as described and DEVIATION 1944's
                                  mechanism needs re-reading. Record the
                                  compiler error; it is the finding.

VERDICT: on a box where the bindings hang, BAD or FORCED must hang on call 2
(or in call 1's teardown) and GOOD must print DONE. Read a both-pass against
the table above and write which row it was into the leg's verdicts.txt rather
than assuming this file's thesis. Every phase prints a flushed line, so a
hang names its phase.
"""

from std.sys import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

comptime DT = DType.float32
comptime ORDER_BAD = is_defined["ORDER_BAD"]()
comptime ORDER_GOOD = is_defined["ORDER_GOOD"]()
comptime ORDER_FORCED = is_defined["ORDER_FORCED"]()
comptime N = 1 << 16


def _say(msg: String, t0: Int):
    print(
        "orderprobe +" + String((perf_counter_ns() - t0) // 1_000_000)
        + "ms: " + msg,
        flush=True,
    )


def one_call(tag: String, t0: Int) raises -> Float32:
    """The binding's body: a context created HERE, a pinned host buffer and a
    device buffer created on it, an upload, a synchronize, then the releases.
    The only difference between the two arms is where the context dies."""
    _say(tag + ": ctx", t0)
    var ctx = DeviceContext()
    var h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        h.unsafe_ptr().unsafe_store(i, Float32(i % 97) * Float32(0.5))
    var d = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _say(tag + ": uploaded", t0)
    var probe = h.unsafe_ptr().unsafe_load(N - 1)
    comptime if ORDER_BAD:
        # PRE-1946: `ctx`'s last use is the `synchronize()` above, so it is
        # destroyed HERE, before its own buffers.
        _ = d^
        _ = h^
    comptime if ORDER_GOOD:
        _ = d^
        _ = h^
        _ = ctx^
    _say(tag + ": released", t0)
    return probe


def forced_call(tag: String, t0: Int) raises -> Float32:
    """The hazard with the context's death STRUCTURAL, not inferred.

    `_open` builds the context and lets the two buffers escape by transfer,
    so the context handle is gone at the return and the caller releases the
    buffers afterwards. Nothing here depends on where a compiler decides a
    value's last use is. See "WHY A THIRD ARM"."""
    _say(tag + ": ctx (forced)", t0)

    def _open() raises -> Tuple[DeviceBuffer[DT], HostBuffer[DT]]:
        var ctx = DeviceContext()
        var h = ctx.enqueue_create_host_buffer[DT](N)
        ctx.synchronize()
        for i in range(N):
            h.unsafe_ptr().unsafe_store(i, Float32(i % 97) * Float32(0.5))
        var d = ctx.enqueue_create_buffer[DT](N)
        ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
        ctx.synchronize()
        return d^, h^

    var pair = _open()
    _say(tag + ": uploaded, context handle out of scope", t0)
    var probe = pair[1].unsafe_ptr().unsafe_load(N - 1)
    _ = pair^
    _say(tag + ": released", t0)
    return probe


def main() raises:
    comptime if not (ORDER_BAD or ORDER_GOOD or ORDER_FORCED):
        print(
            "orderprobe: build with -D ORDER_BAD=1, -D ORDER_GOOD=1 or"
            " -D ORDER_FORCED=1"
        )
        return
    var t0 = perf_counter_ns()
    var a = Float32(0.0)
    var b = Float32(0.0)
    comptime if ORDER_FORCED:
        a = forced_call("call1", t0)
        b = forced_call("call2", t0)
    else:
        a = one_call("call1", t0)
        b = one_call("call2", t0)
    _say("both calls returned, probe equal " + String(a == b), t0)
    print("orderprobe DONE", flush=True)
