"""DEVIATION 1946 in 60 lines: does a buffer outliving its DeviceContext
wedge the process?

Written 2026-08-29 after leg 2 of `tools/diag/rtx4090_hang.sh`
(`bench/results/e1/2026-08-29_165644-runpod-nvidia/diag/verdicts.txt`) left
one question standing: through the Python bindings the FIRST call of the rf
or iforest binding returns and the NEXT GPU call in the process never does,
while `ensemble/mojo_only/rf_ctx_probe.mojo` -- billed as "the binding's
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

    -D ORDER_BAD=1    the pre-1946 binding: buffers freed after ctx's last use
    -D ORDER_GOOD=1   the fix: `_ = ctx^` last, so the context dies last

    mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_BAD=1 \\
        ensemble/mojo_only/rf_ctx_order_probe.mojo -o /tmp/order_bad && /tmp/order_bad
    mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_GOOD=1 \\
        ensemble/mojo_only/rf_ctx_order_probe.mojo -o /tmp/order_good && /tmp/order_good

VERDICT: on a box where the bindings hang, BAD must hang on call 2 (or in
call 1's teardown) and GOOD must print DONE. If BOTH print DONE, the ordering
is NOT the poison on that box and DEVIATION 1946 is not the whole answer --
say so in the leg's verdicts.txt rather than assuming this file's thesis.
Every phase prints a flushed line, so a hang names its phase.
"""

from std.sys import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

comptime DT = DType.float32
comptime ORDER_BAD = is_defined["ORDER_BAD"]()
comptime ORDER_GOOD = is_defined["ORDER_GOOD"]()
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


def main() raises:
    comptime if not (ORDER_BAD or ORDER_GOOD):
        print("orderprobe: build with -D ORDER_BAD=1 or -D ORDER_GOOD=1")
        return
    var t0 = perf_counter_ns()
    var a = one_call("call1", t0)
    var b = one_call("call2", t0)
    _say("both calls returned, probe equal " + String(a == b), t0)
    print("orderprobe DONE", flush=True)
