# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHERE AN OPERATION RUNS. `Exec` is the one interface the lane's algorithms
are written against (`sequence/recurrent.mojo`); `HostExec` (here, no GPU
import) runs each operation as an ascending host loop over its elements and
`DeviceExec` (`sequence/exec_device.mojo`) as one GPU thread per element.
The element body is the same function (`sequence/ops.mojo::apply`), so the
two agree bit for bit by construction.

THE HOST THREAD SPLIT (lane sequence-cpu, 2026-09-28). A launch's elements
are the GPU's threads: no element reads what another element of the same
launch writes (the device runs them with no barrier and no order), and each
element's arithmetic is its own body's, in its own order. So `HostExec`
splits the element range of a launch into contiguous chunks and runs them
on `core/host_parallel.mojo::host_parallelize` (every task in the caller's
floating-point environment, DEVIATION 5900). Which thread runs an element
never moves a bit of it: the result is the ascending serial walk's at
every thread count. The task count is `core/host_predict_threads.mojo`'s
policy (MOJOLEARN_CPU_THREADS, else one per physical core), cut down so a
task gets at least `HOST_LAUNCH_GRAIN` units of work (`_element_weight`),
so the many tiny launches of a small recurrent step stay on the calling
thread. A single task runs on the calling thread without touching the
pool."""
from std.memory import memcpy, memset_zero

from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from sequence.dispatch import apply
from sequence.host_gemm import host_gemm_pack, host_gemm_rows
from sequence.ops import (
    FP,
    Args,
    OP_GEMM,
    OP_COLSUM,
    OP_CELL_FWD,
    OP_CELL_BWD,
    OP_CE,
    OP_SOFTMAX,
    OP_MLP_ROWLOSS,
    OP_AF_ROW,
    OP_AF_COL,
    OP_LN_FWD,
    OP_LN_BWD_X,
    OP_LN_BWD_W,
    OP_SEG_SUMSQ,
    OP_LAMB_RATIO,
    OP_STL,
    OP_THETA,
    OP_CROSTON,
    OP_ETS,
    OP_GARCH,
    OP_PROPHET_FEATURES,
    OP_PROPHET_FIT,
    OP_PROPHET_PREDICT,
    OP_CHOLSOLVE,
    OP_VAR_FORECAST,
    OP_MOE_ROUTE,
    OP_MOE_HIDDEN,
    OP_MOE_OUT,
)

#: The least work (in `_element_weight` units, roughly one fused
#: multiply-add each) one host task is given. Below it the launch runs on
#: the calling thread: a pool dispatch costs microseconds, and the lane's
#: recurrent steps issue many launches of a few thousand elements.
comptime HOST_LAUNCH_GRAIN = 1 << 15

#: The weight of one element of an operation whose element is a whole
#: series fit or solve (STL, Theta, Croston, ETS, GARCH, Prophet, the VAR
#: solve): a single element already clears the grain.
comptime _HEAVY = HOST_LAUNCH_GRAIN


@always_inline
def _element_weight[OP: Int](a: Args) -> Int:
    """The rough cost of one element of OP, for the task count only (it
    decides how many threads, never what a thread computes)."""
    comptime if OP == OP_GEMM:
        return max(a.i2, 1)
    elif OP == OP_COLSUM:
        return max(a.i0, 1)
    elif OP == OP_LN_BWD_W:
        return max(a.i0, 1)
    elif (
        OP == OP_STL or OP == OP_THETA or OP == OP_CROSTON or OP == OP_ETS
        or OP == OP_GARCH or OP == OP_PROPHET_FIT or OP == OP_CHOLSOLVE
        or OP == OP_VAR_FORECAST or OP == OP_SEG_SUMSQ or OP == OP_LAMB_RATIO
    ):
        return _HEAVY
    elif (
        OP == OP_CE or OP == OP_SOFTMAX or OP == OP_MLP_ROWLOSS
        or OP == OP_AF_ROW or OP == OP_AF_COL or OP == OP_LN_FWD
        or OP == OP_LN_BWD_X or OP == OP_PROPHET_FEATURES
        or OP == OP_PROPHET_PREDICT or OP == OP_MOE_ROUTE
        or OP == OP_MOE_HIDDEN or OP == OP_MOE_OUT
    ):
        return 256
    elif OP == OP_CELL_FWD or OP == OP_CELL_BWD:
        return 32
    else:
        return 2


def host_launch_tasks(n: Int, weight: Int, workers: Int) -> Int:
    """How many contiguous element chunks a launch of `n` elements of the
    given weight splits into, under a ceiling of `workers`: at least 1,
    never more than `n`, and never so many that a task falls below
    HOST_LAUNCH_GRAIN."""
    if n <= 1 or workers <= 1:
        return 1
    var work = n * weight
    var by_work = work // HOST_LAUNCH_GRAIN
    var t = workers
    if by_work < t:
        t = by_work
    if t > n:
        t = n
    if t < 1:
        t = 1
    return t


trait Exec:
    def alloc(mut self, n: Int) raises -> FP:
        """A zero-filled buffer of n floats that lives as long as the Exec."""
        ...

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        """Host memory `src` -> this Exec's buffer `dst` (n floats)."""
        ...

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        """This Exec's buffer `src` -> host memory `dst` (n floats)."""
        ...

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        """Run operation OP over elements 0..n-1."""
        ...

    def sync(mut self) raises:
        ...


struct HostExec(Exec):
    var bufs: List[FP]
    #: The task ceiling, read once per Exec (one entry call).
    var workers: Int
    #: The GEMM's [K, N] panel of B (`sequence/host_gemm.mojo`), reused
    #: across launches, grown on demand.
    var panel: FP
    var panel_cap: Int

    def __init__(out self):
        self.bufs = List[FP]()
        self.workers = host_predict_task_count(1 << 30)
        self.panel = FP(unsafe_from_address=Int(alloc[Float32](1)))
        self.panel_cap = 1

    def __deinit__(deinit self):
        for i in range(len(self.bufs)):
            self.bufs[i].free()
        self.panel.free()

    def alloc(mut self, n: Int) raises -> FP:
        var count = n if n > 0 else 1
        var p = alloc[Float32](count)
        memset_zero(p, count)
        var q = FP(unsafe_from_address=Int(p))
        self.bufs.append(q)
        return q

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        if n > 0:
            memcpy(dest=dst, src=src, count=n)

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        if n > 0:
            memcpy(dest=dst, src=src, count=n)

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        if n <= 0:
            return
        comptime if OP == OP_GEMM:
            self._gemm(a)
            return
        var tasks = host_launch_tasks(n, _element_weight[OP](a), self.workers)
        if tasks <= 1:
            for t in range(n):
                apply[OP](t, a)
            return
        var chunk = host_predict_chunk(n, tasks)

        def _chunk(c: Int) {imm a, imm chunk, imm n}:
            var lo = c * chunk
            var hi = min(lo + chunk, n)
            for t in range(lo, hi):
                apply[OP](t, a)

        host_parallelize(_chunk, tasks)

    def _gemm(mut self, a: Args):
        """OP_GEMM over `sequence/host_gemm.mojo`: row ranges on threads,
        each row's cells W columns at a time; every cell `op_gemm`'s bits."""
        var M = a.i0
        var N = a.i1
        var K = a.i2
        var bp = a.p1
        var ldb = a.i5
        if a.i6 != 1 and K > 0:
            if K * N > self.panel_cap:
                self.panel.free()
                self.panel = FP(unsafe_from_address=Int(alloc[Float32](K * N)))
                self.panel_cap = K * N
            host_gemm_pack(a, self.panel)
            bp = self.panel
            ldb = N
        var tasks = host_launch_tasks(M, max(N * K, 1), self.workers)
        if tasks <= 1:
            host_gemm_rows(a, bp, ldb, 0, M)
            return
        var chunk = host_predict_chunk(M, tasks)

        def _rows(c: Int) {imm a, imm bp, imm ldb, imm chunk, imm M}:
            var lo = c * chunk
            host_gemm_rows(a, bp, ldb, lo, min(lo + chunk, M))

        host_parallelize(_rows, tasks)

    def sync(mut self) raises:
        pass
