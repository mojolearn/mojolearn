# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHERE AN OPERATION RUNS. `Exec` (`sequence/exec_trait.mojo`) is the one interface the lane's algorithms
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
from sequence.exec_trait import Exec
from sequence.host_gemm import host_gemm_pack, host_gemm_rows
from sequence.ops import (
    FP,
    Args,
    gates_of,
    OP_GEMM,
    OP_GEMM_SPLITK,
    OP_COLSUM,
    OP_CELL_FWD,
    OP_CELL_BWD,
    OP_BIAS,
    OP_CELL_FWD_H,
    OP_CELL_BWD_H,
    OP_GEMM_EPI,
    OP_GEMM_EPI_TAIL,
    OP_COLSUM_DIV,
    OP_CE,
    OP_SOFTMAX,
    OP_MLP_ROWLOSS,
    OP_AF_ROW,
    OP_AF_COL,
    OP_LN_FWD,
    OP_LN_BWD_X,
    OP_LN_BWD_W,
    OP_SEG_SUMSQ,
    OP_CHUNK_SUMSQ,
    OP_AF_BLK_SUMSQ,
    OP_MLP_L2PART,
    OP_MLP_ROWPART,
    OP_LAMB_RATIO,
    OP_LAMB_BLK,
    OP_LAMB_SEGFOLD,
    OP_LAMB_CLIP,
    OP_LAMB_TRUST,
    OP_STL,
    OP_THETA,
    OP_CROSTON,
    OP_ETS,
    OP_GARCH,
    OP_PROPHET_FEATURES,
    OP_PROPHET_FIT,
    OP_PROPHET_PREDICT,
    OP_PROPHET_FG_PART,
    OP_CHOLSOLVE,
    OP_VAR_FORECAST,
    OP_MOE_ROUTE,
    OP_MOE_HIDDEN,
    OP_MOE_OUT,
    OP_VAR_RESID,
    OP_VAR_SIGMA,
    OP_STL_SEAS,
    OP_STL_MA,
    OP_STL_LOESS,
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
    elif OP == OP_GEMM_SPLITK:
        return max(a.i10, 1) if a.i11 == 0 else max(a.i9, 1)
    elif OP == OP_COLSUM or OP == OP_COLSUM_DIV:
        return max(a.i0, 1)
    elif OP == OP_LN_BWD_W:
        return max(a.i0, 1)
    elif OP == OP_AF_BLK_SUMSQ or OP == OP_MLP_L2PART or OP == OP_MLP_ROWPART:
        return max(a.i1, 1)
    elif OP == OP_LAMB_BLK:
        return max(a.i2, 1) * (2 if a.i1 != 0 else 1)
    elif OP == OP_LAMB_SEGFOLD or OP == OP_LAMB_CLIP or OP == OP_LAMB_TRUST:
        return _HEAVY
    elif (
        OP == OP_STL or OP == OP_THETA or OP == OP_CROSTON or OP == OP_ETS
        or OP == OP_GARCH or OP == OP_PROPHET_FIT or OP == OP_CHOLSOLVE
        or OP == OP_VAR_FORECAST or OP == OP_SEG_SUMSQ or OP == OP_LAMB_RATIO
    ):
        return _HEAVY
    elif (
        OP == OP_CE or OP == OP_SOFTMAX or OP == OP_MLP_ROWLOSS
        or OP == OP_AF_ROW or OP == OP_AF_COL or OP == OP_LN_FWD or OP == OP_CHUNK_SUMSQ
        or OP == OP_LN_BWD_X or OP == OP_PROPHET_FEATURES
        or OP == OP_PROPHET_PREDICT or OP == OP_MOE_ROUTE or OP == OP_PROPHET_FG_PART
        or OP == OP_MOE_HIDDEN or OP == OP_MOE_OUT
    ):
        return 256
    elif OP == OP_CELL_FWD or OP == OP_CELL_BWD:
        return 32
    # lane/apple-fast-tsa2 (reached only under TSA2_VAR / TSA2_STL): the
    # chain length of one point
    elif OP == OP_VAR_RESID or OP == OP_VAR_SIGMA or OP == OP_STL_MA:
        return max(a.i1, 1)
    elif OP == OP_STL_SEAS:
        return 4 * max(a.i2, 1)
    elif OP == OP_STL_LOESS:
        return 4 * max(a.i1, 1)
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
        if n > 0 and Int(dst) != Int(src):
            memcpy(dest=dst, src=src, count=n)

    def download_async(mut self, dst: FP, src: FP, n: Int) raises:
        self.download(dst, src, n)

    def bind(mut self, src: FP, n: Int) raises -> FP:
        return src

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        if n <= 0:
            return
        comptime if OP == OP_GEMM:
            self._gemm(a)
            return
        # The device's fused launches (Apple speed, 2026-09-28) run here as
        # the launches they fuse, so the GEMM keeps the host kernel: the
        # same cells in the same order either way.
        comptime if OP == OP_GEMM_EPI:
            self._gemm(a)
            self.launch[OP_GEMM_EPI_TAIL](a, n)
            return
        comptime if OP == OP_CELL_FWD_H:
            # h_prev @ W_hh^T into GH, + b_hh, then the cell
            var B = a.i1
            var H = a.i2
            var GH = gates_of(a.i0) * H
            var g = Args()
            g.p0 = a.p3
            g.p1 = a.p7
            g.p2 = a.p1
            g.i0 = B
            g.i1 = GH
            g.i2 = H
            g.i3 = H
            g.i4 = 1
            g.i5 = 1
            g.i6 = H
            g.i8 = GH
            self._gemm(g)
            var bb = Args()
            bb.p0 = a.p1
            bb.p1 = a.p8
            bb.p2 = a.p1
            bb.i0 = B
            bb.i1 = GH
            bb.i2 = GH
            bb.i3 = GH
            self.launch[OP_BIAS](bb, B * GH)
            self.launch[OP_CELL_FWD](a, n)
            return
        comptime if OP == OP_CELL_BWD_H:
            # dh += dGH_{s+1} @ W_hh (when a later step exists), then the cell
            if a.i3 != 0:
                var B = a.i1
                var H = a.i2
                var GH = gates_of(a.i0) * H
                var g = Args()
                g.p0 = a.p9 + a.i4
                g.p1 = a.p11
                g.p2 = a.p5
                g.i0 = B
                g.i1 = H
                g.i2 = GH
                g.i3 = GH
                g.i4 = 1
                g.i5 = H
                g.i6 = 1
                g.i7 = 1
                g.i8 = H
                self._gemm(g)
            self.launch[OP_CELL_BWD](a, n)
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
