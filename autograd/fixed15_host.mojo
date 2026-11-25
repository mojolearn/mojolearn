# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE HOST ARM of the fixed15 products of the tape, the Sequential layers
and the recurrent estimators (lane fixed15-train, 2026-11-25).

    GEMM profile  mojolearn.identical.gemm.int15i64.v1
                  (`gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 6)

`fixed15_nt_host` is the FLAT plan of clause W-8 on the CPU column: each
operand quantized from its float32 values along the product's own contracted
extent (W-1, W-2, W-9), then one Int64 sum of Int32 code products a cell, `p`
ascending (`int15_dot_cell`, THE DEFINITION), then the pinned conversion and
dequantization (W-6, W-7). Every plan of the profile gives the same bits
(W-8), so this arm and the device arm (`autograd/fixed15_gemm.mojo`, the
pieces or matrix-unit plan of `gemm/checks/gemm_int15.mojo`) agree bit for bit
by the contract; `checks/native_tests/fixed15_train_check.mojo` holds both to
`gemm/host/gemm_int15_oracle.mojo` and to a float64 oracle.

Layout. `C[m x n] = A[m x k] . B[k x n]` with `A(i, p) = a[i ar + p ac]` and
`B(p, j) = b[p br + j bc]`, OP_GEMM's words (sequence/ops.mojo op_gemm). The
left operand is quantized along `p` per row `i` (per token for a forward and
an input gradient); the right operand per column `j` along `p` (per output
feature for a forward, per input feature for an input gradient). Any strides
are read; the codes are made from the float32 values each time (W-9).

Parallel (AGENTS.md "write code to run in parallel"): the two quantizations
split over their rows and the product over the rows of C, each through
`host_parallelize` with the caller's worker count (HostExec.workers, the
allocation). A row's code and exponent are functions of that row alone and a
cell of its own two rows, so the split moves no bit.
"""

from checks.numerics import ftz
from checks.numerics_int15 import dequant_int15_pinned, int15_row_exponent, quantize_int15_value
from core.host_parallel import host_parallelize
from gemm.contract import INT15_MAX_K, gemm_oracle_sabotage_value_flip
from sequence.ops import FP
from std.sys.compile import is_defined

#: THE HOST SABOTAGE ARM: every cell of a fixed15 product on the host steps
#: one unit (`gemm_oracle_sabotage_value_flip`), so the host column of every
#: fixed15 verify lane must be seen to disagree with the device columns.
comptime FIXED15_TRAIN_SABOTAGE_HOST = is_defined["MOJOLEARN_FIXED15_TRAIN_SABOTAGE_HOST"]()

#: Fewer multiply-accumulates than this per task run in the calling thread:
#: a task's dispatch costs more than the work it would carry.
comptime FIXED15_HOST_GRAIN = 1 << 16


def fixed15_refuse_k(k: Int, what: String) raises:
    """Clause W-4: the profile's integer sums are exact for `1 <= k <=
    INT15_MAX_K` (65536) and a larger contraction is refused BY NAME, by
    every arm, before any work."""
    if k < 1 or k > INT15_MAX_K:
        raise Error(
            "ValueError: numeric_profile fixed15: " + what + " contracts k=" + String(k)
            + " values, outside [1, " + String(INT15_MAX_K)
            + "] (INT15_MAX_K, gemm/IDENTICAL_LOWBIT_CONTRACT.md W-4); refused by name. "
            + "Use numeric_profile='fp32_v1' for this model"
        )


def _tasks(units: Int, work_per_unit: Int, workers: Int) -> Int:
    if units <= 1 or workers <= 1:
        return 1
    var by_work = (units * max(work_per_unit, 1)) // FIXED15_HOST_GRAIN
    return max(1, min(units, min(workers, by_work)))


def _quantize_rows(
    q: MutPointer[Int16, MutUntrackedOrigin],
    e: MutPointer[Int32, MutUntrackedOrigin],
    x: FP,
    rows: Int,
    extent: Int,
    s_row: Int,
    s_p: Int,
    workers: Int,
):
    """Clauses W-1 and W-2 on row `r` of the logical `rows x extent` matrix
    whose element `(r, p)` is `x[r s_row + p s_p]`: the absmax after the
    flush (a maximum, exact and order-free, which a NaN never wins), its
    exponent, then every code. Written as row `r` of `q` (`rows x extent`)."""
    var tasks = _tasks(rows, extent, workers)
    var chunk = (rows + tasks - 1) // tasks

    def _rows(task: Int) {imm q, imm e, imm x, imm rows, imm extent, imm s_row, imm s_p, imm chunk}:
        for r in range(task * chunk, min((task + 1) * chunk, rows)):
            var best = Float32(0.0)
            for p in range(extent):
                var v = ftz(x.unsafe_load(r * s_row + p * s_p))
                if v < Float32(0.0):
                    v = -v
                if v > best:
                    best = v
            var ex = int15_row_exponent(best)
            e.unsafe_store(r, Int32(ex))
            for p in range(extent):
                q.unsafe_store(r * extent + p, quantize_int15_value(x.unsafe_load(r * s_row + p * s_p), ex))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)


def fixed15_nt_host(
    dst: FP, a: FP, ar: Int, ac: Int, b: FP, br: Int, bc: Int, m: Int, n: Int, k: Int, workers: Int,
) raises:
    """`C[m x n] = A . B` under `int15i64.v1` on the host (module docstring),
    written over `dst` (row stride n). `k` outside `[1, INT15_MAX_K]` is
    refused by name by the caller (`fixed15_gemm`) and again here."""
    fixed15_refuse_k(k, "a product")
    if m <= 0 or n <= 0:
        return
    var qa = alloc[Int16](m * k)
    var ea = alloc[Int32](m)
    var qb = alloc[Int16](n * k)
    var eb = alloc[Int32](n)
    var pqa = MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=Int(qa))
    var pea = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(ea))
    var pqb = MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=Int(qb))
    var peb = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(eb))
    # A row i along p: a[i ar + p ac]. B^T row j along p: b[p br + j bc].
    _quantize_rows(pqa, pea, a, m, k, ar, ac, workers)
    _quantize_rows(pqb, peb, b, n, k, bc, br, workers)
    var tasks = _tasks(m, n * k, workers)
    var chunk = (m + tasks - 1) // tasks

    def _cells(task: Int) {imm dst, imm pqa, imm pea, imm pqb, imm peb, imm m, imm n, imm k, imm chunk}:
        for i in range(task * chunk, min((task + 1) * chunk, m)):
            var ei = Int(pea.unsafe_load(i))
            for j in range(n):
                # THE DEFINITION (int15_dot_cell): Int32 products, Int64 sum,
                # p ascending; exact, so any order would give these bits
                var acc = Int64(0)
                for p in range(k):
                    acc += Int64(Int32(pqa.unsafe_load(i * k + p)) * Int32(pqb.unsafe_load(j * k + p)))
                var v = dequant_int15_pinned(acc, ei + Int(peb.unsafe_load(j)))
                comptime if FIXED15_TRAIN_SABOTAGE_HOST:
                    v = gemm_oracle_sabotage_value_flip(v)
                dst.unsafe_store(i * n + j, v)

    if tasks <= 1:
        _cells(0)
    else:
        host_parallelize(_cells, tasks)
    qa.free()
    ea.free()
    qb.free()
    eb.free()
