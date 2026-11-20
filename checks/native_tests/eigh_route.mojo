# SPDX-License-Identifier: Apache-2.0
"""The device dense eigh route (`DevExec.eigh` -> `DevExec.eigh_into`, lane
perf-decomp-eigh 2026-11-13) against its host twin `eigh_op_sym_host`, word
for word, on the three routes the restored host-driven path takes:

  n = 24   the one-launch batched Jacobi (n <= XD_EIGH_ONE_BLOCK_N)
  n = 48   the per-round multi-block Jacobi (`_eigh_par_dev`)
  n = 96   full rank: the tridiagonal solve (`_eigh_td_try_dev`)
  n = 96   rank 12 (C C^T): repeated zero eigenvalues, the tridiagonal
           solve refuses and the multi-block Jacobi answers (MinCovDet's
           singular covariances on a 220-wide dataset)

IDENTICAL builds on a GPU column only. The inputs are float32 words built on
the host and handed to both sides unchanged."""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from x_decomp.eigh_ops_host import eigh_op_sym_host
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report


# ---- The device column (was eigh_route_gpu_suite.mojo) ----
def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _full_rank(n: Int) -> List[Float32]:
    # a symmetric matrix with a spread spectrum: a dominant diagonal plus a
    # bounded, deterministic off-diagonal pattern
    var a = List[Float32](length=n * n, fill=0.0)
    for i in range(n):  # test fixture: n x n words
        for j in range(i + 1):
            var x = Float32(((i * 37 + j * 11) % 23) - 11) / 32.0
            if i == j:
                x = Float32(i + 1) + x
            a[i * n + j] = x
            a[j * n + i] = x
    return a^


def _low_rank(n: Int, r: Int) -> List[Float32]:
    # C C^T with C n x r: rank r, n - r eigenvalues at zero (a refused gap)
    var c = List[Float32](length=n * r, fill=0.0)
    for i in range(n * r):  # test fixture: n x r words
        c[i] = Float32((i * 29 % 17) - 8) / 8.0
    var a = List[Float32](length=n * n, fill=0.0)
    for i in range(n):  # test fixture: n x n words
        for j in range(i + 1):
            var s = Float32(0.0)
            for k in range(r):
                s += c[i * r + k] * c[j * r + k]
            a[i * n + j] = s
            a[j * n + i] = s
    return a^


def _same_words(dev: List[Float32], host: List[Float32], what: String) raises:
    _require(len(dev) == len(host), what + ": lengths differ")
    for i in range(len(dev)):  # test fixture: compare every word
        if bitcast[DType.uint32](dev[i]) != bitcast[DType.uint32](host[i]):
            raise Error(what + ": word " + String(i) + " differs (device " + String(dev[i]) + ", host "
                        + String(host[i]) + ")")


def _case(var a: List[Float32], n: Int, name: String) raises:
    var host_in = a.copy()
    var wd = List[Float32](length=n, fill=0.0)
    var vd = List[Float32](length=n * n, fill=0.0)
    var wh = List[Float32](length=n, fill=0.0)
    var vh = List[Float32](length=n * n, fill=0.0)
    DevExec.eigh(F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(wd.unsafe_ptr())),
                 F32Ptr(unsafe_from_address=Int(vd.unsafe_ptr())), n, 0)
    var bad = eigh_op_sym_host(F32Ptr(unsafe_from_address=Int(host_in.unsafe_ptr())),
                               F32Ptr(unsafe_from_address=Int(wh.unsafe_ptr())),
                               F32Ptr(unsafe_from_address=Int(vh.unsafe_ptr())), n)
    _require(bad == 0, name + ": the host twin did not converge")
    _same_words(wd, wh, name + " eigenvalues")
    _same_words(vd, vh, name + " eigenvectors")
    for i in range(1, n):  # test fixture: ascending order
        _require(wd[i - 1] <= wd[i], name + ": eigenvalues not ascending at " + String(i))
    _ = a^
    _ = host_in^
    print("PASS eigh_route_gpu", name)


def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    _case(_full_rank(24), 24, "one-block n=24")
    _case(_full_rank(48), 48, "multi-block n=48")
    _case(_full_rank(96), 96, "tridiagonal n=96")
    _case(_low_rank(96, 12), 96, "refused tridiagonal -> multi-block n=96 rank 12")


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("eigh_route", cpu=False, gpu=True, serial="", run=run_cases)
