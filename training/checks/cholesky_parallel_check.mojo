# SPDX-License-Identifier: Apache-2.0
"""Cloud-only operation-level multi-GPU Cholesky gate.

For every shape, factor on one device and with whole trailing-update rows on
all devices; require equal identity traces (the whole matrix after every
panel's factor, solve and trailing stage), equal factor bits, info, nb and
logdet. Solve with whole right-hand-side columns and require equal traces and
solution bits. A non-positive-definite matrix must fail at the same info with
the same partial factor. Build with -D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1
to see this gate fail.
"""
from std.os import getenv, setenv
from std.memory import bitcast

from core.identity_trace import first_divergence
from cholesky.checks.potrf import chol_jitter_pinned
from cholesky.estimator import cholesky_factor_host, cholesky_solve_host


def _unit(seed: Int, a: Int) -> Float32:
    var u = UInt64(seed) * UInt64(0x9E3779B97F4A7C15) + UInt64(a) * UInt64(0xBF58476D1CE4E5B9)
    u ^= u >> 31
    u *= UInt64(0x94D049BB133111EB)
    u ^= u >> 29
    return Float32(Int((u >> 40) & 0xFFFF)) / Float32(32768.0) - Float32(1.0)


def spd(n: Int, seed: Int, broken: Bool) -> List[Float32]:
    """M M^T / n + I on the host in float64, symmetric by construction."""
    var m = List[Float64]()
    for i in range(n * n):
        m.append(Float64(_unit(seed, i)))
    var a = List[Float32]()
    for i in range(n):
        for j in range(n):
            var acc = Float64(0.0)
            for p in range(n):
                acc += m[i * n + p] * m[j * n + p]
            acc /= Float64(n)
            if i == j:
                acc += Float64(1.0)
                if broken and i == n // 2:
                    acc = Float64(-1.0)
            a.append(Float32(acc))
    # Exact symmetry in float32 bits.
    for i in range(n):
        for j in range(i):
            a[j * n + i] = a[i * n + j]
    return a^


def _clear(path: String) raises:
    with open(path, "w") as fh:
        fh.write("")


def _set(count: Int, path: String) raises:
    if not setenv("MOJOLEARN_CHOLESKY_DEVICE_COUNT", String(count), True):
        raise Error("setenv failed")
    _clear(path)
    if not setenv("MOJOLEARN_IDENTITY_TRACE", path, True):
        raise Error("setenv failed")


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    if len(a) != len(b):
        raise Error(what + " length differs")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(what + " bits differ at " + String(i))


def run_case(n: Int, nrhs: Int, broken: Bool, jitter: Float32, count: Int, root: String) raises:
    var a = spd(n, n * 7 + nrhs, broken)
    var name = "n" + String(n) + "-r" + String(nrhs) + ("-broken" if broken else "")
    var fone = root + "/" + name + "-factor-one.trace"
    var fmany = root + "/" + name + "-factor-many.trace"
    _set(1, fone)
    var one = cholesky_factor_host(a, n, jitter)
    _set(count, fmany)
    var many = cholesky_factor_host(a, n, jitter)
    var d = first_divergence(fone, fmany)
    if d != "":
        raise Error("factor trace differs " + name + ": " + d)
    _same(one.l, many.l, name + " factor")
    if one.info != many.info or one.nb != many.nb:
        raise Error(name + " info or nb differs")
    if bitcast[DType.uint32](one.logdet) != bitcast[DType.uint32](many.logdet):
        raise Error(name + " logdet differs")
    if broken:
        if one.info == 0:
            raise Error(name + " broken matrix factored")
        print("PASS cholesky equal failure", n, "info", one.info, "devices", count)
        return
    var b = List[Float32]()
    for i in range(n * nrhs):
        b.append(Float32(4.0) * _unit(n + 99, i))
    var sone = root + "/" + name + "-solve-one.trace"
    var smany = root + "/" + name + "-solve-many.trace"
    _set(1, sone)
    var x1 = cholesky_solve_host(one, b, nrhs)
    _set(count, smany)
    var x2 = cholesky_solve_host(many, b, nrhs)
    d = first_divergence(sone, smany)
    if d != "":
        raise Error("solve trace differs " + name + ": " + d)
    _same(x1, x2, name + " solution")
    print("PASS cholesky factor+solve", n, nrhs, "info", one.info, "devices", count)


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("RunPod required; no local execution")
    var count = Int(String(getenv("MOJOLEARN_CHOLESKY_CHECK_DEVICES", "2")))
    var root = String(getenv("MOJOLEARN_CHOLESKY_CHECK_DIR", "/tmp"))
    var ns: List[Int] = [1, 2, 31, 32, 33, 65, 100, 257, 513]
    var rhs: List[Int] = [1, 2, 3, 7]
    for n in ns:
        for r in rhs:
            run_case(n, r, False, chol_jitter_pinned(), count, root)
    run_case(129, 2, False, Float32(0.0), count, root)
    run_case(100, 1, True, Float32(0.0), count, root)
    run_case(257, 1, True, Float32(0.0), count, root)
    if not setenv("MOJOLEARN_IDENTITY_TRACE", "", True):
        raise Error("setenv failed")
    print("PASS cholesky parallel gate")
