# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LogisticRegression's probability links against a host oracle
(DEVIATION 549, IDENTITY_PATHS row 36).

`predict_proba` runs on the host in Float64 through `identical_exp64` on
BOTH paths: the GPU binding calls `glm/estimator.mojo::qn_sigmoid_host` /
`qn_softmax_host`, the CPU binding `core/classical_host_predict.mojo::
host_qn_sigmoid_into` / `host_qn_softmax_into`. Two spellings of one
formula are two chances to drift. This check holds both to the oracle
below, written from the definition and nothing else:

    sigmoid   p1 = 1 / (1 + exp(-z)),  p0 = 1 - p1
    softmax   m = first max under a strict `>`; s = serial ascending sum of
              exp(z_c - m); p_c = exp(z_c - m) / s

First the fixture must SEPARATE the pinned spelling from a plausible other
one (`e / (1 + e)` for the sigmoid, a descending sum for the softmax), else
the check is VACUOUS and fails. Then each shipped function must equal the
oracle bit for bit at every cell.

    tools/with_identical_mode.sh pixi run mojo run -I . glm/checks/link_seams_check.mojo

Sabotage arms (tools/identity_lanes/linear.checks):
glm/checks/sabotage/seam_549_sigmoid_ratio.patch (the GPU path's sigmoid
spelled `e / (1 + e)`), seam_549_softmax_host_desc.patch (the CPU path's
softmax sum descending).
"""
from std.memory import bitcast

from checks.fixture_rng import u01_row
from checks.numerics import identical_exp64
from core.classical_host_predict import host_qn_sigmoid_into, host_qn_softmax_into
from core.host_predict_threads import HostF64Ptr, host_list_ptr
from glm.estimator import qn_sigmoid_host, qn_softmax_host


comptime N_ROWS = 4099
comptime N_CLASSES = 5
comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime F64Ptr = MutPointer[Float64, MutUntrackedOrigin]


def _scores(n: Int, salt: Int) -> List[Float32]:
    """Scores in (-12, 12), with every 97th cell a repeat of its neighbour
    (ties for the softmax max) and a few exact zeros."""
    var out = List[Float32]()
    for i in range(n):
        if i % 97 == 1:
            out.append(out[i - 1])
        elif i % 211 == 0:
            out.append(Float32(0.0))
        else:
            out.append(Float32(24.0 * u01_row(i, 0, salt) - 12.0))
    return out^


def _bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def _oracle_sigmoid(z32: Float32) -> Float64:
    var z = Float64(z32)
    return 1.0 / (1.0 + identical_exp64(-z))


def _ratio_sigmoid(z32: Float32) -> Float64:
    var e = identical_exp64(Float64(z32))
    return e / (1.0 + e)


def _oracle_softmax(s: List[Float32], i: Int, c_n: Int, descending: Bool) -> List[Float64]:
    var m = Float64(s[i * c_n])
    for c in range(1, c_n):
        var v = Float64(s[i * c_n + c])
        if v > m:
            m = v
    var e = List[Float64]()
    for c in range(c_n):
        e.append(identical_exp64(Float64(s[i * c_n + c]) - m))
    var tot = Float64(0.0)
    if descending:
        for c in range(c_n - 1, -1, -1):
            tot += e[c]
    else:
        for c in range(c_n):
            tot += e[c]
    var out = List[Float64]()
    for c in range(c_n):
        out.append(e[c] / tot)
    return out^


def _compare(name: String, got: List[Float64], want: List[Float64]) raises:
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if _bits(got[i]) != _bits(want[i]):
            bad += 1
            if first < 0:
                first = i
    if bad:
        raise Error(
            name + ": " + String(bad) + " of " + String(len(want))
            + " cells differ from the oracle; first cell " + String(first)
            + " got " + String(got[first]) + " oracle " + String(want[first])
        )
    print(name, "OK:", len(want), "cells equal the oracle bit for bit")


def check_sigmoid() raises:
    var z = _scores(N_ROWS, 549)
    var want = List[Float64]()
    var separated = 0
    for i in range(N_ROWS):
        var p = _oracle_sigmoid(z[i])
        if _bits(p) != _bits(_ratio_sigmoid(z[i])):
            separated += 1
        want.append(1.0 - p)
        want.append(p)
    if separated == 0:
        raise Error("check_sigmoid: VACUOUS, no row separates 1/(1+exp(-z)) from e/(1+e)")
    print("check_sigmoid: the fixture separates the spellings on", separated, "of", N_ROWS, "rows")
    var gpu_path = List[Float64](length=2 * N_ROWS, fill=0.0)
    qn_sigmoid_host(F32Ptr(unsafe_from_address=Int(z.unsafe_ptr())),
                    F64Ptr(unsafe_from_address=Int(gpu_path.unsafe_ptr())), N_ROWS)
    _compare("check_sigmoid (GPU binding's qn_sigmoid_host)", gpu_path, want)
    var cpu_path = List[Float64](length=2 * N_ROWS, fill=0.0)
    host_qn_sigmoid_into(host_list_ptr(z), rebind[HostF64Ptr](cpu_path.unsafe_ptr()), N_ROWS, 3)
    _compare("check_sigmoid (CPU binding's host_qn_sigmoid_into)", cpu_path, want)


def check_softmax() raises:
    var n = N_ROWS * N_CLASSES
    var z = _scores(n, 706)
    var want = List[Float64]()
    var separated = 0
    for i in range(N_ROWS):
        var a = _oracle_softmax(z, i, N_CLASSES, False)
        var b = _oracle_softmax(z, i, N_CLASSES, True)
        var moved = False
        for c in range(N_CLASSES):
            want.append(a[c])
            if _bits(a[c]) != _bits(b[c]):
                moved = True
        if moved:
            separated += 1
    if separated == 0:
        raise Error("check_softmax: VACUOUS, no row separates the ascending sum from the descending one")
    print("check_softmax: the fixture separates the sum orders on", separated, "of", N_ROWS, "rows")
    var gpu_path = List[Float64](length=n, fill=0.0)
    qn_softmax_host(F32Ptr(unsafe_from_address=Int(z.unsafe_ptr())),
                    F64Ptr(unsafe_from_address=Int(gpu_path.unsafe_ptr())), N_ROWS, N_CLASSES)
    _compare("check_softmax (GPU binding's qn_softmax_host)", gpu_path, want)
    var cpu_path = List[Float64](length=n, fill=0.0)
    host_qn_softmax_into(host_list_ptr(z), rebind[HostF64Ptr](cpu_path.unsafe_ptr()), N_ROWS, N_CLASSES, 3)
    _compare("check_softmax (CPU binding's host_qn_softmax_into)", cpu_path, want)


def main() raises:
    check_sigmoid()
    check_softmax()
    print("link_seams_check: PASS")
