# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KernelExplainer over the resident linear block (lane vdf-explain, 2026-11-24): the device route
`agnostic_device.kshap_solve_linear` (masks, synthetic rows, the linear decision, means, link and solve without
leaving the device) against

  * the HOST COLUMN: `agnostic_host.kshap_synth`, x_linear's `decision_one` (the linear member's predict word) on
    every synthetic row and on the explained rows, `agnostic_host.kshap_solve`;
  * the DEVICE MEMBER ROUTE it replaces on a GPU build: `agnostic_device.kshap_synth` (downloaded), the same host
    decision, `agnostic_device.kshap_solve` (uploaded again).

phi is asserted bit for bit (binary64 words) under IDENTICAL, over three cells: (link identity, no logit), (link
exp, no logit), (link sigmoid, logit); two chunkings of the same rows (one chunk of 3, chunks of 2 + 1) must also
agree, since the route's chunk budget differs from the member route's. Under FAST the route is not built
(PSHAP_IDN_RESIDENT is IDENTICAL only): the check says so and passes.

    tools/with_identical_mode.sh pixi run mojo run -I . checks/native_tests/xtrees_agnostic_linear_check.mojo
"""
from std.math import min, max
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_linear.dispatch import decision_one
from x_linear.ops import FP
from xtrees import agnostic_device as agn_dev
from xtrees import agnostic_host as agn_host
from xtrees.api_kernels import kshap_schedule


comptime _R = 3
comptime _D = 4
comptime _NB = 5
comptime _K = 2
comptime _SEED = 7


struct _Sched(Movable):
    var off: List[Int64]
    var sw: List[Float64]
    var cdf: List[Float64]
    var res: List[Int64]

    def __init__(out self, M: Int, ns: Int):
        var h = max(1, M // 2)
        self.off = List[Int64](length=h + 1, fill=Int64(0))
        self.sw = List[Float64](length=h, fill=0.0)
        self.cdf = List[Float64](length=h, fill=0.0)
        self.res = List[Int64](length=6, fill=Int64(0))
        _ = kshap_schedule(M, ns, self.off.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                           self.sw.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                           self.cdf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                           self.res.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())

    def m(self) -> Int:
        return Int(self.res[0])

    def nfixed(self) -> Int:
        return Int(self.res[1])

    def nfull(self) -> Int:
        return Int(self.res[2])

    def npaired(self) -> Int:
        return Int(self.res[3])

    def L(self) -> Int:
        return Int(self.res[4])

    def wrand(self) -> UInt64:
        return UInt64(self.res[5])


def _fixture(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture values)
        v[i] = Float32(((i * 37 + salt * 11) % 23) - 11) / Float32(8.0)
    return v^


def _decide(rows: Int, x: Int, wb: Int, model_link: Int) -> List[Float32]:
    """x_linear `decision_one` (the linear member's predict word) on `rows` float32 rows of _D features."""
    var y = List[Float32](length=max(rows * _K, 1), fill=Float32(0))
    var xp = FP(unsafe_from_address=x)
    var wp = FP(unsafe_from_address=wb)
    for i in range(rows):  # small-loop(fixture rows)
        for c in range(_K):  # small-loop(fixture outputs)
            y[i * _K + c] = decision_one(xp, i, _D, wp, c, model_link)
    return y^


def _member_route(device: Bool, x: List[Float32], bg: List[Float32], wb: List[Float32], fnull: List[Float64],
                  s: _Sched, model_link: Int, logit: Bool) raises -> List[Float64]:
    """The member route of one chunk of _R rows: synth, the host decision, solve (host or device glue)."""
    var m = s.m()
    var phi = List[Float64](length=_R * _D * _K, fill=0.0)
    var syn = List[Float32](length=max(_R * m * _NB * _D, 1), fill=Float32(0))
    var so = Int(s.off.unsafe_ptr())
    var sw = Int(s.sw.unsafe_ptr())
    var cd = Int(s.cdf.unsafe_ptr())
    if device:
        agn_dev.kshap_synth(Int(x.unsafe_ptr()), Int(bg.unsafe_ptr()), so, sw, cd, Int(syn.unsafe_ptr()), _R, _NB,
                            _D, m, s.nfixed(), s.nfull(), s.npaired(), s.L(), _SEED, 0, s.wrand())
    else:
        agn_host.kshap_synth(Int(x.unsafe_ptr()), Int(bg.unsafe_ptr()), so, sw, cd, Int(syn.unsafe_ptr()), _R, _NB,
                             _D, m, s.nfixed(), s.nfull(), s.npaired(), s.L(), _SEED, 0, s.wrand())
    var out = _decide(_R * m * _NB, Int(syn.unsafe_ptr()), Int(wb.unsafe_ptr()), model_link)
    var fx = _decide(_R, Int(x.unsafe_ptr()), Int(wb.unsafe_ptr()), model_link)
    if device:
        agn_dev.kshap_solve(Int(out.unsafe_ptr()), Int(fx.unsafe_ptr()), Int(fnull.unsafe_ptr()), so, sw, cd,
                            Int(phi.unsafe_ptr()), _R, _NB, _D, _K, m, s.nfixed(), s.nfull(), s.npaired(), s.L(),
                            _SEED, 0, s.wrand(), logit)
    else:
        agn_host.kshap_solve(Int(out.unsafe_ptr()), Int(fx.unsafe_ptr()), Int(fnull.unsafe_ptr()), so, sw, cd,
                             Int(phi.unsafe_ptr()), _R, _NB, _D, _K, m, s.nfixed(), s.nfull(), s.npaired(), s.L(),
                             _SEED, 0, s.wrand(), logit)
    _ = syn^
    _ = out^
    _ = fx^
    return phi^


def _resident(x: List[Float32], bg: List[Float32], wb: List[Float32], fnull: List[Float64], s: _Sched,
              model_link: Int, logit: Bool, first: Int) raises -> List[Float64]:
    """The resident linear route over the _R rows in two calls (rows 0 .. first-1, then the rest; first == _R:
    one call), as `agn_kernel_values_run` chunks them."""
    var m = s.m()
    var phi = List[Float64](length=_R * _D * _K, fill=0.0)
    var so = Int(s.off.unsafe_ptr())
    var sw = Int(s.sw.unsafe_ptr())
    var cd = Int(s.cdf.unsafe_ptr())
    var r0 = 0
    while r0 < _R:  # driver-loop: over the fixture's row chunks
        var rows = first if r0 == 0 else _R - r0
        agn_dev.kshap_solve_linear(Int(x.unsafe_ptr()) + 4 * r0 * _D, Int(bg.unsafe_ptr()), Int(wb.unsafe_ptr()),
                                   Int(fnull.unsafe_ptr()), so, sw, cd, Int(phi.unsafe_ptr()) + 8 * r0 * _D * _K,
                                   rows, _NB, _D, _K, m, s.nfixed(), s.nfull(), s.npaired(), s.L(), _SEED, r0,
                                   s.wrand(), logit, model_link)
        r0 += rows
    return phi^


def _same(a: List[Float64], b: List[Float64], what: String) raises:
    if len(a) != len(b):
        raise Error("FAIL " + what + ": lengths differ")
    for i in range(len(a)):  # small-loop(fixture words)
        if bitcast[DType.uint64](a[i]) != bitcast[DType.uint64](b[i]):
            raise Error("FAIL " + what + ": phi word " + String(i) + " differs: " + String(a[i]) + " vs "
                        + String(b[i]))


def check_cell(model_link: Int, logit: Bool) raises:
    var x = _fixture(_R * _D, 1)
    var bg = _fixture(_NB * _D, 2)
    var wb = _fixture(_K * (_D + 1), 3)
    var fnull: List[Float64] = [0.125, -0.25]
    var s = _Sched(_D, min(2 * _D + 2048, (1 << _D) - 2))
    var cell = String("link ") + String(model_link)
    if logit:
        cell += " logit"
    var host = _member_route(False, x, bg, wb, fnull, s, model_link, logit)
    var member = _member_route(True, x, bg, wb, fnull, s, model_link, logit)
    var one = _resident(x, bg, wb, fnull, s, model_link, logit, _R)
    var two = _resident(x, bg, wb, fnull, s, model_link, logit, 2)
    _same(member, host, cell + ": device member route vs host column")
    _same(one, host, cell + ": resident linear vs host column")
    _same(two, one, cell + ": resident linear, chunks 2 + 1 vs one chunk")
    print("PASS kshap_solve_linear", cell, ": resident == member route == host column (", len(host), "words )")


def run_checks() raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and agn_dev.PSHAP_IDN_RESIDENT:
        check_cell(0, False)
        check_cell(1, False)
        check_cell(2, True)
    else:
        print("SKIP kshap_solve_linear: the resident linear route is built in IDENTICAL only")
